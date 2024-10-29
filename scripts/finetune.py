import tqdm
import tensorflow as tf
import torch
import wandb
import os
import copy
import json
import logging
import pathlib
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig
from PIL import Image
from accelerate import PartialState
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.distributed as dist
from dataset.dataset import RLDSDataset, save_statistics_to_json
from mobilevlm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from mobilevlm import conversation as conversation_lib
from mobilevlm.utils import tokenizer_image_token
from mobilevlm.model.mobilevlm import load_pretrained_vlm_for_vla
from mobilevlm.model.mobilellama import MobileLlamaForCausalLM, SpatialVLAForCausalLM, MobileVLMConfig, SpatialVLAConfig
from mobilevlm.train.train import find_all_linear_names, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3
from mergelora import merge_lora

from spatialvla_config import dataset_kwargs, traj_transform_kwargs, frame_transform_kwargs, ModelArguments, TrainingArguments
tf.config.set_visible_devices([], "GPU") ## Ensure dataloader did not access to gpu
os.environ["TOKENIZERS_PARALLELISM"] = "false"

distributed_state = PartialState()
torch.cuda.set_device(device_id := distributed_state.local_process_index)
torch.cuda.empty_cache()

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

## Argument parsing
parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
model_args, training_args = parser.parse_args_into_dataclasses()

## Load Pretrained VLM
load_4bit = training_args.bits == 4
load_8bit = training_args.bits == 8

tokenizer, model, image_processor, _ = load_pretrained_vlm_for_vla(
    model_args.model_path, 
    load_8bit, 
    load_4bit,
    device='cuda',
    action_len=model_args.action_len,
    action_dim=model_args.action_dim,
    action_hidden_size=model_args.action_hidden_size
)
model.config.use_cache = False
print('Pretrained VLM Loaded')

dataset = RLDSDataset(
    dataset_kwargs,
    tokenizer=tokenizer,
    image_processor=image_processor,
    shuffle_buffer_size=training_args.shuffle_buffer_size,
    traj_transform_kwargs=traj_transform_kwargs,
    frame_transform_kwargs=frame_transform_kwargs,
    train=True,
)
dataloader = DataLoader(
    dataset,
    batch_size=training_args.batch_size,
    sampler=None,
    num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
)
# Save statistics to a JSON file
save_statistics_to_json(dataset.dataset.dataset_statistics, f"{training_args.output_dir}/dataset_statistics.json")
print('Dataset Loaded')

## Model initial setup
if model_args.freeze_backbone:
    model.model.requires_grad_(False)

# Quantization
if training_args.bits in [4, 8]: 
    from peft import prepare_model_for_kbit_training
    model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

# Gradient Checkpointing
if training_args.gradient_checkpointing: 
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

## LORA SETTING
if training_args.lora_enable:
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=find_all_linear_names(model),
        lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias,
        init_lora_weights="gaussian",
    )
    if training_args.bits == 16:
        if training_args.bf16:
            model.to(torch.bfloat16)
        if training_args.fp16:
            model.to(torch.float16)
    rank0_print("Adding LoRA adapters...")
    model = get_peft_model(model, lora_config)


# Quantization LoRA
if training_args.bits in [4, 8]:
    model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)
    from peft.tuners.lora import LoraLayer
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if training_args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name or 'action_head' in name or 'action_hidden_size' in name:
            if hasattr(module, 'weight'):
                if training_args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
model.print_trainable_parameters()
print('LoRA Loaded')

trainable_params = [param for param in model.parameters() if param.requires_grad]
optimizer = AdamW(trainable_params, lr=training_args.learning_rate)
loss_fn = MSELoss()

## Wandb Init
if distributed_state.is_main_process:
    wandb.init(
        entity=training_args.wandb_entity,
        project=training_args.wandb_project,
        name=f"{dataset_kwargs['name']}_chunk{model_args.action_len}"
    )


## Training LOOP!
print('Training Start')
with tqdm(total=training_args.max_steps, leave=False) as progress:
    model.train()
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(dataloader):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            action = model.forward(
                input_ids=batch['input_ids'][:, 0, :].to(device_id),
                images=batch['pixel_values'].to(device_id),
            )
            loss = loss_fn(action, batch['action'].to(device_id))
        normalized_loss = loss / training_args.gradient_accumulation_steps
        normalized_loss.backward()

        # Compute gradient step index
        gradient_step_idx = batch_idx // training_args.gradient_accumulation_steps

        # Update
        if (batch_idx + 1) % training_args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            progress.update()

        # Wandb Log
        if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
            wandb.log(
                {
                    "action_loss": loss.item(),
                },
                step=gradient_step_idx,
            )

        # Save Checkpoint
        if gradient_step_idx > 0 and gradient_step_idx % training_args.save_steps == 0:
            if distributed_state.is_main_process:
                print(f"Saving Model Checkpoint for Step {gradient_step_idx}")
            dist.barrier()

            model.config.save_pretrained(training_args.temp_dir)
            state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
            model.save_pretrained(training_args.temp_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.temp_dir, 'non_lora_trainables.bin'))

            # Merge lora model into output dir
            merge_lora(model_args.model_path, training_args.temp_dir, training_args.output_dir)
            dist.barrier()

        if gradient_step_idx == training_args.max_steps:
            print(f"Max step {training_args.max_steps} reached! Stopping training...")


            

# Save configs and state_dicts into temp dir
model.config.save_pretrained(training_args.temp_dir)
state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
model.save_pretrained(training_args.temp_dir, state_dict=state_dict)
torch.save(non_lora_state_dict, os.path.join(training_args.temp_dir, 'non_lora_trainables.bin'))

# Merge lora model into output dir
merge_lora(model_args.model_path, training_args.temp_dir, training_args.output_dir)
print('Checkpoint Saved')