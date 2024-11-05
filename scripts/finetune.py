import tqdm
import tensorflow as tf
import torch
import wandb
import os
import copy
import json
import logging
import pathlib
from tqdm import tqdm
import transformers
from transformers.trainer import ALL_LAYERNORM_LAYERS, get_parameter_names
from PIL import Image
from accelerate import PartialState
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, List
from torch.nn import MSELoss, L1Loss, SmoothL1Loss
from torch.optim import AdamW, Adam
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from spatialvla.datasets import RLDSDataset, RLDSBatchTransform
from spatialvla.datasets.rlds.utils.data_utils import save_dataset_statistics
from spatialvla.datasets.rlds.utils.data_utils import PaddedCollatorForActionPrediction

from spatialvla.mobilevlm.model.mobilevlm import load_pretrained_vlm_for_vla
from spatialvla.mobilevlm.train.train import find_all_linear_names, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3
from mergelora import merge_lora

from spatialvla_config import ModelArguments, TrainingArguments

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
    model_args, 
    load_8bit, 
    load_4bit,
    device='cuda',
)
model.config.use_cache = False
model = model.to(device_id)
print('Pretrained VLM Loaded')

batch_transform = RLDSBatchTransform(
    tokenizer,
    image_processor,
)

dataset = RLDSDataset(
    data_root_dir=training_args.data_root_dir,
    data_mix=training_args.data_mix,
    batch_transform=batch_transform,
    shuffle_buffer_size=training_args.shuffle_buffer_size,
    train=True,
    window_size=1,
    future_action_window_size=model_args.action_len - 1
)

collator = PaddedCollatorForActionPrediction(tokenizer.model_max_length, tokenizer.pad_token_id, padding_side='right')

dataloader = DataLoader(
    dataset,
    batch_size=training_args.batch_size,
    sampler=None,
    collate_fn=collator,
    num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
)
# Save statistics to a JSON file
if not os.path.exists(training_args.output_dir):
    os.makedirs(training_args.output_dir)
temp_dir = f'{training_args.output_dir}_tmp'

save_dataset_statistics(dataset.dataset_statistics, training_args.output_dir)
print('Dataset Loaded')

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

print('LoRA applied to ', find_all_linear_names(model))

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
        task_type="CAUSAL_LM",
    )
    if training_args.bits == 16:
        if training_args.bf16:
            model.to(torch.bfloat16)
        if training_args.fp16:
            model.to(torch.float16)
    rank0_print("Adding LoRA adapters...")
    model = get_peft_model(model, lora_config)

print(model)

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

for p in model.get_model().mm_projector.parameters():
    p.requires_grad = True
for p in model.action_head.parameters():
    p.requires_grad = True
model.print_trainable_parameters()
print('LoRA Loaded')

model = DDP(model, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
decay_parameters = [name for name in decay_parameters if "bias" not in name]
unused_parameters = [name for name, _ in model.named_parameters() if "vision_tower" in name and "layers" not in name]
projector_parameters = [name for name, _ in model.named_parameters() if "mm_projector" in name]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if (n in decay_parameters and n not in projector_parameters and n not in unused_parameters and p.requires_grad)],
        "lr": training_args.learning_rate,
        "eps":1e-5,
    },
    {
        "params": [
            p for n, p in model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and n not in unused_parameters and p.requires_grad)
        ],
        "weight_decay": 0.0,
        "lr": training_args.learning_rate,
        "eps":1e-5
    },
    {
        "params": [
            p for n, p in model.named_parameters() if (n in decay_parameters and n in projector_parameters and n not in unused_parameters and p.requires_grad)
        ],
        "lr": training_args.learning_rate,
        "eps":1e-5
    },
    {
        "params": [
            p for n, p in model.named_parameters() if (n not in decay_parameters and n in projector_parameters and n not in unused_parameters and p.requires_grad)
        ],
        "weight_decay": 0.0,
        "lr": training_args.learning_rate,
        "eps":1e-5
    },
]
optimizer = optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
# lora_params = [param for name, param in model.named_parameters() if 'lora' in name]
# non_lora_params = [param for name, param in model.named_parameters() if 'lora' not in name and param.requires_grad]
# print([name for name, param in model.named_parameters() if 'lora' not in name and param.requires_grad])
# layernorm_params = [param for name, param in model.named_parameters() if 'lora' not in name and 'layernorm' in name and param.requires_grad]


# loss_fn = MSELoss()
# loss_fn = L1Loss()
loss_fn = SmoothL1Loss()


## Wandb Init
if distributed_state.is_main_process:
    wandb.init(
        entity=training_args.wandb_entity,
        project=training_args.wandb_project,
        name=f"{training_args.data_mix}_chunk{model_args.action_len}"
    )

## Training LOOP!
print('Training Start')
with tqdm(total=training_args.max_steps, leave=False) as progress:
    model.train()
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(dataloader):
        with torch.autocast('cuda', dtype=torch.float16):
            action = model.forward(
                input_ids=batch['input_ids'].to(device_id),
                images=batch['pixel_values'].to(device_id),
                attention_mask=batch['attention_mask'].to(device_id)
            )
            loss = loss_fn(action, batch['action'].to(device_id))
            # print(loss.item())
        normalized_loss = loss / training_args.gradient_accumulation_steps
        normalized_loss.backward()

        # Compute gradient step index
        gradient_step_idx = batch_idx // training_args.gradient_accumulation_steps

        # Wandb Log
        if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
            max_grad = 0.0
            total_grad_sum = 0.0
            total_grad_count = 0
            global_norm = 0.0
            
            # Aggregate gradient statistics
            for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                    # Update max gradient
                    max_grad = max(max_grad, param.grad.abs().max().item())
                    # Sum up gradients for mean calculation
                    total_grad_sum += param.grad.data.sum().item()
                    total_grad_count += param.grad.numel()
                    # Update the global gradient norm
                    global_norm += param.grad.data.norm(2).item() ** 2  # L2 norm

            # Calculate mean gradient
            mean_grad = total_grad_sum / total_grad_count if total_grad_count > 0 else 0.0
            # Calculate the global norm (take square root of the sum of squares)
            global_norm = global_norm ** 0.5

            # Log global gradient statistics to WandB
            wandb.log({
                "grad/max": max_grad,
                "grad/mean": mean_grad,
                "grad/norm": global_norm,
                "action_loss": loss.item(),  # Log loss globally as well
            },
            step=gradient_step_idx             
            )
        
        # Update
        if (batch_idx + 1) % training_args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            progress.update()

        # Save Checkpoint
        if gradient_step_idx > 0 and gradient_step_idx % training_args.save_steps == 0:
            if distributed_state.is_main_process:
                print(f"Saving Model Checkpoint for Step {gradient_step_idx}")

                dist.barrier()

                model.module.config.save_pretrained(temp_dir)
                state_dict = get_peft_state_maybe_zero_3(model.module.named_parameters(), training_args.lora_bias)
                non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.module.named_parameters())
                model.module.save_pretrained(temp_dir, state_dict=state_dict)
                torch.save(non_lora_state_dict, os.path.join(temp_dir, 'non_lora_trainables.bin'))

                # Merge lora model into output dir
                merge_lora(model_args.model_path, temp_dir, training_args.output_dir)

                dist.barrier()

        if gradient_step_idx == training_args.max_steps:
            print(f"Max step {training_args.max_steps} reached! Stopping training...")
           

# Save configs and state_dicts into temp dir
model.config.save_pretrained(temp_dir)
state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
model.save_pretrained(temp_dir, state_dict=state_dict)
torch.save(non_lora_state_dict, os.path.join(temp_dir, 'non_lora_trainables.bin'))

# Merge lora model into output dir
merge_lora(model_args.model_path, temp_dir, training_args.output_dir)
print('Checkpoint Saved')