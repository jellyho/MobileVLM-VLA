import tqdm
import tensorflow as tf
import torch
import torch.nn as nn
import wandb
import os
import time
from tqdm import tqdm
import transformers
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers.trainer import ALL_LAYERNORM_LAYERS, get_parameter_names
from accelerate import PartialState
from dataclasses import  asdict
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from spatialvla.datasets import RLDSDataset, RLDSBatchTransform
from spatialvla.datasets.rlds.utils.data_utils import save_dataset_statistics
from spatialvla.datasets.rlds.utils.data_utils import PaddedCollatorForActionPrediction

from spatialvla.mobilevlm.model.mobilevlm import load_pretrained_vlm_for_vla
from spatialvla.mobilevlm.train.train import find_all_linear_names, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3
from spatialvla_config import ModelArguments, TrainingArguments, HEAD_ARGS
from mergelora import merge_lora

tf.config.set_visible_devices([], "GPU") ## Ensure dataloader did not access to gpu
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from pynvml import *
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)

distributed_state = PartialState()
device_id = distributed_state.local_process_index
torch.cuda.set_device(device_id)
torch.cuda.empty_cache()

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

## Argument parsing
parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
model_args, training_args = parser.parse_args_into_dataclasses()
model_args.head_args = HEAD_ARGS[model_args.action_head]

## Load Pretrained VLM
load_4bit = training_args.bits == 4
load_8bit = training_args.bits == 8
if training_args.bf16:
    dtype = torch.bfloat16
if training_args.fp16:
    dtype = torch.float16
tokenizer, model, image_processor, _ = load_pretrained_vlm_for_vla(
    model_args, 
    load_8bit, 
    load_4bit,
    device=device_id,
    dtype=dtype
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
    future_action_window_size=model_args.action_len - 1,
    enable_autotune=training_args.enable_autotune,
    use_state_input=model_args.use_state_input
)

# sampler = DistributedSampler(dataset)
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
    try:
        os.makedirs(training_args.output_dir)
    except:
        pass
if distributed_state.is_main_process:
    save_dataset_statistics(dataset.dataset_statistics, training_args.output_dir)
temp_dir = f'{training_args.output_dir}_tmp'
print('Dataset Loaded')

# Gradient Checkpointing
if training_args.gradient_checkpointing: 
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

# print(model)
print(sum(p.numel() for p in model.model.parameters() if p.requires_grad))
print(sum(p.numel() for p in model.action_head.parameters() if p.requires_grad))
model = DDP(model, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)
decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
decay_parameters = [name for name in decay_parameters if "bias" not in name]
print(sum(p.numel() for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)))
print(sum(p.numel() for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)))
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)],
        "weight_decay": training_args.weight_decay,
        "lr": training_args.learning_rate,
        "eps":training_args.adam_epsilon,
    },
    {
        "params": [p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
        "weight_decay": 0.0,
        "lr": training_args.learning_rate,
        "eps":training_args.adam_epsilon
    },
]
optimizer = AdamW(optimizer_grouped_parameters)

if training_args.lr_scheduler_type == 'linear':
    warmup_steps = int(training_args.max_steps * training_args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_args.max_steps)
elif training_args.lr_scheduler_type == 'cosine':
    warmup_steps = int(training_args.max_steps * training_args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_args.max_steps)
loss_fn = MSELoss()

## Wandb Init
if distributed_state.is_main_process:
    wandb.init(
        entity=training_args.wandb_entity,
        project=training_args.wandb_project,
        name=f"{training_args.data_mix}_{model_args.action_head}_chunk{model_args.action_len}",
        config={**asdict(training_args), **asdict(model_args)}
    )

## Training LOOP!
print('Training Start')
with tqdm(total=training_args.max_steps, leave=False) as progress:
    model.train()
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(dataloader):
        ## Forward
        model.train()
        optimizer.zero_grad()
        with torch.autocast('cuda', dtype=dtype):
            loss = model.forward(
                input_ids=batch['input_ids'].to(device_id),
                images=batch['pixel_values'].to(device_id),
                attention_mask=batch['attention_mask'].to(device_id),
                actions=batch['action'].to(device_id),
                states=batch['observation']['proprio'] if model_args.use_state_input else None
            )
        normalized_loss = loss / training_args.gradient_accumulation_steps
        normalized_loss.backward()
        gradient_step_idx = batch_idx // training_args.gradient_accumulation_steps

        ## Model Update
        if (batch_idx + 1) % training_args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=training_args.max_grad_norm)
            optimizer.step()
            if training_args.lr_scheduler_type != 'constant':
                scheduler.step() 
            progress.update()
            # info = nvmlDeviceGetMemoryInfo(handle)
            # print(f"Used memory: {info.used / 1024 ** 2} MB")

        ## Logging        
        log_dict = {}
        if distributed_state.is_main_process and gradient_step_idx % training_args.log_steps == 0:
            if training_args.log_grad:  # Log Grad
                max_grad, total_grad_sum, total_grad_count, global_norm = 0.0, 0.0, 0, 0.0
                for param in model.parameters():
                    if param.requires_grad and param.grad is not None:
                        max_grad = max(max_grad, param.grad.abs().max().item())
                        total_grad_sum += param.grad.data.sum().item()
                        total_grad_count += param.grad.numel()
                        global_norm += param.grad.data.norm(2).item() ** 2  # L2 norm
                mean_grad = total_grad_sum / total_grad_count if total_grad_count > 0 else 0.0
                global_norm = global_norm ** 0.5
                log_dict['grad/max'] = max_grad
                log_dict['grad/mean'] = mean_grad
                log_dict['grad/norm'] = global_norm

            model.eval()
            with torch.no_grad():
                with torch.autocast('cuda', dtype=dtype):
                    start = time.time()
                    predicted_action = model.module.predict_action(
                        input_ids=batch['input_ids'].to(device_id),
                        images=batch['pixel_values'].to(device_id),
                        attention_mask=batch['attention_mask'].to(device_id),
                        use_cache=True,
                        states=batch['observation']['proprio'] if model_args.use_state_input else None
                    )
                    prediction_time = float(time.time() - start) / training_args.batch_size
                action_loss = nn.functional.mse_loss(batch['action'].to(device_id), predicted_action, reduction='mean')
                log_dict['action_loss'] = action_loss.item()
                log_dict['pred_time'] = prediction_time

        if distributed_state.is_main_process:
            log_dict['loss'] = normalized_loss.item()
            log_dict['learning_rate'] = scheduler.get_last_lr()[0] if training_args.lr_scheduler_type != 'constant' else training_args.learning_rate
            wandb.log(log_dict, step=gradient_step_idx)

        # Save Checkpoint
        if gradient_step_idx > 0 and gradient_step_idx % training_args.save_steps == 0:
            if distributed_state.is_main_process:
                print(f"Saving Model Checkpoint for Step {gradient_step_idx}")
                model.module.config.save_pretrained(training_args.output_dir)
                model.module.save_pretrained(training_args.output_dir)
                tokenizer.save_pretrained(training_args.output_dir)

            dist.barrier()

        if gradient_step_idx >= training_args.max_steps:
            print(f"Max step {training_args.max_steps} reached! Stopping training...")
            break

print('Checkpoint Saved. Finished.')