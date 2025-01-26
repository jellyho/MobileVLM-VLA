import tqdm
import tensorflow as tf
import torch
import torch.nn as nn
import wandb
import os
import time
import numpy as np
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

from spatialvla.mobilevlm.model.mobilevlm import load_pretrained_vlm_for_vla, load_vla
from spatialvla.mobilevlm.train.train import find_all_linear_names, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3
from spatialvla_config import ModelArguments, TrainingArguments, HEAD_ARGS
from mergelora import merge_lora
from spatialvla.mobilevlm.action_tokenizer import ActionTokenizer
from spatialvla.mobilevlm.conversation import conv_templates, SeparatorStyle
from spatialvla.mobilevlm.utils import disable_torch_init, process_images, tokenizer_image_token, KeywordsStoppingCriteria
from spatialvla.mobilevlm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from PIL import Image

tf.config.set_visible_devices([], "GPU") ## Ensure dataloader did not access to gpu
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"] = "offline"

distributed_state = PartialState()
device_id = distributed_state.local_process_index
torch.cuda.set_device(device_id)
torch.cuda.empty_cache()

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

if training_args.resume:
    tokenizer, model, image_processor, _ = load_vla(
        training_args.output_dir,
        load_8bit=False, 
        load_4bit=False,
        device=device_id,
        dtype=dtype
    )
else:
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

## Important!
if model.config.head_args['head_type'] == 'BR':
    action_tokenizer = ActionTokenizer(tokenizer)
    model.action_tokenizer = action_tokenizer
else:
    action_tokenizer = None

batch_transform = RLDSBatchTransform(
    tokenizer,
    image_processor,
    use_state_input=model_args.use_state_input,
    action_tokenizer=action_tokenizer,
    window_size=1,
    future_action_window_size=model_args.action_len - 1,
    use_hz_input=model_args.use_hz_input
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
    use_state_input=model_args.use_state_input,
    num_parallel_calls=training_args.num_parallel_calls
)

collator = PaddedCollatorForActionPrediction(
    tokenizer.model_max_length, 
    tokenizer.pad_token_id, 
    padding_side='right',
    use_state_input=model_args.use_state_input,
    use_label=(model.config.head_args['head_type'] == 'BR'),
    use_hz_input=model_args.use_hz_input
)

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

if training_args.freeze_vision_backbone:
    for param in model.model.vision_tower.parameters():
        param.requires_grad = False

model = DDP(model, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)
decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
decay_parameters_names = [name for name in decay_parameters if "bias" not in name]
optimizer_grouped_parameters = []
if model.module.config.head_args['head_type'] == 'BR':
    si_parameters_names = []
    for n, p in model.named_parameters():
        if 'si.' in n:
            si_parameters_names.append(n)
    decay_params = [p for n, p in model.named_parameters() if (n in decay_parameters_names and n not in si_parameters_names and p.requires_grad)]
    nondecay_params = [p for n, p in model.named_parameters() if (n not in decay_parameters_names and n not in si_parameters_names and p.requires_grad)]
    si_params = [p for n , p in model.named_parameters() if (n in si_parameters_names and p.requires_grad)]
    optimizer_grouped_parameters.append(
            {
            "params": si_params,
            "weight_decay": 0.0,
            "lr": 5e-6,
            "eps":training_args.adam_epsilon,
        }
    )
else:    
    decay_params = [p for n, p in model.named_parameters() if (n in decay_parameters_names and p.requires_grad)]
    nondecay_params = [p for n, p in model.named_parameters() if (n not in decay_parameters_names and p.requires_grad)]
optimizer_grouped_parameters.extend([
    {
        "params": decay_params,
        "weight_decay": training_args.weight_decay,
        "lr": training_args.learning_rate,
        "eps":training_args.adam_epsilon,
    },
    {
        "params": nondecay_params,
        "weight_decay": 0.0,
        "lr": training_args.learning_rate,
        "eps":training_args.adam_epsilon
    },
])
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

if training_args.resume:
    if os.path.exists(f'{training_args.output_dir}/training_states.pth'):
        ckpt = torch.load(f'{training_args.output_dir}/training_states.pth', map_location="cpu")
        optimizer.load_state_dict(ckpt['optim'])
        scheduler.load_state_dict(ckpt['scheduler'])
        step = ckpt['step']
    else:
        step = 0
else:
    step = 0

## Training LOOP!
print('Training Start')
with tqdm(total=training_args.max_steps, initial=step, leave=False) as progress:
    model.train()
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(dataloader, start=step):
        ## Forward
        model.train()
        optimizer.zero_grad()
        with torch.autocast('cuda', dtype=dtype):
            loss = model.forward(
                input_ids=batch['input_ids'].to(device_id),
                images=batch['pixel_values'].to(device_id),
                attention_mask=batch['attention_mask'].to(device_id),
                actions=batch['action'].to(device_id),
                states=batch['proprio'].to(device_id) if model_args.use_state_input else None,
                labels=batch['labels'] if model.module.config.head_args['head_type'] == 'BR' else None,
                hz=batch['hz'].to(device_id) if model_args.use_hz_input else None
            )
            if model.module.config.head_args['head_type'] == 'BR':
                action_logits = loss.logits[:, -51:-1]
                action_preds = action_logits.argmax(dim=2)
                action_gt = batch['labels'][:, -50:].to(action_logits.device)
                mask = action_gt > action_tokenizer.action_token_begin_idx
                correct_preds = (action_preds == action_gt) & mask
                action_accuracy = correct_preds.sum().float() / mask.sum().float()
                eps_loss = loss.loss[5]
                ce_loss = loss.loss[1]
                v_loss = loss.loss[2]
                s_loss = loss.loss[3]
                b_loss = loss.loss[4]
                loss = loss.loss[0]

        normalized_loss = loss / training_args.gradient_accumulation_steps
        normalized_loss.backward()
        gradient_step_idx = batch_idx // training_args.gradient_accumulation_steps

        ## Model Update
        if (batch_idx + 1) % training_args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=training_args.max_grad_norm)
            optimizer.step()
            if training_args.lr_scheduler_type != 'constant':
                scheduler.step()
            if model.module.config.head_args['head_type'] == 'BR':
                model.module.si.ema.update()
            progress.update()

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
                        states=batch['proprio'].to(device_id) if model_args.use_state_input else None,
                        prior_actions=batch['action'].to(device_id),
                        hz=batch['hz'].to(device_id) if model_args.use_hz_input else None,
                    )
                    prediction_time = float(time.time() - start) / training_args.batch_size
                action_loss = nn.functional.mse_loss(batch['action'].to(device_id), predicted_action, reduction='mean')
                log_dict['action_loss'] = action_loss.item()
                log_dict['pred_time'] = prediction_time

        if distributed_state.is_main_process:
            log_dict['loss'] = normalized_loss.item()
            if model.module.config.head_args['head_type'] == 'BR':
                log_dict['token acc'] = action_accuracy
                log_dict['eps_loss'] = eps_loss.detach().cpu().numpy()
                log_dict['ce_loss'] = ce_loss.cpu().numpy()
                log_dict['v_loss'] = v_loss.detach().cpu().numpy()
                log_dict['s_loss'] = s_loss.detach().cpu().numpy()
                log_dict['b_loss'] = b_loss.detach().cpu().numpy()
            log_dict['learning_rate'] = scheduler.get_last_lr()[0] if training_args.lr_scheduler_type != 'constant' else training_args.learning_rate
            wandb.log(log_dict, step=gradient_step_idx)

        # Save Checkpoint
        if gradient_step_idx > 0 and gradient_step_idx % training_args.save_steps == 0:
            if distributed_state.is_main_process:
                print(f"Saving Model Checkpoint for Step {gradient_step_idx}")
                model.module.config.save_pretrained(training_args.output_dir)
                model.module.save_pretrained(training_args.output_dir)
                tokenizer.save_pretrained(training_args.output_dir)
                other_states = {
                    'step': gradient_step_idx,
                    'optim': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }
                torch.save(other_states, f'{training_args.output_dir}/training_states.pth')
                del other_states
                if model.module.config.head_args['head_type'] == 'BR':
                    model.module.si.save_ema(training_args.output_dir)

            dist.barrier()

        if gradient_step_idx >= training_args.max_steps:
            print(f"Max step {training_args.max_steps} reached! Stopping training...")
            break

print('Checkpoint Saved. Finished.')