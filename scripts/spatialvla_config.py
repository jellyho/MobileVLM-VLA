from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Sequence, List, Any
import transformers

## head args template
# MLP
MLPHead = {
    'head_type': 'MLP',
    'action_hidden_sizes': [256],
    'hidden_projection' : 'mean'
}

# MAP
MAPHead = {
    'head_type': 'MAP',
    'num_heads': 8,
    'hidden_projection': 'pass', # Always pass
    'max_action': 5.0,
    'use_tanh': True
}

# Diffusion
DiffusionHead = {
    'head_type': 'Diffusion',
    'hidden_projection': 'pass', # If use_map is true, set this to pass
    'use_map' : False,
    'max_action': 5.0,
    'time_dim':  32,
    'num_blocks': 3,
    'dropout_rate': 0.0,
    'hidden_dim': 512,
    'use_layer_norm':True,
    'diffusion_steps': 20,
    'n_diffusion_samples': 1,
}

FlowMatchingHead = {
    'head_type': 'FlowMatching',
    'hidden_projection': 'pass', # If use_map is true, set this to pass
    'use_map' : False,
    'max_action': 5.0,
    'time_dim':  32,
    'num_blocks': 3,
    'dropout_rate': 0.0,
    'hidden_dim': 512,
    'use_layer_norm':True,
    'diffusion_steps': 10,
    'n_diffusion_samples': 1,
}

DiffusionPolicyHead = {
    'head_type': 'DiffusionPolicy',
    'hidden_projection': 'pass',  # If use_map is true, set this to pass
    'use_map' : True,
    'max_action': 5.0,
    'time_dim':  256,
    'dropout_rate': 0.0,
    'hidden_dim': 256,
    'use_layer_norm':True,
    'diffusion_steps': 100,
}

DiffusionPolicyHead2 = {
    'head_type': 'DiffusionPolicy2',
    'hidden_projection': 'pass',  # If use_map is true, set this to pass
    'use_map' : True,
    'max_action': 5.0,
    'time_dim':  256,
    'dropout_rate': 0.0,
    'hidden_dim': 256,
    'use_layer_norm':True,
    'diffusion_steps': 100,
}

FlowMatchingDiffusionPolicyHead = {
    'head_type': 'FlowMatchingDiffusionPolicy',
    'hidden_projection': 'pass',  # If use_map is true, set this to pass
    'use_map' : True,
    'max_action': 5.0,
    'time_dim':  256,
    'dropout_rate': 0.0,
    'hidden_dim': 256,
    'use_layer_norm':True,
    'diffusion_steps': 10,
}

DiT = {
    'head_type':'DiT',
    'hidden_projection': 'pass', # Always pass
    'use_map' : False, # Always False
    'max_action': 5.0,
    'time_dim': 256,
    'hidden_dim': 256,
    'diffusion_steps': 100,
    'sched':'DDIM'
}

FAST = {
    'head_type' : 'FAST',
    'hidden_projection': 'pass', # Always pass
    'use_map' : False,
}

BR = {
    'head_type':'BR',
    'hidden_projection': 'pass', # Always pass
    "net_type": "unet1D_si",
    "interpolant_type": "power3",
    "gamma_type": "(2t(t-1))^0.5",
    "epsilon_type": "1-t",
    "beta_max": 0.03,
    "t0": 1e-4,
    "T": 1,
    "clip_denoise": False,
    "pretrain": False,
    'obs_dim': 512,
    'obs_horizon': 1,
    'prior_policy': None,
    'num_denoise_steps': 5,
}

HEAD_ARGS = {
    'MLP': MLPHead,
    'MAP': MAPHead,
    'Diffusion': DiffusionHead,
    'DiffusionPolicy': DiffusionPolicyHead,
    'DiffusionPolicy2': DiffusionPolicyHead2,
    'FlowMatchingDiffusionPolicy': FlowMatchingDiffusionPolicyHead,
    'DiT': DiT,
    'BR': BR,
    'FlowMatching' : FlowMatchingHead,
    'FAST' : FAST
}

@dataclass
class ModelArguments:
    model_path: Optional[str] = field(default="remyxai/SpaceLLaVA-lite")
    action_head: str = field(default='FlowMatching')
    head_args: Dict[str, Any] = field(default_factory=lambda: FlowMatchingHead) 
    action_dim: int = field(default=7)
    action_len: int = field(default=8)
    use_state_input: bool = field(default=False)
    use_hz_input: bool = field(default=True)
    state_dim: int = field(default=8)
    # double_action: bool = field(defualt=False)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # resume?
    resume: bool = field(default=False)

    # Directory Paths
    output_dir: str = field(default='checkpoints/SpatialVLA_highlr')
    data_root_dir: str = field(default='/home/shared/rlds_datasets')
    data_mix: str = field(default='libero_object_no_noops')

    # Wandb
    wandb_project: str = field(default='SpatialVLA')
    wandb_entity: str = field(default='jellyho_')

    # LoRA
    lora_enable: bool = True
    lora_rank: int = 64
    lora_alpha: int = 128
    use_rslora: bool = False
    lora_dropout: float = 0.01
    lora_weight_path: str = ""
    lora_bias: str = "none"

    # Training
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    seed: int = field(default=42)
    batch_size: int = field(default=32)
    shuffle_buffer_size: int = field(default=10000)
    enable_autotune: bool = field(default=True)
    num_parallel_calls: int = field(default=16)
    traj_transform_threads: int = field(default=10)
    traj_read_treads: int = field(default=10)
    image_aug: bool = field(default=False)
    max_steps: int = field(default=50000)  
    save_steps: int = field(default=1000)
    log_grad: bool = field(default=False)
    log_steps: int = field(default=100)

    learning_rate: float = field(default=1e-4)
    lr_scheduler_type: str = field(default='constant')
    freeze_vision_backbone: bool = field(default=False)
    warmup_ratio: float = field(default=0.06)
    max_grad_norm: float = field(default=1.0)
    weight_decay: float = field(default=0.01)
    adam_epsilon: float= field(default=1e-8)
    gradient_accumulation_steps: int = field(default=1)

    # Quantization
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    bits: int = field(default=16)

    # etc
    group_by_modality_length: bool = field(default=False)