from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import transformers
from dataset.spec import ModuleSpec

## Dataset arguments#####
dataset_kwargs = dict(
    name= "lg_cup_color_rightarm",
    data_dir= "/home/jellyho/tensorflow_datasets",
    image_obs_keys= {"primary": "image"},
    proprio_obs_key= None,
    language_key= "language_instruction",
    action_proprio_normalization_type= "normal",
    action_normalization_mask= [True, True, True, True, True, True, False],
    standardize_fn= ModuleSpec.create(
        "dataset.oxe.oxe_standardization_transforms:lg_dataset_transform_single_arm_delta_ee",
    ),
)
traj_transform_kwargs = dict(
    window_size=1, # Fix to 1
    action_horizon=1,
    goal_relabeling_strategy=None,
    task_augment_strategy="delete_task_conditioning",
    task_augment_kwargs=dict(
        keep_image_prob=0.0,
    ),
    # If the default data loading speed is too slow, try these:
    # num_parallel_calls=16,  # for less CPU-intensive ops
)
workspace_augment_kwargs = dict(
        random_resized_crop=dict(scale=[1.0, 1.0], ratio=[1.0, 1.0]),
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )
wrist_augment_kwargs = dict(
    random_brightness=[0.1],
    random_contrast=[0.9, 1.1],
    random_saturation=[0.9, 1.1],
    random_hue=[0.05],
    augment_order=[
        "random_brightness",
        "random_contrast",
        "random_saturation",
        "random_hue",
    ],
)
frame_transform_kwargs = dict(
    resize_size={
        "primary": (256, 256),  # workspace (3rd person) camera is at 256x256
    },
    image_augment_kwargs=dict(
        primary=workspace_augment_kwargs,
        wrist=wrist_augment_kwargs,
    ),
    num_parallel_calls=16
)
######## Dataset Argument ends##############

@dataclass
class ModelArguments:
    model_path: Optional[str] = field(default="remyxai/SpaceLLaVA-lite")
    action_dim: int = field(default=14)
    action_len: int = field(default=1)
    action_hidden_size: int = field(default=256)
    freeze_backbone: bool = False
    tune_mm_mlp_adapter: bool = False

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    temp_dir: str = field(default='SpatialVLA_tmp')
    output_dir: str = field(default='SpatialVLA')
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = True
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    batch_size = 32
    shuffle_buffer_size = 10000
    max_steps = 50000
    save_steps = 1000
    seed = 42
    wandb_project = 'SpatialVLA'
    wandb_entity = 'jellyho_'
    learning_rate = 0.001
    gradient_accumulation_steps = 1