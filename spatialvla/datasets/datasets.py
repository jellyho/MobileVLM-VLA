"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import numpy as np
import torch
import copy
from PIL import Image

from torch.utils.data import Dataset, IterableDataset

from transformers import PreTrainedTokenizerBase
from transformers import AutoTokenizer, BitsAndBytesConfig

from spatialvla.mobilevlm.utils import disable_torch_init, process_images, tokenizer_image_token, KeywordsStoppingCriteria
# from prismatic.models.backbones.llm.prompting import PromptBuilder
# from prismatic.models.backbones.vision import ImageTransform

from spatialvla.mobilevlm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from spatialvla.mobilevlm.conversation import conv_templates, SeparatorStyle

from spatialvla.datasets.rlds.utils.data_utils import tree_map
# from prismatic.vla.action_tokenizer import ActionTokenizer
from spatialvla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from spatialvla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from spatialvla.datasets.rlds.utils.data_utils import NormalizationType
from transformers import PreTrainedTokenizerBase
# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


@dataclass
class RLDSBatchTransform:
    tokenizer: AutoTokenizer
    image_processor: Any
    window_size: int = 1
    future_action_window_size: int = 0
    use_state_input: bool = False
    action_tokenizer: PreTrainedTokenizerBase = None

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], torch.Tensor(np.array(rlds_batch["action"])).to(torch.float16)
        imgs = []
        for img in rlds_batch["observation"]["image_primary"]:
            imgs.append(Image.fromarray(img))
        # pil = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        new_img = process_images(imgs, self.image_processor, {'image_aspect_ratio' : 'pad'}).to(torch.float16)

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        conv = conv_templates['v1'].copy()
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        prompt = f'What action should the robot take to {lang}?'
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
        conv.append_message(conv.roles[1], None)
        # if self.action_tokenizer is not None:
        #     conv.append_message(conv.roles[1], self.action_tokenizer(action))
        # else:
        #     conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        #
        # Tokenize (w/ `base_tokenizer`)
        input_ids = (tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"))
            
        if self.action_tokenizer is not None:
            input_ids = torch.cat([input_ids, torch.Tensor(self.action_tokenizer.discretize(action)).to(dtype=torch.long)], axis=0)
            labels = list(input_ids)
            labels = torch.tensor(labels)
            labels[: -(self.window_size + self.future_action_window_size)] = IGNORE_INDEX
        else:
            labels = None

        proprio = torch.Tensor(rlds_batch['observation']['proprio']) if self.use_state_input else None

        return dict(pixel_values=new_img, input_ids=input_ids, labels=labels, action=action, proprio=proprio, dataset_name=dataset_name, img=img)


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        use_state_input: bool = False,
        # resize_resolution: Tuple[int, int] = (336, 336), # resize is done by image processor
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        window_size = 1,
        future_action_window_size=0,
        enable_autotune=False
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        #INFO dataset kwargs
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=use_state_input,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.NORMAL,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=window_size,                        # If we wanted to feed / predict more than one step
                future_action_window_size=future_action_window_size,                        # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
            ),
            frame_transform_kwargs=dict(
                resize_size={}, # resize will be done by image processor
                num_parallel_calls=16,        # They used 100!!! how?                  # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
            enable_autotune=enable_autotune
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
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
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out