import sys
import torch
import argparse
from PIL import Image
from pathlib import Path
import numpy as np
from torchvision import transforms
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from spatialvla.mobilevlm.model.mobilevlm import load_vla, load_pretrained_model
from spatialvla.mobilevlm.model.bimanual import load_twinvla_from_singlevla, load_twinvla
from spatialvla.mobilevlm.conversation import conv_templates, SeparatorStyle
from spatialvla.mobilevlm.utils import disable_torch_init, process_images, tokenizer_image_token, KeywordsStoppingCriteria
from spatialvla.mobilevlm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

class VLAModel:
    def __init__(self, model_path, dtype=torch.bfloat16):
        disable_torch_init()
        self.tokenizer, self.model, self.image_processor, self.dataset_statistics = load_vla(model_path=model_path, dtype=dtype)
        self.dtype = dtype

    def unnorm_action(self, unnorm_key, action):
        mask = self.dataset_statistics[unnorm_key]['action']['mask']
        if self.model.config.head_args['head_type'] == 'FAST':
            low = self.dataset_statistics[unnorm_key]['action']['q01']
            high = self.dataset_statistics[unnorm_key]['action']['q99']
            action = np.where(
                mask,  # Condition: apply unnormalization where mask is True
                (action + 1) * (high - low + 1e-6) / 2 + low,
                action  # Original action where mask is False
            )
        else:
            action = np.where(
                mask,  # Condition: apply unnormalization where mask is True
                action * self.dataset_statistics[unnorm_key]['action']['std'] + self.dataset_statistics[unnorm_key]['action']['mean'],  # Unnormalized action
                action  # Original action where mask is False
            )
        return action

    def inference_prompt(self, image, prompt, max_new_tokens=100):
        images = [image]
        images_tensor = process_images(images, self.image_processor, {'image_aspect_ratio' : 'pad'}).to(self.model.device, dtype=self.dtype)
        conv = conv_templates['v1'].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        # Input
        input_ids = (tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda())
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        return f"{outputs.strip()}"

    def inference_action(self, unnorm_key, image, prompt, state=None, num_denoise_steps=None, hz=None):
        # image = self.transform(image)
        images = [image]
        # Check whether this process_images is same as dataset
        images_tensor = process_images(images, self.image_processor, {'image_aspect_ratio' : 'pad'}).to(self.model.device, dtype=self.dtype)
        prompt = f'What action should the robot take to {prompt}?'
        conv = conv_templates['v1'].copy() # Hard-coded
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        # Input
        input_ids = (tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda())
        
        if self.model.config.head_args['head_type'] == 'BR':
            with torch.inference_mode():
                action = self.model.predict_action_br(
                    input_ids=input_ids,
                    images=images_tensor,
                    num_denoise_steps=5,
                    hz=torch.Tensor(hz).unsqueeze(0)
                )
        elif self.model.config.head_args['head_type'] == 'FAST':
            with torch.inference_mode():
                action = self.model.predict_action_fast(
                    input_ids=input_ids,
                    images=images_tensor,
                    hz=hz
                )
        else:
            with torch.inference_mode():
                action = self.model.predict_action(
                    input_ids=input_ids,
                    images=images_tensor,
                    use_cache=True,
                    num_denoise_steps=num_denoise_steps,
                    hz=torch.tensor([hz]).unsqueeze(0).to(images_tensor.device) if hz is not None else None
                )
        action = action.float().cpu().numpy()[0]
        action = self.unnorm_action(unnorm_key, action)
        return action

class TwinVLAModel:
    def __init__(self, model_path, dtype=torch.bfloat16):
        disable_torch_init()
        self.tokenizer, self.model, self.image_processor, self.dataset_statistics = load_twinvla(model_path=model_path, dtype=dtype)
        # self.tokenizer, self.model, self.image_processor, self.dataset_statistics = load_twinvla_from_singlevla(single_model_path=model_path, dtype=dtype)
        self.dtype = dtype
        # self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(model_path)

    def unnorm_action(self, unnorm_key, action):
        mask = self.dataset_statistics[unnorm_key]['action']['mask']
        action = np.where(
            mask,  # Condition: apply unnormalization where mask is True
            action * self.dataset_statistics[unnorm_key]['action']['std'] + self.dataset_statistics[unnorm_key]['action']['mean'],  # Unnormalized action
            action  # Original action where mask is False
        )
        return action

    def inference_prompt(self, image, prompt, max_new_tokens=100):
        images = [image]
        images_tensor = process_images(images, self.image_processor, {'image_aspect_ratio' : 'pad'}).to(self.model.device, dtype=self.dtype)
        conv = conv_templates['v1'].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        # Input
        input_ids = (tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda())
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        return f"{outputs.strip()}"

    def inference_action(self, unnorm_key, image, prompt, secondary_image=None, state=None, output_attn=False, hz=None):
        images = [image]
        # Check whether this process_images is same as dataset
        images_tensor = process_images(images, self.image_processor, {'image_aspect_ratio' : 'pad'}).to(self.model.device, dtype=self.dtype)
        if secondary_image is not None:
            secondary_images = [secondary_image]
            # Check whether this process_images is same as dataset
            secondary_images_tensor = process_images(secondary_images, self.image_processor, {'image_aspect_ratio' : 'pad'}).to(self.model.device, dtype=self.dtype)            
        else:
            secondary_images_tensor = None
        prompt = f'What action should the robot take to {prompt}?'
        conv = conv_templates['v1'].copy() # Hard-coded
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        # Input
        input_ids = (tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda())
        stopping_criteria = KeywordsStoppingCriteria(['|'], self.tokenizer, input_ids)

        if self.model.config.head_args['head_type'] == 'BR':
            with torch.inference_mode():
                action = self.model.predict_action_br(
                    input_ids=input_ids,
                    images=images_tensor,
                    num_denoise_steps=5
                )
        else:
            with torch.inference_mode():
                action = self.model.predict_action(
                    input_ids=input_ids,
                    images=images_tensor,
                    images_secondary=secondary_images_tensor,
                    use_cache=True,
                    output_attn=output_attn,
                    hz=torch.tensor([hz]).unsqueeze(0).to(images_tensor.device) if hz is not None else None
                )
        if output_attn:
            attn = action[1]
            action = action[0]
        action = action.cpu().numpy()[0]
        # action = action[:, :7]
        action = self.unnorm_action(unnorm_key, action)

        if output_attn:
            return action, attn
        return action

