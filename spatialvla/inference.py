import sys
import torch
import argparse
from PIL import Image
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))

from spatialvla.mobilevlm.model.mobilevlm import load_vla, load_pretrained_model
from spatialvla.mobilevlm.conversation import conv_templates, SeparatorStyle
from spatialvla.mobilevlm.utils import disable_torch_init, process_images, tokenizer_image_token, KeywordsStoppingCriteria
from spatialvla.mobilevlm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN


class VLAModel:
    def __init__(self, model_path):
        disable_torch_init()
        self.tokenizer, self.model, self.image_processor, self.dataset_statistics = load_vla(model_path=model_path)

    def unnorm_action(self, unnorm_key, action):
        mask = self.dataset_statistics[unnorm_key]['action']['mask']
        action = np.where(
            mask,  # Condition: apply unnormalization where mask is True
            action * self.dataset_statistics[unnorm_key]['action']['std'] + self.dataset_statistics[unnorm_key]['action']['mean'],  # Unnormalized action
            action  # Original action where mask is False
        )
        return action

    def inference_prompt(self, image, prompt):
        images = [image]
        images_tensor = process_images(images, self.image_processor, {'image_aspect_ratio' : 'pad'}).to(self.model.device, dtype=torch.float16)
        prompt = f'What action should the robot take to {prompt}?'
        # prompt = 'what objects can you see?'
        conv = conv_templates[self.args.conv_mode].copy()
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
                do_sample=True if self.args.temperature > 0 else False,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                num_beams=self.args.num_beams,
                max_new_tokens=self.args.max_new_tokens,
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

    def inference_action(self, unnorm_key, image, prompt):
        images = [image]
        # Check whether this process_images is same as dataset
        images_tensor = process_images(images, self.image_processor, {'image_aspect_ratio' : 'pad'}).to(self.model.device, dtype=torch.float16)
        prompt = f'What action should the robot take to {prompt}?'

        conv = conv_templates['v1'].copy() # Hard-coded
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # Input
        input_ids = (tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda())
        with torch.inference_mode():
            action = self.model.forward(
                input_ids=input_ids,
                images=images_tensor,
                use_cache=True,
            )
        action = action.cpu().numpy()[0]
        action = self.unnorm_action(unnorm_key, action)
        return action

def inference_once(args):
    disable_torch_init()
    model_name = args.model_path.split('/')[-1]
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.load_8bit, args.load_4bit)

    images = [Image.open(args.image_file).convert("RGB")]
    images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + args.prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    # Input
    input_ids = (tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda())
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
    # Inference
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
    # Result-Decode
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="mtgv/MobileVLM-1.7B")
    parser.add_argument("--conv-mode", type=str, default="v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--load_8bit", type=bool, default=False)
    parser.add_argument("--load_4bit", type=bool, default=False)
    args = parser.parse_args()

    inference_once(args)
