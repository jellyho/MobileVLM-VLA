import torch
from typing import List, Optional, Tuple, Union
import time
import torch
import torch.nn as nn
from einops import repeat
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler 
from spatialvla.mobilevlm.model.mobilevlm import MobileVLMMetaModel, MobileVLMMetaForCausalLM
from spatialvla.mobilevlm.model.action_heads import MLPHead, ContinuousActionHead, MAPHead
from spatialvla.mobilevlm.model.diffusion_heads import DiffusionActionHead, DiffusionPolicyHead, DiTModules
from spatialvla.mobilevlm.action_tokenizer import ActionTokenizer
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, BitsAndBytesConfig
from spatialvla.mobilevlm.model.vision_encoder import build_vision_tower
from spatialvla.mobilevlm.model.vision_projector import build_vision_projector
from spatialvla.mobilevlm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, \
    DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from spatialvla.datasets.rlds.utils.data_utils import load_statistics_from_json
from spatialvla.mobilevlm.model.mobilellama import SpatialVLAModel, SpatialVLAForCausalLM, SpatialVLAConfig
import copy
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
import math

class TwinVLAConfig(SpatialVLAConfig):
    model_type = 'twinvla'
    def __init__(self, **kwargs):
        self.training = True
        super().__init__(**kwargs)

class TwinVLA(SpatialVLAForCausalLM):
    config_class = TwinVLAConfig

    def __init__(self, config):
        super(TwinVLA, self).__init__(config)
        self.prepared = False
        self.post_init()
        if config.training == False:
            self.prepare_twinvla()
        

    def prepare_twinvla(self):
        # make twin tower
        self._modules['model_r'] = self._modules.pop('model')
        self.model_l = copy.deepcopy(self.model_r)

        # remove one vision tower
        vision_tower_r = self._modules['model_r']._modules.pop('vision_tower')
        del vision_tower_r

        self.vision_tower = self._modules['model_l']._modules.pop('vision_tower')

        self.tower_flag = 'left'
        self.prepared = True

    def get_model(self):
        if self.prepared:
            if self.tower_flag == 'left':
                return self.model_l
            else:
                return self.model_r
        else:
            return self.model
    
    def get_vision_tower(self):
        if self.prepared:
            return self.vision_tower
        else:
            return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.vision_tower(images)
        if self.tower_flag == 'left':
            image_features = self.model_l.mm_projector(image_features).contiguous()
        elif self.tower_flag == 'right':
            image_features = self.model_r.mm_projector(image_features).contiguous()

        return image_features

    def prepare_inputs_labels_for_multimodal_twinvla(self, input_ids, attention_mask, past_key_values, labels, images, additional_modality=None):
        self.tower_flag = 'left'
        inputs_left = self.prepare_inputs_labels_for_multimodal(
            input_ids,
            attention_mask,
            past_key_values,
            labels,
            images,
            additional_modality
        )
        self.tower_flag = 'right'
        inputs_right = self.prepare_inputs_labels_for_multimodal(
            input_ids,
            attention_mask,
            past_key_values,
            labels,
            images,
            additional_modality
        )
        return inputs_left, inputs_right

    def in_ln(self, model, idx):
        return model.layers[idx].input_layernorm

    def attn(self, model, idx):
        return model.layers[idx].self_attn

    def out_ln(self, model, idx):
        return model.layers[idx].post_attention_layernorm

    def mlp(self, model, idx):
        return model.layers[idx].mlp

    def joint_forward(self, 
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = True,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = True,
        actions: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict 

        additional_modality = []

        ## Prepare state tokens
        if hasattr(self.config, "use_state_input") and self.config.use_state_input and states is not None:
            additional_modality.append(self.get_state_embeds(states))

        ## Prepare action positional token ## TURN OFF for older verison of octo policy
        if self.config.head_args['head_type'] == 'Diffusion':
            additional_modality.append(self.get_action_pos_embeds(input_ids.shape[0]))

        inputs_lr = self.prepare_inputs_labels_for_multimodal_twinvla(
            input_ids,
            attention_mask,
            past_key_values,
            labels, 
            images, 
            additional_modality
        )

        # pos_ids and attn_mask caluation only once ###
        batch_size, seq_length, _ = inputs_lr[0][3].shape
        device =  inputs_lr[0][3].device
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        if inputs_lr[0][1] is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=device
            )
        atm= self.model_r._prepare_decoder_attention_mask(
            inputs_lr[0][1], (batch_size, seq_length), inputs_lr[0][3], past_key_values_length
        )
        atm = torch.cat([atm, atm], axis=2)
        attention_mask = torch.cat([atm, atm], axis=3)
    
        ######

        models = [self.model_l, self.model_r]
        hidden_statess = []
        ## prepare inputs
        for inputs in inputs_lr:
            hidden_statess.append(inputs[3])

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        ## norm, attn, norm, mlp
        for layer_idx in range(24):
            # input_layernorm
            residuals = []
            for idx, model in enumerate(models):
                residuals.append(hidden_statess[idx])
                hidden_statess[idx] = self.in_ln(model, layer_idx)(hidden_statess[idx])

            # attention
            q_states, k_states, v_states = [], [], []
            # pre_attn
            for idx, model in enumerate(models):
                hidden_states = hidden_statess[idx]
                bsz, q_len, _ = hidden_states.size()
                attn = self.attn(model, layer_idx)

                query_states = attn.q_proj(hidden_states)
                key_states = attn.k_proj(hidden_states)
                value_states = attn.v_proj(hidden_states)

                query_states = query_states.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
                key_states = key_states.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
                value_states = value_states.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)

                kv_seq_len = key_states.shape[-2]

                cos, sin = attn.rotary_emb(value_states, seq_len=kv_seq_len)
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

                key_states = repeat_kv(key_states, attn.num_key_value_groups)
                value_states = repeat_kv(value_states, attn.num_key_value_groups)

                q_states.append(query_states)
                k_states.append(key_states)
                v_states.append(value_states)

                head_dim = attn.head_dim
                hidden_size = attn.hidden_size
            
            # eager_attention_forward
            new_q_states = torch.cat(q_states, axis=-2)
            new_k_states = torch.cat(k_states, axis=-2)
            new_v_states = torch.cat(v_states, axis=-2)

            attn_weights = torch.matmul(new_q_states, new_k_states.transpose(2, 3)) / math.sqrt(head_dim)
            attn_weights = attn_weights + attention_mask
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, new_v_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, -1, hidden_size)

            attn_outputs = torch.split(attn_output, [q_len, q_len], dim=1)

            for idx, model in enumerate(models):
                hidden_statess[idx] = self.attn(model, layer_idx).o_proj(attn_outputs[idx]) + residuals[idx]

            if output_attentions:
                all_self_attns += (attn_weights, )
            ## attn finish

            ## post attn
            for idx, model in enumerate(models):
                hidden_states = hidden_statess[idx]
                residuals = hidden_states
                hidden_states = self.out_ln(model, layer_idx)(hidden_states)
                hidden_states = self.mlp(model, layer_idx)(hidden_states)
                hidden_states = hidden_states + residuals
                hidden_statess[idx] = hidden_states
        
        outputs = []
        for idx, model in enumerate(models):
            outputs.append(model.norm(hidden_statess[idx]))

        return outputs, all_hidden_states, all_self_attns, inputs_lr, attention_mask

    def forward(self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = True,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = True,
        actions: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
    ):
        outputs, all_hidden_states, all_self_attns, inputs_lr, attention_mask = self.joint_forward(
            input_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            images,
            return_dict,
            actions,
            states
        )

        loss = 0
        for idx in range(2):
            hidden =  outputs[idx]
            # Token aggregation
            if self.config.head_args['hidden_projection'] == 'last':
                action_hidden = hidden[:, -1].contiguous() # [batch, dim]
            elif self.config.head_args['hidden_projection'] == 'mean':
                action_hidden = torch.mean(hidden, axis=1)
            elif self.config.head_args['hidden_projection'] == 'pass':
                action_hidden = hidden # [batch, token_num, dim]

            # Actions Batch, len, dim

            # Action decoding
            if self.config.head_args['head_type'] == 'Diffusion':
                loss += self.action_head.loss(action_hidden, actions[:, :, idx*self.config.action_dim:(idx+1)*self.config.action_dim], attention_mask=inputs_lr[idx][1])
            elif self.config.head_args['head_type'] == 'DiffusionPolicy':
                loss += self.action_head.loss(action_hidden, actions[:, :, idx*self.config.action_dim:(idx+1)*self.config.action_dim])
            else: #MLP, MAP
                predicted_action = self.action_head(action_hidden)
                predicted_action = predicted_action.reshape(-1, self.config.action_len, self.config.action_dim)
                loss += nn.functional.mse_loss(actions, predicted_action, reduction='mean')

        return loss

    def predict_action(self, 
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        num_denoise_steps: Optional[int] = None,
        states: Optional[torch.Tensor] = None,
        prior_actions = None
    ):
        output_attentions = False
        output_hidden_states = False
        return_dict=False
        outputs, all_hidden_states, all_self_attns, inputs_lr, attention_mask = self.joint_forward(
            input_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            images,
            return_dict,
            prior_actions,
            states
        )

        predicted_actions = []
        for idx in range(2):
            hidden =  outputs[idx]
            # Token aggregation
            if self.config.head_args['hidden_projection'] == 'last':
                action_hidden = hidden[:, -1].contiguous() # [batch, dim]
            elif self.config.head_args['hidden_projection'] == 'mean':
                action_hidden = torch.mean(hidden, axis=1)
            elif self.config.head_args['hidden_projection'] == 'pass':
                action_hidden = hidden # [batch, token_num, dim]

            if self.config.head_args['head_type'] == 'Diffusion':
                predicted_action = self.action_head.predict_action(action_hidden, attention_mask=inputs_lr[idx][1])
            elif self.config.head_args['head_type'] == 'DiffusionPolicy':
                predicted_action = self.action_head.predict_action(action_hidden)
            else:
                predicted_action = self.action_head(action_hidden)

            predicted_action = predicted_action.reshape(-1, self.config.action_len, self.config.action_dim)
            predicted_actions.append(predicted_action)

        final_actions = torch.cat(predicted_actions, dim=-1)
        return final_actions


def load_twinvla_from_singlevla(single_model_path, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", dtype=torch.bfloat16):
    kwargs = {}
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = dtype

    print('Loading with', dtype)
    tokenizer = AutoTokenizer.from_pretrained(single_model_path, use_fast=False)
    config = TwinVLAConfig.from_pretrained(single_model_path)

    config.model_type='twinvla'
    model = TwinVLA.from_pretrained(single_model_path, config=config, low_cpu_mem_usage=False, **kwargs)
    model.to(device)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if 'v2' in getattr(model.config, "mm_projector_type", "ldpnet"):
        vision_tower.load_image_processor()
    elif not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=dtype)

    model.get_model().mm_projector.to(device=device, dtype=dtype)

    dataset_statistics = None

    if model.action_head:
        model.action_head.to(device=device)
    dataset_statistics = load_statistics_from_json(single_model_path)

    image_processor = vision_tower.image_processor

    if model.config.head_args['head_type'] == 'BR':
        action_tokenizer = ActionTokenizer(tokenizer)
        model.action_tokenizer = action_tokenizer
        model.si.load_ema(model_path)

    model.prepare_twinvla()
    model.config.trainig = False

    return tokenizer, model, image_processor, dataset_statistics

def load_twinvla(model_path, load_8bit=False, load_4bit=False, device_map='auto', device='cuda', dtype=torch.bfloat16):
    kwargs = {}
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = dtype

    print('Loading with', dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    config = TwinVLAConfig.from_pretrained(model_path)

    model = TwinVLA.from_pretrained(model_path, config=config, low_cpu_mem_usage=False, **kwargs)
    model.to(device)

    vision_tower = model.get_vision_tower()
    if 'v2' in getattr(model.config, "mm_projector_type", "ldpnet"):
        vision_tower.load_image_processor()
    elif not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=dtype)

    if model.action_head:
        model.action_head.to(device=device)
    # dataset_statistics = load_statistics_from_json(model_path)

    image_processor = vision_tower.image_processor

    return tokenizer, model, image_processor, None