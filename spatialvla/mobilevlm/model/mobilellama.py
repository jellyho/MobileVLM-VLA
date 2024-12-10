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

class MobileVLMConfig(LlamaConfig):
    model_type = "mobilevlm"
    

class SpatialVLAConfig(MobileVLMConfig):
    model_type = 'spatialvla'
    def __init__(self, **kwargs):
        self.action_dim = None
        self.action_len = None
        self.head_args = None
        super().__init__(**kwargs)


class MobileLlamaModel(MobileVLMMetaModel, LlamaModel):
    config_class = MobileVLMConfig

    def __init__(self, config: LlamaConfig):
        super(MobileLlamaModel, self).__init__(config)

class SpatialVLAModel(MobileVLMMetaModel, LlamaModel):
    config_class = SpatialVLAConfig

    def __init__(self, config: LlamaConfig):
        super(SpatialVLAModel, self).__init__(config)

class SpatialVLAForCausalLM(LlamaForCausalLM, MobileVLMMetaForCausalLM):
    config_class = SpatialVLAConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)

        self.model = SpatialVLAModel(config)
        # For compatibility, lm_head is only used for token embedding resizing
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config = config
        self.post_init()  # Initialize weights and apply final processing
        if hasattr(config, "use_state_input") and config.use_state_input:
            self.state_proj = nn.Sequential(
                nn.Linear(action_dim, action_dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(action_dim, embed_size),
            )
            self.register_parameter(
                "state_pos",
                nn.Parameter(torch.empty(1, 1, config.state_dim), requires_grad=True),
            )
            nn.init.xavier_uniform_(self.state_pos.data)
        if config.head_args:
            if config.head_args['head_type'] == 'MLP':
                self.action_head = MLPHead(config.hidden_size, config.head_args['action_hidden_sizes'], config.action_dim * config.action_len)
            elif config.head_args['head_type'] == 'MAP':
                self.action_head = ContinuousActionHead(config.hidden_size, config.action_dim * config.action_len, config.head_args['num_heads'])
            elif config.head_args['head_type'] == 'Diffusion':
                self.action_head = DiffusionActionHead(
                    config.head_args,
                    config.hidden_size,
                    config.action_len,
                    config.action_dim,
                    self.model.dtype
                )
                self.register_parameter(
                    "action_pos",
                    nn.Parameter(torch.empty(1, config.action_len, config.hidden_size), requires_grad=True),
                )
                nn.init.xavier_uniform_(self.action_pos.data)
            elif config.head_args['head_type'] == 'DiffusionPolicy':
                self.action_head = DiffusionPolicyHead(
                    config.head_args,
                    config.hidden_size,
                    config.action_len,
                    config.action_dim,
                    self.model.dtype
                )
            elif config.head_args['head_type'] == 'DiT':
                self.action_head = DiTModules(
                    config.action_dim, 
                    config.action_len, 
                    config.hidden_size, 
                    config.head_args
                )
        else:
            self.action_head = False
        
    def get_model(self):
        return self.model

    def forward(self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = True,
        actions: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict 

        # Prepare language, image tokens
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = \
            self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

        # Prepare state tokens
        if hasattr(config, "use_state_input") and self.config.use_state_input and state is not None:
            batch_size = inputs_embeds.shape[0]
            attention_mask = None
            inputs_embeds = torch.cat([inputs_embeds, repeat(self.state_pos, '1 l a -> B l a', B=batch_size)], axis=1)

        # Prepare noisy action tokens
        if self.config.head_args['head_type'] == 'DiT':
            input_ids, attention_mask, past_key_values, inputs_embeds, labels, time_enc, noise = \
            self.action_head.prepare_inputs_for_DiT_training(actions, input_ids, attention_mask, past_key_values, inputs_embeds, labels)
        elif self.config.head_args['head_type'] == 'Diffusion':
            batch_size = inputs_embeds.shape[0]
            attention_mask = None
            inputs_embeds = torch.cat([inputs_embeds, repeat(self.action_pos, '1 l a -> B l a', B=batch_size)], axis=1)


        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden = outputs[0].contiguous()

        # Token aggregation
        if self.config.head_args['hidden_projection'] == 'last':
            action_hidden = hidden[:, -1].contiguous() # [batch, dim]
        elif self.config.head_args['hidden_projection'] == 'mean':
            action_hidden = torch.mean(hidden, axis=1)
        elif self.config.head_args['hidden_projection'] == 'pass':
            action_hidden = hidden # [batch, token_num, dim]
        
        # Action decoding
        if self.config.head_args['head_type'] in ['Diffusion', 'DiffusionPolicy']:
            loss = self.action_head.loss(action_hidden, actions)
        elif self.config.head_args['head_type'] == 'DiT':
            loss = self.action_head.loss(action_hidden, time_enc, noise)
        else: #MLP, MAP
            predicted_action = self.action_head(action_hidden)
            predicted_action = predicted_action.reshape(-1, self.config.action_len, self.config.action_dim)
            loss = nn.functional.mse_loss(actions, predicted_action, reduction='mean')
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
    ):
        # Prepare denoising step for DiT
        batch_size = input_ids.shape[0]
        if self.config.head_args['head_type'] == 'DiT':
            num_denoise_steps = num_denoise_steps or (self.action_head.diffusion_steps if self.config.head_args['sched'] == 'DDPM' else self.action_head.diffusion_steps // 10)
            action_shape = (batch_size, self.config.action_len, self.config.action_dim)
            noisy_actions = torch.randn(action_shape, device=input_ids.device)
            self.action_head.scheduler.set_timesteps(num_denoise_steps)
            step_iterator = self.action_head.scheduler.timesteps

        loop_num =  num_denoise_steps if self.config.head_args['head_type'] == 'DiT' else 1

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = \
                self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)
        
        for i in range(loop_num):
            if hasattr(config, "use_state_input") and self.config.use_state_input and state is not None:
                batch_size = inputs_embeds.shape[0]
                attention_mask = None
                inputs_embeds = torch.cat([inputs_embeds, repeat(self.state_pos, '1 l a -> B l a', B=batch_size)], axis=1)
            if self.config.head_args['head_type'] == 'DiT':
                step = step_iterator[i]
                timesteps = torch.full((batch_size,), step, device=inputs_embeds.device, dtype=torch.long)
                new_input_ids, new_attention_mask, past_key_values, new_inputs_embeds, new_labels, time_enc = \
                self.action_head.prepare_inputs_for_DiT_evaluate(noisy_actions, timesteps, input_ids, attention_mask, past_key_values, inputs_embeds, labels)

                outputs = self.model(
                    input_ids=new_input_ids,
                    attention_mask=new_attention_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=new_inputs_embeds,
                    use_cache=use_cache,
                )
                
                if use_cache and past_key_values is None:
                    pkv = outputs.past_key_values
                    sliced_kv = []
                    n_tokens = 1 + self.config.action_len
                    for layer_kv in pkv:
                        keys, values = layer_kv
                        sliced_keys = keys[:, :, :-n_tokens, :]  # Slicing the last n_tokens
                        sliced_values = values[:, :, :-n_tokens, :]  # Slicing the last n_tokens
                        sliced_kv.append((sliced_keys, sliced_values))
                    past_key_values = tuple(sliced_kv)
            else:
                if self.config.head_args['head_type'] == 'Diffusion':
                    batch_size = inputs_embeds.shape[0]
                    attention_mask = None
                    inputs_embeds = torch.cat([inputs_embeds, repeat(self.action_pos, '1 l a -> B l a', B=batch_size)], axis=1)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                )    

            hidden = outputs[0].contiguous()

            if self.config.head_args['hidden_projection'] == 'last':
                action_hidden = hidden[:, -1].contiguous() # [batch, dim]
            elif self.config.head_args['hidden_projection'] == 'mean':
                action_hidden = torch.mean(hidden, axis=1)
            elif self.config.head_args['hidden_projection'] == 'pass':
                action_hidden = hidden # [batch, token_num, dim]
            
            if self.config.head_args['head_type'] in ['Diffusion', 'DiffusionPolicy']: # Maybe diffusion
                predicted_action = self.action_head.predict_action(action_hidden)
            elif self.config.head_args['head_type'] == 'DiT':
                noisy_actions = self.action_head.denoise_action(noisy_actions, action_hidden, step, time_enc)
                predicted_action = noisy_actions
            else:
                predicted_action = self.action_head(action_hidden)

        predicted_action = predicted_action.reshape(-1, self.config.action_len, self.config.action_dim)

        return predicted_action

        
class MobileLlamaForCausalLM(LlamaForCausalLM, MobileVLMMetaForCausalLM):
    config_class = MobileVLMConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = MobileLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()  # Initialize weights and apply final processing

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict 

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = \
            self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

        # attention_mask = None
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0].contiguous()
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

AutoConfig.register("mobilevlm", MobileVLMConfig)
AutoModelForCausalLM.register(MobileVLMConfig, MobileLlamaForCausalLM)

AutoConfig.register("spatialvla", SpatialVLAConfig)
AutoModelForCausalLM.register(SpatialVLAConfig, SpatialVLAForCausalLM)
