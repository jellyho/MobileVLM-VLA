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
from spatialvla.mobilevlm.model.diffusion_heads import DiffusionActionHead, DiffusionPolicyHead, DiTModules, FlowMatchingActionHead, DiffusionPolicyHead2, FlowMatchingDiffusionPolicyHead, TimestepEmbedder
from spatialvla.mobilevlm.action_tokenizer import ActionTokenizer
from spatialvla.mobilevlm.model.bridger.model.stochastic_interpolants import StochasticInterpolants

IGNORE_INDEX = -100

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
        super(SpatialVLAForCausalLM, self).__init__(config)

        self.model = SpatialVLAModel(config)
        # For compatibility, lm_head is only used for token embedding resizing
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config = config
        self.post_init()  # Initialize weights and apply final processing
        if hasattr(config, "use_state_input") and config.use_state_input:
            state_dim = config.state_dim
            self.state_proj = nn.Sequential(
                nn.Linear(state_dim, state_dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(state_dim, config.hidden_size),
            )
            self.register_parameter(
                "state_pos",
                nn.Parameter(torch.empty(1, 1, config.hidden_size), requires_grad=True),
            )
            nn.init.xavier_uniform_(self.state_pos.data)

        if hasattr(config, "use_hz_input") and config.use_hz_input:
            self.hz_proj = TimestepEmbedder(config.hidden_size)
            self.register_parameter(
                "hz_pos",
                nn.Parameter(torch.empty(1, 1, config.hidden_size), requires_grad=True),
            )
            nn.init.xavier_uniform_(self.hz_pos.data)

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
            elif config.head_args['head_type'] == 'FlowMatching':
                self.action_head = FlowMatchingActionHead(
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
            elif config.head_args['head_type'] == 'DiffusionPolicy2':
                self.action_head = DiffusionPolicyHead2(
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
            elif config.head_args['head_type'] == 'FlowMatchingDiffusionPolicy':
                self.action_head = FlowMatchingDiffusionPolicyHead(
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
            elif config.head_args['head_type'] == 'DiT':
                self.action_head = DiTModules(
                    config.action_dim, 
                    config.action_len, 
                    config.hidden_size, 
                    config.head_args
                )
            elif config.head_args['head_type'] == 'BR':
                self.condition_projector = ContinuousActionHead(config.hidden_size, config.head_args['obs_dim'], 1)
                model_args = config.head_args
                model_args['action_dim'] = config.action_dim
                model_args['action_horizon'] = config.action_len
                self.si = StochasticInterpolants(model_args)
                self.si.load_model(model_args, device='cuda')
                self.action_head = False
            else:
                self.action_head = False
        else:
            print('no need for action head')
            self.action_head = False
        
    def get_model(self):
        return self.model

    def get_state_embeds(self, states):
        state_embeds = self.state_proj(states) + self.state_pos # B, 1, 2048
        return state_embeds

    def get_hz_embeds(self, hz):
        hz_embeds = self.hz_proj(hz) + self.hz_pos # B, 1, 2048
        return hz_embeds

    def get_action_pos_embeds(self, batch_size):
        action_pos_embeds = repeat(self.action_pos, '1 l a -> B l a', B=batch_size)
        return action_pos_embeds

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
        hz: Optional[torch.Tensor] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict 

        additional_modality = []
        ## Prepare state tokens
        if hasattr(self.config, "use_state_input") and self.config.use_state_input and states is not None:
            additional_modality.append(self.get_state_embeds(states))

        if hasattr(self.config, "use_hz_input") and self.config.use_hz_input and hz is not None:
            additional_modality.append(self.get_hz_embeds(hz))

        ## Prepare action positional token
        if self.config.head_args['head_type'] in ['Diffusion', 'FlowMatching', 'DiffusionPolicy2', 'FlowMathingDiffusionPolicy']:
            additional_modality.append(self.get_action_pos_embeds(input_ids.shape[0]))
        
        # Prepare language, image , addtional tokens
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = \
            self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images, additional_modality)

        # Prepare noisy action tokens
        if self.config.head_args['head_type'] == 'DiT':
            input_ids, attention_mask, past_key_values, inputs_embeds, labels, time_enc, noise = \
            self.action_head.prepare_inputs_for_DiT_training(actions, input_ids, attention_mask, past_key_values, inputs_embeds, labels)

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
        if self.config.head_args['head_type'] == 'BR':
            logits = self.lm_head(hidden)
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model/pipeline parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
            else:
                loss = None

            if actions is not None:
                condition = self.condition_projector(hidden)
                prior_action = torch.tensor(self.action_tokenizer.detokenize(self.action_tokenizer.discretize(actions.cpu().numpy())))
                batch_dict = {'obs':condition, 'action':actions, 'tokenized_action':prior_action}
                loss_args = {'prior_policy':'tokenized_action'}
                denoising_loss, loss_info = self.si.get_loss(batch_dict, loss_args, actions.device)
                ce_loss = loss.detach()
                loss = loss + denoising_loss

                ## For conditioning token, apply attention mask, and choose fial embedding to feed in.
                return CausalLMOutputWithPast(
                    loss=(loss, ce_loss, loss_info['v_loss'], loss_info['s_loss'], loss_info['b_loss'], denoising_loss),
                    logits=logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
            else:
                return CausalLMOutputWithPast(
                    loss=loss,
                    logits=logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
        elif self.config.head_args['head_type'] == 'FAST':
            logits = self.lm_head(hidden)
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # print(shift_logits.shape, shift_labels.shape)
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model/pipeline parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
            else:
                loss = None  
            return CausalLMOutputWithPast(
                    loss=loss,
                    logits=logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )

        # Token aggregation
        if self.config.head_args['hidden_projection'] == 'last':
            action_hidden = hidden[:, -1].contiguous() # [batch, dim]
        elif self.config.head_args['hidden_projection'] == 'mean':
            action_hidden = torch.mean(hidden, axis=1)
        elif self.config.head_args['hidden_projection'] == 'pass':
            action_hidden = hidden # [batch, token_num, dim]
        
        # Action decoding
        if self.config.head_args['head_type'] in ['Diffusion', 'FlowMatching', 'DiffusionPolicy2', 'FlowMatchingDiffusionPolicy']:
            loss = self.action_head.loss(action_hidden, actions, attention_mask=attention_mask)
        elif self.config.head_args['head_type'] == 'DiffusionPolicy':
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
        prior_actions = None, # For BR,
        hz=None
    ):
        # Prepare denoising step for DiT
        batch_size = input_ids.shape[0]

        additional_modality = []
        ## Prepare state tokens
        if hasattr(self.config, "use_state_input") and self.config.use_state_input and states is not None:
            additional_modality.append(self.get_state_embeds(states))

        if hasattr(self.config, "use_hz_input") and self.config.use_hz_input and hz is not None:
            additional_modality.append(self.get_hz_embeds(hz))

        ## Prepare action positional token ## TURN OFF for older verison of octo policy
        if self.config.head_args['head_type'] in ['Diffusion', 'FlowMatching', 'DiffusionPolicy2', 'FlowMatchingDiffusionPolicy']:
            additional_modality.append(self.get_action_pos_embeds(input_ids.shape[0]))
            
        if self.config.head_args['head_type'] == 'DiT':
            num_denoise_steps = num_denoise_steps or (self.action_head.diffusion_steps if self.config.head_args['sched'] == 'DDPM' else self.action_head.diffusion_steps // 10)
            action_shape = (batch_size, self.config.action_len, self.config.action_dim)
            noisy_actions = torch.randn(action_shape, device=input_ids.device)
            self.action_head.scheduler.set_timesteps(num_denoise_steps)
            step_iterator = self.action_head.scheduler.timesteps

        loop_num =  num_denoise_steps if self.config.head_args['head_type'] == 'DiT' else 1

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = \
                self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images, additional_modality)
        
        for i in range(loop_num):
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
            
            if self.config.head_args['head_type'] in ['Diffusion', 'FlowMatching', 'DiffusionPolicy2', 'FlowMatchingDiffusionPolicy']:
                predicted_action = self.action_head.predict_action(action_hidden, attention_mask=attention_mask, num_denoise_steps=num_denoise_steps)
            elif self.config.head_args['head_type'] == 'DiffusionPolicy':
                predicted_action = self.action_head.predict_action(action_hidden, num_denoise_steps=num_denoise_steps)
            elif self.config.head_args['head_type'] == 'DiT':
                noisy_actions = self.action_head.denoise_action(noisy_actions, action_hidden, step, time_enc)
                predicted_action = noisy_actions
            elif self.config.head_args['head_type'] == 'BR':
                condition = self.condition_projector(action_hidden)
                prior_action = torch.tensor(self.action_tokenizer.detokenize(self.action_tokenizer.discretize(prior_actions.cpu().numpy())))
                predicted_action = self.si.sample(
                    x_prior=prior_action.cuda().to(dtype=torch.bfloat16),
                    cond=condition.float().flatten(1),
                    diffuse_step=num_denoise_steps if num_denoise_steps is not None else 5
                )
            else:
                predicted_action = self.action_head(action_hidden)

        predicted_action = predicted_action.reshape(-1, self.config.action_len, self.config.action_dim)

        return predicted_action

    def predict_action_br(self, 
        input_ids: torch.LongTensor = None,
        images: Optional[torch.FloatTensor] = None,
        num_denoise_steps: Optional[int] = 5,
        states: Optional[torch.Tensor] = None,
    ):
        # GENERATE Actions with hidden state return, take -1 index hidden state
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output_ids = self.generate(
                input_ids,
                images=images,
                max_new_tokens=self.config.action_len,
                use_cache=True,
                do_sample=False,
                num_beams=1,
                top_p=None,                
            )
            
            action_token = output_ids[:, -self.config.action_len:].cpu()
            prior_actions = torch.tensor(self.action_tokenizer.detokenize(action_token), device=input_ids.device)

            model_actions = self.predict_action(
                input_ids=output_ids,
                images=images,
                use_cache=True,
                prior_actions=prior_actions,
                num_denoise_steps=num_denoise_steps
            )
            # model_actions = self.si.sample(x_prior=prior_action.to(dtype=torch.bfloat16), cond=condition.float().flatten(1), diffuse_step=5)
        return model_actions

    def predict_action_fast(self, 
        input_ids: torch.LongTensor = None,
        images: Optional[torch.FloatTensor] = None,
        states: Optional[torch.Tensor] = None,
        hzs = None,
        action_dim = None
    ):
        # GENERATE Actions with hidden state return, take -1 index hidden state
        with torch.autocast('cuda', dtype=torch.bfloat16):
            # 1, length
            input_length = input_ids.shape[1]
            output_ids = self.generate(
                input_ids,
                images=images,
                max_new_tokens=512,
                use_cache=True,
                do_sample=False,
                num_beams=1,
                top_p=None,                
            )
            
            action_token = output_ids[:, input_length:].cpu()
            actions = torch.tensor(self.action_tokenizer.detokenize(action_token, hzs, action_dim), device=input_ids.device)
        return actions

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
        position_ids = None
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
