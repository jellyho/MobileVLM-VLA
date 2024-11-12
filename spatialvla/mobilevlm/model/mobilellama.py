from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from spatialvla.mobilevlm.model.mobilevlm import MobileVLMMetaModel, MobileVLMMetaForCausalLM
from spatialvla.mobilevlm.model.action_heads import MLPHead, ContinuousActionHead, MAPHead
from spatialvla.mobilevlm.model.diffusion_heads import DiffusionActionHead

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

        # NLP Head Version
        if config.head_args['head_type'] == 'MLP':
            self.action_head = MLPHead(config.hidden_size, config.head_args['action_hidden_sizes'], config.action_dim * config.action_len)
        elif config.head_args['head_type'] == 'MAP':
            self.action_head = ContinuousActionHead(config.hidden_size, config.action_dim * config.action_len, config.head_args['num_heads'])
        elif config.head_args['head_type'] == 'Diffusion':
            self.action_head = DiffusionActionHead(
                config.hidden_size,
                config.action_len,
                config.action_dim,
                config.head_args['max_action'],
                config.head_args['loss_type'],
                config.head_args['time_dim'],
                config.head_args['num_blocks'],
                config.head_args['dropout_rate'],
                config.head_args['hidden_dim'],
                config.head_args['use_layer_norm'],
                config.head_args['n_diffusion_samples']
            )

        self.action_head.to()
    
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
        chat: Optional[bool] = True,
        action = None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict 

        # [batch, input_ids] this may have 

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = \
            self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

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
        loss = None
        hidden = outputs[0]
        if self.config.head_args['hidden_projection'] == 'last':
            action_hidden = hidden[:, -1] # [batch, dim]
        elif self.config.head_args['hidden_projection'] == 'mean':
            action_hidden = torch.mean(hidden, axis=1)
        elif self.config.head_args['hidden_projection'] == 'pass':
            action_hidden = hidden # [batch, token_num, dim]
        
        if action is not None:
            loss = self.action_head.loss(action_hidden, action)
            with torch.no_grad():
                predicted_action = self.action_head.predict_action(action_hidden)
            return loss, predicted_action

        predicted_action = self.action_head(action_hidden)
        predicted_action = predicted_action.reshape(-1, self.config.action_len, self.config.action_dim)

        if not chat:
            return predicted_action
        else:
            logits = self.lm_head(hidden_states)
            return CausalLMOutputWithAction(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                action=predicted_action
            )

        
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

        hidden_states = outputs[0]
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
