import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from typing import Dict, Optional, Tuple
from spatialvla.mobilevlm.model.action_heads import MAPHead
from spatialvla.mobilevlm.model.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
############## Pytorch Version of Octo Diffusion Head #################
def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.bfloat16):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos(((t + s) / (1 + s) * np.pi * 0.5).float()) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999).to(dtype=dtype)


class FourierFeatures(nn.Module):
    def __init__(self, output_size, learnable=True, dtype=torch.bfloat16):
        super().__init__()
        self.output_size = output_size
        self.learnable = learnable
        if learnable:
            self.kernel = nn.Parameter(
                torch.randn(1, output_size // 2), requires_grad=True
            )
        self.dtype = dtype

    def forward(self, x):
        if self.learnable:
            f = 2.0 * np.pi * torch.matmul(x.float(), self.kernel).float()
        else:
            half_dim = self.output_size // 2
            f = torch.log(torch.Tensor([10000.0])) / (half_dim - 1)
            f = torch.exp(torch.arange(half_dim) * -f).to(x.device)
            f = x.float() * f
            f = f.float()
        return torch.cat([torch.cos(f).to(dtype=self.dtype), torch.sin(f).to(dtype=self.dtype)], dim=-1).float()


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation=nn.SiLU(), activate_final=False, dropout_rate=None, use_layer_norm=False):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dims[0])]
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            
            if dropout_rate is not None and dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
            
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dims[i + 1]))
    
            if i + 1 < len(hidden_dims) - 1 or activate_final:
                layers.append(activation)
                
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MLPResNetBlock(nn.Module):
    def __init__(self, features, activation=nn.SiLU(), dropout_rate=None, use_layer_norm=False):
        super().__init__()
        self.features = features
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate else None
        self.layer_norm = nn.LayerNorm(features) if use_layer_norm else None
        self.dense1 = nn.Linear(features, features * 4)
        self.dense2 = nn.Linear(features * 4, features)
        self.dense_residual = nn.Linear(features, features)

    def forward(self, x):
        residual = x
        if self.dropout:
            x = self.dropout(x)
        if self.layer_norm:
            x = self.layer_norm(x)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        if residual.shape != x.shape:
            residual = self.dense_residual(residual)
        return residual + x


class MLPResNet(nn.Module):
    def __init__(self, num_blocks, in_dim, out_dim, dropout_rate, hidden_dim, use_layer_norm=True, activation=nn.SiLU()):
        super().__init__()
        self.in_dense = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([MLPResNetBlock(hidden_dim, activation=activation, dropout_rate=dropout_rate, use_layer_norm=use_layer_norm) for _ in range(num_blocks)])       
        self.out_dense = nn.Linear(hidden_dim, out_dim)
        self.activation = activation

    def forward(self, x):
        x = self.in_dense(x)
        for layer in self.layers:
            x = layer(x)
        return self.out_dense(self.activation(x))

## Time projector from DiT-Policy
class TimeNetwork(nn.Module):
    def __init__(self, time_dim, out_dim, learnable_w=False):
        assert time_dim % 2 == 0, "time_dim must be even!"
        half_dim = int(time_dim // 2)
        super().__init__()

        w = np.log(10000) / (half_dim - 1)
        w = torch.exp(torch.arange(half_dim) * -w).float()
        self.register_parameter("w", nn.Parameter(w, requires_grad=learnable_w))

        self.out_net = nn.Sequential(
            nn.Linear(time_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        assert len(x.shape) == 1, "assumes 1d input timestep array"
        x = x[:, None] * self.w[None]
        x = torch.cat((torch.cos(x), torch.sin(x)), dim=1)
        return self.out_net(x).unsqueeze(1)

## Eps decoder from DiT-Policy
class EpsLayer(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, t): # Non-Shared modulation
        # x B, L, H
        # process the conditioning vector first
        cond = t # B, 1, H
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1) # B, 1, H
        x = x * scale + shift
        x = self.linear(x)
        return x

    # def forward(self, x, t): # Shared Modulation
    #     # x B, L, H
    #     cond = t.mean(axis=0).squeeze(0) # 1, H
    #     shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1) # 1, H
    #     x = x * scale[None] + shift[None] # B, L, H * 1 1 H
    #     x = self.linear(x) # B, L, A_dim
    #     return x

    def reset_parameters(self):
        for p in self.parameters():
            nn.init.zeros_(p)


def create_diffusion_model(obs_dim, out_dim, time_dim, num_blocks, dropout_rate, hidden_dim, use_layer_norm, dtype):
    return ScoreActor(
        FourierFeatures(time_dim, learnable=True, dtype=dtype),
        MLP(time_dim, [2 * time_dim, time_dim]),
        MLPResNet(num_blocks, (obs_dim + time_dim + out_dim), out_dim, dropout_rate, hidden_dim, use_layer_norm=use_layer_norm),
        dtype=dtype
    )


class ScoreActor(nn.Module):
    def __init__(self, time_preprocess, cond_encoder, reverse_network, dtype=torch.bfloat16):
        super().__init__()
        self.time_preprocess = time_preprocess
        self.cond_encoder = cond_encoder
        self.reverse_network = reverse_network
        self.dtype = dtype

    def forward(self, obs_enc, actions, time):
        # print(obs_enc.dtype, actions.dtype, time.dtype)
        t_ff = self.time_preprocess(time.float()) # Input : N, B, 1, Output : N, B, time_dim
        cond_enc = self.cond_encoder(t_ff) # Input : N, B, time_dim, Output: N, B, time_dim
        if len(cond_enc.shape) == 3: # training time
            obs_enc = obs_enc.unsqueeze(0).expand(cond_enc.shape[0], obs_enc.shape[0], -1) # Input : B, embedding_size, Output : N, B, embeding_size
        reverse_input = torch.cat([cond_enc, obs_enc, actions], dim=-1) # N, B, (time_dim + embedding_size + L*A)
        return self.reverse_network(reverse_input)


class DiffusionActionHead(nn.Module):
    def __init__(self, head_args, in_dim=2048, action_len=1, action_dim=7, dtype=torch.bfloat16):
        super().__init__()
        self.action_len = action_len
        self.action_dim = action_dim
        self.max_action = head_args['max_action']
        self.time_dim = head_args['time_dim']
        self.num_blocks = head_args['num_blocks']
        self.dropout_rate = head_args['dropout_rate']
        self.hidden_dim = head_args['hidden_dim']
        self.use_layer_norm = head_args['use_layer_norm']
        self.diffusion_steps = head_args['diffusion_steps']
        self.n_diffusion_samples = head_args['n_diffusion_samples']
        self.use_map = head_args['use_map']
        self.dtype = dtype

        self.diffusion_model = create_diffusion_model(
            obs_dim=self.hidden_dim, 
            out_dim=self.action_dim * self.action_len,
            time_dim=self.time_dim,
            num_blocks=self.num_blocks,
            dropout_rate=self.dropout_rate,
            hidden_dim=self.hidden_dim,
            use_layer_norm=self.use_layer_norm,
            dtype = self.dtype
        )

        if self.use_map:
            self.map = MAPHead(in_dim, num_heads=1, num_readouts=1)
            self.proj = nn.Linear(in_dim, self.hidden_dim)
        betas = cosine_beta_schedule(self.diffusion_steps).float()
        self.betas = betas
        self.alphas = 1 - betas
        self.alpha_hats = torch.cumprod(self.alphas, dim=0)

    def forward(self, embeddings, time=None, noisy_actions=None):
        if time is None or noisy_actions is None:
            # B, embedding
            time = torch.zeros((embeddings.shape[0], 1), device=embeddings.device) # (Batch , 1)
            noisy_actions = torch.zeros(
                (embeddings.shape[0], self.action_dim * self.action_len,), device=embeddings.device
            ) # (Batch, L * A)
        pred_eps = self.diffusion_model(embeddings, noisy_actions, time) # (Batch, L * A) or (N, Batch, L * A)
        return pred_eps

    # No window size
    def loss(self, embeddings, actions):
        # size of embeddings will be [Batch, embedding_size]
        if self.use_map:
            embeddings = self.map(embeddings)
            embeddings = embeddings.squeeze(1)
            embeddings = self.proj(embeddings)
        batch_size, embeddings_size = embeddings.shape
        actions_flat = rearrange(actions, "b h a -> b (h a)")
        actions_flat = actions_flat.clamp(-self.max_action, self.max_action)

        time = torch.randint(0, self.diffusion_steps, (self.n_diffusion_samples, batch_size, 1))
        noise = torch.randn(self.n_diffusion_samples, *actions_flat.shape, device=actions_flat.device)

        scale = torch.sqrt(self.alpha_hats[time]).to(actions_flat.device)
        std = torch.sqrt(1 - self.alpha_hats[time]).to(actions_flat.device)
        noisy_actions = scale * actions_flat.unsqueeze(0) + std * noise
        pred_eps = self(embeddings, time=time.to(actions_flat.device), noisy_actions=noisy_actions)

        loss = F.mse_loss(pred_eps, noise, reduction="none")
        loss = loss.mean()
        return loss

    def predict_action(self, embeddings, num_denoise_steps=None, return_history=False, dtype=torch.bfloat16):
        """
        Predict the action from the given embeddings using denoising steps.

        Args:
            embeddings (Tensor): Input tensor of embeddings, shape (Batch, embedding_dim)
            num_denoise_steps (int): Optional, number of steps for denoising. Defaults to `self.diffusion_steps`.

        Returns:
            Tensor: Predicted denoised actions of shape (Batch, action_len * action_dim)
        """
        if self.use_map:
            embeddings = self.map(embeddings)
            embeddings = embeddings.squeeze(1)
            embeddings = self.proj(embeddings)
        num_denoise_steps = num_denoise_steps or self.diffusion_steps
        batch_size = embeddings.shape[0]
        action_shape = (batch_size, self.action_len * self.action_dim)

        # Start with pure noise as the initial noisy action
        noisy_actions = torch.randn(action_shape, device=embeddings.device)
        if return_history:
            hist = []
            hist.append(noisy_actions)
        with torch.autocast(device_type='cuda', dtype=dtype):
            for step in reversed(range(num_denoise_steps)):
                # Get alpha, beta, and sqrt(1 - alpha_hat) for current step
                alpha_1 = 1 / torch.sqrt(self.alphas[step])
                alpha_2 = (1 - self.alphas[step]) / (torch.sqrt(1 - self.alpha_hats[step]))
                beta = self.betas[step]

                # Prepare time tensor for this specific step
                time_tensor = torch.full((batch_size, 1), step, device=embeddings.device, dtype=torch.long)

                # Predict the noise in the current noisy actions
                pred_eps = self.diffusion_model(embeddings, noisy_actions, time_tensor)

                # Remove the predicted noise to get a less noisy estimate of the action
                noisy_actions = alpha_1 * (noisy_actions - alpha_2 * pred_eps)
                if return_history:
                    hist.append(noisy_actions)
                # Optional noise addition for non-final steps
                if step > 0:
                    noise = torch.randn_like(noisy_actions)
                    noisy_actions = noisy_actions + torch.sqrt(beta) * noise

        # Clamp the final prediction to the max action range
        predicted_actions = noisy_actions.clamp(-self.max_action, self.max_action)
        if return_history:
            return predicted_actions, hist
        return predicted_actions

## Diffusion Policy(Chi et al.) version of Action head
class DiffusionPolicyHead(nn.Module):
    def __init__(self, head_args, in_dim=2048, action_len=1, action_dim=7, dtype=torch.bfloat16):
        super().__init__()
        self.action_len = action_len
        self.action_dim = action_dim
        self.max_action = head_args['max_action']
        self.time_dim = head_args['time_dim']
        self.hidden_dim = head_args['hidden_dim']
        self.diffusion_steps = head_args['diffusion_steps']
        self.use_map = head_args['use_map']
        self.dtype = dtype

        self.diffusion_model = ConditionalUnet1D(
            input_dim=self.action_dim,
            local_cond_dim=None,
            global_cond_dim=self.hidden_dim,
            diffusion_step_embed_dim=self.time_dim,
            down_dims=[256, 512, 1024],
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True
        )

        if self.use_map:
            self.map = MAPHead(in_dim, num_heads=1, num_readouts=1)
        self.proj = nn.Linear(in_dim, self.hidden_dim)

        self.scheduler = DDPMScheduler(
            num_train_timesteps=self.diffusion_steps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule='squaredcos_cap_v2',
            variance_type='fixed_samll',
            clip_sample=True,
            clip_sample_range=self.max_action,
            prediction_type='epsilon'
        )

    def forward(self, embeddings, time=None, noisy_actions=None):
        if time is None or noisy_actions is None:
            # B, embedding
            time = torch.zeros((embeddings.shape[0],), device=embeddings.device) # (Batch , 1)
            noisy_actions = torch.zeros(
                (embeddings.shape[0], self.action_len, self.action_dim,), device=embeddings.device
            ) # (Batch, L, A)
        pred_eps = self.diffusion_model(embeddings, noisy_actions, time) # (Batch, L, A) or (N, Batch, L, A)
        return pred_eps

    # No window size
    def loss(self, embeddings, actions):
        # size of embeddings will be [Batch, embedding_size]
        self.scheduler.set_timesteps(self.diffusion_steps)
        if self.use_map:
            embeddings = self.map(embeddings)
            embeddings = embeddings.squeeze(1)
        embeddings = self.proj(embeddings)
        batch_size, embeddings_size = embeddings.shape
        # actions_flat = rearrange(actions, "b h a -> b (h a)")
        actions = actions.clamp(-self.max_action, self.max_action)

        time = torch.randint(0, self.diffusion_steps, (batch_size,)) # B, 
        noise = torch.randn(batch_size, self.action_len, self.action_dim, device=actions.device)
        noisy_actions = self.scheduler.add_noise(actions, noise, time)
        pred_eps = self(embeddings, time=time.to(actions.device), noisy_actions=noisy_actions)

        loss = F.mse_loss(pred_eps, noise, reduction="none")
        loss = loss.mean()
        return loss

    def predict_action(self, embeddings, num_denoise_steps=None, return_history=False, dtype=torch.bfloat16):
        """
        Predict the action from the given embeddings using denoising steps.

        Args:
            embeddings (Tensor): Input tensor of embeddings, shape (Batch, embedding_dim)
            num_denoise_steps (int): Optional, number of steps for denoising. Defaults to `self.diffusion_steps`.

        Returns:
            Tensor: Predicted denoised actions of shape (Batch, action_len * action_dim)
        """
        if self.use_map:
            embeddings = self.map(embeddings)
            embeddings = embeddings.squeeze(1)
            embeddings = self.proj(embeddings)
        num_denoise_steps = num_denoise_steps or self.diffusion_steps
        batch_size = embeddings.shape[0]
        action_shape = (batch_size, self.action_len, self.action_dim)

        # Start with pure noise as the initial noisy action
        noisy_actions = torch.randn(action_shape, device=embeddings.device)
        self.scheduler.set_timesteps(num_denoise_steps)
        if return_history:
            hist = []
            hist.append(noisy_actions)
        with torch.autocast(device_type='cuda', dtype=dtype):
            for step in self.scheduler.timesteps:
                # Prepare time tensor for this specific step
                time_tensor = torch.full((batch_size,), step, device=embeddings.device, dtype=torch.long)

                # Predict the noise in the current noisy actions
                pred_eps = self.diffusion_model(embeddings, noisy_actions, time_tensor)
                noisy_actions = self.scheduler.step(pred_eps, step, noisy_actions).prev_sample

                if return_history:
                    hist.append(noisy_actions)

        # Clamp the final prediction to the max action range
        predicted_actions = noisy_actions.clamp(-self.max_action, self.max_action).reshape(batch_size, -1) # B, L*A(for compatibility)
        if return_history:
            return predicted_actions, hist
        return predicted_actions

# Meta module for DiT Implementation
class DiTModules(nn.Module):
    def __init__(self, action_dim, action_len, embed_size, head_args):
        super().__init__()
        self.register_parameter(
            "noise_pos",
            nn.Parameter(torch.empty(1, action_len, embed_size), requires_grad=True),
        )
        self.register_parameter(
            'timestep_pos',
            nn.Parameter(torch.empty(1, 1, embed_size), requires_grad=True),
        )
        nn.init.xavier_uniform_(self.noise_pos.data)
        nn.init.xavier_uniform_(self.timestep_pos.data)
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, action_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(action_dim, embed_size),
        )
        self.time_net = TimeNetwork(head_args['time_dim'], embed_size)
        self.eps_net = EpsLayer(embed_size, action_dim)
        self.diffusion_steps = head_args['diffusion_steps']

        if head_args['sched'] == 'DDPM':
            self.scheduler = DDPMScheduler(
                num_train_timesteps=self.diffusion_steps,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule='squaredcos_cap_v2',
                variance_type='fixed_small',
                clip_sample=True,
                clip_sample_range=head_args['max_action'],
                prediction_type='epsilon'
            )
        elif head_args['sched'] == 'DDIM':
            self.scheduler = DDIMScheduler(
                num_train_timesteps=self.diffusion_steps,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                clip_sample_range=head_args['max_action'],
                prediction_type='epsilon',
                set_alpha_to_one=True
            )

        self.action_dim = action_dim
        self.action_len = action_len
        self.head_args = head_args
        self.embed_size = embed_size

    def prepare_inputs_for_DiT_training(self, actions, input_ids, attention_mask, past_key_values, inputs_embeds, labels):
        B, action_len, embed_dim = actions.shape[0], self.action_len, inputs_embeds.shape[-1]

        # Calculate Noisy Actions
        timesteps = torch.randint(low=0, high=self.diffusion_steps, size=(B,), device=inputs_embeds.device)

        noise = torch.randn_like(actions, device=inputs_embeds.device)
        noisy_actions = self.scheduler.add_noise(actions, noise, timesteps)
        noisy_actions_proj = self.action_proj(noisy_actions)

        # Create tokens for noisy actions & timestep
        noisy_actions_token = noisy_actions_proj + self.noise_pos
        time_enc = self.time_net(timesteps)
        time_token = time_enc + self.timestep_pos
        DiT_tokens = torch.cat([time_token, noisy_actions_token], dim=1)  # Add time_token as the first token

        # Calculate the lengths and expand attention mask
        valid_lengths = attention_mask.sum(dim=1, keepdim=True)  # Shape: (B, 1)

        new_attention_mask = torch.cat(
            [attention_mask, torch.zeros((B, DiT_tokens.shape[1]), device=attention_mask.device, dtype=attention_mask.dtype)],
            dim=1
        )

        # Expand inputs_embeds and insert DiT_tokens
        new_inputs_embeds = torch.cat(
            [inputs_embeds, torch.zeros((B, DiT_tokens.shape[1], embed_dim), device=inputs_embeds.device, dtype=inputs_embeds.dtype)],
            dim=1
        )
        # Scatter update attention mask and inputs_embeds
        indices = valid_lengths + torch.arange(DiT_tokens.shape[1], device=inputs_embeds.device).unsqueeze(0)  # Offset per batch
        new_attention_mask.scatter_(1, indices, 1)  # Scatter updates valid tokens in the extended attention mask
        new_inputs_embeds.scatter_(1, indices.unsqueeze(-1).expand(-1, -1, embed_dim), DiT_tokens)

        return input_ids, new_attention_mask, past_key_values, new_inputs_embeds, labels, time_enc, noise
    
    def prepare_inputs_for_DiT_evaluate(self, noisy_actions, timesteps, input_ids, attention_mask, past_key_values, inputs_embeds, labels):
        B, action_len, embed_dim = noisy_actions.shape[0], self.action_len, inputs_embeds.shape[-1]
        noisy_actions_proj = self.action_proj(noisy_actions)

        # Create tokens for noisy actions & timestep
        noisy_actions_token = noisy_actions_proj + self.noise_pos
        time_enc = self.time_net(timesteps)
        time_token = time_enc + self.timestep_pos
        DiT_tokens = torch.cat([time_token, noisy_actions_token], dim=1)  # Add time_token as the first token

        # Calculate the lengths and expand attention mask
        if attention_mask is None:
            attention_mask = torch.ones((B, inputs_embeds.shape[1]), device=inputs_embeds.device, dtype=torch.long)
        valid_lengths = attention_mask.sum(dim=1, keepdim=True)  # Shape: (B, 1)
        indices = valid_lengths + torch.arange(DiT_tokens.shape[1], device=inputs_embeds.device).unsqueeze(0)

        new_attention_mask = torch.cat(
            [attention_mask, torch.zeros((B, DiT_tokens.shape[1]), device=attention_mask.device, dtype=attention_mask.dtype)],
            dim=1)
        new_attention_mask.scatter_(1, indices, 1)  # Scatter updates valid tokens in the extended attention mask

        if past_key_values is not None: # Cached
            new_inputs_embeds = DiT_tokens
            return input_ids, new_attention_mask, past_key_values, new_inputs_embeds, labels, time_enc

        # Expand inputs_embeds and insert DiT_tokens
        new_inputs_embeds = torch.cat(
            [inputs_embeds, torch.zeros((B, DiT_tokens.shape[1], embed_dim), device=inputs_embeds.device, dtype=inputs_embeds.dtype)],
            dim=1)
        new_inputs_embeds.scatter_(1, indices.unsqueeze(-1).expand(-1, -1, embed_dim), DiT_tokens)       

        return input_ids, new_attention_mask, past_key_values, new_inputs_embeds, labels, time_enc


    def loss(self, action_hidden, time_enc, noise):
        output_tokens = action_hidden[:, -self.action_len:, :]
        eps_out = self.eps_net(output_tokens, time_enc)

        loss = nn.functional.mse_loss(eps_out, noise, reduction='mean')
        return loss

    def denoise_action(self, noisy_actions, action_hidden, step, time_enc):
        output_tokens = action_hidden[:, -self.action_len:, :]
        eps_out = self.eps_net(output_tokens, time_enc)
        noisy_actions = self.scheduler.step(eps_out, step, noisy_actions).prev_sample

        return noisy_actions


