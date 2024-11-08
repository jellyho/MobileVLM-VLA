import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from typing import Dict, Optional, Tuple

############## Pytorch Version of Octo Diffusion Head #################

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos(((t + s) / (1 + s) * np.pi * 0.5).float()) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999).half()


class FourierFeatures(nn.Module):
    def __init__(self, output_size, learnable=True):
        super().__init__()
        self.output_size = output_size
        self.learnable = learnable
        if learnable:
            self.kernel = nn.Parameter(
                torch.randn(1, output_size // 2), requires_grad=True
            )

    def forward(self, x):
        if self.learnable:
            f = 2 * np.pi * torch.matmul(x, self.kernel).float()
        else:
            half_dim = self.output_size // 2
            f = torch.log(10000) / (half_dim - 1)
            f = torch.exp(torch.arange(half_dim) * -f)
            f = x * f
            f = f.float()
        return torch.cat([torch.cos(f).half(), torch.sin(f).half()], dim=-1)


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
    def __init__(self, num_blocks, out_dim, dropout_rate, hidden_dim, use_layer_norm=True, activation=nn.SiLU()):
        super().__init__()
        self.layers = nn.ModuleList([
            MLPResNetBlock(hidden_dim, activation=activation, dropout_rate=dropout_rate, use_layer_norm=use_layer_norm)
            for _ in range(num_blocks)
        ])
        self.out_dense = nn.Linear(hidden_dim, out_dim)
        self.activation = activation

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.out_dense(self.activation(x))


def create_diffusion_model(out_dim, time_dim, num_blocks, dropout_rate, hidden_dim, use_layer_norm):
    return ScoreActor(
        FourierFeatures(time_dim, learnable=True),
        MLP(time_dim, [2 * time_dim, time_dim]),
        MLPResNet(num_blocks, out_dim, dropout_rate, hidden_dim, use_layer_norm=use_layer_norm)
    )


class ScoreActor(nn.Module):
    def __init__(self, time_preprocess, cond_encoder, reverse_network):
        super().__init__()
        self.time_preprocess = time_preprocess
        self.cond_encoder = cond_encoder
        self.reverse_network = reverse_network

    def forward(self, obs_enc, actions, time):
        t_ff = self.time_preprocess(time) # Input : N, B, 1, Output : N, B, time_dim
        cond_enc = self.cond_encoder(t_ff) # Input : N, B, time_dim, Output: N, B, time_dim
        obs_enc = obs_enc.expand_as(cond_enc) # Input : B, embedding_size, Output : N, B, embeding_size
        reverse_input = torch.cat([cond_enc, obs_enc, actions], dim=-1) # N, B, (time_dim + embedding_size + L*A)
        return self.reverse_network(reverse_input)


class DiffusionActionHead(nn.Module):
    def __init__(self, action_len=1, action_dim=7, max_action=5.0,
                 loss_type="mse", time_dim=32, num_blocks=3, dropout_rate=0.0, hidden_dim=256,
                 use_layer_norm=True, diffusion_steps=20, n_diffusion_samples=1):
        super().__init__()
        self.action_len = action_len
        self.action_dim = action_dim
        self.max_action = max_action
        self.loss_type = loss_type
        self.time_dim = time_dim
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm
        self.diffusion_steps = diffusion_steps
        self.n_diffusion_samples = n_diffusion_samples

        self.diffusion_model = create_diffusion_model(
            self.action_dim * self.action_len,
            time_dim=self.time_dim,
            num_blocks=self.num_blocks,
            dropout_rate=self.dropout_rate,
            hidden_dim=self.hidden_dim,
            use_layer_norm=self.use_layer_norm,
        )

        betas = cosine_beta_schedule(self.diffusion_steps)
        self.betas = betas
        self.alphas = 1 - betas
        self.alpha_hats = torch.cumprod(self.alphas.float(), dim=0).half()

    def forward(self, embeddings, time=None, noisy_actions=None):
        if time is None or noisy_actions is None:
            # B, embedding
            time = torch.zeros((embeddings.shape[0], 1), device=embeddings.device) # (Batch , 1)
            noisy_actions = torch.zeros(
                (embeddings.shape[0], self.action_dim * self.action_len,), device=embeddings.device
            ) # (Batch, L * A)
        pred_eps = self.diffusion_model(embeddings, noisy_actions, time) # (Batch, L * A) or (N, Bathc, L * A)
        return pred_eps

    # No window size
    def loss(self, embeddings, actions):
        # size of embeddings will be [Batch, embedding_size]
        batch_size, embeddings_size = embeddings.shape
        actions_flat = rearrange(actions, "b h a -> b (h a)")
        actions_flat = actions_flat.clamp(-self.max_action, self.max_action)

        time = torch.randint(0, self.diffusion_steps, (self.n_diffusion_samples, batch_size, 1))
        noise = torch.randn(self.n_diffusion_samples, *actions_flat.shape, device=actions_flat.device)

        scale = torch.sqrt(self.alpha_hats[time])
        std = torch.sqrt(1 - self.alpha_hats[time])
        noisy_actions = scale * actions_flat[None] + std * noise

        pred_eps = self(embeddings, time=time, noisy_actions=noisy_actions)

        loss = F.mse_loss(pred_eps, noise, reduction="none")
        loss = loss.sum() / mask.sum()
        return loss

    def predict_action(self, embeddings, num_denoise_steps=None):
        """
        Predict the action from the given embeddings using denoising steps.

        Args:
            embeddings (Tensor): Input tensor of embeddings, shape (Batch, embedding_dim)
            num_denoise_steps (int): Optional, number of steps for denoising. Defaults to `self.diffusion_steps`.

        Returns:
            Tensor: Predicted denoised actions of shape (Batch, action_len * action_dim)
        """
        num_denoise_steps = num_denoise_steps or self.diffusion_steps
        batch_size = embeddings.shape[0]
        action_shape = (batch_size, self.action_len * self.action_dim)

        # Start with pure noise as the initial noisy action
        noisy_actions = torch.randn(action_shape, device=embeddings.device)

        for step in reversed(range(num_denoise_steps)):
            # Get alpha, beta, and sqrt(1 - alpha_hat) for current step
            alpha_hat = self.alpha_hats[step]
            beta_t = self.betas[step]
            sqrt_alpha_hat = torch.sqrt(alpha_hat)
            sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat)

            # Prepare time tensor for this specific step
            time_tensor = torch.full((batch_size, 1), step, device=embeddings.device, dtype=torch.long)

            # Predict the noise in the current noisy actions
            pred_eps = self.diffusion_model(embeddings, noisy_actions, time_tensor)

            # Remove the predicted noise to get a less noisy estimate of the action
            noisy_actions = (noisy_actions - sqrt_one_minus_alpha_hat * pred_eps) / sqrt_alpha_hat

            # Optional noise addition for non-final steps
            if step > 0:
                noise = torch.randn_like(noisy_actions)
                noisy_actions = noisy_actions + torch.sqrt(beta_t) * noise

        # Clamp the final prediction to the max action range
        # predicted_actions = noisy_actions.clamp(-self.max_action, self.max_action)
        return predicted_actions