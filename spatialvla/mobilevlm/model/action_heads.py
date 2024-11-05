import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPHead(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLPHead, self).__init__()
        
        # Define a list to hold each layer
        layers = []
        
        # First hidden layer (input to first hidden size)
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())  # Add activation
        
        # Additional hidden layers (for each pair in hidden_sizes)
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.ReLU())
        
        # Output layer (last hidden size to output)
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # Combine all layers into a sequential module
        self.model = nn.Sequential(*layers)

    def _init_weights(self):
        # Apply custom initialization to each module in self.modules()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)

class MAPHead(nn.Module):
    """Multihead Attention Pooling using PyTorch.

    Conversion of the original JAX implementation for PyTorch.
    """

    def __init__(self, input_dim, mlp_dim=None, num_heads=8, num_readouts=1):
        super(MAPHead, self).__init__()
        
        self.num_heads = num_heads
        self.num_readouts = num_readouts
        self.input_dim = input_dim
        self.mlp_dim = mlp_dim if mlp_dim is not None else 4 * input_dim

        # Multihead Attention layer
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)

        # Probe parameter (initialized like in JAX)
        self.probe = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(1, num_readouts, input_dim))
        )

        # LayerNorm layer
        self.layer_norm = nn.LayerNorm(input_dim)

        # MLP Block
        self.mlp_block = MlpBlock(input_dim, self.mlp_dim)
    
    def forward(self, x, mask=None):
        batch_size, l, d = x.shape
        probe = self.probe.expand(batch_size, -1, -1)  # Expand probe to match batch size

        if mask is not None:
            mask = mask.unsqueeze(1)  # Expand for multihead attention compatibility

        # Multihead Attention with the probe
        out, _ = self.attention(probe, x, x, key_padding_mask=mask)

        # Apply LayerNorm and MLP Block
        y = self.layer_norm(out)
        out = out + self.mlp_block(y)
        
        return out


class MlpBlock(nn.Module):
    """MLP Block with two linear layers and GELU activation."""

    def __init__(self, input_dim, hidden_dim):
        super(MlpBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

class ContinuousActionHead(nn.Module):
    def __init__(self, input_size, output_size):
        super(ContinuousActionHead, self).__init__()
        self.map_head = MAPHead(input_size)
        self.projection = nn.Linear(input_size, output_size)

    def forward(self, x):
        pooled = self.map_head(x)
        output = self.projection(pooled)
        return output

