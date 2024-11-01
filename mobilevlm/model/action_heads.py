import torch
import torch.nn as nn

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
            elif isinstance(m, nn.LayerNorm):
                # Set LayerNorm weights to 1 and biases to 0
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)
