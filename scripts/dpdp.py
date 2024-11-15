import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb

# Initialize wandb
wandb.init(project="diffusion-model-mnist", entity="jellyho_")

# Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dim = 28 * 28    # Flattened size of MNIST images
hidden_dim = 256      # Hidden layer size in MLP
num_steps = 1000      # Number of diffusion steps
batch_size = 64       # Batch size
lr = 1e-3         # Learning rate

# Define MLP model
class MLP(nn.Module):
    def __init__(self, data_dim, hidden_dim=256):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(data_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim)
        )

    def forward(self, x, t):
        t_embedding = t.unsqueeze(-1).float() / num_steps  # Normalize time step
        x_t = torch.cat([x, t_embedding], dim=-1)
        return self.model(x_t)

# Diffusion process helpers
def q_sample(x_0, t, betas):
    noise = torch.randn_like(x_0)
    alpha_cumprod = torch.cumprod(1 - betas, dim=0)  # Calculate cumulative product over the whole sequence
    alpha_cumprod_t = alpha_cumprod[t].unsqueeze(-1)  # Gather alpha_cumprod for each t in the batch
    sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)

    # Apply the forward diffusion step
    x_t = sqrt_alpha_cumprod_t * x_0 + torch.sqrt(1 - sqrt_alpha_cumprod_t**2) * noise
    return x_t, noise

# Loss function
def diffusion_loss(model, x_0, t, betas):
    x_t, noise = q_sample(x_0, t, betas)
    noise_pred = model(x_t, t)
    return ((noise - noise_pred) ** 2).mean()

# Training setup
betas = torch.linspace(1e-4, 0.02, num_steps).to(device)  # Linear schedule for betas
model = MLP(data_dim=data_dim, hidden_dim=hidden_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(10):  # Training for 10 epochs
    for i, (images, _) in enumerate(train_loader):
        images = images.to(device)
        t = torch.randint(0, num_steps, (images.shape[0],)).to(device)
        
        # Compute loss
        loss = diffusion_loss(model, images, t, betas)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Log to wandb
        if i % 100 == 0:
            wandb.log({"epoch": epoch, "batch": i, "loss": loss.item()})
            print(f"Epoch {epoch}, Batch {i}: Loss = {loss.item()}")

# Sampling function (reverse process)
@torch.no_grad()
def p_sample_loop(model, shape, betas):
    x_t = torch.randn(shape).to(device)
    for t in reversed(range(num_steps)):
        t_tensor = torch.full((shape[0],), t, dtype=torch.long).to(device)
        noise_pred = model(x_t, t_tensor)
        alpha_t = 1 - betas[t]
        x_t = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
    return x_t

# Generate and log samples
samples = p_sample_loop(model, (batch_size, data_dim), betas)
samples = samples.view(-1, 1, 28, 28)  # Reshape for MNIST
wandb.log({"Generated Samples": [wandb.Image(sample) for sample in samples]})
print("Generated samples logged to wandb")
