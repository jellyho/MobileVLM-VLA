import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from einops import rearrange
import wandb
import matplotlib.pyplot as plt

# Initialize W&B
wandb.init(project="MNIST-Diffusion")

# Diffusion MLP Model
class DiffusionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, time_dim):
        super(DiffusionMLP, self).__init__()
        self.time_emb = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, t):
        t = self.time_emb(t.unsqueeze(-1).float())
        x = torch.cat([x, t], dim=1)
        return self.mlp(x)


# Diffusion Model Trainer
class DiffusionTrainer:
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.model = model
        self.timesteps = timesteps
        self.beta_schedule = torch.linspace(beta_start, beta_end, timesteps).to(next(model.parameters()).device)
        self.alpha = 1 - self.beta_schedule
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def sample_noise(self, x, t):
        device = x.device  # Ensure all calculations are on the same device as `x`
        alpha_hat_t = self.alpha_hat[t].view(-1, 1).to(device)
        noise = torch.randn_like(x, device=device)
        noisy_x = torch.sqrt(alpha_hat_t) * x + torch.sqrt(1 - alpha_hat_t) * noise
        return noisy_x, noise

    def compute_loss(self, x, t):
        device = x.device  # Ensure all calculations are on the same device as `x`
        t = t.to(device)
        noisy_x, noise = self.sample_noise(x, t)
        predicted_noise = self.model(noisy_x, t.float())
        return nn.MSELoss()(predicted_noise, noise)

    def denoise(self, x, steps=None):
        if steps is None:
            steps = self.timesteps
        device = x.device  # Ensure all calculations are on the same device as `x`
        for t in reversed(range(steps)):
            t = torch.full((x.size(0),), t, dtype=torch.long, device=device)
            alpha_hat_t = self.alpha_hat[t].view(-1, 1).to(device)
            noise = self.model(x, t.float())
            x = (x - (1 - alpha_hat_t).sqrt() * noise) / (alpha_hat_t.sqrt())
            if t[0] > 0:
                x += torch.sqrt(self.beta_schedule[t]).view(-1, 1).to(device) * torch.randn_like(x, device=device)
        return x



# Training Loop with W&B Logging
def train(model, trainer, data_loader, optimizer, epochs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.alpha_hat.to(device)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (images, _) in enumerate(data_loader):
            images = images.view(images.size(0), -1).to(device)  # Flatten MNIST images to (batch_size, 784)
            t = torch.randint(0, trainer.timesteps, (images.size(0),), device=device)  # Random timestep
            optimizer.zero_grad()
            loss = trainer.compute_loss(images, t)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Log batch loss to W&B
            wandb.log({"batch_loss": loss.item()})

        # Log epoch loss to W&B
        avg_epoch_loss = epoch_loss / len(data_loader)
        wandb.log({"epoch_loss": avg_epoch_loss})
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss}")

        # Generate and log a sample image at the end of each epoch
        model.eval()
        sample_noise = torch.randn((1, 784), device=device)
        with torch.no_grad():
            sample = trainer.denoise(sample_noise).clamp(0, 1)
        sample_image = sample.view(28, 28).cpu().numpy()

        # Log the generated sample image to W&B
        wandb.log({"generated_sample": wandb.Image(sample_image, caption=f"Generated Image Epoch {epoch+1}")})

        # Optionally, display the generated sample image
        plt.imshow(sample_image, cmap='gray')
        plt.title(f"Generated Image Epoch {epoch+1}")
        plt.show()


# Prepare Dataset and DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten the image
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Model and Training Configurations
input_dim = 28 * 28  # Flattened MNIST images (28x28 pixels)
hidden_dim = 256
time_dim = 64
timesteps = 1000
epochs = 10
learning_rate = 1e-4

# Initialize model, trainer, and optimizer
model = DiffusionMLP(input_dim, hidden_dim, time_dim)
trainer = DiffusionTrainer(model, timesteps=timesteps)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Start training
train(model, trainer, train_loader, optimizer, epochs)

# Finalize W&B run
wandb.finish()
