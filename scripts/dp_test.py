import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import wandb
from einops import rearrange
from spatialvla.mobilevlm.model.diffusion_heads import DiffusionActionHead, FourierFeatures

# Initialize W&B
wandb.init(project="MNIST-Diffusion-Validation")

# Model Hyperparameters
digit_dim = 10               # For digit encoding
obs_dim = 32           # Input dimension is the digit
time_dim = 32                 # Fourier feature dimension for the time step
action_len = 1                # Single action output (MNIST flattened)
action_dim = 28 * 28          # Flattened MNIST image dimension
hidden_dim = 256              # Hidden dimensions for MLP
diffusion_steps = 20          # Number of diffusion steps
num_blocks = 3                # Number of residual blocks
max_action = 5.0              # Range for MNIST data (normalized to [0, 1])

# Initialize the model
model = DiffusionActionHead(
    in_dim=obs_dim,  # Now represents the Fourier-encoded time step (digit)
    action_len=action_len,
    action_dim=action_dim,
    max_action=max_action,
    time_dim=time_dim,
    num_blocks=num_blocks,
    dropout_rate=0.0,
    hidden_dim=hidden_dim,
    use_layer_norm=False,
    diffusion_steps=diffusion_steps,
    n_diffusion_samples=2,
).to('cuda' if torch.cuda.is_available() else 'cpu')

print(model)

# Fourier feature encoder
fourier_encoder = FourierFeatures(output_size=obs_dim, learnable=False).to('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1] for MNIST
    transforms.Lambda(lambda x: x.view(-1))  # Flatten the image to (784,)
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

# Training Hyperparameters
epochs = 100
learning_rate = 1e-3

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
# Helper function to log sample predictions
def log_sample_predictions(model, fourier_encoder, num_samples=1):
    model.eval()
    with torch.no_grad():
        for i in range(num_samples):
            # Select a random image
            image, label = train_dataset[i]
            image = image.unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')  # Add batch dimension
            label = torch.Tensor(label).to('cuda' if torch.cuda.is_available() else 'cpu').float().view(-1, 1)
            # Fourier encode the label (digit)
            label_encoded = fourier_encoder(label[0].reshape(1, 1))
            # print(label.shape)            
            # Run denoising prediction
            
            pred_action = model.predict_action(label_encoded)

            # Unnormalize both the original and predicted images (assuming [-1, 1] normalization)
            unnormalized_image = (image * 0.5 + 0.5).clamp(0, 1) # Original image
            unnormalized_pred = (pred_action * 0.5 + 0.5).clamp(0, 1)  # Predicted image

            # Log the original and predicted images
            wandb.log({
                f"sample_{i}_original": wandb.Image(unnormalized_image.view(28, 28).cpu().numpy(), caption="Original"),
                f"sample_{i}_predicted": wandb.Image(unnormalized_pred.view(28, 28).cpu().numpy(), caption="Predicted")
            })


model.train()
# Training loop
for epoch in range(epochs):
    running_loss = 0.0
    model.train()
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
        images = images.to('cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16)  # Flattened MNIST images
        labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16)
        # print(images.shape, labels.shape)
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # Fourier encode the digit ( time step encoding)
            labels_encoded = fourier_encoder(labels.float().view(-1, 1))
            # labels_encoded = torch.zeros(images.shape[0], 32).to('cuda')
            # Generate noisy target actions
            actions = images  # MNIST flattened images as the target
            # Compute diffusion loss
            # print(labels_encoded.shape, actions.shape)
            loss = model.loss(labels_encoded, actions.unsqueeze(1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
            
    avg_loss = running_loss / len(train_loader) / 1024
    print(f"Batch {len(train_loader)}, Loss: {avg_loss:.4f}")
    wandb.log({"batch_loss": avg_loss})
    running_loss = 0.0
    # Log predictions after training
    log_sample_predictions(model, fourier_encoder, 1)

    # Log epoch loss to W&B
    # wandb.log({"epoch_loss": running_loss / len(train_loader)})

