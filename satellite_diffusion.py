import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
import math
import json
import pandas as pd

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Verify CUDA is available
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA Version: {torch.version.cuda}')
    torch.cuda.set_device(0)

class EuroSATDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Check if directory exists
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset directory not found: {root_dir}")
        
        # Get all class directories
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        if not self.classes:
            raise ValueError(f"No class directories found in {root_dir}. Please ensure the dataset is properly extracted.")
            
        print(f"Found classes: {self.classes}")
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Create train/val split
        self.image_paths = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            images = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]  # EuroSAT uses .jpg files
            
            if not images:
                print(f"Warning: No .jpg images found in {class_dir}")
                continue
                
            print(f"Found {len(images)} images in class {class_name}")
            
            # Split into train/val (80/20)
            split_idx = int(len(images) * 0.8)
            if split == 'train':
                selected_images = images[:split_idx]
            else:
                selected_images = images[split_idx:]
            
            for img_name in selected_images:
                self.image_paths.append(os.path.join(class_name, img_name))
                self.labels.append(self.class_to_idx[class_name])
        
        if not self.image_paths:
            raise ValueError(f"No images found in the dataset. Please check if the dataset is properly downloaded and extracted.")
            
        print(f'Loaded {split} dataset with {len(self.image_paths)} images')
        print(f'Number of classes: {len(self.classes)}')
        print(f'Classes: {self.classes}')
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f'Image not found: {img_path}')
            
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels) if time_emb_dim else None
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, t=None):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        
        if self.time_mlp and t is not None:
            time_emb = self.time_mlp(t)[:, :, None, None]
            h = h + time_emb
            
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        
        return h + self.shortcut(x)

class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=3, model_channels=32, out_channels=3, num_classes=10):
        super().__init__()
        
        # Time embedding
        time_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Class embedding
        self.label_embed = nn.Embedding(num_classes, time_dim)
        
        # Encoder (reduced depth)
        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        self.down1 = ResidualBlock(model_channels, model_channels*2, time_dim)
        self.down2 = ResidualBlock(model_channels*2, model_channels*4, time_dim)
        
        # Middle (reduced)
        self.middle1 = ResidualBlock(model_channels*4, model_channels*4, time_dim)
        
        # Decoder (reduced depth)
        self.up1 = ResidualBlock(model_channels*8, model_channels*2, time_dim)
        self.up2 = ResidualBlock(model_channels*4, model_channels, time_dim)
        
        self.conv_out = nn.Conv2d(model_channels, out_channels, 3, padding=1)
        
    def forward(self, x, timesteps, labels):
        # Time embedding
        t_emb = self.time_embed(timestep_embedding(timesteps, self.time_embed[0].in_features))  # Fixed dimension
        
        # Label embedding
        l_emb = self.label_embed(labels)
        emb = t_emb + l_emb
        
        # Encoder
        h = self.conv_in(x)
        h1 = self.down1(h, emb)
        h2 = self.down2(h1, emb)
        
        # Middle
        h = self.middle1(h2, emb)
        
        # Decoder with skip connections
        h = self.up1(torch.cat([h, h2], dim=1), emb)
        h = self.up2(torch.cat([h, h1], dim=1), emb)
        
        return self.conv_out(h)

def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class DiffusionModel:
    def __init__(self, timesteps=100):
        """Initialize the diffusion process parameters."""
        self.timesteps = timesteps
        
        # Define beta schedule
        self.betas = linear_beta_schedule(timesteps)
        
        # Pre-calculate different terms for closed form
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
        # Move tensors to device
        if torch.cuda.is_available():
            self.betas = self.betas.to(device)
            self.alphas = self.alphas.to(device)
            self.alphas_cumprod = self.alphas_cumprod.to(device)
            self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
            self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
            self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
            self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
            self.posterior_variance = self.posterior_variance.to(device)
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process (adding noise)."""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    @torch.no_grad()
    def p_sample(self, model, x, t, labels):
        """Reverse diffusion process (removing noise)."""
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        
        # Equation 11 in the paper
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t, labels) / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, model, shape, labels, device):
        """Generate samples by iteratively denoising."""
        b = shape[0]
        img = torch.randn(shape, device=device)
        
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step'):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, labels)
        return img
    
    def p_losses(self, denoise_model, x_start, t, labels, noise=None):
        """Calculate the loss for training."""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, t, labels)
        
        loss = F.mse_loss(noise, predicted_noise)
        return loss

def linear_beta_schedule(timesteps):
    """Linear schedule of beta values for the diffusion process."""
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def extract(a, t, x_shape):
    """Extract appropriate timestep values from a tensor."""
    batch_size = t.shape[0]
    t = t.to(a.device)  # Ensure t is on the same device as a
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

class SuperResolution:
    def __init__(self, model, diffusion):
        self.model = model
        self.diffusion = diffusion
    
    @torch.no_grad()
    def enhance(self, low_res_image, scale_factor=2, label=None):
        """Enhance a low-resolution image using the diffusion model."""
        if label is None:
            label = torch.zeros(1, device=device, dtype=torch.long)
        
        # Upsample the low-res image
        upsampled = F.interpolate(low_res_image, scale_factor=scale_factor, mode='bicubic')
        
        # Use the diffusion model to improve quality
        t = torch.zeros(1, device=device, dtype=torch.long)
        enhanced = self.model(upsampled, t, label.repeat(upsampled.shape[0]))
        
        return enhanced

class SemanticSegmentation:
    def __init__(self, model, diffusion, num_classes):
        self.model = model
        self.diffusion = diffusion
        self.num_classes = num_classes
        
    @torch.no_grad()
    def segment(self, image):
        # Get model output
        t = torch.zeros(1, device=image.device, dtype=torch.long)
        output = self.model(image, t, torch.zeros_like(t))
        # Take the argmax across channels as a pseudo-class prediction
        mask = output.argmax(dim=1, keepdim=True)  # shape: [B, 1, H, W]
        return mask

def train_model(model, diffusion, train_loader, val_loader, n_epochs=100):
    """Train the diffusion model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{n_epochs}') as pbar:
            for batch, (images, labels) in enumerate(pbar):
                # Move data to GPU
                images = images.to(device)
                labels = labels.to(device)
                
                # Sample random timesteps
                t = torch.randint(0, diffusion.timesteps, (images.shape[0],), device=device).long()
                
                # Calculate loss
                loss = diffusion.p_losses(model, images, t, labels)
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1} average loss: {avg_loss:.4f}')
        
        # Generate samples every 5 epochs
        if (epoch + 1) % 5 == 0:
            generate_samples(model, diffusion, 4, train_loader.dataset.classes)

@torch.no_grad()
def generate_samples(model, diffusion, num_samples, classes, labels=None):
    """Generate synthetic satellite images."""
    model.eval()
    
    if labels is None:
        labels = torch.randint(0, len(classes), (num_samples,), device=device)
    
    # Sample images
    samples = diffusion.p_sample_loop(
        model,
        shape=(num_samples, 3, 64, 64),
        labels=labels,
        device=device
    )
    
    # Display results
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(samples, nrow=int(num_samples**0.5)).permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5)
    print('Generated labels:', [classes[l] for l in labels.cpu().numpy()])
    plt.show()
    
    return samples

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Define dataset path
    dataset_path = 'C:/Inhouse Project/Datas/eurosat-dataset/EuroSAT'
    
    # Check if directory exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Reduced image size for faster training
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Initialize datasets
    train_dataset = EuroSATDataset(
        root_dir=dataset_path,
        split='train',
        transform=transform
    )
    
    val_dataset = EuroSATDataset(
        root_dir=dataset_path,
        split='validation',
        transform=transform
    )
    
    # Create data loaders with smaller batch size
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)  # Reduced batch size
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # Initialize model and diffusion process
    model = ConditionalUNet(
        in_channels=3,
        model_channels=32,  # Reduced model size
        out_channels=3,
        num_classes=len(train_dataset.classes)
    ).to(device)
    
    diffusion = DiffusionModel(timesteps=100)  # Reduced timesteps
    
    # Train the model with just 1 epoch
    print("Training the model...")
    train_model(model, diffusion, train_loader, val_loader, n_epochs=1)
    
    # Save the trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'diffusion': diffusion
    }, 'satellite_diffusion_model.pth')
    
    # Test super-resolution
    print("\nTesting super-resolution...")
    sr_model = SuperResolution(model, diffusion)
    with torch.no_grad():
        real_image = next(iter(val_loader))[0][:1].to(device)
        low_res = F.interpolate(real_image, scale_factor=0.5, mode='bicubic')
        enhanced = sr_model.enhance(low_res, scale_factor=2)
        
        # Display results
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        for ax, img, title in zip([ax1, ax2, ax3], 
                                [real_image, low_res, enhanced],
                                ['Original', 'Low-res', 'Enhanced']):
            ax.imshow(img[0].permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5)
            ax.set_title(title)
            ax.axis('off')
        plt.show()
    
    # Test pseudo-semantic segmentation
    print("\nTesting pseudo-semantic segmentation...")
    seg_model = SemanticSegmentation(model, diffusion, len(train_dataset.classes))
    with torch.no_grad():
        real_image = next(iter(val_loader))[0][:1].to(device)
        mask = seg_model.segment(real_image)
        # Display the argmax of the mask as a segmentation map
        plt.figure(figsize=(6, 6))
        plt.imshow(mask[0, 0].cpu(), cmap='tab10')
        plt.title('Pseudo-Segmentation')
        plt.axis('off')
        plt.show()

    # Generate synthetic images for each class
    print("\nGenerating synthetic images for each class...")
    for class_idx, class_name in enumerate(train_dataset.classes):
        print(f"Class: {class_name}")
        generate_samples(model, diffusion, num_samples=4, classes=train_dataset.classes, labels=torch.full((4,), class_idx, device=device))

if __name__ == "__main__":
    main() 