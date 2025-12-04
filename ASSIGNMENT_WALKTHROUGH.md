# AS4: Image Generation Models - Complete Walkthrough

**Due: Tuesday, Dec 9 @5:30pm**

This guide provides a step-by-step walkthrough to complete the assignment on designing, training, and evaluating image generation models (DCGAN and Diffusion Models).

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Timeline (3-Day Plan)](#timeline-3-day-accelerated-plan)
4. [Task 1: Environment and Container Setup](#task-1-environment-and-container-setup)
5. [Task 2: DCGAN Implementation](#task-2-dcgan-implementation)
6. [Task 3: Diffusion Model Analysis](#task-3-diffusion-model-analysis)
7. [Task 4: Comparison and Evaluation](#task-4-comparison-and-evaluation)
8. [Task 5: Documentation and Report](#task-5-documentation-and-report)
9. [Submission Checklist](#submission-checklist)
10. [Troubleshooting](#troubleshooting)

---

## Overview

You will:
1. Set up a GPU environment using NVIDIA NGC container + Apptainer + Slurm
2. Implement and train a DCGAN on MNIST/Fashion-MNIST/CIFAR-10
3. Run and analyze a diffusion model
4. Compare both approaches
5. Document everything in a report

---

## Prerequisites

- Access to a Slurm-based GPU cluster
- Basic knowledge of PyTorch/TensorFlow
- Understanding of CNNs and GANs from lectures
- Familiarity with Linux command line

---

## Timeline (3-Day Accelerated Plan)

### Day 1: Environment Setup + DCGAN Implementation (8-10 hours)
**Morning (3-4 hours):**
- Test container and GPU access (30 min)
- Create project structure (15 min)
- Implement DCGAN generator and discriminator (2-3 hours)
- Set up training script with basic monitoring (1 hour)

**Afternoon/Evening (5-6 hours):**
- Launch DCGAN training job (runs 3-4 hours in background)
- While training: Prepare diffusion model setup
- Install diffusion libraries
- Test pretrained diffusion model
- Monitor DCGAN training progress

### Day 2: Diffusion Model + Initial Analysis (8-10 hours)
**Morning (4-5 hours):**
- Review DCGAN results from overnight
- Implement diffusion training script (2-3 hours)
- Launch diffusion training job (runs 4-6 hours in background)
- Generate DCGAN samples and save outputs

**Afternoon/Evening (4-5 hours):**
- While diffusion trains: Start comparison script
- Implement FID/IS metric calculations
- Create visualization code
- Monitor diffusion training
- Generate initial comparisons

### Day 3: Final Analysis + Documentation (6-8 hours)
**Morning (3-4 hours):**
- Review all results
- Generate final samples from both models
- Complete quantitative comparisons
- Create all visualizations and plots

**Afternoon (3-4 hours):**
- Write technical report
- Document all findings
- Organize code and outputs
- Final testing and verification
- Submit assignment

**Key Strategy for 3-Day Timeline:**
- Use smaller datasets (MNIST instead of CIFAR-10) for faster training
- Reduce epochs (30-40 instead of 50+) but monitor quality
- Run training jobs overnight/in background
- Work on documentation and analysis while models train
- Focus on core requirements, skip optional explorations

---

## Task 1: Environment and Container Setup

### Step 1.1: Select NVIDIA NGC Container

1. Visit [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/)
2. Browse to Containers → PyTorch
3. Recommended: `nvcr.io/nvidia/pytorch:25.09-py3`
   - Includes PyTorch 2.5, CUDA 12.6, Python 3.10
   - Has most required libraries pre-installed

**Document in README:**
```markdown
## Container Information
- Base Image: nvcr.io/nvidia/pytorch:25.09-py3
- PyTorch Version: 2.5
- CUDA Version: 12.6
- Python Version: 3.10
```

### Step 1.2: Test Apptainer Access

```bash
# Check if Apptainer is available
apptainer --version

# If not available, load module (cluster-specific)
module load apptainer
# or
module load singularity
```

### Step 1.3: Pull the Container

Create a directory for containers:
```bash
mkdir -p /home1/michael2024/ML_Course/container
cd /home1/michael2024/ML_Course/container

# Pull the NVIDIA container (this may take 10-20 minutes)
apptainer pull docker://nvcr.io/nvidia/pytorch:25.09-py3
```

This creates a `.sif` file (e.g., `pytorch_25.09-py3.sif`)

### Step 1.4: Create a Test Slurm Job

Create `test_gpu.slurm`:
```bash
#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --output=logs/test_gpu_%j.out
#SBATCH --error=logs/test_gpu_%j.err
#SBATCH --partition=gpu          # Use your cluster's GPU partition name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --mem=16G
#SBATCH --time=00:10:00

# Create logs directory
mkdir -p logs

# Load Apptainer if needed
# module load apptainer

# Path to your container
CONTAINER=/home1/michael2024/ML_Course/container/pytorch_25.09-py3.sif

# Test GPU access
apptainer exec --nv $CONTAINER python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
"
```

### Step 1.5: Submit and Verify

```bash
# Create logs directory
mkdir -p logs

# Submit the job
sbatch test_gpu.slurm

# Check job status
squeue -u $USER

# View output when complete
cat logs/test_gpu_*.out
```

**Expected Output:**
```
PyTorch version: 2.5.0
CUDA available: True
CUDA version: 12.6
GPU device: [Your GPU Name]
Number of GPUs: 1
```

### Step 1.6: Create Project Structure

```bash
cd ~/ML_Course/as4
mkdir -p {data,models,outputs,logs,scripts,notebooks}

# Project structure:
# as4/
# ├── data/              # Datasets
# ├── models/            # Saved model checkpoints
# ├── outputs/           # Generated images
# ├── logs/              # Slurm logs
# ├── scripts/           # Python scripts
# ├── notebooks/         # Jupyter notebooks (optional)
# ├── *.slurm           # Slurm job files
# └── README.md         # Documentation
```

### Step 1.7: Document Container Setup

Update your `README.md` with:
- Container image and version
- How to pull the container
- How to run test jobs
- Any cluster-specific instructions

✅ **Checkpoint:** You should now have a working GPU environment with the NVIDIA container.

---

## Task 2: DCGAN Implementation

### Step 2.1: Understand DCGAN Architecture

**Key Components:**

1. **Generator**: Converts noise → image
   - Input: Random noise vector (e.g., 100-dimensional)
   - Uses transpose convolutions to upsample
   - Output: Generated image (e.g., 28×28 or 32×32)

2. **Discriminator**: Classifies real vs. fake
   - Input: Image
   - Uses regular convolutions to downsample
   - Output: Probability (real or fake)

**DCGAN Guidelines:**
- Use batch normalization (except generator output and discriminator input)
- Use ReLU in generator (except output uses Tanh)
- Use LeakyReLU in discriminator
- Use strided convolutions instead of pooling

### Step 2.2: Choose Dataset

Pick one dataset:

**Option A: MNIST (Recommended for beginners)**
- Grayscale, 28×28 images
- Simple, trains quickly
- Good for debugging

**Option B: Fashion-MNIST**
- Same format as MNIST
- More challenging patterns
- Better for demonstrating capability

**Option C: CIFAR-10**
- Color (RGB), 32×32 images
- More complex
- Requires more training time

### Step 2.3: Implement DCGAN

Create `scripts/dcgan.py`:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Hyperparameters
LATENT_DIM = 100
IMAGE_SIZE = 28  # 28 for MNIST, 32 for CIFAR-10
CHANNELS = 1     # 1 for MNIST, 3 for CIFAR-10
BATCH_SIZE = 128
NUM_EPOCHS = 50
LR = 0.0002
BETA1 = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim=100, channels=1):
        super(Generator, self).__init__()
        
        # For 28x28 images (MNIST)
        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 256, 7, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # State: 256 x 7 x 7
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # State: 128 x 14 x 14
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # State: 64 x 28 x 28
            
            nn.Conv2d(64, channels, 3, 1, 1),
            nn.Tanh()
            # Output: channels x 28 x 28
        )
    
    def forward(self, x):
        return self.main(x)


# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, channels=1):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # Input: channels x 28 x 28
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 64 x 14 x 14
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 128 x 7 x 7
            
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 256 x 4 x 4
            
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Output: 1 x 1 x 1
        )
    
    def forward(self, x):
        return self.main(x).view(-1, 1)


# Weight initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train_dcgan(dataset_name='mnist', num_epochs=50, save_dir='outputs/dcgan'):
    """
    Train DCGAN on specified dataset
    """
    print(f"Training DCGAN on {dataset_name}")
    print(f"Device: {DEVICE}")
    
    # Create output directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/samples", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Load dataset
    if dataset_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        dataset = datasets.MNIST(root='./data', train=True, 
                                download=True, transform=transform)
        channels = 1
    elif dataset_name.lower() == 'fashion-mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        dataset = datasets.FashionMNIST(root='./data', train=True,
                                       download=True, transform=transform)
        channels = 1
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, 
                          shuffle=True, num_workers=4)
    
    # Initialize models
    generator = Generator(LATENT_DIM, channels).to(DEVICE)
    discriminator = Discriminator(channels).to(DEVICE)
    
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # Loss and optimizers
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, 0.999))
    
    # Fixed noise for visualization
    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=DEVICE)
    
    # Training loop
    print("Starting Training...")
    G_losses = []
    D_losses = []
    
    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(DEVICE)
            
            # Labels
            real_labels = torch.ones(batch_size, 1, device=DEVICE)
            fake_labels = torch.zeros(batch_size, 1, device=DEVICE)
            
            # ============================================
            # Train Discriminator
            # ============================================
            optimizer_D.zero_grad()
            
            # Real images
            output_real = discriminator(real_images)
            loss_D_real = criterion(output_real, real_labels)
            
            # Fake images
            noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=DEVICE)
            fake_images = generator(noise)
            output_fake = discriminator(fake_images.detach())
            loss_D_fake = criterion(output_fake, fake_labels)
            
            # Total discriminator loss
            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            optimizer_D.step()
            
            # ============================================
            # Train Generator
            # ============================================
            optimizer_G.zero_grad()
            
            output = discriminator(fake_images)
            loss_G = criterion(output, real_labels)  # Generator wants D to think fake is real
            loss_G.backward()
            optimizer_G.step()
            
            # Save losses
            G_losses.append(loss_G.item())
            D_losses.append(loss_D.item())
            
            # Print progress
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i}/{len(dataloader)}] "
                      f"Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}")
        
        # Generate and save sample images
        if (epoch + 1) % 5 == 0 or epoch == 0:
            with torch.no_grad():
                fake = generator(fixed_noise).detach().cpu()
                img_grid = vutils.make_grid(fake, padding=2, normalize=True, nrow=8)
                plt.figure(figsize=(10, 10))
                plt.imshow(np.transpose(img_grid, (1, 2, 0)))
                plt.axis('off')
                plt.title(f'Epoch {epoch+1}')
                plt.savefig(f'{save_dir}/samples/epoch_{epoch+1:03d}.png')
                plt.close()
    
    # Save final models
    torch.save(generator.state_dict(), f'models/dcgan_generator_{dataset_name}.pth')
    torch.save(discriminator.state_dict(), f'models/dcgan_discriminator_{dataset_name}.pth')
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(G_losses, label='Generator Loss', alpha=0.7)
    plt.plot(D_losses, label='Discriminator Loss', alpha=0.7)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('DCGAN Training Losses')
    plt.savefig(f'{save_dir}/training_losses.png')
    plt.close()
    
    print(f"Training complete! Models saved to models/")
    print(f"Samples saved to {save_dir}/samples/")
    
    return generator, discriminator, G_losses, D_losses


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DCGAN')
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'fashion-mnist'],
                       help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--output-dir', type=str, default='outputs/dcgan',
                       help='Output directory')
    
    args = parser.parse_args()
    
    train_dcgan(args.dataset, args.epochs, args.output_dir)
```

### Step 2.4: Create Slurm Job for DCGAN Training

Create `train_dcgan.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=train_dcgan
#SBATCH --output=logs/dcgan_%j.out
#SBATCH --error=logs/dcgan_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# Create necessary directories
mkdir -p logs outputs/dcgan models

# Load Apptainer if needed
# module load apptainer

# Path to container
CONTAINER=/home1/michael2024/ML_Course/container/pytorch_25.09-py3.sif

# Dataset to use (mnist or fashion-mnist)
DATASET=mnist
EPOCHS=50

echo "Starting DCGAN training on $DATASET"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Run training
apptainer exec --nv $CONTAINER python3 scripts/dcgan.py \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --output-dir outputs/dcgan

echo "Training complete!"
```

### Step 2.5: Submit Training Job

```bash
# Submit the job
sbatch train_dcgan.slurm

# Monitor progress
tail -f logs/dcgan_*.out

# Check job status
squeue -u $USER
```

### Step 2.6: Analyze DCGAN Results

After training completes:

1. **Check generated images:**
   ```bash
   ls outputs/dcgan/samples/
   ```

2. **View training losses:**
   ```bash
   display outputs/dcgan/training_losses.png
   ```

3. **Generate new samples:**
   Create `scripts/generate_dcgan.py`:
   ```python
   import torch
   from dcgan import Generator
   import torchvision.utils as vutils
   import matplotlib.pyplot as plt
   
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
   # Load generator
   generator = Generator(latent_dim=100, channels=1).to(device)
   generator.load_state_dict(torch.load('models/dcgan_generator_mnist.pth'))
   generator.eval()
   
   # Generate samples
   with torch.no_grad():
       noise = torch.randn(64, 100, 1, 1, device=device)
       fake_images = generator(noise).cpu()
   
   # Display
   img_grid = vutils.make_grid(fake_images, padding=2, normalize=True, nrow=8)
   plt.figure(figsize=(10, 10))
   plt.imshow(img_grid.permute(1, 2, 0))
   plt.axis('off')
   plt.savefig('outputs/dcgan/final_samples.png')
   plt.show()
   ```

✅ **Checkpoint:** You should have trained DCGAN model with generated image samples.

---

## Task 3: Diffusion Model Analysis

### Step 3.1: Choose Diffusion Library

**Recommended Options:**

**Option A: Diffusers (Hugging Face) - RECOMMENDED**
- Easy to use
- Pretrained models available
- Good documentation
- Supports Stable Diffusion, DDPM, etc.

**Option B: PyTorch Implementation**
- More control
- Learn internals
- More work required

### Step 3.2: Install Diffusers Library

Create `install_diffusers.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=install_diff
#SBATCH --output=logs/install_diff_%j.out
#SBATCH --error=logs/install_diff_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00

CONTAINER=/home1/michael2024/ML_Course/container/pytorch_25.09-py3.sif

# Install diffusers and dependencies
apptainer exec --nv $CONTAINER pip3 install --user \
    diffusers \
    transformers \
    accelerate \
    safetensors
```

### Step 3.3: Implement Diffusion Model Script

Create `scripts/diffusion_model.py`:

```python
import torch
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_pretrained_diffusion():
    """
    Run a pretrained diffusion model (DDPM on MNIST-like data)
    """
    print("Loading pretrained diffusion model...")
    
    # Load pretrained model (trained on similar data)
    # Note: For MNIST, you may need to train your own or use a similar model
    # Here we use a general DDPM pipeline
    
    try:
        pipeline = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
        pipeline = pipeline.to(DEVICE)
        
        print("Generating samples...")
        
        # Generate images
        images = pipeline(
            batch_size=16,
            num_inference_steps=50,  # Can adjust: more steps = better quality
        ).images
        
        # Save samples
        os.makedirs("outputs/diffusion", exist_ok=True)
        
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        for idx, ax in enumerate(axes.flat):
            ax.imshow(images[idx])
            ax.axis('off')
        plt.tight_layout()
        plt.savefig('outputs/diffusion/pretrained_samples.png')
        print("Samples saved to outputs/diffusion/pretrained_samples.png")
        
        return images
        
    except Exception as e:
        print(f"Error loading pretrained model: {e}")
        print("You may need to train a diffusion model from scratch")
        return None


def train_simple_diffusion(dataset_name='mnist', num_epochs=50):
    """
    Train a simple DDPM diffusion model from scratch
    """
    print(f"Training diffusion model on {dataset_name}")
    print(f"Device: {DEVICE}")
    
    # Create output directory
    os.makedirs("outputs/diffusion", exist_ok=True)
    
    # Load dataset
    if dataset_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.Resize(32),  # Resize to 32x32 for easier processing
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        dataset = datasets.MNIST(root='./data', train=True,
                                download=True, transform=transform)
        channels = 1
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
    
    # Initialize UNet model
    model = UNet2DModel(
        sample_size=32,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 256),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    ).to(DEVICE)
    
    # Initialize scheduler (defines noise schedule)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(dataloader) * num_epochs),
    )
    
    # Training loop
    print("Starting training...")
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (images, _) in enumerate(progress_bar):
            images = images.to(DEVICE)
            
            # Sample random timesteps
            batch_size = images.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (batch_size,), device=DEVICE
            ).long()
            
            # Add noise to images according to timesteps
            noise = torch.randn_like(images)
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
            
            # Predict the noise
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            
            # Compute loss
            loss = F.mse_loss(noise_pred, noise)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            epoch_loss += loss.item()
            losses.append(loss.item())
            
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Generate samples every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                # Start from pure noise
                sample = torch.randn(16, 1, 32, 32).to(DEVICE)
                
                # Iteratively denoise
                for t in noise_scheduler.timesteps:
                    with torch.no_grad():
                        residual = model(sample, t).sample
                    sample = noise_scheduler.step(residual, t, sample).prev_sample
                
                # Save samples
                sample = (sample + 1) / 2  # Denormalize
                sample = sample.cpu().numpy()
                
                fig, axes = plt.subplots(4, 4, figsize=(10, 10))
                for idx, ax in enumerate(axes.flat):
                    ax.imshow(sample[idx, 0], cmap='gray')
                    ax.axis('off')
                plt.suptitle(f'Epoch {epoch+1}')
                plt.tight_layout()
                plt.savefig(f'outputs/diffusion/samples_epoch_{epoch+1:03d}.png')
                plt.close()
    
    # Save model
    torch.save(model.state_dict(), 'models/diffusion_model_mnist.pth')
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses, alpha=0.7)
    plt.xlabel('Iterations')
    plt.ylabel('MSE Loss')
    plt.title('Diffusion Model Training Loss')
    plt.savefig('outputs/diffusion/training_loss.png')
    plt.close()
    
    print("Training complete!")
    return model, losses


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Diffusion Model')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'pretrained'],
                       help='Run pretrained model or train from scratch')
    parser.add_argument('--dataset', type=str, default='mnist',
                       help='Dataset to use for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    
    args = parser.parse_args()
    
    if args.mode == 'pretrained':
        run_pretrained_diffusion()
    else:
        train_simple_diffusion(args.dataset, args.epochs)
```

### Step 3.4: Create Slurm Job for Diffusion Model

Create `train_diffusion.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=train_diff
#SBATCH --output=logs/diffusion_%j.out
#SBATCH --error=logs/diffusion_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00

mkdir -p logs outputs/diffusion models

CONTAINER=/home1/michael2024/ML_Course/container/pytorch_25.09-py3.sif

echo "Starting diffusion model training"
echo "Job ID: $SLURM_JOB_ID"

# Train from scratch
apptainer exec --nv $CONTAINER python3 scripts/diffusion_model.py \
    --mode train \
    --dataset mnist \
    --epochs 50

echo "Training complete!"
```

### Step 3.5: Run Diffusion Model

```bash
# Submit job
sbatch train_diffusion.slurm

# Monitor
tail -f logs/diffusion_*.out
```

✅ **Checkpoint:** You should have diffusion model results with generated samples.

---

## Task 4: Comparison and Evaluation

### Step 4.1: Create Comparison Script

Create `scripts/compare_models.py`:

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import time
import os

def calculate_metrics(real_images, fake_images):
    """
    Calculate FID and IS scores
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # FID Score
    fid = FrechetInceptionDistance(feature=2048).to(device)
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)
    fid_score = fid.compute()
    
    # Inception Score
    inception = InceptionScore().to(device)
    inception.update(fake_images)
    is_score = inception.compute()
    
    return fid_score.item(), is_score[0].item()


def compare_generation_time(generator_dcgan, diffusion_model, num_samples=100):
    """
    Compare generation time for both models
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # DCGAN generation time
    start_time = time.time()
    with torch.no_grad():
        noise = torch.randn(num_samples, 100, 1, 1, device=device)
        _ = generator_dcgan(noise)
    dcgan_time = time.time() - start_time
    
    # Diffusion generation time (simplified)
    # Note: Actual implementation depends on your diffusion setup
    start_time = time.time()
    # ... diffusion generation ...
    diffusion_time = time.time() - start_time
    
    return dcgan_time, diffusion_time


def create_comparison_report():
    """
    Generate comprehensive comparison report
    """
    report = {
        'DCGAN': {
            'Training Time': 'measured_value',
            'Generation Time (per 100 samples)': 'measured_value',
            'Image Quality (qualitative)': 'description',
            'FID Score': 'measured_value',
            'Inception Score': 'measured_value',
            'Mode Collapse': 'observed/not observed',
            'Training Stability': 'stable/unstable',
        },
        'Diffusion Model': {
            'Training Time': 'measured_value',
            'Generation Time (per 100 samples)': 'measured_value',
            'Image Quality (qualitative)': 'description',
            'FID Score': 'measured_value',
            'Inception Score': 'measured_value',
            'Diversity': 'high/medium/low',
            'Training Stability': 'stable/unstable',
        }
    }
    
    return report


def visualize_comparison():
    """
    Create side-by-side comparison visualizations
    """
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    
    # Load DCGAN samples
    # ... load images ...
    
    # Load Diffusion samples
    # ... load images ...
    
    # Display
    for i in range(8):
        axes[0, i].imshow(dcgan_samples[i])
        axes[0, i].axis('off')
        axes[0, i].set_title('DCGAN' if i == 0 else '')
        
        axes[1, i].imshow(diffusion_samples[i])
        axes[1, i].axis('off')
        axes[1, i].set_title('Diffusion' if i == 0 else '')
    
    plt.tight_layout()
    plt.savefig('outputs/comparison/side_by_side.png')
    plt.close()


if __name__ == "__main__":
    print("Running model comparison...")
    
    # Create comparison directory
    os.makedirs('outputs/comparison', exist_ok=True)
    
    # Run comparisons
    # ... your comparison code ...
    
    print("Comparison complete!")
```

### Step 4.2: Metrics to Compare

**Quantitative Metrics:**
1. **FID Score** (Fréchet Inception Distance)
   - Lower is better
   - Measures quality and diversity
   
2. **Inception Score (IS)**
   - Higher is better
   - Measures quality and diversity

3. **Generation Time**
   - Time to generate N samples

4. **Training Time**
   - Total time to train

5. **GPU Memory Usage**
   - Peak memory during training

**Qualitative Analysis:**
1. **Image Quality**
   - Sharpness
   - Detail
   - Realism

2. **Diversity**
   - Variety in generated samples
   - Mode collapse detection

3. **Training Stability**
   - Loss curves
   - Convergence behavior

### Step 4.3: Create Comparison Table

Document your findings in a table:

| Metric | DCGAN | Diffusion | Winner |
|--------|-------|-----------|--------|
| Training Time | X hours | Y hours | ? |
| Generation Time (100 samples) | X seconds | Y seconds | ? |
| FID Score | X | Y | Lower is better |
| Inception Score | X | Y | Higher is better |
| Image Quality | Good/Fair/Poor | Good/Fair/Poor | Subjective |
| Diversity | High/Med/Low | High/Med/Low | ? |
| Training Stability | Stable/Unstable | Stable/Unstable | ? |
| GPU Memory | X GB | Y GB | Lower is better |

✅ **Checkpoint:** You should have quantitative and qualitative comparisons.

---

## Task 5: Documentation and Report

### Step 5.1: Report Structure

Create `REPORT.md` with the following sections:

```markdown
# AS4: Image Generation Models Report

**Student Name:** [Your Name]
**Date:** [Submission Date]

## 1. Executive Summary
- Brief overview of what you implemented
- Key findings from comparison
- Main conclusions

## 2. Environment Setup

### 2.1 Container Configuration
- Container image used
- Version information
- Installation steps

### 2.2 Cluster Configuration
- GPU type and specifications
- Slurm configuration
- Resource allocation

### 2.3 Software Dependencies
- PyTorch version
- Additional libraries
- Python version

## 3. DCGAN Implementation

### 3.1 Architecture Design
- Generator architecture diagram/description
- Discriminator architecture diagram/description
- Design choices and rationale

### 3.2 Training Configuration
- Hyperparameters used
- Loss functions
- Optimization strategy

### 3.3 Training Process
- Training time
- Convergence behavior
- Challenges encountered

### 3.4 Results
- Generated samples (include images)
- Training loss curves
- Qualitative analysis

### 3.5 Observations
- Mode collapse (if any)
- Training stability
- Quality of generated images

## 4. Diffusion Model

### 4.1 Model Selection
- Which diffusion model used
- Architecture details
- Pretrained vs. trained from scratch

### 4.2 Implementation Details
- Forward diffusion process
- Reverse diffusion process
- UNet architecture (if applicable)

### 4.3 Training/Inference
- Configuration
- Time requirements
- Challenges

### 4.4 Results
- Generated samples (include images)
- Quality analysis

## 5. Comparative Analysis

### 5.1 Quantitative Comparison
- FID scores
- Inception scores
- Generation times
- Training times
- Resource usage

### 5.2 Qualitative Comparison
- Image quality comparison
- Diversity analysis
- Visual examples

### 5.3 Trade-offs
- Advantages of each approach
- Disadvantages of each approach
- Use case recommendations

## 6. Discussion

### 6.1 Key Findings
- What worked well
- What didn't work
- Unexpected results

### 6.2 Challenges and Solutions
- Technical difficulties
- How you overcame them

### 6.3 Lessons Learned
- Understanding of convolution/transpose convolution
- GAN training dynamics
- Diffusion process understanding

## 7. Conclusions
- Summary of results
- Recommendations
- Future work

## 8. References
- Papers cited
- Libraries used
- Resources consulted

## Appendix A: Code Structure
- File organization
- How to run experiments
- Reproducibility instructions

## Appendix B: Sample Outputs
- Additional generated images
- Training curves
- Comparison visualizations
```

### Step 5.2: Update README.md

Your `README.md` should include:

```markdown
# AS4: Image Generation Models

## Project Structure

```
as4/
├── data/                  # Datasets (auto-downloaded)
├── models/                # Saved model checkpoints
│   ├── dcgan_generator_mnist.pth
│   ├── dcgan_discriminator_mnist.pth
│   └── diffusion_model_mnist.pth
├── outputs/               # Generated images and results
│   ├── dcgan/
│   │   ├── samples/
│   │   └── training_losses.png
│   ├── diffusion/
│   │   ├── samples/
│   │   └── training_loss.png
│   └── comparison/
├── logs/                  # Slurm job logs
├── scripts/               # Python scripts
│   ├── dcgan.py
│   ├── diffusion_model.py
│   ├── generate_dcgan.py
│   └── compare_models.py
├── test_gpu.slurm         # GPU test job
├── train_dcgan.slurm      # DCGAN training job
├── train_diffusion.slurm  # Diffusion training job
├── README.md              # This file
└── REPORT.md              # Final report
```

## Container Information

- **Base Image:** nvcr.io/nvidia/pytorch:25.09-py3
- **PyTorch Version:** 2.5
- **CUDA Version:** 12.6
- **Python Version:** 3.10

## How to Run

### 1. Pull Container
```bash
cd /home1/michael2024/ML_Course/container
apptainer pull docker://nvcr.io/nvidia/pytorch:25.09-py3
```

### 2. Test GPU Access
```bash
sbatch test_gpu.slurm
```

### 3. Train DCGAN
```bash
sbatch train_dcgan.slurm
```

### 4. Train Diffusion Model
```bash
sbatch train_diffusion.slurm
```

### 5. Generate Comparison
```bash
python3 scripts/compare_models.py
```

## Results

[Include summary of your results here]

## Dependencies

Core libraries (included in container):
- PyTorch 2.5
- torchvision
- NumPy
- Matplotlib

Additional libraries (install if needed):
- diffusers
- transformers
- torchmetrics
```

### Step 5.3: Code Documentation

Ensure all Python scripts have:
- Docstrings for functions
- Comments explaining complex sections
- Type hints where appropriate
- Clear variable names

✅ **Checkpoint:** Complete documentation ready for submission.

---

## Submission Checklist

Before submitting, verify you have:

### Code and Scripts
- [ ] All Python scripts in `scripts/` directory
- [ ] All Slurm job files (`.slurm`)
- [ ] Container pull/setup instructions
- [ ] Requirements file (if additional packages needed)

### Models and Results
- [ ] Trained DCGAN model checkpoints
- [ ] Trained/run diffusion model checkpoints
- [ ] Generated sample images from both models
- [ ] Training loss curves
- [ ] Comparison visualizations

### Documentation
- [ ] Comprehensive `README.md`
- [ ] Detailed `REPORT.md` with all sections
- [ ] Container information documented
- [ ] Instructions for reproducing results
- [ ] Code comments and docstrings

### Analysis
- [ ] Quantitative comparison (FID, IS, times)
- [ ] Qualitative comparison (visual quality, diversity)
- [ ] Discussion of trade-offs
- [ ] Observations on training dynamics

### Reproducibility
- [ ] Clear steps to reproduce
- [ ] All file paths are correct
- [ ] Slurm scripts are cluster-agnostic (or documented)
- [ ] Container version locked

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: Container won't pull
```bash
# Check Apptainer version
apptainer --version

# Try alternative method
singularity pull docker://nvcr.io/nvidia/pytorch:25.09-py3

# Check disk space
df -h ~
```

#### Issue: GPU not detected
```bash
# Verify NVIDIA drivers
nvidia-smi

# Check Slurm GPU allocation
squeue -u $USER -o "%.18i %.9P %.50j %.8u %.2t %.10M %.6D %R %b"

# Ensure --nv flag is used
apptainer exec --nv $CONTAINER python3 -c "import torch; print(torch.cuda.is_available())"
```

#### Issue: Out of memory during training
- Reduce batch size
- Use gradient accumulation
- Reduce model size
- Request more GPU memory in Slurm

```python
# In your script, reduce batch size
BATCH_SIZE = 64  # Instead of 128
```

#### Issue: DCGAN not converging
- Adjust learning rates
- Try different beta values for Adam
- Add label smoothing
- Adjust discriminator/generator update ratio

```python
# Label smoothing
real_labels = torch.ones(batch_size, 1, device=DEVICE) * 0.9  # Instead of 1.0
fake_labels = torch.zeros(batch_size, 1, device=DEVICE) + 0.1  # Instead of 0.0
```

#### Issue: Mode collapse in GAN
- Use different loss functions (Wasserstein loss)
- Add noise to discriminator inputs
- Use batch normalization
- Adjust architecture

#### Issue: Diffusion model too slow
- Reduce number of diffusion steps
- Use DDIM scheduler instead of DDPM
- Use latent diffusion instead of pixel-space
- Generate fewer samples for comparison

```python
# Use fewer steps
num_inference_steps = 25  # Instead of 50
```

#### Issue: Jobs stuck in queue
```bash
# Check queue status
squeue

# Check your job specifically
squeue -u $USER

# See why job is pending
squeue -u $USER --start

# Check partition availability
sinfo
```

---

## Tips for Success

1. **Start Early**: Don't wait until the last few days

2. **Test Incrementally**: Test each component before moving to the next

3. **Monitor Resources**: Keep an eye on GPU memory and training time

4. **Save Checkpoints**: Save model checkpoints regularly

5. **Version Control**: Use git to track your changes

6. **Document as You Go**: Don't leave documentation for the end

7. **Ask for Help**: Use office hours if stuck

8. **Compare Results**: Regularly check generated samples

9. **Time Management**: Allocate time for each task based on the 10-day plan

10. **Backup**: Keep backups of trained models and results

---

## Additional Resources

### Papers to Read
- **DCGAN Paper**: "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" (Radford et al., 2016)
- **Diffusion Models**: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
- **Stable Diffusion**: "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., 2022)

### Useful Links
- [PyTorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/index)
- [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/)
- [Apptainer Documentation](https://apptainer.org/docs/)

### Metrics
- [FID Score Explanation](https://wandb.ai/ayush-thakur/gan-evaluation/reports/How-to-Evaluate-GANs-using-Frechet-Inception-Distance-FID---Vmlldzo0MTAxOTI)
- [Inception Score](https://arxiv.org/abs/1606.03498)

---

## Contact and Support

If you encounter issues:
1. Check this walkthrough thoroughly
2. Review lecture slides
3. Search for error messages online
4. Ask on course forum
5. Attend office hours
6. Email instructor as last resort

---

**Good luck with your assignment! 🚀**

Remember: The goal is to learn about image generation models, not just to get them working. Make sure you understand the "why" behind each step.
