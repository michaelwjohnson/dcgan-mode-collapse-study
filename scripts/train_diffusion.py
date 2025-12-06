import torch
import torch.nn.functional as F
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import argparse
import random
import time
import json

# Fixed seed for reproducibility (matching DCGAN setup)
SEED = 77
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def denoise_images_experiment(output_dir='output/diffusion_denoise', num_samples=8, noise_levels=[0.3, 0.5, 0.7]):
    """
    Experiment: Add Gaussian noise to MNIST images and denoise them
    This demonstrates the forward and reverse diffusion process
    
    Args:
        output_dir: Directory to save outputs
        num_samples: Number of images to process
        noise_levels: List of noise standard deviations to test
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Denoising Experiment on {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load MNIST test images
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    
    # Get a batch of real images
    real_images, labels = next(iter(dataloader))
    real_images = real_images.to(device)
    
    # Load or create a simple diffusion model
    print("Loading diffusion model...")
    try:
        # Try to load pre-trained DDPM model (trained on similar sized images)
        model = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to(device)
        scheduler = model.scheduler
        unet = model.unet
    except Exception as e:
        print(f"Could not load pretrained model: {e}")
        print("Using custom UNet for denoising...")
        unet = UNet2DModel(
            sample_size=32,
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(64, 128, 256),
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        ).to(device)
        scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    results = []
    
    for noise_std in noise_levels:
        print(f"\nProcessing noise level: {noise_std}")
        
        # Forward process: Add Gaussian noise (simulating forward diffusion)
        noise = torch.randn_like(real_images) * noise_std
        noisy_images = real_images + noise
        noisy_images = torch.clamp(noisy_images, -1, 1)
        
        # Reverse process: Denoise using the model
        denoised_images = noisy_images.clone()
        
        # Run denoising steps
        num_inference_steps = 50
        scheduler.set_timesteps(num_inference_steps)
        
        start_time = time.time()
        
        with torch.no_grad():
            for t in tqdm(scheduler.timesteps, desc=f"Denoising (noise={noise_std})"):
                # Predict noise residual
                model_output = unet(denoised_images, t).sample
                # Compute previous sample
                denoised_images = scheduler.step(model_output, t, denoised_images).prev_sample
        
        denoise_time = time.time() - start_time
        
        results.append({
            'noise_level': noise_std,
            'time_seconds': denoise_time,
            'time_per_sample_ms': (denoise_time / num_samples) * 1000,
            'num_steps': num_inference_steps
        })
        
        # Visualize: Original, Noisy, Denoised
        fig, axes = plt.subplots(3, num_samples, figsize=(16, 6))
        
        for i in range(num_samples):
            # Original
            axes[0, i].imshow((real_images[i, 0].cpu() + 1) / 2, cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_ylabel('Original', rotation=0, labelpad=40, fontsize=12)
            
            # Noisy
            axes[1, i].imshow((noisy_images[i, 0].cpu() + 1) / 2, cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_ylabel(f'Noisy\n(σ={noise_std})', rotation=0, labelpad=40, fontsize=12)
            
            # Denoised
            axes[2, i].imshow((denoised_images[i, 0].cpu() + 1) / 2, cmap='gray')
            axes[2, i].axis('off')
            if i == 0:
                axes[2, i].set_ylabel('Denoised', rotation=0, labelpad=40, fontsize=12)
        
        plt.suptitle(f'Denoising Diffusion Process (noise std = {noise_std})', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/denoise_noise_{noise_std}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Time: {denoise_time:.2f}s ({denoise_time/num_samples*1000:.2f}ms per sample)")
    
    # Save performance metrics
    metrics = {
        "experiment": "denoising_diffusion",
        "num_samples": num_samples,
        "num_inference_steps": num_inference_steps,
        "results": results
    }
    
    with open(f'{output_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\nDenoising experiment complete! Results saved to {output_dir}/")
    return results


def unconditional_generation_experiment(output_dir='output/diffusion_generation', 
                                       num_samples=64, num_inference_steps=50):
    """
    Experiment: Generate images from pure noise using pretrained diffusion model
    Demonstrates unconditional generation and sampling process
    
    Args:
        output_dir: Directory to save outputs
        num_samples: Number of images to generate
        num_inference_steps: Number of denoising steps
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning Unconditional Generation Experiment on {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load pretrained DDPM model
        print("Loading pretrained DDPM model...")
        pipeline = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
        pipeline = pipeline.to(device)
        
        print(f"Generating {num_samples} samples with {num_inference_steps} steps...")
        
        # Test different numbers of inference steps
        step_configs = [10, 25, 50, 100]
        results = []
        
        for steps in step_configs:
            print(f"\n  Testing with {steps} inference steps...")
            start_time = time.time()
            
            # Generate images
            output = pipeline(
                batch_size=min(16, num_samples),  # Generate in batches
                num_inference_steps=steps,
            )
            images = output.images
            
            gen_time = time.time() - start_time
            
            results.append({
                'num_steps': steps,
                'time_seconds': gen_time,
                'time_per_sample_ms': (gen_time / len(images)) * 1000
            })
            
            # Save samples
            fig, axes = plt.subplots(4, 4, figsize=(10, 10))
            for idx, ax in enumerate(axes.flat):
                ax.imshow(images[idx])
                ax.axis('off')
            plt.suptitle(f'Generated Samples ({steps} inference steps)', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/generated_steps_{steps}.png', dpi=150)
            plt.close()
            
            print(f"    Time: {gen_time:.2f}s ({gen_time/len(images)*1000:.2f}ms per sample)")
        
        # Save metrics
        metrics = {
            "experiment": "unconditional_generation",
            "model": "google/ddpm-cifar10-32",
            "results": results
        }
        
        with open(f'{output_dir}/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"\nGeneration experiment complete! Results saved to {output_dir}/")
        return results
        
    except Exception as e:
        print(f"Error: {e}")
        print("Pretrained model not available. Skipping unconditional generation.")
        return None


def compare_with_gan(diffusion_dir='output/diffusion_denoise', 
                    gan_baseline_dir='output/dcgan_baseline',
                    gan_modified_dir='output/dcgan_modified',
                    output_dir='output/gan_diff_comparison'):
    """
    Compare diffusion model results with GAN results
    
    Args:
        diffusion_dir: Directory with diffusion results
        gan_baseline_dir: Directory with GAN baseline results
        gan_modified_dir: Directory with GAN modified results
        output_dir: Directory to save comparison
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n=== Comparison: Diffusion vs GAN ===")
    
    # Load metrics
    comparison = {
        "diffusion": {},
        "gan_baseline": {},
        "gan_modified": {}
    }
    
    # Load diffusion metrics
    if os.path.exists(f'{diffusion_dir}/metrics.json'):
        with open(f'{diffusion_dir}/metrics.json', 'r') as f:
            comparison["diffusion"] = json.load(f)
    
    # Load GAN metrics
    if os.path.exists(f'{gan_baseline_dir}/metrics.json'):
        with open(f'{gan_baseline_dir}/metrics.json', 'r') as f:
            comparison["gan_baseline"] = json.load(f)
    
    if os.path.exists(f'{gan_modified_dir}/metrics.json'):
        with open(f'{gan_modified_dir}/metrics.json', 'r') as f:
            comparison["gan_modified"] = json.load(f)
    
    # Save comparison
    with open(f'{output_dir}/diffusion_vs_gan_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=4)
    
    # Print summary
    print("\nDiffusion Model Characteristics:")
    print("  - Training Stability: High (no adversarial dynamics)")
    print("  - Mode Collapse: Not susceptible (likelihood-based)")
    print("  - Inference Speed: Slower (iterative denoising)")
    print("  - Sample Quality: Generally high")
    
    print("\nGAN Characteristics:")
    print("  - Training Stability: Lower (adversarial min-max game)")
    print("  - Mode Collapse: Susceptible (as seen in modified config)")
    print("  - Inference Speed: Fast (single forward pass)")
    print("  - Sample Quality: Variable (depends on training stability)")
    
    print(f"\nComparison saved to {output_dir}/diffusion_vs_gan_comparison.json")
    
    return comparison


def train_diffusion(dataset_name='mnist', num_epochs=50, output_dir='output/diffusion',
                    batch_size=128, lr=1e-4, num_train_timesteps=1000):
    """
    Train a DDPM diffusion model from scratch
    
    Args:
        dataset_name: Name of dataset ('mnist' or 'fashion-mnist')
        num_epochs: Number of training epochs
        output_dir: Directory to save outputs
        batch_size: Batch size for training
        lr: Learning rate
        num_train_timesteps: Number of diffusion timesteps
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Training Diffusion Model on {dataset_name}")
    print(f"Device: {device}")
    print(f"Configuration: timesteps={num_train_timesteps}, lr={lr}, batch_size={batch_size}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/samples", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Load dataset
    if dataset_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.Resize(32),  # Resize to 32x32 for UNet
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        dataset = datasets.MNIST(root='./data', train=True,
                                download=True, transform=transform)
        channels = 1
    elif dataset_name.lower() == 'fashion-mnist':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        dataset = datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform=transform)
        channels = 1
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    dataloader = DataLoader(dataset, batch_size=batch_size,
                           shuffle=True, num_workers=4)
    
    # Initialize UNet model
    model = UNet2DModel(
        sample_size=32,
        in_channels=channels,
        out_channels=channels,
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
    ).to(device)
    
    # Initialize scheduler (defines noise schedule)
    noise_scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
    
    # Optimizer with cosine learning rate schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(dataloader) * num_epochs),
    )
    
    # Training loop
    print("Starting Training...")
    losses = []
    start_time = time.time()
    
    pbar = tqdm(total=num_epochs * len(dataloader), desc="Training Diffusion", unit="batch")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            
            # Sample random timesteps for each image
            batch_size_actual = images.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (batch_size_actual,), device=device
            ).long()
            
            # Add noise to images according to timesteps
            noise = torch.randn_like(images)
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
            
            # Predict the noise
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            
            # Compute MSE loss
            loss = F.mse_loss(noise_pred, noise)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            epoch_loss += loss.item()
            losses.append(loss.item())
            
            # Update progress bar
            pbar.set_postfix({
                "Epoch": f"{epoch+1}/{num_epochs}",
                "Loss": f"{loss.item():.4f}"
            })
            pbar.update(1)
        
        avg_loss = epoch_loss / len(dataloader)
        
        # Save checkpoints every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_dir = f'{output_dir}/checkpoints'
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'losses': losses,
            }, f'{checkpoint_dir}/checkpoint_epoch_{epoch+1:03d}.pth')
            print(f"\nCheckpoint saved at epoch {epoch+1}")
        
        # Generate samples every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                # Start from pure noise
                sample = torch.randn(64, channels, 32, 32).to(device)
                
                # Iteratively denoise
                for t in tqdm(noise_scheduler.timesteps, desc=f"Generating samples (Epoch {epoch+1})", leave=False):
                    with torch.no_grad():
                        residual = model(sample, t).sample
                    sample = noise_scheduler.step(residual, t, sample).prev_sample
                
                # Denormalize and save
                sample = (sample + 1) / 2
                sample = sample.cpu()
                
                # Create grid
                fig, axes = plt.subplots(8, 8, figsize=(10, 10))
                for idx, ax in enumerate(axes.flat):
                    ax.imshow(sample[idx, 0], cmap='gray')
                    ax.axis('off')
                plt.suptitle(f'Epoch {epoch+1}')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/samples/epoch_{epoch+1:03d}.png')
                plt.close()
    
    pbar.close()
    
    training_time = time.time() - start_time
    
    # Save final model
    torch.save(model.state_dict(),
               f'models/diffusion_model_{dataset_name}.pth')
    
    # Save losses to CSV
    with open(f'{output_dir}/losses.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Iteration', 'Loss'])
        for i, loss in enumerate(losses):
            writer.writerow([i, loss])
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses, alpha=0.7, label='Diffusion Loss')
    plt.xlabel('Iterations')
    plt.ylabel('MSE Loss')
    plt.title('Diffusion Model Training Loss')
    plt.legend()
    plt.savefig(f'{output_dir}/training_losses.png')
    plt.close()
    
    # Save metrics
    metrics = {
        "configuration": {
            "dataset": dataset_name,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "num_train_timesteps": num_train_timesteps,
        },
        "training_performance": {
            "total_training_time_seconds": round(training_time, 2),
            "total_training_time_minutes": round(training_time / 60, 2),
            "time_per_epoch_seconds": round(training_time / num_epochs, 2),
            "total_iterations": len(losses)
        },
        "loss_metrics": {
            "final_loss": round(losses[-1], 4),
            "min_loss": round(min(losses), 4),
            "avg_loss": round(sum(losses) / len(losses), 4)
        }
    }
    
    with open(f'{output_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\nTraining complete! Model saved to models/")
    print(f"Samples saved to {output_dir}/samples/")
    print(f"Training time: {training_time/60:.2f} minutes")
    
    return model, losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Diffusion Model Experiments')
    parser.add_argument('--mode', type=str, default='denoise',
                       choices=['denoise', 'generate', 'train', 'compare', 'all'],
                       help='Experiment mode')
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'fashion-mnist'],
                       help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs (for training mode)')
    parser.add_argument('--output-dir', type=str, default='output/diffusion',
                       help='Output directory')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--timesteps', type=int, default=1000,
                       help='Number of diffusion timesteps')
    parser.add_argument('--num-samples', type=int, default=8,
                       help='Number of samples for experiments')
    parser.add_argument('--inference-steps', type=int, default=50,
                       help='Number of inference steps for generation')
    
    args = parser.parse_args()
    
    if args.mode == 'denoise':
        print("\n" + "="*60)
        print("EXPERIMENT 1: Denoising Diffusion Process")
        print("="*60)
        denoise_images_experiment(
            output_dir='output/diffusion_denoise',
            num_samples=args.num_samples,
            noise_levels=[0.3, 0.5, 0.7]
        )
    
    elif args.mode == 'generate':
        print("\n" + "="*60)
        print("EXPERIMENT 2: Unconditional Generation")
        print("="*60)
        unconditional_generation_experiment(
            output_dir='output/diffusion_generation',
            num_samples=64,
            num_inference_steps=args.inference_steps
        )
    
    elif args.mode == 'train':
        print("\n" + "="*60)
        print("EXPERIMENT 3: Train Diffusion Model from Scratch")
        print("="*60)
        train_diffusion(
            dataset_name=args.dataset,
            num_epochs=args.epochs,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            lr=args.lr,
            num_train_timesteps=args.timesteps
        )
    
    elif args.mode == 'compare':
        print("\n" + "="*60)
        print("COMPARISON: Diffusion vs GAN")
        print("="*60)
        compare_with_gan()
    
    elif args.mode == 'all':
        print("\n" + "="*60)
        print("RUNNING ALL EXPERIMENTS")
        print("="*60)
        
        # Experiment 1: Denoising
        denoise_images_experiment(
            output_dir='output/diffusion_denoise',
            num_samples=args.num_samples,
            noise_levels=[0.3, 0.5, 0.7]
        )
        
        # Experiment 2: Generation
        unconditional_generation_experiment(
            output_dir='output/diffusion_generation',
            num_samples=64,
            num_inference_steps=args.inference_steps
        )
        
        # Experiment 3: Comparison
        compare_with_gan()
        
        print("\n" + "="*60)
        print("ALL EXPERIMENTS COMPLETE")
        print("="*60)

