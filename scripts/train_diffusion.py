import torch
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
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
        # Load pre-trained DDPM model (trained on RGB CIFAR-10)
        model = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to(device)
        scheduler = model.scheduler
        unet = model.unet
        
        # Convert grayscale MNIST to 3 channels to match CIFAR-10 model
        print("Converting grayscale MNIST to RGB format for pretrained model...")
        real_images = real_images.repeat(1, 3, 1, 1)  # Repeat channels: [B, 1, H, W] -> [B, 3, H, W]
        
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
            # Convert back to grayscale for visualization (take first channel)
            orig_img = (real_images[i, 0].cpu() + 1) / 2
            noisy_img = (noisy_images[i, 0].cpu() + 1) / 2
            denoise_img = (denoised_images[i, 0].cpu() + 1) / 2
            
            # Original
            axes[0, i].imshow(orig_img, cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_ylabel('Original', rotation=0, labelpad=40, fontsize=12)
            
            # Noisy
            axes[1, i].imshow(noisy_img, cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_ylabel(f'Noisy\n(σ={noise_std})', rotation=0, labelpad=40, fontsize=12)
            
            # Denoised
            axes[2, i].imshow(denoise_img, cmap='gray')
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Diffusion Model Denoising Experiments')
    parser.add_argument('--mode', type=str, default='denoise',
                       choices=['denoise', 'compare'],
                       help='Experiment mode: denoise or compare')
    parser.add_argument('--num-samples', type=int, default=8,
                       help='Number of samples for denoising experiment')
    parser.add_argument('--inference-steps', type=int, default=50,
                       help='Number of inference steps for denoising')
    
    args = parser.parse_args()
    
    if args.mode == 'denoise':
        print("\n" + "="*60)
        print("EXPERIMENT: Denoising Diffusion Process")
        print("="*60)
        denoise_images_experiment(
            output_dir='output/diffusion_denoise',
            num_samples=args.num_samples,
            noise_levels=[0.3, 0.5, 0.7]
        )
    
    elif args.mode == 'compare':
        print("\n" + "="*60)
        print("COMPARISON: Diffusion vs GAN")
        print("="*60)
        compare_with_gan()


