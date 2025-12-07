"""
Compute quantitative metrics for GAN and Diffusion models
Uses torchmetrics for FID and Inception Score
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import numpy as np
import json
import os
from PIL import Image
import argparse
import time

# Fixed seed
SEED = 77
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_generated_samples(image_path, num_samples=64):
    """Load generated samples from grid image"""
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    
    # Assuming 8x8 grid
    grid_size = 8
    h, w = img_array.shape
    cell_h, cell_w = h // grid_size, w // grid_size
    
    samples = []
    for i in range(grid_size):
        for j in range(grid_size):
            cell = img_array[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            # Resize to 28x28
            cell_img = Image.fromarray(cell).resize((28, 28), Image.BILINEAR)
            # Convert to 3-channel for Inception
            cell_rgb = np.stack([cell_img]*3, axis=0)  # [3, 28, 28]
            samples.append(cell_rgb)
    
    samples = np.array(samples[:num_samples])  # [N, 3, 28, 28]
    samples = torch.from_numpy(samples).float()
    # Normalize to [0, 255] uint8 for torchmetrics
    samples = ((samples / 255.0) * 255).byte()
    
    return samples


def compute_fid_score(real_images, fake_images, device='cuda'):
    """Compute FID score between real and fake images"""
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    
    # Update with real images
    fid.update(real_images.to(device), real=True)
    
    # Update with fake images
    fid.update(fake_images.to(device), real=False)
    
    # Compute FID
    fid_score = fid.compute()
    
    return fid_score.item()


def compute_inception_score(fake_images, device='cuda'):
    """Compute Inception Score for fake images"""
    inception = InceptionScore(normalize=True).to(device)
    
    # Update with fake images
    inception.update(fake_images.to(device))
    
    # Compute IS (returns mean and std)
    is_mean, is_std = inception.compute()
    
    return is_mean.item(), is_std.item()


def compute_diversity_metrics(samples):
    """Compute diversity metrics from samples"""
    # Convert to float for calculations
    samples_float = samples.float() / 255.0
    
    # Pixel-level variance
    per_sample_variance = samples_float.reshape(samples.size(0), -1).var(dim=1).mean().item()
    inter_sample_variance = samples_float.reshape(samples.size(0), -1).var(dim=0).mean().item()
    
    # Pairwise distance (subsample for efficiency)
    n_pairs = min(100, samples.size(0) * (samples.size(0) - 1) // 2)
    distances = []
    indices = torch.randperm(samples.size(0))[:20]  # Use 20 samples
    for i in range(len(indices)):
        for j in range(i+1, len(indices)):
            dist = F.mse_loss(samples_float[indices[i]], samples_float[indices[j]])
            distances.append(dist.item())
    
    mean_pairwise_dist = np.mean(distances) if distances else 0
    
    return {
        'intra_sample_variance': per_sample_variance,
        'inter_sample_variance': inter_sample_variance,
        'mean_pairwise_distance': mean_pairwise_dist
    }


def compute_denoising_metrics(denoised_path):
    """Compute reconstruction error for diffusion denoising"""
    img = Image.open(denoised_path).convert('L')
    img_array = np.array(img)
    
    # Split into 3 rows: original, noisy, denoised
    h, w = img_array.shape
    row_h = h // 3
    
    original_row = img_array[:row_h, :]
    denoised_row = img_array[2*row_h:, :]
    
    # Split each row into individual samples
    num_samples = 8
    cell_w = w // num_samples
    
    mse_values = []
    psnr_values = []
    ssim_values = []
    
    for i in range(num_samples):
        orig = original_row[:, i*cell_w:(i+1)*cell_w].astype(float)
        den = denoised_row[:, i*cell_w:(i+1)*cell_w].astype(float)
        
        # MSE
        mse = np.mean((orig - den) ** 2)
        mse_values.append(mse)
        
        # PSNR
        if mse > 0:
            psnr = 10 * np.log10(255**2 / mse)
            psnr_values.append(psnr)
        
        # Simple SSIM approximation (correlation)
        orig_flat = orig.flatten()
        den_flat = den.flatten()
        if orig_flat.std() > 0 and den_flat.std() > 0:
            corr = np.corrcoef(orig_flat, den_flat)[0, 1]
            ssim_values.append(corr)
    
    return {
        'mean_mse': np.mean(mse_values),
        'mean_psnr': np.mean(psnr_values) if psnr_values else 0,
        'mean_correlation': np.mean(ssim_values) if ssim_values else 0,
        'std_mse': np.std(mse_values),
        'std_psnr': np.std(psnr_values) if psnr_values else 0
    }


def load_real_mnist_samples(num_samples=1000, device='cuda'):
    """Load real MNIST samples for FID computation"""
    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    
    images, _ = next(iter(loader))
    
    # Convert to 3-channel RGB and uint8 [0, 255]
    images_rgb = images.repeat(1, 3, 1, 1)  # [N, 3, 28, 28]
    images_rgb = (images_rgb * 255).byte()
    
    return images_rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output-dir', type=str, default='output/metrics')
    parser.add_argument('--num-real-samples', type=int, default=1000, help='Number of real MNIST samples for FID')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("COMPUTING QUANTITATIVE METRICS")
    print("="*60)
    
    results = {}
    
    # Load real MNIST samples once for FID
    print("\nLoading real MNIST samples...")
    real_samples = load_real_mnist_samples(num_samples=args.num_real_samples, device=device)
    print(f"  Loaded {real_samples.size(0)} real samples")
    
    # ===== Evaluate DCGAN Baseline =====
    print("\n--- Evaluating DCGAN Baseline ---")
    gan_baseline_path = 'output/dcgan_baseline/samples/epoch_050.png'
    if os.path.exists(gan_baseline_path):
        print("  Loading generated samples...")
        fake_samples = load_generated_samples(gan_baseline_path, num_samples=64)
        
        print("  Computing FID score...")
        start = time.time()
        # Use subset of real samples matching fake samples size for fair comparison
        real_subset = real_samples[:64]
        fid_score = compute_fid_score(real_subset, fake_samples, device)
        fid_time = time.time() - start
        print(f"    FID Score: {fid_score:.2f} (computed in {fid_time:.2f}s)")
        
        print("  Computing Inception Score...")
        start = time.time()
        is_mean, is_std = compute_inception_score(fake_samples, device)
        is_time = time.time() - start
        print(f"    Inception Score: {is_mean:.3f} ± {is_std:.3f} (computed in {is_time:.2f}s)")
        
        print("  Computing diversity metrics...")
        diversity = compute_diversity_metrics(fake_samples)
        print(f"    Inter-sample Variance: {diversity['inter_sample_variance']:.4f}")
        print(f"    Mean Pairwise Distance: {diversity['mean_pairwise_distance']:.4f}")
        
        results['dcgan_baseline'] = {
            'fid_score': fid_score,
            'inception_score_mean': is_mean,
            'inception_score_std': is_std,
            **diversity
        }
    else:
        print(f"  ⚠ File not found: {gan_baseline_path}")
    
    # ===== Evaluate DCGAN Modified =====
    print("\n--- Evaluating DCGAN Modified ---")
    gan_modified_path = 'output/dcgan_modified/samples/epoch_050.png'
    if os.path.exists(gan_modified_path):
        print("  Loading generated samples...")
        fake_samples = load_generated_samples(gan_modified_path, num_samples=64)
        
        print("  Computing FID score...")
        real_subset = real_samples[:64]
        fid_score = compute_fid_score(real_subset, fake_samples, device)
        print(f"    FID Score: {fid_score:.2f}")
        
        print("  Computing Inception Score...")
        is_mean, is_std = compute_inception_score(fake_samples, device)
        print(f"    Inception Score: {is_mean:.3f} ± {is_std:.3f}")
        
        print("  Computing diversity metrics...")
        diversity = compute_diversity_metrics(fake_samples)
        print(f"    Inter-sample Variance: {diversity['inter_sample_variance']:.4f}")
        print(f"    Mean Pairwise Distance: {diversity['mean_pairwise_distance']:.4f}")
        
        results['dcgan_modified'] = {
            'fid_score': fid_score,
            'inception_score_mean': is_mean,
            'inception_score_std': is_std,
            **diversity
        }
    else:
        print(f"  ⚠ File not found: {gan_modified_path}")
    
    # ===== Evaluate Diffusion Denoising =====
    print("\n--- Evaluating Diffusion Model Denoising ---")
    diffusion_results = {}
    for noise_level in [0.3, 0.5, 0.7]:
        denoise_path = f'output/diffusion_denoise/denoise_noise_{noise_level}.png'
        if os.path.exists(denoise_path):
            print(f"  Noise σ={noise_level}:")
            metrics = compute_denoising_metrics(denoise_path)
            print(f"    MSE: {metrics['mean_mse']:.2f} ± {metrics['std_mse']:.2f}")
            print(f"    PSNR: {metrics['mean_psnr']:.2f} ± {metrics['std_psnr']:.2f} dB")
            print(f"    Correlation: {metrics['mean_correlation']:.3f}")
            
            diffusion_results[f'noise_{noise_level}'] = metrics
        else:
            print(f"  ⚠ File not found: {denoise_path}")
    
    results['diffusion_denoising'] = diffusion_results
    
    # Save results
    output_path = os.path.join(args.output_dir, 'quantitative_metrics.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Metrics saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("METRIC INTERPRETATION GUIDE")
    print("="*60)
    print("\nFID Score (Lower is better):")
    print("  - Measures quality and diversity compared to real data")
    print("  - <10: Excellent, 10-50: Good, 50-100: Fair, >100: Poor")
    print("  - Baseline vs Modified shows quality difference")
    
    print("\nInception Score (Higher is better):")
    print("  - Measures quality and diversity of generated images")
    print("  - For MNIST: 5-10 is typical, >10 is excellent")
    print("  - Low score indicates mode collapse or poor quality")
    
    print("\nDiversity Metrics:")
    print("  - Inter-sample Variance: Higher = more diverse outputs")
    print("  - Mean Pairwise Distance: Higher = less mode collapse")
    
    print("\nDenoising Metrics:")
    print("  - MSE: Lower = better reconstruction")
    print("  - PSNR: Higher = better quality (>20dB is good, >30dB is excellent)")
    print("  - Correlation: Closer to 1.0 = better reconstruction")
    
    # Create comparison summary
    if 'dcgan_baseline' in results and 'dcgan_modified' in results:
        print("\n" + "="*60)
        print("QUICK COMPARISON")
        print("="*60)
        baseline_fid = results['dcgan_baseline']['fid_score']
        modified_fid = results['dcgan_modified']['fid_score']
        baseline_is = results['dcgan_baseline']['inception_score_mean']
        modified_is = results['dcgan_modified']['inception_score_mean']
        
        print(f"\nFID Score:        Baseline: {baseline_fid:.2f}  |  Modified: {modified_fid:.2f}  |  Winner: {'Baseline' if baseline_fid < modified_fid else 'Modified'}")
        print(f"Inception Score:  Baseline: {baseline_is:.3f}  |  Modified: {modified_is:.3f}  |  Winner: {'Baseline' if baseline_is > modified_is else 'Modified'}")


if __name__ == "__main__":
    main()
