import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tqdm import tqdm
import argparse

import random
import numpy as np
import torch
import csv
import time
import json

SEED = 77  # You can choose any integer
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim=100, channels=1, depth=3, dropout=0.0):
        super(Generator, self).__init__()

        layers = []
        # Define channel progression based on depth
        channel_multipliers = [256, 128, 64, 32][:depth]
        
        # For 28x28 images (MNIST)
        # Input: latent_dim x 1 x 1
        layers.extend([
            nn.ConvTranspose2d(latent_dim, channel_multipliers[0], 7, 1, 0, bias=False),
            nn.BatchNorm2d(channel_multipliers[0]),
            nn.ReLU(True)
        ])
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        # State: channel_multipliers[0] x 7 x 7

        # Middle layers based on depth
        for i in range(1, depth):
            layers.extend([
                nn.ConvTranspose2d(channel_multipliers[i-1], channel_multipliers[i], 4, 2, 1, bias=False),
                nn.BatchNorm2d(channel_multipliers[i]),
                nn.ReLU(True)
            ])
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))

        # Output layer
        layers.extend([
            nn.Conv2d(channel_multipliers[depth-1], channels, 3, 1, 1),
            nn.Tanh()
        ])
        # Output: channels x 28 x 28
        
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, channels=1, depth=3, dropout=0.0):
        super(Discriminator, self).__init__()

        layers = []
        # Define channel progression based on depth
        channel_multipliers = [64, 128, 256, 512][:depth]
        
        # Input: channels x 28 x 28
        layers.extend([
            nn.Conv2d(channels, channel_multipliers[0], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        # State: channel_multipliers[0] x 14 x 14

        # Middle layers based on depth
        for i in range(1, depth):
            stride = 2
            layers.extend([
                nn.Conv2d(channel_multipliers[i-1], channel_multipliers[i], 4, stride, 1, bias=False),
                nn.BatchNorm2d(channel_multipliers[i]),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))

        self.main = nn.Sequential(*layers)
        
        # Adaptive pooling to ensure 1x1 output regardless of depth
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        
        # Final classification layer
        self.final = nn.Sequential(
            nn.Conv2d(channel_multipliers[depth-1], 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.main(x)
        out = self.adaptive_pool(out)
        out = self.final(out)
        return out.view(x.size(0), 1)


# Weight initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train_dcgan(dataset_name='mnist', num_epochs=50, output_dir='output/',
                latent_dim=100, lr=0.0002, batch_size=128, beta1=0.5,
                gen_depth=3, disc_depth=3, dropout=0.0):
    """
    Train DCGAN on specified dataset

    Args:
        dataset_name: Name of dataset ('mnist' or 'fashion-mnist')
        num_epochs: Number of training epochs
        output_dir: Directory to save outputs
        latent_dim: Dimension of latent space
        lr: Learning rate
        batch_size: Batch size for training
        beta1: Beta1 parameter for Adam optimizer
        gen_depth: Number of layers in generator
        disc_depth: Number of layers in discriminator
        dropout: Dropout rate for regularization
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Training DCGAN on {dataset_name}")
    print(f"Device: {device}")
    print(
        f"Configuration: latent_dim={latent_dim}, lr={lr}, batch_size={batch_size}, beta1={beta1}, gen_depth={gen_depth}, disc_depth={disc_depth}, dropout={dropout}")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/samples", exist_ok=True)
    os.makedirs(f"{output_dir}/models", exist_ok=True)

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

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=8, pin_memory=True)

    # Initialize models
    generator = Generator(latent_dim, channels, depth=gen_depth, dropout=dropout).to(device)
    discriminator = Discriminator(channels, depth=disc_depth, dropout=dropout).to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Loss and optimizers
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(),
                             lr=lr, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(),
                             lr=lr, betas=(beta1, 0.999))

    # Fixed noise for visualization
    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

    # Training loop
    print("Starting Training...")
    G_losses = []
    D_losses = []
    
    # Start timing
    start_time = time.time()

    pbar = tqdm(total=num_epochs * len(dataloader), desc="Training DCGAN", unit="batch")
    
    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)

            # Labels
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # ============================================
            # Train Discriminator
            # ============================================
            optimizer_D.zero_grad()

            # Real images
            output_real = discriminator(real_images)
            loss_D_real = criterion(output_real, real_labels)

            # Fake images
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
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
            # Generator wants D to think fake is real
            loss_G = criterion(output, real_labels)
            loss_G.backward()
            optimizer_G.step()

            # Save losses
            G_losses.append(loss_G.item())
            D_losses.append(loss_D.item())

            # Update progress bar every batch (interactive terminal)
            pbar.set_postfix({
                "Epoch": f"{epoch+1}/{num_epochs}",
                "Loss D": f"{loss_D.item():.4f}",
                "Loss G": f"{loss_G.item():.4f}"
            })
            pbar.update(1)

        # Save checkpoints every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_dir = f'{output_dir}/checkpoints'
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'G_losses': G_losses,
                'D_losses': D_losses,
            }, f'{checkpoint_dir}/checkpoint_epoch_{epoch+1:03d}.pth')
            print(f"\nCheckpoint saved at epoch {epoch+1}")

        # Generate and save sample images
        if (epoch + 1) % 5 == 0 or epoch == 0:
            with torch.no_grad():
                fake = generator(fixed_noise).detach().cpu()
                img_grid = vutils.make_grid(
                    fake, padding=2, normalize=True, nrow=8)
                plt.figure(figsize=(10, 10))
                plt.imshow(np.transpose(img_grid, (1, 2, 0)))
                plt.axis('off')
                plt.title(f'Epoch {epoch+1}')
                plt.savefig(f'{output_dir}/samples/epoch_{epoch+1:03d}.png')
                plt.close()

    pbar.close()
    
    # End timing
    end_time = time.time()
    total_training_time = end_time - start_time
    
    print(f"\nTotal training time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
    
    # Calculate inference time (average over 100 samples)
    print("Measuring inference time...")
    inference_times = []
    generator.eval()
    with torch.no_grad():
        for _ in range(100):
            test_noise = torch.randn(1, latent_dim, 1, 1, device=device)
            inf_start = time.time()
            _ = generator(test_noise)
            inf_end = time.time()
            inference_times.append(inf_end - inf_start)
    
    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    
    print(f"Average inference time: {avg_inference_time*1000:.2f} ms (±{std_inference_time*1000:.2f} ms)")
    
    # Calculate metrics
    final_G_loss = np.mean(G_losses[-100:])  # Average of last 100 iterations
    final_D_loss = np.mean(D_losses[-100:])
    min_G_loss = np.min(G_losses)
    min_D_loss = np.min(D_losses)
    
    print(f"\nFinal Generator Loss (avg last 100): {final_G_loss:.4f}")
    print(f"Final Discriminator Loss (avg last 100): {final_D_loss:.4f}")
    print(f"Min Generator Loss: {min_G_loss:.4f}")
    print(f"Min Discriminator Loss: {min_D_loss:.4f}")
    
    # Save final models directly to output_dir
    torch.save(generator.state_dict(),
               f'{output_dir}/models/dcgan_generator_{dataset_name}.pth')
    torch.save(discriminator.state_dict(),
               f'{output_dir}/models/dcgan_discriminator_{dataset_name}.pth')

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(G_losses, label='Generator Loss', alpha=0.7)
    plt.plot(D_losses, label='Discriminator Loss', alpha=0.7)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('DCGAN Training Losses')
    plt.savefig(f'{output_dir}/training_losses.png')
    plt.close()

    # Save losses to CSV
    # Save losses to CSV
    with open(f'{output_dir}/losses.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Iteration', 'Generator Loss', 'Discriminator Loss'])
        for i, (g_loss, d_loss) in enumerate(zip(G_losses, D_losses)):
            writer.writerow([i, g_loss, d_loss])
    
    # Save metrics to JSON
    metrics = {
        "configuration": {
            "dataset": dataset_name,
            "epochs": num_epochs,
            "latent_dim": latent_dim,
            "learning_rate": lr,
            "batch_size": batch_size,
            "generator_depth": gen_depth,
            "discriminator_depth": disc_depth,
            "dropout": dropout,
            "beta1": beta1
        },
        "training_performance": {
            "total_training_time_seconds": round(total_training_time, 2),
            "total_training_time_minutes": round(total_training_time / 60, 2),
            "time_per_epoch_seconds": round(total_training_time / num_epochs, 2),
            "total_iterations": len(G_losses)
        },
        "inference_performance": {
            "average_inference_time_ms": round(avg_inference_time * 1000, 2),
            "std_inference_time_ms": round(std_inference_time * 1000, 2)
        },
        "loss_metrics": {
            "final_generator_loss": round(final_G_loss, 4),
            "final_discriminator_loss": round(final_D_loss, 4),
            "min_generator_loss": round(min_G_loss, 4),
            "min_discriminator_loss": round(min_D_loss, 4)
        }
    }
    
    with open(f'{output_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Training complete! Models saved to {output_dir}/models/")
    print(f"Samples saved to {output_dir}/samples/")
    print(f"Metrics saved to {output_dir}/metrics.json")

    return generator, discriminator, G_losses, D_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DCGAN')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion-mnist'],
                        help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory')
    # Configuration parameters for experiments
    parser.add_argument('--latent-dim', type=int, default=100,
                        help='Latent dimension for generator')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Beta1 for Adam optimizer')
    parser.add_argument('--gen-depth', type=int, default=3,
                        help='Number of layers in generator (default: 3)')
    parser.add_argument('--disc-depth', type=int, default=3,
                        help='Number of layers in discriminator (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate for regularization (default: 0.0)')

    args = parser.parse_args()

    # Pass all parameters to train_dcgan
    train_dcgan(
        dataset_name=args.dataset,
        num_epochs=args.epochs,
        output_dir=args.output_dir,
        latent_dim=args.latent_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        beta1=args.beta1,
        gen_depth=args.gen_depth,
        disc_depth=args.disc_depth,
        dropout=args.dropout
    )
