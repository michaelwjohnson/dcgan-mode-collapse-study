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


def train_dcgan(dataset_name='mnist', num_epochs=50, save_dir='outputs/dcgan',
                latent_dim=100, lr=0.0002, batch_size=128, beta1=0.5):
    """
    Train DCGAN on specified dataset

    Args:
        dataset_name: Name of dataset ('mnist' or 'fashion-mnist')
        num_epochs: Number of training epochs
        save_dir: Directory to save outputs
        latent_dim: Dimension of latent space
        lr: Learning rate
        batch_size: Batch size for training
        beta1: Beta1 parameter for Adam optimizer
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Training DCGAN on {dataset_name}")
    print(f"Device: {device}")
    print(
        f"Configuration: latent_dim={latent_dim}, lr={lr}, batch_size={batch_size}, beta1={beta1}")

    # Create output directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/samples", exist_ok=True)
    os.makedirs(f"{save_dir}/models", exist_ok=True)

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
                            shuffle=True, num_workers=4, pin_memory=True)

    # Initialize models
    generator = Generator(latent_dim, channels).to(device)
    discriminator = Discriminator(channels).to(device)

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
            checkpoint_dir = f'{save_dir}/checkpoints'
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
                plt.savefig(f'{save_dir}/samples/epoch_{epoch+1:03d}.png')
                plt.close()

    pbar.close()
    
    # Save final models directly to save_dir
    torch.save(generator.state_dict(),
               f'{save_dir}/models/dcgan_generator_{dataset_name}.pth')
    torch.save(discriminator.state_dict(),
               f'{save_dir}/models/dcgan_discriminator_{dataset_name}.pth')

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
    parser = argparse.ArgumentParser(description='Train DCGAN')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion-mnist'],
                        help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--output-dir', type=str, default='outputs/dcgan',
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

    args = parser.parse_args()

    # Pass all parameters to train_dcgan
    train_dcgan(
        dataset_name=args.dataset,
        num_epochs=args.epochs,
        save_dir=args.output_dir,
        latent_dim=args.latent_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        beta1=args.beta1
    )
