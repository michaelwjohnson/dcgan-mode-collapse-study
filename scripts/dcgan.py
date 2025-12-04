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
import itertools

# Hyperparameters
LATENT_DIM = 100
IMAGE_SIZE = 28  # For MNIST dataset (28x28 images)
BATCH_SIZE = 128  # Batch size for training
NUM_EPOCHS = 50  # Number of epochs to train
LEARNING_RATE = 0.0002  # Learning rate for optimizers
BETA1 = 0.5  # Beta1 hyperparameter for Adam optimizers
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator Network


class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, channels=1):
        super(Generator, self).__init__()

        # For MNIST (28x28), we use a smaller architecture
        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1 (Z vector)
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
            nn.Tanh(),
            # Output: channels x 28 x 28
        )

        def forward(self, input):
            return self.main(input)

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
            nn.Sigmoid(),
            # Output: 1 x 1 x 1
        )

    def forward(self, input):
        return self.main(input).view(-1, 1)

# Weight Initialization


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != 1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train_dcgan(dataset='mnist', epochs=NUM_EPOCHS, save_dir='outputs/dcgan'):
    '''
    Train DCGAN on specified dataset
    '''
    print(f'Training DCGAN on {dataset}')
    print(f'Device: {DEVICE}')

    # Create output directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f'{save_dir}/samples', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = datasets.MNIST(root='./data', train=True,
                             download=True, transform=transform)
    channels = 1  # MNIST has 1 channel (grayscale)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=4)
    # Initialize models
    generator = Generator(latent_dim=LATENT_DIM, channels=channels).to(DEVICE)
    discriminator = Discriminator(channels=channels).to(DEVICE)

    generator.apply(weights_init)  # Initialize generator weights
    discriminator.apply(weights_init)  # Initialize discriminator weights

    # Loss and optimizers
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer_G = optim.Adam(generator.parameters(),
                             lr=LEARNING_RATE, betas=(BETA1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(),
                             lr=LEARNING_RATE, betas=(BETA1, 0.999))

    # Fixed noise for visualization
    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=DEVICE)

    # Training loop
    G_losses = []
    D_losses = []

    total_steps = epochs * len(dataloader)
    progress_bar = tqdm(itertools.product(range(epochs), enumerate(
        dataloader)), total=total_steps, desc="Training DCGAN")

    for epoch, (i, (real_images, _)) in itertools.product(range(epochs), enumerate(dataloader)):
        batch_size = real_images.size(0)
        real_images = real_images.to(DEVICE)

        # Labels
        real_labels = torch.ones(batch_size, 1, device=DEVICE)
        fake_labels = torch.zeros(batch_size, 1, device=DEVICE)

        # =======================================================
        # Train Discriminator
        # =======================================================
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

        # =======================================================
        # Train Generator
        # ======================================================
        optimizer_G.zero_grad()

        output = discriminator(fake_images)
        # Generator wants Discriminator to think fake is real
        loss_G = criterion(output, real_labels)
        loss_G.backward()
        optimizer_G.step()

        # Save losses
        G_losses.append(loss_G.item())
        D_losses.append(loss_D.item())

        # Update progress bar with loss info and current epoch
        progress_bar.set_postfix({
            'Epoch': epoch + 1, 'Loss D': loss_D.item(), 'Loss G': loss_G.item()})

        # Generate and save sample images
        