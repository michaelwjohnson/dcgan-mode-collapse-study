#!/usr/bin/env python3
"""
Generate GAN architecture diagram using PlotNeuralNet
Creates a LaTeX-based visualization with actual MNIST images
"""

import os
import sys
import torch
import torchvision
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

# Add PlotNeuralNet to path
sys.path.append("../PlotNeuralNet")


def save_mnist_samples(
    output_dir="../PlotNeuralNet/examples/mnist_samples", num_samples=16
):
    """Load and save MNIST sample images"""
    os.makedirs(output_dir, exist_ok=True)

    # Load MNIST dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    mnist_data = datasets.MNIST(
        root="../data", train=True, download=False, transform=transform
    )

    # Get diverse samples (one per digit class, plus extras)
    indices = []
    seen_classes = set()

    for idx, (img, label) in enumerate(mnist_data):
        if label not in seen_classes:
            indices.append(idx)
            seen_classes.add(label)
        if len(indices) >= num_samples:
            break

    # If we need more samples, add random ones
    while len(indices) < num_samples:
        indices.append(np.random.randint(0, len(mnist_data)))

    # Save samples
    for i, idx in enumerate(indices):
        img, label = mnist_data[idx]
        # Convert to PIL and save
        img_np = img.squeeze().numpy()
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8), mode="L")
        img_pil.save(os.path.join(output_dir, f"mnist_sample_{i:02d}.png"))

    print(f"Saved {len(indices)} MNIST samples to {output_dir}")
    return output_dir


def generate_gan_diagram():
    """Generate LaTeX file for GAN architecture diagram"""

    # First, save MNIST samples
    mnist_dir = save_mnist_samples()

    # Create LaTeX file for DCGAN architecture
    latex_content = r"""\documentclass[border=8pt, multi, tikz]{standalone}
\usepackage{import}
\subimport{../layers/}{init}
\usetikzlibrary{positioning}
\usetikzlibrary{3d} %for including external image

\def\ConvColor{rgb:yellow,5;red,2.5;white,5}
\def\ConvReluColor{rgb:yellow,5;red,5;white,5}
\def\PoolColor{rgb:red,1;black,0.3}
\def\UnpoolColor{rgb:blue,2;green,1;black,0.3}
\def\FcColor{rgb:blue,5;red,2.5;white,5}
\def\FcReluColor{rgb:blue,5;red,5;white,4}
\def\SoftmaxColor{rgb:magenta,5;black,7}
\def\NoiseColor{rgb:green,1;black,0.3}

\newcommand{\copymidarrow}{\tikz \draw[-Stealth,line width=0.8mm,draw={rgb:blue,4;red,1;green,1;black,3}] (-0.3,0) -- ++(0.3,0);}

\begin{document}
\begin{tikzpicture}
\tikzstyle{connection}=[ultra thick,every node/.style={sloped,allow upside down},draw=\edgecolor,opacity=0.7]
\tikzstyle{copyconnection}=[ultra thick,every node/.style={sloped,allow upside down},draw={rgb:blue,4;red,1;green,1;black,3},opacity=0.7]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DCGAN Architecture - Baseline Configuration
%% Generator: 256x1x1 -> 256x7x7 -> 128x14x14 -> 64x28x28 -> 1x28x28
%% Discriminator: 1x28x28 -> 64x14x14 -> 128x7x7 -> 256x3x3 -> 1x1x1 -> Real/Fake
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Real MNIST Training Data
\pic[shift={(0,0,0)}] at (0,0,0) {Box={name=mnist_real,caption=Real MNIST,fill=\ConvColor,height=28,width=2,depth=28}};
\node[canvas is zy plane at x=0] at (mnist_real-west) {\includegraphics[width=4cm,height=4cm]{mnist_samples/mnist_sample_00.png}};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Generator Network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Noise Input
\pic[shift={(3,0,0)}] at (0,0,0) {Box={name=noise,caption=Latent Vector $z$,fill=\NoiseColor,height=1,width=1,depth=1,zlabel=256}};

%% Generator Layer 1: 256x1x1 -> 256x7x7 (ConvTranspose2d)
\pic[shift={(2,0,0)}] at (noise-east) {RightBandedBox={name=gen1,caption=ConvT 4x4,fill=\UnpoolColor,bandfill=\ConvReluColor,height=7,width=8,depth=7,zlabel=256}};

%% Generator Layer 2: 256x7x7 -> 128x14x14 (ConvTranspose2d)
\pic[shift={(2,0,0)}] at (gen1-east) {RightBandedBox={name=gen2,caption=ConvT 4x4,fill=\UnpoolColor,bandfill=\ConvReluColor,height=14,width=6,depth=14,zlabel=128}};

%% Generator Layer 3: 128x14x14 -> 64x28x28 (ConvTranspose2d)
\pic[shift={(2,0,0)}] at (gen2-east) {RightBandedBox={name=gen3,caption=ConvT 4x4,fill=\UnpoolColor,bandfill=\ConvReluColor,height=28,width=4,depth=28,zlabel=64}};

%% Generator Output: 64x28x28 -> 1x28x28 (ConvTranspose2d + Tanh)
\pic[shift={(2,0,0)}] at (gen3-east) {RightBandedBox={name=gen_out,caption=ConvT 4x4 + Tanh,fill=\UnpoolColor,bandfill=\SoftmaxColor,height=28,width=2,depth=28,zlabel=1}};

%% Fake Generated Image
\pic[shift={(3,0,0)}] at (gen_out-east) {Box={name=fake_img,caption=Fake Image,fill=\ConvColor,height=28,width=2,depth=28}};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Discriminator Network (processes both real and fake images)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Position discriminator below generator
\pic[shift={(0,-15,0)}] at (fake_img-south) {Box={name=disc_input,caption=Input Image,fill=\ConvColor,height=28,width=2,depth=28,zlabel=1}};

%% Discriminator Layer 1: 1x28x28 -> 64x14x14 (Conv2d)
\pic[shift={(2,0,0)}] at (disc_input-east) {RightBandedBox={name=disc1,caption=Conv 4x4,fill=\PoolColor,bandfill=\ConvReluColor,height=14,width=4,depth=14,zlabel=64}};

%% Discriminator Layer 2: 64x14x14 -> 128x7x7 (Conv2d)
\pic[shift={(2,0,0)}] at (disc1-east) {RightBandedBox={name=disc2,caption=Conv 4x4,fill=\PoolColor,bandfill=\ConvReluColor,height=7,width=6,depth=7,zlabel=128}};

%% Discriminator Layer 3: 128x7x7 -> 256x3x3 (Conv2d)
\pic[shift={(2,0,0)}] at (disc2-east) {RightBandedBox={name=disc3,caption=Conv 4x4,fill=\PoolColor,bandfill=\ConvReluColor,height=3,width=8,depth=3,zlabel=256}};

%% Discriminator Output: 256x3x3 -> 1x1x1 (AdaptiveAvgPool + Conv 1x1 + Sigmoid)
\pic[shift={(2,0,0)}] at (disc3-east) {RightBandedBox={name=disc_out,caption=Pool + Conv 1x1,fill=\PoolColor,bandfill=\SoftmaxColor,height=1,width=4,depth=1,zlabel=1}};

%% Classification Output
\pic[shift={(2,0,0)}] at (disc_out-east) {Box={name=class_out,caption=Real/Fake,fill=\FcReluColor,height=3,width=3,depth=3}};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Connections
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Generator connections
\draw [connection]  (noise-east)        -- node {\midarrow} (gen1-west);
\draw [connection]  (gen1-east)         -- node {\midarrow} (gen2-west);
\draw [connection]  (gen2-east)         -- node {\midarrow} (gen3-west);
\draw [connection]  (gen3-east)         -- node {\midarrow} (gen_out-west);
\draw [connection]  (gen_out-east)      -- node {\midarrow} (fake_img-west);

%% Connection from fake image to discriminator
\draw [connection]  (fake_img-south)    -- node {\midarrow} (disc_input-north) node[midway,right] {Fake};

%% Connection from real MNIST to discriminator (curved)
\path (mnist_real-south) -- (disc_input-north) coordinate[midway] (middle);
\draw [connection] (mnist_real-south) -- (middle) -- (disc_input-north) node[midway,left] {Real};

%% Discriminator connections
\draw [connection]  (disc_input-east)   -- node {\midarrow} (disc1-west);
\draw [connection]  (disc1-east)        -- node {\midarrow} (disc2-west);
\draw [connection]  (disc2-east)        -- node {\midarrow} (disc3-west);
\draw [connection]  (disc3-east)        -- node {\midarrow} (disc_out-west);
\draw [connection]  (disc_out-east)     -- node {\midarrow} (class_out-west);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Labels
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Add Generator label
\node[draw, fill=white, rounded corners, align=center] at ($(gen2-north)+(0,3,0)$) {\Large \textbf{Generator $G$}};

%% Add Discriminator label
\node[draw, fill=white, rounded corners, align=center] at ($(disc2-north)+(0,3,0)$) {\Large \textbf{Discriminator $D$}};

\end{tikzpicture}
\end{document}
"""

    # Write LaTeX file
    output_path = "../PlotNeuralNet/examples/dcgan_architecture.tex"
    with open(output_path, "w") as f:
        f.write(latex_content)

    print(f"\nGenerated LaTeX file: {output_path}")
    print("\nTo compile the diagram:")
    print(f"  cd PlotNeuralNet/examples")
    print(f"  pdflatex dcgan_architecture.tex")
    print(f"\nThis will create: dcgan_architecture.pdf")

    return output_path


if __name__ == "__main__":
    generate_gan_diagram()
