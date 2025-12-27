# Comparative Analysis of DCGANs and Diffusion Models for MNIST Generation

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 12.2](https://img.shields.io/badge/CUDA-12.2-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

A comprehensive comparative study examining Deep Convolutional Generative Adversarial Networks (DCGANs) and diffusion models for MNIST digit generation, with focus on training stability, mode collapse, and computational tradeoffs.

## Research Overview

This project investigates fundamental tradeoffs between two prominent generative model families:

- **DCGANs**: Fast inference (1.29ms/sample) but suffer from training instability and mode collapse
- **Diffusion Models**: Stable training and consistent quality but 115-230× slower inference (153-297ms/sample)

### Key Findings

- **Baseline DCGAN** (latent_dim=256, depth=3, dropout=0.0): Produces diverse, high-quality samples throughout 50 epochs
- **Modified DCGAN** (latent_dim=64, depth=4, dropout=0.3): Exhibits complete mode collapse by epoch 35, generating identical grid patterns
- **Diffusion Model**: Demonstrates stable generative behavior with consistent quality but significantly higher computational cost

**Full Paper**: [documents/report/main.pdf](documents/report/main.pdf)

## Visual Results

### Baseline DCGAN Training Evolution
Progressive improvement from random noise to diverse, high-quality digit generation:

- **Epoch 1**: Recognizable digit-like shapes emerge immediately
- **Epochs 10-30**: Rapid quality improvement with increasing diversity
- **Epochs 35-50**: Stable, high-quality generation across all digit classes

### Modified DCGAN Mode Collapse
Demonstrates classic mode collapse failure pattern:

- **Epochs 1-10**: Amorphous noise and grid artifacts
- **Epochs 20-30**: Rapid quality degradation
- **Epochs 35-50**: Complete collapse to identical checkerboard patterns

### Quantitative Metrics

| Metric | Baseline | Modified |
|--------|----------|----------|
| Final Generator Loss | 1.75 | 10.40 |
| Final Discriminator Loss | 0.65 | 0.0001 |
| Inter-sample Variance | 0.226 | 0.082 (-63.8%) |
| Mean Pairwise Distance | 0.451 | 0.154 (-65.8%) |
| Training Time | 7.49 min | 12.22 min |
| Inference Time | 1.29 ms | 0.70 ms |

## Quick Start

### Prerequisites

- HPC cluster with SLURM scheduler
- NVIDIA GPU (A100 40GB recommended)
- Apptainer/Singularity container runtime

### 1. Build Container

```bash
cd as4
module load apptainer/1.3.4-gcc-14.2.0-g7o5w4g
apptainer build container/dcgan_diffusion.sif container/apptainer.def
```

### 2. Train Models

**Baseline DCGAN:**
```bash
sbatch slurm/train_dcgan_baseline.slurm
```

**Modified DCGAN:**
```bash
sbatch slurm/train_dcgan_modified.slurm
```

**Diffusion Model Denoising:**
```bash
sbatch slurm/train_diffusion.slurm
```

### 3. Monitor Training

```bash
# Check job status
squeue -u $USER

# View training logs
tail -f logs/dcgan_baseline_*.out

# Monitor GPU utilization
watch -n 1 nvidia-smi
```

### 4. Compute Metrics

```bash
sbatch slurm/compute_metrics.slurm
```

## Project Structure

```
as4/
├── python/                          # Source code
│   ├── train_dcgan.py              # DCGAN training script
│   ├── train_diffusion.py          # Diffusion model inference
│   ├── compute_metrics.py          # Quantitative evaluation
│   └── generate_gan_diagram.py     # Architecture visualization
├── container/                       # Container definitions
│   ├── apptainer.def               # Container build recipe
│   └── dcgan_diffusion.sif         # Built container image
├── slurm/                          # SLURM job scripts
│   ├── train_dcgan_baseline.slurm
│   ├── train_dcgan_modified.slurm
│   ├── train_diffusion.slurm
│   └── compute_metrics.slurm
├── documents/report/               # Academic paper
│   └── main.pdf                    # IEEE conference paper (PDF)
├── output/                         # Experimental results
│   ├── dcgan_baseline/            # Baseline samples & metrics
│   ├── dcgan_modified/            # Modified samples & metrics
│   ├── diffusion_denoise/         # Diffusion denoising results
│   └── metrics/                   # Quantitative comparisons
├── models/                         # Trained model checkpoints
│   ├── dcgan_baseline/
│   └── dcgan_modified/
└── data/                          # MNIST dataset (auto-downloaded)
```

## Technical Details

### Environment

- **Base Image**: NVIDIA NGC PyTorch 25.09-py3
- **Python**: 3.11
- **PyTorch**: 2.1.0+
- **CUDA**: 12.2
- **GPU**: NVIDIA A100 40GB
- **Key Dependencies**: torchvision, diffusers, transformers, matplotlib, seaborn

### Reproducibility

All experiments use fixed random seeds (seed=77) and deterministic cuDNN settings:

```python
torch.manual_seed(77)
torch.cuda.manual_seed_all(77)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### DCGAN Architecture

- **Generator**: Transposed convolutions with batch normalization, ReLU, and Tanh output
- **Discriminator**: Strided convolutions with batch normalization, LeakyReLU, and adaptive pooling
- **Loss**: Binary cross-entropy with alternating updates
- **Optimizer**: Adam (lr=0.0002, betas=(0.5, 0.999))

### Diffusion Model

- **Architecture**: DDPM with UNet backbone and timestep conditioning
- **Pretrained**: HuggingFace `1aurent/ddpm-mnist`
- **Resolution**: 28×28 grayscale
- **Inference Steps**: 100 sequential denoising steps

## Results Summary

### Training Stability

- **Baseline**: Stable adversarial dynamics with balanced oscillating losses
- **Modified**: Rapid loss divergence - generator loss climbs from 3→10+, discriminator collapses to near-zero

### Mode Collapse Analysis

Three architectural factors combine to cause collapse in the modified configuration:

1. **Reduced Generator Capacity**: Latent dimension 64 vs. 256 constrains representation power
2. **Increased Discriminator Depth**: 4 vs. 3 layers creates power imbalance
3. **Dropout Regularization**: 0.3 dropout further slows generator learning

### Computational Tradeoffs

- **GAN Advantages**: Fast single-pass generation (1.29ms), real-time applications
- **GAN Disadvantages**: Training instability, mode collapse risk, vanishing gradients
- **Diffusion Advantages**: Stable likelihood-based training, consistent quality, no collapse
- **Diffusion Disadvantages**: 115-230× slower inference, 100 sequential denoising steps

## Citation

If you use this work in your research, please cite:

```bibtex
@article{johnson2025dcgan,
  title={Comparative Analysis of DCGANs and Diffusion Models for MNIST Generation: 
         Training Stability, Mode Collapse, and Computational Tradeoffs},
  author={Johnson, Michael},
  journal={ECE5570 Machine Learning at Scale},
  year={2025}
}
```

## References

- Goodfellow et al. (2014): [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
- Radford et al. (2016): [Unsupervised Representation Learning with DCGANs](https://arxiv.org/abs/1511.06434)
- Ho et al. (2020): [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

## License

This project is part of academic coursework for ECE5570 - Machine Learning at Scale.

## Acknowledgments

- **Course**: ECE5570 - Machine Learning at Scale
- **Institution**: Florida Institute of Technology
- **Compute Resources**: AI-Panther HPC Cluster
