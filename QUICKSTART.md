# Quick Start Guide

This guide will help you get the DCGAN-Diffusion comparison project running quickly.

## ⚡ 5-Minute Setup

### Step 1: Clone and Setup (1 min)

```bash
git clone https://github.com/yourusername/dcgan-diffusion-comparison.git
cd dcgan-diffusion-comparison

# Load Apptainer module (adjust for your cluster)
module load apptainer/1.3.4
```

### Step 2: Build Container (2 min)

```bash
apptainer build container/dcgan_diffusion.sif container/apptainer.def
```

### Step 3: Run Your First Experiment (2 min)

**Test GPU access:**
```bash
sbatch slurm/test_gpu.slurm
# Wait ~30 seconds, then check:
cat logs/test_gpu_*.out
```

**Train baseline DCGAN:**
```bash
sbatch slurm/train_dcgan_baseline.slurm
# Training takes ~7 minutes
```

**Monitor progress:**
```bash
# Check if job is running
squeue -u $USER

# Watch live training logs
tail -f logs/dcgan_baseline_*.out
```

### Step 4: View Results

After training completes (~7 minutes):

```bash
# Generated samples
ls output/dcgan_baseline/samples/

# Training curves
ls output/dcgan_baseline/training_losses.png

# Model checkpoint
ls models/dcgan_baseline/
```

## 🎯 Common Tasks

### Compare Baseline vs Modified DCGAN

```bash
# Train both configurations
sbatch slurm/train_dcgan_baseline.slurm
sbatch slurm/train_dcgan_modified.slurm

# Compute comparison metrics
sbatch slurm/compute_metrics.slurm

# View results
cat output/metrics/comparison.txt
```

### Run Diffusion Model Denoising

```bash
sbatch slurm/train_diffusion.slurm

# View denoised results
ls output/diffusion_denoise/
```

### Custom Hyperparameters

Edit the SLURM script to modify training parameters:

```bash
# Edit slurm/train_dcgan_baseline.slurm
--latent_dim 128        # Change latent dimension
--depth 4               # Change network depth
--dropout 0.2           # Add dropout
--epochs 100            # Train longer
--batch_size 256        # Larger batches
```

## 🔧 Troubleshooting

### GPU Not Available
```bash
# Check available partitions
sinfo

# Check GPU nodes
scontrol show partition gpu1

# Use different partition in SLURM script
#SBATCH --partition=gpu2
```

### Out of Memory
```bash
# Reduce batch size in training script
--batch_size 64  # Instead of 128

# Request more memory in SLURM script
#SBATCH --mem=64G  # Instead of 32G
```

### Container Build Fails
```bash
# Use sandbox build for debugging
apptainer build --sandbox sandbox/ container/apptainer.def

# Or pull pre-built base image
apptainer pull docker://nvcr.io/nvidia/pytorch:25.09-py3
```

### Jobs Pending Too Long
```bash
# Check job priority
squeue -u $USER --start

# Use different partition
#SBATCH --partition=gpu-short

# Reduce resource requirements
#SBATCH --time=01:00:00  # Shorter time limit
#SBATCH --gres=gpu:1     # Single GPU
```

## 📊 Expected Results

### Baseline DCGAN (Success)
- **Training time**: ~7-8 minutes for 50 epochs
- **Final G loss**: ~1.75
- **Final D loss**: ~0.65
- **Samples**: Diverse, recognizable digits
- **Variance**: >0.2

### Modified DCGAN (Mode Collapse)
- **Training time**: ~12 minutes for 50 epochs
- **Final G loss**: >10 (diverged)
- **Final D loss**: ~0.0001 (collapsed)
- **Samples**: Identical grid patterns
- **Variance**: <0.1

### Diffusion Model
- **Inference time**: ~150-300ms per sample
- **Quality**: Consistent, high-quality digits
- **Behavior**: Generative, not reconstructive

## 🚀 Next Steps

1. **Explore the paper**: Read [documents/report/main.tex](documents/report/main.tex) for detailed analysis
2. **Modify architectures**: Edit [python/train_dcgan.py](python/train_dcgan.py) to test new configurations
3. **Add metrics**: Implement FID/IS scores in [python/compute_metrics.py](python/compute_metrics.py)
4. **Try new datasets**: Extend to CIFAR-10 or CelebA
5. **Implement improvements**: Add spectral normalization, progressive growing, etc.

## 💡 Tips

- **Save checkpoints regularly**: Training can be interrupted
- **Use tmux/screen**: Keep sessions alive during long runs
- **Monitor GPU usage**: `watch -n 1 nvidia-smi`
- **Compare configurations**: Use WandB or TensorBoard for tracking
- **Document experiments**: Keep a lab notebook of hyperparameters and results

## 📚 Further Reading

- [DCGAN Paper (Radford et al., 2016)](https://arxiv.org/abs/1511.06434)
- [DDPM Paper (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- [GAN Training Tips](https://github.com/soumith/ganhacks)

## 🆘 Getting Help

- **Check logs**: Always start with `cat logs/jobname_*.err`
- **Read error messages**: They usually indicate the exact problem
- **Open an issue**: Provide logs, system info, and steps to reproduce
- **Ask questions**: No question is too basic!

Happy experimenting! 🎨
