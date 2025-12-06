#!/bin/bash
#SBATCH --job-name=diffusion_exp
#SBATCH --output=logs/diffusion_%j.out
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

mkdir -p logs output/diffusion_denoise output/gan_diff_comparison

# Load apptainer
module load apptainer/1.3.4-gcc-14.2.0-g7o5w4g

CONTAINER=container/dcgan_diffusion.sif

echo "Starting Diffusion Model Denoising Experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Run denoising experiment (demonstrates forward and reverse diffusion)
echo ""
echo "=== EXPERIMENT: Denoising Diffusion Process ==="
echo "Adding Gaussian noise to MNIST images and denoising them"
echo "Noise levels: 0.3, 0.5, 0.7"
echo ""
apptainer exec --nv $CONTAINER python3 scripts/train_diffusion.py \
    --mode denoise \
    --num-samples 8 \
    --inference-steps 50

# Comparison with GAN
echo ""
echo "=== COMPARISON: Diffusion vs GAN ==="
apptainer exec --nv $CONTAINER python3 scripts/train_diffusion.py \
    --mode compare

echo ""
echo "Diffusion experiment complete!"
echo "Results saved to:"
echo "  - output/diffusion_denoise/"
echo "  - output/gan_diff_comparison/"