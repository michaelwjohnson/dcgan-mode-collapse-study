#!/bin/bash
#SBATCH --job-name=diffusion_exp
#SBATCH --output=logs/diffusion_%j.out
#SBATCH --error=logs/diffusion_%j.err
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

mkdir -p logs output/diffusion_denoise output/diffusion_generation output/gan_diff_comparison

# Load apptainer
module load apptainer/1.3.4-gcc-14.2.0-g7o5w4g

CONTAINER=container/dcgan_diffusion.sif

echo "Starting Diffusion Model Experiments"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Run all diffusion experiments
# Experiment 1: Denoising (forward and reverse diffusion)
echo ""
echo "=== EXPERIMENT 1: Denoising Diffusion Process ==="
apptainer exec --nv $CONTAINER python3 scripts/train_diffusion.py \
    --mode denoise \
    --num-samples 8

# Experiment 2: Unconditional generation with varying inference steps
echo ""
echo "=== EXPERIMENT 2: Unconditional Generation ==="
apptainer exec --nv $CONTAINER python3 scripts/train_diffusion.py \
    --mode generate \
    --inference-steps 50

# Experiment 3: Comparison with GAN
echo ""
echo "=== EXPERIMENT 3: Comparison with GAN ==="
apptainer exec --nv $CONTAINER python3 scripts/train_diffusion.py \
    --mode compare

echo ""
echo "All diffusion experiments complete!"
echo "Results saved to:"
echo "  - output/diffusion_denoise/"
echo "  - output/diffusion_generation/"
echo "  - output/gan_diff_comparison/"