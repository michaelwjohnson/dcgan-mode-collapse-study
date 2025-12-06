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

mkdir -p logs output/diffusion_text2img output/gan_diff_comparison

# Load apptainer
module load apptainer/1.3.4-gcc-14.2.0-g7o5w4g

CONTAINER=container/dcgan_diffusion.sif

echo "Starting Stable Diffusion Experiments"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Run text-to-image experiment with Stable Diffusion
echo ""
echo "=== EXPERIMENT: Text-to-Image with Stable Diffusion ==="
apptainer exec --nv $CONTAINER python3 scripts/train_diffusion.py \
    --mode text2img \
    --num-images 4

# Comparison with GAN
echo ""
echo "=== EXPERIMENT: Comparison with GAN ==="
apptainer exec --nv $CONTAINER python3 scripts/train_diffusion.py \
    --mode compare

echo ""
echo "All diffusion experiments complete!"
echo "Results saved to:"
echo "  - output/diffusion_text2img/"
echo "  - output/gan_diff_comparison/"