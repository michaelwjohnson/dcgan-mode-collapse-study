#!/bin/bash
#SBATCH --job-name=train_dcgan_modified
#SBATCH --output=logs/dcgan_modified_%j.out
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

mkdir -p logs output/dcgan_modified models/dcgan_modified

# Load apptainer
module load apptainer/1.3.4-gcc-14.2.0-g7o5w4g

CONTAINER=container/dcgan_diffusion.sif

echo "Starting DCGAN MODIFIED training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Run training with modified configuration
apptainer exec --nv $CONTAINER python3 scripts/train_dcgan.py \
    --dataset mnist \
    --epochs 50 \
    --latent-dim 64 \
    --lr 0.0002 \
    --batch-size 128 \
    --beta1 0.5 \
    --gen-depth 4 \
    --disc-depth 4 \
    --dropout 0.3 \
    --output-dir output/dcgan_modified

# Rationale:

# Reduced latent dimension (64): Should produce less diverse images and potentially show mode collapse more clearly
# Increased depth (4 layers): Adds model complexity, which can improve quality but may also increase training instability
# Dropout (0.3): Adds regularization to prevent overfitting and potentially improve generalization
# Same learning rate: Keeps training speed comparable for fair comparison
# This combination will help observe:

# Effects of latent space size on diversity
# Impact of network depth on image quality and training stability
# How regularization affects mode collapse and generalization


echo "Modified training complete!"

