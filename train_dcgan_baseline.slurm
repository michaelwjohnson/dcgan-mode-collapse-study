#!/bin/bash
#SBATCH --job-name=train_dcgan_baseline
#SBATCH --output=logs/dcgan_baseline_%j.out
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

mkdir -p logs outputs/dcgan_baseline models/dcgan_baseline

# Load apptainer
module load apptainer/1.3.4-gcc-14.2.0-g7o5w4g

CONTAINER=container/dcgan_diffusion.sif

echo "Starting DCGAN BASELINE training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Run training with baseline configuration
apptainer exec --nv $CONTAINER python3 scripts/train_dcgan.py \
    --dataset mnist \
    --epochs 50 \
    --latent-dim 256 \
    --lr 0.0002 \
    --batch-size 128 \
    --beta1 0.5 \
    --gen-depth 3 \
    --disc-depth 3 \
    --dropout 0.0 \
    --output-dir outputs/dcgan_baseline

echo "Baseline training complete!"