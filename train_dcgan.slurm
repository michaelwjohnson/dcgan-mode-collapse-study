#!/bin/bash
#SBATCH --job-name=train_dcgan
#SBATCH --output=logs/dcgan_%j.out
#SBATCH --error=logs/dcgan_%j.err
#SBATCH --partition=gpu1         
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8   
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --mem=32G
#SBATCH --time=04:00:00          # Set a time limit for the job (hh:mm:ss)

# Create necessary directories
mkdir -p logs outputs/dcgan models

# Load apptainer
module load apptainer/1.3.4-gcc-14.2.0-g7o5w4g

# Path to the container image
CONTAINER="container/dcgan_diffusion.sif"

# Dataset to use
DATASET="mnist"
EPOCHS=50

# Display job information
echo "Starting DCGAN training on ${DATASET} dataset for ${EPOCHS} epochs."
echo "Job ID: $SLURM_JOB_ID"
echo "GPU Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Run the training script inside the container
apptainer exec --nv ${CONTAINER} python3 scripts/train_dcgan.py \
    --epochs ${EPOCHS} \
    --save_dir outputs/dcgan \

echo "DCGAN training complete"

