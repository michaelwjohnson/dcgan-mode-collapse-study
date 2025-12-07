#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --output=logs/test_gpu_%j.out
#SBATCH --partition=gpu1         # Use your cluster's GPU partition name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --mem=16G
#SBATCH --time=00:10:00        # 10 minutes

# Create logs directory
mkdir -p logs

# Load Apptainer if needed
module load apptainer/1.3.4-gcc-14.2.0-g7o5w4g

# Path to your container
CONTAINER=container/dcgan_diffusion.sif

# Test GPU access
apptainer exec --nv $CONTAINER python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
"