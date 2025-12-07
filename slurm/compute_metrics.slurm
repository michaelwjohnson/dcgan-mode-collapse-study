#!/bin/bash
#SBATCH --job-name=compute_metrics
#SBATCH --output=logs/metrics_%j.out
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00

mkdir -p logs output/metrics

# Load apptainer
module load apptainer/1.3.4-gcc-14.2.0-g7o5w4g

CONTAINER=container/dcgan_diffusion.sif

echo "Starting Metrics Computation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Run metrics computation
apptainer exec --nv $CONTAINER python3 python/compute_metrics.py \
    --device cuda \
    --output-dir output/metrics \
    --num-real-samples 1000

echo ""
echo "Metrics computation complete!"
echo "Results saved to: output/metrics/"
