# AS4: Image Generation Models

## Environment Setup

### Base Container Image and Version
- **Base Image:** nvcr.io/nvidia/pytorch:25.09-py3
- **PyTorch Version:** 2.5
- **CUDA Version:** 12.6
- **Python Version:** 3.10

### Custom Container Definition Location
- **Definition file:** `container/apptainer.def`
- **Built image:** `container/dcgan_diffusion.sif`

### How to Build the Custom Container
```bash
cd /home1/michael2024/ML_Course/as4
apptainer build container/dcgan_diffusion.sif container/apptainer.def
```

### How to Run Test Jobs
- Submit the GPU test job:
```bash
sbatch test_gpu.slurm
```
- Check job status:
```bash
squeue -u $USER
```
- View output:
```bash
cat logs/test_gpu_*.out
```

### Cluster-Specific Instructions
- Load Apptainer module before running jobs:
```bash
module load apptainer/1.3.4-gcc-14.2.0-g7o5w4g
```
- Use the correct GPU partition (e.g., `gpu1`) in your Slurm scripts.

---

✅ **Checkpoint:** You should now have a working GPU environment with your custom container built from `container/apptainer.def`.
