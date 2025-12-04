# AS4: Image Generation Models

**Student Name:** [Your Name]  
**Student ID:** [Your ID]  
**Date:** December 2025

## Project Overview

This project implements and compares two image generation approaches:
1. **DCGAN** (Deep Convolutional Generative Adversarial Network)
2. **Diffusion Model** (Denoising Diffusion Probabilistic Model)

Both models are trained on [MNIST/Fashion-MNIST/CIFAR-10] dataset using GPU acceleration in an NVIDIA container on a Slurm cluster.

---

## Project Structure

```
as4/
├── data/                  # Datasets (auto-downloaded)
├── models/                # Saved model checkpoints
│   ├── dcgan_generator_mnist.pth
│   ├── dcgan_discriminator_mnist.pth
│   └── diffusion_model_mnist.pth
├── outputs/               # Generated images and results
│   ├── dcgan/
│   │   ├── samples/
│   │   └── training_losses.png
│   ├── diffusion/
│   │   ├── samples/
│   │   └── training_loss.png
│   └── comparison/
├── logs/                  # Slurm job logs
├── scripts/               # Python scripts
│   ├── dcgan.py
│   ├── diffusion_model.py
│   ├── generate_dcgan.py
│   └── compare_models.py
├── test_gpu.slurm         # GPU test job
├── train_dcgan.slurm      # DCGAN training job
├── train_diffusion.slurm  # Diffusion training job
├── README.md              # This file
└── REPORT.md              # Detailed technical report
```

---

## Environment Setup

### Container Information

- **Base Image:** `nvcr.io/nvidia/pytorch:25.09-py3`
- **PyTorch Version:** 2.5.0
- **CUDA Version:** 12.6
- **Python Version:** 3.10
- **Container Location:** `container/dcgan_diffusion.sif`

### Cluster Configuration

- **Cluster Name:** [Your cluster name]
- **GPU Type:** [e.g., NVIDIA A100, V100, etc.]
- **GPU Memory:** [e.g., 40GB]
- **Partition Used:** `gpu`

### Setup Instructions

1. **Pull Container:**
   ```bash
   cd /home1/michael2024/ML_Course/container
   apptainer pull docker://nvcr.io/nvidia/pytorch:25.09-py3
   ```

2. **Create Directory Structure:**
   ```bash
   cd ~/ML_Course/as4
   mkdir -p {data,models,outputs,logs,scripts}
   ```

3. **Test GPU Access:**
   ```bash
   sbatch test_gpu.slurm
   cat logs/test_gpu_*.out
   ```

### Additional Dependencies

Libraries installed beyond base container:
```bash
uv pip install diffusers transformers accelerate torchmetrics
```

---

## How to Run

### 1. Test Environment
```bash
sbatch test_gpu.slurm
# Verify output shows GPU is available
```

### 2. Train DCGAN
```bash
sbatch train_dcgan.slurm
# Training takes approximately [X] hours
# Monitor: tail -f logs/dcgan_*.out
```

### 3. Train Diffusion Model
```bash
sbatch train_diffusion.slurm
# Training takes approximately [X] hours
# Monitor: tail -f logs/diffusion_*.out
```

### 4. Generate Samples
```bash
# DCGAN samples
apptainer exec --nv /home1/michael2024/ML_Course/container/pytorch_25.09-py3.sif \
    python3 scripts/generate_dcgan.py

# Diffusion samples (included in training script)
```

### 5. Run Comparison
```bash
apptainer exec --nv /home1/michael2024/ML_Course/container/pytorch_25.09-py3.sif \
    python3 scripts/compare_models.py
```

---

## Results Summary

### DCGAN Results

- **Training Time:** [X] hours
- **Final Generator Loss:** [X]
- **Final Discriminator Loss:** [X]
- **Image Quality:** [Brief qualitative description]
- **Observations:** [Mode collapse? Stability? etc.]

**Sample Generated Images:**

[Include 2-3 sample images or reference to outputs/dcgan/samples/]

### Diffusion Model Results

- **Training Time:** [X] hours
- **Final Loss:** [X]
- **Inference Time (50 steps):** [X] seconds per batch
- **Image Quality:** [Brief qualitative description]
- **Observations:** [Quality improvement with steps? etc.]

**Sample Generated Images:**

[Include 2-3 sample images or reference to outputs/diffusion/samples/]

### Comparison

| Metric | DCGAN | Diffusion | Winner |
|--------|-------|-----------|--------|
| Training Time | [X] hours | [Y] hours | - |
| Generation Speed | [X] sec/100 | [Y] sec/100 | - |
| FID Score | [X] | [Y] | Lower better |
| Inception Score | [X] | [Y] | Higher better |
| Image Quality | [rating] | [rating] | - |
| Diversity | [High/Med/Low] | [High/Med/Low] | - |
| Training Stability | [rating] | [rating] | - |

**Key Findings:**
- [Bullet point 1]
- [Bullet point 2]
- [Bullet point 3]

---

## Challenges Encountered

### Challenge 1: [Description]
**Solution:** [How you solved it]

### Challenge 2: [Description]
**Solution:** [How you solved it]

---

## Key Learnings

1. **Transpose Convolutions:** [What you learned]
2. **GAN Training Dynamics:** [What you learned]
3. **Diffusion Process:** [What you learned]
4. **GPU Computing:** [What you learned]

---

## Reproducibility

To reproduce these results:

1. Use the exact container specified above
2. Follow setup instructions
3. Submit jobs in order: test → DCGAN → diffusion
4. All random seeds are set in scripts for reproducibility
5. Expected total compute time: [X] GPU-hours

---

## Files Submitted

- [ ] All Python scripts in `scripts/` directory
- [ ] All Slurm job files
- [ ] Trained model checkpoints
- [ ] Sample outputs and visualizations
- [ ] This README.md
- [ ] REPORT.md with detailed analysis
- [ ] Training logs

---

## References

1. Radford, A., et al. (2016). Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. ICLR.
2. Ho, J., et al. (2020). Denoising Diffusion Probabilistic Models. NeurIPS.
3. [Add other references you used]

---

## Contact

**Student:** [Your Name]  
**Email:** [Your Email]  
**Course:** [Course Code and Name]  
**Semester:** Fall 2025

---

## Acknowledgments

- NVIDIA NGC for container images
- [Cluster name] for computational resources
- [Any other acknowledgments]
