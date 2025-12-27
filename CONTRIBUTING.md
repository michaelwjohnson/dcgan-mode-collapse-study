# Contributing to DCGAN-Diffusion Comparison

Thank you for your interest in contributing to this project! This research project compares DCGANs and diffusion models for MNIST generation.

## How to Contribute

### Reporting Issues

If you find bugs or have suggestions for improvements:

1. Check if the issue already exists in the GitHub Issues
2. If not, create a new issue with:
   - Clear, descriptive title
   - Detailed description of the problem or suggestion
   - Steps to reproduce (for bugs)
   - Expected vs. actual behavior
   - System information (OS, GPU, CUDA version, etc.)

### Suggesting Enhancements

Enhancement suggestions are welcome! Consider:

- **New model architectures**: StyleGAN, Progressive GAN, etc.
- **Additional metrics**: FID, IS, Precision/Recall
- **Datasets**: CIFAR-10, CelebA, ImageNet
- **Training techniques**: Spectral normalization, progressive growing
- **Visualization improvements**: t-SNE plots, attention maps

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following the code style below
4. Test your changes thoroughly
5. Commit with clear messages (`git commit -m 'Add amazing feature'`)
6. Push to your branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request with:
   - Clear description of changes
   - Motivation and context
   - Any relevant issue numbers
   - Test results or screenshots

### Code Style

**Python:**
- Follow PEP 8 style guide
- Use meaningful variable names
- Add docstrings for functions and classes
- Keep functions focused and modular
- Add type hints where appropriate

**Example:**
```python
def train_dcgan(
    dataloader: DataLoader,
    generator: nn.Module,
    discriminator: nn.Module,
    epochs: int = 50,
    device: str = "cuda"
) -> Dict[str, List[float]]:
    """
    Train DCGAN model on provided data.
    
    Args:
        dataloader: PyTorch DataLoader with training data
        generator: Generator network
        discriminator: Discriminator network
        epochs: Number of training epochs
        device: Training device ('cuda' or 'cpu')
        
    Returns:
        Dictionary containing training losses
    """
    # Implementation
    pass
```

**SLURM Scripts:**
- Use descriptive job names
- Comment resource requirements
- Include module loads
- Set appropriate time limits

### Testing

Before submitting:

1. **Smoke test**: Verify the code runs without errors
2. **Visual inspection**: Check generated samples look reasonable
3. **Metrics**: Ensure losses converge appropriately
4. **Reproducibility**: Test with fixed random seeds

### Commit Message Guidelines

Use clear, descriptive commit messages:

```
Add spectral normalization to discriminator

- Implement spectral normalization layer
- Apply to all discriminator conv layers
- Update training loop to handle new architecture
- Add configuration parameter for enabling/disabling
```

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dcgan-diffusion-comparison.git
cd dcgan-diffusion-comparison
```

2. Build the container:
```bash
apptainer build container/dcgan_diffusion.sif container/apptainer.def
```

3. Test your changes:
```bash
sbatch slurm/test_gpu.slurm
```

## Research Reproducibility

When adding new experiments:

1. **Fix random seeds**: Use consistent seeds across runs
2. **Document hyperparameters**: Add to config files or README
3. **Save checkpoints**: Include model saving logic
4. **Log metrics**: Use wandb, tensorboard, or CSV logging
5. **Version control**: Commit code before running experiments

## Questions?

Feel free to open an issue for any questions about contributing!

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers and beginners
- Focus on the research and code quality
- Provide helpful feedback in reviews

---

Thank you for contributing to generative modeling research! 🚀
