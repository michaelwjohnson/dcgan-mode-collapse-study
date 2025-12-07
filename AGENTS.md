## PR instructions

- Always run all experiments and check outputs before committing.
- Make sure code is documented and type-hinted.
- Add or update tests for any code changes.

## Agent usage

Add AGENTS.md at the project root. Coding agents and teammates should:

## Academic writing style guide for agents

When editing `main.tex` or any academic writing in this project, always follow these principles:

1. **Be clear**

- Use short, direct sentences.
- Avoid overly formal or obscure words.
- Present points simply and unambiguously.

2. **Be concise**

- Remove unnecessary words and filler phrases.
- Avoid nominalisations unless needed for clarity.
- Do not state the obvious if context is clear.

3. **Avoid personal language**

- Do not use personal pronouns (I, we, my, our) unless required for reflection.
- Focus on objective, evidence-based statements.

4. **Avoid informal language**

- Use neutral, academic vocabulary.
- Avoid contractions and phrasal verbs.

5. **Make claims using cautious language**

- Use tentative language for weak evidence (may, could, might, possible explanation).
- Use neutral or confident language only when strongly supported by evidence.
- Avoid words like "prove" or "always".

6. **Focus on what is important**

- Use active sentences for clarity, unless passive is more appropriate for focus.
- Use verb tenses appropriately (past for completed work, present for general concepts).

Refer to the full academic writing style guide above for examples and further details.

7. **Structure your writing well**
   
- Use clear headings and subheadings.
- Organize paragraphs logically with topic sentences.
- Use lists or bullet points for clarity when appropriate.
- Ensure smooth transitions between sections and ideas.

8. **Avoid AI disclosure statements**
- Do not include statements about AI assistance in the writing.
- Focus on the content and quality of the writing itself.
- Ensure the writing meets academic standards without referencing AI tools.

9. **Avoid plagiarism**
- Always cite sources for ideas, data, or direct quotes.
- Paraphrase information in your own words while maintaining original meaning.
- Use quotation marks for direct quotes and provide proper citations.

10. **Avoid AI wrtiting style**
- Do not use overly complex or repetitive phrases.
- Maintain a formal and academic tone without sounding mechanical.
- Ensure the writing flows naturally and is engaging to read.
- Focus on clarity and precision rather than trying to sound "academic".
- Avoid generic statements that lack specificity or depth.
- Use varied sentence structures to enhance readability.
- Incorporate critical analysis and original thought rather than relying on clichés or common knowledge.
- Do not use the em dash (—) excessively; prefer commas or parentheses for clarity.

---

## Project overview

This project implements and compares Deep Convolutional Generative Adversarial Networks (DCGANs) for MNIST digit generation as part of Assignment 4 for ECE5570 - Machine Learning at Scale.

### Key Components
- **Training scripts**: `scripts/train_dcgan.py` - Main DCGAN training with configurable architecture
- **SLURM scripts**: `slurm/train_dcgan_baseline.slurm` and `slurm/train_dcgan_modified.slurm` - HPC job submissions
- **Containers**: `container/dcgan_diffusion.sif` (Apptainer) - Reproducible environment
- **Report**: `documents/report/main.tex` - IEEE-format conference paper
- **Outputs**: `output/dcgan_baseline/` and `output/dcgan_modified/` - Models, samples, metrics, losses

### Goals
- Compare architectural choices and their impact on GAN training
- Identify mode collapse and training instability
- Generate quantitative metrics and visual examples for analysis
- Document findings in academic report format

---

## Build and test commands

### Training
```bash
# Local training (if GPU available)
python scripts/train_dcgan.py --dataset mnist --epochs 50 --output-dir output/test

# Submit to HPC cluster
sbatch slurm/train_dcgan_baseline.slurm
sbatch slurm/train_dcgan_modified.slurm

# Check job status
squeue --me

# View logs
tail -f logs/dcgan_baseline_*.out
tail -f logs/dcgan_modified_*.out
```

### Generate report
```bash
cd documents/report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Compute metrics
```bash
# If metrics computation script exists
sbatch slurm/compute_metrics.slurm
```

---

## Code style guidelines

- **Python**: PEP 8 compliant, use type hints where appropriate
- **Documentation**: Docstrings for functions and classes
- **Reproducibility**: Always set random seeds (SEED=77) and deterministic settings
- **Naming**: Use descriptive variable names (e.g., `latent_dim`, `gen_depth`)
- **Modularity**: Keep training logic separate from model definitions

### Key Code Patterns
```python
# Set seeds for reproducibility
SEED = 77
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Save metrics in structured format (JSON)
metrics = {
    "configuration": {...},
    "training_performance": {...},
    "loss_metrics": {...}
}
```

---

## Testing instructions

### Verify training script
```bash
# Quick test run (1 epoch)
python scripts/train_dcgan.py --epochs 1 --output-dir output/test

# Check outputs
ls -la output/test/samples/
ls -la output/test/models/
cat output/test/metrics.json
```

### Validate SLURM scripts
```bash
# Dry run (syntax check)
sbatch --test-only slurm/train_dcgan_baseline.slurm

# Check resource allocation
scontrol show job <JOB_ID>
```

### Verify container
```bash
# Test container execution
apptainer exec --nv container/dcgan_diffusion.sif python --version
apptainer exec --nv container/dcgan_diffusion.sif python -c "import torch; print(torch.cuda.is_available())"
```

---

## HPC and containerization

### SLURM configuration
- **Partition**: `gpu1`
- **Resources**: 1 GPU, 8 CPUs, 32GB RAM
- **Time limit**: 4 hours
- **Module**: `apptainer/1.3.4-gcc-14.2.0-g7o5w4g`

### Container usage
```bash
# Load module
module load apptainer/1.3.4-gcc-14.2.0-g7o5w4g

# Run command in container
apptainer exec --nv container/dcgan_diffusion.sif <command>

# Interactive shell
apptainer shell --nv container/dcgan_diffusion.sif
```

---

## Output structure

Each experiment produces:
- `samples/` - Generated images per epoch (PNG)
- `models/` - Trained generator and discriminator (PTH)
- `checkpoints/` - Epoch checkpoints (every 10 epochs)
- `losses.csv` - Per-iteration losses (CSV)
- `metrics.json` - Structured metrics (JSON)
- `training_losses.png` - Loss curve visualization (PNG)

---

## Common issues

### Import errors in Python scripts
- Ensure you're using the correct Python environment or container
- Check that all dependencies are installed

### SLURM job failures
- Check error logs in `logs/` directory
- Verify GPU availability: `sinfo -p gpu1`
- Ensure container file exists: `ls -lh container/dcgan_diffusion.sif`

### Out of memory errors
- Reduce `--batch-size` parameter
- Reduce network depth (`--gen-depth`, `--disc-depth`)

### Mode collapse
- Adjust hyperparameters (latent dimension, learning rate, dropout)
- Monitor sample quality and loss curves during training
- Refer to GAN training best practices in course notes


### Academic writing style guide for agents

When editing `main.tex` or any academic writing in this project, always follow these principles:

1. **Be clear**

- Use short, direct sentences.
- Avoid overly formal or obscure words.
- Present points simply and unambiguously.

2. **Be concise**

- Remove unnecessary words and filler phrases.
- Avoid nominalisations unless needed for clarity.
- Do not state the obvious if context is clear.

3. **Avoid personal language**

- Do not use personal pronouns (I, we, my, our) unless required for reflection.
- Focus on objective, evidence-based statements.

4. **Avoid informal language**

- Use neutral, academic vocabulary.
- Avoid contractions and phrasal verbs.

5. **Make claims using cautious language**

- Use tentative language for weak evidence (may, could, might, possible explanation).
- Use neutral or confident language only when strongly supported by evidence.
- Avoid words like "prove" or "always".

6. **Focus on what is important**

- Use active sentences for clarity, unless passive is more appropriate for focus.
- Use verb tenses appropriately (past for completed work, present for general concepts).

Refer to the full academic writing style guide above for examples and further details.

7. **Structure your writing well**
   
- Use clear headings and subheadings.
- Organize paragraphs logically with topic sentences.
- Use lists or bullet points for clarity when appropriate.
- Ensure smooth transitions between sections and ideas.

8. **Avoid AI disclosure statements**
- Do not include statements about AI assistance in the writing.
- Focus on the content and quality of the writing itself.
- Ensure the writing meets academic standards without referencing AI tools.

9. **Avoid plagiarism**
- Always cite sources for ideas, data, or direct quotes.
- Paraphrase information in your own words while maintaining original meaning.
- Use quotation marks for direct quotes and provide proper citations.

10. **Avoid AI wrtiting style**
- Do not use overly complex or repetitive phrases.
- Maintain a formal and academic tone without sounding mechanical.
- Ensure the writing flows naturally and is engaging to read.
- Focus on clarity and precision rather than trying to sound "academic".
- Avoid generic statements that lack specificity or depth.
- Use varied sentence structures to enhance readability.
- Incorporate critical analysis and original thought rather than relying on clichés or common knowledge.
- Do not use the em dash (—) excessively; prefer commas or parentheses for clarity.