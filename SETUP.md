# EdgeWisePersona Project Setup

## Environment Setup

This project uses a Python virtual environment for dependency management and portability.

### Initial Setup (First Time)

1. **Activate the virtual environment:**

```bash
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

2. **Install dependencies:**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. **Verify installation:**

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
```

### Daily Usage

**Activate the environment** before working:

```bash
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows
```

**Deactivate** when done:

```bash
deactivate
```

### Transferring to Another Device

1. **On the source device**, commit your code:
```bash
git add .
git commit -m "Update project"
git push
```

2. **On the new device**, clone and set up:
```bash
git clone <repo-url>
cd adaptcommand
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Adding New Dependencies

If you install new packages:

```bash
pip install <package-name>
pip freeze > requirements.txt  # Update requirements
```

## Project Structure

```
adaptcommand/
├── data/
│   ├── raw/              # Raw EdgeWisePersona data
│   ├── cleaned/          # Cleaned dialogues
│   ├── splits/           # Train/val/test splits
│   ├── profiles/         # User embeddings/profiles
│   └── persona_pred/     # Persona prediction data
├── models/
│   ├── prefix_all/       # Global prefix models
│   ├── lora_all/         # Global LoRA models
│   ├── prefix_per_user/  # Per-user prefix models
│   └── lora_per_user/    # Per-user LoRA models
├── results/
│   ├── baseline/         # Baseline results
│   ├── prefix_all/       # Global prefix results
│   ├── lora_all/         # Global LoRA results
│   ├── prefix_per_user/  # Per-user prefix results
│   ├── lora_per_user/    # Per-user LoRA results
│   ├── persona_pred/     # Persona prediction results
│   ├── stats/            # Statistical analyses
│   ├── ablations/        # Ablation study results
│   └── figures/          # All plots and visualizations
├── scripts/              # Python scripts for each phase
├── configs/              # Configuration files
├── design.md             # Project design document
├── plan.md               # Implementation plan
└── requirements.txt      # Python dependencies
```

## Next Steps

See `plan.md` for the full implementation roadmap. Start with Phase A (Data Readiness).
