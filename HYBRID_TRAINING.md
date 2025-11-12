# Hybrid LoRA Training Guide

This guide explains how to use the hybrid training approach where small persona-specific LoRA adapters are trained on top of the unified LoRA model.

## Architecture

```
Base Model (Qwen 0.5B)
    ↓
Unified LoRA (rank=16, frozen)
    ↓
Persona LoRA (rank=8, trainable) ← Train this per persona
```

**Benefits:**
- Leverages shared knowledge from unified model
- Only trains small persona adapter (~few MB)
- Prevents overfitting on small per-persona datasets
- Much faster training than from scratch
- Better performance expected than standalone per-persona models

## Prerequisites

1. **Unified model must be trained first:**
   ```bash
   python scripts/train_unified_lora.py
   ```

   This creates `models/lora_unified/` with the base unified adapter.

## Usage

### Train Single Persona

```bash
python scripts/train_persona_on_unified.py \
    --persona_id persona_001 \
    --unified_model models/lora_unified \
    --config configs/lora_training.json \
    --output_dir models/lora_hybrid/persona_001 \
    --persona_rank 8
```

**Arguments:**
- `--persona_id`: Which persona to train (e.g., persona_001)
- `--unified_model`: Path to unified LoRA model (default: models/lora_unified)
- `--config`: Training config file (default: configs/lora_training.json)
- `--output_dir`: Where to save persona adapter (default: models/lora_hybrid/{persona_id})
- `--persona_rank`: LoRA rank for persona adapter (default: 8, smaller = fewer parameters)
- `--no_val`: Skip validation during training

### Train All Personas

```bash
python scripts/train_all_hybrid.py \
    --unified_model models/lora_unified \
    --output_dir models/lora_hybrid \
    --persona_rank 8
```

**Arguments:**
- `--unified_model`: Path to unified LoRA model
- `--config`: Training config file
- `--output_dir`: Base directory for all persona adapters
- `--splits_file`: Path to splits file (default: data/splits/edgesplits.json)
- `--persona_rank`: LoRA rank for all persona adapters (default: 8)
- `--start_index`: Start from this persona index (for resuming)
- `--end_index`: End at this persona index (for partial runs)
- `--no_val`: Skip validation during training
- `--skip_existing`: Skip personas that already have trained models

**Example - Train subset:**
```bash
# Train first 10 personas
python scripts/train_all_hybrid.py --end_index 10

# Resume from persona 50 to 100
python scripts/train_all_hybrid.py --start_index 50 --end_index 100

# Skip already trained personas
python scripts/train_all_hybrid.py --skip_existing
```

## Training Configuration

The hybrid approach uses modified training settings optimized for fine-tuning:

- **Epochs**: 5 (vs 10 for standalone) - fewer needed since starting from pretrained
- **Learning rate**: 1e-4 (vs 3e-4) - lower for fine-tuning
- **LoRA rank**: 8 (vs 16) - smaller adapter to prevent overfitting
- **Dropout**: 0.2 (vs 0.1) - higher regularization for small datasets

You can adjust these in `scripts/train_persona_on_unified.py` if needed.

## Output Structure

```
models/lora_hybrid/
├── persona_000/
│   ├── adapter_config.json       # Persona adapter config
│   ├── adapter_model.bin          # Persona adapter weights (small, ~5MB)
│   ├── test_metrics.json          # Evaluation results
│   └── training_config.json       # Full training config
├── persona_001/
│   └── ...
└── persona_199/
    └── ...

results/hybrid/
└── hybrid_summary.json            # Aggregated results across all personas
```

## Using Trained Models

To use a hybrid model for inference:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# Load unified adapter
model = PeftModel.from_pretrained(
    model,
    "models/lora_unified",
    adapter_name="unified"
)

# Load persona adapter
model.load_adapter("models/lora_hybrid/persona_001", adapter_name="persona")

# Enable both adapters (they're active by default in most PEFT versions)
try:
    model.set_adapter(["unified", "persona"])
except:
    model.enable_adapters()  # Fallback for older versions

# Now generate with both adapters active
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
inputs = tokenizer("Your prompt here", return_tensors="pt")
outputs = model.generate(**inputs)
```

## Hyperparameter Tuning

You can experiment with different `persona_rank` values:

```bash
# Very small adapter (fewer params, more regularization)
python scripts/train_persona_on_unified.py --persona_id persona_001 --persona_rank 4

# Medium adapter (default)
python scripts/train_persona_on_unified.py --persona_id persona_001 --persona_rank 8

# Larger adapter (more capacity, risk of overfitting)
python scripts/train_persona_on_unified.py --persona_id persona_001 --persona_rank 16
```

**Rule of thumb:**
- rank=4: Very limited data (<20 examples)
- rank=8: Small data (20-50 examples) ← recommended starting point
- rank=16: More data (50+ examples)

## Evaluation

After training all personas, compare with baseline and standalone per-persona models:

```bash
python scripts/compare_results.py \
    --baseline results/baseline/baseline_results.json \
    --personalized results/hybrid/hybrid_summary.json \
    --output results/comparison_hybrid.json
```

## Expected Results

The hybrid approach should:
- Outperform standalone per-persona models (which overfit on small data)
- Potentially match or approach unified model performance
- Have much less variance across personas
- Train faster (5 epochs vs 10 epochs)

## Troubleshooting

**Error: "Unified model not found"**
→ Train unified model first: `python scripts/train_unified_lora.py`

**CUDA out of memory**
→ Reduce `per_device_train_batch_size` or `gradient_accumulation_steps` in the training script

**Poor performance**
→ Try different `persona_rank` values (4, 8, 16, 32)
→ Adjust learning rate or number of epochs
→ Check if unified model is performing well first

**Adapter conflict errors**
→ Make sure adapter names are unique ("unified" vs "persona")
→ Check PEFT library version: `pip install -U peft`
