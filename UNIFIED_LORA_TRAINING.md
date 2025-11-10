# Unified LoRA Training Guide

This guide explains how to train a single LoRA adapter on all training data from all 200 personas.

## Overview

The `train_unified_lora.py` script trains **one unified LoRA model** on the entire training set:
- **200 personas** × **30 training dialogues** = **6,000 total dialogues**
- Multiple assistant turns per dialogue = **~40,000+ training examples**

This is different from `train_lora_per_user.py` which trains separate adapters for each individual persona.

## Quick Start

### Option 1: Using the run script
```bash
bash run_unified_lora_training.sh
```

### Option 2: Direct Python execution
```bash
python scripts/train_unified_lora.py \
    --config configs/lora_training.json \
    --output_dir models/lora_unified
```

## Command Line Arguments

### `--config` (default: `configs/lora_training.json`)
Path to the training configuration file containing:
- Model name and LoRA parameters
- Training hyperparameters
- Data paths

### `--output_dir` (default: `models/lora_unified`)
Directory where the trained adapter will be saved. The script will create:
- `adapter_model.safetensors` - The LoRA weights
- `adapter_config.json` - LoRA configuration
- `training_config.json` - Full training configuration
- `test_metrics.json` - Test set evaluation results
- `val_metrics.json` - Validation set results (if using validation)
- `trainer_state.json` - Training state
- `logs/` - Training logs

### `--no_persona`
Don't include persona descriptions in the system prompts.

**Default behavior (persona included):**
```
System: You are a helpful and personalized smart home assistant.

User Profile: Ethan is a reserved librarian with a penchant for mystery novels...
```

**With --no_persona flag:**
```
System: You are a helpful and personalized smart home assistant.
```

### `--no_val`
Skip validation set evaluation during training (faster, but no early stopping).

## Training Configuration

The default configuration in `configs/lora_training.json`:

```json
{
  "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
  "lora_config": {
    "r": 8,                    // LoRA rank
    "lora_alpha": 16,          // LoRA alpha
    "target_modules": ["q_proj", "v_proj"],  // Which layers to adapt
    "lora_dropout": 0.05,
    "bias": "none"
  },
  "training_args": {
    "num_train_epochs": 5,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,  // Effective batch size = 4
    "learning_rate": 5e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "save_strategy": "epoch",
    "evaluation_strategy": "epoch"
  },
  "max_length": 512,
  "seed": 42
}
```

## Training Process

1. **Load base model**: Qwen/Qwen2.5-0.5B-Instruct
2. **Apply LoRA**: Add low-rank adapters to attention layers
3. **Load training data**: All 6,000 training dialogues from 200 personas
4. **Create examples**: Extract all assistant turns as training examples (~40K+)
5. **Train**: 5 epochs with gradient accumulation
6. **Evaluate on validation**: Embedding similarity, device/param accuracy
7. **Evaluate on test**: Final metrics on 2,000 test dialogues

## Expected Training Time

- **CPU**: ~12-24 hours (not recommended)
- **MPS (Apple Silicon)**: ~4-8 hours
- **CUDA GPU**: ~1-3 hours (depending on GPU)

With 6,000 dialogues and ~40,000+ training examples at batch size 1 with accumulation=4:
- ~10,000 steps per epoch
- 5 epochs = ~50,000 total steps
- At ~1-2 steps/second on MPS: ~6-14 hours

## Evaluation Metrics

The script evaluates the model on:

### Embedding Similarity
- Semantic similarity between predicted and reference responses
- Range: 0.0 to 1.0 (higher is better)

### Device-Level Metrics
- **Device Precision**: Are activated devices correct?
- **Device Recall**: Are all needed devices activated?

### Parameter-Level Metrics
- **Param Precision/Recall**: Overall parameter accuracy
- **Numerical Precision/Recall**: Exact match for numerical values (temperature, volume, etc.)
- **Param F1**: Harmonic mean of precision and recall

### Example Output
```
Test Metrics:
  embedding_similarity          : 0.6824
  device_precision              : 0.8123
  device_recall                 : 0.8045
  param_precision               : 0.7891
  param_recall                  : 0.7654
  param_f1                      : 0.7771
  numerical_precision           : 0.2456
  numerical_recall              : 0.2389
  avg_pred_length               : 18.23
  avg_ref_length                : 14.56
  length_ratio                  : 1.25
```

## Comparison to Baseline

The baseline (no fine-tuning) achieved:
- Embedding similarity: **0.638**
- Device precision: **0.785**
- Numerical precision: **0.202**

Fine-tuning should improve these metrics, especially:
- Better device and parameter selection
- More accurate numerical values
- Improved persona-specific responses

## Using the Trained Model

After training, load and use the model:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "models/lora_unified")

# Generate
messages = [
    {"role": "system", "content": "You are a helpful smart home assistant."},
    {"role": "user", "content": "Could you turn on the lights at 60% brightness?"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Monitoring Training

Training logs are saved to `{output_dir}/logs/`. You can monitor:
- Loss curves
- Learning rate schedule
- Validation metrics per epoch

Check the logs:
```bash
cat models/lora_unified/trainer_state.json
```

## Troubleshooting

### Out of Memory (OOM)
Reduce batch size in config:
```json
"per_device_train_batch_size": 1,
"gradient_accumulation_steps": 8
```

### Training too slow
- Use CUDA GPU if available
- Reduce number of epochs
- Enable fp16 (CUDA only): `"fp16": true`

### Poor performance
- Increase number of epochs
- Adjust learning rate (try 1e-4 or 1e-3)
- Increase LoRA rank: `"r": 16`
- Add more target modules: `["q_proj", "v_proj", "k_proj", "o_proj"]`

## Next Steps

After training:
1. Compare test metrics to baseline results
2. Analyze per-persona performance
3. Try training with `--no_persona` flag for comparison
4. Experiment with different LoRA configurations
5. Test on specific challenging personas

## Directory Structure After Training

```
models/lora_unified/
├── adapter_config.json          # LoRA configuration
├── adapter_model.safetensors    # Trained LoRA weights
├── training_config.json         # Full training config
├── test_metrics.json           # Test evaluation results
├── val_metrics.json            # Validation results
├── trainer_state.json          # Training state
├── tokenizer_config.json       # Tokenizer config
├── special_tokens_map.json     # Special tokens
└── logs/                       # Training logs
    └── events.out.tfevents.*
```
