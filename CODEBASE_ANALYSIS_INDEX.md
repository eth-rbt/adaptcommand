# AdaptCommand Codebase Analysis - Complete Index

Generated: November 9, 2025

## Overview

This directory now contains comprehensive documentation for implementing LoRA (Low-Rank Adaptation) training for per-user personalized models in the AdaptCommand project.

## Documentation Files

### 1. **EXPLORATION_SUMMARY.txt** (18 KB)
**Start here for a complete overview**

A detailed walkthrough of the entire codebase containing:
- Scripts directory analysis (8 key scripts explained)
- Data format and structure specifications
- Model architecture and configuration details
- Evaluation infrastructure overview
- Training utilities and code patterns
- Key insights for LoRA implementation
- Complete dependency list
- Recommended next steps with checklist
- File reference guide

**Best for**: Getting the full picture of what exists and what needs to be built

---

### 2. **lora_training_guide.md** (20 KB)
**Detailed implementation guide**

A comprehensive 10-section guide covering:
- Executive summary
- Codebase structure overview (detailed)
- Data format specifications with examples
- Base model information and loading patterns
- Complete analysis of existing training infrastructure
- Code patterns and examples for:
  - Data loading
  - Prompt formatting
  - Model generation
- Key insights for LoRA implementation
- Recommended LoRA hyperparameters
- Implementation roadmap with phases
- File locations summary

**Best for**: Understanding how to implement LoRA training and what patterns to follow

---

### 3. **quick_reference.md** (4.2 KB)
**Quick code snippets and reference**

A condensed reference containing:
- Dataset overview (quick stats)
- Data loading code (copy-paste ready)
- Base model information
- Evaluation metrics summary
- LoRA setup code with PEFT
- Training loop template
- Inference pattern with LoRA
- File paths table

**Best for**: Quick lookups while coding, copy-paste snippets

---

## Quick Navigation

### If you want to...

**Understand what data exists**
→ Read: EXPLORATION_SUMMARY.txt Section 2, lora_training_guide.md Section 2

**See code patterns**
→ Read: quick_reference.md, lora_training_guide.md Section 6

**Learn about the baseline evaluation**
→ Read: EXPLORATION_SUMMARY.txt Section 4, lora_training_guide.md Section 4

**Get started coding LoRA**
→ Read: quick_reference.md, then lora_training_guide.md Section 5

**Find file paths**
→ Read: quick_reference.md "File Paths Reference", EXPLORATION_SUMMARY.txt Section 9

**Understand the training infrastructure**
→ Read: EXPLORATION_SUMMARY.txt Section 5, lora_training_guide.md Section 4

**See recommended hyperparameters**
→ Read: lora_training_guide.md Section 8, EXPLORATION_SUMMARY.txt Section 6

---

## Key Facts at a Glance

### Dataset
- **Total dialogues**: 10,000
- **Total personas**: 200
- **Per-persona size**: ~50 dialogues
- **Train/val/test split**: 60% / 20% / 20% per persona
- **Format**: JSONL (one dialogue per line)
- **Location**: `data/cleaned/dialogs_clean.jsonl`

### Model
- **Name**: Qwen/Qwen2.5-0.5B-Instruct
- **Type**: CausalLM (HuggingFace)
- **Size**: 0.5B parameters
- **Config**: `configs/baseline_v1.0.json`

### Evaluation Metrics
- ROUGE (1, 2, L)
- Embedding similarity (via sentence-transformers)
- Smart home action accuracy (device/parameter level)
- Length statistics

### LoRA Implementation
- **Rank**: 8 (small due to small per-user dataset)
- **Target modules**: q_proj, v_proj
- **Learning rate**: 5e-4
- **Epochs**: 5
- **Batch size**: 4
- **Storage per persona**: ~5MB
- **Total for 200 personas**: ~1GB

---

## Files and Directories

### Data
```
data/
  cleaned/dialogs_clean.jsonl        ← 10,000 dialogues
  splits/edgesplits.json             ← Train/val/test indices per persona
  raw/                               ← Original source files
```

### Code
```
scripts/
  run_baseline_benchmark.py          ← Main baseline evaluation
  action_metrics.py                  ← Smart home metric extraction
  create_splits.py                   ← Data split creation
  [YOUR NEW FILES]:
  train_lora_per_user.py            ← Train LoRA per persona (to create)
  eval_lora.py                      ← Evaluate LoRA (to create)
```

### Configuration
```
configs/
  baseline_v1.0.json                 ← Existing baseline config
  lora_training.json                 ← Training config (to create)
```

### Models
```
models/
  lora_per_user/
    persona_000/                     ← Adapter storage (to create)
    persona_001/
    ...
```

### Results
```
results/
  baseline/                          ← Baseline results (existing)
  lora/                             ← LoRA results (to create)
```

---

## Implementation Checklist

### Phase 1: Setup & Debug
- [ ] Read all three documentation files
- [ ] Understand data format and structure
- [ ] Review baseline script and metrics
- [ ] Create `scripts/train_lora_per_user.py`
- [ ] Create `scripts/eval_lora.py`
- [ ] Create `configs/lora_training.json`
- [ ] Test on persona_000 only

### Phase 2: Validation
- [ ] Verify LoRA adapter saves correctly
- [ ] Verify adapter loads and inference works
- [ ] Compare metrics vs baseline for persona_000
- [ ] Check no data leakage from test set

### Phase 3: Scale
- [ ] Loop training over all 200 personas
- [ ] Save each adapter with progress tracking
- [ ] Run evaluation for all personas
- [ ] Aggregate results

### Phase 4: Analysis
- [ ] Create analysis script
- [ ] Compare improvements by persona
- [ ] Identify which personas benefit most
- [ ] Create visualizations
- [ ] Write findings

---

## Code Snippets Quick Reference

### Load persona's training data
```python
import json

with open("data/cleaned/dialogs_clean.jsonl") as f:
    dialogues = [json.loads(line) for line in f]

with open("data/splits/edgesplits.json") as f:
    splits = json.load(f)

persona_id = "persona_000"
train_indices = splits[persona_id]["train"]
train_dialogues = [dialogues[i] for i in train_indices]
```

### Initialize LoRA
```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = get_peft_model(model, lora_config)
```

### Train with HF Trainer
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir=f"models/lora_per_user/{persona_id}",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    learning_rate=5e-4,
    weight_decay=0.01,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
```

### Load and use LoRA adapter
```python
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    "models/lora_per_user/persona_000"
)

# Use normally - adapter automatically applied
inputs = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(inputs)
```

---

## Key Insights

### Critical Success Factors
1. **Small per-user datasets** (~30 examples) → Use r=8, strong regularization
2. **Per-user adapters** → Independent training, flexible inference
3. **Evaluation metrics** → Reuse existing baseline metrics
4. **Data integrity** → Strict persona-level splits, no leakage

### Common Pitfalls to Avoid
1. Using too high LoRA rank for small datasets
2. Not using early stopping
3. Insufficient regularization (weight decay)
4. Mixing test data with training
5. Training for too many epochs

### Recommended Hyperparameters
- LoRA rank: 8 (not 16 or 32)
- Learning rate: 5e-4
- Batch size: 4
- Epochs: 5
- Weight decay: 0.01
- Dropout: 0.05

---

## Dependencies Already Installed
- torch >= 2.0.0
- transformers >= 4.35.0
- **peft >= 0.7.0** ← LoRA support
- accelerate >= 0.25.0
- datasets >= 2.14.0
- sentence-transformers >= 2.2.0
- rouge-score >= 0.1.2
- wandb >= 0.15.0
- tensorboard >= 2.13.0

No additional installs needed!

---

## Resources in Repository

- `design.md` - High-level project design
- `plan.md` - Implementation roadmap (Phases A-D)
- `QUICKSTART.md` - Quick start guide for baseline
- `requirements.txt` - Python dependencies

---

## Timeline Estimate

- **Implementation**: 1-2 days
- **Single persona debug**: 30 minutes
- **Training all 200 personas on GPU**: 1-2 hours
- **Analysis**: 1-2 hours
- **Total**: ~3-5 days end-to-end

---

## Next Steps

1. **Read the summaries** (start with EXPLORATION_SUMMARY.txt)
2. **Review the code patterns** (look at quick_reference.md)
3. **Study the baseline script** (scripts/run_baseline_benchmark.py)
4. **Create train_lora_per_user.py** (use patterns from baseline)
5. **Create eval_lora.py** (similar structure to baseline)
6. **Debug on one persona** (persona_000)
7. **Scale to all personas** (with progress tracking)
8. **Analyze results** (compare vs baseline)

---

## Questions?

All three documentation files contain detailed explanations, code examples, and rationale. They're designed to be self-contained but also complementary:

- **EXPLORATION_SUMMARY.txt**: Comprehensive walkthrough
- **lora_training_guide.md**: Implementation deep-dive
- **quick_reference.md**: Quick code lookups

Start with what fits your learning style!

---

Generated with comprehensive codebase analysis
November 9, 2025
