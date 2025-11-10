# Quick Reference: AdaptCommand LoRA Training Setup

## Dataset Overview
```
Total dialogues: 10,000
Total personas: 200
Per persona: ~50 dialogues
  - Train: 30 (60%)
  - Val: 10 (20%)
  - Test: 10 (20%)

Format: JSONL (each line = one dialogue)
Location: data/cleaned/dialogs_clean.jsonl
Splits: data/splits/edgesplits.json
```

## Data Loading Quick Code
```python
import json

# Load all dialogues
with open("data/cleaned/dialogs_clean.jsonl") as f:
    dialogues = [json.loads(line) for line in f]

# Load splits
with open("data/splits/edgesplits.json") as f:
    splits = json.load(f)

# Get persona's training dialogues
persona_id = "persona_000"
train_indices = splits[persona_id]["train"]
train_dialogues = [dialogues[i] for i in train_indices]

# Each dialogue has:
# - persona_id: str
# - session_id: int
# - character: str (persona description)
# - messages: list of {"role": "user"|"assistant", "text": str}
```

## Base Model
- **Name**: Qwen/Qwen2.5-0.5B-Instruct
- **Type**: CausalLM (HuggingFace)
- **Size**: 0.5B parameters
- **Tokenizer**: Has chat template

**Load it**:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
```

## Evaluation Metrics Available
1. **ROUGE**: rouge1, rouge2, rougeL F1 scores
2. **Embedding Similarity**: Cosine sim via sentence-transformers
3. **Action Accuracy** (smart home specific):
   - Device precision/recall/F1
   - Parameter precision/recall/F1
   - Numerical params: precision/recall/F1/MAE
   - Categorical params: precision/recall/F1

**Code to use**:
```python
from scripts.action_metrics import ActionExtractor, ActionMetrics

extractor = ActionExtractor()
pred_actions = extractor.extract_actions("Turn on the lights to 50%")
ref_actions = extractor.extract_actions("Set lights brightness to 50")
metrics = ActionMetrics.compare_actions(pred_actions, ref_actions)
```

## LoRA Setup (using PEFT)
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,                          # Rank (small due to small dataset)
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Load base model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Shows how many params are trainable
```

## Training Loop (using HF Trainer)
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir=f"models/lora_per_user/{persona_id}",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    learning_rate=5e-4,
    weight_decay=0.01,
    save_strategy="steps",
    save_steps=50,
    eval_strategy="steps",
    eval_steps=50,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
```

## Inference with LoRA
```python
from peft import AutoPeftModelForCausalLM

# Load model + adapter
model = AutoPeftModelForCausalLM.from_pretrained(
    "models/lora_per_user/persona_000"
)

# Generate as normal (adapter automatically applied)
inputs = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(inputs, max_length=256)
```

## File Paths Reference
| What | Path |
|------|------|
| Dialogues | `data/cleaned/dialogs_clean.jsonl` |
| Splits | `data/splits/edgesplits.json` |
| Config template | `configs/baseline_v1.0.json` |
| Baseline script | `scripts/run_baseline_benchmark.py` |
| Metrics | `scripts/action_metrics.py` |
| LoRA storage | `models/lora_per_user/{persona_id}/` |

## Next: Create These Scripts
1. `scripts/train_lora_per_user.py` - Training loop
2. `scripts/eval_lora.py` - Evaluation
3. `configs/lora_training.json` - Training config

## Key Requirements Already Installed
- torch >= 2.0.0
- transformers >= 4.35.0
- **peft >= 0.7.0** (LoRA support!)
- accelerate >= 0.25.0
- datasets >= 2.14.0
