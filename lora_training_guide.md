# LoRA Training Guide for AdaptCommand Per-User Models

## Executive Summary

This is a comprehensive guide to implementing LoRA (Low-Rank Adaptation) training for per-user personalized models in the AdaptCommand project. The codebase is well-structured with:
- 10,000 cleaned dialogues across 200 unique personas
- Pre-split train/val/test data per persona (60/20/20)
- Established baseline evaluation infrastructure
- PEFT library already in requirements (supports LoRA)
- Smart home task-specific metrics and evaluation

---

## 1. CODEBASE STRUCTURE OVERVIEW

### Root Directory Layout
```
/Users/ethrbt/code/adaptcommand/
├── data/                    # Data directory
│   ├── raw/                 # Original EdgeWisePersona dataset files
│   ├── cleaned/             # Processed dialogues
│   │   └── dialogs_clean.jsonl    (10,000 dialogues)
│   └── splits/              # Data splits
│       └── edgesplits.json  (train/val/test indices per persona)
├── models/                  # Model storage directories
│   ├── lora_all/           # Global LoRA adapter (for future)
│   ├── lora_per_user/      # Per-user LoRA adapters (for future)
│   ├── prefix_all/         # Global prefix adapter (for future)
│   └── prefix_per_user/    # Per-user prefix adapters (for future)
├── scripts/                 # Utility and evaluation scripts
│   ├── run_baseline_benchmark.py   (MAIN: Baseline evaluation)
│   ├── action_metrics.py           (Smart home action extraction)
│   ├── create_splits.py            (Data split creation)
│   ├── clean_data.py               (Data cleaning)
│   ├── inspect_jsonl.py            (Data inspection)
│   ├── inspect_data_structure.py   (Data analysis)
│   └── validate_data.py            (Data validation)
├── configs/                 # Configuration files
│   └── baseline_v1.0.json   (Model and evaluation config)
├── results/                 # Evaluation results directory
├── requirements.txt         # Python dependencies (PEFT included!)
├── design.md               # High-level project design
├── plan.md                 # Implementation roadmap
└── QUICKSTART.md           # Quick reference guide
```

---

## 2. DATA FORMAT & STRUCTURE

### Dialogue Format (JSONL)
Each line in `data/cleaned/dialogs_clean.jsonl` contains:

```json
{
  "persona_id": "persona_000",
  "session_id": 0,
  "character": "Ethan is a reserved librarian with a penchant for mystery novels...",
  "routines": [
    {
      "triggers": {
        "time_of_day": "evening",
        "day_of_week": "weekday",
        "sun_phase": "after_sunset",
        "weather": "rainy",
        "outdoor_temp": "cold"
      },
      "actions": {
        "tv": null,
        "ac": {"temperature": 22, "mode": "heat", "fan_speed": 1},
        "lights": {"brightness": 50, "color": "warm", "mode": "static"},
        "speaker": {"volume": 30, "equalizer": "balanced"},
        "security": null
      }
    }
    // ... more routines
  ],
  "messages": [
    {
      "role": "user",
      "text": "Could you turn on the lights? It's quite dim in here."
    },
    {
      "role": "assistant",
      "text": "Of course. What brightness level would you like for the lights?"
    },
    // ... more messages
  ]
}
```

### Key Data Characteristics
- **Total dialogues**: 10,000
- **Total unique personas**: 200
- **Avg dialogues per persona**: 50
- **Avg messages per dialogue**: ~14 (turns)
- **Split per persona**: 
  - Train: ~30 dialogues (60%)
  - Val: ~10 dialogues (20%)
  - Test: ~10 dialogues (20%)

### Data Split File (`data/splits/edgesplits.json`)
```json
{
  "persona_000": {
    "train": [0, 1, 2, ..., 29],      // First 60% (train indices into dialogs_clean.jsonl)
    "val": [30, 31, 32, ..., 39],     // Next 20%
    "test": [40, 41, 42, ..., 49]     // Final 20%
  },
  "persona_001": { ... },
  // ... 200 personas total
}
```

### Persona Characteristics
Each persona has:
- A **character description** (guides the assistant's style)
- **Routines** (smart home preferences under different conditions)
- **Multiple dialogue sessions** (interaction history)

---

## 3. BASE MODEL & CONFIGURATION

### Current Baseline Model
- **Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Architecture**: Causal Language Model (CausalLM)
- **Size**: 0.5B parameters (very small, for fast iteration)
- **Tokenizer**: Uses chat template via `apply_chat_template()`

### Configuration File (`configs/baseline_v1.0.json`)
```json
{
  "model": {
    "name": "Qwen/Qwen2.5-0.5B-Instruct",
    "alternatives": [
      "Qwen/Qwen2.5-1.5B-Instruct",
      "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
      "microsoft/phi-2"
    ]
  },
  "generation": {
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": true,
    "repetition_penalty": 1.1
  },
  "prompt": {
    "system_template": "You are a helpful smart home assistant. Help the user control their devices by understanding their preferences and the current context.",
    "format": "chat",
    "include_context": true,
    "max_context_messages": 5
  },
  "evaluation": {
    "metrics": ["embedding_similarity", "action_accuracy"],
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "compute_per_user": true
  },
  "data": {
    "dialogues_file": "data/cleaned/dialogs_clean.jsonl",
    "splits_file": "data/splits/edgesplits.json",
    "eval_split": "test",
    "max_examples": null
  },
  "device": "auto",
  "batch_size": 8,
  "seed": 42
}
```

### How to Load Model
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cuda"  # or "mps" for Mac, "cpu" for CPU
)
model.eval()
```

---

## 4. EXISTING TRAINING INFRASTRUCTURE & UTILITIES

### Baseline Evaluation Script (`scripts/run_baseline_benchmark.py`)

**What it does**:
1. Loads model and configuration
2. Loads test dialogues using persona-based splits
3. For each dialogue, runs **turn-by-turn prediction** with ground truth context
4. Generates model response
5. Computes multiple metrics (ROUGE, embedding similarity, action accuracy)
6. Aggregates results globally and per-persona
7. Saves results to JSON and CSV

**Key functions**:
- `load_data()`: Loads dialogues and filters by split
- `format_prompt()`: Formats dialogue into chat messages
- `generate_response()`: Runs model inference
- `compute_metrics()`: Calculates ROUGE, embedding similarity, action accuracy
- `compute_per_persona_metrics()`: Aggregates metrics by persona

**How to run baseline**:
```bash
python scripts/run_baseline_benchmark.py \
  --config configs/baseline_v1.0.json \
  --max_examples 50 \
  --output_dir results/baseline_quick
```

**Output files**:
- `baseline_results.json` - Global metrics
- `per_persona_results.json` - Per-persona metrics  
- `per_persona_summary.csv` - Readable summary table
- `sample_outputs.jsonl` - First 20 prediction examples

### Action Metrics (`scripts/action_metrics.py`)

**Purpose**: Extract and compare smart home device actions

**Key classes**:
- `ActionExtractor`: Parses text to extract device actions
- `ActionMetrics`: Compares predicted vs reference actions

**Supported devices**:
- TV (volume, brightness, input_source)
- AC (temperature, mode, fan_speed)
- Lights (brightness, color, mode)
- Speaker (volume, equalizer)
- Security (armed, alarm_volume)

**Metrics computed**:
- Device-level: precision, recall
- Parameter-level: precision, recall, F1
- Numerical parameters: precision, recall, F1, MAE
- Categorical parameters: precision, recall, F1

**Example usage**:
```python
from scripts.action_metrics import ActionExtractor, ActionMetrics

extractor = ActionExtractor()
pred_actions = extractor.extract_actions("Set the AC to 22 degrees in heat mode")
ref_actions = extractor.extract_actions("Set temperature to 22, mode heat")

metrics = ActionMetrics.compare_actions(pred_actions, ref_actions)
```

### Data Loading Pattern (from baseline script)

```python
def load_data(dialogues_file, splits_file, split_name, max_examples=None):
    """Load and filter dialogues by split."""
    dialogues = []
    with open(dialogues_file, "r") as f:
        for line in f:
            dialogues.append(json.loads(line))
    
    with open(splits_file, "r") as f:
        splits = json.load(f)
    
    split_dialogues = []
    for persona_splits in splits.values():
        for idx in persona_splits[split_name]:
            split_dialogues.append(dialogues[idx])
    
    if max_examples:
        split_dialogues = split_dialogues[:max_examples]
    
    return split_dialogues
```

### Prompt Formatting Pattern

```python
def format_prompt(dialogue, config, up_to_turn=-1):
    """Format dialogue into chat messages."""
    messages = dialogue["messages"]
    
    if up_to_turn == -1:
        up_to_turn = len(messages) - 1
    
    context_messages = messages[:up_to_turn]
    target_message = messages[up_to_turn]
    
    # Build prompt
    prompt_messages = []
    if config["prompt"]["system_template"]:
        prompt_messages.append({
            "role": "system",
            "content": config["prompt"]["system_template"]
        })
    
    # Add context (limit to max_context_messages)
    max_context = config["prompt"].get("max_context_messages", 5)
    start_idx = max(0, len(context_messages) - max_context * 2)
    
    for msg in context_messages[start_idx:]:
        prompt_messages.append({
            "role": msg["role"],
            "content": msg["text"]
        })
    
    return prompt_messages, target_message["text"]
```

### Generation Pattern

```python
def generate_response(model, tokenizer, messages, config, device):
    """Generate model response."""
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config["generation"]["max_new_tokens"],
            temperature=config["generation"]["temperature"],
            top_p=config["generation"]["top_p"],
            do_sample=config["generation"]["do_sample"],
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    
    return generated_text.strip()
```

### Evaluation Metrics

The baseline computes:

1. **ROUGE Metrics**:
   - ROUGE-1 F1 (unigram overlap)
   - ROUGE-2 F1 (bigram overlap)
   - ROUGE-L F1 (longest common subsequence)

2. **Embedding Similarity**:
   - Uses `sentence-transformers/all-MiniLM-L6-v2`
   - Cosine similarity between prediction and reference embeddings

3. **Action Accuracy** (smart-home specific):
   - Device detection (TP/FP/TN/FN)
   - Device precision, recall, F1
   - Parameter precision, recall, F1
   - Numerical parameter metrics (MAE)
   - Categorical parameter metrics

4. **Length Statistics**:
   - Average prediction length
   - Average reference length
   - Length ratio

---

## 5. KEY INSIGHTS FOR LoRA IMPLEMENTATION

### What You Need to Implement

1. **LoRA Adapter Integration**:
   - Use PEFT library (already in requirements!)
   - Initialize LoRA config (rank, alpha, target modules)
   - Attach to base model
   - Train only LoRA parameters while keeping base model frozen

2. **Per-User Training Loop**:
   - For each persona:
     - Load their training data from split indices
     - Create/load their LoRA adapter
     - Fine-tune on training set
     - Evaluate on validation set
     - Save adapter weights to `models/lora_per_user/<persona_id>/`

3. **Training Data Preparation**:
   - Convert dialogues to input/output pairs for language modeling
   - Each training sample = context messages + assistant response
   - Use HuggingFace Datasets for efficient loading

4. **Training Configuration**:
   - Learning rate: typically 2e-4 to 5e-4 for LoRA
   - Batch size: 4-8 (for small models)
   - Epochs: 3-5 (avoid overfitting with small per-user datasets)
   - Max sequence length: ~1024 tokens

5. **Inference with LoRA**:
   - Load base model
   - Load specific LoRA adapter using PEFT
   - Generate responses same as baseline
   - LoRA parameters automatically applied during forward pass

### Critical Considerations

1. **Small Per-User Datasets** (~30 training examples per persona):
   - High risk of overfitting
   - Consider regularization (weight decay, early stopping)
   - May want to use lower ranks (r=4-8 instead of r=16-32)

2. **Evaluation Strategy**:
   - Use baseline metrics (ROUGE, embedding similarity, action accuracy)
   - Compute improvement over baseline per-persona
   - Track which personas benefit most from personalization
   - Check if model memorizes vs. learns generalizable patterns

3. **Storage & Computation**:
   - LoRA adapters are small (typically 1-10MB per persona)
   - 200 personas × 5MB = 1GB total storage (very manageable)
   - Training time: minutes per persona on GPU

4. **Data Distribution**:
   - Ensure you're not leaking test data to training
   - Use strict persona-level splits (no data sharing)
   - Evaluate on held-out test set

---

## 6. CODE STRUCTURE & PATTERNS TO FOLLOW

### Example: Load Persona's Training Data

```python
import json
from pathlib import Path

def load_persona_training_data(persona_id, dialogues_file, splits_file):
    """Load training data for a specific persona."""
    
    # Load all dialogues
    dialogues = []
    with open(dialogues_file, "r") as f:
        for line in f:
            dialogues.append(json.loads(line))
    
    # Load splits
    with open(splits_file, "r") as f:
        splits = json.load(f)
    
    # Get training indices for this persona
    if persona_id not in splits:
        return []
    
    train_indices = splits[persona_id]["train"]
    
    # Filter dialogues by indices
    persona_train_dialogues = [
        dialogues[idx] for idx in train_indices 
        if dialogues[idx]["persona_id"] == persona_id
    ]
    
    return persona_train_dialogues
```

### Example: Convert Dialogue to Training Samples

```python
def dialogue_to_training_samples(dialogue, tokenizer, max_context=5):
    """Convert a dialogue to language modeling samples."""
    
    messages = dialogue["messages"]
    samples = []
    
    # Find all assistant message indices
    assistant_indices = [i for i, msg in enumerate(messages) 
                        if msg["role"] == "assistant"]
    
    # Skip first 2, train on rest (like baseline)
    for assist_idx in assistant_indices[2:]:
        # Build context (up to but not including target)
        context_start = max(0, assist_idx - max_context * 2)
        context_messages = []
        
        # Add system prompt
        context_messages.append({
            "role": "system",
            "content": "You are a helpful smart home assistant..."
        })
        
        # Add dialogue history
        for msg in messages[context_start:assist_idx]:
            context_messages.append({
                "role": msg["role"],
                "content": msg["text"]
            })
        
        # Target response
        target_text = messages[assist_idx]["text"]
        
        samples.append({
            "messages": context_messages,
            "target": target_text
        })
    
    return samples
```

### Example: Tokenize for Training

```python
def tokenize_chat_sample(sample, tokenizer, max_length=1024):
    """Tokenize a chat sample for language modeling."""
    
    # Format using chat template
    prompt = tokenizer.apply_chat_template(
        sample["messages"],
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Append target and tokenize full sequence
    full_text = prompt + sample["target"] + tokenizer.eos_token
    
    encodings = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # For causal LM, labels = input_ids (auto-shifted in HF Trainer)
    encodings["labels"] = encodings["input_ids"].clone()
    
    return encodings
```

---

## 7. REQUIRED DEPENDENCIES

### Already in requirements.txt:
```
torch>=2.0.0
transformers>=4.35.0
peft>=0.7.0                 # <-- LoRA support!
accelerate>=0.25.0          # <-- Distributed training support
sentence-transformers>=2.2.0
datasets>=2.14.0            # <-- For efficient data loading
tqdm>=4.65.0
pandas>=2.0.0
numpy>=1.24.0
```

### Additional considerations:
- No additional dependencies needed for basic LoRA
- Consider adding `wandb>=0.15.0` for experiment tracking (already in requirements)
- `tensorboard>=2.13.0` for training monitoring (already in requirements)

---

## 8. RECOMMENDED LoRA HYPERPARAMETERS

Based on the small per-user dataset size (~30 examples):

```python
lora_config = {
    "r": 8,                      # Rank (small due to small dataset)
    "lora_alpha": 16,            # Scaling factor (typically 2x rank)
    "target_modules": [
        "q_proj",
        "v_proj"                 # Just attention (cheaper than all linear layers)
    ],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

training_config = {
    "learning_rate": 5e-4,       # Higher LR for small data
    "num_train_epochs": 5,       # More epochs since small dataset
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 8,
    "weight_decay": 0.01,        # Regularization
    "warmup_steps": 10,
    "max_grad_norm": 1.0,
    "save_strategy": "steps",
    "save_steps": 50,
    "eval_strategy": "steps",
    "eval_steps": 50,
    "logging_steps": 10,
    "load_best_model_at_end": True,
    "metric_for_best_model": "loss",
}
```

---

## 9. NEXT STEPS FOR IMPLEMENTATION

1. **Create LoRA Training Script** (`scripts/train_lora_per_user.py`):
   - Load persona data
   - Prepare training dataset
   - Configure PEFT LoRA
   - Train using HuggingFace Trainer
   - Save adapter

2. **Create LoRA Evaluation Script** (`scripts/eval_lora.py`):
   - Load trained adapters for each persona
   - Evaluate on test set
   - Compare against baseline
   - Compute improvement metrics

3. **Create Configuration Files**:
   - `configs/lora_training.json` - Training hyperparameters
   - `configs/lora_eval.json` - Evaluation config

4. **Track Experiments**:
   - Use WandB or tensorboard for monitoring
   - Log per-persona improvements
   - Compare global vs per-user metrics

5. **Analysis & Reporting**:
   - Which personas benefit most from LoRA?
   - How does performance scale with training set size?
   - Trade-offs between compute and improvement

---

## 10. FILE LOCATIONS SUMMARY

| Purpose | Location | Status |
|---------|----------|--------|
| Training data | `data/cleaned/dialogs_clean.jsonl` | Ready |
| Data splits | `data/splits/edgesplits.json` | Ready |
| Base model | HuggingFace (auto-download) | Ready |
| Config template | `configs/baseline_v1.0.json` | Ready |
| Baseline script | `scripts/run_baseline_benchmark.py` | Ready |
| Metrics utilities | `scripts/action_metrics.py` | Ready |
| LoRA adapter storage | `models/lora_per_user/<persona_id>/` | (Create) |
| Training script | `scripts/train_lora_per_user.py` | (Create) |
| Eval script | `scripts/eval_lora.py` | (Create) |
| Results | `results/lora/` | (Create) |

---

## Final Notes

**You're in a great position to implement LoRA training!** The codebase has:
- Clean, well-structured data with proper splits
- Established evaluation infrastructure
- PEFT library already installed
- Clear patterns for data loading, prompt formatting, and metrics
- Excellent documentation (design.md, plan.md)

The main work is creating:
1. A training loop using PEFT's LoRA
2. Per-persona data preparation
3. Evaluation against baseline

Start with training a single persona to debug, then scale to all 200 personas.

