# Phase A Scripts — Data Readiness

This directory contains scripts for Phase A (Data Readiness) of the EdgeWisePersona personalization study.

## Prerequisites

Make sure you've activated the virtual environment and installed dependencies:

```bash
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

## Phase A Workflow

Follow these steps in order:

### A.1 — Load Dataset

**Purpose:** Acquire and explore the EdgeWisePersona dataset

**Script:** `load_edgewise_data.py`

**Usage:**

```bash
# From HuggingFace (example - replace with actual dataset name)
python scripts/load_edgewise_data.py \
  --source "edgewise/persona-dataset" \
  --output_dir data/raw

# From local file
python scripts/load_edgewise_data.py \
  --source /path/to/dataset.jsonl \
  --output_dir data/raw

# Just explore without saving
python scripts/load_edgewise_data.py \
  --source "dataset-name" \
  --explore_only
```

**Outputs:**
- `data/raw/edgewise_raw.jsonl` — Raw dialogues
- `data/raw/dataset_stats.json` — Dataset statistics

---

### A.2 — Clean Data

**Purpose:** Normalize whitespace, standardize fields, filter degenerate exchanges

**Script:** `clean_data.py`

**Usage:**

```bash
# Default settings
python scripts/clean_data.py \
  --input data/raw/edgewise_raw.jsonl \
  --output data/cleaned/dialogs_clean.jsonl

# Custom thresholds
python scripts/clean_data.py \
  --input data/raw/edgewise_raw.jsonl \
  --output data/cleaned/dialogs_clean.jsonl \
  --min_msg_length 5 \
  --min_messages 3 \
  --min_total_chars 20
```

**Parameters:**
- `--min_msg_length`: Minimum characters per message (default: 2)
- `--min_messages`: Minimum messages per dialogue (default: 2)
- `--min_total_chars`: Minimum total characters per dialogue (default: 10)

**Outputs:**
- `data/cleaned/dialogs_clean.jsonl` — Cleaned dialogues
- `data/cleaned/cleaning_stats.json` — Cleaning statistics

---

### A.3 — Create Splits

**Purpose:** Generate time-aware train/val/test splits per user

**Script:** `create_splits.py`

**Usage:**

```bash
# Default 60/20/20 split
python scripts/create_splits.py \
  --input data/cleaned/dialogs_clean.jsonl \
  --output data/splits/edgesplits.json

# Custom ratios
python scripts/create_splits.py \
  --input data/cleaned/dialogs_clean.jsonl \
  --output data/splits/edgesplits.json \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --test_ratio 0.15 \
  --online_batch_size 12

# With cold-start data
python scripts/create_splits.py \
  --input data/cleaned/dialogs_clean.jsonl \
  --output data/splits/edgesplits.json \
  --create_cold_start \
  --cold_start_turns 5
```

**Parameters:**
- `--train_ratio`: Training proportion (default: 0.6)
- `--val_ratio`: Validation proportion (default: 0.2)
- `--test_ratio`: Test proportion (default: 0.2)
- `--online_batch_size`: Examples for online updates (default: 10)
- `--min_examples`: Minimum examples per split to include user (default: 1)
- `--create_cold_start`: Generate cold-start test set
- `--cold_start_turns`: Number of initial turns for cold-start (default: 5)

**Outputs:**
- `data/splits/edgesplits.json` — Per-user splits
- `data/splits/split_stats.json` — Split statistics
- `data/splits/cold_start_test.jsonl` — Cold-start data (optional)

---

### A.4 — Validate Splits

**Purpose:** Verify splits are valid (no overlap, proper coverage, balanced)

**Script:** `validate_data.py`

**Usage:**

```bash
# Validate with default thresholds
python scripts/validate_data.py \
  --splits data/splits/edgesplits.json \
  --dialogues data/cleaned/dialogs_clean.jsonl \
  --output data/splits/validation_report.json

# Custom validation parameters
python scripts/validate_data.py \
  --splits data/splits/edgesplits.json \
  --dialogues data/cleaned/dialogs_clean.jsonl \
  --min_examples 2 \
  --max_user_proportion 0.15 \
  --output data/splits/validation_report.json
```

**Parameters:**
- `--min_examples`: Minimum examples per split (default: 1)
- `--max_user_proportion`: Max proportion for single user (default: 0.1)

**Outputs:**
- `data/splits/validation_report.json` — Validation report
- Exit code 0 if valid, 1 if invalid

**Validation Checks:**
1. ✓ No overlap between train/val/test
2. ✓ Minimum coverage per user
3. ✓ Dataset balance (no dominant users)
4. ✓ Temporal ordering preserved

---

## Complete Phase A Pipeline

Run all steps sequentially:

```bash
# 1. Load data
python scripts/load_edgewise_data.py \
  --source "your-dataset-source" \
  --output_dir data/raw

# 2. Clean data
python scripts/clean_data.py \
  --input data/raw/edgewise_raw.jsonl \
  --output data/cleaned/dialogs_clean.jsonl

# 3. Create splits
python scripts/create_splits.py \
  --input data/cleaned/dialogs_clean.jsonl \
  --output data/splits/edgesplits.json \
  --create_cold_start

# 4. Validate splits
python scripts/validate_data.py \
  --splits data/splits/edgesplits.json \
  --dialogues data/cleaned/dialogs_clean.jsonl \
  --output data/splits/validation_report.json
```

---

## Expected Directory Structure After Phase A

```
data/
├── raw/
│   ├── edgewise_raw.jsonl
│   └── dataset_stats.json
├── cleaned/
│   ├── dialogs_clean.jsonl
│   └── cleaning_stats.json
└── splits/
    ├── edgesplits.json
    ├── split_stats.json
    ├── validation_report.json
    └── cold_start_test.jsonl (optional)
```

---

## Troubleshooting

### Issue: Dataset not found

Make sure you have the correct dataset name or local path. Check HuggingFace datasets hub or verify your local file exists.

### Issue: Too many personas excluded

Lower the `--min_examples` threshold in `create_splits.py`, or increase the minimum message requirements in `clean_data.py`.

### Issue: Validation fails (overlap detected)

This indicates a bug in the splitting logic. Please report this issue.

### Issue: Dataset is unbalanced

Some users may naturally have more dialogues. You can filter out dominant users or cap their examples in preprocessing.

---

## Next Steps

Once Phase A is complete and validation passes:
- Proceed to **Phase B** (Baseline Establishment)
- See `plan.md` for the next steps

---

## Questions or Issues?

See `design.md` for the overall project design and `plan.md` for the implementation roadmap.
