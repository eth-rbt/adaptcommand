# Quick Start: Hybrid Training

Train persona-specific LoRA adapters on top of your unified model in 3 easy steps.

## Step 1: Verify Setup

```bash
python scripts/test_hybrid_setup.py
```

This checks:
- ✓ Unified model exists
- ✓ Data files are available
- ✓ PEFT library supports adapter stacking
- ✓ GPU availability

If any checks fail, follow the error messages to fix them.

## Step 2: Train Single Persona (Test)

Test on one persona first:

```bash
python scripts/train_persona_on_unified.py --persona_id persona_000
```

**Expected time:** ~5-10 minutes on RTX 3070

**Output:** `models/lora_hybrid/persona_000/`

**Check results:**
```bash
cat models/lora_hybrid/persona_000/test_metrics.json
```

Look for:
- `embedding_similarity` > 0.75 (higher is better)
- `param_f1` > 0.85 (higher is better)

## Step 3: Train All Personas

Once the test looks good, train all 200 personas:

```bash
python scripts/train_all_hybrid.py --persona_rank 8
```

**Expected time:** ~16-20 hours for 200 personas on RTX 3070

**To run in background:**
```bash
nohup python scripts/train_all_hybrid.py > hybrid_training.log 2>&1 &
```

**To resume if interrupted:**
```bash
python scripts/train_all_hybrid.py --skip_existing
```

**To train subset:**
```bash
# First 50 personas
python scripts/train_all_hybrid.py --end_index 50

# Next 50 personas
python scripts/train_all_hybrid.py --start_index 50 --end_index 100
```

## Step 4: Compare Results

Compare hybrid approach vs standalone per-persona models:

```bash
python scripts/compare_results.py \
    --baseline results/personalized/personalized_summary.json \
    --personalized results/hybrid/hybrid_summary.json \
    --output results/comparison_hybrid.json
```

## Configuration Options

### Adjust Persona LoRA Rank

```bash
# Smaller adapter (more regularization, faster)
python scripts/train_persona_on_unified.py --persona_id persona_000 --persona_rank 4

# Larger adapter (more capacity)
python scripts/train_persona_on_unified.py --persona_id persona_000 --persona_rank 16
```

**Recommendation:** Start with rank=8, increase if underfitting

### Other Options

```bash
python scripts/train_persona_on_unified.py \
    --persona_id persona_000 \
    --unified_model models/lora_unified \
    --config configs/lora_training.json \
    --output_dir models/lora_hybrid/persona_000 \
    --persona_rank 8 \
    --no_val  # Skip validation during training
```

## Troubleshooting

**Problem:** Test fails with "unified model not found"
```bash
# Train unified model first
python scripts/train_unified_lora.py
```

**Problem:** CUDA out of memory
```bash
# Edit scripts/train_persona_on_unified.py
# Change: per_device_train_batch_size=1, gradient_accumulation_steps=8
```

**Problem:** Training is slow
```bash
# Use smaller persona rank
python scripts/train_all_hybrid.py --persona_rank 4
```

**Problem:** Poor performance on specific persona
```bash
# Try larger rank for that persona
python scripts/train_persona_on_unified.py --persona_id persona_XXX --persona_rank 16
```

## Expected Results

Hybrid models should:
- **Outperform** standalone per-persona models (~5-15% improvement)
- **Approach** unified model performance (within 1-3%)
- Have **less variance** across personas
- Train **faster** (5 epochs vs 10 epochs)

## Architecture Summary

```
Base Model (Qwen 0.5B - 494M params)
    ↓
Unified LoRA (r=16, ~2.1M params, frozen)
    ↓
Persona LoRA (r=8, ~1.0M params, trainable) ← Only this is trained
```

**Total trainable params per persona:** ~1M (0.2% of base model)

## Next Steps

After training, consider:

1. **Analyze per-persona variance** - which personas benefit most?
2. **Try different ranks** - optimal rank may vary per persona
3. **Experiment with data mixing** - add similar personas' data
4. **Test inference speed** - compare unified vs hybrid
