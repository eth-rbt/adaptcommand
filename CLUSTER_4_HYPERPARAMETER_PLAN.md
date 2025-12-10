# Cluster 4 Hyperparameter Tuning Plan

## Why Cluster 4?

**Cluster size comparison:**
```
Cluster 0: 16 personas,  480 examples  → 72.65% ❌ (FAILED)
Cluster 1: 52 personas, 1560 examples
Cluster 2: 25 personas,  750 examples
Cluster 3: 35 personas, 1050 examples
Cluster 4: 72 personas, 2160 examples → BEST CHOICE! 🎯
```

**Why cluster 4 is best:**
- 4.5x more data than cluster 0 (2160 vs 480 examples)
- Largest cluster = most diverse = less overfitting risk
- 72 personas = good generalization potential

## Why Cluster 0 Failed

**Hypothesis:**
1. **Too little data**: 480 examples still causes overfitting
2. **Poor cluster quality**: Silhouette score 0.022 (very low)
3. **Wrong hyperparameters**: 3 epochs, LR=5e-4 might be suboptimal
4. **Smallest cluster**: 16 personas is too specific

## Hyperparameter Tuning Strategy

We'll test **3 configurations** on cluster 4:

### Configuration 1: More Epochs (Conservative)
```
Epochs: 5 (vs 3)
Batch size: 2
Learning rate: 5e-4
LoRA rank: 8
Time: ~70 min
```
**Rationale**: More training steps, same LR

### Configuration 2: Lower LR + More Epochs (Recommended)
```
Epochs: 5
Batch size: 2
Learning rate: 2e-4 (lower)
LoRA rank: 8
Time: ~70 min
```
**Rationale**: Slower learning = better convergence, less overfitting

### Configuration 3: Higher Rank + Lower LR (Aggressive)
```
Epochs: 5
Batch size: 2
Learning rate: 2e-4
LoRA rank: 16 (double)
Time: ~80 min
```
**Rationale**: More parameters + careful learning

## Step-by-Step Execution Plan

### Phase 1: Train 3 Configurations (~4 hours)

```bash
# Config 1: More epochs
python scripts/train_cluster_lora.py \
    --cluster_id 4 \
    --epochs 5 \
    --batch_size 2 \
    --lr 5e-4 \
    --rank 8 \
    2>&1 | tee logs/cluster_4_config1.log

# Config 2: Lower LR (RECOMMENDED)
python scripts/train_cluster_lora.py \
    --cluster_id 4 \
    --epochs 5 \
    --batch_size 2 \
    --lr 2e-4 \
    --rank 8 \
    2>&1 | tee logs/cluster_4_config2.log

# Config 3: Higher rank
python scripts/train_cluster_lora.py \
    --cluster_id 4 \
    --epochs 5 \
    --batch_size 2 \
    --lr 2e-4 \
    --rank 16 \
    2>&1 | tee logs/cluster_4_config3.log
```

**Note**: Can run sequentially or pick one configuration to start.

### Phase 2: Quick Validation Check (~5 min each)

After each training, check validation loss from logs:
```bash
tail -100 logs/cluster_4_config1.log | grep eval_loss
tail -100 logs/cluster_4_config2.log | grep eval_loss
tail -100 logs/cluster_4_config3.log | grep eval_loss
```

Pick the config with **lowest eval_loss**.

### Phase 3: Full Evaluation (~1 hour)

```bash
# Evaluate best configuration
python scripts/eval_cluster_lora.py --cluster_id 4

# This will take ~1 hour (72 personas × 10 test examples)
```

### Phase 4: MoE Comparison (Run in Parallel!)

**While cluster 4 trains**, start MoE in another terminal:

```bash
# Terminal 2: MoE merging (45 min, no GPU needed)
python scripts/moe_merge_sparse.py --k 5

# Then evaluate (30 min)
python scripts/eval_moe_sparse.py
```

### Phase 5: Comprehensive Comparison

```bash
# Generate final comparison
python scripts/compare_all_final.py
```

## Expected Timeline

### Sequential Execution:
```
Phase 1: Train Config 1    →  70 min
         Train Config 2    →  70 min
         Train Config 3    →  80 min
Phase 2: Quick validation  →   5 min
Phase 3: Full evaluation   →  60 min
Phase 4: MoE comparison    →  75 min
Phase 5: Final comparison  →   5 min
─────────────────────────────────────
TOTAL:                        365 min (~6 hours)
```

### Parallel Execution (RECOMMENDED):
```
Terminal 1: Train cluster 4 configs  →  220 min (3.7 hrs)
Terminal 2: MoE merging + eval       →   75 min (1.25 hrs)
─────────────────────────────────────
TOTAL:                                 220 min (~3.7 hours)
```

## Success Criteria

**Cluster 4 must achieve:**
- Embedding similarity > 82.14% (beat unified)
- Better than cluster 0 (72.65%)
- Ideally: 85-87% (original goal)

**If cluster 4 fails (<82%):**
- MoE is our best bet
- Consider that cluster-based training doesn't work for this dataset
- Focus on MoE and other merging strategies

## Decision Tree

```
After Phase 3:

Is cluster 4 > 82.14%?
├─ YES (82-87%)
│   └─ Compare with MoE
│       ├─ MoE better? → Use MoE (main result)
│       └─ Cluster better? → Use cluster (main result)
│
└─ NO (<82.14%)
    └─ Abandon cluster training
        └─ MoE is best approach
```

## Recommended Quick Start

**Start with Config 2 only** (most likely to succeed):

```bash
# Just train config 2 first (70 min)
python scripts/train_cluster_lora.py \
    --cluster_id 4 \
    --epochs 5 \
    --batch_size 2 \
    --lr 2e-4 \
    --rank 8 \
    2>&1 | tee logs/cluster_4_best.log

# While it trains, run MoE in parallel (different terminal)
python scripts/moe_merge_sparse.py --k 5
```

Then evaluate both and compare!

## Files Created

After completion:
- `models/lora_clusters/cluster_04/` - Trained model
- `logs/cluster_4_*.log` - Training logs
- `results/cluster_lora/cluster_lora_results.json` - Evaluation results
- `results/moe_sparse/moe_sparse_results.json` - MoE results
- `results/figures/comprehensive_comparison.png` - Final comparison

## Next Command

**Start now with recommended config:**
```bash
python scripts/train_cluster_lora.py --cluster_id 4 --epochs 5 --batch_size 2 --lr 2e-4
```
