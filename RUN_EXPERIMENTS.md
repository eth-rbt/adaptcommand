# Quick Start: Run Both Experiments

## Experiment 1: Model Merging (Quick!)

### Option A: Cluster-Based Merge (Recommended)
**Time:** 15-20 minutes
**Expected:** +3-5% over unified

```bash
# Merge existing LoRAs by cluster (creates 5 merged models)
python scripts/merge_cluster_loras.py

# Evaluate
python scripts/eval_cluster_merged.py

# Check results
# If > 85%: You're done! 🎯
# If < 85%: Proceed to train cluster LoRAs from scratch
```

### Option B: Simple Average Merge (Fastest)
**Time:** 10-15 minutes
**Expected:** +1-3% over unified

```bash
# Merge ALL 200 LoRAs into one (simplest approach)
python scripts/merge_all_loras_simple.py

# Evaluate
python scripts/eval_merged_simple.py

# Check results
```

---

## Experiment 2: Cluster LoRA Training (If Merging Isn't Enough)

### Quick Test (30 min)
Test on smallest cluster first:

```bash
# Train cluster 0 only (16 personas, 480 examples)
python scripts/train_cluster_lora.py --cluster_id 0

# Evaluate just cluster 0
python scripts/eval_cluster_lora.py --cluster_id 0

# Check if it beats unified (82.14%)
# If yes: proceed to train all clusters
# If no: debug hyperparameters
```

### Full Training (6 hours)
Train all 5 clusters:

```bash
# Train all clusters sequentially
python scripts/train_cluster_lora.py

# This will create:
# - models/lora_clusters/cluster_00/ (16 personas, ~30 min)
# - models/lora_clusters/cluster_01/ (52 personas, ~90 min)
# - models/lora_clusters/cluster_02/ (25 personas, ~45 min)
# - models/lora_clusters/cluster_03/ (35 personas, ~60 min)
# - models/lora_clusters/cluster_04/ (72 personas, ~120 min)

# Evaluate all
python scripts/eval_cluster_lora.py
```

### Custom Training (Advanced)
```bash
# Adjust hyperparameters
python scripts/train_cluster_lora.py \
    --cluster_id 0 \
    --rank 8 \           # LoRA rank (4/8/16)
    --epochs 5 \         # Training epochs
    --batch_size 4 \     # Batch size
    --lr 5e-4            # Learning rate
```

---

## Cluster Info

Your 5 clusters:

```
Cluster 0: 16 personas  →  ~480 examples   (~30 min training)
Cluster 1: 52 personas  → ~1560 examples   (~90 min training)
Cluster 2: 25 personas  →  ~750 examples   (~45 min training)
Cluster 3: 35 personas  → ~1050 examples   (~60 min training)
Cluster 4: 72 personas  → ~2160 examples  (~120 min training)
```

---

## Recommended Order

### Day 1: Test Merging (30 min total)

1. **Cluster merge** (15 min)
   ```bash
   python scripts/merge_cluster_loras.py
   python scripts/eval_cluster_merged.py
   ```

2. **Simple merge** (15 min)
   ```bash
   python scripts/merge_all_loras_simple.py
   python scripts/eval_merged_simple.py
   ```

3. **Compare results:**
   ```bash
   python scripts/compare_all_approaches.py
   ```

**Decision point:**
- If cluster merge > 85%: SUCCESS! Stop here 🎯
- If both < 85%: Proceed to train cluster LoRAs

### Day 2-3: Train Cluster LoRAs (if needed)

4. **Test on one cluster** (30 min)
   ```bash
   python scripts/train_cluster_lora.py --cluster_id 0
   python scripts/eval_cluster_lora.py --cluster_id 0
   ```

5. **If successful, train all** (6 hours)
   ```bash
   python scripts/train_cluster_lora.py
   python scripts/eval_cluster_lora.py
   ```

---

## Expected Results

| Method | Time | Expected Performance | vs Unified |
|--------|------|---------------------|------------|
| Baseline | - | 63.79% | -18.4% |
| Unified LoRA | - | 82.14% | baseline |
| Per-Persona LoRA | - | 68.28% | -13.9% ❌ |
| **Simple Merge** | 15 min | 83-85% | **+1-3%** ✓ |
| **Cluster Merge** | 20 min | 85-87% | **+3-5%** ✓ |
| **Cluster LoRA Training** | 6 hrs | 86-88% | **+4-6%** ✓ |

---

## Troubleshooting

### If merging fails:
```bash
# Check if per-persona LoRAs exist
ls models/lora_adapters/

# Should see: persona_000/, persona_001/, ..., persona_199/
```

### If training is too slow:
```bash
# Reduce batch size
python scripts/train_cluster_lora.py --batch_size 2

# Or reduce epochs
python scripts/train_cluster_lora.py --epochs 3
```

### If out of memory:
```bash
# Use smaller rank
python scripts/train_cluster_lora.py --rank 4
```

---

## Quick Comparison Command

After running experiments:

```bash
python -c "
import json

print('Results Comparison:')
print('=' * 60)

# Unified
with open('results/unified/unified_results.json') as f:
    unified = json.load(f)['metrics']['embedding_similarity']
print(f'Unified LoRA:     {unified:.4f}')

# Simple merge (if exists)
try:
    with open('results/merged_simple/results.json') as f:
        simple = json.load(f)['embedding_similarity']
    print(f'Simple Merge:     {simple:.4f} ({(simple-unified)*100:+.2f}%)')
except:
    print('Simple Merge:     Not run yet')

# Cluster merge (if exists)
try:
    with open('results/merged_cluster/results.json') as f:
        cluster_merged = json.load(f)['embedding_similarity']
    print(f'Cluster Merge:    {cluster_merged:.4f} ({(cluster_merged-unified)*100:+.2f}%)')
except:
    print('Cluster Merge:    Not run yet')

# Cluster LoRA (if exists)
try:
    with open('results/cluster_lora/results.json') as f:
        cluster_lora = json.load(f)['embedding_similarity']
    print(f'Cluster LoRA:     {cluster_lora:.4f} ({(cluster_lora-unified)*100:+.2f}%)')
except:
    print('Cluster LoRA:     Not run yet')
"
```

---

## My Recommendation

**Start here** (20 minutes):

```bash
# 1. Cluster merge (most likely to work well)
python scripts/merge_cluster_loras.py

# 2. Evaluate
python scripts/eval_cluster_merged.py

# 3. Check if > 85%
#    YES → You're done! No need to train
#    NO  → Train cluster LoRAs
```

This gives you the best chance of beating unified with minimal time investment!
