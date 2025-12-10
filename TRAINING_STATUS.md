# Cluster LoRA Training Status

## Currently Running

**Cluster 0 Training**
- Personas: 16
- Training examples: ~480 (16 personas × 30 examples)
- Expected time: ~30-40 minutes
- Configuration:
  - Epochs: 3
  - Batch size: 2
  - Learning rate: 5e-4
  - LoRA rank: 8

## Why This Will Beat Unified Model

| Model | Training Examples | Overfitting Risk | Expected Performance |
|-------|------------------|------------------|---------------------|
| Per-Persona LoRA | 30 | ❌ VERY HIGH | 68.3% (failed) |
| **Cluster 0 LoRA** | **480** | **✓ LOW** | **~85-87%** |
| Unified LoRA | 6,000 | ✓ NONE | 82.1% (baseline) |

**Key insight:** 480 examples is the sweet spot:
- Enough data to avoid overfitting (16x more than per-persona)
- Specialized enough to capture cluster-specific patterns
- Should beat unified by +3-5%

## Training Pipeline

### Phase 1: Test Cluster 0 (Running Now - 30 min)
```
Cluster 0: 16 personas, 480 examples
↓
If successful (>83%):
  Proceed to train all 5 clusters
If not successful (<83%):
  Debug hyperparameters
```

### Phase 2: Train All 5 Clusters (~6 hours total)
```
Cluster 0: 16 personas   →  480 examples  (~30 min) ✓
Cluster 1: 52 personas   → 1560 examples  (~90 min)
Cluster 2: 25 personas   →  750 examples  (~45 min)
Cluster 3: 35 personas   → 1050 examples  (~60 min)
Cluster 4: 72 personas   → 2160 examples (~120 min)
```

### Phase 3: Evaluation (~30 min)
```bash
python scripts/eval_cluster_lora.py
```

### Phase 4: Comprehensive Comparison (~5 min)
```bash
python scripts/compare_all_final.py
```

This will create:
- Comprehensive visualization comparing ALL methods
- Summary table with all metrics
- Results saved to `results/figures/comprehensive_comparison.png`

## Expected Final Results

```
Method                    Emb Sim    vs Unified
Baseline                  63.79%     -18.4%
Unified LoRA             82.14%     baseline
Per-Persona LoRA          68.28%     -13.9% ❌
Hybrid LoRA               75.91%     -6.2% ❌
Prefix Per-User           63.24%     -18.9% ❌
Selective Routing         82.99%     +1.0% ✓
Cluster LoRA (GOAL)       86-88%     +4-6% 🎯
```

## Next Steps After Training

1. **Evaluate cluster 0** (5 min)
   ```bash
   python scripts/eval_cluster_lora.py --cluster_id 0
   ```

2. **If cluster 0 > 83%**: Train remaining clusters
   ```bash
   python scripts/train_cluster_lora.py --cluster_id 1
   python scripts/train_cluster_lora.py --cluster_id 2
   python scripts/train_cluster_lora.py --cluster_id 3
   python scripts/train_cluster_lora.py --cluster_id 4
   ```

3. **Evaluate all clusters** (30 min)
   ```bash
   python scripts/eval_cluster_lora.py
   ```

4. **Create final comparison** (5 min)
   ```bash
   python scripts/compare_all_final.py
   ```

## Monitoring Training

Check logs:
```bash
tail -f logs/cluster_0_training.log
```

Or check manually:
```bash
ls -lh models/lora_clusters/cluster_00/
```

## If Training Fails

**Out of memory:**
```bash
python scripts/train_cluster_lora.py --cluster_id 0 --batch_size 1
```

**Too slow:**
```bash
python scripts/train_cluster_lora.py --cluster_id 0 --epochs 2
```

**Want to test quickly:**
```bash
python scripts/train_cluster_lora.py --cluster_id 0 --epochs 1 --batch_size 1
```

## Alternative: Model Merging (If You Want Quick Results)

While training runs, you could also test model merging:

```bash
# In another terminal:
python scripts/merge_cluster_loras.py  # 15 min, expected +3-5%
```

This uses your existing 200 per-persona LoRAs and merges them by cluster. It's FREE (no training) and might give similar results!

## Timeline

```
Now:        Cluster 0 training started
+30 min:    Cluster 0 done → Evaluate
+40 min:    If good, start cluster 1-4 training
+7 hours:   All clusters trained
+7.5 hours: All evaluated
+8 hours:   Final comparison complete 🎯
```

## Success Criteria

**Cluster 0 must achieve:**
- Embedding similarity > 83% (better than unified 82.14%)
- No catastrophic failures (min per-persona > 70%)
- Improvement for 60%+ of personas in cluster 0

**If achieved:** Proceed to train all clusters
**If not:** Debug and retry with different hyperparameters
