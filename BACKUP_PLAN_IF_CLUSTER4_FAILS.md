# Backup Plan: If Cluster 4 Still Fails

## Current Status

**Cluster 0 Results**: 72.65% (FAILED - worse than unified 82.14%)

**Cluster 4**: Training complete, evaluation running...
- 72 personas, 2160 training examples
- Lower learning rate (2e-4 vs 5e-4)
- 5 epochs vs 3
- Expected: 83-87%

**IF Cluster 4 < 82.14% → Execute backup plan**

---

## Backup Strategy 1: Different LoRA Merge Weights

### Problem
Current cluster training starts from base model, not leveraging existing per-persona LoRAs.

### Solution: Weighted Merge of Per-Persona LoRAs
Instead of training from scratch, merge existing per-persona LoRAs within cluster with task-adaptive weights:

```python
# For cluster 4 (72 personas):
# 1. Load all 72 per-persona LoRAs
# 2. Merge with smart weights based on:
#    - Validation performance
#    - Similarity to cluster centroid
#    - Training data quality

weights = []
for persona in cluster_4_personas:
    val_score = persona_val_performance[persona]
    similarity = cosine_sim(persona_emb, cluster_centroid)
    weight = val_score * similarity
    weights.append(weight)

# Normalize
weights = softmax(weights)

# Merge
cluster_4_lora = weighted_average(loras, weights)
```

**Expected improvement**: 84-86% (uses proven per-persona signal)
**Time**: 15 minutes (no training!)

---

## Backup Strategy 2: Train from Unified, Not Base

### Problem
Training cluster LoRA from base model discards unified model's knowledge.

### Solution: Fine-tune FROM unified model
```python
# Instead of:
model = AutoModelForCausalLM.from_pretrained(base_model)
model = get_peft_model(model, lora_config)  # New LoRA

# Do:
model = AutoModelForCausalLM.from_pretrained(base_model)
unified_lora_path = 'models/lora_unified'
model = PeftModel.from_pretrained(model, unified_lora_path)  # Load unified
# Then add NEW adapter on top for cluster-specific tuning
model.add_adapter("cluster_4", lora_config)
model.set_adapter("cluster_4")
```

**Key insight**: Unified (82.14%) is strong baseline. Don't throw it away!

**Expected improvement**: 85-88%
**Time**: ~70 min training

---

## Backup Strategy 3: Hybrid Unified + Cluster

### Problem
Cluster-only loses global patterns, Unified-only loses personalization.

### Solution: Mixture at inference
```python
# At inference for persona in cluster 4:
unified_output = unified_model.generate(input)
cluster_output = cluster_4_model.generate(input)

# Ensemble
final_output = 0.6 * unified_output + 0.4 * cluster_output
```

**Expected improvement**: 83-85%
**Time**: Instant (no training)

---

## Backup Strategy 4: Constrained LoRA (Less Overfitting)

### Problem
LoRA might still be overfitting despite more data.

### Solution: Add regularization + constraints
```python
lora_config = LoraConfig(
    r=4,  # LOWER rank (less params)
    lora_alpha=8,  # Lower alpha
    lora_dropout=0.1,  # ADD dropout (was 0.05)
    target_modules=["q_proj"],  # ONLY q_proj (not v_proj)
)

# Add L2 regularization
training_args = TrainingArguments(
    weight_decay=0.05,  # Stronger (was 0.01)
    warmup_ratio=0.1,   # More warmup
)
```

**Expected improvement**: 84-86%
**Time**: ~70 min training

---

## Backup Strategy 5: Data Augmentation

### Problem
Even 2160 examples might not be enough.

### Solution: Augment training data
```python
# For each training example:
# 1. Paraphrase user queries
# 2. Add noise to context
# 3. Back-translation
# 4. Synonym replacement

# Effectively 3x data: 2160 → 6480 examples
```

**Expected improvement**: 83-86%
**Time**: ~90 min (data prep + training)

---

## Decision Tree

```
After Cluster 4 Evaluation:

Score > 82.14%?
├─ YES (82-85%)
│   └─ Victory! Use cluster 4
│       └─ Compare with MoE, pick best
│
└─ NO (<82%)
    ├─ Try Strategy 1: Weighted merge (15 min)
    │   ├─ Score > 82%? → Use it!
    │   └─ Still < 82%? → Try Strategy 2
    │
    ├─ Try Strategy 2: Train from unified (70 min)
    │   ├─ Score > 82%? → Use it!
    │   └─ Still < 82%? → Try Strategy 3
    │
    └─ Give up on cluster training
        └─ Use MoE or unified as final result
```

---

## Quick Implementation: Strategy 1 (Weighted Merge)

**If cluster 4 < 82%, run this immediately:**

```bash
# Create weighted merge script
python scripts/weighted_merge_cluster.py --cluster_id 4

# Evaluate
python scripts/eval_weighted_merge.py --cluster_id 4

# Time: 15 minutes total
```

This leverages existing per-persona LoRAs with smart weighting!

---

## Quick Implementation: Strategy 2 (Train from Unified)

```bash
# Modify training to start from unified
python scripts/train_cluster_from_unified.py \
    --cluster_id 4 \
    --epochs 3 \
    --batch_size 2 \
    --lr 1e-4  # Even lower LR

# Time: 70 minutes
```

---

## Expected Timeline

```
Now:              Cluster 4 eval running (2 hrs)
+2 hours:         Check results

IF cluster 4 < 82%:
  +15 min:        Try Strategy 1 (weighted merge)
  +30 min:        Evaluate Strategy 1

  IF still < 82%:
    +70 min:      Try Strategy 2 (train from unified)
    +60 min:      Evaluate Strategy 2

Total worst case: 2 + 2.5 = 4.5 hours
```

---

## Success Criteria

**Minimum acceptable**: > 82.14% (beat unified)
**Good**: 84-86%
**Excellent**: 87-89%

**If we can't beat 82.14%:**
- Use unified or MoE as main result
- Cluster training documented as "attempted but not effective"
- Focus report on MoE and other successful approaches

---

## Implementation Priority

1. **Wait for cluster 4 results** (~2 hrs)
2. **If < 82%**: Try Strategy 1 first (fastest, no training)
3. **If still < 82%**: Try Strategy 2 (most promising)
4. **If still < 82%**: Accept that cluster training doesn't work for this dataset

---

## Key Insight

**The problem might be fundamental:**
- Clustering quality is poor (silhouette score 0.022)
- Personas might not cluster well
- Small model (0.5B params) might not have capacity for cluster-level specialization

**If cluster training fails, MoE merging is the winner!**
