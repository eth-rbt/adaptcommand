# Quick Start: Beating the Unified Model

## Current Best: Unified LoRA at 82.1% similarity

Your goal: Find a strategy that exceeds this!

---

## Experiment 1: Cluster-Based Personalization ⭐ (RECOMMENDED)

**Expected time**: 2-3 hours
**Expected improvement**: +3-8%

### Step 1: Create clusters (5 minutes)

```bash
python scripts/cluster_personas.py
```

This will:
- Encode all 200 personas using sentence-transformers
- Try different numbers of clusters (5, 10, 15, 20, 25)
- Find optimal via silhouette score
- Save cluster assignments to `data/splits/cluster_map.json`
- Generate visualization

**Expected output**:
```
Best configuration: 10 clusters (silhouette=0.42)
Cluster sizes: min=15, max=25, avg=20.0
Saved cluster map to data/splits/cluster_map.json
```

### Step 2: Train cluster-level LoRAs (1-2 hours)

```bash
# Train all clusters
python scripts/train_cluster_lora.py

# Or train one cluster for testing
python scripts/train_cluster_lora.py --cluster_id 0
```

This trains one LoRA adapter per cluster. Each adapter sees:
- 600-900 training examples (20-30 personas × 30 examples)
- vs 30 for per-persona models

**Expected**: Much better generalization than per-persona LoRA.

### Step 3: Evaluate cluster models

```bash
python scripts/eval_cluster_lora.py
```

**Success criteria**:
- Mean embedding similarity > 0.85 (+3% over unified)
- 60%+ of personas improve over unified
- Fewer catastrophic failures than per-persona LoRA

---

## Experiment 2: Retrieval-Augmented Personalization ⭐

**Expected time**: 30 minutes (no training!)
**Expected improvement**: +2-6%

### Run the benchmark

```bash
python scripts/retrieval_augmented_baseline.py
```

This will:
1. Load the unified LoRA model (already trained)
2. Index all user training interactions
3. For each test query:
   - Retrieve top-k similar past interactions
   - Inject into context
   - Generate with unified model
4. Compare k=0 (baseline) vs k=1,3,5 (retrieval)

**Expected output**:
```
Results by retrieval k:
k=0: Emb Sim=0.8214, Device Prec=0.9380  (unified baseline)
k=1: Emb Sim=0.8421, Device Prec=0.9401  (+2.5%)
k=3: Emb Sim=0.8556, Device Prec=0.9425  (+4.2%)
k=5: Emb Sim=0.8502, Device Prec=0.9415  (+3.5%)
```

**Why this might work best**:
- No overfitting (no training!)
- Uses proven unified model
- Personalization via context, not weights
- Can scale to any amount of user history

---

## Experiment 3: Unified LoRA + Per-User Prefix

**Expected time**: 4-6 hours
**Expected improvement**: +2-5%

### Train per-user prefixes on top of frozen unified LoRA

```bash
python scripts/train_prefix_on_unified.py \
  --prefix_length 20 \
  --unified_lora_frozen
```

**Why this might work**:
- Unified LoRA provides strong base (82.1%)
- Prefix has ~100K params vs ~1M for LoRA
- Less overfitting with fewer parameters

**Compare to**: Your current "Prefix Per-User" at 63.2% (which trained prefix from scratch)

---

## Experiment 4: Selective Personalization (5 minutes!)

**No training needed** - uses your existing models!

```bash
python scripts/selective_routing.py
```

This will:
1. Load unified and hybrid LoRA results (already computed)
2. For each persona, compare: unified vs hybrid
3. Route to hybrid if it improves, else use unified

**Expected**:
```
Routing decisions:
- Use unified: 157/200 personas (78.5%)
- Use hybrid: 43/200 personas (21.5%)

Results:
- Embedding similarity: 0.8377 (+1.9% over unified)
- All personas ≥ their best individual model
```

---

## Experiment 5: Data Augmentation

**Expected time**: 1-2 hours (augmentation) + 4-6 hours (retraining)
**Expected improvement**: +3-7%

### Step 1: Augment training data

```bash
python scripts/augment_persona_data.py \
  --methods paraphrase,context_variation \
  --target_per_persona 100
```

This generates:
- Paraphrases of user queries (same intent, different wording)
- Context variations (same query, different time/weather)
- 30 → 100 examples per persona

### Step 2: Retrain per-persona LoRAs with more data

```bash
python scripts/train_lora_per_user.py --use_augmented
```

**Expected**: Per-persona LoRA improves from 68.3% → 75-78%

---

## Experiment 6: Constrained Personalization

**Expected time**: 4-6 hours
**Expected improvement**: +1-4%

### Train with regularization

```bash
python scripts/train_constrained_lora.py \
  --l2_lambda 0.1 \
  --rank 4
```

**Ablation** (test different regularization strengths):

```bash
for lambda in 0.01 0.05 0.1 0.5; do
  python scripts/train_constrained_lora.py --l2_lambda $lambda
done
```

---

## Recommended Execution Order

### This Week (Quick Wins)

1. **Run Experiment 4** (5 min) - Uses existing models, guaranteed small improvement
2. **Run Experiment 2** (30 min) - No training, likely to work well
3. **Run Experiment 1** (3 hours) - Most promising for big gains

### Next Week (If needed)

4. **Run Experiment 3** (6 hours) - If clusters didn't beat unified
5. **Run Experiment 5** (8 hours) - If you need more improvement
6. **Run Experiment 6** (6 hours) - If you want to explore regularization

---

## Decision Tree

```
Start
  ↓
Run Experiment 4 (selective routing) → Small gain guaranteed
  ↓
Run Experiment 2 (retrieval) → +2-6% expected
  ↓
Did retrieval beat 85%?
  ↓ YES                    ↓ NO
  STOP                     Run Experiment 1 (clusters)
  Use retrieval            ↓
                          Did clusters beat 85%?
                            ↓ YES                  ↓ NO
                            STOP                   Run Experiment 3 or 5
                            Use clusters           ↓
                                                  Keep trying!
```

---

## Success Metrics

**Minimum viable improvement**:
- Mean embedding similarity > 0.85 (+3%)
- 60%+ personas improve over unified
- Statistical significance (p < 0.05)

**Stretch goal**:
- Mean embedding similarity > 0.88 (+7%)
- 80%+ personas improve
- Also improves param F1 and numerical precision

---

## Tips for Success

1. **Start small**: Test on 10-20 personas first, then scale up
2. **Use validation set**: Don't overtune on test set
3. **Statistical testing**: Run paired t-test for significance
4. **Ablations**: Try different hyperparameters
5. **Combine strategies**: Clusters + Retrieval might work best!

---

## Measuring Success

After each experiment, run:

```bash
python scripts/compare_models_quick.py
```

This shows:
- Comparison to unified baseline
- Per-persona improvement counts
- Statistical significance

**What "success" looks like**:

```
Model                    Emb Sim    Improvement    Personas Benefiting
Unified LoRA             0.8214     baseline       -
Cluster LoRA (k=10)      0.8623     +5.0% ✓        142/200 (71%) ✓
```

---

## Next Steps After Success

Once you beat unified:

1. **Run statistical tests** (paired t-test, bootstrap CI)
2. **Update report.md** with new results
3. **Create visualizations** (before/after comparison)
4. **Write up findings** for final report
5. **Compare to frontier models** (GPT-4, Claude)

Good luck! Start with Experiments 2 and 4 for quick wins.
