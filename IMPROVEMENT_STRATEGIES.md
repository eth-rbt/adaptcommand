# Strategies to Beat the Unified Model

## Current Situation

**Problem**: Personalized models UNDERPERFORM the unified model due to overfitting

```
Model                 Emb Sim    Improvement    Personas Benefiting
Unified LoRA          0.8214     baseline       -
Per-Persona LoRA      0.6828     -13.9%         9/200 (4.5%)
Hybrid LoRA           0.7591     -6.2%          43/200 (21.5%)
Prefix Per-User       0.6324     -18.9%         6/200 (3.0%)
```

**Root Cause**: Only ~30 training examples per persona → severe overfitting when training individual adapters.

---

## Strategy Ranking (by likelihood of success)

### 🥇 Strategy 1: Cluster-Based Personalization
**Expected improvement: +3-8% over unified**

**Idea**: Group similar personas → train cluster-level adapters with more data

**Why it works**:
- Each cluster has ~600-900 training examples (20-30 personas × 30 examples)
- 20-30× more data reduces overfitting
- Still captures persona-specific patterns at cluster level

**Implementation**:
```bash
# 1. Cluster personas
python scripts/cluster_personas.py

# 2. Train cluster-level adapters (one per cluster)
python scripts/train_cluster_lora.py --n_clusters 10

# 3. (Optional) Add tiny per-user deltas on top
python scripts/train_cluster_plus_user.py
```

**Expected results**:
- 10-15 clusters optimal (based on silhouette score)
- Each cluster adapter sees 600-900 examples
- Should improve 40-60% of personas over unified
- Target: 0.85-0.88 embedding similarity

**Ablations to run**:
- Number of clusters: [5, 10, 15, 20, 25]
- Cluster method: [KMeans, Hierarchical, DBSCAN]
- With/without per-user deltas on top

---

### 🥈 Strategy 2: Retrieval-Augmented Personalization
**Expected improvement: +2-6% over unified**

**Idea**: Keep unified model, but augment context with retrieved user history

**Why it works**:
- No model overfitting (weights stay unified)
- Personalization via context, not weights
- Can leverage unlimited user history
- Dynamic adaptation without retraining

**Implementation**:
```bash
python scripts/retrieval_augmented_baseline.py
```

**Key components**:
1. Index all user training interactions with embeddings
2. For new query, retrieve top-k similar past interactions
3. Inject into context: "You previously responded to similar queries like..."
4. Generate with unified model + augmented context

**Ablations to run**:
- Retrieval k: [0, 1, 3, 5, 10]
- Retrieval method: [semantic similarity, recency, hybrid]
- Context format: [examples, summary, preferences]

**Advantages**:
- No retraining needed
- Scales to unlimited user history
- Transparent (can inspect retrieved examples)
- Works with any model (including frontier LLMs)

---

### 🥉 Strategy 3: Constrained Personalization
**Expected improvement: +1-4% over unified**

**Idea**: Regularize per-user adapters to stay close to unified adapter

**Why it works**:
- Allows personalization but prevents catastrophic drift
- L2 penalty on (LoRA_persona - LoRA_unified)
- Smaller rank (r=2-4) reduces capacity to overfit

**Implementation**:
```python
# Train with L2 regularization to unified adapter
python scripts/train_constrained_lora.py \
  --l2_lambda 0.1 \
  --rank 4
```

**Key parameters**:
- `l2_lambda`: Regularization strength (0.0 = no constraint, 1.0 = strong)
- `rank`: LoRA rank (lower = less overfitting)
- `alpha`: LoRA alpha (lower = smaller updates)

**Ablations to run**:
- L2 lambda: [0.01, 0.05, 0.1, 0.5]
- Rank: [2, 4, 8]
- Compare to knowledge distillation variant

---

### Strategy 4: Selective Personalization
**Expected improvement: +2-5% over unified**

**Idea**: Only personalize for users who benefit; use unified for others

**Why it works**:
- Current data shows only 21.5% of personas benefit from hybrid LoRA
- Personalize only those 21.5%, use unified for the rest
- Could train a meta-classifier to predict who benefits

**Implementation**:
1. Train unified model (already done)
2. Train per-persona adapters (already done)
3. On validation set, identify which personas improve
4. At inference: route to personalized adapter if beneficial, else unified

**Decision rule**:
```python
if persona_id in beneficial_personas:
    use_adapter(f"models/lora_per_user/{persona_id}")
else:
    use_adapter("models/lora_unified")
```

**Meta-learning approach**:
- Train classifier: persona features → "will benefit from personalization?"
- Features: persona embedding, verbosity, sentiment, interaction frequency
- Use this to predict for new users

---

### Strategy 5: Data Augmentation
**Expected improvement: +3-7% over unified**

**Idea**: Synthesize more per-user training examples to reduce data scarcity

**Approaches**:
1. **Paraphrase existing queries** (keep same actions)
2. **Context variation** (same query, different time/weather)
3. **Back-translation** (translate to another language and back)
4. **LLM synthesis** (use GPT-4 to generate persona-consistent examples)

**Implementation**:
```python
# Augment training data
python scripts/augment_persona_data.py \
  --target_size 100 \  # 30 → 100 examples per persona
  --methods paraphrase,context_variation
```

**Expected results**:
- 30 → 100 examples per persona
- Reduces overfitting
- Should improve per-persona LoRA from 68% → 75-78%

**Risk**: Synthetic data may not match real distribution

---

### Strategy 6: Multi-Task Learning
**Expected improvement: +2-4% over unified**

**Idea**: Joint training on multiple related tasks for better regularization

**Tasks**:
1. Primary: Generate response
2. Auxiliary: Predict persona ID from query
3. Auxiliary: Predict device actions (classification)
4. Auxiliary: Predict numerical parameters (regression)

**Why it works**:
- Auxiliary tasks provide regularization
- Shared representations improve generalization
- Forces model to learn user-distinguishing features

**Implementation**:
- Modify training to include multiple loss terms
- Weighted combination: `loss = 1.0*L_gen + 0.1*L_persona + 0.05*L_device`

---

### Strategy 7: Mixture of Experts (MoE) LoRA
**Expected improvement: +3-6% over unified**

**Idea**: Train multiple specialized LoRA experts, route inputs dynamically

**Architecture**:
```
Input → Router → Select LoRA Expert(s) → Generate
                  ↓
         [Expert 1: Formal users]
         [Expert 2: Casual users]
         [Expert 3: Technical users]
         [Expert 4: Brief users]
         ...
```

**Implementation**:
1. Cluster personas into K groups
2. Train K expert LoRAs
3. Train lightweight router to select expert(s) based on input
4. Optionally: soft routing (weighted combination of experts)

**Advantages**:
- Each expert sees more data than per-persona
- Dynamic routing adapts to input
- Can combine multiple experts

---

### Strategy 8: Meta-Learning (MAML/Reptile)
**Expected improvement: +4-8% over unified (if works)**

**Idea**: Learn an initialization that adapts quickly with few examples

**Why it might work**:
- Designed exactly for few-shot learning
- Learns to learn from small datasets
- Could find initialization that adapts well with 5-10 examples

**Implementation**:
```python
# Train meta-learner
python scripts/train_maml_lora.py \
  --inner_lr 1e-4 \
  --outer_lr 1e-5 \
  --support_size 5  # Learn from 5 examples per persona
```

**Challenge**:
- Computationally expensive
- Requires careful hyperparameter tuning
- May not beat simpler methods

---

### Strategy 9: Hybrid: Unified LoRA + Per-User Prefix
**Expected improvement: +2-5% over unified**

**Idea**: Keep unified LoRA (strong), add lightweight per-user prefix/soft prompt

**Why it works**:
- Unified LoRA provides strong base (82% similarity)
- Prefix tuning has fewer parameters → less overfitting
- ~100K params (prefix) vs ~1M params (LoRA) per user

**Implementation**:
```python
# Train per-user prefix on top of frozen unified LoRA
python scripts/train_prefix_on_unified.py \
  --prefix_length 20 \
  --unified_lora_path models/lora_unified
```

**Expected results**:
- Should beat hybrid LoRA (which is at 75.9%)
- Target: 0.83-0.86 embedding similarity
- Much cheaper than per-user LoRA

---

### Strategy 10: User Embedding Conditioning
**Expected improvement: +3-6% over unified**

**Idea**: Learn compact user embeddings, condition model on them

**Architecture**:
```
User ID → Embedding Lookup → User Vector (64-128 dim)
                              ↓
                       Prepend to input
                              ↓
                       Unified Model → Generate
```

**Implementation**:
1. Learn user embeddings jointly with unified model
2. At inference: lookup user embedding + prepend to input
3. Model learns to condition on user vector

**Advantages**:
- Very parameter efficient (64 dims × 200 users = 12.8K params)
- Can interpolate between users
- Can initialize new users with mean embedding

---

## Recommended Experiment Plan

### Phase 1: Quick Wins (1 week)
1. ✅ **Cluster-based personalization** (Strategy 1) - Most likely to succeed
2. ✅ **Retrieval augmentation** (Strategy 2) - Easy to implement, no training
3. ✅ **Selective personalization** (Strategy 4) - Use existing models

### Phase 2: Promising Approaches (1-2 weeks)
4. **Unified LoRA + Per-User Prefix** (Strategy 9) - Cheap, low risk
5. **Constrained personalization** (Strategy 3) - Test regularization
6. **Data augmentation** (Strategy 5) - Address root cause

### Phase 3: Advanced Techniques (2-3 weeks, if time)
7. **Mixture of Experts** (Strategy 7) - Novel, publishable
8. **User embedding conditioning** (Strategy 10) - Elegant solution
9. **Meta-learning** (Strategy 8) - High risk, high reward

---

## Success Criteria

For any strategy to be considered successful:

1. **Improves >60% of personas** over unified (vs current 21.5%)
2. **Mean embedding similarity > 0.85** (vs unified 0.821)
3. **No catastrophic failures** (min per-persona > 0.70)
4. **Statistical significance** (p < 0.05 on paired t-test)

Target metrics:
```
Embedding similarity: 0.85-0.90  (+3-9% over unified)
Device precision:     0.94-0.96  (+0-2% over unified)
Param F1:             0.91-0.94  (+1-4% over unified)
```

---

## Implementation Priority

**Run first** (this week):
1. `python scripts/cluster_personas.py` - Takes 5 min
2. `python scripts/retrieval_augmented_baseline.py` - Takes 30 min

**Run second** (next week):
3. Train cluster-level LoRAs (10 adapters × 1 hour = 10 hours)
4. Test selective personalization (use existing models)
5. Train unified LoRA + per-user prefix

**Evaluate**:
- If clusters work → stop, use that
- If retrieval works → combine with clusters
- If both fail → try data augmentation or MoE

---

## Key Insight

The unified model is strong because it sees ALL the data (6000 examples).
To beat it, you need to either:
1. **Increase data per personalized model** (clustering, augmentation)
2. **Avoid overfitting** (regularization, smaller capacity)
3. **Personalize without weights** (retrieval, conditioning)

The worst approach is what you tried: train separate models on 30 examples each.
