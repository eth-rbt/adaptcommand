# Prefix Tuning on Unified LoRA - Implementation Plan

## Core Idea

**Start from the winner (Unified LoRA @ 82.14%) and add lightweight prefix-based personalization**

Instead of training new LoRAs, we:
1. Load the unified LoRA model (frozen)
2. Learn small "prefix" embeddings (virtual tokens) per persona/cluster
3. Prepend these prefixes to the input to provide personalization context

**Why this might work:**
- Builds on proven strong base (82.14%)
- Much fewer parameters (only prefix embeddings)
- Less overfitting risk
- Prefix acts as "soft prompt" for personalization

---

## Method Overview

### Traditional Prefix Tuning
```
Input: [PREFIX_TOKENS] + [USER_INPUT]
       ↓
    Transformer
       ↓
    Output
```

### Our Approach: Prefix Tuning on Unified LoRA
```
Input: [PERSONA_PREFIX] + [USER_INPUT]
       ↓
  Unified LoRA Model (FROZEN)
       ↓
    Output
```

**Key difference**: We freeze the unified LoRA and only train prefix embeddings

---

## Architecture Details

### Prefix Configuration

**Option 1: Continuous Prefix (Recommended)**
- Learn embedding vectors directly
- Prefix length: 10-20 tokens
- Embedding dim: 896 (Qwen 0.5B hidden size)
- Parameters: `prefix_length × 896 = 10 × 896 = 8,960 params per persona`

**Option 2: Discrete Prefix**
- Learn from existing vocabulary
- Less flexible but interpretable
- Not recommended for this task

### Training Configuration

```python
prefix_config = {
    'prefix_length': 10,  # Number of virtual tokens
    'hidden_size': 896,   # Qwen 0.5B embedding dimension
    'init_strategy': 'random',  # or 'vocab_sample'
    'learning_rate': 1e-3,  # Higher LR since only training prefix
    'epochs': 10,  # More epochs since fewer params
    'batch_size': 2,
    'warmup_ratio': 0.1,
}
```

---

## Experiment Plan

### Experiment 1: Per-Persona Prefix Tuning

**Goal**: Learn 200 personalized prefixes (one per persona)

**Setup**:
- Base model: Unified LoRA (frozen)
- Train 200 prefixes (10 tokens × 896 dim each)
- Training data: 30 examples per persona
- Training time: ~2-3 min per persona (only 8,960 params!)
- Total time: ~10 hours for all 200 personas

**Expected outcome**: 83-85% (slight improvement over unified)

**Advantages**:
- Very fast training (only prefix params)
- Low overfitting risk (8,960 params vs millions in LoRA)
- Can still use unified model for new users

### Experiment 2: Cluster-Based Prefix Tuning

**Goal**: Learn 5 cluster-specific prefixes

**Setup**:
- Base model: Unified LoRA (frozen)
- Train 5 cluster prefixes (one per cluster)
- Training data: 480-2160 examples per cluster
- Training time: ~10-15 min per cluster
- Total time: ~1 hour for all 5 clusters

**Expected outcome**: 84-86% (better than unified)

**Advantages**:
- Much faster than per-persona
- More data per prefix (less overfitting)
- Easy to scale to new personas (assign to cluster)

---

## Implementation Steps

### Phase 1: Single Persona Proof of Concept (30 min)

**Test on one persona to validate approach**

```bash
# 1. Implement prefix tuning script
python scripts/train_prefix_on_unified.py \
    --persona_id persona_000 \
    --unified_lora_path models/lora_unified \
    --prefix_length 10 \
    --epochs 10 \
    --lr 1e-3

# 2. Evaluate
python scripts/eval_prefix_unified.py \
    --persona_id persona_000 \
    --prefix_model_path models/prefix_per_persona/persona_000

# Expected: 70-75% (should beat per-persona LoRA 69.2% for persona_000)
```

**Success criteria**: > 69.2% (persona_000's per-persona LoRA score)

### Phase 2: Full Per-Persona Prefix Training (10 hours)

```bash
# Train all 200 personas in parallel (if multiple GPUs)
# Or sequentially (10 hours)
python scripts/train_all_persona_prefixes.py \
    --unified_lora_path models/lora_unified \
    --output_dir models/prefix_per_persona \
    --prefix_length 10 \
    --epochs 10

# Evaluate all
python scripts/eval_all_prefix_unified.py \
    --prefix_dir models/prefix_per_persona
```

**Success criteria**: > 82.14% (beat unified baseline)

### Phase 3: Cluster Prefix Training (1 hour)

**Try cluster 4 first (72 personas, 2160 examples)**

```bash
# Train cluster 4 prefix
python scripts/train_cluster_prefix.py \
    --cluster_id 4 \
    --unified_lora_path models/lora_unified \
    --prefix_length 10 \
    --epochs 10 \
    --lr 1e-3

# Evaluate cluster 4
python scripts/eval_cluster_prefix.py \
    --cluster_id 4 \
    --prefix_model_path models/prefix_clusters/cluster_04
```

**Success criteria**: > 82.14% (beat unified)

### Phase 4: All Cluster Prefixes (1 hour)

```bash
# Train all 5 clusters
for cluster in 0 1 2 3 4; do
    python scripts/train_cluster_prefix.py \
        --cluster_id $cluster \
        --unified_lora_path models/lora_unified \
        --output_dir models/prefix_clusters
done

# Evaluate all
python scripts/eval_all_cluster_prefixes.py
```

---

## Code Structure

### New Scripts to Create

1. **`scripts/train_prefix_on_unified.py`**
   - Load unified LoRA model (frozen)
   - Add prefix tuning layer
   - Train on persona/cluster data
   - Save prefix embeddings

2. **`scripts/eval_prefix_unified.py`**
   - Load unified LoRA + prefix
   - Evaluate on test set
   - Compare with baselines

3. **`scripts/train_all_persona_prefixes.py`**
   - Batch training for all personas
   - Parallel processing support

4. **`scripts/train_cluster_prefix.py`**
   - Train cluster-specific prefixes
   - Cluster data loading

### Key Implementation Details

```python
class PrefixTuning(nn.Module):
    def __init__(self, prefix_length, hidden_size):
        super().__init__()
        # Learnable prefix embeddings
        self.prefix_embeddings = nn.Parameter(
            torch.randn(prefix_length, hidden_size)
        )

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0]

        # Expand prefix for batch
        prefix = self.prefix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        # Get input embeddings from model
        input_embeds = self.model.get_input_embeddings()(input_ids)

        # Concatenate prefix with input
        combined_embeds = torch.cat([prefix, input_embeds], dim=1)

        # Extend attention mask for prefix tokens
        prefix_mask = torch.ones(batch_size, self.prefix_length,
                                  device=attention_mask.device)
        combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        return combined_embeds, combined_mask
```

### Training Loop

```python
def train_prefix(unified_model, prefix_layer, train_data, config):
    # Freeze unified model
    for param in unified_model.parameters():
        param.requires_grad = False

    # Only train prefix
    optimizer = AdamW(prefix_layer.parameters(), lr=config.lr)

    for epoch in range(config.epochs):
        for batch in train_data:
            # Get prefix-augmented inputs
            embeds, mask = prefix_layer(batch['input_ids'], batch['attention_mask'])

            # Forward through frozen unified model
            outputs = unified_model(
                inputs_embeds=embeds,
                attention_mask=mask,
                labels=batch['labels']
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
```

---

## Expected Results

### Best Case Scenario
| Method | Score | vs Unified | Improvement |
|--------|-------|------------|-------------|
| Unified LoRA | 82.14% | baseline | - |
| **Per-Persona Prefix** | **84-85%** | **+2-3%** | ✓ |
| **Cluster Prefix** | **85-87%** | **+3-5%** | ✓✓ |

### Worst Case Scenario
| Method | Score | vs Unified | Status |
|--------|-------|------------|--------|
| Unified LoRA | 82.14% | baseline | - |
| Per-Persona Prefix | 80-82% | -2 to 0% | ❌ No improvement |
| Cluster Prefix | 81-83% | -1 to +1% | ❌ Marginal |

---

## Why This Might Work (Unlike Previous Methods)

### Previous Failures:
1. **Per-Persona LoRA**: Severe overfitting (68.28%)
2. **Cluster LoRA**: Poor clustering + insufficient data (74.14%)
3. **MoE Merging**: Destroys knowledge (66.38%)

### Prefix Tuning Advantages:
1. **Starts from strong base**: Builds on 82.14% unified model
2. **Minimal parameters**: Only 8,960 params/persona (vs millions in LoRA)
3. **Less overfitting**: Much harder to overfit with so few params
4. **Additive not destructive**: Doesn't modify unified weights
5. **Efficient**: Very fast training (2-3 min per persona)

---

## Risk Mitigation

### If Per-Persona Fails Again:
- Try cluster prefixes (more data, less overfitting)
- Increase prefix length (more capacity)
- Try different initialization strategies

### If Cluster Prefixes Fail:
- Try hierarchical prefixes (cluster + persona)
- Combine with retrieval augmentation
- Accept that unified is optimal for this task

---

## Timeline

### Quick Start (Proof of Concept): 1 hour
1. Implement prefix tuning (30 min)
2. Train + eval persona_000 (15 min)
3. Train + eval cluster 4 (15 min)

### Full Evaluation: 12 hours
1. Train all 200 persona prefixes (10 hrs)
2. Evaluate all personas (1 hr)
3. Train + eval all clusters (1 hr)

### Parallelized (if 4 GPUs): 3 hours
1. Split personas across GPUs (2.5 hrs)
2. Evaluate all (30 min)

---

## Next Steps

**Immediate**:
1. Implement `scripts/train_prefix_on_unified.py`
2. Test on persona_000 as proof of concept
3. If successful (>69.2%), proceed with full training

**Expected commands**:
```bash
# Step 1: Test on one persona
python scripts/train_prefix_on_unified.py --persona_id persona_000

# Step 2: If successful, train all
python scripts/train_all_persona_prefixes.py

# Step 3: Train cluster prefixes
python scripts/train_cluster_prefix.py --cluster_id 4
```

---

## Success Criteria

**Minimum viable**: > 82.14% (beat unified baseline)
**Good**: 83-85% (+1-3% improvement)
**Excellent**: 85-87% (+3-5% improvement)

**If < 82.14%**: Accept that unified LoRA is optimal for this dataset
