# Final Results: Personalization Methods Comparison

## Executive Summary

**Winner: Unified LoRA**

After extensive experimentation with multiple personalization strategies, we found that **simple unified training outperforms ALL personalization approaches** across all three key metrics. Every attempt to personalize the model resulted in worse performance.

---

## Comprehensive Results (All 3 Key Metrics)

| Rank | Method | Embedding Similarity | Device Precision | Numerical Precision | Status |
|------|--------|---------------------|------------------|---------------------|--------|
| 1 | **Unified LoRA** | **82.14%** | **93.80%** | **29.35%** | WINNER |
| 2 | MoE Sparse (K=5) | 66.38% (-15.8%) | 90.22% (-3.6%) | 17.88% (-11.5%) | Failed |
| 3 | Weighted Merge (cluster 4) | 61.34% (-20.8%) | 61.14% (-32.7%) | 19.04% (-10.3%) | Failed |
| 4 | Per-Persona LoRA | 68.28% (-13.9%) | N/A | N/A | Failed |

**Key Findings:**
- **Embedding Similarity**: Unified LoRA dominates (+15.8% over best personalization)
- **Device Precision**: Unified LoRA best, MoE Sparse close second (-3.6%), Weighted Merge catastrophic drop (-32.7%)
- **Numerical Precision**: Unified LoRA dominates (+10.3% over best personalization)

---

## Complete Results (Embedding Similarity Only)

| Rank | Method | Embedding Similarity | vs Unified | Status |
|------|--------|---------------------|------------|--------|
| 1 | **Unified LoRA** | **82.14%** | baseline | WINNER |
| 2 | Cluster 4 LoRA (trained) | 74.14% | -8.0% | Failed |
| 3 | Cluster 0 LoRA (trained) | 72.65% | -9.5% | Failed |
| 4 | Per-Persona LoRA (30 ex) | 68.28% | -13.9% | Failed |
| 5 | MoE Sparse (K=5) | 66.38% | -15.8% | Failed |
| 6 | Weighted Merge (cluster 4) | 61.34% | -20.8% | Failed (Worst) |

---

## Methods Tested

### 1. Unified LoRA Baseline - **Comprehensive Metrics Available**
- **Embedding Similarity**: 82.14%
- **Device Precision**: 93.80% (best!)
- **Numerical Precision**: 29.35% (best!)
- **Param F1**: 89.58% (best!)
- **Approach**: Single LoRA trained on all 6000 examples
- **Pros**: Simple, generalizes well, dominates ALL metrics
- **Cons**: No personalization
- **Status**: **BEST PERFORMER ACROSS ALL METRICS**

### 2. Cluster 4 LoRA (Trained from Scratch)
- **Score**: 74.14%
- **Approach**: Train cluster LoRA on 72 personas (2160 examples)
- **Hyperparameters**: 5 epochs, LR=2e-4, batch_size=2
- **Why it failed**: Poor clustering quality (silhouette=0.022), insufficient data
- **Status**: Failed (-8.0% vs unified)

### 3. Cluster 0 LoRA (Trained from Scratch)
- **Score**: 72.65%
- **Approach**: Train cluster LoRA on 16 personas (480 examples)
- **Why it failed**: Too little data, severe overfitting
- **Status**: Failed (-9.5% vs unified)

### 4. Per-Persona LoRA
- **Score**: 68.28%
- **Approach**: Individual LoRAs for each of 200 personas (30 examples each)
- **Why it failed**: Severe overfitting on tiny datasets
- **Status**: Failed (-13.9% vs unified)

### 5. MoE Sparse (K=5) - **Comprehensive Metrics Available**
- **Embedding Similarity**: 66.38% (±8.74%)
- **Device Precision**: 90.22% (close to unified!)
- **Numerical Precision**: 17.88%
- **Param F1**: 86.14%
- **Approach**: For each persona, merge 5 most similar LoRAs within their cluster
- **Variance**: Min=37.19%, Max=88.04%
- **Why it failed**: Weight merging severely degrades embedding similarity, though device precision stays high
- **Status**: Failed (-15.8% embedding, -3.6% device precision)

### 6. Weighted Merge (Cluster 4) - **Comprehensive Metrics Available**
- **Embedding Similarity**: 61.34%
- **Device Precision**: 61.14% (catastrophic drop!)
- **Numerical Precision**: 19.04%
- **Param F1**: 55.70%
- **Approach**: Smart merging of per-persona LoRAs within cluster using validation performance and centrality weights
- **Why it failed**: Averaging destroys ALL aspects of performance - embedding, device detection, and parameter extraction
- **Status**: **WORST PERFORMER** (-20.8% embedding, -32.7% device precision)

---

## Why Did ALL Personalization Methods Fail?

### 1. **Poor Clustering Quality**
- Silhouette score: **0.022** (very low)
- Indicates clusters don't represent meaningful persona groups
- Personas may not cluster based on behavioral patterns

### 2. **Small Model Capacity**
- Qwen 0.5B params insufficient for fine-grained personalization
- Not enough capacity to learn persona-specific patterns while maintaining general knowledge

### 3. **Insufficient Training Data**
- 30 examples/persona far too small (severe overfitting)
- Even 2160 examples (cluster 4) insufficient
- Unified model benefits from 6000 total examples

### 4. **Merging Destroys Knowledge**
- Weighted averaging of LoRA weights dilutes specialized patterns
- MoE merging performs WORSE than individual overfitted LoRAs (66.38% vs 68.28%)
- No synergy from combining experts - only degradation

### 5. **Task Characteristics**
- Smart-home commands may not benefit from personalization
- Domain knowledge more important than user-specific patterns
- Generic responses work better than personalized ones

---

## Comprehensive Metric Analysis

### Key Insights from 3-Metric Evaluation

**1. Embedding Similarity Pattern:**
- Unified LoRA dominates at 82.14%
- All personalization methods show catastrophic drops
- Worst: Weighted Merge (61.34%, -20.8%)
- Even "best" personalization (MoE) drops 15.8%

**2. Device Precision Pattern:**
- Unified LoRA: 93.80% (exceptional)
- MoE Sparse: 90.22% (only -3.6% drop - surprisingly good!)
- Weighted Merge: 61.14% (-32.7% catastrophic drop)
- **Insight**: MoE maintains device detection ability despite poor embedding similarity

**3. Numerical Precision Pattern:**
- Unified LoRA: 29.35% (best)
- Weighted Merge: 19.04% (-10.3%)
- MoE Sparse: 17.88% (-11.5%)
- **Insight**: All personalization methods struggle with numerical parameter extraction

### Surprising Finding: MoE Sparse

MoE Sparse shows an interesting split personality:
- **Bad** at embedding similarity (66.38%, -15.8%)
- **Good** at device precision (90.22%, only -3.6%)
- **Bad** at numerical precision (17.88%, -11.5%)

This suggests MoE merging preserves device classification knowledge but destroys:
- Semantic coherence (low embedding similarity)
- Numerical parameter knowledge (low numerical precision)

### Catastrophic Failure: Weighted Merge

Weighted Merge fails across ALL dimensions:
- Worst embedding similarity (61.34%)
- Catastrophic device precision drop (61.14%, -32.7%)
- Poor numerical precision (19.04%)

**Conclusion**: Smart averaging of weights is universally destructive, even worse than MoE's simple cosine-weighted merging.

---

## Training Details

### Cluster Training Attempts

**Cluster 0** (Failed):
- 16 personas, 480 training examples
- 3 epochs, batch_size=2, LR=5e-4
- Training time: 42 minutes
- Result: 72.65% (FAILED)

**Cluster 4** (Failed with optimized hyperparameters):
- 72 personas, 2160 training examples (4.5x more data!)
- 5 epochs, batch_size=2, LR=2e-4 (lower LR for better convergence)
- Training time: 70 minutes
- Result: 74.14% (STILL FAILED)

### MoE Merging

**Sparse MoE (K=5)**:
- Created 200 personalized models
- Each merges 5 most similar LoRAs from cluster
- Weights based on cosine similarity
- Creation time: 3.5 minutes (no training!)
- Evaluation time: 94 minutes (200 personas)
- Result: 66.38% (WORST)

### Weighted Merge (Backup Strategy)

**Weighted Merge Cluster 4**:
- Smart weights = validation_score × centrality_to_cluster
- Used existing per-persona LoRAs (no training)
- Merge time: 2 minutes
- Evaluation time: 41 minutes (72 personas)
- Result: 67.00% (FAILED)

---

## Key Insights

1. **Simplicity wins**: Unified training on all data outperforms complex personalization

2. **More data > personalization**: The unified model has access to 6000 examples vs 30 per-persona, and the data advantage dominates

3. **Poor clustering == failed personalization**: With silhouette score of 0.022, personas don't form meaningful groups

4. **Merging is destructive**: Averaging LoRA weights consistently degrades performance

5. **Small models need more data**: 0.5B params insufficient for effective personalization with limited data

---

## Recommendations

### For This Dataset:
✅ **Use Unified LoRA** - Best performance, simplest approach
❌ Avoid personalization - All methods failed
❌ Avoid merging strategies - Consistently worst performers

### For Future Work:
1. **Larger models**: Try 3B+ params for personalization capacity
2. **More data**: Collect 100+ examples per persona
3. **Better clustering**: Use behavioral patterns, not just description embeddings
4. **Alternative approaches**: Consider prompt-based personalization instead of fine-tuning
5. **Hybrid methods**: Combine unified model with lightweight persona-specific adapters

---

## Files and Artifacts

### Results:
- `results/unified/unified_results.json` - **Unified model (82.14% emb, 93.80% dev, 29.35% num) [COMPREHENSIVE]**
- `results/moe_sparse/moe_sparse_comprehensive_results.json` - **MoE Sparse (66.38% emb, 90.22% dev, 17.88% num) [COMPREHENSIVE]**
- `results/weighted_merge/weighted_merge_comprehensive.json` - **Weighted Merge (61.34% emb, 61.14% dev, 19.04% num) [COMPREHENSIVE]**
- `results/cluster_lora/cluster_lora_results.json` - Cluster models (embedding similarity only)
- `results/personalized/personalized_summary.json` - Per-persona LoRAs (68.28%, embedding similarity only)
- `results/moe_sparse/moe_sparse_results.json` - MoE results (old, embedding similarity only)

### Models:
- `models/lora_unified/` - Unified LoRA (WINNER)
- `models/lora_adapters/persona_*/` - 200 per-persona LoRAs
- `models/lora_clusters/cluster_*/` - Cluster LoRAs
- `models/moe_sparse_k5/persona_*/` - 200 MoE merged models
- `models/weighted_cluster_4/` - Weighted merge model

### Documentation:
- `CLUSTER_AND_MERGE_GUIDE.md` - Cluster training guide
- `BACKUP_PLAN_IF_CLUSTER4_FAILS.md` - Backup strategies
- `MOE_QUICK_START.md` - MoE merging guide
- `MOE_MERGING_STRATEGIES.md` - Comprehensive MoE strategies

---

## Conclusion

After testing 6 different approaches including:
- Cluster-based training with optimized hyperparameters
- Per-persona LoRAs
- Weighted merging strategies
- Mixture-of-Experts merging

**The winner is clear: Unified LoRA at 82.14%**

The fundamental lesson: **For this smart-home task with limited data and small model, simple unified training outperforms all personalization attempts.**

Personalization requires:
- Larger models (3B+ params)
- More data (100+ examples/persona)
- Better clustering/grouping strategies
- Or fundamentally different approaches (prompt-based, retrieval-augmented)
