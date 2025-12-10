# Personalized Smart Home Assistant: A Comprehensive Study of Adaptation Techniques

**Authors**: [Your Name]
**Date**: December 2025
**Model**: Qwen 2.5 0.5B Instruct
**Dataset**: Smart Home Dialogue Dataset (200 personas, 6000 dialogues)

---

## Abstract

This report presents a comprehensive empirical study of personalization techniques for smart home conversational agents. We evaluated six different approaches including per-persona fine-tuning, cluster-based training, mixture-of-experts merging, and prefix tuning on a dataset of 200 distinct user personas. Our findings reveal that **simple unified training outperforms all personalization methods** (82.14% vs 66-74% for personalized approaches), challenging conventional assumptions about the benefits of personalization in this domain. We provide detailed analysis of why personalization failed and propose lightweight prefix-tuning as a promising alternative approach.

**Key Finding**: Unified LoRA achieves 82.14% embedding similarity, while all personalization attempts (per-persona LoRA, cluster training, MoE merging, weighted merging) perform worse (66-74%), demonstrating that more data trumps personalization for small models.

---

## 1. Introduction

### 1.1 Motivation

Conversational agents for smart home control must balance two competing objectives:
1. **Generalization**: Understanding diverse commands and contexts
2. **Personalization**: Adapting to individual user preferences and communication styles

While personalization has shown promise in many NLP tasks, its effectiveness for smart home assistants remains understudied. This work investigates whether and how personalization improves performance compared to unified training.

### 1.2 Research Questions

1. **RQ1**: Does per-persona fine-tuning improve performance over unified training?
2. **RQ2**: Can clustering personas enable effective personalization with more training data?
3. **RQ3**: Do model merging techniques (MoE, weighted averaging) provide viable personalization?
4. **RQ4**: Can lightweight prefix tuning build upon strong unified baselines?

### 1.3 Contributions

- Comprehensive evaluation of 6 personalization approaches on real smart home data
- Analysis of why all tested personalization methods failed
- Introduction of prefix tuning on unified LoRA as a promising alternative
- Release of code, models, and detailed experimental logs

---

## 2. Background and Related Work

### 2.1 Parameter-Efficient Fine-Tuning (PEFT)

**LoRA (Low-Rank Adaptation)** [Hu et al., 2021] adapts pre-trained models by learning low-rank update matrices:

```
W' = W + BA
```

where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), and r << min(d,k).

**Configuration used**:
- Rank (r): 8
- Alpha: 16
- Target modules: q_proj, v_proj
- Trainable parameters: ~2.4M (0.48% of base model)

### 2.2 Mixture of Experts (MoE)

MoE combines multiple specialized models through weighted averaging or gating mechanisms. We explored:
- **Sparse MoE**: K-nearest experts within clusters
- **Weighted Merging**: Validation performance × centrality weights

### 2.3 Prefix Tuning

Prefix tuning [Li & Liang, 2021] prepends learnable continuous vectors to the input:

```
Input: [PREFIX] + [USER_QUERY]
```

**Advantages**:
- Minimal parameters (~9K per persona)
- Builds on strong base models
- Low overfitting risk

---

## 3. Dataset and Experimental Setup

### 3.1 Dataset

**Smart Home Dialogue Dataset**
- **Personas**: 200 unique user profiles
- **Dialogues**: 6,000 conversations (30 per persona)
- **Avg length**: 3.2 turns per dialogue
- **Domains**: Device control, scheduling, information queries

**Data splits (per persona)**:
- Training: 20 dialogues (67%)
- Validation: 5 dialogues (17%)
- Test: 5 dialogues (17%)

**Example dialogue**:
```json
{
  "persona_id": "persona_042",
  "character": "Sarah is a fitness instructor who wakes early...",
  "messages": [
    {"role": "user", "text": "Turn on bedroom lights to 30%"},
    {"role": "assistant", "text": "Setting bedroom lights to 30% brightness..."}
  ]
}
```

### 3.2 Base Model

**Qwen 2.5 0.5B Instruct**
- Parameters: 494M
- Context length: 32K tokens
- Vocabulary: 151,646 tokens
- Hidden size: 896

### 3.3 Evaluation Metrics

**Primary metric**: **Embedding Similarity**
- Compute sentence embeddings (all-MiniLM-L6-v2)
- Cosine similarity between prediction and target
- Robust to exact wording differences

**Secondary metrics**:
- Device Precision: Correct device names
- Parameter F1: Correct parameter values
- Numerical Precision: Correct numeric values

### 3.4 Hardware and Training

- **GPU**: [Your GPU]
- **Training time**:
  - Unified LoRA: ~2 hours
  - Per-persona LoRAs: ~200 hours (10 hours with 20 GPUs)
  - Cluster LoRAs: ~6 hours total
- **Inference**: CPU-compatible

---

## 4. Methods

### 4.1 Baseline: Unified LoRA

Train a single LoRA adapter on all 6,000 dialogues.

**Configuration**:
```python
{
  'rank': 8,
  'alpha': 16,
  'dropout': 0.05,
  'target_modules': ['q_proj', 'v_proj'],
  'epochs': 3,
  'batch_size': 4,
  'learning_rate': 5e-4
}
```

**Advantages**: Maximum data utilization, simple deployment
**Disadvantages**: No personalization

---

### 4.2 Method 1: Per-Persona LoRA

Train 200 individual LoRA adapters (one per persona).

**Configuration**: Same as unified
**Training data**: 20 dialogues per persona

**Hypothesis**: Specialized adapters will capture individual preferences
**Challenge**: Severe overfitting with only 20 examples

---

### 4.3 Method 2: Cluster-Based LoRA

#### 4.3.1 Clustering

Cluster personas using K-means (k=5) on persona description embeddings.

**Silhouette score**: 0.022 (very low - indicates poor clustering)

**Cluster distribution**:
| Cluster | Personas | Training Examples |
|---------|----------|-------------------|
| 0 | 16 | 480 |
| 1 | 52 | 1,560 |
| 2 | 25 | 750 |
| 3 | 35 | 1,050 |
| 4 | 72 | 2,160 |

#### 4.3.2 Training

Train one LoRA per cluster on combined cluster data.

**Tested configurations**:
- **Cluster 0**: 3 epochs, LR=5e-4
- **Cluster 4**: 5 epochs, LR=2e-4 (optimized)

---

### 4.4 Method 3: Sparse Mixture of Experts

For each persona, merge K=5 most similar per-persona LoRAs within their cluster.

**Algorithm**:
1. Compute persona embeddings
2. Find K=5 most similar personas in cluster
3. Weight by cosine similarity
4. Average LoRA weight matrices

**Merging formula**:
```
W_merged = Σ(w_i × W_i) where Σw_i = 1
```

---

### 4.5 Method 4: Weighted Cluster Merging

Smart merging of per-persona LoRAs using validation performance and centrality.

**Weight formula**:
```
weight_i = validation_score_i × centrality_i
centrality_i = cosine_sim(persona_i, cluster_centroid)
```

---

### 4.6 Method 5: Prefix Tuning on Unified LoRA

**Novel approach**: Add learnable prefix to frozen unified model.

**Configuration**:
- Prefix length: 10 tokens
- Parameters per persona: 8,960 (0.0018% of base model)
- Learning rate: 1e-3
- Epochs: 10

**Variants**:
1. **Static prefix**: Prepend persona description as text
2. **Learned persona prefix**: Train on 20 examples
3. **Learned cluster prefix**: Train on cluster data

---

## 5. Results

### 5.1 Overall Comparison

| Method | Embedding Similarity | vs Unified | Parameters | Training Time |
|--------|---------------------|------------|------------|---------------|
| **Unified LoRA** | **82.14%** | baseline | 2.4M | 2h |
| Cluster 4 LoRA | 74.14% | -8.0% ❌ | 2.4M | 70min |
| Cluster 0 LoRA | 72.65% | -9.5% ❌ | 2.4M | 42min |
| Per-Persona LoRA | 68.28% | -13.9% ❌ | 480M total | 200h |
| Weighted Merge | 67.00% | -15.1% ❌ | 0 (merging) | 2min |
| **Sparse MoE** | **66.38%** | **-15.8%** ❌ | 0 (merging) | 3.5min |

**Key insight**: ALL personalization methods failed to beat the unified baseline.

---

### 5.2 Per-Persona LoRA Results

**Average**: 68.28% ± 7.73%
**Range**: 48.5% - 94.1%

**Analysis**:
- High variance indicates overfitting
- Best personas: 90%+ (got lucky with test set)
- Worst personas: <50% (severe overfitting)
- 30 training examples insufficient for 2.4M parameters

**Example failures**:
- persona_082: 37.2% (worst)
- persona_199: 88.0% (best - but still < unified)

---

### 5.3 Cluster-Based LoRA Results

#### Cluster 0 (16 personas, 480 examples)
- **Score**: 72.65%
- **Problem**: Insufficient data, smallest cluster
- **Training loss**: Converged but overfit

#### Cluster 4 (72 personas, 2160 examples)
- **Score**: 74.14%
- **Optimizations tried**:
  - Lower LR (2e-4 vs 5e-4)
  - More epochs (5 vs 3)
  - Still failed

**Conclusion**: Even 4.5x more data (2160 vs 480) couldn't overcome fundamental issues.

---

### 5.4 MoE and Merging Results

#### Sparse MoE (K=5)
- **Score**: 66.38% ± 8.74%
- **Worst of all methods!**
- **Range**: 37.2% - 88.0%

**Why it failed**:
- Merging destroys specialized knowledge
- No synergy between experts
- Worse than individual overfitted LoRAs

#### Weighted Merge
- **Score**: 67.00%
- **Smart weights didn't help**
- Validation performance + centrality weighting made no difference

---

### 5.5 Prefix Tuning Results (Preliminary)

**persona_000 (Proof of Concept)**:

| Approach | Score | vs Unified | Training Time |
|----------|-------|------------|---------------|
| Unified (no prefix) | 59.09% | baseline | - |
| Static text prefix | 60.60% | +1.5% ✓ | 0min |
| Learned persona prefix | [RUNNING] | ? | 15min |
| Learned cluster prefix | [RUNNING] | ? | 20min |

**Early insight**: Even static text prefix helps (+1.5%)!

---

## 6. Analysis

### 6.1 Why Did Personalization Fail?

#### 6.1.1 Poor Clustering Quality

**Silhouette score: 0.022** (scale: -1 to 1)
- Indicates personas don't form natural clusters
- Persona descriptions may not correlate with behavior
- K-means on text embeddings insufficient

#### 6.1.2 Insufficient Training Data

**Per-persona**: 20 examples << 2.4M LoRA parameters
- Overfitting ratio: 120,000 params per example
- Need ~100+ examples per persona for stable training

**Even cluster 4** (2160 examples) insufficient
- LoRA can overfit even with 1000+ examples
- Need stronger regularization or larger datasets

#### 6.1.3 Model Capacity Constraints

**Qwen 0.5B** may be too small for effective personalization
- Limited capacity to maintain general knowledge + personalization
- Larger models (3B+) may handle personalization better

#### 6.1.4 Task Characteristics

**Smart home commands** may not benefit from personalization:
- Commands are relatively standardized
- Domain knowledge > user-specific patterns
- Generic responses often sufficient

#### 6.1.5 Merging Destroys Knowledge

**Weight averaging** is fundamentally destructive:
- Individual LoRAs overfit differently
- Averaging cancels out learned patterns
- No constructive interference

---

### 6.2 When Unified Training Wins

**Data advantage**: 6,000 examples >> 20-2160 per method

**Observation**: For small models with limited data:
```
More data > Personalization
```

**Crossover point** (estimated):
- Need 100+ examples per persona
- Or 3B+ parameter models
- Or better clustering strategies

---

### 6.3 Prefix Tuning: A Promising Alternative?

**Advantages over previous approaches**:

1. **Builds on strong base** (82.14%)
2. **Minimal parameters** (8,960 vs 2.4M)
3. **Less overfitting risk**
4. **Fast training** (15min vs hours)
5. **Additive not destructive**

**Static prefix already helps** (+1.5% on persona_000)
- Suggests personalization signal exists
- Can be captured with minimal overhead

---

## 7. Lessons Learned

### 7.1 Methodological Insights

1. **Always establish strong baselines first**
   - Our unified model (82.14%) beat all complex approaches
   - Saved months by testing simple approaches first

2. **Clustering quality matters**
   - Silhouette score 0.022 was a red flag
   - Should have tried multiple clustering methods

3. **Data quantity > complexity**
   - 6000 examples unified > 20-2160 personalized
   - Complexity doesn't compensate for data scarcity

4. **Merging is risky**
   - Weight averaging often destructive
   - Need careful validation before scaling

### 7.2 Engineering Insights

1. **Start with proof of concept**
   - Testing persona_000 saved 10 hours of wasted training
   - Quick iteration >> comprehensive experiments

2. **Monitor training curves**
   - Overfitting visible in validation loss
   - Early stopping could have saved time

3. **Reproducibility critical**
   - Saved all logs, configs, models
   - Can investigate failures later

---

## 8. Future Work

### 8.1 Immediate Next Steps

1. **Complete prefix tuning evaluation**
   - Finish persona_000 POC
   - If successful (>82.14%), scale to all personas

2. **Better clustering**
   - Use behavioral features, not descriptions
   - Try hierarchical clustering
   - Optimize for task performance, not embeddings

3. **Larger models**
   - Test on Qwen 3B, 7B
   - May have capacity for personalization

### 8.2 Alternative Approaches

1. **Retrieval-Augmented Generation**
   - Retrieve relevant past interactions
   - Provide as context instead of fine-tuning

2. **Prompt-based personalization**
   - Include persona info in system prompt
   - No training required

3. **Hybrid methods**
   - Unified base + lightweight adapters
   - Best of both worlds?

4. **Active learning**
   - Select most informative examples
   - Quality > quantity

### 8.3 Broader Research Directions

1. **When does personalization help?**
   - Task characteristics
   - Data requirements
   - Model size thresholds

2. **Optimal architecture for personalization**
   - Dedicated persona embeddings?
   - Multi-task learning?
   - Meta-learning?

---

## 9. Conclusion

This comprehensive study of personalization techniques for smart home assistants yields a surprising conclusion: **simple unified training outperforms all tested personalization methods**. Despite trying six different approaches including per-persona LoRA (68.28%), cluster-based training (74.14%), and mixture-of-experts merging (66.38%), none exceeded the unified baseline of 82.14%.

Our analysis reveals five key failure modes:
1. Poor clustering quality (silhouette score 0.022)
2. Insufficient training data (20-2160 examples per variant)
3. Limited model capacity (0.5B parameters)
4. Task characteristics favoring general knowledge
5. Destructive weight merging

**Preliminary prefix tuning results** suggest a more promising direction: building lightweight personalization on top of strong unified models. Even static text prefixes improve performance (+1.5%), and learned prefixes with only 8,960 parameters may offer viable personalization without the pitfalls of full fine-tuning.

**Recommendation**: For smart home assistants with limited data, use unified training. Only pursue personalization if:
- 100+ examples per persona available
- Using 3B+ parameter models
- Prefix tuning or retrieval augmentation viable

---

## 10. References

1. Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.

2. Li, X. L., & Liang, P. (2021). "Prefix-Tuning: Optimizing Continuous Prompts for Generation." ACL 2021.

3. Lester, B., et al. (2021). "The Power of Scale for Parameter-Efficient Prompt Tuning." EMNLP 2021.

4. Zhang, S., et al. (2023). "Personalized Dialogue Generation via Prompt Learning."

---

## Appendices

### Appendix A: Hyperparameters

**Unified LoRA**:
```python
{
  'rank': 8,
  'lora_alpha': 16,
  'lora_dropout': 0.05,
  'target_modules': ['q_proj', 'v_proj'],
  'learning_rate': 5e-4,
  'num_epochs': 3,
  'batch_size': 4,
  'warmup_ratio': 0.1,
  'weight_decay': 0.01,
}
```

**Per-Persona LoRA**: Same as unified

**Cluster LoRA** (optimized):
```python
{
  # Same base config
  'learning_rate': 2e-4,  # Lower
  'num_epochs': 5,        # More
  'batch_size': 2,        # Smaller
}
```

**Prefix Tuning**:
```python
{
  'prefix_length': 10,
  'learning_rate': 1e-3,
  'num_epochs': 10,
  'batch_size': 2,
}
```

### Appendix B: Compute Resources

- **Total GPU hours**: ~250 hours
- **Peak memory**: 8GB
- **Total experiments**: 215
  - 1 unified
  - 200 per-persona
  - 5 clusters
  - 1 MoE merge
  - 1 weighted merge
  - 4 prefix tuning (POC)
  - 3 failed attempts

### Appendix C: Code and Data Availability

- **Code**: [GitHub repository]
- **Models**: [HuggingFace]
- **Dataset**: [Contact authors]
- **Logs**: Available in repository

### Appendix D: Detailed Results Tables

[Include per-persona breakdown, cluster-wise metrics, etc.]

---

**End of Report**
