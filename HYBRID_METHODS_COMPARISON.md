# Hybrid Methods Comparison

## Summary

I've compared the three hybrid/cluster methods you tested. **TL;DR: All three fail to beat unified LoRA (82.14%).**

---

## The Three Methods

### 1. **Hybrid LoRA** (Best of the three, but still fails)
- **Approach**: Unified LoRA (frozen) + Per-Persona LoRA fine-tuning on top
- **Score**: 75.91% (-7.6% vs unified)
- **Training**: Two-stage (unified base + 200 persona adapters)
- **Time**: ~52 GPU hours (2h unified + 50h persona training)
- **Data**: 20 examples per persona

**Why it fails:**
- Benefits from strong unified foundation (82.14%)
- BUT: Per-persona fine-tuning overfits on just 20 examples
- High variance (std: 6.89%) = inconsistent results
- Some personas improve, most get worse
- **Trade-off**: 26x more training time for 7.6% worse performance

---

### 2. **Cluster LoRA** (Middle performer)
- **Approach**: Group personas into 5 clusters, train one LoRA per cluster FROM SCRATCH
- **Score**: 74.14% (-9.7% vs unified)
- **Training**: Single-stage cluster training
- **Time**: ~2 GPU hours (only trained 2 clusters)
- **Data**: 480-2,160 examples per cluster

**Why it fails:**
- Poor clustering quality (silhouette score: 0.022)
- Text similarity ≠ behavioral similarity
- Even largest cluster (2,160 examples) still insufficient vs unified (6,000)
- Loses the benefit of unified's full dataset
- **Trade-off**: Same training time as unified but 9.7% worse

---

### 3. **Weighted Merge** (Worst method)
- **Approach**: Merge K=5 similar per-persona LoRAs using smart weighting
- **Score**: 67.00% (-18.4% vs unified)
- **Training**: Zero training (just merges existing LoRAs)
- **Time**: ~2 minutes (merging only)
- **Data**: Uses pre-trained per-persona LoRAs

**Why it fails:**
- Merging overfitted models is destructive
- Smart weighting (validation score × centrality) doesn't help
- Linear averaging of nonlinear weights fails
- Very high variance (std: 20.11%) = extremely unstable
- Even worse than individual overfitted LoRAs (67% vs 68%)
- **Trade-off**: Fast but useless

---

## Performance Ranking

```
0. Unified (Baseline):  82.14%  ★ WINNER
1. Hybrid LoRA:         75.91%  (-7.6%)
2. Cluster LoRA:        74.14%  (-9.7%)
3. Weighted Merge:      67.00%  (-18.4%)
```

---

## Key Insights

### 1. **All Three Fail to Beat Unified**
- Best hybrid: 75.91% (Hybrid LoRA)
- Unified: 82.14%
- Gap: -7.6% despite 26x more training time

### 2. **Data Quantity Dominates**
```
Unified:        6,000 examples → 82.14%  ✓
Cluster 4:      2,160 examples → 74.14%
Cluster 0:        480 examples → 72.65%
Hybrid:            20 examples → 75.91% (but overfits)
Weighted:     merged weights  → 67.00% (destructive)
```

### 3. **Training Time vs Performance**
```
Method           Time       Score    Efficiency
─────────────────────────────────────────────────
Unified          2h        82.14%   ★ Best
Cluster LoRA     2h        74.14%   Same time, worse
Hybrid LoRA     52h        75.91%   26x time, worse!
Weighted Merge   2min      67.00%   Fast but useless
```

### 4. **Variance = Instability**
```
Unified:         0.0% std  (perfectly consistent)
Cluster:         3.7% std  (moderate)
Hybrid:          6.9% std  (high - inconsistent)
Weighted:       20.1% std  (very high - unstable)
```

---

## Visualization

A comprehensive comparison visualization has been saved to:
**`results/figures/hybrid_methods_comparison.png`**

This includes:
1. **Performance bar chart** with error bars
2. **Distribution box plots** showing variance
3. **Training time vs performance scatter plot**
4. **Detailed comparison table** with all metrics

---

## What Went Wrong?

### Hybrid LoRA
- ✓ Good idea: Build on strong unified foundation
- ✗ Problem: 20 examples insufficient for 2.4M parameters
- ✗ Result: Overfitting destroys gains from unified base

### Cluster LoRA
- ✓ Good idea: More data per cluster than per-persona
- ✗ Problem: Poor clustering (silhouette: 0.022)
- ✗ Problem: Still less data than unified
- ✗ Result: Loses unified's data advantage with no benefit

### Weighted Merge
- ✓ Good idea: Fast, no training needed
- ✗ Problem: Linear averaging destroys learned patterns
- ✗ Problem: Merging overfitted models compounds errors
- ✗ Result: Worst of all methods

---

## When Might Hybrid Methods Work?

Based on these failures, hybrid approaches might succeed with:

1. **More data per persona**: 100+ examples (vs 20)
2. **Larger models**: 3B+ parameters (vs 0.5B) for dual objectives
3. **Better clustering**: Silhouette score > 0.3 (vs 0.022)
4. **Highly personal tasks**: >50% personality-dependent (vs 25%)

**Current situation**: None of these conditions are met!

---

## Recommendation

### For Your Dataset and Task:

**✓ USE: Unified LoRA**
- Score: 82.14%
- Time: 2 hours
- Simple: Single model
- Consistent: 0% variance

**✗ AVOID: All Hybrid Methods**
- Worse performance (-7.6% to -18.4%)
- More complexity (multiple models, routing, merging)
- More training time (2min to 52h)
- Higher variance (inconsistent results)

---

## Files Generated

1. **`results/figures/hybrid_methods_comparison.png`** - Comprehensive visualization
2. **`scripts/compare_hybrid_methods.py`** - Reusable comparison script
3. **`HYBRID_METHODS_COMPARISON.md`** - This summary (you're reading it!)

---

## Bottom Line

**Simple unified training beats sophisticated personalization.**

The data tells a clear story:
- 6,000 unified examples > 20-2,160 personalized examples
- 2 hours unified training > 2 minutes to 52 hours hybrid training
- 82.14% unified score > 67-76% hybrid scores

Don't overthink it. Use unified LoRA.
