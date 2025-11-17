"""
Plot comprehensive comparison between Baseline Individual, Hybrid, and Unified models
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load results
with open('results/personalized/personalized_summary.json') as f:
    baseline_individual = json.load(f)

with open('results/hybrid/hybrid_summary.json') as f:
    hybrid = json.load(f)

with open('results/unified/unified_results.json') as f:
    unified = json.load(f)

# Extract per-persona data
baseline_personas = {p['persona_id']: p for p in baseline_individual['per_persona_metrics']}
hybrid_personas = {p['persona_id']: p for p in hybrid['per_persona_metrics']}

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# ============================================================================
# 1. Overall Comparison Bar Chart
# ============================================================================
ax1 = fig.add_subplot(gs[0, :])

metrics = ['embedding_similarity', 'param_f1', 'device_precision', 'device_recall']
metric_labels = ['Embedding\nSimilarity', 'Param F1', 'Device\nPrecision', 'Device\nRecall']

x = np.arange(len(metrics))
width = 0.25

baseline_vals = [baseline_individual[f'{m}_mean'] for m in metrics]
hybrid_vals = [hybrid[f'{m}_mean'] for m in metrics]
unified_vals = [unified['metrics'][m] for m in metrics]

bars1 = ax1.bar(x - width, baseline_vals, width, label='Baseline Individual', color='#3498db', alpha=0.8)
bars2 = ax1.bar(x, hybrid_vals, width, label='Hybrid (Gentle)', color='#2ecc71', alpha=0.8)
bars3 = ax1.bar(x + width, unified_vals, width, label='Unified', color='#e74c3c', alpha=0.8)

ax1.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_title('Overall Performance Comparison: Baseline Individual vs Hybrid vs Unified',
              fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(metric_labels)
ax1.legend(fontsize=11)
ax1.set_ylim(0, 1.05)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8)

# ============================================================================
# 2. Per-Persona Improvement Histogram (Hybrid vs Baseline)
# ============================================================================
ax2 = fig.add_subplot(gs[1, 0])

# Calculate improvements
improvements_vs_baseline = []
for pid in baseline_personas.keys():
    baseline_emb = baseline_personas[pid]['embedding_similarity']
    hybrid_emb = hybrid_personas[pid]['embedding_similarity']
    improvement = hybrid_emb - baseline_emb
    improvements_vs_baseline.append(improvement)

ax2.hist(improvements_vs_baseline, bins=30, color='#2ecc71', alpha=0.7, edgecolor='black')
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Change')
ax2.axvline(x=np.mean(improvements_vs_baseline), color='blue', linestyle='--', linewidth=2,
            label=f'Mean: {np.mean(improvements_vs_baseline):.4f}')

improved = sum(1 for x in improvements_vs_baseline if x > 0)
degraded = sum(1 for x in improvements_vs_baseline if x < 0)

ax2.set_xlabel('Improvement in Embedding Similarity', fontsize=11, fontweight='bold')
ax2.set_ylabel('Number of Personas', fontsize=11, fontweight='bold')
ax2.set_title(f'Hybrid vs Baseline Individual\n({improved} improved, {degraded} degraded)',
              fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3)

# ============================================================================
# 3. Per-Persona Gap to Unified (Hybrid vs Baseline)
# ============================================================================
ax3 = fig.add_subplot(gs[1, 1])

unified_emb = unified['metrics']['embedding_similarity']

# Calculate gaps
baseline_gaps = []
hybrid_gaps = []

for pid in baseline_personas.keys():
    baseline_gap = unified_emb - baseline_personas[pid]['embedding_similarity']
    hybrid_gap = unified_emb - hybrid_personas[pid]['embedding_similarity']
    baseline_gaps.append(baseline_gap)
    hybrid_gaps.append(hybrid_gap)

ax3.hist(baseline_gaps, bins=30, alpha=0.5, label='Baseline Gap', color='#3498db', edgecolor='black')
ax3.hist(hybrid_gaps, bins=30, alpha=0.5, label='Hybrid Gap', color='#2ecc71', edgecolor='black')

ax3.axvline(x=np.mean(baseline_gaps), color='#3498db', linestyle='--', linewidth=2,
            label=f'Baseline Mean: {np.mean(baseline_gaps):.4f}')
ax3.axvline(x=np.mean(hybrid_gaps), color='#2ecc71', linestyle='--', linewidth=2,
            label=f'Hybrid Mean: {np.mean(hybrid_gaps):.4f}')

ax3.set_xlabel('Gap to Unified Performance', fontsize=11, fontweight='bold')
ax3.set_ylabel('Number of Personas', fontsize=11, fontweight='bold')
ax3.set_title('How Close Are Personas to Unified Performance?', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(axis='y', alpha=0.3)

# ============================================================================
# 4. Scatter: Baseline vs Hybrid Performance
# ============================================================================
ax4 = fig.add_subplot(gs[2, 0])

baseline_emb_list = [baseline_personas[pid]['embedding_similarity'] for pid in sorted(baseline_personas.keys())]
hybrid_emb_list = [hybrid_personas[pid]['embedding_similarity'] for pid in sorted(hybrid_personas.keys())]

# Color by improvement
colors = ['green' if h > b else 'red' for h, b in zip(hybrid_emb_list, baseline_emb_list)]

ax4.scatter(baseline_emb_list, hybrid_emb_list, alpha=0.6, c=colors, s=50)
ax4.plot([0.5, 1], [0.5, 1], 'k--', lw=2, label='y=x (no change)')
ax4.plot([0.5, 1], [unified_emb, unified_emb], 'r--', lw=2, label=f'Unified: {unified_emb:.3f}')

ax4.set_xlabel('Baseline Individual Embedding Similarity', fontsize=11, fontweight='bold')
ax4.set_ylabel('Hybrid Embedding Similarity', fontsize=11, fontweight='bold')
ax4.set_title('Persona-by-Persona Comparison\n(Green=Improved, Red=Degraded)',
              fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)
ax4.set_xlim(0.5, 1.0)
ax4.set_ylim(0.5, 1.0)

# ============================================================================
# 5. Distribution Box Plots
# ============================================================================
ax5 = fig.add_subplot(gs[2, 1])

data_for_box = [
    baseline_emb_list,
    hybrid_emb_list,
    [unified_emb] * len(baseline_emb_list)  # Unified is constant
]

bp = ax5.boxplot(data_for_box, labels=['Baseline\nIndividual', 'Hybrid', 'Unified'],
                 patch_artist=True, showmeans=True)

colors_box = ['#3498db', '#2ecc71', '#e74c3c']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax5.set_ylabel('Embedding Similarity', fontsize=11, fontweight='bold')
ax5.set_title('Distribution Comparison', fontsize=12, fontweight='bold')
ax5.grid(axis='y', alpha=0.3)
ax5.set_ylim(0.5, 1.0)

# Add mean values as text
for i, (data, label) in enumerate(zip(data_for_box, ['Baseline', 'Hybrid', 'Unified']), 1):
    mean_val = np.mean(data)
    ax5.text(i, mean_val, f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('results/hybrid/comprehensive_comparison.png', dpi=300, bbox_inches='tight')
print(f"[OK] Saved comprehensive comparison to results/hybrid/comprehensive_comparison.png")

# ============================================================================
# Print Summary Statistics
# ============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE COMPARISON SUMMARY")
print("="*80)

print("\nEmbedding Similarity:")
print(f"  Baseline Individual: {baseline_individual['embedding_similarity_mean']:.4f} ± {baseline_individual['embedding_similarity_std']:.4f}")
print(f"  Hybrid (Gentle):     {hybrid['embedding_similarity_mean']:.4f} ± {hybrid['embedding_similarity_std']:.4f}")
print(f"  Unified:             {unified_emb:.4f}")

print("\nImprovements:")
print(f"  Hybrid vs Baseline:  {(hybrid['embedding_similarity_mean'] - baseline_individual['embedding_similarity_mean']):.4f} ({((hybrid['embedding_similarity_mean'] - baseline_individual['embedding_similarity_mean']) / baseline_individual['embedding_similarity_mean'] * 100):.1f}%)")
print(f"  Personas improved:   {improved}/{len(improvements_vs_baseline)} ({100*improved/len(improvements_vs_baseline):.1f}%)")
print(f"  Personas degraded:   {degraded}/{len(improvements_vs_baseline)} ({100*degraded/len(improvements_vs_baseline):.1f}%)")

print("\nGap to Unified:")
print(f"  Baseline gap:        {np.mean(baseline_gaps):.4f}")
print(f"  Hybrid gap:          {np.mean(hybrid_gaps):.4f}")
print(f"  Gap reduction:       {(np.mean(baseline_gaps) - np.mean(hybrid_gaps)):.4f} ({((np.mean(baseline_gaps) - np.mean(hybrid_gaps)) / np.mean(baseline_gaps) * 100):.1f}%)")

print("\nParam F1:")
print(f"  Baseline Individual: {baseline_individual['param_f1_mean']:.4f}")
print(f"  Hybrid (Gentle):     {hybrid['param_f1_mean']:.4f}")
print(f"  Unified:             {unified['metrics']['param_f1']:.4f}")

print("\n" + "="*80)
