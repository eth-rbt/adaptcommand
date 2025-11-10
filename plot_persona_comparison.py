#!/usr/bin/env python3
"""Plot histogram comparison of per-persona results vs baseline."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load the data
per_persona_path = Path("results/baseline/per_persona_results.json")
baseline_path = Path("results/baseline/baseline_results.json")

print("Loading data...")
with open(per_persona_path, 'r') as f:
    per_persona_data = json.load(f)

with open(baseline_path, 'r') as f:
    baseline_data = json.load(f)

# Extract metrics for each persona
embedding_similarities = []
numerical_precisions = []

for persona_id, persona_data in per_persona_data.items():
    if persona_id.startswith('persona_'):
        metrics = persona_data.get('metrics', {})

        emb_sim = metrics.get('embedding_similarity')
        num_prec = metrics.get('numerical_precision')

        if emb_sim is not None:
            embedding_similarities.append(emb_sim)
        if num_prec is not None:
            numerical_precisions.append(num_prec)

# Get baseline values
baseline_emb_sim = baseline_data['metrics']['embedding_similarity']
baseline_num_prec = baseline_data['metrics']['numerical_precision']

print(f"Found {len(embedding_similarities)} personas")
print(f"Baseline embedding similarity: {baseline_emb_sim:.4f}")
print(f"Baseline numerical precision: {baseline_num_prec:.4f}")
print(f"Per-persona embedding similarity: mean={np.mean(embedding_similarities):.4f}, std={np.std(embedding_similarities):.4f}")
print(f"Per-persona numerical precision: mean={np.mean(numerical_precisions):.4f}, std={np.std(numerical_precisions):.4f}")

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Embedding Similarity
ax1 = axes[0]
ax1.hist(embedding_similarities, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
ax1.axvline(baseline_emb_sim, color='red', linestyle='--', linewidth=2,
            label=f'Baseline: {baseline_emb_sim:.4f}')
ax1.axvline(np.mean(embedding_similarities), color='green', linestyle='--', linewidth=2,
            label=f'Per-Persona Mean: {np.mean(embedding_similarities):.4f}')
ax1.set_xlabel('Embedding Similarity', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Distribution of Embedding Similarity Across Personas', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Numerical Precision
ax2 = axes[1]
ax2.hist(numerical_precisions, bins=30, alpha=0.7, color='coral', edgecolor='black')
ax2.axvline(baseline_num_prec, color='red', linestyle='--', linewidth=2,
            label=f'Baseline: {baseline_num_prec:.4f}')
ax2.axvline(np.mean(numerical_precisions), color='green', linestyle='--', linewidth=2,
            label=f'Per-Persona Mean: {np.mean(numerical_precisions):.4f}')
ax2.set_xlabel('Numerical Precision', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Distribution of Numerical Precision Across Personas', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Save the plot
output_path = Path("results/baseline/persona_comparison_histogram.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")
plt.close()

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print("\nEmbedding Similarity:")
print(f"  Min:    {np.min(embedding_similarities):.4f}")
print(f"  Max:    {np.max(embedding_similarities):.4f}")
print(f"  Mean:   {np.mean(embedding_similarities):.4f}")
print(f"  Median: {np.median(embedding_similarities):.4f}")
print(f"  Std:    {np.std(embedding_similarities):.4f}")
print(f"  Baseline: {baseline_emb_sim:.4f}")

print("\nNumerical Precision:")
print(f"  Min:    {np.min(numerical_precisions):.4f}")
print(f"  Max:    {np.max(numerical_precisions):.4f}")
print(f"  Mean:   {np.mean(numerical_precisions):.4f}")
print(f"  Median: {np.median(numerical_precisions):.4f}")
print(f"  Std:    {np.std(numerical_precisions):.4f}")
print(f"  Baseline: {baseline_num_prec:.4f}")
