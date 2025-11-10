#!/usr/bin/env python3
"""Plot the importance (explained variance) of PCA components."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# Load characters
characters_path = Path("data/raw/characters.jsonl")
print("Loading personas...")

personas = []
with open(characters_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        personas.append(data['character'])

print(f"Loaded {len(personas)} personas")

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Embed all personas
print("Generating embeddings...")
embeddings = model.encode(personas, show_progress_bar=True, convert_to_numpy=True)
print(f"Embedding shape: {embeddings.shape}")

# Apply PCA with more components to see the full variance distribution
n_components = min(50, len(personas))  # Get up to 50 components or max available
print(f"Applying PCA with {n_components} components...")
pca = PCA(n_components=n_components)
embeddings_pca = pca.fit_transform(embeddings)

# Get explained variance ratio
explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

print(f"\nTop 10 components explain {cumulative_var[9]*100:.2f}% of variance")
print(f"Top 20 components explain {cumulative_var[19]*100:.2f}% of variance")
print(f"All {n_components} components explain {cumulative_var[-1]*100:.2f}% of variance")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Bar chart of explained variance for top 20 components
components_to_show = min(20, n_components)
x_pos = np.arange(1, components_to_show + 1)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, components_to_show))

ax1.bar(x_pos, explained_var[:components_to_show] * 100,
        color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
ax1.set_xlabel('Principal Component', fontsize=12)
ax1.set_ylabel('Explained Variance (%)', fontsize=12)
ax1.set_title('Explained Variance by Principal Component (Top 20)',
              fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f'PC{i}' for i in x_pos], rotation=45, ha='right')
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels on top of bars for top 5
for i in range(min(5, components_to_show)):
    ax1.text(i+1, explained_var[i]*100 + 0.1, f'{explained_var[i]*100:.2f}%',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 2: Cumulative explained variance
ax2.plot(x_pos, cumulative_var[:components_to_show] * 100,
         marker='o', linewidth=2, markersize=6, color='steelblue')
ax2.axhline(y=50, color='red', linestyle='--', linewidth=1.5,
            label='50% variance threshold', alpha=0.7)
ax2.axhline(y=80, color='orange', linestyle='--', linewidth=1.5,
            label='80% variance threshold', alpha=0.7)
ax2.axhline(y=95, color='green', linestyle='--', linewidth=1.5,
            label='95% variance threshold', alpha=0.7)
ax2.set_xlabel('Number of Principal Components', fontsize=12)
ax2.set_ylabel('Cumulative Explained Variance (%)', fontsize=12)
ax2.set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(x_pos)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='lower right', fontsize=10)
ax2.set_ylim([0, 105])

# Add annotations for key milestones
milestones = [50, 80, 95]
for milestone in milestones:
    # Find how many components needed for this milestone
    n_comp_needed = np.argmax(cumulative_var >= (milestone/100)) + 1
    if n_comp_needed <= components_to_show:
        ax2.annotate(f'{n_comp_needed} PCs',
                    xy=(n_comp_needed, cumulative_var[n_comp_needed-1]*100),
                    xytext=(5, -15),
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', lw=1))

plt.tight_layout()

# Save plot
output_path = Path("results/baseline/pca_component_importance.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")
plt.close()

# Print detailed statistics
print("\n" + "="*60)
print("PCA COMPONENT IMPORTANCE ANALYSIS")
print("="*60)

print(f"\nTop 10 Components:")
for i in range(min(10, n_components)):
    print(f"  PC{i+1}: {explained_var[i]*100:.3f}% (cumulative: {cumulative_var[i]*100:.2f}%)")

# Find how many components needed for different thresholds
for threshold in [50, 80, 90, 95, 99]:
    n_needed = np.argmax(cumulative_var >= (threshold/100)) + 1
    if n_needed < len(cumulative_var):
        print(f"\nComponents needed for {threshold}% variance: {n_needed}")

print("\nDone!")
