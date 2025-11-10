#!/usr/bin/env python3
"""Visualize persona embeddings using PCA."""

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

# Load per-persona results to get embedding similarity scores
print("Loading per-persona results...")
per_persona_path = Path("results/baseline/per_persona_results.json")
with open(per_persona_path, 'r') as f:
    per_persona_data = json.load(f)

# Extract embedding similarities in order
embedding_similarities = []
for i in range(len(personas)):
    persona_id = f"persona_{i:03d}"
    if persona_id in per_persona_data:
        emb_sim = per_persona_data[persona_id]['metrics']['embedding_similarity']
        embedding_similarities.append(emb_sim)
    else:
        embedding_similarities.append(None)  # Handle missing data

print(f"Loaded embedding similarities for {len([x for x in embedding_similarities if x is not None])} personas")

# Load embedding model (same one used in evaluation)
print("Loading embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Embed all personas
print("Generating embeddings...")
embeddings = model.encode(personas, show_progress_bar=True, convert_to_numpy=True)
print(f"Embedding shape: {embeddings.shape}")

# Apply PCA to reduce to 2D
print("Applying PCA...")
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# Calculate explained variance
explained_var = pca.explained_variance_ratio_
print(f"PC1 explains {explained_var[0]*100:.2f}% of variance")
print(f"PC2 explains {explained_var[1]*100:.2f}% of variance")
print(f"Total explained variance: {sum(explained_var)*100:.2f}%")

# Create scatter plot
fig, ax = plt.subplots(figsize=(12, 10))

# Plot all personas colored by embedding similarity
scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                     c=embedding_similarities,
                     cmap='RdYlGn',  # Red-Yellow-Green colormap (red=low, green=high)
                     alpha=0.7,
                     s=60,
                     edgecolors='black',
                     linewidth=0.5,
                     vmin=min(embedding_similarities),
                     vmax=max(embedding_similarities))

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax, label='Embedding Similarity')

# Labels and title
ax.set_xlabel(f'PC1 ({explained_var[0]*100:.2f}% variance)', fontsize=12)
ax.set_ylabel(f'PC2 ({explained_var[1]*100:.2f}% variance)', fontsize=12)
ax.set_title('2D PCA Visualization of 200 Personas\nColored by Embedding Similarity Performance',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Find best and worst performing personas
best_idx = np.argmax(embedding_similarities)
worst_idx = np.argmin(embedding_similarities)

# Add annotations for best and worst performers
for idx, label_text in [(best_idx, 'Best'), (worst_idx, 'Worst')]:
    name = personas[idx].split()[0]
    emb_sim = embedding_similarities[idx]
    ax.annotate(f'{label_text}: {name}\n(sim={emb_sim:.3f})',
                xy=(embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=9,
                alpha=0.9,
                bbox=dict(boxstyle='round,pad=0.4',
                         facecolor='lightgreen' if idx == best_idx else 'lightcoral',
                         alpha=0.7),
                arrowprops=dict(arrowstyle='->', lw=1.5))

plt.tight_layout()

# Save plot
output_path = Path("results/baseline/persona_embeddings_pca.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")
plt.close()

# Print some statistics about the distribution
print("\n" + "="*60)
print("EMBEDDING STATISTICS")
print("="*60)
print(f"\nOriginal embedding dimension: {embeddings.shape[1]}")
print(f"PCA components: 2")
print(f"Total variance explained: {sum(explained_var)*100:.2f}%")
print(f"\nPC1 statistics:")
print(f"  Min:  {embeddings_2d[:, 0].min():.4f}")
print(f"  Max:  {embeddings_2d[:, 0].max():.4f}")
print(f"  Mean: {embeddings_2d[:, 0].mean():.4f}")
print(f"  Std:  {embeddings_2d[:, 0].std():.4f}")
print(f"\nPC2 statistics:")
print(f"  Min:  {embeddings_2d[:, 1].min():.4f}")
print(f"  Max:  {embeddings_2d[:, 1].max():.4f}")
print(f"  Mean: {embeddings_2d[:, 1].mean():.4f}")
print(f"  Std:  {embeddings_2d[:, 1].std():.4f}")

# Calculate and display pairwise distances to show diversity
from scipy.spatial.distance import pdist, squareform
distances = pdist(embeddings_2d, metric='euclidean')
print(f"\nPairwise distances in 2D space:")
print(f"  Min distance:  {distances.min():.4f}")
print(f"  Max distance:  {distances.max():.4f}")
print(f"  Mean distance: {distances.mean():.4f}")
print(f"  Std distance:  {distances.std():.4f}")

# Print embedding similarity statistics
print("\n" + "="*60)
print("EMBEDDING SIMILARITY PERFORMANCE")
print("="*60)
emb_sims_array = np.array(embedding_similarities)
print(f"\nEmbedding Similarity:")
print(f"  Min:    {emb_sims_array.min():.4f} (Persona {np.argmin(emb_sims_array)}: {personas[np.argmin(emb_sims_array)].split()[0]})")
print(f"  Max:    {emb_sims_array.max():.4f} (Persona {np.argmax(emb_sims_array)}: {personas[np.argmax(emb_sims_array)].split()[0]})")
print(f"  Mean:   {emb_sims_array.mean():.4f}")
print(f"  Median: {np.median(emb_sims_array):.4f}")
print(f"  Std:    {emb_sims_array.std():.4f}")

print("\nDone!")
