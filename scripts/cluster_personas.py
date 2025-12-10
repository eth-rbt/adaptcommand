"""
Strategy 1: Cluster-based Personalization

Group personas by similarity, train cluster-level adapters with more data.
This gives ~600 examples per cluster (20 personas × 30 examples) instead of 30.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

def load_personas():
    """Load all persona descriptions"""
    with open('data/cleaned/dialogs_clean.jsonl') as f:
        dialogues = [json.loads(line) for line in f]

    # Get unique personas and their descriptions
    personas = {}
    for d in dialogues:
        if d['persona_id'] not in personas:
            personas[d['persona_id']] = d['character']

    return personas

def cluster_personas(personas, n_clusters_range=range(5, 31, 5)):
    """
    Cluster personas using their semantic embeddings.
    Try different numbers of clusters to find optimal.
    """

    # Encode persona descriptions
    print("Encoding persona descriptions...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    persona_ids = sorted(personas.keys())
    descriptions = [personas[pid] for pid in persona_ids]
    embeddings = model.encode(descriptions, show_progress_bar=True)

    # Try different numbers of clusters
    results = []
    for n_clusters in n_clusters_range:
        print(f"\nTrying {n_clusters} clusters...")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Calculate silhouette score
        score = silhouette_score(embeddings, cluster_labels)

        # Calculate cluster sizes
        cluster_sizes = [np.sum(cluster_labels == i) for i in range(n_clusters)]

        results.append({
            'n_clusters': n_clusters,
            'silhouette_score': score,
            'cluster_sizes': cluster_sizes,
            'min_size': min(cluster_sizes),
            'max_size': max(cluster_sizes),
            'labels': cluster_labels,
            'kmeans': kmeans
        })

        print(f"  Silhouette score: {score:.4f}")
        print(f"  Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, avg={np.mean(cluster_sizes):.1f}")

    # Find best by silhouette score
    best = max(results, key=lambda x: x['silhouette_score'])
    print(f"\nBest configuration: {best['n_clusters']} clusters (silhouette={best['silhouette_score']:.4f})")

    # Create cluster map
    cluster_map = {
        persona_ids[i]: int(best['labels'][i])
        for i in range(len(persona_ids))
    }

    # Save cluster map
    output_dir = Path('data/splits')
    output_dir.mkdir(exist_ok=True, parents=True)

    with open(output_dir / 'cluster_map.json', 'w') as f:
        json.dump({
            'n_clusters': best['n_clusters'],
            'silhouette_score': float(best['silhouette_score']),
            'cluster_map': cluster_map,
            'cluster_sizes': [int(x) for x in best['cluster_sizes']]
        }, f, indent=2)

    print(f"\nSaved cluster map to {output_dir / 'cluster_map.json'}")

    # Plot results
    plot_clustering_analysis(results, embeddings, best, persona_ids, personas)

    return cluster_map, best

def plot_clustering_analysis(results, embeddings, best, persona_ids, personas):
    """Visualize clustering results"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Silhouette scores vs number of clusters
    ax = axes[0, 0]
    n_clusters = [r['n_clusters'] for r in results]
    scores = [r['silhouette_score'] for r in results]
    ax.plot(n_clusters, scores, 'o-', linewidth=2, markersize=8)
    ax.axvline(best['n_clusters'], color='red', linestyle='--', alpha=0.5, label='Best')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Clustering Quality vs Number of Clusters')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. Cluster size distribution (best config)
    ax = axes[0, 1]
    ax.bar(range(best['n_clusters']), best['cluster_sizes'])
    ax.axhline(np.mean(best['cluster_sizes']), color='red', linestyle='--', alpha=0.5, label='Mean')
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Number of Personas')
    ax.set_title(f'Cluster Sizes (n={best["n_clusters"]})')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()

    # 3. PCA visualization of clusters
    from sklearn.decomposition import PCA
    ax = axes[1, 0]
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)

    scatter = ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=best['labels'],
        cmap='tab20',
        alpha=0.6,
        s=50
    )
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('Persona Clusters (PCA Projection)')
    plt.colorbar(scatter, ax=ax, label='Cluster ID')

    # 4. Training data per cluster
    ax = axes[1, 1]
    examples_per_persona = 30  # training examples
    cluster_training_examples = [size * examples_per_persona for size in best['cluster_sizes']]
    ax.bar(range(best['n_clusters']), cluster_training_examples)
    ax.axhline(examples_per_persona, color='red', linestyle='--', alpha=0.5, label='Single persona')
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Training Examples')
    ax.set_title('Training Data per Cluster')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()

    plt.tight_layout()
    output_path = Path('results/figures/clustering_analysis.png')
    output_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved clustering analysis to {output_path}")
    plt.close()

if __name__ == '__main__':
    print("Clustering Personas for Cluster-Level Personalization")
    print("=" * 80)

    personas = load_personas()
    print(f"Loaded {len(personas)} personas")

    cluster_map, best = cluster_personas(personas)

    print("\nNext steps:")
    print("1. Train cluster-level LoRA adapters (one per cluster)")
    print("2. Each cluster has ~600-900 training examples (vs 30 for per-persona)")
    print("3. Optionally: add tiny per-user deltas on top of cluster adapter")
