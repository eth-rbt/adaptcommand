"""
Sparse MoE-style merging: Top-K experts within cluster

For each persona:
1. Identify their cluster
2. Find K most similar personas WITHIN that cluster
3. Merge those K LoRAs with similarity-based weights

This is more efficient than full MoE (only K experts) but more
personalized than cluster-only merging.
"""

import json
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def compute_persona_similarities(cluster_id, cluster_map_path='data/splits/cluster_map.json'):
    """Compute pairwise similarities within a cluster"""

    # Load cluster map
    with open(cluster_map_path) as f:
        cluster_data = json.load(f)

    cluster_map = cluster_data['cluster_map']

    # Get personas in this cluster
    cluster_personas = [
        p for p, c in cluster_map.items() if c == cluster_id
    ]

    # Load persona descriptions
    with open('data/cleaned/dialogs_clean.jsonl') as f:
        dialogues = [json.loads(line) for line in f]

    persona_descriptions = {}
    for dialogue in dialogues:
        persona_id = dialogue['persona_id']
        if persona_id in cluster_personas and persona_id not in persona_descriptions:
            persona_descriptions[persona_id] = dialogue['character']

    # Encode with sentence transformer
    encoder = SentenceTransformer('all-MiniLM-L6-v2')

    persona_ids = sorted(persona_descriptions.keys())
    descriptions = [persona_descriptions[p] for p in persona_ids]
    embeddings = encoder.encode(descriptions, show_progress_bar=False)

    # Compute similarity matrix (cosine similarity)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarity_matrix = embeddings @ embeddings.T

    return persona_ids, similarity_matrix


def merge_top_k_loras(
    target_persona_id,
    persona_ids,
    similarity_matrix,
    k=5,
    base_model='Qwen/Qwen2.5-0.5B-Instruct',
    lora_dir='models/lora_adapters'
):
    """Merge top-K most similar LoRAs for a target persona"""

    # Find target persona index
    target_idx = persona_ids.index(target_persona_id)

    # Get similarities (exclude self)
    similarities = similarity_matrix[target_idx].copy()
    similarities[target_idx] = -1  # Exclude self

    # Get top-K indices
    top_k_indices = np.argsort(similarities)[-k:]
    top_k_personas = [persona_ids[i] for i in top_k_indices]
    top_k_similarities = similarities[top_k_indices]

    # Normalize similarities to weights
    weights = top_k_similarities / top_k_similarities.sum()

    print(f"\n{target_persona_id} experts:")
    for p, w in zip(top_k_personas, weights):
        print(f"  {p}: {w:.3f}")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(base_model)

    # Merge LoRAs with weighted average
    merged_state_dict = {}

    for persona, weight in zip(top_k_personas, weights):
        lora_path = Path(lora_dir) / persona

        if not lora_path.exists():
            print(f"  Warning: {lora_path} not found, skipping")
            continue

        # Load LoRA
        peft_model = PeftModel.from_pretrained(model, str(lora_path))
        lora_state_dict = peft_model.state_dict()

        # Add weighted parameters
        for name, param in lora_state_dict.items():
            if 'lora' in name:
                if name not in merged_state_dict:
                    merged_state_dict[name] = weight * param.cpu().numpy()
                else:
                    merged_state_dict[name] += weight * param.cpu().numpy()

    return merged_state_dict, top_k_personas, weights


def create_sparse_moe_models(k=5, output_dir='models/moe_sparse_k5'):
    """Create sparse MoE merged models for all personas"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating Sparse MoE models (K={k})")
    print("=" * 80)

    # Load cluster map
    with open('data/splits/cluster_map.json') as f:
        cluster_data = json.load(f)

    cluster_map = cluster_data['cluster_map']
    n_clusters = cluster_data['n_clusters']

    # For each cluster, compute similarities once
    cluster_similarities = {}
    for cluster_id in range(n_clusters):
        print(f"\nComputing similarities for cluster {cluster_id}...")
        persona_ids, similarity_matrix = compute_persona_similarities(cluster_id)
        cluster_similarities[cluster_id] = (persona_ids, similarity_matrix)

    # Save routing information
    routing_info = {}

    # Process each persona
    all_personas = sorted(cluster_map.keys())

    for target_persona in tqdm(all_personas, desc="Creating MoE models"):
        cluster_id = cluster_map[target_persona]
        persona_ids, similarity_matrix = cluster_similarities[cluster_id]

        # Merge top-K LoRAs
        merged_state_dict, experts, weights = merge_top_k_loras(
            target_persona,
            persona_ids,
            similarity_matrix,
            k=k
        )

        # Save merged model
        persona_output_dir = output_dir / target_persona
        persona_output_dir.mkdir(exist_ok=True)

        # Save merged weights
        import torch
        torch.save(merged_state_dict, persona_output_dir / 'merged_lora.pt')

        # Save routing info
        routing_info[target_persona] = {
            'cluster': cluster_id,
            'experts': experts,
            'weights': weights.tolist(),
        }

    # Save routing information
    with open(output_dir / 'routing_info.json', 'w') as f:
        json.dump(routing_info, f, indent=2)

    print(f"\n✓ Saved {len(routing_info)} sparse MoE models to {output_dir}")
    print(f"✓ Routing info saved to {output_dir / 'routing_info.json'}")

    return routing_info


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create sparse MoE merged models')
    parser.add_argument('--k', type=int, default=5,
                       help='Number of experts per persona')

    args = parser.parse_args()

    create_sparse_moe_models(k=args.k)
