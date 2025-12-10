"""
Strategy 1: Weighted merge of per-persona LoRAs within cluster

Instead of training from scratch, intelligently merge existing per-persona LoRAs
using weights based on validation performance and similarity.
"""

import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import argparse


def load_persona_val_scores():
    """Load validation scores for each persona"""
    try:
        with open('results/personalized/personalized_summary.json') as f:
            data = json.load(f)
            per_persona_metrics = data['per_persona_metrics']

            scores = {}
            for persona_data in per_persona_metrics:
                persona_id = persona_data['persona_id']
                # Use embedding similarity as quality metric
                scores[persona_id] = persona_data['embedding_similarity']

            return scores
    except:
        # If no validation scores, use uniform weights
        return {}


def compute_cluster_weights(cluster_id, cluster_map_path='data/splits/cluster_map.json'):
    """Compute smart weights for merging personas in cluster"""

    # Load cluster map
    with open(cluster_map_path) as f:
        cluster_data = json.load(f)

    cluster_map = cluster_data['cluster_map']

    # Get personas in this cluster
    cluster_personas = [p for p, c in cluster_map.items() if c == cluster_id]

    print(f"Cluster {cluster_id}: {len(cluster_personas)} personas")

    # Load persona embeddings
    encoder = SentenceTransformer('all-MiniLM-L6-v2')

    with open('data/cleaned/dialogs_clean.jsonl') as f:
        dialogues = [json.loads(line) for line in f]

    persona_descriptions = {}
    for dialogue in dialogues:
        persona_id = dialogue['persona_id']
        if persona_id in cluster_personas and persona_id not in persona_descriptions:
            persona_descriptions[persona_id] = dialogue['character']

    # Encode personas
    persona_ids = sorted(persona_descriptions.keys())
    descriptions = [persona_descriptions[p] for p in persona_ids]
    embeddings = encoder.encode(descriptions, show_progress_bar=False)

    # Compute cluster centroid
    centroid = embeddings.mean(axis=0)

    # Compute similarity to centroid (how "central" each persona is)
    similarities = embeddings @ centroid / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(centroid))

    # Load validation scores
    val_scores = load_persona_val_scores()

    # Compute weights
    weights = {}
    for persona_id, similarity in zip(persona_ids, similarities):
        val_score = val_scores.get(persona_id, 0.7)  # Default if not available

        # Weight = validation quality * centrality
        weight = val_score * (0.5 + similarity)  # Bias toward central personas
        weights[persona_id] = weight

    # Normalize
    total = sum(weights.values())
    weights = {k: v/total for k, v in weights.items()}

    # Sort by weight
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)

    print("\nTop 10 weighted personas:")
    for persona, weight in sorted_weights[:10]:
        val_score = val_scores.get(persona, 0.7)
        print(f"  {persona}: {weight:.4f} (val={val_score:.3f})")

    return weights


def weighted_merge_loras(
    cluster_id,
    base_model='Qwen/Qwen2.5-0.5B-Instruct',
    lora_dir='models/lora_adapters',
    output_dir=None
):
    """Merge per-persona LoRAs with smart weights"""

    if output_dir is None:
        output_dir = f'models/weighted_cluster_{cluster_id}'

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f"Weighted Merge: Cluster {cluster_id}")
    print(f"{'=' * 80}")

    # Compute weights
    weights = compute_cluster_weights(cluster_id)

    # Load base model
    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Merge LoRAs with weights
    print("\nMerging LoRAs...")
    merged_state_dict = {}

    for persona_id, weight in weights.items():
        lora_path = Path(lora_dir) / persona_id

        if not lora_path.exists():
            print(f"  Warning: {lora_path} not found, skipping")
            continue

        # Load LoRA
        try:
            peft_model = PeftModel.from_pretrained(model, str(lora_path))
            lora_state_dict = peft_model.state_dict()

            # Add weighted parameters
            for name, param in lora_state_dict.items():
                if 'lora' in name:
                    if name not in merged_state_dict:
                        merged_state_dict[name] = weight * param.cpu().numpy()
                    else:
                        merged_state_dict[name] += weight * param.cpu().numpy()
        except Exception as e:
            print(f"  Error loading {persona_id}: {e}")
            continue

    # Create merged model
    print("\nCreating merged model...")

    # Use first persona as template structure
    template_persona = list(weights.keys())[0]
    template_path = Path(lora_dir) / template_persona

    final_model = PeftModel.from_pretrained(model, str(template_path))

    # Replace with merged weights
    state_dict = final_model.state_dict()
    for name in merged_state_dict:
        if name in state_dict:
            state_dict[name] = torch.from_numpy(merged_state_dict[name])

    final_model.load_state_dict(state_dict)

    # Save
    print(f"\nSaving to {output_dir}...")
    final_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save weight info
    with open(output_dir / 'merge_weights.json', 'w') as f:
        json.dump(weights, f, indent=2)

    print(f"\n✓ Weighted merge complete!")
    print(f"  Used {len(weights)} personas")
    print(f"  Saved to {output_dir}")

    return final_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Weighted merge of cluster LoRAs')
    parser.add_argument('--cluster_id', type=int, required=True,
                       help='Cluster ID to merge')

    args = parser.parse_args()

    weighted_merge_loras(args.cluster_id)
