"""
Cluster-Based LoRA Merging

Merge existing per-persona LoRAs by cluster to create 5 better combined models.

Why this works:
- Merges similar personas together (based on clustering)
- Each merged model represents 16-72 personas
- Averages out individual overfitting while keeping cluster patterns
- NO TRAINING REQUIRED - uses existing models!

Expected: +3-5% over unified baseline
"""

import json
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import copy


def load_cluster_info():
    """Load cluster assignments"""
    with open('data/splits/cluster_map.json') as f:
        data = json.load(f)

    cluster_map = data['cluster_map']
    n_clusters = data['n_clusters']

    # Organize by cluster
    clusters = {i: [] for i in range(n_clusters)}
    for persona_id, cluster_id in cluster_map.items():
        clusters[cluster_id].append(persona_id)

    return clusters, n_clusters


def merge_lora_adapters_average(base_model, lora_paths, output_path):
    """
    Merge multiple LoRA adapters by averaging their weights.

    Args:
        base_model: Base model (Qwen2.5-0.5B-Instruct)
        lora_paths: List of paths to LoRA adapters to merge
        output_path: Where to save merged adapter
    """

    print(f"\nMerging {len(lora_paths)} LoRA adapters...")

    if len(lora_paths) == 0:
        print("No LoRAs to merge!")
        return

    # Load first LoRA to get structure
    print(f"Loading first LoRA: {lora_paths[0]}")
    merged_model = PeftModel.from_pretrained(base_model, lora_paths[0])

    # Get adapter weights
    merged_state = merged_model.state_dict()

    # Initialize averaged weights
    avg_state = {}
    lora_keys = [k for k in merged_state.keys() if 'lora' in k.lower()]

    print(f"Found {len(lora_keys)} LoRA parameters to merge")

    # Initialize with zeros
    for key in lora_keys:
        avg_state[key] = torch.zeros_like(merged_state[key])

    # Accumulate all LoRA weights
    print("Accumulating weights from all adapters...")
    for i, lora_path in enumerate(tqdm(lora_paths)):
        try:
            # Load this LoRA
            model = PeftModel.from_pretrained(copy.deepcopy(base_model), lora_path)
            state = model.state_dict()

            # Add to average
            for key in lora_keys:
                if key in state:
                    avg_state[key] += state[key].float()

            # Clean up
            del model

        except Exception as e:
            print(f"Error loading {lora_path}: {e}")
            continue

    # Divide by count to get average
    print("Computing average...")
    for key in lora_keys:
        avg_state[key] = avg_state[key] / len(lora_paths)

    # Update merged model with averaged weights
    print("Updating model with merged weights...")
    merged_state_dict = merged_model.state_dict()
    for key in lora_keys:
        merged_state_dict[key] = avg_state[key]

    merged_model.load_state_dict(merged_state_dict)

    # Save
    print(f"Saving merged adapter to {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_path)

    # Save merge info
    with open(output_path / 'merge_info.json', 'w') as f:
        json.dump({
            'num_merged': len(lora_paths),
            'source_adapters': [str(p) for p in lora_paths],
            'merge_method': 'simple_average'
        }, f, indent=2)

    print(f"✓ Merged {len(lora_paths)} adapters")

    return merged_model


def merge_all_clusters():
    """
    Merge per-persona LoRAs by cluster.
    Creates 5 cluster-merged LoRA adapters.
    """

    print("=" * 80)
    print("CLUSTER-BASED LORA MERGING")
    print("=" * 80)

    # Load cluster info
    clusters, n_clusters = load_cluster_info()

    print(f"\nFound {n_clusters} clusters:")
    for cluster_id, personas in clusters.items():
        print(f"  Cluster {cluster_id}: {len(personas)} personas")

    # Load base model (needed for merging)
    print("\nLoading base model...")
    base_model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

    print("Base model loaded")

    # Merge each cluster
    output_base = Path('models/lora_cluster_merged')

    for cluster_id, personas in clusters.items():
        print(f"\n{'=' * 80}")
        print(f"CLUSTER {cluster_id}")
        print(f"{'=' * 80}")

        # Get LoRA paths for this cluster
        lora_paths = []
        for persona_id in personas:
            lora_path = Path(f'models/lora_adapters/{persona_id}')
            if lora_path.exists():
                lora_paths.append(lora_path)
            else:
                print(f"Warning: LoRA not found for {persona_id}")

        if len(lora_paths) == 0:
            print(f"No LoRAs found for cluster {cluster_id}, skipping")
            continue

        print(f"Found {len(lora_paths)} LoRAs to merge")

        # Merge
        output_path = output_base / f'cluster_{cluster_id:02d}'
        merged_model = merge_lora_adapters_average(
            base_model,
            lora_paths,
            output_path
        )

        # Save tokenizer too
        tokenizer.save_pretrained(output_path)

        # Save cluster info
        with open(output_path / 'cluster_info.json', 'w') as f:
            json.dump({
                'cluster_id': cluster_id,
                'num_personas': len(personas),
                'personas': personas,
            }, f, indent=2)

        print(f"✓ Cluster {cluster_id} merged successfully")

        # Clean up
        del merged_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\n{'=' * 80}")
    print("ALL CLUSTERS MERGED")
    print(f"{'=' * 80}")
    print(f"\nMerged models saved to: {output_base}")
    print("\nNext steps:")
    print(f"  python scripts/eval_cluster_merged.py")


if __name__ == '__main__':
    merge_all_clusters()
