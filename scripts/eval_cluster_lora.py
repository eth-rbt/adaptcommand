"""
Evaluate cluster-level LoRA adapters

For each persona, load their cluster's LoRA and evaluate on their test set.
Compare to unified baseline.
"""

import json
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
import argparse

# Import metrics
import sys
sys.path.append('scripts')
from action_metrics import ActionExtractor, ActionMetrics


def load_cluster_info():
    """Load cluster assignments"""
    with open('data/splits/cluster_map.json') as f:
        data = json.load(f)
    return data['cluster_map'], data['n_clusters']


def evaluate_cluster_lora(cluster_id=None, max_examples_per_persona=None):
    """
    Evaluate cluster LoRA models.

    Args:
        cluster_id: If specified, only evaluate this cluster
        max_examples_per_persona: Limit examples per persona (for quick testing)
    """

    print("=" * 80)
    print("CLUSTER LORA EVALUATION")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    with open('data/cleaned/dialogs_clean.jsonl') as f:
        dialogues = [json.loads(line) for line in f]

    with open('data/splits/edgesplits.json') as f:
        splits = json.load(f)

    cluster_map, n_clusters = load_cluster_info()

    # Load tokenizer
    base_model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize metrics
    extractor = ActionExtractor()
    similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Determine which clusters to evaluate
    clusters_to_eval = [cluster_id] if cluster_id is not None else range(n_clusters)

    # Results storage
    all_results = []

    for cid in clusters_to_eval:
        print(f"\n{'=' * 80}")
        print(f"EVALUATING CLUSTER {cid}")
        print(f"{'=' * 80}")

        # Load cluster model
        cluster_path = Path(f'models/lora_clusters/cluster_{cid:02d}')

        if not cluster_path.exists():
            print(f"Cluster {cid} model not found at {cluster_path}")
            print("Skipping...")
            continue

        print(f"\nLoading cluster {cid} model from {cluster_path}")
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(base_model, cluster_path)
        model.eval()

        # Get personas in this cluster
        personas_in_cluster = [pid for pid, c in cluster_map.items() if c == cid]
        print(f"Cluster {cid} contains {len(personas_in_cluster)} personas")

        # Evaluate each persona in this cluster
        cluster_results = []

        for persona_id in tqdm(personas_in_cluster, desc=f"Cluster {cid}"):
            # Get test dialogues for this persona
            test_indices = splits[persona_id]['test']
            test_dialogues = [dialogues[i] for i in test_indices]

            if max_examples_per_persona:
                test_dialogues = test_dialogues[:max_examples_per_persona]

            persona_predictions = []
            persona_references = []

            # Evaluate each test dialogue
            for dialogue in test_dialogues:
                messages = dialogue['messages']

                # Process each user-assistant pair
                for i in range(0, len(messages) - 1, 2):
                    if messages[i]['role'] != 'user' or messages[i+1]['role'] != 'assistant':
                        continue

                    user_msg = messages[i]['text']
                    reference = messages[i+1]['text']

                    # Build prompt
                    system_parts = [
                        "You are a helpful smart home assistant.",
                        f"\nUser Profile: {dialogue['character']}"
                    ]

                    if 'meta' in dialogue and dialogue['meta']:
                        context = dialogue['meta']
                        context_items = []
                        for k, v in context.items():
                            if k != 'routines' and v:
                                context_items.append(f"{k}: {v}")
                        if context_items:
                            system_parts.append(f"\nContext: {', '.join(context_items)}")

                    system_message = "\n".join(system_parts)

                    # Format with chat template
                    chat_messages = [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_msg}
                    ]

                    prompt = tokenizer.apply_chat_template(
                        chat_messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )

                    # Generate
                    inputs = tokenizer(prompt, return_tensors="pt")
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=256,
                            temperature=0.7,
                            top_p=0.9,
                            do_sample=True,
                            repetition_penalty=1.1
                        )

                    prediction = tokenizer.decode(
                        outputs[0][len(inputs.input_ids[0]):],
                        skip_special_tokens=True
                    ).strip()

                    persona_predictions.append(prediction)
                    persona_references.append(reference)

            # Calculate metrics for this persona
            if len(persona_predictions) == 0:
                continue

            # Embedding similarity
            pred_embs = similarity_model.encode(persona_predictions)
            ref_embs = similarity_model.encode(persona_references)

            similarities = [
                np.dot(p, r) / (np.linalg.norm(p) * np.linalg.norm(r))
                for p, r in zip(pred_embs, ref_embs)
            ]

            avg_similarity = float(np.mean(similarities))

            # Action accuracy
            all_metrics = []
            for pred, ref in zip(persona_predictions, persona_references):
                pred_actions = extractor.extract_actions(pred)
                ref_actions = extractor.extract_actions(ref)
                metrics = ActionMetrics.compare_actions(pred_actions, ref_actions)
                all_metrics.append(metrics)

            avg_device_precision = float(np.mean([m['device_precision'] for m in all_metrics]))
            avg_param_f1 = float(np.mean([m['param_f1'] for m in all_metrics]))
            avg_numerical_precision = float(np.mean([m['numerical_precision'] for m in all_metrics]))

            persona_result = {
                'persona_id': persona_id,
                'cluster_id': cid,
                'embedding_similarity': avg_similarity,
                'device_precision': avg_device_precision,
                'param_f1': avg_param_f1,
                'numerical_precision': avg_numerical_precision,
                'num_examples': len(persona_predictions)
            }

            cluster_results.append(persona_result)
            all_results.append(persona_result)

        # Print cluster summary
        if cluster_results:
            print(f"\nCluster {cid} Results:")
            print(f"  Embedding Similarity: {np.mean([r['embedding_similarity'] for r in cluster_results]):.4f}")
            print(f"  Device Precision:     {np.mean([r['device_precision'] for r in cluster_results]):.4f}")
            print(f"  Param F1:             {np.mean([r['param_f1'] for r in cluster_results]):.4f}")
            print(f"  Numerical Precision:  {np.mean([r['numerical_precision'] for r in cluster_results]):.4f}")

        # Clean up
        del model
        del base_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Overall summary
    if all_results:
        print(f"\n{'=' * 80}")
        print("OVERALL RESULTS (ALL CLUSTERS)")
        print(f"{'=' * 80}")
        print(f"Personas evaluated: {len(all_results)}")
        print(f"Embedding Similarity: {np.mean([r['embedding_similarity'] for r in all_results]):.4f}")
        print(f"Device Precision:     {np.mean([r['device_precision'] for r in all_results]):.4f}")
        print(f"Param F1:             {np.mean([r['param_f1'] for r in all_results]):.4f}")
        print(f"Numerical Precision:  {np.mean([r['numerical_precision'] for r in all_results]):.4f}")

        # Save results
        output_dir = Path('results/cluster_lora')
        output_dir.mkdir(exist_ok=True, parents=True)

        with open(output_dir / 'cluster_lora_results.json', 'w') as f:
            json.dump({
                'per_persona_results': all_results,
                'summary': {
                    'num_personas': len(all_results),
                    'embedding_similarity_mean': float(np.mean([r['embedding_similarity'] for r in all_results])),
                    'embedding_similarity_std': float(np.std([r['embedding_similarity'] for r in all_results])),
                    'device_precision_mean': float(np.mean([r['device_precision'] for r in all_results])),
                    'param_f1_mean': float(np.mean([r['param_f1'] for r in all_results])),
                    'numerical_precision_mean': float(np.mean([r['numerical_precision'] for r in all_results])),
                }
            }, f, indent=2)

        print(f"\nSaved results to {output_dir / 'cluster_lora_results.json'}")

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster_id', type=int, default=None,
                       help='Evaluate specific cluster only')
    parser.add_argument('--max_examples', type=int, default=None,
                       help='Max examples per persona for quick testing')

    args = parser.parse_args()

    evaluate_cluster_lora(
        cluster_id=args.cluster_id,
        max_examples_per_persona=args.max_examples
    )
