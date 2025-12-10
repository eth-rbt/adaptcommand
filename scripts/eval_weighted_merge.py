"""
Evaluate weighted merge cluster model

Adapted from eval_cluster_lora.py to evaluate weighted merge models
"""

import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_test_data(cluster_id):
    """Load test dialogues for personas in cluster"""

    # Load cluster map
    with open('data/splits/cluster_map.json') as f:
        cluster_data = json.load(f)

    cluster_map = cluster_data['cluster_map']

    # Get personas in this cluster
    cluster_personas = [p for p, c in cluster_map.items() if c == cluster_id]

    # Load all dialogues
    with open('data/cleaned/dialogs_clean.jsonl') as f:
        all_dialogues = [json.loads(line) for line in f]

    # Load splits
    with open('data/splits/edgesplits.json') as f:
        splits = json.load(f)

    # Collect test data per persona
    test_data = {}

    for idx, dialogue in enumerate(all_dialogues):
        persona_id = dialogue['persona_id']

        if persona_id not in cluster_personas:
            continue

        if persona_id not in test_data:
            test_data[persona_id] = []

        # Check if this dialogue is in test set
        if idx in splits[persona_id]['test']:
            test_data[persona_id].append(dialogue)

    return test_data


def format_dialogue_for_generation(dialogue, tokenizer):
    """Format dialogue for generation (all turns except last)"""

    # Build system message
    system_parts = [
        "You are a helpful smart home assistant.",
        f"\nUser Profile: {dialogue['character']}"
    ]

    # Add context
    if 'meta' in dialogue and dialogue['meta']:
        context = dialogue['meta']
        context_items = []
        for k, v in context.items():
            if k != 'routines' and v:
                context_items.append(f"{k}: {v}")
        if context_items:
            system_parts.append(f"\nContext: {', '.join(context_items)}")

    system_message = "\n".join(system_parts)

    # Format conversation (exclude last assistant turn)
    messages = [{"role": "system", "content": system_message}]

    for msg in dialogue['messages'][:-1]:  # Exclude last turn
        messages.append({
            "role": msg['role'],
            "content": msg['text']
        })

    # Apply chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return text


def generate_response(model, tokenizer, prompt, max_new_tokens=128):
    """Generate response from model"""

    inputs = tokenizer(prompt, return_tensors='pt')

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    return response.strip()


def compute_embedding_similarity(pred, target, encoder):
    """Compute embedding similarity between prediction and target"""

    pred_emb = encoder.encode([pred], show_progress_bar=False)[0]
    target_emb = encoder.encode([target], show_progress_bar=False)[0]

    # Cosine similarity
    similarity = np.dot(pred_emb, target_emb) / (
        np.linalg.norm(pred_emb) * np.linalg.norm(target_emb)
    )

    return float(similarity)


def evaluate_weighted_merge(
    cluster_id,
    model_dir='models/weighted_cluster_4',
    base_model='Qwen/Qwen2.5-0.5B-Instruct',
):
    """Evaluate weighted merge model for a cluster"""

    print("=" * 80)
    print(f"EVALUATING WEIGHTED MERGE MODEL - CLUSTER {cluster_id}")
    print("=" * 80)

    # Load test data
    print("\nLoading test data...")
    test_data = load_test_data(cluster_id)

    # Load model
    print(f"\nLoading weighted merge model from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model)
    model = PeftModel.from_pretrained(model, model_dir)
    model.eval()

    # Initialize sentence encoder
    encoder = SentenceTransformer('all-MiniLM-L6-v2')

    print(f"Cluster {cluster_id} contains {len(test_data)} personas")

    # Evaluate each persona
    all_similarities = []

    for persona_id in tqdm(sorted(test_data.keys()), desc=f"Cluster {cluster_id}"):
        persona_similarities = []

        for dialogue in test_data[persona_id]:
            # Format input
            prompt = format_dialogue_for_generation(dialogue, tokenizer)

            # Generate
            prediction = generate_response(model, tokenizer, prompt)

            # Get ground truth
            target = dialogue['messages'][-1]['text']

            # Compute similarity
            similarity = compute_embedding_similarity(prediction, target, encoder)
            persona_similarities.append(similarity)

        all_similarities.extend(persona_similarities)

    # Compute overall metrics
    results = {
        'cluster_id': cluster_id,
        'embedding_similarity': float(np.mean(all_similarities)),
        'embedding_similarity_std': float(np.std(all_similarities)),
        'num_personas': len(test_data),
        'num_test_examples': len(all_similarities),
    }

    print(f"\nCluster {cluster_id} Results:")
    print(f"  Embedding Similarity: {results['embedding_similarity']:.4f}")
    print(f"  Std Dev:              {results['embedding_similarity_std']:.4f}")
    print(f"  Personas:             {results['num_personas']}")
    print(f"  Test Examples:        {results['num_test_examples']}")

    # Save results
    output_dir = Path('results/weighted_merge')
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f'weighted_merge_cluster{cluster_id}_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {output_dir / f'weighted_merge_cluster{cluster_id}_results.json'}")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate weighted merge cluster model')
    parser.add_argument('--cluster_id', type=int, default=4,
                       help='Cluster ID')
    parser.add_argument('--model_dir', type=str, default='models/weighted_cluster_4',
                       help='Model directory')

    args = parser.parse_args()

    evaluate_weighted_merge(args.cluster_id, model_dir=args.model_dir)
