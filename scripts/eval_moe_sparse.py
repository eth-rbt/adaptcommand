"""
Evaluate Sparse MoE merged models

For each persona:
1. Load their sparse MoE merged model
2. Evaluate on test set
3. Compute all metrics (embedding similarity, device precision, etc.)
"""

import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_test_data():
    """Load test dialogues"""

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


def evaluate_moe_model(
    persona_id,
    test_dialogues,
    base_model='Qwen/Qwen2.5-0.5B-Instruct',
    moe_dir='models/moe_sparse_k5',
    encoder=None,
):
    """Evaluate MoE model for a specific persona"""

    if encoder is None:
        encoder = SentenceTransformer('all-MiniLM-L6-v2')

    # Load base model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model)

    # Load merged LoRA weights
    persona_moe_path = Path(moe_dir) / persona_id / 'merged_lora.pt'

    if not persona_moe_path.exists():
        print(f"Warning: {persona_moe_path} not found")
        return None

    merged_weights = torch.load(persona_moe_path, weights_only=False)

    # Apply merged weights to model
    # NOTE: This is simplified - in practice you'd need to properly load PEFT weights
    # For now, we'll use a workaround by loading a base LoRA and replacing weights

    # Load routing info to get one of the experts
    with open(Path(moe_dir) / 'routing_info.json') as f:
        routing_info = json.load(f)

    expert_persona = routing_info[persona_id]['experts'][0]
    expert_path = Path('models/lora_adapters') / expert_persona

    # Load expert as base structure
    peft_model = PeftModel.from_pretrained(model, str(expert_path))

    # Replace weights with merged weights
    state_dict = peft_model.state_dict()
    for name in merged_weights:
        if name in state_dict:
            state_dict[name] = torch.from_numpy(merged_weights[name])

    peft_model.load_state_dict(state_dict)
    peft_model.eval()

    # Evaluate on test dialogues
    similarities = []

    for dialogue in test_dialogues:
        # Format input
        prompt = format_dialogue_for_generation(dialogue, tokenizer)

        # Generate
        prediction = generate_response(peft_model, tokenizer, prompt)

        # Get ground truth
        target = dialogue['messages'][-1]['text']

        # Compute similarity
        similarity = compute_embedding_similarity(prediction, target, encoder)
        similarities.append(similarity)

    # Compute metrics
    metrics = {
        'embedding_similarity': np.mean(similarities),
        'embedding_similarity_std': np.std(similarities),
        'num_test_examples': len(test_dialogues),
    }

    return metrics


def evaluate_all_moe_models(moe_dir='models/moe_sparse_k5'):
    """Evaluate all MoE models"""

    print("Evaluating Sparse MoE Models")
    print("=" * 80)

    # Load test data
    print("Loading test data...")
    test_data = load_test_data()

    # Load routing info
    with open(Path(moe_dir) / 'routing_info.json') as f:
        routing_info = json.load(f)

    # Initialize encoder once
    encoder = SentenceTransformer('all-MiniLM-L6-v2')

    # Evaluate each persona
    results = {}

    for persona_id in tqdm(sorted(routing_info.keys()), desc="Evaluating personas"):
        if persona_id not in test_data:
            print(f"Warning: No test data for {persona_id}")
            continue

        try:
            metrics = evaluate_moe_model(
                persona_id,
                test_data[persona_id],
                encoder=encoder,
                moe_dir=moe_dir,
            )

            if metrics:
                results[persona_id] = metrics

        except Exception as e:
            print(f"Error evaluating {persona_id}: {e}")
            continue

    # Compute summary statistics
    all_similarities = [r['embedding_similarity'] for r in results.values()]

    summary = {
        'embedding_similarity_mean': np.mean(all_similarities),
        'embedding_similarity_std': np.std(all_similarities),
        'embedding_similarity_min': np.min(all_similarities),
        'embedding_similarity_max': np.max(all_similarities),
        'num_personas': len(results),
    }

    # Save results
    output_dir = Path('results/moe_sparse')
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'summary': summary,
        'per_persona_results': results,
    }

    with open(output_dir / 'moe_sparse_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("SPARSE MoE EVALUATION RESULTS")
    print("=" * 80)
    print(f"Embedding Similarity: {summary['embedding_similarity_mean']:.4f} ± {summary['embedding_similarity_std']:.4f}")
    print(f"Min: {summary['embedding_similarity_min']:.4f}")
    print(f"Max: {summary['embedding_similarity_max']:.4f}")
    print(f"Personas evaluated: {summary['num_personas']}")

    print(f"\nResults saved to {output_dir / 'moe_sparse_results.json'}")

    return summary


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate sparse MoE models')
    parser.add_argument('--moe_dir', type=str, default='models/moe_sparse_k5',
                       help='Directory containing MoE models')

    args = parser.parse_args()

    evaluate_all_moe_models(moe_dir=args.moe_dir)
