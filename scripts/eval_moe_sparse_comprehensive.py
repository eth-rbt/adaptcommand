"""
Evaluate Sparse MoE merged models with COMPREHENSIVE METRICS

For each persona:
1. Load their sparse MoE merged model
2. Evaluate on test set
3. Compute ALL metrics (embedding similarity, device precision, numerical precision, etc.)
"""

import json
import numpy as np
import torch
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Import action metrics
from action_metrics import ActionExtractor, ActionMetrics


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


def evaluate_moe_model(
    persona_id,
    test_dialogues,
    base_model='Qwen/Qwen2.5-0.5B-Instruct',
    moe_dir='models/moe_sparse_k5',
    encoder=None,
    action_extractor=None,
):
    """Evaluate MoE model for a specific persona with comprehensive metrics"""

    if encoder is None:
        encoder = SentenceTransformer('all-MiniLM-L6-v2')

    if action_extractor is None:
        action_extractor = ActionExtractor()

    # Load base model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model)

    # Load merged LoRA weights
    persona_moe_path = Path(moe_dir) / persona_id / 'merged_lora.pt'

    if not persona_moe_path.exists():
        print(f"Warning: {persona_moe_path} not found", file=sys.stderr)
        return None

    merged_weights = torch.load(persona_moe_path, weights_only=False)

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

    # Collect predictions and references
    predictions = []
    references = []
    similarities = []
    all_action_metrics = []

    for dialogue in test_dialogues:
        # Format input
        prompt = format_dialogue_for_generation(dialogue, tokenizer)

        # Generate
        prediction = generate_response(peft_model, tokenizer, prompt)
        predictions.append(prediction)

        # Get ground truth
        target = dialogue['messages'][-1]['text']
        references.append(target)

        # Compute embedding similarity
        pred_emb = encoder.encode([prediction], show_progress_bar=False)[0]
        target_emb = encoder.encode([target], show_progress_bar=False)[0]
        similarity = np.dot(pred_emb, target_emb) / (
            np.linalg.norm(pred_emb) * np.linalg.norm(target_emb)
        )
        similarities.append(float(similarity))

        # Extract actions and compute action-based metrics
        pred_actions = action_extractor.extract_actions(prediction)
        ref_actions = action_extractor.extract_actions(target)
        action_metrics = ActionMetrics.compare_actions(pred_actions, ref_actions)
        all_action_metrics.append(action_metrics)

    # Aggregate metrics
    metrics = {
        'persona_id': persona_id,
        'embedding_similarity': float(np.mean(similarities)),
        'embedding_similarity_std': float(np.std(similarities)),
        'device_precision': float(np.mean([m['device_precision'] for m in all_action_metrics])),
        'param_f1': float(np.mean([m['param_f1'] for m in all_action_metrics])),
        'numerical_precision': float(np.mean([m['numerical_precision'] for m in all_action_metrics])),
        'num_test_examples': len(test_dialogues),
    }

    # Clean up
    del peft_model
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return metrics


def evaluate_all_moe_models(moe_dir='models/moe_sparse_k5'):
    """Evaluate all MoE models with comprehensive metrics"""

    print("Evaluating Sparse MoE Models (COMPREHENSIVE METRICS)")
    print("=" * 80)

    # Load test data
    print("Loading test data...")
    test_data = load_test_data()

    # Load routing info
    with open(Path(moe_dir) / 'routing_info.json') as f:
        routing_info = json.load(f)

    # Initialize shared components
    print("Initializing models...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    action_extractor = ActionExtractor()

    # Evaluate each persona
    results = []

    for persona_id in tqdm(sorted(routing_info.keys()), desc="Evaluating personas"):
        if persona_id not in test_data:
            print(f"Warning: No test data for {persona_id}", file=sys.stderr)
            continue

        try:
            metrics = evaluate_moe_model(
                persona_id,
                test_data[persona_id],
                encoder=encoder,
                action_extractor=action_extractor,
                moe_dir=moe_dir,
            )

            if metrics:
                results.append(metrics)

                # Print progress
                if len(results) % 10 == 0:
                    print(f"Evaluated {len(results)}/{len(routing_info)} personas")

        except Exception as e:
            print(f"Error evaluating {persona_id}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            continue

    # Compute summary statistics
    summary = {
        'embedding_similarity_mean': float(np.mean([r['embedding_similarity'] for r in results])),
        'embedding_similarity_std': float(np.std([r['embedding_similarity'] for r in results])),
        'device_precision_mean': float(np.mean([r['device_precision'] for r in results])),
        'numerical_precision_mean': float(np.mean([r['numerical_precision'] for r in results])),
        'param_f1_mean': float(np.mean([r['param_f1'] for r in results])),
        'num_personas': len(results),
    }

    # Save results
    output_dir = Path('results/moe_sparse')
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'summary': summary,
        'per_persona_results': results,
    }

    with open(output_dir / 'moe_sparse_comprehensive_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("SPARSE MoE COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 80)
    print(f"Embedding Similarity:    {summary['embedding_similarity_mean']:.4f} ± {summary['embedding_similarity_std']:.4f}")
    print(f"Device Precision:        {summary['device_precision_mean']:.4f}")
    print(f"Numerical Precision:     {summary['numerical_precision_mean']:.4f}")
    print(f"Param F1:                {summary['param_f1_mean']:.4f}")
    print(f"Personas evaluated:      {summary['num_personas']}")

    print(f"\nResults saved to {output_dir / 'moe_sparse_comprehensive_results.json'}")

    return summary


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate sparse MoE models with comprehensive metrics')
    parser.add_argument('--moe_dir', type=str, default='models/moe_sparse_k5',
                       help='Directory containing MoE models')

    args = parser.parse_args()

    evaluate_all_moe_models(moe_dir=args.moe_dir)
