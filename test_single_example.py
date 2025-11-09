"""
Test Single Example - Run baseline on ONE dialogue and show detailed output

This script helps you understand what the benchmark is doing by showing:
- Input context
- Reference response
- Model prediction
- Computed metrics
"""

import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import numpy as np
from scripts.action_metrics import ActionExtractor, ActionMetrics


def load_one_example():
    """Load just one test example."""
    # Load dialogues
    with open("data/cleaned/dialogs_clean.jsonl", "r") as f:
        dialogues = [json.loads(line) for line in f]

    # Load splits
    with open("data/splits/edgesplits.json", "r") as f:
        splits = json.load(f)

    # Get first test example
    for persona_id, persona_splits in splits.items():
        if len(persona_splits["test"]) > 0:
            idx = persona_splits["test"][0]
            return dialogues[idx], persona_id

    return None, None


def format_dialogue(dialogue, system_prompt):
    """Format dialogue for the model."""
    messages = dialogue["messages"]

    # Find last assistant message
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            # Get context (everything before this turn)
            context_messages = messages[:i]
            reference = messages[i]["text"]

            # Build prompt
            prompt_messages = [{"role": "system", "content": system_prompt}]

            # Add last 5 turns (max_context_messages)
            start_idx = max(0, len(context_messages) - 10)  # 5 pairs * 2
            for msg in context_messages[start_idx:]:
                prompt_messages.append({
                    "role": msg["role"],
                    "content": msg["text"]
                })

            return prompt_messages, reference, i

    return None, None, None


def generate_response(model, tokenizer, messages, device):
    """Generate a response."""
    # Format messages
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode
    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return generated_text.strip()


def compute_single_metrics(prediction, reference):
    """Compute metrics for a single example."""
    metrics = {}

    # Embedding similarity
    print("\n" + "="*60)
    print("COMPUTING EMBEDDING SIMILARITY")
    print("="*60)
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    pred_emb = embedding_model.encode([prediction])[0]
    ref_emb = embedding_model.encode([reference])[0]

    sim = np.dot(pred_emb, ref_emb) / (np.linalg.norm(pred_emb) * np.linalg.norm(ref_emb))
    metrics["embedding_similarity"] = float(sim)
    print(f"Embedding similarity: {sim:.4f}")

    # Action accuracy
    print("\n" + "="*60)
    print("COMPUTING ACTION ACCURACY")
    print("="*60)
    extractor = ActionExtractor()

    pred_actions = extractor.extract_actions(prediction)
    ref_actions = extractor.extract_actions(reference)

    print(f"\nReference actions extracted:")
    print(json.dumps(ref_actions, indent=2))

    print(f"\nPrediction actions extracted:")
    print(json.dumps(pred_actions, indent=2))

    action_metrics = ActionMetrics.compare_actions(pred_actions, ref_actions)

    print(f"\nDevice-level metrics:")
    print(f"  device_precision: {action_metrics['device_precision']:.4f}")
    print(f"  device_recall: {action_metrics['device_recall']:.4f}")

    print(f"\nParameter metrics (overall):")
    print(f"  param_precision: {action_metrics['param_precision']:.4f}")
    print(f"  param_recall: {action_metrics['param_recall']:.4f}")
    print(f"  param_f1: {action_metrics['param_f1']:.4f}")

    print(f"\nNumerical parameters:")
    print(f"  TP: {action_metrics['numerical_tp']}, FP: {action_metrics['numerical_fp']}, FN: {action_metrics['numerical_fn']}")
    print(f"  Precision: {action_metrics['numerical_precision']:.4f}")
    print(f"  Recall: {action_metrics['numerical_recall']:.4f}")
    print(f"  F1: {action_metrics['numerical_f1']:.4f}")
    if action_metrics['numerical_mae'] > 0:
        print(f"  MAE: {action_metrics['numerical_mae']:.2f}")

    print(f"\nCategorical parameters:")
    print(f"  TP: {action_metrics['categorical_tp']}, FP: {action_metrics['categorical_fp']}, FN: {action_metrics['categorical_fn']}")
    print(f"  Precision: {action_metrics['categorical_precision']:.4f}")
    print(f"  Recall: {action_metrics['categorical_recall']:.4f}")
    print(f"  F1: {action_metrics['categorical_f1']:.4f}")

    # Store all metrics
    for key, value in action_metrics.items():
        if isinstance(value, (int, float)):
            metrics[key] = value
        else:
            metrics[key] = float(value)

    # Length stats
    pred_len = len(prediction.split())
    ref_len = len(reference.split())
    metrics["pred_length"] = pred_len
    metrics["ref_length"] = ref_len
    metrics["length_ratio"] = pred_len / ref_len if ref_len > 0 else 0

    return metrics


def main():
    print("="*60)
    print("SINGLE EXAMPLE TEST")
    print("="*60)

    # Load one example
    print("\nLoading one test example...")
    dialogue, persona_id = load_one_example()

    if dialogue is None:
        print("Error: Could not load test example")
        return

    print(f"✓ Loaded dialogue from {persona_id}")
    print(f"  Session ID: {dialogue.get('session_id', 'N/A')}")
    print(f"  Total messages: {len(dialogue['messages'])}")

    # Show persona info
    print("\n" + "="*60)
    print("PERSONA INFORMATION")
    print("="*60)
    print(f"Character: {dialogue.get('character', 'N/A')[:200]}...")

    if 'meta' in dialogue:
        print(f"\nContext:")
        for key, value in dialogue['meta'].items():
            print(f"  {key}: {value}")

    # Format dialogue
    system_prompt = "You are a helpful smart home assistant. Help the user control their devices by understanding their preferences and the current context."
    prompt_messages, reference, turn_idx = format_dialogue(dialogue, system_prompt)

    if prompt_messages is None:
        print("Error: Could not format dialogue")
        return

    # Show input
    print("\n" + "="*60)
    print("INPUT CONTEXT (what model sees)")
    print("="*60)
    for msg in prompt_messages:
        role_marker = "🤖" if msg["role"] == "assistant" else "👤" if msg["role"] == "user" else "⚙️"
        print(f"\n{role_marker} {msg['role'].upper()}:")
        print(msg["content"])

    # Show reference
    print("\n" + "="*60)
    print("REFERENCE (ground truth assistant response)")
    print("="*60)
    print(reference)

    # Load model
    print("\n" + "="*60)
    print("LOADING MODEL")
    print("="*60)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32,
        device_map=device if device == "cuda" else None
    )

    if device == "mps":
        model = model.to(device)

    model.eval()
    print("✓ Model loaded")

    # Generate prediction
    print("\n" + "="*60)
    print("GENERATING PREDICTION")
    print("="*60)
    prediction = generate_response(model, tokenizer, prompt_messages, device)
    print(prediction)

    # Compute metrics
    metrics = compute_single_metrics(prediction, reference)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nReference length: {metrics['ref_length']} words")
    print(f"Prediction length: {metrics['pred_length']} words")
    print(f"Length ratio: {metrics['length_ratio']:.2f}")
    print(f"\nEmbedding similarity: {metrics['embedding_similarity']:.4f}")

    print(f"\n--- Device Detection ---")
    print(f"Device precision: {metrics['device_precision']:.4f}")
    print(f"Device recall: {metrics['device_recall']:.4f}")

    print(f"\n--- Parameter Metrics (Overall) ---")
    print(f"Param precision: {metrics['param_precision']:.4f}")
    print(f"Param recall: {metrics['param_recall']:.4f}")
    print(f"Param F1: {metrics['param_f1']:.4f}")

    print(f"\n--- Numerical Parameters ---")
    print(f"TP: {metrics['numerical_tp']}, FP: {metrics['numerical_fp']}, FN: {metrics['numerical_fn']}")
    print(f"Precision: {metrics['numerical_precision']:.4f}")
    print(f"Recall: {metrics['numerical_recall']:.4f}")
    print(f"F1: {metrics['numerical_f1']:.4f}")
    if metrics['numerical_mae'] > 0:
        print(f"MAE: {metrics['numerical_mae']:.2f}")

    print(f"\n--- Categorical Parameters ---")
    print(f"TP: {metrics['categorical_tp']}, FP: {metrics['categorical_fp']}, FN: {metrics['categorical_fn']}")
    print(f"Precision: {metrics['categorical_precision']:.4f}")
    print(f"Recall: {metrics['categorical_recall']:.4f}")
    print(f"F1: {metrics['categorical_f1']:.4f}")

    # Save results
    output = {
        "persona_id": persona_id,
        "session_id": dialogue.get('session_id', 'N/A'),
        "turn_idx": turn_idx,
        "reference": reference,
        "prediction": prediction,
        "metrics": metrics
    }

    output_file = Path("results/single_example_test.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()
