"""
Test Sequential Evaluation - Show how the new evaluation method works

This demonstrates:
1. Predicting multiple turns in sequence (not just the last one)
2. Each turn uses GROUND TRUTH context (not predictions)
3. Computing confusion matrix (TP/FP/TN/FN) for device detection
4. Skipping first 2 assistant turns, evaluating turn 3 onwards
"""

import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from scripts.action_metrics import ActionExtractor


def load_one_dialogue():
    """Load one test dialogue with multiple turns."""
    # Load dialogues
    with open("data/cleaned/dialogs_clean.jsonl", "r") as f:
        dialogues = [json.loads(line) for line in f]

    # Load splits
    with open("data/splits/edgesplits.json", "r") as f:
        splits = json.load(f)

    # Find a dialogue with at least 5 assistant turns
    for persona_id, persona_splits in splits.items():
        for idx in persona_splits["test"]:
            dialogue = dialogues[idx]
            messages = dialogue["messages"]
            assistant_count = sum(1 for msg in messages if msg["role"] == "assistant")
            if assistant_count >= 5:
                return dialogue, persona_id

    return None, None


def generate_response(model, tokenizer, messages, device):
    """Generate a response."""
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

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

    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return generated_text.strip()


def compute_device_confusion(pred_actions, ref_actions):
    """Compute confusion matrix for device detection."""
    all_devices = ["tv", "ac", "lights", "speaker", "security"]

    confusion = {"tp": [], "fp": [], "tn": [], "fn": []}

    pred_devices = set(pred_actions.keys())
    ref_devices = set(ref_actions.keys())

    for device in all_devices:
        in_pred = device in pred_devices
        in_ref = device in ref_devices

        if in_pred and in_ref:
            confusion["tp"].append(device)
        elif in_pred and not in_ref:
            confusion["fp"].append(device)
        elif not in_pred and in_ref:
            confusion["fn"].append(device)
        else:
            confusion["tn"].append(device)

    return confusion


def main():
    print("="*70)
    print("SEQUENTIAL EVALUATION DEMONSTRATION")
    print("="*70)

    # Load dialogue
    print("\nLoading dialogue with multiple turns...")
    dialogue, persona_id = load_one_dialogue()

    if dialogue is None:
        print("Error: Could not find suitable dialogue")
        return

    messages = dialogue["messages"]
    assistant_indices = [i for i, msg in enumerate(messages) if msg["role"] == "assistant"]

    print(f"✓ Loaded dialogue from {persona_id}")
    print(f"  Total messages: {len(messages)}")
    print(f"  Assistant turns: {len(assistant_indices)}")
    print(f"  Will evaluate turns: {list(range(2, len(assistant_indices)))}")

    # Load model
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading: {model_name}")

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

    # Sequential evaluation
    print("\n" + "="*70)
    print("SEQUENTIAL TURN-BY-TURN EVALUATION")
    print("="*70)

    system_prompt = "You are a helpful smart home assistant. Help the user control their devices by understanding their preferences and the current context."

    extractor = ActionExtractor()

    all_results = []

    # Evaluate turns 3 onwards (skip first 2)
    for turn_idx, assist_idx in enumerate(assistant_indices[2:], start=2):
        print(f"\n{'─'*70}")
        print(f"TURN {turn_idx + 1} (index {assist_idx})")
        print(f"{'─'*70}")

        # Build prompt - ALWAYS use ground truth for context
        prompt_messages = [{"role": "system", "content": system_prompt}]

        # Add last 5 pairs of context (GROUND TRUTH)
        context_start = max(0, assist_idx - 10)
        for msg in messages[context_start:assist_idx]:
            prompt_messages.append({
                "role": msg["role"],
                "content": msg["text"]
            })

        # Show last user message
        if assist_idx > 0 and messages[assist_idx - 1]["role"] == "user":
            print(f"\n👤 USER: {messages[assist_idx - 1]['text']}")

        # Get reference (ground truth)
        reference = messages[assist_idx]["text"]
        print(f"\n📖 REFERENCE: {reference}")

        # Generate prediction
        print(f"Prompt messages: {prompt_messages}")
        prediction = generate_response(model, tokenizer, prompt_messages, device)

        print(f"\n🤖 PREDICTION: {prediction}")

        # Extract actions
        pred_actions = extractor.extract_actions(prediction)
        ref_actions = extractor.extract_actions(reference)

        print(f"\n📊 ACTIONS:")
        print(f"  Reference devices: {list(ref_actions.keys()) or 'none'}")
        print(f"  Predicted devices: {list(pred_actions.keys()) or 'none'}")

        # Compute confusion matrix
        confusion = compute_device_confusion(pred_actions, ref_actions)

        print(f"\n🎯 DEVICE CONFUSION MATRIX:")
        print(f"  ✓ True Positives (TP):  {confusion['tp'] or 'none'}")
        print(f"  ✗ False Positives (FP): {confusion['fp'] or 'none'}")
        print(f"  ✗ False Negatives (FN): {confusion['fn'] or 'none'}")
        print(f"  ✓ True Negatives (TN):  {len(confusion['tn'])} devices")

        # Compute action metrics with numerical parameters
        from scripts.action_metrics import ActionMetrics
        action_metrics = ActionMetrics.compare_actions(pred_actions, ref_actions)

        print(f"\n📊 PARAMETER METRICS:")
        print(f"  Numerical Parameters:")
        print(f"    TP: {action_metrics['numerical_tp']}, FP: {action_metrics['numerical_fp']}, FN: {action_metrics['numerical_fn']}")
        print(f"    Precision: {action_metrics['numerical_precision']:.3f}")
        print(f"    Recall: {action_metrics['numerical_recall']:.3f}")
        print(f"    F1: {action_metrics['numerical_f1']:.3f}")
        if action_metrics['numerical_mae'] > 0:
            print(f"    MAE: {action_metrics['numerical_mae']:.2f}")

        print(f"  Categorical Parameters:")
        print(f"    TP: {action_metrics['categorical_tp']}, FP: {action_metrics['categorical_fp']}, FN: {action_metrics['categorical_fn']}")
        print(f"    Precision: {action_metrics['categorical_precision']:.3f}")
        print(f"    Recall: {action_metrics['categorical_recall']:.3f}")
        print(f"    F1: {action_metrics['categorical_f1']:.3f}")

        # Store results
        all_results.append({
            "turn": turn_idx + 1,
            "index": assist_idx,
            "reference": reference,
            "prediction": prediction,
            "ref_actions": ref_actions,
            "pred_actions": pred_actions,
            "confusion": confusion,
            "numerical_metrics": {
                "tp": action_metrics['numerical_tp'],
                "fp": action_metrics['numerical_fp'],
                "fn": action_metrics['numerical_fn'],
                "precision": action_metrics['numerical_precision'],
                "recall": action_metrics['numerical_recall'],
                "f1": action_metrics['numerical_f1'],
                "mae": action_metrics['numerical_mae']
            },
            "categorical_metrics": {
                "tp": action_metrics['categorical_tp'],
                "fp": action_metrics['categorical_fp'],
                "fn": action_metrics['categorical_fn'],
                "precision": action_metrics['categorical_precision'],
                "recall": action_metrics['categorical_recall'],
                "f1": action_metrics['categorical_f1']
            }
        })

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    total_tp = sum(len(r["confusion"]["tp"]) for r in all_results)
    total_fp = sum(len(r["confusion"]["fp"]) for r in all_results)
    total_fn = sum(len(r["confusion"]["fn"]) for r in all_results)
    total_tn = sum(len(r["confusion"]["tn"]) for r in all_results)

    print(f"\nEvaluated {len(all_results)} turns")
    print(f"\nDevice-Level Aggregated Confusion Matrix:")
    print(f"  True Positives (TP):  {total_tp}")
    print(f"  False Positives (FP): {total_fp}")
    print(f"  False Negatives (FN): {total_fn}")
    print(f"  True Negatives (TN):  {total_tn}")

    if total_tp + total_fp > 0:
        precision = total_tp / (total_tp + total_fp)
        print(f"\nDevice Precision: {precision:.3f}")

    if total_tp + total_fn > 0:
        recall = total_tp / (total_tp + total_fn)
        print(f"Device Recall: {recall:.3f}")

    if total_tp + total_fp + total_fn + total_tn > 0:
        accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn)
        print(f"Device Accuracy: {accuracy:.3f}")

    # Numerical parameter summary
    num_tp = sum(r["numerical_metrics"]["tp"] for r in all_results)
    num_fp = sum(r["numerical_metrics"]["fp"] for r in all_results)
    num_fn = sum(r["numerical_metrics"]["fn"] for r in all_results)
    avg_num_precision = sum(r["numerical_metrics"]["precision"] for r in all_results) / len(all_results)
    avg_num_recall = sum(r["numerical_metrics"]["recall"] for r in all_results) / len(all_results)
    avg_num_f1 = sum(r["numerical_metrics"]["f1"] for r in all_results) / len(all_results)
    maes = [r["numerical_metrics"]["mae"] for r in all_results if r["numerical_metrics"]["mae"] > 0]
    avg_mae = sum(maes) / len(maes) if maes else 0.0

    print(f"\nNumerical Parameters:")
    print(f"  Total TP: {num_tp}, FP: {num_fp}, FN: {num_fn}")
    print(f"  Avg Precision: {avg_num_precision:.3f}")
    print(f"  Avg Recall: {avg_num_recall:.3f}")
    print(f"  Avg F1: {avg_num_f1:.3f}")
    if avg_mae > 0:
        print(f"  Avg MAE: {avg_mae:.2f}")

    # Categorical parameter summary
    cat_tp = sum(r["categorical_metrics"]["tp"] for r in all_results)
    cat_fp = sum(r["categorical_metrics"]["fp"] for r in all_results)
    cat_fn = sum(r["categorical_metrics"]["fn"] for r in all_results)
    avg_cat_precision = sum(r["categorical_metrics"]["precision"] for r in all_results) / len(all_results)
    avg_cat_recall = sum(r["categorical_metrics"]["recall"] for r in all_results) / len(all_results)
    avg_cat_f1 = sum(r["categorical_metrics"]["f1"] for r in all_results) / len(all_results)

    print(f"\nCategorical Parameters:")
    print(f"  Total TP: {cat_tp}, FP: {cat_fp}, FN: {cat_fn}")
    print(f"  Avg Precision: {avg_cat_precision:.3f}")
    print(f"  Avg Recall: {avg_cat_recall:.3f}")
    print(f"  Avg F1: {avg_cat_f1:.3f}")

    # Save results
    output_file = Path("results/sequential_eval_test.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump({
            "persona_id": persona_id,
            "total_turns_evaluated": len(all_results),
            "aggregated_confusion": {
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
                "tn": total_tn
            },
            "per_turn_results": all_results
        }, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()
