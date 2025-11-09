"""
Quick Baseline Benchmark Script

This script runs a baseline evaluation on the smart home dataset
using a small model for fast iteration.
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
import numpy as np
from action_metrics import ActionExtractor, ActionMetrics


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def load_config(config_file: Path):
    """Load configuration."""
    with open(config_file, "r") as f:
        return json.load(f)


def load_data(dialogues_file: Path, splits_file: Path, split_name: str, max_examples: int = None):
    """Load dialogues and filter by split."""
    # Load dialogues
    dialogues = []
    with open(dialogues_file, "r") as f:
        for line in f:
            dialogues.append(json.loads(line))

    # Load splits
    with open(splits_file, "r") as f:
        splits = json.load(f)

    # Get indices for the specified split
    split_dialogues = []
    for persona_splits in splits.values():
        for idx in persona_splits[split_name]:
            split_dialogues.append(dialogues[idx])

    # Limit examples if specified
    if max_examples:
        split_dialogues = split_dialogues[:max_examples]

    return split_dialogues


def format_prompt(dialogue: dict, config: dict, up_to_turn: int = -1):
    """
    Format a dialogue into a prompt for the model.

    Args:
        dialogue: Dialogue entry
        config: Config dict
        up_to_turn: Generate response for this turn (0-indexed). -1 means last turn.

    Returns:
        (prompt_messages, reference_response)
    """
    messages = dialogue["messages"]

    # Determine which turn to predict
    if up_to_turn == -1:
        up_to_turn = len(messages) - 1

    # Get messages up to (but not including) the target turn
    context_messages = messages[:up_to_turn]
    target_message = messages[up_to_turn]

    # Ensure target is assistant response
    if target_message["role"] != "assistant":
        raise ValueError(f"Target turn must be assistant response, got {target_message['role']}")

    # Build prompt messages
    prompt_messages = []

    # Add system prompt
    if config["prompt"]["system_template"]:
        prompt_messages.append({
            "role": "system",
            "content": config["prompt"]["system_template"]
        })

    # Add context (limit to max_context_messages)
    max_context = config["prompt"].get("max_context_messages", 5)
    start_idx = max(0, len(context_messages) - max_context * 2)  # *2 for user+assistant pairs

    for msg in context_messages[start_idx:]:
        prompt_messages.append({
            "role": msg["role"],
            "content": msg["text"]
        })

    reference_response = target_message["text"]

    return prompt_messages, reference_response


def generate_response(model, tokenizer, messages: list, config: dict, device: str):
    """Generate a response using the model."""
    # Format messages using chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate
    gen_config = config["generation"]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_config["max_new_tokens"],
            temperature=gen_config["temperature"],
            top_p=gen_config["top_p"],
            do_sample=gen_config["do_sample"],
            repetition_penalty=gen_config.get("repetition_penalty", 1.0),
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode
    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return generated_text.strip()


def compute_device_confusion(pred_actions: dict, ref_actions: dict) -> dict:
    """
    Compute device-level confusion matrix (TP/FP/TN/FN).

    Args:
        pred_actions: Predicted device actions
        ref_actions: Reference device actions

    Returns:
        Dict with per-device TP/FP/TN/FN counts
    """
    # All possible devices (from ActionExtractor)
    all_devices = ["tv", "ac", "lights", "speaker", "security"]

    confusion = {
        "tp": [],  # True positives: predicted and in reference
        "fp": [],  # False positives: predicted but not in reference
        "tn": [],  # True negatives: not predicted and not in reference
        "fn": []   # False negatives: not predicted but in reference
    }

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
        else:  # not in_pred and not in_ref
            confusion["tn"].append(device)

    return confusion


def compute_per_persona_metrics(predictions: list, references: list, persona_ids: list,
                                  device_detections: list, config: dict) -> dict:
    """Compute metrics grouped by persona."""
    from collections import defaultdict

    # Group data by persona
    persona_data = defaultdict(lambda: {
        "predictions": [],
        "references": [],
        "device_detections": []
    })

    for pred, ref, persona_id, device_det in zip(predictions, references, persona_ids, device_detections):
        persona_data[persona_id]["predictions"].append(pred)
        persona_data[persona_id]["references"].append(ref)
        persona_data[persona_id]["device_detections"].append(device_det)

    # Compute metrics for each persona
    per_persona_results = {}

    for persona_id, data in persona_data.items():
        persona_metrics = compute_metrics(
            data["predictions"],
            data["references"],
            config,
            data["device_detections"]
        )
        per_persona_results[persona_id] = {
            "num_predictions": len(data["predictions"]),
            "metrics": persona_metrics
        }

    return per_persona_results


def compute_metrics(predictions: list, references: list, config: dict, device_detections: list = None):
    """Compute evaluation metrics."""
    metrics = {}

    # ROUGE scores
    if "rouge" in config["evaluation"]["metrics"]:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = {
            'rouge1': [],
            'rouge2': [],
            'rougeL': []
        }

        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            for key in rouge_scores:
                rouge_scores[key].append(scores[key].fmeasure)

        for key in rouge_scores:
            metrics[f"{key}_f1"] = np.mean(rouge_scores[key])

    # Embedding similarity
    if "embedding_similarity" in config["evaluation"]["metrics"]:
        embedding_model_name = config["evaluation"]["embedding_model"]
        embedding_model = SentenceTransformer(embedding_model_name)

        pred_embeddings = embedding_model.encode(predictions)
        ref_embeddings = embedding_model.encode(references)

        # Cosine similarity
        similarities = []
        for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
            sim = np.dot(pred_emb, ref_emb) / (np.linalg.norm(pred_emb) * np.linalg.norm(ref_emb))
            similarities.append(sim)

        metrics["embedding_similarity"] = np.mean(similarities)

    # Action-based metrics (smart home specific)
    if "action_accuracy" in config["evaluation"]["metrics"]:
        extractor = ActionExtractor()

        # Extract actions from all predictions and references
        device_precisions = []
        device_recalls = []
        param_precisions = []
        param_recalls = []
        param_f1s = []

        # Numerical parameter metrics
        numerical_tps = []
        numerical_fps = []
        numerical_fns = []
        numerical_precisions = []
        numerical_recalls = []
        numerical_f1s = []
        numerical_maes = []

        # Categorical parameter metrics
        categorical_tps = []
        categorical_fps = []
        categorical_fns = []
        categorical_precisions = []
        categorical_recalls = []
        categorical_f1s = []

        for pred, ref in zip(predictions, references):
            pred_actions = extractor.extract_actions(pred)
            ref_actions = extractor.extract_actions(ref)

            action_metrics = ActionMetrics.compare_actions(pred_actions, ref_actions)

            device_precisions.append(action_metrics["device_precision"])
            device_recalls.append(action_metrics["device_recall"])
            param_precisions.append(action_metrics["param_precision"])
            param_recalls.append(action_metrics["param_recall"])
            param_f1s.append(action_metrics["param_f1"])

            # Numerical metrics
            numerical_tps.append(action_metrics["numerical_tp"])
            numerical_fps.append(action_metrics["numerical_fp"])
            numerical_fns.append(action_metrics["numerical_fn"])
            numerical_precisions.append(action_metrics["numerical_precision"])
            numerical_recalls.append(action_metrics["numerical_recall"])
            numerical_f1s.append(action_metrics["numerical_f1"])
            numerical_maes.append(action_metrics["numerical_mae"])

            # Categorical metrics
            categorical_tps.append(action_metrics["categorical_tp"])
            categorical_fps.append(action_metrics["categorical_fp"])
            categorical_fns.append(action_metrics["categorical_fn"])
            categorical_precisions.append(action_metrics["categorical_precision"])
            categorical_recalls.append(action_metrics["categorical_recall"])
            categorical_f1s.append(action_metrics["categorical_f1"])

        metrics["device_precision"] = np.mean(device_precisions)
        metrics["device_recall"] = np.mean(device_recalls)
        metrics["param_precision"] = np.mean(param_precisions)
        metrics["param_recall"] = np.mean(param_recalls)
        metrics["param_f1"] = np.mean(param_f1s)

        # Aggregate numerical metrics
        metrics["numerical_tp_total"] = int(np.sum(numerical_tps))
        metrics["numerical_fp_total"] = int(np.sum(numerical_fps))
        metrics["numerical_fn_total"] = int(np.sum(numerical_fns))
        metrics["numerical_precision"] = np.mean(numerical_precisions)
        metrics["numerical_recall"] = np.mean(numerical_recalls)
        metrics["numerical_f1"] = np.mean(numerical_f1s)
        metrics["numerical_mae"] = np.mean([mae for mae in numerical_maes if mae > 0]) if any(mae > 0 for mae in numerical_maes) else 0.0

        # Aggregate categorical metrics
        metrics["categorical_tp_total"] = int(np.sum(categorical_tps))
        metrics["categorical_fp_total"] = int(np.sum(categorical_fps))
        metrics["categorical_fn_total"] = int(np.sum(categorical_fns))
        metrics["categorical_precision"] = np.mean(categorical_precisions)
        metrics["categorical_recall"] = np.mean(categorical_recalls)
        metrics["categorical_f1"] = np.mean(categorical_f1s)

    # Device confusion matrix (if available)
    if device_detections is not None and len(device_detections) > 0:
        # Aggregate all TP/FP/TN/FN counts
        total_tp = sum(len(d["tp"]) for d in device_detections)
        total_fp = sum(len(d["fp"]) for d in device_detections)
        total_tn = sum(len(d["tn"]) for d in device_detections)
        total_fn = sum(len(d["fn"]) for d in device_detections)

        metrics["device_tp"] = total_tp
        metrics["device_fp"] = total_fp
        metrics["device_tn"] = total_tn
        metrics["device_fn"] = total_fn

        # Compute metrics from confusion matrix
        if total_tp + total_fp > 0:
            metrics["device_precision_from_confusion"] = total_tp / (total_tp + total_fp)
        else:
            metrics["device_precision_from_confusion"] = 0.0

        if total_tp + total_fn > 0:
            metrics["device_recall_from_confusion"] = total_tp / (total_tp + total_fn)
        else:
            metrics["device_recall_from_confusion"] = 0.0

        if total_tp + total_fp + total_fn + total_tn > 0:
            metrics["device_accuracy"] = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn)
        else:
            metrics["device_accuracy"] = 0.0

        # F1 score
        prec = metrics["device_precision_from_confusion"]
        rec = metrics["device_recall_from_confusion"]
        if prec + rec > 0:
            metrics["device_f1"] = 2 * prec * rec / (prec + rec)
        else:
            metrics["device_f1"] = 0.0

        # Per-device breakdown
        device_counts = {"tp": {}, "fp": {}, "tn": {}, "fn": {}}
        for device in ["tv", "ac", "lights", "speaker", "security"]:
            for category in ["tp", "fp", "tn", "fn"]:
                device_counts[category][device] = sum(1 for d in device_detections if device in d[category])

        metrics["device_breakdown"] = device_counts

    # Length statistics
    pred_lengths = [len(p.split()) for p in predictions]
    ref_lengths = [len(r.split()) for r in references]

    metrics["avg_pred_length"] = np.mean(pred_lengths)
    metrics["avg_ref_length"] = np.mean(ref_lengths)
    metrics["length_ratio"] = np.mean(pred_lengths) / np.mean(ref_lengths) if np.mean(ref_lengths) > 0 else 0

    return metrics


def run_benchmark(config_file: Path, max_examples: int = None, output_dir: Path = None):
    """Run the baseline benchmark."""
    print("="*60)
    print("BASELINE BENCHMARK")
    print("="*60)

    # Load config
    print(f"\nLoading config from {config_file}...")
    config = load_config(config_file)

    # Set device
    device = config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model and tokenizer
    model_name = config["model"]["name"]
    print(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32,
        device_map=device if device == "cuda" else None
    )

    if device == "mps":
        model = model.to(device)

    model.eval()
    print(f"[OK] Model loaded")

    # Load data
    data_config = config["data"]
    split_name = data_config["eval_split"]
    max_ex = max_examples or data_config.get("max_examples")

    print(f"\nLoading {split_name} split...")
    dialogues = load_data(
        Path(data_config["dialogues_file"]),
        Path(data_config["splits_file"]),
        split_name,
        max_ex
    )
    print(f"[OK] Loaded {len(dialogues)} dialogues")

    # Run evaluation
    print(f"\nRunning evaluation...")
    print(f"Mode: Sequential turn-by-turn prediction with ground truth context")

    predictions = []
    references = []
    all_device_detections = []  # Track TP/FP/TN/FN for each prediction
    persona_ids = []  # Track which persona each prediction belongs to
    dialogue_ids = []  # Track which dialogue each prediction belongs to
    turn_indices = []  # Track which turn index within dialogue

    for dialogue in tqdm(dialogues, desc="Evaluating"):
        messages = dialogue["messages"]
        persona_id = dialogue.get("persona_id", "unknown")
        session_id = dialogue.get("session_id", "unknown")

        # Find all assistant message indices
        assistant_indices = [i for i, msg in enumerate(messages) if msg["role"] == "assistant"]

        # Skip first 2 assistant messages, evaluate the rest
        if len(assistant_indices) <= 2:
            continue

        # Evaluate each assistant turn (starting from 3rd)
        for turn_idx, assist_idx in enumerate(assistant_indices[2:], start=2):
            try:
                # Build prompt messages manually for this turn
                prompt_messages = []

                # Add system prompt
                if config["prompt"]["system_template"]:
                    prompt_messages.append({
                        "role": "system",
                        "content": config["prompt"]["system_template"]
                    })

                # Add context messages (up to but not including current turn)
                # ALWAYS use ground truth messages for context
                # Limit to max_context_messages
                max_context = config["prompt"].get("max_context_messages", 5)
                context_start = max(0, assist_idx - max_context * 2)  # *2 for user+assistant pairs

                for msg in messages[context_start:assist_idx]:
                    prompt_messages.append({
                        "role": msg["role"],
                        "content": msg["text"]
                    })

                # Get reference (ground truth)
                reference = messages[assist_idx]["text"]

                # Generate prediction
                prediction = generate_response(model, tokenizer, prompt_messages, config, device)

                predictions.append(prediction)
                references.append(reference)
                persona_ids.append(persona_id)
                dialogue_ids.append(session_id)
                turn_indices.append(turn_idx)

                # Track device detections for confusion matrix
                extractor = ActionExtractor()
                pred_actions = extractor.extract_actions(prediction)
                ref_actions = extractor.extract_actions(reference)

                device_detection = compute_device_confusion(pred_actions, ref_actions)
                all_device_detections.append(device_detection)

            except Exception as e:
                print(f"\nError processing dialogue turn {turn_idx}: {e}")
                continue

    # Compute metrics
    print(f"\nComputing metrics...")
    print(f"Total predictions: {len(predictions)}")
    print(f"Unique personas: {len(set(persona_ids))}")

    # Global metrics (across all personas)
    metrics = compute_metrics(predictions, references, config, all_device_detections)

    # Per-persona metrics
    print(f"\nComputing per-persona metrics...")
    per_persona_metrics = compute_per_persona_metrics(
        predictions, references, persona_ids, all_device_detections, config
    )

    # Display results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    # Group metrics for better readability
    print("\n--- Embedding Metrics ---")
    if "embedding_similarity" in metrics:
        print(f"{'embedding_similarity':30s}: {metrics['embedding_similarity']:.4f}")

    print("\n--- Device Detection ---")
    device_metrics = ["device_precision", "device_recall", "device_tp", "device_fp",
                     "device_tn", "device_fn", "device_precision_from_confusion",
                     "device_recall_from_confusion", "device_accuracy", "device_f1"]
    for metric in device_metrics:
        if metric in metrics:
            if isinstance(metrics[metric], float):
                print(f"{metric:30s}: {metrics[metric]:.4f}")
            else:
                print(f"{metric:30s}: {metrics[metric]}")

    print("\n--- Parameter Metrics (Overall) ---")
    param_metrics = ["param_precision", "param_recall", "param_f1"]
    for metric in param_metrics:
        if metric in metrics:
            print(f"{metric:30s}: {metrics[metric]:.4f}")

    print("\n--- Numerical Parameters ---")
    num_metrics = ["numerical_tp_total", "numerical_fp_total", "numerical_fn_total",
                   "numerical_precision", "numerical_recall", "numerical_f1", "numerical_mae"]
    for metric in num_metrics:
        if metric in metrics:
            if isinstance(metrics[metric], float):
                print(f"{metric:30s}: {metrics[metric]:.4f}")
            else:
                print(f"{metric:30s}: {metrics[metric]}")

    print("\n--- Categorical Parameters ---")
    cat_metrics = ["categorical_tp_total", "categorical_fp_total", "categorical_fn_total",
                   "categorical_precision", "categorical_recall", "categorical_f1"]
    for metric in cat_metrics:
        if metric in metrics:
            if isinstance(metrics[metric], float):
                print(f"{metric:30s}: {metrics[metric]:.4f}")
            else:
                print(f"{metric:30s}: {metrics[metric]}")

    print("\n--- Length Statistics ---")
    length_metrics = ["avg_pred_length", "avg_ref_length", "length_ratio"]
    for metric in length_metrics:
        if metric in metrics:
            print(f"{metric:30s}: {metrics[metric]:.4f}")

    # Display per-persona summary
    print(f"\n{'='*60}")
    print("PER-PERSONA SUMMARY")
    print(f"{'='*60}")
    print(f"Total personas evaluated: {len(per_persona_metrics)}")

    # Show sample of per-persona metrics
    sample_personas = list(per_persona_metrics.keys())[:5]
    print(f"\nSample personas (showing first 5):")
    for persona_id in sample_personas:
        persona_data = per_persona_metrics[persona_id]
        print(f"\n  {persona_id}:")
        print(f"    Predictions: {persona_data['num_predictions']}")
        print(f"    Embedding similarity: {persona_data['metrics'].get('embedding_similarity', 0):.4f}")
        print(f"    Device precision: {persona_data['metrics'].get('device_precision', 0):.4f}")
        print(f"    Device recall: {persona_data['metrics'].get('device_recall', 0):.4f}")

    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save global results
        results = {
            "config": config,
            "metrics": metrics,
            "num_examples": len(predictions),
            "num_personas": len(set(persona_ids))
        }

        # Convert numpy types to Python types for JSON serialization
        results = convert_to_serializable(results)

        results_file = output_dir / "baseline_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[OK] Results saved to {results_file}")

        # Save per-persona results
        per_persona_results_file = output_dir / "per_persona_results.json"
        per_persona_serializable = convert_to_serializable(per_persona_metrics)
        with open(per_persona_results_file, "w") as f:
            json.dump(per_persona_serializable, f, indent=2)
        print(f"[OK] Per-persona results saved to {per_persona_results_file}")

        # Save per-persona summary as CSV for easy analysis
        import pandas as pd
        persona_summary = []
        for persona_id, data in per_persona_metrics.items():
            row = {
                "persona_id": persona_id,
                "num_predictions": data["num_predictions"],
                "embedding_similarity": data["metrics"].get("embedding_similarity", 0),
                "device_precision": data["metrics"].get("device_precision", 0),
                "device_recall": data["metrics"].get("device_recall", 0),
                "param_precision": data["metrics"].get("param_precision", 0),
                "param_recall": data["metrics"].get("param_recall", 0),
                "param_f1": data["metrics"].get("param_f1", 0),
                "numerical_precision": data["metrics"].get("numerical_precision", 0),
                "numerical_recall": data["metrics"].get("numerical_recall", 0),
                "numerical_f1": data["metrics"].get("numerical_f1", 0),
                "categorical_precision": data["metrics"].get("categorical_precision", 0),
                "categorical_recall": data["metrics"].get("categorical_recall", 0),
                "categorical_f1": data["metrics"].get("categorical_f1", 0),
            }
            persona_summary.append(row)

        df = pd.DataFrame(persona_summary)
        csv_file = output_dir / "per_persona_summary.csv"
        df.to_csv(csv_file, index=False)
        print(f"[OK] Per-persona summary CSV saved to {csv_file}")

        # Save sample outputs
        samples_file = output_dir / "sample_outputs.jsonl"
        with open(samples_file, "w") as f:
            for i, (pred, ref, persona_id) in enumerate(zip(predictions[:20], references[:20], persona_ids[:20])):
                sample = {
                    "index": i,
                    "persona_id": persona_id,
                    "prediction": pred,
                    "reference": ref
                }
                f.write(json.dumps(sample) + "\n")
        print(f"[OK] Sample outputs saved to {samples_file}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run baseline benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline_v1.0.json",
        help="Path to config file"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate (for quick testing)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/baseline",
        help="Output directory for results"
    )

    args = parser.parse_args()

    run_benchmark(
        Path(args.config),
        args.max_examples,
        Path(args.output_dir)
    )


if __name__ == "__main__":
    main()
