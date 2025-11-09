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
    for persona_id, persona_splits in splits.items():
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


def compute_metrics(predictions: list, references: list, config: dict):
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

        for pred, ref in zip(predictions, references):
            pred_actions = extractor.extract_actions(pred)
            ref_actions = extractor.extract_actions(ref)

            action_metrics = ActionMetrics.compare_actions(pred_actions, ref_actions)

            device_precisions.append(action_metrics["device_precision"])
            device_recalls.append(action_metrics["device_recall"])
            param_precisions.append(action_metrics["param_precision"])
            param_recalls.append(action_metrics["param_recall"])
            param_f1s.append(action_metrics["param_f1"])

        metrics["device_precision"] = np.mean(device_precisions)
        metrics["device_recall"] = np.mean(device_recalls)
        metrics["param_precision"] = np.mean(param_precisions)
        metrics["param_recall"] = np.mean(param_recalls)
        metrics["param_f1"] = np.mean(param_f1s)

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
    predictions = []
    references = []

    for dialogue in tqdm(dialogues, desc="Evaluating"):
        # Get the last assistant turn as target
        messages = dialogue["messages"]

        # Find last assistant message
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "assistant":
                try:
                    prompt_messages, reference = format_prompt(dialogue, config, up_to_turn=i)
                    prediction = generate_response(model, tokenizer, prompt_messages, config, device)

                    predictions.append(prediction)
                    references.append(reference)
                    break
                except Exception as e:
                    print(f"Error processing dialogue: {e}")
                    break

    # Compute metrics
    print(f"\nComputing metrics...")
    metrics = compute_metrics(predictions, references, config)

    # Display results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    for metric, value in metrics.items():
        print(f"{metric:30s}: {value:.4f}")

    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "config": config,
            "metrics": metrics,
            "num_examples": len(predictions)
        }

        # Convert numpy types to Python types for JSON serialization
        results = convert_to_serializable(results)

        results_file = output_dir / "baseline_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[OK] Results saved to {results_file}")

        # Save sample outputs
        samples_file = output_dir / "sample_outputs.jsonl"
        with open(samples_file, "w") as f:
            for i, (pred, ref) in enumerate(zip(predictions[:20], references[:20])):
                sample = {
                    "index": i,
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
