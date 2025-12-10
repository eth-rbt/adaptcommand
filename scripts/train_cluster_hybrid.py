"""
Hybrid Cluster LoRA Training

Train a small cluster-level LoRA adapter on top of the frozen unified adapter.
This mirrors the persona hybrid approach but groups multiple personas together.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from action_metrics import ActionExtractor, ActionMetrics


# ------------------------------------------------------------
# Data loading helpers
# ------------------------------------------------------------

def load_config(config_file: Path) -> dict:
    with open(config_file, "r") as f:
        return json.load(f)


def load_cluster_map(cluster_map_path: Path) -> Tuple[Dict[str, int], int]:
    with open(cluster_map_path, "r") as f:
        data = json.load(f)
    return data["cluster_map"], data.get("n_clusters", len(set(data["cluster_map"].values())))


def load_dialogues(dialogues_path: Path) -> List[Dict]:
    with open(dialogues_path, "r") as f:
        return [json.loads(line) for line in f]


def load_splits(splits_path: Path) -> Dict:
    with open(splits_path, "r") as f:
        return json.load(f)


def collect_cluster_dialogues(
    dialogues: List[Dict],
    splits: Dict,
    cluster_map: Dict[str, int],
    cluster_id: int,
    split_name: str,
) -> List[Dict]:
    persona_ids = [pid for pid, cid in cluster_map.items() if cid == cluster_id]
    indices = []

    for pid in persona_ids:
        persona_splits = splits.get(pid, {})
        indices.extend(persona_splits.get(split_name, []))

    return [dialogues[idx] for idx in indices]


# ------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------

def build_system_message(dialogue: Dict, base_prompt: str) -> str:
    parts = [base_prompt]

    character = dialogue.get("character")
    if character:
        parts.append(f"User Profile: {character}")

    meta = dialogue.get("meta")
    if meta:
        context_items = [f"{k}: {v}" for k, v in meta.items() if k != "routines" and v]
        if context_items:
            parts.append(f"Context: {', '.join(context_items)}")

    return "\n\n".join(parts)


def create_training_examples(
    dialogues: List[Dict],
    tokenizer,
    max_length: int,
    system_prompt: str,
) -> List[str]:
    training_texts = []

    for dialogue in dialogues:
        messages = dialogue["messages"]
        system_message = build_system_message(dialogue, system_prompt)

        for i, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue

            context_messages = messages[: i + 1]
            chat_messages = [{"role": "system", "content": system_message}]

            for m in context_messages:
                chat_messages.append({"role": m["role"], "content": m["text"]})

            text = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            training_texts.append(text)

    return training_texts


def tokenize_function(examples, tokenizer, max_length):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

def evaluate_with_generation(
    model,
    tokenizer,
    dialogues: List[Dict],
    system_prompt: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Dict[str, float]:
    model.eval()

    predictions = []
    references = []

    print(f"\nGenerating responses for {len(dialogues)} dialogues...")

    for dialogue in tqdm(dialogues, desc="Evaluating"):
        messages = dialogue["messages"]
        system_message = build_system_message(dialogue, system_prompt)

        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] != "assistant":
                continue

            context_messages = messages[:i]
            chat_messages = [{"role": "system", "content": system_message}]
            for m in context_messages:
                chat_messages.append({"role": m["role"], "content": m["text"]})

            prompt = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated_text = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            ).strip()

            predictions.append(generated_text)
            references.append(messages[i]["text"])
            break

    print(f"\nComputing metrics on {len(predictions)} examples...")
    metrics = {}

    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    pred_embeddings = embedding_model.encode(predictions)
    ref_embeddings = embedding_model.encode(references)

    similarities = []
    for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
        sim = np.dot(pred_emb, ref_emb) / (np.linalg.norm(pred_emb) * np.linalg.norm(ref_emb))
        similarities.append(sim)

    metrics["embedding_similarity"] = float(np.mean(similarities))

    extractor = ActionExtractor()

    device_precisions = []
    device_recalls = []
    param_precisions = []
    param_recalls = []
    param_f1s = []
    numerical_precisions = []
    numerical_recalls = []

    for pred, ref in zip(predictions, references):
        pred_actions = extractor.extract_actions(pred)
        ref_actions = extractor.extract_actions(ref)

        action_metrics = ActionMetrics.compare_actions(pred_actions, ref_actions)

        device_precisions.append(action_metrics["device_precision"])
        device_recalls.append(action_metrics["device_recall"])
        param_precisions.append(action_metrics["param_precision"])
        param_recalls.append(action_metrics["param_recall"])
        param_f1s.append(action_metrics["param_f1"])
        numerical_precisions.append(action_metrics["numerical_precision"])
        numerical_recalls.append(action_metrics["numerical_recall"])

    metrics["device_precision"] = float(np.mean(device_precisions))
    metrics["device_recall"] = float(np.mean(device_recalls))
    metrics["param_precision"] = float(np.mean(param_precisions))
    metrics["param_recall"] = float(np.mean(param_recalls))
    metrics["param_f1"] = float(np.mean(param_f1s))
    metrics["numerical_precision"] = float(np.mean(numerical_precisions))
    metrics["numerical_recall"] = float(np.mean(numerical_recalls))

    pred_lengths = [len(p.split()) for p in predictions]
    ref_lengths = [len(r.split()) for r in references]

    metrics["avg_pred_length"] = float(np.mean(pred_lengths))
    metrics["avg_ref_length"] = float(np.mean(ref_lengths))
    metrics["length_ratio"] = (
        float(np.mean(pred_lengths) / np.mean(ref_lengths)) if np.mean(ref_lengths) > 0 else 0
    )

    return metrics


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

def train_cluster_hybrid(
    cluster_id: int,
    unified_model_path: Path,
    config: dict,
    cluster_map_path: Path,
    output_base: Path,
    cluster_lora_rank: int = 8,
    use_val: bool = True,
    batch_size: int = 1,
    grad_accum: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 1e-4,
):
    print("\n" + "=" * 80)
    print(f"Training Cluster {cluster_id} Hybrid Adapter (rank={cluster_lora_rank})")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    base_model_name = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    if device == "mps":
        base_model = base_model.to(device)

    print(f"\nLoading unified adapter from {unified_model_path}...")
    model_with_unified = PeftModel.from_pretrained(base_model, str(unified_model_path))
    merged_model = model_with_unified.merge_and_unload()
    print("[OK] Unified adapter merged into base model")

    lora_config = LoraConfig(
        r=cluster_lora_rank,
        lora_alpha=cluster_lora_rank * 2,
        target_modules=config["lora_config"]["target_modules"],
        lora_dropout=0.2,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(merged_model, lora_config)
    model.print_trainable_parameters()

    dialogues = load_dialogues(Path(config["data_path"]))
    splits = load_splits(Path(config.get("splits_path", "data/splits/edgesplits.json")))
    cluster_map, _ = load_cluster_map(cluster_map_path)

    train_dialogues = collect_cluster_dialogues(dialogues, splits, cluster_map, cluster_id, "train")
    val_dialogues = collect_cluster_dialogues(dialogues, splits, cluster_map, cluster_id, "val")
    test_dialogues = collect_cluster_dialogues(dialogues, splits, cluster_map, cluster_id, "test")

    print(f"Train dialogues: {len(train_dialogues)} | Val: {len(val_dialogues)} | Test: {len(test_dialogues)}")

    combined_train = train_dialogues + val_dialogues

    system_prompt = "You are a helpful and personalized smart home assistant."

    train_texts = create_training_examples(combined_train, tokenizer, config["max_length"], system_prompt)
    train_dataset = Dataset.from_dict({"text": train_texts})

    val_dataset = None
    if use_val and test_dialogues:
        val_texts = create_training_examples(test_dialogues, tokenizer, config["max_length"], system_prompt)
        val_dataset = Dataset.from_dict({"text": val_texts})

    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, config["max_length"]),
        batched=True,
        remove_columns=["text"],
    )

    if val_dataset is not None:
        val_dataset = val_dataset.map(
            lambda x: tokenize_function(x, tokenizer, config["max_length"]),
            batched=True,
            remove_columns=["text"],
        )

    output_dir = output_base / f"cluster_{cluster_id:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.05,
        logging_dir=str(output_dir / "logs"),
        logging_steps=20,
        save_strategy="epoch",
        eval_strategy="epoch" if use_val and val_dataset is not None else "no",
        load_best_model_at_end=use_val and val_dataset is not None,
        metric_for_best_model="eval_loss" if use_val and val_dataset is not None else None,
        greater_is_better=False if use_val and val_dataset is not None else None,
        save_total_limit=2,
        fp16=True,
        optim="adamw_torch",
        report_to="none",
        seed=config["seed"],
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("\nStarting training...")
    train_result = trainer.train()

    print(f"\nSaving cluster adapter to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    cluster_personas = [pid for pid, cid in cluster_map.items() if cid == cluster_id]
    with open(output_dir / "cluster_info.json", "w") as f:
        json.dump(
            {
                "cluster_id": cluster_id,
                "personas": cluster_personas,
                "num_personas": len(cluster_personas),
                "num_train_dialogues": len(train_dialogues),
                "num_val_dialogues": len(val_dialogues),
                "num_test_dialogues": len(test_dialogues),
                "cluster_lora_rank": cluster_lora_rank,
                "unified_model_path": str(unified_model_path),
            },
            f,
            indent=2,
        )

    config_to_save = {
        "base_model": base_model_name,
        "unified_model_path": str(unified_model_path),
        "cluster_id": cluster_id,
        "cluster_lora_rank": cluster_lora_rank,
        "training_config": config,
        "hyperparams": {
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
        },
    }
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config_to_save, f, indent=2)

    print("\nEvaluating on test split (cluster adapter + unified base)...")
    test_metrics = evaluate_with_generation(
        model=model,
        tokenizer=tokenizer,
        dialogues=test_dialogues,
        system_prompt=system_prompt,
        device=device,
        max_new_tokens=config["generation"].get("max_new_tokens", 256),
        temperature=config["generation"].get("temperature", 0.7),
        top_p=config["generation"].get("top_p", 0.9),
    )

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print("\nTest Metrics (Cluster Hybrid):")
    for key, value in test_metrics.items():
        print(f"  {key:25s}: {value:.4f}")

    return output_dir, test_metrics, cluster_personas


# ------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------

def aggregate_results(results: List[Dict], output_file: Path):
    if not results:
        print("[WARN] No cluster results to aggregate")
        return

    metric_keys = list(results[0]["metrics"].keys())

    summary = {
        "num_clusters": len(results),
        "clusters": results,
    }

    for key in metric_keys:
        values = [r["metrics"][key] for r in results]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
        summary[f"{key}_min"] = float(np.min(values))
        summary[f"{key}_max"] = float(np.max(values))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] Aggregated results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Train cluster-level hybrid LoRA adapters on top of unified weights",
    )
    parser.add_argument("--cluster_id", type=int, default=None, help="Train a specific cluster (default: all)")
    parser.add_argument(
        "--unified_model",
        type=str,
        default="models/lora_unified",
        help="Path to unified LoRA adapter",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/lora_training.json",
        help="Path to base training config",
    )
    parser.add_argument(
        "--cluster_map",
        type=str,
        default="data/splits/cluster_map.json",
        help="Path to cluster assignment file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/lora_cluster_hybrid",
        help="Base output directory",
    )
    parser.add_argument("--cluster_rank", type=int, default=8, help="LoRA rank for cluster adapter")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--no_val", action="store_true", help="Skip validation during training")
    parser.add_argument("--start_cluster", type=int, default=0, help="Start cluster index (for all mode)")
    parser.add_argument("--end_cluster", type=int, default=None, help="End cluster index (exclusive) for all mode")
    parser.add_argument("--skip_existing", action="store_true", help="Skip clusters with existing test metrics")

    args = parser.parse_args()

    config = load_config(Path(args.config))
    unified_model_path = Path(args.unified_model)
    cluster_map_path = Path(args.cluster_map)
    output_base = Path(args.output_dir)

    if args.cluster_id is not None:
        output_dir, test_metrics, personas = train_cluster_hybrid(
            cluster_id=args.cluster_id,
            unified_model_path=unified_model_path,
            config=config,
            cluster_map_path=cluster_map_path,
            output_base=output_base,
            cluster_lora_rank=args.cluster_rank,
            use_val=not args.no_val,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            num_epochs=args.epochs,
            learning_rate=args.lr,
        )

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Cluster adapter saved to: {output_dir}")
        print(f"Personas covered: {len(personas)}")
        return

    cluster_map, n_clusters = load_cluster_map(cluster_map_path)
    cluster_ids = sorted(set(cluster_map.values()))

    if args.end_cluster is not None:
        cluster_ids = [cid for cid in cluster_ids if args.start_cluster <= cid < args.end_cluster]
    else:
        cluster_ids = [cid for cid in cluster_ids if cid >= args.start_cluster]

    print(f"Training {len(cluster_ids)} clusters (indices {cluster_ids[0]}..{cluster_ids[-1]})")

    results = []

    for idx, cid in enumerate(cluster_ids, 1):
        print(f"\n{'#' * 80}")
        print(f"# Cluster {cid} ({idx}/{len(cluster_ids)})")
        print(f"{'#' * 80}")

        cluster_dir = output_base / f"cluster_{cid:02d}"
        metrics_file = cluster_dir / "test_metrics.json"

        if args.skip_existing and metrics_file.exists():
            print(f"[SKIP] Found existing results at {metrics_file}")
            with open(metrics_file, "r") as f:
                existing_metrics = json.load(f)
            with open(cluster_dir / "cluster_info.json", "r") as f:
                info = json.load(f)
            results.append(
                {
                    "cluster_id": cid,
                    "num_personas": info.get("num_personas"),
                    "metrics": existing_metrics,
                    "path": str(cluster_dir),
                }
            )
            continue

        try:
            output_dir, test_metrics, personas = train_cluster_hybrid(
                cluster_id=cid,
                unified_model_path=unified_model_path,
                config=config,
                cluster_map_path=cluster_map_path,
                output_base=output_base,
                cluster_lora_rank=args.cluster_rank,
                use_val=not args.no_val,
                batch_size=args.batch_size,
                grad_accum=args.grad_accum,
                num_epochs=args.epochs,
                learning_rate=args.lr,
            )
            results.append(
                {
                    "cluster_id": cid,
                    "num_personas": len(personas),
                    "metrics": test_metrics,
                    "path": str(output_dir),
                }
            )
        except Exception as exc:
            print(f"[ERROR] Cluster {cid} failed: {exc}")
            continue

    summary_path = Path("results/hybrid_cluster/cluster_hybrid_summary.json")
    aggregate_results(results, summary_path)


if __name__ == "__main__":
    main()
