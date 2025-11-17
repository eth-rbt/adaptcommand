"""
Hybrid LoRA Training: Persona-Specific Adapter on Top of Unified Model

This script:
1. Loads the unified LoRA model (trained on all personas)
2. Adds a smaller persona-specific LoRA adapter on top
3. Trains only the persona adapter while keeping unified adapter frozen
4. Combines benefits of shared knowledge + personalization
"""

import json
import argparse
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from action_metrics import ActionExtractor, ActionMetrics


def load_config(config_file: Path) -> dict:
    """Load training configuration."""
    with open(config_file, "r") as f:
        return json.load(f)


def load_persona_data(dialogues_file: Path, splits_file: Path, persona_id: str, split_name: str) -> List[Dict]:
    """
    Load all dialogues for a specific persona and split.
    """
    # Load all dialogues
    dialogues = []
    with open(dialogues_file, "r") as f:
        for line in f:
            dialogues.append(json.loads(line))

    # Load splits
    with open(splits_file, "r") as f:
        splits = json.load(f)

    # Get indices for this persona's split
    if persona_id not in splits:
        raise ValueError(f"Persona {persona_id} not found in splits file")

    persona_splits = splits[persona_id]
    if split_name not in persona_splits:
        raise ValueError(f"Split {split_name} not found for persona {persona_id}")

    indices = persona_splits[split_name]

    # Get dialogues for these indices
    persona_dialogues = [dialogues[idx] for idx in indices]

    return persona_dialogues


def format_chat_messages(messages: List[Dict], system_prompt: str = None) -> List[Dict]:
    """Format dialogue messages into chat format."""
    formatted_messages = []

    if system_prompt:
        formatted_messages.append({
            "role": "system",
            "content": system_prompt
        })

    for msg in messages:
        formatted_messages.append({
            "role": msg["role"],
            "content": msg["text"]
        })

    return formatted_messages


def create_training_examples(dialogues: List[Dict], tokenizer, max_length: int = 512,
                            system_prompt: str = None) -> List[str]:
    """
    Create training examples from dialogues.
    Creates one training example for EACH assistant turn.
    """
    training_texts = []

    for dialogue in dialogues:
        messages = dialogue["messages"]

        # Create one training example for each assistant turn
        for i, msg in enumerate(messages):
            if msg["role"] == "assistant":
                context_messages = messages[:i+1]
                formatted_messages = format_chat_messages(context_messages, system_prompt)

                text = tokenizer.apply_chat_template(
                    formatted_messages,
                    tokenize=False,
                    add_generation_prompt=False
                )

                training_texts.append(text)

    return training_texts


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize examples for training."""
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None
    )
    return outputs


def evaluate_with_generation(
    model,
    tokenizer,
    dialogues: List[Dict],
    system_prompt: str,
    device: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> Dict[str, float]:
    """Evaluate model by generating responses and computing metrics."""
    model.eval()

    predictions = []
    references = []

    print(f"\nGenerating responses for {len(dialogues)} dialogues...")

    for dialogue in tqdm(dialogues, desc="Evaluating"):
        messages = dialogue["messages"]

        # Find last assistant message to predict
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "assistant":
                context_messages = messages[:i]

                formatted_messages = []
                if system_prompt:
                    formatted_messages.append({
                        "role": "system",
                        "content": system_prompt
                    })

                for msg in context_messages:
                    formatted_messages.append({
                        "role": msg["role"],
                        "content": msg["text"]
                    })

                # Generate
                prompt = tokenizer.apply_chat_template(
                    formatted_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                inputs = tokenizer(prompt, return_tensors="pt").to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )

                generated_text = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip()

                predictions.append(generated_text)
                references.append(messages[i]["text"])
                break

    # Compute metrics
    print(f"\nComputing metrics on {len(predictions)} examples...")
    metrics = {}

    # 1. Embedding similarity
    print("  - Computing embedding similarity...")
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    pred_embeddings = embedding_model.encode(predictions)
    ref_embeddings = embedding_model.encode(references)

    similarities = []
    for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
        sim = np.dot(pred_emb, ref_emb) / (np.linalg.norm(pred_emb) * np.linalg.norm(ref_emb))
        similarities.append(sim)

    metrics["embedding_similarity"] = float(np.mean(similarities))

    # 2. Action-based metrics
    print("  - Computing action accuracy...")
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

    # 3. Length statistics
    pred_lengths = [len(p.split()) for p in predictions]
    ref_lengths = [len(r.split()) for r in references]

    metrics["avg_pred_length"] = float(np.mean(pred_lengths))
    metrics["avg_ref_length"] = float(np.mean(ref_lengths))
    metrics["length_ratio"] = float(np.mean(pred_lengths) / np.mean(ref_lengths)) if np.mean(ref_lengths) > 0 else 0

    return metrics


def train_persona_on_unified(
    persona_id: str,
    unified_model_path: Path,
    config: dict,
    output_dir: Path = None,
    persona_lora_rank: int = 8,
    use_val: bool = True
):
    """
    Train a persona-specific LoRA adapter on top of unified model.

    Args:
        persona_id: The persona to train for
        unified_model_path: Path to the unified LoRA model
        config: Training configuration
        output_dir: Directory to save the trained adapter
        persona_lora_rank: Rank for persona-specific LoRA (smaller than unified)
        use_val: Whether to use validation set for evaluation
    """
    print(f"\n{'='*60}")
    print(f"Training Persona-Specific Adapter on Unified Model")
    print(f"Persona: {persona_id}")
    print(f"{'='*60}\n")

    # Set device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load base model
    base_model_name = config["model_name"]
    print(f"\nLoading base model: {base_model_name}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )

    if device == "mps":
        base_model = base_model.to(device)

    print(f"[OK] Base model loaded")

    # Load unified LoRA adapter and MERGE it into base model
    print(f"\nLoading unified LoRA from: {unified_model_path}")
    model_with_unified = PeftModel.from_pretrained(
        base_model,
        str(unified_model_path)
    )
    print(f"[OK] Unified LoRA loaded")

    # Merge unified adapter into base model
    print(f"\nMerging unified adapter into base model...")
    merged_model = model_with_unified.merge_and_unload()
    print(f"[OK] Unified adapter merged into base model")

    # Now add persona-specific LoRA adapter on top of merged model
    print(f"\nAdding persona-specific LoRA (rank={persona_lora_rank})...")
    persona_lora_config = LoraConfig(
        r=persona_lora_rank,  # Smaller rank for persona-specific adapter
        lora_alpha=persona_lora_rank * 2,  # Keep alpha = 2*r ratio
        target_modules=config["lora_config"]["target_modules"],
        lora_dropout=0.3,  # High dropout to prevent overfitting on small data
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Apply persona LoRA to merged model
    model = get_peft_model(merged_model, persona_lora_config)

    print(f"[OK] Persona adapter added")

    # Print trainable parameters
    model.print_trainable_parameters()

    # Load persona data
    print(f"\nLoading training data for persona {persona_id}...")
    data_path = Path(config["data_path"])
    splits_path = Path(config.get("splits_path", "data/splits/edgesplits.json"))

    # Load train + val for training (80% of data)
    train_dialogues = load_persona_data(data_path, splits_path, persona_id, "train")
    print(f"[OK] Loaded {len(train_dialogues)} train dialogues")

    val_dialogues = load_persona_data(data_path, splits_path, persona_id, "val")
    print(f"[OK] Loaded {len(val_dialogues)} val dialogues (adding to training set)")

    train_dialogues.extend(val_dialogues)
    print(f"[OK] Combined training set: {len(train_dialogues)} dialogues")

    # Load test split for evaluation (20% of data)
    eval_dialogues = None
    if use_val:
        eval_dialogues = load_persona_data(data_path, splits_path, persona_id, "test")
        print(f"[OK] Loaded {len(eval_dialogues)} test dialogues for evaluation")

    # Create training examples
    system_prompt = "You are a helpful and personalized smart home assistant."

    train_texts = create_training_examples(train_dialogues, tokenizer,
                                          config["max_length"], system_prompt)
    train_dataset = Dataset.from_dict({"text": train_texts})

    if use_val and eval_dialogues:
        val_texts = create_training_examples(eval_dialogues, tokenizer,
                                            config["max_length"], system_prompt)
        val_dataset = Dataset.from_dict({"text": val_texts})

    # Tokenize datasets
    print(f"\nTokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, config["max_length"]),
        batched=True,
        remove_columns=["text"]
    )

    if use_val:
        val_dataset = val_dataset.map(
            lambda x: tokenize_function(x, tokenizer, config["max_length"]),
            batched=True,
            remove_columns=["text"]
        )

    print(f"[OK] Training examples: {len(train_dataset)}")
    if use_val:
        print(f"[OK] Validation examples: {len(val_dataset)}")

    # Setup output directory
    if output_dir is None:
        output_dir = Path("models/lora_hybrid") / persona_id
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training arguments - very gentle fine-tuning to preserve unified performance
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=2,  # Very few epochs to avoid degrading unified model
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,  # Very low learning rate to preserve unified knowledge
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir=str(output_dir / "logs"),
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if use_val else "no",
        load_best_model_at_end=use_val,
        metric_for_best_model="eval_loss" if use_val else None,
        greater_is_better=False,
        save_total_limit=2,
        fp16=False,  # Disabled to avoid FP16 gradient issues with frozen adapters
        optim="adamw_torch",
        report_to="none",
        seed=config["seed"]
    )

    # Data collator with padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if use_val else None,
        data_collator=data_collator,
    )

    # Train
    print(f"\nStarting training...")
    train_result = trainer.train()

    # Save the persona adapter
    print(f"\nSaving persona adapter to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Save config
    config_to_save = {
        "base_model": base_model_name,
        "unified_model_path": str(unified_model_path),
        "persona_id": persona_id,
        "persona_lora_rank": persona_lora_rank,
        "training_config": config
    }
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config_to_save, f, indent=2)

    print(f"\n[OK] Training complete!")
    print(f"[OK] Persona adapter saved to: {output_dir}")

    # Evaluation on test set with both adapters active
    print(f"\n{'='*60}")
    print(f"TEST SET EVALUATION (Unified + Persona)")
    print(f"{'='*60}")

    test_dialogues = load_persona_data(data_path, splits_path, persona_id, "test")
    print(f"[OK] Loaded {len(test_dialogues)} test dialogues")

    # Model already has persona adapter active on top of merged unified model

    test_metrics = evaluate_with_generation(
        model=model,
        tokenizer=tokenizer,
        dialogues=test_dialogues,
        system_prompt=system_prompt,
        device=device,
        max_new_tokens=config["generation"].get("max_new_tokens", 256),
        temperature=config["generation"].get("temperature", 0.7),
        top_p=config["generation"].get("top_p", 0.9)
    )

    print(f"\nTest Metrics (Hybrid Model):")
    for key, value in test_metrics.items():
        print(f"  {key:30s}: {value:.4f}")

    # Save test metrics
    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\n[OK] Test metrics saved to {output_dir / 'test_metrics.json'}")

    return output_dir, test_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train persona-specific LoRA on top of unified model"
    )
    parser.add_argument(
        "--persona_id",
        type=str,
        required=True,
        help="Persona ID to train for (e.g., 'persona_001')"
    )
    parser.add_argument(
        "--unified_model",
        type=str,
        default="models/lora_unified",
        help="Path to unified LoRA model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/lora_training.json",
        help="Path to training config file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: models/lora_hybrid/{persona_id})"
    )
    parser.add_argument(
        "--persona_rank",
        type=int,
        default=8,
        help="LoRA rank for persona-specific adapter (default: 8)"
    )
    parser.add_argument(
        "--no_val",
        action="store_true",
        help="Don't use validation set during training"
    )

    args = parser.parse_args()

    # Load config
    config = load_config(Path(args.config))

    # Train
    output_dir, test_metrics = train_persona_on_unified(
        persona_id=args.persona_id,
        unified_model_path=Path(args.unified_model),
        config=config,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        persona_lora_rank=args.persona_rank,
        use_val=not args.no_val
    )

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Adapter saved to: {output_dir}")
    print(f"\nTest Metrics:")
    print(f"  Embedding Similarity: {test_metrics['embedding_similarity']:.4f}")
    print(f"  Param F1:             {test_metrics['param_f1']:.4f}")
    print(f"  Device Precision:     {test_metrics['device_precision']:.4f}")


if __name__ == "__main__":
    main()
