"""
Unified LoRA Training Script

This script trains a single LoRA adapter on ALL training data from all 200 personas.
Unlike train_lora_per_user.py which trains separate adapters per persona,
this trains one unified model on the combined dataset.
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
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from action_metrics import ActionExtractor, ActionMetrics
import sys


def load_config(config_file: Path) -> dict:
    """Load training configuration."""
    with open(config_file, "r") as f:
        return json.load(f)


def load_all_split_data(dialogues_file: Path, splits_file: Path, split_name: str) -> List[Dict]:
    """
    Load all dialogues for a specific split across ALL personas.

    Args:
        dialogues_file: Path to dialogues JSONL file
        splits_file: Path to splits JSON file
        split_name: "train", "val", or "test"

    Returns:
        List of dialogue dictionaries from all personas for the specified split
    """
    # Load all dialogues
    dialogues = []
    with open(dialogues_file, "r") as f:
        for line in f:
            dialogues.append(json.loads(line))

    # Load splits
    with open(splits_file, "r") as f:
        splits = json.load(f)

    # Collect indices for all personas' split
    all_indices = []
    for persona_id, persona_splits in splits.items():
        if split_name in persona_splits:
            all_indices.extend(persona_splits[split_name])

    # Get dialogues for these indices
    split_dialogues = [dialogues[idx] for idx in all_indices]

    return split_dialogues


def format_chat_messages(messages: List[Dict], system_prompt: str = None,
                         include_persona: bool = True, character: str = None) -> List[Dict]:
    """
    Format dialogue messages into chat format.

    Args:
        messages: List of message dicts with "role" and "text"
        system_prompt: Optional system prompt to prepend
        include_persona: Whether to include persona description in system prompt
        character: Character description to include

    Returns:
        List of formatted messages for chat template
    """
    formatted_messages = []

    # Add system prompt if provided
    if system_prompt:
        final_system = system_prompt

        # Optionally append persona description
        if include_persona and character:
            final_system = f"{system_prompt}\n\nUser Profile: {character}"

        formatted_messages.append({
            "role": "system",
            "content": final_system
        })

    # Add dialogue messages
    for msg in messages:
        formatted_messages.append({
            "role": msg["role"],
            "content": msg["text"]
        })

    return formatted_messages


def create_training_examples(dialogues: List[Dict], tokenizer, max_length: int = 512,
                            system_prompt: str = None, include_persona: bool = True) -> List[str]:
    """
    Create training examples from dialogues.

    Creates one training example for EACH assistant turn in each dialogue.
    This ensures the model learns from all assistant responses.

    Args:
        dialogues: List of dialogue dictionaries
        tokenizer: Tokenizer with chat template
        max_length: Maximum sequence length
        system_prompt: Optional system prompt
        include_persona: Whether to include persona description

    Returns:
        List of formatted training texts (one per assistant turn)
    """
    training_texts = []

    for dialogue in dialogues:
        messages = dialogue["messages"]
        character = dialogue.get("character", None)

        # Create one training example for each assistant turn
        for i, msg in enumerate(messages):
            if msg["role"] == "assistant":
                # Get all messages up to and including this assistant turn
                context_messages = messages[:i+1]

                # Format messages
                formatted_messages = format_chat_messages(
                    context_messages,
                    system_prompt,
                    include_persona,
                    character
                )

                # Apply chat template
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
    include_persona: bool = True,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    batch_size: int = 1
) -> Dict[str, float]:
    """
    Evaluate model by generating responses and computing metrics.

    Args:
        model: The model to evaluate
        tokenizer: Tokenizer
        dialogues: List of dialogue dictionaries
        system_prompt: System prompt to use
        device: Device to run on
        include_persona: Whether to include persona in prompts
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        batch_size: Batch size for generation

    Returns:
        Dictionary of metrics
    """
    model.eval()

    predictions = []
    references = []

    print(f"\nGenerating responses for {len(dialogues)} dialogues...")

    for dialogue in tqdm(dialogues, desc="Evaluating"):
        messages = dialogue["messages"]
        character = dialogue.get("character", None)

        # Find last assistant message to predict
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "assistant":
                # Get context (everything before this assistant turn)
                context_messages = messages[:i]

                # Format for generation
                formatted_messages = format_chat_messages(
                    context_messages,
                    system_prompt,
                    include_persona,
                    character
                )

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

                # Decode
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
    pred_embeddings = embedding_model.encode(predictions, show_progress_bar=True)
    ref_embeddings = embedding_model.encode(references, show_progress_bar=True)

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


def train_unified_lora(
    config: dict,
    output_dir: Path,
    include_persona: bool = True,
    use_val: bool = True
):
    """
    Train a unified LoRA adapter on all training data.

    Args:
        config: Training configuration
        output_dir: Directory to save the trained adapter
        include_persona: Whether to include persona descriptions in prompts
        use_val: Whether to use validation set for evaluation
    """
    print(f"\n{'='*60}")
    print(f"Training Unified LoRA on All Training Data")
    print(f"{'='*60}\n")

    # Set device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load base model and tokenizer
    model_name = config["model_name"]
    print(f"\nLoading base model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )

    if device == "mps":
        base_model = base_model.to(device)

    print(f"[OK] Base model loaded")

    # Configure LoRA
    print(f"\nConfiguring LoRA...")
    lora_config = LoraConfig(
        r=config["lora_config"]["r"],
        lora_alpha=config["lora_config"]["lora_alpha"],
        target_modules=config["lora_config"]["target_modules"],
        lora_dropout=config["lora_config"]["lora_dropout"],
        bias=config["lora_config"]["bias"],
        task_type=TaskType.CAUSAL_LM
    )

    # Apply LoRA to model
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # Load data
    print(f"\nLoading training data from all personas...")
    data_path = Path(config["data_path"])
    splits_path = Path(config.get("splits_path", "data/splits/edgesplits.json"))

    train_dialogues = load_all_split_data(data_path, splits_path, "train")
    print(f"[OK] Loaded {len(train_dialogues)} training dialogues from all personas")

    if use_val:
        val_dialogues = load_all_split_data(data_path, splits_path, "val")
        print(f"[OK] Loaded {len(val_dialogues)} validation dialogues from all personas")

    # Create training examples
    system_prompt = "You are a helpful and personalized smart home assistant."

    print(f"\nCreating training examples...")
    train_texts = create_training_examples(
        train_dialogues,
        tokenizer,
        config["max_length"],
        system_prompt,
        include_persona
    )
    train_dataset = Dataset.from_dict({"text": train_texts})
    print(f"[OK] Created {len(train_texts)} training examples")

    if use_val:
        val_texts = create_training_examples(
            val_dialogues,
            tokenizer,
            config["max_length"],
            system_prompt,
            include_persona
        )
        val_dataset = Dataset.from_dict({"text": val_texts})
        print(f"[OK] Created {len(val_texts)} validation examples")

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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training arguments
    training_args_config = config["training_args"]
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_args_config["num_train_epochs"],
        per_device_train_batch_size=training_args_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_args_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_args_config["gradient_accumulation_steps"],
        learning_rate=training_args_config["learning_rate"],
        weight_decay=training_args_config["weight_decay"],
        warmup_ratio=training_args_config["warmup_ratio"],
        logging_dir=str(output_dir / "logs"),
        logging_steps=training_args_config["logging_steps"],
        save_strategy=training_args_config["save_strategy"],
        eval_strategy=training_args_config["evaluation_strategy"] if use_val else "no",
        load_best_model_at_end=training_args_config["load_best_model_at_end"] if use_val else False,
        metric_for_best_model=training_args_config["metric_for_best_model"] if use_val else None,
        greater_is_better=training_args_config["greater_is_better"] if use_val else None,
        save_total_limit=training_args_config["save_total_limit"],
        fp16=training_args_config["fp16"] and device == "cuda",
        optim=training_args_config["optim"],
        report_to=training_args_config["report_to"],
        seed=config["seed"]
    )

    # Data collator with padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
        pad_to_multiple_of=8  # Pad to multiple of 8 for efficiency
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
    print(f"\n{'='*60}")
    print(f"Starting training...")
    print(f"{'='*60}\n")
    train_result = trainer.train()

    # Save final model
    print(f"\nSaving LoRA adapter to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Save config with additional info
    config_to_save = config.copy()
    config_to_save["training_info"] = {
        "num_train_dialogues": len(train_dialogues),
        "num_train_examples": len(train_dataset),
        "num_val_dialogues": len(val_dialogues) if use_val else 0,
        "num_val_examples": len(val_dataset) if use_val else 0,
        "include_persona": include_persona
    }

    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config_to_save, f, indent=2)

    print(f"\n[OK] Training complete!")
    print(f"[OK] LoRA adapter saved to: {output_dir}")

    # Evaluation with generation-based metrics
    if use_val:
        print(f"\n{'='*60}")
        print(f"VALIDATION SET EVALUATION")
        print(f"{'='*60}")

        val_metrics = evaluate_with_generation(
            model=model,
            tokenizer=tokenizer,
            dialogues=val_dialogues,
            system_prompt=system_prompt,
            device=device,
            include_persona=include_persona,
            max_new_tokens=config["generation"].get("max_new_tokens", 256),
            temperature=config["generation"].get("temperature", 0.7),
            top_p=config["generation"].get("top_p", 0.9)
        )

        print(f"\nValidation Metrics:")
        for key, value in val_metrics.items():
            print(f"  {key:30s}: {value:.4f}")

        # Save validation metrics
        with open(output_dir / "val_metrics.json", "w") as f:
            json.dump(val_metrics, f, indent=2)
        print(f"\n[OK] Validation metrics saved to {output_dir / 'val_metrics.json'}")

    # Evaluate on test set
    print(f"\n{'='*60}")
    print(f"TEST SET EVALUATION")
    print(f"{'='*60}")

    test_dialogues = load_all_split_data(data_path, splits_path, "test")
    print(f"[OK] Loaded {len(test_dialogues)} test dialogues from all personas")

    test_metrics = evaluate_with_generation(
        model=model,
        tokenizer=tokenizer,
        dialogues=test_dialogues,
        system_prompt=system_prompt,
        device=device,
        include_persona=include_persona,
        max_new_tokens=config["generation"].get("max_new_tokens", 256),
        temperature=config["generation"].get("temperature", 0.7),
        top_p=config["generation"].get("top_p", 0.9)
    )

    print(f"\nTest Metrics:")
    for key, value in test_metrics.items():
        print(f"  {key:30s}: {value:.4f}")

    # Save test metrics
    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\n[OK] Test metrics saved to {output_dir / 'test_metrics.json'}")

    return output_dir, test_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train unified LoRA adapter on all training data"
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
        default="models/lora_unified",
        help="Output directory for trained adapter"
    )
    parser.add_argument(
        "--no_persona",
        action="store_true",
        help="Don't include persona descriptions in prompts"
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
    output_dir, test_metrics = train_unified_lora(
        config=config,
        output_dir=Path(args.output_dir),
        include_persona=not args.no_persona,
        use_val=not args.no_val
    )

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Adapter saved to: {output_dir}")
    print(f"\nFinal Test Metrics:")
    print(f"  Embedding Similarity: {test_metrics['embedding_similarity']:.4f}")
    print(f"  Device Precision:     {test_metrics['device_precision']:.4f}")
    print(f"  Device Recall:        {test_metrics['device_recall']:.4f}")
    print(f"  Param Precision:      {test_metrics['param_precision']:.4f}")
    print(f"  Param Recall:         {test_metrics['param_recall']:.4f}")
    print(f"  Numerical Precision:  {test_metrics['numerical_precision']:.4f}")


if __name__ == "__main__":
    main()
