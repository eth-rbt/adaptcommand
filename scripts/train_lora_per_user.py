"""
LoRA Training Script for Per-User Personalization

This script trains a LoRA adapter for a specific persona using their training data.
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
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
from typing import List, Dict
import numpy as np


def load_config(config_file: Path) -> dict:
    """Load training configuration."""
    with open(config_file, "r") as f:
        return json.load(f)


def load_persona_data(dialogues_file: Path, splits_file: Path, persona_id: str, split_name: str) -> List[Dict]:
    """
    Load all dialogues for a specific persona and split.

    Args:
        dialogues_file: Path to dialogues JSONL file
        splits_file: Path to splits JSON file
        persona_id: The persona ID to load data for
        split_name: "train", "val", or "test"

    Returns:
        List of dialogue dictionaries
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
    """
    Format dialogue messages into chat format.

    Args:
        messages: List of message dicts with "role" and "text"
        system_prompt: Optional system prompt to prepend

    Returns:
        List of formatted messages for chat template
    """
    formatted_messages = []

    # Add system prompt if provided
    if system_prompt:
        formatted_messages.append({
            "role": "system",
            "content": system_prompt
        })

    # Add dialogue messages
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

    Each dialogue is converted to a full chat conversation using the model's chat template.

    Args:
        dialogues: List of dialogue dictionaries
        tokenizer: Tokenizer with chat template
        max_length: Maximum sequence length
        system_prompt: Optional system prompt

    Returns:
        List of formatted training texts
    """
    training_texts = []

    for dialogue in dialogues:
        messages = dialogue["messages"]

        # Format messages
        formatted_messages = format_chat_messages(messages, system_prompt)

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
    # The DataCollatorForLanguageModeling will automatically create labels from input_ids
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None
    )
    return outputs


def compute_metrics(eval_pred):
    """Compute metrics during evaluation."""
    logits, labels = eval_pred

    # Calculate perplexity from loss
    # Note: Trainer computes loss automatically, we just format it here
    return {}


def train_lora_for_persona(
    persona_id: str,
    config: dict,
    output_dir: Path = None,
    use_val: bool = True
):
    """
    Train a LoRA adapter for a specific persona.

    Args:
        persona_id: The persona to train for
        config: Training configuration
        output_dir: Directory to save the trained adapter
        use_val: Whether to use validation set for evaluation
    """
    print(f"\n{'='*60}")
    print(f"Training LoRA for Persona: {persona_id}")
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
        dtype=torch.float16 if device == "cuda" else torch.float32,
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
    print(f"\nLoading training data for persona {persona_id}...")
    data_path = Path(config["data_path"])
    splits_path = Path(config.get("splits_path", "data/splits/edgesplits.json"))

    train_dialogues = load_persona_data(data_path, splits_path, persona_id, "train")
    print(f"[OK] Loaded {len(train_dialogues)} training dialogues")

    if use_val:
        val_dialogues = load_persona_data(data_path, splits_path, persona_id, "val")
        print(f"[OK] Loaded {len(val_dialogues)} validation dialogues")

    # Create training examples
    system_prompt = "You are a helpful and personalized smart home assistant."

    train_texts = create_training_examples(train_dialogues, tokenizer,
                                          config["max_length"], system_prompt)
    train_dataset = Dataset.from_dict({"text": train_texts})

    if use_val:
        val_texts = create_training_examples(val_dialogues, tokenizer,
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
        output_dir = Path(config["output_dir"]) / persona_id
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
    print(f"\nStarting training...")
    train_result = trainer.train()

    # Save final model
    print(f"\nSaving LoRA adapter to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Save config
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n[OK] Training complete!")
    print(f"[OK] LoRA adapter saved to: {output_dir}")

    # Evaluation on validation set
    if use_val:
        print(f"\nRunning final evaluation on validation set...")
        eval_results = trainer.evaluate()

        print(f"\nValidation Results:")
        for key, value in eval_results.items():
            print(f"  {key}: {value:.4f}")

        trainer.log_metrics("eval", eval_results)
        trainer.save_metrics("eval", eval_results)

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapter for a specific persona")
    parser.add_argument(
        "--persona_id",
        type=str,
        required=True,
        help="Persona ID to train for (e.g., 'persona_001')"
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
        help="Output directory (default: models/lora_adapters/{persona_id})"
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
    output_dir = train_lora_for_persona(
        persona_id=args.persona_id,
        config=config,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        use_val=not args.no_val
    )

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Adapter saved to: {output_dir}")


if __name__ == "__main__":
    main()
