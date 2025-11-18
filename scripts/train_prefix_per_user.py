"""
Per-Persona Prefix Training Script

This script trains a separate prefix adapter for each persona using their training data.
This is the prefix tuning equivalent of train_lora_per_user.py.

Each persona gets their own prefix trained on their 30 training examples.
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
from peft import PrefixTuningConfig, get_peft_model, TaskType
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

                # Format messages
                formatted_messages = format_chat_messages(
                    context_messages,
                    system_prompt
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
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> Dict[str, float]:
    """
    Evaluate model by generating responses and computing metrics.
    """
    model.eval()

    predictions = []
    references = []

    for dialogue in dialogues:
        messages = dialogue["messages"]

        # Find last assistant message to predict
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "assistant":
                # Get context (everything before this assistant turn)
                context_messages = messages[:i]

                # Format for generation
                formatted_messages = format_chat_messages(
                    context_messages,
                    system_prompt
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
    metrics = {}

    # 1. Embedding similarity
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    pred_embeddings = embedding_model.encode(predictions, show_progress_bar=False)
    ref_embeddings = embedding_model.encode(references, show_progress_bar=False)

    similarities = []
    for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
        sim = np.dot(pred_emb, ref_emb) / (np.linalg.norm(pred_emb) * np.linalg.norm(ref_emb))
        similarities.append(sim)

    metrics["embedding_similarity"] = float(np.mean(similarities))

    # 2. Action-based metrics
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


def train_persona_prefix(
    persona_id: str,
    config: dict,
    output_dir: Path
):
    """
    Train a prefix adapter for a single persona.

    Args:
        persona_id: Persona identifier (e.g., "persona_000")
        config: Training configuration
        output_dir: Base output directory
    """
    print(f"\n{'='*60}")
    print(f"Training Prefix for {persona_id}")
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
        device_map=None,  # Disable auto device mapping for prefix tuning compatibility
        attn_implementation="eager"  # Use eager attention to avoid cache issues
    )

    base_model = base_model.to(device)

    print(f"[OK] Base model loaded")

    # Configure Prefix Tuning
    print(f"\nConfiguring Prefix Tuning...")
    prefix_config = PrefixTuningConfig(
        num_virtual_tokens=config["prefix_config"]["num_virtual_tokens"],
        prefix_projection=config["prefix_config"]["prefix_projection"],
        task_type=TaskType.CAUSAL_LM
    )

    # Apply Prefix Tuning to model
    model = get_peft_model(base_model, prefix_config)
    model.print_trainable_parameters()

    # Load persona's training data
    print(f"\nLoading training data for {persona_id}...")
    data_path = Path(config["data_path"])
    splits_path = Path(config["splits_path"])

    train_dialogues = load_persona_data(data_path, splits_path, persona_id, "train")
    print(f"[OK] Loaded {len(train_dialogues)} training dialogues")

    # Create training examples
    system_prompt = "You are a helpful and personalized smart home assistant."

    print(f"\nCreating training examples...")
    train_texts = create_training_examples(
        train_dialogues,
        tokenizer,
        config["max_length"],
        system_prompt
    )
    train_dataset = Dataset.from_dict({"text": train_texts})
    print(f"[OK] Created {len(train_texts)} training examples")

    # Tokenize dataset
    print(f"\nTokenizing dataset...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, config["max_length"]),
        batched=True,
        remove_columns=["text"]
    )
    print(f"[OK] Training examples: {len(train_dataset)}")

    # Setup output directory
    persona_output_dir = output_dir / persona_id
    persona_output_dir.mkdir(parents=True, exist_ok=True)

    # Training arguments
    training_args_config = config["training_args"]
    training_args = TrainingArguments(
        output_dir=str(persona_output_dir),
        num_train_epochs=training_args_config["num_train_epochs"],
        per_device_train_batch_size=training_args_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_args_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_args_config["gradient_accumulation_steps"],
        learning_rate=training_args_config["learning_rate"],
        weight_decay=training_args_config["weight_decay"],
        warmup_ratio=training_args_config["warmup_ratio"],
        logging_dir=str(persona_output_dir / "logs"),
        logging_steps=training_args_config["logging_steps"],
        save_strategy=training_args_config["save_strategy"],
        eval_strategy=training_args_config["evaluation_strategy"],
        save_total_limit=training_args_config["save_total_limit"],
        fp16=training_args_config["fp16"] and device == "cuda",
        optim=training_args_config["optim"],
        report_to=training_args_config["report_to"],
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
        data_collator=data_collator,
    )

    # Train
    print(f"\n{'='*60}")
    print(f"Starting training...")
    print(f"{'='*60}\n")
    trainer.train()

    # Save model
    print(f"\nSaving prefix adapter to {persona_output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(persona_output_dir)

    # Evaluate on test set
    print(f"\n{'='*60}")
    print(f"TEST SET EVALUATION")
    print(f"{'='*60}")

    test_dialogues = load_persona_data(data_path, splits_path, persona_id, "test")
    print(f"[OK] Loaded {len(test_dialogues)} test dialogues")

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

    print(f"\nTest Metrics:")
    for key, value in test_metrics.items():
        print(f"  {key:30s}: {value:.4f}")

    # Save test metrics
    with open(persona_output_dir / "eval.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\n[OK] Test metrics saved to {persona_output_dir / 'eval.json'}")

    return test_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train prefix adapter for a single persona"
    )
    parser.add_argument(
        "--persona_id",
        type=str,
        required=True,
        help="Persona identifier (e.g., persona_000)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/prefix_per_user.json",
        help="Path to training config file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/prefix_per_user",
        help="Base output directory for trained adapters"
    )

    args = parser.parse_args()

    # Load config
    config = load_config(Path(args.config))

    # Train
    test_metrics = train_persona_prefix(
        persona_id=args.persona_id,
        config=config,
        output_dir=Path(args.output_dir)
    )

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE FOR {args.persona_id}")
    print(f"{'='*60}")
    print(f"\nFinal Test Metrics:")
    print(f"  Embedding Similarity: {test_metrics['embedding_similarity']:.4f}")
    print(f"  Device Precision:     {test_metrics['device_precision']:.4f}")
    print(f"  Device Recall:        {test_metrics['device_recall']:.4f}")
    print(f"  Param F1:             {test_metrics['param_f1']:.4f}")
    print(f"  Numerical Precision:  {test_metrics['numerical_precision']:.4f}")


if __name__ == "__main__":
    main()
