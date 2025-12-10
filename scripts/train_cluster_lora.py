"""
Train cluster-level LoRA adapters.

After running cluster_personas.py, this trains one LoRA adapter per cluster.
Each cluster has 600-900 training examples (vs 30 for per-persona).
"""

import json
import numpy as np
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import argparse
from tqdm import tqdm


def load_cluster_data(cluster_id, cluster_map_path='data/splits/cluster_map.json'):
    """Load training data for a specific cluster"""

    # Load cluster assignments
    with open(cluster_map_path) as f:
        cluster_data = json.load(f)

    cluster_map = cluster_data['cluster_map']

    # Get personas in this cluster
    cluster_personas = [
        persona_id for persona_id, cid in cluster_map.items()
        if cid == cluster_id
    ]

    print(f"Cluster {cluster_id} contains {len(cluster_personas)} personas")

    # Load all dialogues
    with open('data/cleaned/dialogs_clean.jsonl') as f:
        all_dialogues = [json.loads(line) for line in f]

    # Load splits
    with open('data/splits/edgesplits.json') as f:
        splits = json.load(f)

    # Collect training and validation data for this cluster
    train_dialogues = []
    val_dialogues = []

    for idx, dialogue in enumerate(all_dialogues):
        persona_id = dialogue['persona_id']
        if persona_id not in cluster_personas:
            continue

        # Splits contain INDICES, not session_ids
        if idx in splits[persona_id]['train']:
            train_dialogues.append(dialogue)
        elif idx in splits[persona_id]['val']:
            val_dialogues.append(dialogue)

    print(f"  Training dialogues: {len(train_dialogues)}")
    print(f"  Validation dialogues: {len(val_dialogues)}")

    return train_dialogues, val_dialogues, cluster_personas


def format_dialogue_for_training(dialogue, tokenizer):
    """Convert dialogue to training format"""

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

    # Format conversation - convert 'text' to 'content' for chat template
    messages = [{"role": "system", "content": system_message}]

    for msg in dialogue['messages']:
        messages.append({
            "role": msg['role'],
            "content": msg['text']  # Convert 'text' to 'content'
        })

    # Apply chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False)

    return text


def prepare_dataset(dialogues, tokenizer, max_length=1024):
    """Prepare dataset from dialogues"""

    texts = []
    for dialogue in tqdm(dialogues, desc="Formatting dialogues"):
        text = format_dialogue_for_training(dialogue, tokenizer)
        texts.append(text)

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    dataset = Dataset.from_dict({'text': texts})
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text'],
        desc="Tokenizing"
    )

    return tokenized


def train_cluster_lora(
    cluster_id,
    output_dir=None,
    base_model='Qwen/Qwen2.5-0.5B-Instruct',
    rank=8,
    num_epochs=5,
    batch_size=4,
    learning_rate=5e-4,
):
    """Train LoRA adapter for a specific cluster"""

    if output_dir is None:
        output_dir = f'models/lora_clusters/cluster_{cluster_id:02d}'

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f"Training Cluster {cluster_id} LoRA Adapter")
    print(f"{'=' * 80}")

    # Load data
    train_dialogues, val_dialogues, cluster_personas = load_cluster_data(cluster_id)

    # Save cluster info
    with open(output_dir / 'cluster_info.json', 'w') as f:
        json.dump({
            'cluster_id': cluster_id,
            'personas': cluster_personas,
            'num_personas': len(cluster_personas),
            'train_dialogues': len(train_dialogues),
            'val_dialogues': len(val_dialogues),
        }, f, indent=2)

    # Load model and tokenizer
    print("\nLoading base model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model)

    # Apply LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset = prepare_dataset(train_dialogues, tokenizer)
    val_dataset = prepare_dataset(val_dialogues, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
        warmup_steps=100,
        fp16=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save
    print(f"\nSaving to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"✓ Cluster {cluster_id} training complete!")

    return model


def train_all_clusters(n_clusters=None):
    """Train LoRA adapters for all clusters"""

    # Load cluster info
    with open('data/splits/cluster_map.json') as f:
        cluster_data = json.load(f)

    if n_clusters is None:
        n_clusters = cluster_data['n_clusters']

    print(f"Training {n_clusters} cluster LoRA adapters")
    print(f"{'=' * 80}\n")

    for cluster_id in range(n_clusters):
        try:
            train_cluster_lora(cluster_id)
        except Exception as e:
            print(f"ERROR training cluster {cluster_id}: {e}")
            continue

    print(f"\n{'=' * 80}")
    print("All cluster training complete!")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train cluster-level LoRA adapters')
    parser.add_argument('--cluster_id', type=int, default=None,
                       help='Train specific cluster (default: train all)')
    parser.add_argument('--rank', type=int, default=8,
                       help='LoRA rank')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='Learning rate')

    args = parser.parse_args()

    if args.cluster_id is not None:
        # Train single cluster
        train_cluster_lora(
            cluster_id=args.cluster_id,
            rank=args.rank,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
    else:
        # Train all clusters
        train_all_clusters()
