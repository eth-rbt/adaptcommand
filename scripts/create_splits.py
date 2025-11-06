"""
Phase A.3: Per-User Time-Aware Splits

This script creates time-aware train/val/test splits for each persona/user:
- 60% train, 20% val, 20% test (time-ordered)
- Small online batch (8-12 examples) from train for micro-updates
- Optional cold-start and cluster splits
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm


def create_time_aware_split(dialogue_indices: list,
                            train_ratio: float = 0.6,
                            val_ratio: float = 0.2,
                            test_ratio: float = 0.2,
                            online_batch_size: int = 10):
    """
    Create time-aware splits for a single user's dialogues.

    Args:
        dialogue_indices: List of dialogue indices (assumed to be time-ordered)
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        online_batch_size: Number of examples for online micro-updates

    Returns:
        Dictionary with train/val/test/online splits
    """
    n = len(dialogue_indices)

    # Calculate split points
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    # Create splits (maintaining temporal order)
    train_indices = dialogue_indices[:train_end]
    val_indices = dialogue_indices[train_end:val_end]
    test_indices = dialogue_indices[val_end:]

    # Reserve online batch from end of train split
    online_size = min(online_batch_size, len(train_indices))
    online_indices = train_indices[-online_size:] if online_size > 0 else []

    splits = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
        "online": online_indices
    }

    return splits


def generate_splits(input_file: Path,
                   output_file: Path,
                   train_ratio: float = 0.6,
                   val_ratio: float = 0.2,
                   test_ratio: float = 0.2,
                   online_batch_size: int = 10,
                   min_examples_per_split: int = 1):
    """
    Generate time-aware splits for all users.

    Args:
        input_file: Path to cleaned dialogues
        output_file: Path to save splits
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        online_batch_size: Number of examples for online updates
        min_examples_per_split: Minimum examples required per split

    Returns:
        Dictionary of splits and statistics
    """
    print(f"Loading cleaned data from: {input_file}")

    # Load cleaned dialogues
    dialogues = []
    with open(input_file, "r") as f:
        for line in f:
            dialogues.append(json.loads(line))

    print(f"Loaded {len(dialogues)} dialogues")

    # Group dialogues by persona_id (maintain order within each persona)
    persona_dialogues = defaultdict(list)
    for idx, dialogue in enumerate(dialogues):
        persona_id = dialogue["persona_id"]
        persona_dialogues[persona_id].append(idx)

    print(f"Found {len(persona_dialogues)} unique personas")

    # Create splits for each persona
    all_splits = {}
    valid_personas = []
    excluded_personas = []

    for persona_id, indices in tqdm(persona_dialogues.items(), desc="Creating splits"):
        splits = create_time_aware_split(
            indices,
            train_ratio,
            val_ratio,
            test_ratio,
            online_batch_size
        )

        # Check if persona has sufficient data
        min_required = min_examples_per_split * 3  # train, val, test
        if (len(splits["train"]) >= min_examples_per_split and
            len(splits["val"]) >= min_examples_per_split and
            len(splits["test"]) >= min_examples_per_split):
            all_splits[persona_id] = splits
            valid_personas.append(persona_id)
        else:
            excluded_personas.append(persona_id)

    # Compute statistics
    stats = {
        "total_personas": len(persona_dialogues),
        "valid_personas": len(valid_personas),
        "excluded_personas": len(excluded_personas),
        "total_dialogues": len(dialogues),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "online_batch_size": online_batch_size
    }

    # Split-level statistics
    total_train = sum(len(s["train"]) for s in all_splits.values())
    total_val = sum(len(s["val"]) for s in all_splits.values())
    total_test = sum(len(s["test"]) for s in all_splits.values())
    total_online = sum(len(s["online"]) for s in all_splits.values())

    stats["total_train_examples"] = total_train
    stats["total_val_examples"] = total_val
    stats["total_test_examples"] = total_test
    stats["total_online_examples"] = total_online

    # Per-persona statistics
    examples_per_persona = [len(indices) for indices in persona_dialogues.values()]
    stats["avg_examples_per_persona"] = float(np.mean(examples_per_persona))
    stats["min_examples_per_persona"] = int(np.min(examples_per_persona))
    stats["max_examples_per_persona"] = int(np.max(examples_per_persona))

    # Save splits
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(all_splits, f, indent=2)

    print(f"\n{'='*60}")
    print("SPLIT GENERATION STATISTICS")
    print(f"{'='*60}")
    print(f"Total personas:       {stats['total_personas']}")
    print(f"Valid personas:       {stats['valid_personas']}")
    print(f"Excluded personas:    {stats['excluded_personas']}")
    print(f"\nTotal dialogues:      {stats['total_dialogues']}")
    print(f"Train examples:       {stats['total_train_examples']}")
    print(f"Val examples:         {stats['total_val_examples']}")
    print(f"Test examples:        {stats['total_test_examples']}")
    print(f"Online examples:      {stats['total_online_examples']}")
    print(f"\nAvg examples/persona: {stats['avg_examples_per_persona']:.2f}")
    print(f"Min examples/persona: {stats['min_examples_per_persona']}")
    print(f"Max examples/persona: {stats['max_examples_per_persona']}")
    print(f"\n✓ Splits saved to: {output_file}")

    if excluded_personas:
        print(f"\n⚠ Warning: {len(excluded_personas)} personas excluded due to insufficient data")

    return all_splits, stats


def extract_cold_start_data(dialogues: list,
                           splits: dict,
                           output_file: Path,
                           num_turns: int = 5):
    """
    Extract cold-start test data (first few turns from users).

    Args:
        dialogues: List of all dialogues
        splits: Dictionary of splits per persona
        output_file: Path to save cold-start data
        num_turns: Number of initial turns to extract
    """
    print(f"\nExtracting cold-start data (first {num_turns} turns)...")

    cold_start_data = []

    for persona_id, persona_splits in splits.items():
        # Use first dialogue from test split
        if persona_splits["test"]:
            first_test_idx = persona_splits["test"][0]
            dialogue = dialogues[first_test_idx]

            # Extract first N turns
            messages = dialogue["messages"][:num_turns]

            if len(messages) >= 2:  # At least one exchange
                cold_start_entry = {
                    "persona_id": persona_id,
                    "dialogue_idx": first_test_idx,
                    "messages": messages
                }
                cold_start_data.append(cold_start_entry)

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for entry in cold_start_data:
            f.write(json.dumps(entry) + "\n")

    print(f"✓ Extracted {len(cold_start_data)} cold-start examples")
    print(f"✓ Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Create time-aware splits for EdgeWisePersona")
    parser.add_argument(
        "--input",
        type=str,
        default="data/cleaned/dialogs_clean.jsonl",
        help="Input cleaned dialogues file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/splits/edgesplits.json",
        help="Output splits file"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.6,
        help="Training set ratio"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation set ratio"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Test set ratio"
    )
    parser.add_argument(
        "--online_batch_size",
        type=int,
        default=10,
        help="Number of examples for online micro-updates"
    )
    parser.add_argument(
        "--min_examples",
        type=int,
        default=1,
        help="Minimum examples per split to include persona"
    )
    parser.add_argument(
        "--create_cold_start",
        action="store_true",
        help="Create cold-start test set"
    )
    parser.add_argument(
        "--cold_start_turns",
        type=int,
        default=5,
        help="Number of initial turns for cold-start"
    )

    args = parser.parse_args()

    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.01:
        raise ValueError("Train, val, and test ratios must sum to 1.0")

    # Generate splits
    splits, stats = generate_splits(
        Path(args.input),
        Path(args.output),
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.online_batch_size,
        args.min_examples
    )

    # Save statistics
    stats_file = Path(args.output).parent / "split_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Statistics saved to: {stats_file}")

    # Create cold-start data if requested
    if args.create_cold_start:
        dialogues = []
        with open(args.input, "r") as f:
            for line in f:
                dialogues.append(json.loads(line))

        cold_start_file = Path(args.output).parent / "cold_start_test.jsonl"
        extract_cold_start_data(dialogues, splits, cold_start_file, args.cold_start_turns)


if __name__ == "__main__":
    main()
