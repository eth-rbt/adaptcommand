"""
Phase A.1: Dataset Acquisition & Loading

This script loads the EdgeWisePersona dataset and performs initial exploration.
It expects the dataset to be in HuggingFace format or a local directory.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


def load_edgewise_dataset(data_source: str, cache_dir: str = None):
    """
    Load EdgeWisePersona dataset from HuggingFace or local path.

    Args:
        data_source: HuggingFace dataset name or local path
        cache_dir: Optional cache directory for HF datasets

    Returns:
        dataset: Loaded dataset object
    """
    print(f"Loading dataset from: {data_source}")

    try:
        # Try loading from HuggingFace
        dataset = load_dataset(data_source, cache_dir=cache_dir)
        print(f"✓ Loaded from HuggingFace: {data_source}")
    except Exception as e:
        # Try loading from local path
        try:
            data_path = Path(data_source)
            if data_path.is_dir():
                dataset = load_dataset("json", data_dir=str(data_path))
            elif data_path.suffix == ".jsonl":
                dataset = load_dataset("json", data_files=str(data_path))
            else:
                raise ValueError(f"Unsupported file format: {data_path.suffix}")
            print(f"✓ Loaded from local path: {data_source}")
        except Exception as e2:
            raise ValueError(f"Failed to load dataset.\nHF error: {e}\nLocal error: {e2}")

    return dataset


def explore_dataset(dataset):
    """
    Explore the dataset structure and compute basic statistics.

    Args:
        dataset: HuggingFace dataset object

    Returns:
        stats: Dictionary of dataset statistics
    """
    print("\n" + "="*60)
    print("DATASET EXPLORATION")
    print("="*60)

    # Get the first split (might be 'train' or the only split)
    split_name = list(dataset.keys())[0]
    data = dataset[split_name]

    print(f"\nDataset split: {split_name}")
    print(f"Total examples: {len(data)}")
    print(f"\nColumn names: {data.column_names}")
    print(f"\nFirst example:")
    print(json.dumps(data[0], indent=2, default=str))

    # Compute statistics
    stats = {
        "total_examples": len(data),
        "columns": data.column_names,
        "split_name": split_name
    }

    # Persona-level statistics
    if "persona_id" in data.column_names or "user_id" in data.column_names:
        persona_col = "persona_id" if "persona_id" in data.column_names else "user_id"
        persona_ids = [ex[persona_col] for ex in data]
        persona_counts = Counter(persona_ids)

        stats["num_personas"] = len(persona_counts)
        stats["avg_dialogues_per_persona"] = sum(persona_counts.values()) / len(persona_counts)
        stats["min_dialogues_per_persona"] = min(persona_counts.values())
        stats["max_dialogues_per_persona"] = max(persona_counts.values())

        print(f"\n--- Persona Statistics ---")
        print(f"Number of unique personas: {stats['num_personas']}")
        print(f"Avg dialogues per persona: {stats['avg_dialogues_per_persona']:.2f}")
        print(f"Min dialogues per persona: {stats['min_dialogues_per_persona']}")
        print(f"Max dialogues per persona: {stats['max_dialogues_per_persona']}")

        # Show distribution
        print(f"\nDialogues per persona distribution:")
        for count in sorted(set(persona_counts.values())):
            num_personas = sum(1 for c in persona_counts.values() if c == count)
            print(f"  {count} dialogues: {num_personas} personas")

    # Message-level statistics (if dialogues are nested)
    if "messages" in data.column_names or "dialogue" in data.column_names:
        msg_col = "messages" if "messages" in data.column_names else "dialogue"
        total_messages = sum(len(ex[msg_col]) for ex in data)
        avg_messages = total_messages / len(data)

        # Compute message lengths
        message_lengths = []
        for ex in data:
            for msg in ex[msg_col]:
                if isinstance(msg, dict) and "text" in msg:
                    message_lengths.append(len(msg["text"]))
                elif isinstance(msg, dict) and "content" in msg:
                    message_lengths.append(len(msg["content"]))

        stats["total_messages"] = total_messages
        stats["avg_messages_per_dialogue"] = avg_messages
        stats["avg_message_length"] = sum(message_lengths) / len(message_lengths) if message_lengths else 0

        print(f"\n--- Message Statistics ---")
        print(f"Total messages: {stats['total_messages']}")
        print(f"Avg messages per dialogue: {stats['avg_messages_per_dialogue']:.2f}")
        print(f"Avg message length (chars): {stats['avg_message_length']:.2f}")

    return stats


def save_raw_data(dataset, output_dir: Path):
    """
    Save the dataset to data/raw/ directory for reference.

    Args:
        dataset: HuggingFace dataset object
        output_dir: Output directory path
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    split_name = list(dataset.keys())[0]
    data = dataset[split_name]

    output_file = output_dir / "edgewise_raw.jsonl"

    print(f"\nSaving raw data to: {output_file}")
    with open(output_file, "w") as f:
        for ex in tqdm(data, desc="Saving"):
            f.write(json.dumps(ex) + "\n")

    print(f"✓ Saved {len(data)} examples to {output_file}")

    # Save statistics
    stats_file = output_dir / "dataset_stats.json"
    stats = explore_dataset(dataset)
    with open(stats_file, "w") as f:
        json.dumps(stats, indent=2)
    print(f"✓ Saved statistics to {stats_file}")


def main():
    parser = argparse.ArgumentParser(description="Load and explore EdgeWisePersona dataset")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="HuggingFace dataset name or local path to dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw",
        help="Output directory for raw data (default: data/raw)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for HuggingFace datasets"
    )
    parser.add_argument(
        "--explore_only",
        action="store_true",
        help="Only explore the dataset, don't save it"
    )

    args = parser.parse_args()

    # Load dataset
    dataset = load_edgewise_dataset(args.source, args.cache_dir)

    # Explore
    stats = explore_dataset(dataset)

    # Save if requested
    if not args.explore_only:
        output_dir = Path(args.output_dir)
        save_raw_data(dataset, output_dir)

        # Save stats separately
        stats_file = output_dir / "dataset_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\n✓ All done! Raw data saved to {output_dir}")
    else:
        print("\n✓ Exploration complete (not saved)")


if __name__ == "__main__":
    main()
