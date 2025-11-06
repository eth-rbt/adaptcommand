"""
Phase A.2: Data Cleaning & Normalization

This script cleans and normalizes the EdgeWisePersona dataset:
- Normalizes whitespace
- Standardizes message fields
- Filters degenerate exchanges
- Preserves interaction order per persona
"""

import json
import argparse
import re
from pathlib import Path
from collections import Counter
from tqdm import tqdm


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    # Replace multiple newlines with single newline
    text = re.sub(r'\n+', '\n', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def standardize_message(message: dict) -> dict:
    """
    Standardize a single message to have consistent fields.

    Expected output format:
    {
        "role": "user" or "assistant",
        "text": "message content"
    }

    Args:
        message: Input message dictionary

    Returns:
        Standardized message dictionary
    """
    standardized = {}

    # Normalize role field
    role = message.get("role") or message.get("from") or message.get("speaker")
    if role:
        role = role.lower().strip()
        # Map common variants to standard roles
        if role in ["user", "human", "person", "customer"]:
            standardized["role"] = "user"
        elif role in ["assistant", "agent", "bot", "system"]:
            standardized["role"] = "assistant"
        else:
            standardized["role"] = role
    else:
        # Default to user if no role specified
        standardized["role"] = "user"

    # Normalize text field
    text = message.get("text") or message.get("content") or message.get("message") or ""
    standardized["text"] = normalize_whitespace(str(text))

    return standardized


def is_degenerate_message(message: dict, min_length: int = 2) -> bool:
    """
    Check if a message is degenerate (too short or empty).

    Args:
        message: Message dictionary
        min_length: Minimum character length

    Returns:
        True if degenerate, False otherwise
    """
    text = message.get("text", "")
    return len(text) < min_length


def is_degenerate_dialogue(dialogue: list, min_messages: int = 2, min_total_chars: int = 10) -> bool:
    """
    Check if a dialogue is degenerate (too few messages or too short overall).

    Args:
        dialogue: List of message dictionaries
        min_messages: Minimum number of messages
        min_total_chars: Minimum total characters across all messages

    Returns:
        True if degenerate, False otherwise
    """
    if len(dialogue) < min_messages:
        return True

    total_chars = sum(len(msg.get("text", "")) for msg in dialogue)
    if total_chars < min_total_chars:
        return True

    return False


def clean_dialogue(dialogue_entry: dict, min_msg_length: int = 2,
                   min_messages: int = 2, min_total_chars: int = 10) -> dict:
    """
    Clean a single dialogue entry.

    Args:
        dialogue_entry: Dictionary containing dialogue data
        min_msg_length: Minimum message length in characters
        min_messages: Minimum number of messages per dialogue
        min_total_chars: Minimum total characters per dialogue

    Returns:
        Cleaned dialogue entry or None if it should be discarded
    """
    # Extract messages (could be under different keys)
    messages = dialogue_entry.get("messages") or dialogue_entry.get("dialogue") or []

    # Standardize all messages
    cleaned_messages = []
    for msg in messages:
        standardized = standardize_message(msg)
        # Skip degenerate individual messages
        if not is_degenerate_message(standardized, min_msg_length):
            cleaned_messages.append(standardized)

    # Check if dialogue is degenerate after cleaning
    if is_degenerate_dialogue(cleaned_messages, min_messages, min_total_chars):
        return None

    # Create cleaned entry
    cleaned_entry = {
        "persona_id": dialogue_entry.get("persona_id") or dialogue_entry.get("user_id"),
        "messages": cleaned_messages
    }

    # Preserve any additional metadata
    for key in ["session_id", "timestamp", "metadata"]:
        if key in dialogue_entry:
            cleaned_entry[key] = dialogue_entry[key]

    return cleaned_entry


def clean_dataset(input_file: Path, output_file: Path,
                 min_msg_length: int = 2,
                 min_messages: int = 2,
                 min_total_chars: int = 10):
    """
    Clean the entire dataset.

    Args:
        input_file: Path to raw data file
        output_file: Path to save cleaned data
        min_msg_length: Minimum message length
        min_messages: Minimum messages per dialogue
        min_total_chars: Minimum total characters per dialogue

    Returns:
        Dictionary of cleaning statistics
    """
    print(f"Loading data from: {input_file}")

    # Load raw data
    raw_dialogues = []
    with open(input_file, "r") as f:
        for line in f:
            raw_dialogues.append(json.loads(line))

    print(f"Loaded {len(raw_dialogues)} raw dialogues")

    # Clean dialogues
    cleaned_dialogues = []
    discarded_count = 0

    for entry in tqdm(raw_dialogues, desc="Cleaning dialogues"):
        cleaned = clean_dialogue(entry, min_msg_length, min_messages, min_total_chars)
        if cleaned is not None:
            cleaned_dialogues.append(cleaned)
        else:
            discarded_count += 1

    # Compute statistics
    stats = {
        "raw_count": len(raw_dialogues),
        "cleaned_count": len(cleaned_dialogues),
        "discarded_count": discarded_count,
        "retention_rate": len(cleaned_dialogues) / len(raw_dialogues) if raw_dialogues else 0
    }

    # Per-persona statistics
    persona_counts = Counter(d["persona_id"] for d in cleaned_dialogues)
    stats["num_personas"] = len(persona_counts)
    stats["avg_dialogues_per_persona"] = sum(persona_counts.values()) / len(persona_counts) if persona_counts else 0

    # Message statistics
    total_messages = sum(len(d["messages"]) for d in cleaned_dialogues)
    stats["total_messages"] = total_messages
    stats["avg_messages_per_dialogue"] = total_messages / len(cleaned_dialogues) if cleaned_dialogues else 0

    # Save cleaned data
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for entry in cleaned_dialogues:
            f.write(json.dumps(entry) + "\n")

    print(f"\n{'='*60}")
    print("CLEANING STATISTICS")
    print(f"{'='*60}")
    print(f"Raw dialogues:       {stats['raw_count']}")
    print(f"Cleaned dialogues:   {stats['cleaned_count']}")
    print(f"Discarded dialogues: {stats['discarded_count']}")
    print(f"Retention rate:      {stats['retention_rate']:.2%}")
    print(f"\nUnique personas:     {stats['num_personas']}")
    print(f"Avg dialogues/persona: {stats['avg_dialogues_per_persona']:.2f}")
    print(f"Total messages:      {stats['total_messages']}")
    print(f"Avg messages/dialogue: {stats['avg_messages_per_dialogue']:.2f}")
    print(f"\n✓ Cleaned data saved to: {output_file}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Clean EdgeWisePersona dataset")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/edgewise_raw.jsonl",
        help="Input raw data file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/cleaned/dialogs_clean.jsonl",
        help="Output cleaned data file"
    )
    parser.add_argument(
        "--min_msg_length",
        type=int,
        default=2,
        help="Minimum message length in characters"
    )
    parser.add_argument(
        "--min_messages",
        type=int,
        default=2,
        help="Minimum number of messages per dialogue"
    )
    parser.add_argument(
        "--min_total_chars",
        type=int,
        default=10,
        help="Minimum total characters per dialogue"
    )

    args = parser.parse_args()

    # Clean dataset
    stats = clean_dataset(
        Path(args.input),
        Path(args.output),
        args.min_msg_length,
        args.min_messages,
        args.min_total_chars
    )

    # Save stats
    stats_file = Path(args.output).parent / "cleaning_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Statistics saved to: {stats_file}")


if __name__ == "__main__":
    main()
