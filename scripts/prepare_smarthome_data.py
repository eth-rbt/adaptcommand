"""
Prepare Smart Home Dataset for Personalization Study

This script:
1. Merges characters.jsonl, routines.jsonl, and sessions.jsonl
2. Flattens sessions into individual dialogues
3. Creates a clean dataset ready for splitting and benchmarking
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm


def load_jsonl(file_path: Path):
    """Load a JSONL file into a list."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def merge_and_flatten(characters_file: Path, routines_file: Path, sessions_file: Path):
    """
    Merge the three files and flatten sessions into individual dialogues.

    Returns:
        List of dialogue entries with persona metadata
    """
    print("Loading data files...")
    characters = load_jsonl(characters_file)
    routines = load_jsonl(routines_file)
    sessions = load_jsonl(sessions_file)

    print(f"Loaded {len(characters)} characters, {len(routines)} routine sets, {len(sessions)} session sets")

    # Verify they're all the same length
    assert len(characters) == len(routines) == len(sessions), "File lengths don't match!"

    # Flatten all sessions into individual dialogues
    all_dialogues = []

    for persona_idx in tqdm(range(len(characters)), desc="Processing personas"):
        persona_id = f"persona_{persona_idx:03d}"
        character_desc = characters[persona_idx]["character"]
        persona_routines = routines[persona_idx]["routines"]
        persona_sessions = sessions[persona_idx]["sessions"]

        # Process each session for this persona
        for session in persona_sessions:
            dialogue_entry = {
                "persona_id": persona_id,
                "session_id": session["session_id"],
                "character": character_desc,
                "routines": persona_routines,
                "meta": session["meta"],
                "messages": session["messages"],
                "applied_routines": session.get("applied_routines", [])
            }
            all_dialogues.append(dialogue_entry)

    print(f"\nCreated {len(all_dialogues)} individual dialogues")

    # Compute statistics
    personas = set(d["persona_id"] for d in all_dialogues)
    messages_per_dialogue = [len(d["messages"]) for d in all_dialogues]

    stats = {
        "total_dialogues": len(all_dialogues),
        "total_personas": len(personas),
        "dialogues_per_persona": len(all_dialogues) / len(personas),
        "avg_messages_per_dialogue": sum(messages_per_dialogue) / len(messages_per_dialogue),
        "min_messages_per_dialogue": min(messages_per_dialogue),
        "max_messages_per_dialogue": max(messages_per_dialogue)
    }

    return all_dialogues, stats


def clean_dialogue(dialogue: dict):
    """
    Clean a single dialogue entry.

    - Normalize whitespace in messages
    - Ensure consistent structure
    - Remove any empty messages
    """
    cleaned_messages = []

    for msg in dialogue["messages"]:
        text = msg.get("text", "").strip()
        role = msg.get("role", "").lower().strip()

        # Skip empty messages
        if not text or not role:
            continue

        cleaned_messages.append({
            "role": role,
            "text": text
        })

    # Create cleaned dialogue
    cleaned = {
        "persona_id": dialogue["persona_id"],
        "session_id": dialogue["session_id"],
        "character": dialogue["character"].strip(),
        "routines": dialogue["routines"],
        "meta": dialogue["meta"],
        "messages": cleaned_messages,
        "applied_routines": dialogue.get("applied_routines", [])
    }

    return cleaned


def clean_all_dialogues(dialogues: list):
    """Clean all dialogues."""
    cleaned = []
    discarded = 0

    for dialogue in tqdm(dialogues, desc="Cleaning dialogues"):
        cleaned_dialogue = clean_dialogue(dialogue)

        # Keep dialogues with at least 2 messages (one exchange)
        if len(cleaned_dialogue["messages"]) >= 2:
            cleaned.append(cleaned_dialogue)
        else:
            discarded += 1

    print(f"\nCleaned {len(cleaned)} dialogues, discarded {discarded}")
    return cleaned


def save_cleaned_data(dialogues: list, output_file: Path):
    """Save cleaned dialogues to JSONL."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for dialogue in dialogues:
            f.write(json.dumps(dialogue) + "\n")

    print(f"✓ Saved {len(dialogues)} dialogues to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Prepare smart home dataset for personalization study")
    parser.add_argument(
        "--characters",
        type=str,
        default="data/raw/characters.jsonl",
        help="Path to characters.jsonl"
    )
    parser.add_argument(
        "--routines",
        type=str,
        default="data/raw/routines.jsonl",
        help="Path to routines.jsonl"
    )
    parser.add_argument(
        "--sessions",
        type=str,
        default="data/raw/sessions.jsonl",
        help="Path to sessions.jsonl"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/cleaned/dialogs_clean.jsonl",
        help="Output file for cleaned dialogues"
    )

    args = parser.parse_args()

    # Merge and flatten
    dialogues, stats = merge_and_flatten(
        Path(args.characters),
        Path(args.routines),
        Path(args.sessions)
    )

    print("\n" + "="*60)
    print("MERGED DATA STATISTICS")
    print("="*60)
    print(f"Total dialogues:           {stats['total_dialogues']}")
    print(f"Total personas:            {stats['total_personas']}")
    print(f"Dialogues per persona:     {stats['dialogues_per_persona']:.1f}")
    print(f"Avg messages per dialogue: {stats['avg_messages_per_dialogue']:.1f}")
    print(f"Min messages per dialogue: {stats['min_messages_per_dialogue']}")
    print(f"Max messages per dialogue: {stats['max_messages_per_dialogue']}")

    # Clean
    cleaned_dialogues = clean_all_dialogues(dialogues)

    # Save
    save_cleaned_data(cleaned_dialogues, Path(args.output))

    # Save statistics
    stats_file = Path(args.output).parent / "cleaning_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Statistics saved to {stats_file}")

    print("\n✓ Data preparation complete!")
    print(f"\nNext step: Run create_splits.py to generate train/val/test splits")


if __name__ == "__main__":
    main()
