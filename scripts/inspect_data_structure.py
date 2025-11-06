"""
Quick inspection to understand how characters, routines, and sessions relate.
"""

import json
from pathlib import Path
from collections import Counter


def inspect_structure():
    """Inspect the relationship between the three JSONL files."""

    data_dir = Path("data/raw")

    print("="*80)
    print("DATA STRUCTURE ANALYSIS")
    print("="*80)

    # Load characters
    characters = []
    with open(data_dir / "characters.jsonl", "r") as f:
        for line in f:
            characters.append(json.loads(line))

    print(f"\n1. CHARACTERS: {len(characters)} total")
    print(f"   First character snippet: {characters[0]['character'][:80]}...")

    # Load routines
    routines = []
    with open(data_dir / "routines.jsonl", "r") as f:
        for line in f:
            routines.append(json.loads(line))

    routine_counts = [len(r['routines']) for r in routines]
    print(f"\n2. ROUTINES: {len(routines)} entries (one per character)")
    print(f"   Routines per character: min={min(routine_counts)}, max={max(routine_counts)}, avg={sum(routine_counts)/len(routine_counts):.1f}")

    # Load sessions
    sessions = []
    with open(data_dir / "sessions.jsonl", "r") as f:
        for line in f:
            sessions.append(json.loads(line))

    session_counts = [len(s['sessions']) for s in sessions]
    total_sessions = sum(session_counts)

    print(f"\n3. SESSIONS: {len(sessions)} entries (one per character)")
    print(f"   Sessions per character: min={min(session_counts)}, max={max(session_counts)}, avg={sum(session_counts)/len(session_counts):.1f}")
    print(f"   Total dialogues across all characters: {total_sessions}")

    # Analyze message counts per session
    all_message_counts = []
    for persona_sessions in sessions:
        for session in persona_sessions['sessions']:
            all_message_counts.append(len(session['messages']))

    print(f"\n4. MESSAGES PER SESSION:")
    print(f"   Total sessions: {len(all_message_counts)}")
    print(f"   Messages per session: min={min(all_message_counts)}, max={max(all_message_counts)}, avg={sum(all_message_counts)/len(all_message_counts):.1f}")

    # Show data mapping
    print(f"\n{'='*80}")
    print("DATA MAPPING")
    print(f"{'='*80}")
    print(f"Each line represents ONE character/persona:")
    print(f"  - characters.jsonl line N = character description")
    print(f"  - routines.jsonl line N = that character's device preferences")
    print(f"  - sessions.jsonl line N = that character's dialogue history")
    print(f"\nThis is PERFECT for personalization research!")
    print(f"  - {len(characters)} unique personas")
    print(f"  - {total_sessions} total dialogue sessions")
    print(f"  - Each persona has their own interaction history and preferences")


if __name__ == "__main__":
    inspect_structure()
