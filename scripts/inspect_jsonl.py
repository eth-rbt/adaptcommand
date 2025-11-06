"""
Utility script to inspect JSONL files in the raw data folder.
Displays random entries from each file to understand the data structure.
"""

import json
import argparse
import random
from pathlib import Path


def inspect_jsonl_files(directory: Path, num_entries: int = 1, pretty: bool = True, randomize: bool = True):
    """
    Inspect JSONL files in a directory and display random or first N entries.

    Args:
        directory: Directory containing JSONL files
        num_entries: Number of entries to display per file
        pretty: Whether to pretty-print JSON
        randomize: Whether to show random entries (vs first N)
    """
    # Find all JSONL files
    jsonl_files = list(directory.glob("*.jsonl"))

    if not jsonl_files:
        print(f"No .jsonl files found in {directory}")
        return

    print(f"Found {len(jsonl_files)} JSONL file(s) in {directory}\n")
    print("="*80)

    for jsonl_file in sorted(jsonl_files):
        print(f"\nFile: {jsonl_file.name}")
        print("-"*80)

        try:
            # Load all lines
            with open(jsonl_file, "r") as f:
                lines = f.readlines()

            total_lines = len(lines)
            print(f"Total lines in file: {total_lines}")

            # Select which lines to show
            if randomize:
                # Pick random lines
                num_to_show = min(num_entries, total_lines)
                selected_indices = random.sample(range(total_lines), num_to_show)
                selected_indices.sort()  # Sort for easier reading
            else:
                # Pick first N lines
                selected_indices = list(range(min(num_entries, total_lines)))

            # Display selected entries
            for i, line_idx in enumerate(selected_indices):
                line = lines[line_idx]
                line_num = line_idx + 1  # 1-indexed for display

                try:
                    data = json.loads(line)

                    if randomize:
                        print(f"\nRandom entry (line {line_num}):")
                    else:
                        print(f"\nEntry {line_num}:")

                    if pretty:
                        print(json.dumps(data, indent=2, ensure_ascii=False))
                    else:
                        print(json.dumps(data, ensure_ascii=False))

                    # Show keys and basic info
                    if isinstance(data, dict):
                        print(f"\nKeys in this entry: {list(data.keys())}")

                except json.JSONDecodeError as e:
                    print(f"  ⚠ Line {line_num}: Invalid JSON - {e}")

        except Exception as e:
            print(f"  ✗ Error reading file: {e}")

        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect JSONL files and show random entries from each"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="data/raw",
        help="Directory containing JSONL files (default: data/raw)"
    )
    parser.add_argument(
        "--num",
        type=int,
        default=1,
        help="Number of entries to show per file (default: 1)"
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Display JSON in compact format (no pretty printing)"
    )
    parser.add_argument(
        "--first",
        action="store_true",
        help="Show first N entries instead of random entries"
    )

    args = parser.parse_args()

    directory = Path(args.dir)

    if not directory.exists():
        print(f"✗ Directory not found: {directory}")
        return

    inspect_jsonl_files(directory, args.num, pretty=not args.compact, randomize=not args.first)


if __name__ == "__main__":
    main()
