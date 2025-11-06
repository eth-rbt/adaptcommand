"""
Phase A.4: Data Validation

This script validates the dataset splits to ensure:
- No overlap between train/val/test splits
- Each user has minimum examples in each split
- No single user dominates the dataset
- Temporal ordering is preserved
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np


def load_splits(splits_file: Path):
    """Load the splits JSON file."""
    with open(splits_file, "r") as f:
        return json.load(f)


def load_dialogues(dialogues_file: Path):
    """Load the cleaned dialogues."""
    dialogues = []
    with open(dialogues_file, "r") as f:
        for line in f:
            dialogues.append(json.loads(line))
    return dialogues


def check_no_overlap(splits: dict):
    """
    Verify that there's no overlap between train/val/test splits.

    Args:
        splits: Dictionary of splits per persona

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    for persona_id, persona_splits in splits.items():
        train = set(persona_splits["train"])
        val = set(persona_splits["val"])
        test = set(persona_splits["test"])
        online = set(persona_splits["online"])

        # Check main splits don't overlap
        train_val_overlap = train & val
        train_test_overlap = train & test
        val_test_overlap = val & test

        if train_val_overlap:
            errors.append(f"Persona {persona_id}: train/val overlap: {train_val_overlap}")
        if train_test_overlap:
            errors.append(f"Persona {persona_id}: train/test overlap: {train_test_overlap}")
        if val_test_overlap:
            errors.append(f"Persona {persona_id}: val/test overlap: {val_test_overlap}")

        # Online batch should be subset of train
        online_not_in_train = online - train
        if online_not_in_train:
            errors.append(f"Persona {persona_id}: online batch not in train: {online_not_in_train}")

    is_valid = len(errors) == 0
    return is_valid, errors


def check_minimum_coverage(splits: dict, min_examples: int = 1):
    """
    Verify that each user has minimum examples in each split.

    Args:
        splits: Dictionary of splits per persona
        min_examples: Minimum required examples per split

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    for persona_id, persona_splits in splits.items():
        if len(persona_splits["train"]) < min_examples:
            errors.append(f"Persona {persona_id}: train has only {len(persona_splits['train'])} examples")
        if len(persona_splits["val"]) < min_examples:
            errors.append(f"Persona {persona_id}: val has only {len(persona_splits['val'])} examples")
        if len(persona_splits["test"]) < min_examples:
            errors.append(f"Persona {persona_id}: test has only {len(persona_splits['test'])} examples")

    is_valid = len(errors) == 0
    return is_valid, errors


def check_balance(splits: dict, max_user_proportion: float = 0.1):
    """
    Verify that no single user dominates the dataset.

    Args:
        splits: Dictionary of splits per persona
        max_user_proportion: Maximum proportion of data a single user can have

    Returns:
        Tuple of (is_valid, warning_messages, stats)
    """
    warnings = []

    # Count total examples per user across all splits
    user_totals = {}
    for persona_id, persona_splits in splits.items():
        total = len(persona_splits["train"]) + len(persona_splits["val"]) + len(persona_splits["test"])
        user_totals[persona_id] = total

    # Compute global total
    global_total = sum(user_totals.values())

    # Find users with high proportions
    dominant_users = []
    for persona_id, count in user_totals.items():
        proportion = count / global_total if global_total > 0 else 0
        if proportion > max_user_proportion:
            dominant_users.append((persona_id, count, proportion))
            warnings.append(
                f"Persona {persona_id} has {count} examples ({proportion:.1%} of total), "
                f"exceeds threshold of {max_user_proportion:.1%}"
            )

    stats = {
        "total_examples": global_total,
        "num_personas": len(user_totals),
        "max_user_examples": max(user_totals.values()) if user_totals else 0,
        "max_user_proportion": max(user_totals.values()) / global_total if global_total > 0 else 0,
        "dominant_users": dominant_users
    }

    is_valid = len(dominant_users) == 0
    return is_valid, warnings, stats


def check_temporal_order(splits: dict, dialogues: list):
    """
    Verify that temporal ordering is preserved (train < val < test).

    This assumes dialogue indices are roughly time-ordered.

    Args:
        splits: Dictionary of splits per persona
        dialogues: List of all dialogues

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    for persona_id, persona_splits in splits.items():
        train = persona_splits["train"]
        val = persona_splits["val"]
        test = persona_splits["test"]

        # Check that indices are roughly ordered
        if train and val:
            if max(train) >= min(val):
                errors.append(f"Persona {persona_id}: max train index >= min val index")

        if val and test:
            if max(val) >= min(test):
                errors.append(f"Persona {persona_id}: max val index >= min test index")

        if train and test:
            if max(train) >= min(test):
                errors.append(f"Persona {persona_id}: max train index >= min test index")

    is_valid = len(errors) == 0
    return is_valid, errors


def compute_split_statistics(splits: dict):
    """
    Compute comprehensive statistics about the splits.

    Args:
        splits: Dictionary of splits per persona

    Returns:
        Dictionary of statistics
    """
    # Per-split totals
    total_train = sum(len(s["train"]) for s in splits.values())
    total_val = sum(len(s["val"]) for s in splits.values())
    total_test = sum(len(s["test"]) for s in splits.values())
    total_online = sum(len(s["online"]) for s in splits.values())
    total_all = total_train + total_val + total_test

    # Per-persona statistics
    train_per_persona = [len(s["train"]) for s in splits.values()]
    val_per_persona = [len(s["val"]) for s in splits.values()]
    test_per_persona = [len(s["test"]) for s in splits.values()]

    stats = {
        "num_personas": len(splits),
        "total_examples": total_all,
        "train": {
            "total": total_train,
            "proportion": total_train / total_all if total_all > 0 else 0,
            "avg_per_persona": np.mean(train_per_persona),
            "std_per_persona": np.std(train_per_persona)
        },
        "val": {
            "total": total_val,
            "proportion": total_val / total_all if total_all > 0 else 0,
            "avg_per_persona": np.mean(val_per_persona),
            "std_per_persona": np.std(val_per_persona)
        },
        "test": {
            "total": total_test,
            "proportion": total_test / total_all if total_all > 0 else 0,
            "avg_per_persona": np.mean(test_per_persona),
            "std_per_persona": np.std(test_per_persona)
        },
        "online": {
            "total": total_online,
            "avg_per_persona": np.mean([len(s["online"]) for s in splits.values()])
        }
    }

    return stats


def validate_splits(splits_file: Path, dialogues_file: Path,
                   min_examples: int = 1,
                   max_user_proportion: float = 0.1):
    """
    Run all validation checks on the splits.

    Args:
        splits_file: Path to splits JSON
        dialogues_file: Path to cleaned dialogues
        min_examples: Minimum examples per split
        max_user_proportion: Maximum proportion for single user

    Returns:
        Tuple of (is_valid, validation_report)
    """
    print(f"Loading splits from: {splits_file}")
    splits = load_splits(splits_file)

    print(f"Loading dialogues from: {dialogues_file}")
    dialogues = load_dialogues(dialogues_file)

    print(f"\n{'='*60}")
    print("VALIDATION CHECKS")
    print(f"{'='*60}\n")

    all_valid = True
    report = {}

    # Check 1: No overlap
    print("1. Checking for overlap between splits...")
    valid, errors = check_no_overlap(splits)
    report["no_overlap"] = {"valid": valid, "errors": errors}
    if valid:
        print("   ✓ No overlap detected")
    else:
        print(f"   ✗ Found {len(errors)} overlap issues")
        for error in errors[:5]:  # Show first 5
            print(f"     - {error}")
        all_valid = False

    # Check 2: Minimum coverage
    print(f"\n2. Checking minimum coverage ({min_examples} examples per split)...")
    valid, errors = check_minimum_coverage(splits, min_examples)
    report["minimum_coverage"] = {"valid": valid, "errors": errors}
    if valid:
        print("   ✓ All personas have sufficient coverage")
    else:
        print(f"   ✗ Found {len(errors)} coverage issues")
        for error in errors[:5]:
            print(f"     - {error}")
        all_valid = False

    # Check 3: Balance
    print(f"\n3. Checking balance (max user proportion: {max_user_proportion:.1%})...")
    valid, warnings, balance_stats = check_balance(splits, max_user_proportion)
    report["balance"] = {"valid": valid, "warnings": warnings, "stats": balance_stats}
    if valid:
        print("   ✓ Dataset is well-balanced")
    else:
        print(f"   ⚠ Found {len(warnings)} balance warnings")
        for warning in warnings[:5]:
            print(f"     - {warning}")

    # Check 4: Temporal order
    print("\n4. Checking temporal ordering...")
    valid, errors = check_temporal_order(splits, dialogues)
    report["temporal_order"] = {"valid": valid, "errors": errors}
    if valid:
        print("   ✓ Temporal ordering preserved")
    else:
        print(f"   ✗ Found {len(errors)} temporal ordering issues")
        for error in errors[:5]:
            print(f"     - {error}")
        all_valid = False

    # Compute statistics
    print("\n5. Computing split statistics...")
    stats = compute_split_statistics(splits)
    report["statistics"] = stats

    print(f"\n{'='*60}")
    print("SPLIT STATISTICS")
    print(f"{'='*60}")
    print(f"Total personas:      {stats['num_personas']}")
    print(f"Total examples:      {stats['total_examples']}")
    print(f"\nTrain:")
    print(f"  Total:             {stats['train']['total']} ({stats['train']['proportion']:.1%})")
    print(f"  Avg per persona:   {stats['train']['avg_per_persona']:.2f} ± {stats['train']['std_per_persona']:.2f}")
    print(f"\nVal:")
    print(f"  Total:             {stats['val']['total']} ({stats['val']['proportion']:.1%})")
    print(f"  Avg per persona:   {stats['val']['avg_per_persona']:.2f} ± {stats['val']['std_per_persona']:.2f}")
    print(f"\nTest:")
    print(f"  Total:             {stats['test']['total']} ({stats['test']['proportion']:.1%})")
    print(f"  Avg per persona:   {stats['test']['avg_per_persona']:.2f} ± {stats['test']['std_per_persona']:.2f}")
    print(f"\nOnline:")
    print(f"  Total:             {stats['online']['total']}")
    print(f"  Avg per persona:   {stats['online']['avg_per_persona']:.2f}")

    print(f"\n{'='*60}")
    if all_valid:
        print("✓ ALL VALIDATION CHECKS PASSED")
    else:
        print("✗ SOME VALIDATION CHECKS FAILED")
    print(f"{'='*60}\n")

    return all_valid, report


def main():
    parser = argparse.ArgumentParser(description="Validate EdgeWisePersona splits")
    parser.add_argument(
        "--splits",
        type=str,
        default="data/splits/edgesplits.json",
        help="Path to splits file"
    )
    parser.add_argument(
        "--dialogues",
        type=str,
        default="data/cleaned/dialogs_clean.jsonl",
        help="Path to cleaned dialogues"
    )
    parser.add_argument(
        "--min_examples",
        type=int,
        default=1,
        help="Minimum examples per split"
    )
    parser.add_argument(
        "--max_user_proportion",
        type=float,
        default=0.1,
        help="Maximum proportion of data for single user"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/splits/validation_report.json",
        help="Path to save validation report"
    )

    args = parser.parse_args()

    # Run validation
    is_valid, report = validate_splits(
        Path(args.splits),
        Path(args.dialogues),
        args.min_examples,
        args.max_user_proportion
    )

    # Save report
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"✓ Validation report saved to: {output_file}")

    # Exit with appropriate code
    exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
