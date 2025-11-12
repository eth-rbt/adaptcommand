"""
Train All Personas with Hybrid Approach (Persona adapter on Unified base)

Trains persona-specific adapters for all 200 personas using the unified model as base.
"""

import json
import argparse
from pathlib import Path
import subprocess
import sys
from datetime import datetime
import numpy as np


def load_persona_ids(splits_file: Path):
    """Load all persona IDs from splits file."""
    with open(splits_file, "r") as f:
        splits = json.load(f)
    return list(splits.keys())


def train_single_persona(
    persona_id: str,
    unified_model_path: Path,
    config_path: Path,
    output_base_dir: Path,
    persona_rank: int,
    no_val: bool
):
    """Train a single persona using the hybrid approach."""
    output_dir = output_base_dir / persona_id

    cmd = [
        sys.executable,
        "scripts/train_persona_on_unified.py",
        "--persona_id", persona_id,
        "--unified_model", str(unified_model_path),
        "--config", str(config_path),
        "--output_dir", str(output_dir),
        "--persona_rank", str(persona_rank)
    ]

    if no_val:
        cmd.append("--no_val")

    print(f"\n{'='*80}")
    print(f"Training {persona_id}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    result = subprocess.run(cmd, capture_output=False)

    return result.returncode == 0


def aggregate_results(output_base_dir: Path, output_file: Path):
    """Aggregate test metrics from all trained personas."""
    all_results = []

    for persona_dir in sorted(output_base_dir.iterdir()):
        if persona_dir.is_dir():
            metrics_file = persona_dir / "test_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)
                    metrics["persona_id"] = persona_dir.name
                    all_results.append(metrics)

    if not all_results:
        print("[WARNING] No results found to aggregate")
        return

    # Compute aggregate statistics
    metric_keys = [k for k in all_results[0].keys() if k != "persona_id"]

    summary = {
        "num_personas": len(all_results),
        "per_persona_metrics": all_results
    }

    # Add mean, std, min, max for each metric
    for key in metric_keys:
        values = [r[key] for r in all_results]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
        summary[f"{key}_min"] = float(np.min(values))
        summary[f"{key}_max"] = float(np.max(values))

    # Save summary
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print("AGGREGATE RESULTS")
    print(f"{'='*80}")
    print(f"\nPersonas trained: {len(all_results)}")
    print(f"\nAverage Metrics:")
    for key in metric_keys:
        mean = summary[f"{key}_mean"]
        std = summary[f"{key}_std"]
        print(f"  {key:30s}: {mean:.4f} ± {std:.4f}")

    print(f"\n[OK] Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Train all personas with hybrid approach (persona adapter on unified base)"
    )
    parser.add_argument(
        "--unified_model",
        type=str,
        default="models/lora_unified",
        help="Path to unified LoRA model"
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
        default="models/lora_hybrid",
        help="Base output directory for all persona adapters"
    )
    parser.add_argument(
        "--splits_file",
        type=str,
        default="data/splits/edgesplits.json",
        help="Path to splits file"
    )
    parser.add_argument(
        "--persona_rank",
        type=int,
        default=8,
        help="LoRA rank for persona-specific adapters (default: 8)"
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start from this persona index (for resuming)"
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="End at this persona index (for partial runs)"
    )
    parser.add_argument(
        "--no_val",
        action="store_true",
        help="Don't use validation set during training"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip personas that already have trained models"
    )

    args = parser.parse_args()

    # Load persona IDs
    splits_file = Path(args.splits_file)
    print(f"Loading persona IDs from {splits_file}")
    persona_ids = load_persona_ids(splits_file)
    print(f"[OK] Found {len(persona_ids)} personas")

    # Filter by index range
    if args.end_index is not None:
        persona_ids = persona_ids[args.start_index:args.end_index]
    else:
        persona_ids = persona_ids[args.start_index:]

    print(f"[OK] Will train {len(persona_ids)} personas (indices {args.start_index} to {args.start_index + len(persona_ids) - 1})")

    # Setup paths
    unified_model_path = Path(args.unified_model)
    config_path = Path(args.config)
    output_base_dir = Path(args.output_dir)

    if not unified_model_path.exists():
        print(f"[ERROR] Unified model not found at {unified_model_path}")
        print("Please train the unified model first using scripts/train_unified_lora.py")
        sys.exit(1)

    # Train each persona
    start_time = datetime.now()
    success_count = 0
    failed_personas = []
    skipped_count = 0

    for i, persona_id in enumerate(persona_ids, 1):
        # Check if already exists
        if args.skip_existing:
            output_dir = output_base_dir / persona_id
            if (output_dir / "test_metrics.json").exists():
                print(f"\n[SKIP] {persona_id} already trained ({i}/{len(persona_ids)})")
                skipped_count += 1
                continue

        print(f"\n{'#'*80}")
        print(f"# Training {i}/{len(persona_ids)}: {persona_id}")
        print(f"{'#'*80}")

        success = train_single_persona(
            persona_id=persona_id,
            unified_model_path=unified_model_path,
            config_path=config_path,
            output_base_dir=output_base_dir,
            persona_rank=args.persona_rank,
            no_val=args.no_val
        )

        if success:
            success_count += 1
            print(f"[OK] {persona_id} trained successfully")
        else:
            failed_personas.append(persona_id)
            print(f"[ERROR] {persona_id} training failed")

    # Aggregate results
    print(f"\n{'='*80}")
    print("AGGREGATING RESULTS")
    print(f"{'='*80}")

    summary_file = Path("results/hybrid/hybrid_summary.json")
    aggregate_results(output_base_dir, summary_file)

    # Print final summary
    elapsed_time = datetime.now() - start_time

    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total personas: {len(persona_ids)}")
    print(f"Successfully trained: {success_count}")
    print(f"Skipped (already exists): {skipped_count}")
    print(f"Failed: {len(failed_personas)}")
    print(f"Elapsed time: {elapsed_time}")

    if failed_personas:
        print(f"\nFailed personas:")
        for persona_id in failed_personas:
            print(f"  - {persona_id}")

    print(f"\n[OK] All done!")
    print(f"Results saved to: {summary_file}")


if __name__ == "__main__":
    main()
