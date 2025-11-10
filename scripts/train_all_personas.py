"""
Train LoRA Adapters for Multiple Personas

Trains personalized LoRA adapters for a specified number of personas
and aggregates their results for comparison with baseline.
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from train_lora_per_user import train_lora_for_persona, load_config


def get_persona_ids(splits_file: Path, num_personas: int = None):
    """Get list of persona IDs to train."""
    with open(splits_file, "r") as f:
        splits = json.load(f)

    persona_ids = sorted(splits.keys())

    if num_personas:
        persona_ids = persona_ids[:num_personas]

    return persona_ids


def aggregate_results(results_dir: Path, persona_ids: list):
    """
    Aggregate test metrics across all personas.

    Args:
        results_dir: Base directory containing persona results
        persona_ids: List of persona IDs

    Returns:
        Dictionary with aggregated metrics
    """
    all_metrics = []

    for persona_id in persona_ids:
        test_metrics_file = results_dir / persona_id / "test_metrics.json"

        if test_metrics_file.exists():
            with open(test_metrics_file, "r") as f:
                metrics = json.load(f)
                metrics["persona_id"] = persona_id
                all_metrics.append(metrics)

    if not all_metrics:
        print("No metrics found!")
        return {}

    # Aggregate metrics
    aggregated = {
        "num_personas": len(all_metrics),
        "per_persona_metrics": all_metrics
    }

    # Compute mean and std for each metric
    metric_keys = [k for k in all_metrics[0].keys() if k != "persona_id"]

    for key in metric_keys:
        values = [m[key] for m in all_metrics]
        aggregated[f"{key}_mean"] = float(np.mean(values))
        aggregated[f"{key}_std"] = float(np.std(values))
        aggregated[f"{key}_min"] = float(np.min(values))
        aggregated[f"{key}_max"] = float(np.max(values))

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapters for multiple personas")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/lora_training.json",
        help="Path to training config file"
    )
    parser.add_argument(
        "--num_personas",
        type=int,
        default=50,
        help="Number of personas to train (default: 50, use -1 for all)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/lora_adapters",
        help="Base output directory for adapters"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/personalized",
        help="Directory to save aggregated results"
    )
    parser.add_argument(
        "--start_from",
        type=int,
        default=0,
        help="Start from this persona index (for resuming)"
    )

    args = parser.parse_args()

    # Load config
    config = load_config(Path(args.config))

    # Get persona IDs
    splits_file = Path(config.get("splits_path", "data/splits/edgesplits.json"))
    num_personas = None if args.num_personas == -1 else args.num_personas
    persona_ids = get_persona_ids(splits_file, num_personas)

    # Apply start_from
    if args.start_from > 0:
        persona_ids = persona_ids[args.start_from:]
        print(f"Resuming from persona index {args.start_from}")

    print("="*60)
    print("TRAINING LORA ADAPTERS FOR MULTIPLE PERSONAS")
    print("="*60)
    print(f"Number of personas: {len(persona_ids)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Personas: {persona_ids[0]} to {persona_ids[-1]}")
    print("="*60)

    # Train each persona
    output_base = Path(args.output_dir)
    failed_personas = []

    for i, persona_id in enumerate(tqdm(persona_ids, desc="Training personas")):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(persona_ids)}] Training {persona_id}")
        print(f"{'='*60}")

        try:
            persona_output_dir = output_base / persona_id

            # Skip if already trained
            if (persona_output_dir / "test_metrics.json").exists():
                print(f"[SKIP] {persona_id} already trained")
                continue

            train_lora_for_persona(
                persona_id=persona_id,
                config=config,
                output_dir=persona_output_dir,
                use_val=True
            )

            print(f"[OK] {persona_id} completed successfully")

        except Exception as e:
            print(f"[ERROR] Failed to train {persona_id}: {e}")
            failed_personas.append(persona_id)

    # Aggregate results
    print("\n" + "="*60)
    print("AGGREGATING RESULTS")
    print("="*60)

    aggregated = aggregate_results(output_base, persona_ids)

    # Save aggregated results
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_file = results_dir / "personalized_summary.json"
    with open(summary_file, "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"\n[OK] Aggregated results saved to {summary_file}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Personas trained: {aggregated['num_personas']}")
    print(f"Failed: {len(failed_personas)}")

    if failed_personas:
        print(f"\nFailed personas: {failed_personas}")

    print("\nKey Metrics (mean ± std):")
    print(f"  Embedding Similarity:    {aggregated.get('embedding_similarity_mean', 0):.4f} ± {aggregated.get('embedding_similarity_std', 0):.4f}")
    print(f"  Device Precision:        {aggregated.get('device_precision_mean', 0):.4f} ± {aggregated.get('device_precision_std', 0):.4f}")
    print(f"  Device Recall:           {aggregated.get('device_recall_mean', 0):.4f} ± {aggregated.get('device_recall_std', 0):.4f}")
    print(f"  Param Precision:         {aggregated.get('param_precision_mean', 0):.4f} ± {aggregated.get('param_precision_std', 0):.4f}")
    print(f"  Param Recall:            {aggregated.get('param_recall_mean', 0):.4f} ± {aggregated.get('param_recall_std', 0):.4f}")
    print(f"  Param F1:                {aggregated.get('param_f1_mean', 0):.4f} ± {aggregated.get('param_f1_std', 0):.4f}")
    print(f"  Numerical Precision:     {aggregated.get('numerical_precision_mean', 0):.4f} ± {aggregated.get('numerical_precision_std', 0):.4f}")
    print(f"  Numerical Recall:        {aggregated.get('numerical_recall_mean', 0):.4f} ± {aggregated.get('numerical_recall_std', 0):.4f}")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
