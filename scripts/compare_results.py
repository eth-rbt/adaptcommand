"""
Compare Baseline vs Personalized LoRA Results

Compares the baseline model performance against personalized LoRA adapters.
"""

import json
import argparse
from pathlib import Path
import numpy as np


def load_baseline_results(baseline_file: Path):
    """Load baseline benchmark results."""
    with open(baseline_file, "r") as f:
        data = json.load(f)
    return data["metrics"]


def load_personalized_results(personalized_file: Path):
    """Load aggregated personalized results."""
    with open(personalized_file, "r") as f:
        return json.load(f)


def print_comparison(baseline_metrics, personalized_metrics):
    """Print detailed comparison."""
    print("\n" + "="*80)
    print("BASELINE vs PERSONALIZED COMPARISON")
    print("="*80)

    # Key metrics to compare
    metrics = [
        ("embedding_similarity", "Embedding Similarity"),
        ("device_precision", "Device Precision"),
        ("device_recall", "Device Recall"),
        ("param_precision", "Parameter Precision"),
        ("param_recall", "Parameter Recall"),
        ("param_f1", "Parameter F1"),
        ("numerical_precision", "Numerical Precision"),
        ("numerical_recall", "Numerical Recall"),
        ("avg_pred_length", "Avg Prediction Length"),
        ("avg_ref_length", "Avg Reference Length"),
        ("length_ratio", "Length Ratio"),
    ]

    print(f"\n{'Metric':<30} {'Baseline':>12} {'Personalized':>20} {'Delta':>12} {'Improvement':>12}")
    print("-" * 90)

    improvements = []

    for metric_key, metric_name in metrics:
        baseline_val = baseline_metrics.get(metric_key, 0)
        personalized_mean = personalized_metrics.get(f"{metric_key}_mean", 0)
        personalized_std = personalized_metrics.get(f"{metric_key}_std", 0)

        delta = personalized_mean - baseline_val

        # Calculate improvement percentage
        if baseline_val != 0:
            improvement_pct = (delta / baseline_val) * 100
        else:
            improvement_pct = 0

        # Track improvements for summary
        if metric_key in ["embedding_similarity", "device_precision", "device_recall",
                          "param_precision", "param_recall", "param_f1",
                          "numerical_precision", "numerical_recall"]:
            improvements.append((metric_name, improvement_pct))

        # Format output
        personalized_str = f"{personalized_mean:.4f}±{personalized_std:.4f}"
        delta_str = f"{delta:+.4f}"
        improvement_str = f"{improvement_pct:+.1f}%"

        print(f"{metric_name:<30} {baseline_val:>12.4f} {personalized_str:>20} {delta_str:>12} {improvement_str:>12}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    avg_improvement = np.mean([imp for _, imp in improvements])
    print(f"\nAverage improvement across key metrics: {avg_improvement:+.2f}%")

    # Count significant improvements
    significant_pos = sum(1 for _, imp in improvements if imp > 2)
    significant_neg = sum(1 for _, imp in improvements if imp < -2)

    print(f"Metrics with >2% improvement: {significant_pos}/{len(improvements)}")
    print(f"Metrics with >2% degradation: {significant_neg}/{len(improvements)}")

    # Best improvements
    print("\nTop Improvements:")
    sorted_improvements = sorted(improvements, key=lambda x: x[1], reverse=True)
    for metric_name, imp_pct in sorted_improvements[:5]:
        print(f"  {metric_name:<30}: {imp_pct:+.2f}%")

    # Additional stats
    print("\n" + "="*80)
    print("ADDITIONAL STATISTICS")
    print("="*80)
    print(f"Baseline examples evaluated: {baseline_metrics.get('num_examples', 'N/A')}")
    print(f"Personalized personas trained: {personalized_metrics.get('num_personas', 'N/A')}")

    # Per-persona variance
    print("\nPer-Persona Variance (showing spread in personalization):")
    variance_metrics = ["embedding_similarity", "param_f1", "numerical_precision"]
    for metric_key in variance_metrics:
        mean = personalized_metrics.get(f"{metric_key}_mean", 0)
        std = personalized_metrics.get(f"{metric_key}_std", 0)
        min_val = personalized_metrics.get(f"{metric_key}_min", 0)
        max_val = personalized_metrics.get(f"{metric_key}_max", 0)

        print(f"  {metric_key:<25}: {mean:.4f} ± {std:.4f} (min: {min_val:.4f}, max: {max_val:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs personalized results")
    parser.add_argument(
        "--baseline",
        type=str,
        default="results/baseline/baseline_results.json",
        help="Path to baseline results JSON"
    )
    parser.add_argument(
        "--personalized",
        type=str,
        default="results/personalized/personalized_summary.json",
        help="Path to personalized summary JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/comparison.json",
        help="Path to save comparison results"
    )

    args = parser.parse_args()

    # Load results
    print("Loading results...")
    baseline_metrics = load_baseline_results(Path(args.baseline))
    personalized_metrics = load_personalized_results(Path(args.personalized))

    # Print comparison
    print_comparison(baseline_metrics, personalized_metrics)

    # Save comparison
    comparison = {
        "baseline": baseline_metrics,
        "personalized": {
            "num_personas": personalized_metrics.get("num_personas"),
            "metrics": {
                k: v for k, v in personalized_metrics.items()
                if k.endswith("_mean") or k.endswith("_std") or k.endswith("_min") or k.endswith("_max")
            }
        },
        "deltas": {
            metric_key: personalized_metrics.get(f"{metric_key}_mean", 0) - baseline_metrics.get(metric_key, 0)
            for metric_key in ["embedding_similarity", "device_precision", "device_recall",
                              "param_precision", "param_recall", "param_f1",
                              "numerical_precision", "numerical_recall"]
        }
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\n[OK] Comparison saved to {output_path}")


if __name__ == "__main__":
    main()
