"""
Comprehensive Comparison of All Personalization Methods

Compares:
- Baseline
- Unified LoRA
- Per-Persona LoRA
- Hybrid LoRA
- Global Prefix
- Per-User Prefix (when available)
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

def load_results():
    """Load all available results"""
    results = {}

    # Baseline
    baseline_path = Path("results/baseline/baseline_results.json")
    if baseline_path.exists():
        with open(baseline_path) as f:
            data = json.load(f)
            results["Baseline"] = data.get("metrics", {})

    # Unified LoRA
    unified_path = Path("results/unified/unified_results.json")
    if unified_path.exists():
        with open(unified_path) as f:
            data = json.load(f)
            results["Unified LoRA"] = data.get("metrics", {})

    # Per-Persona LoRA
    perpersona_path = Path("results/personalized/personalized_summary.json")
    if perpersona_path.exists():
        with open(perpersona_path) as f:
            data = json.load(f)
            # Extract mean metrics
            results["Per-Persona LoRA"] = {
                "embedding_similarity": data.get("embedding_similarity_mean", 0),
                "device_precision": data.get("device_precision_mean", 0),
                "param_f1": data.get("param_f1_mean", 0),
                "numerical_precision": data.get("numerical_precision_mean", 0)
            }

    # Hybrid LoRA
    hybrid_path = Path("results/hybrid/hybrid_summary.json")
    if hybrid_path.exists():
        with open(hybrid_path) as f:
            data = json.load(f)
            results["Hybrid LoRA"] = {
                "embedding_similarity": data.get("embedding_similarity_mean", 0),
                "device_precision": data.get("device_precision_mean", 0),
                "param_f1": data.get("param_f1_mean", 0),
                "numerical_precision": data.get("numerical_precision_mean", 0)
            }

    # Global Prefix
    prefix_all_path = Path("models/prefix_all/test_metrics.json")
    if prefix_all_path.exists():
        with open(prefix_all_path) as f:
            results["Global Prefix"] = json.load(f)

    # Per-User Prefix
    prefix_peruser_path = Path("results/prefix_per_user/prefix_per_user_summary.json")
    if prefix_peruser_path.exists():
        with open(prefix_peruser_path) as f:
            data = json.load(f)
            results["Per-User Prefix"] = {
                "embedding_similarity": data.get("embedding_similarity_mean", 0),
                "device_precision": data.get("device_precision_mean", 0),
                "param_f1": data.get("param_f1_mean", 0),
                "numerical_precision": data.get("numerical_precision_mean", 0)
            }

    return results


def create_comparison_plots(results):
    """Create comprehensive comparison plots"""

    # Extract methods and metrics
    methods = list(results.keys())

    # Define key metrics to compare
    metrics = [
        "embedding_similarity",
        "device_precision",
        "param_f1",
        "numerical_precision"
    ]

    metric_labels = {
        "embedding_similarity": "Embedding Similarity",
        "device_precision": "Device Precision",
        "param_f1": "Parameter F1",
        "numerical_precision": "Numerical Precision"
    }

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Color palette
    colors = plt.cm.Set2(range(len(methods)))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Extract values for this metric
        values = []
        method_names = []
        for method in methods:
            if metric in results[method]:
                values.append(results[method][metric])
                method_names.append(method)

        # Create bar plot
        bars = ax.bar(range(len(values)), values, color=colors[:len(values)])

        # Customize plot
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'{metric_labels[metric]}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(method_names)))
        ax.set_xticklabels(method_names, rotation=45, ha='right')
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}',
                   ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('results/figures/method_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: results/figures/method_comparison.png")
    plt.close()

    # Create overall comparison table plot
    create_table_plot(results, methods, metrics, metric_labels)

    # Create radar chart
    create_radar_chart(results, methods, metrics, metric_labels)


def create_table_plot(results, methods, metrics, metric_labels):
    """Create a table visualization of all results"""

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    table_data = []
    for method in methods:
        row = [method]
        for metric in metrics:
            if metric in results[method]:
                row.append(f"{results[method][metric]:.4f}")
            else:
                row.append("N/A")
        table_data.append(row)

    # Create table
    headers = ['Method'] + [metric_labels[m] for m in metrics]
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.25] + [0.15]*len(metrics))

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color code rows
    colors_list = plt.cm.Set2(range(len(methods)))
    for i in range(len(methods)):
        for j in range(len(headers)):
            table[(i+1, j)].set_facecolor(colors_list[i])
            table[(i+1, j)].set_alpha(0.3)

    plt.title('Performance Comparison Across All Methods',
             fontsize=16, fontweight='bold', pad=20)

    plt.savefig('results/figures/comparison_table.png', dpi=300, bbox_inches='tight')
    print(f"Saved: results/figures/comparison_table.png")
    plt.close()


def create_radar_chart(results, methods, metrics, metric_labels):
    """Create radar chart comparing all methods"""

    # Filter to methods with all metrics
    valid_methods = []
    for method in methods:
        if all(metric in results[method] for metric in metrics):
            valid_methods.append(method)

    if len(valid_methods) < 2:
        print("Not enough methods with complete metrics for radar chart")
        return

    # Number of variables
    num_vars = len(metrics)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Plot each method
    colors = plt.cm.Set2(range(len(valid_methods)))
    for idx, method in enumerate(valid_methods):
        values = [results[method][metric] for metric in metrics]
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])

    # Fix axis to go in the right order
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([metric_labels[m] for m in metrics], fontsize=11)

    # Set y-axis limits
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

    plt.title('Multi-Method Performance Comparison\n(Radar Chart)',
             fontsize=14, fontweight='bold', pad=20)

    plt.savefig('results/figures/radar_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: results/figures/radar_comparison.png")
    plt.close()


def create_parameter_efficiency_plot(results):
    """Plot performance vs model size"""

    # Model sizes (approximate)
    model_sizes = {
        "Baseline": 0,
        "Global Prefix": 0.8,  # MB
        "Per-User Prefix": 0.4,  # per user
        "Unified LoRA": 20,
        "Per-Persona LoRA": 5,  # per user
        "Hybrid LoRA": 10  # unified + persona
    }

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot points
    colors = plt.cm.Set2(range(len(results)))
    for idx, (method, metrics) in enumerate(results.items()):
        if method in model_sizes and "embedding_similarity" in metrics:
            x = model_sizes[method]
            y = metrics["embedding_similarity"]
            ax.scatter(x, y, s=300, alpha=0.6, c=[colors[idx]],
                      edgecolors='black', linewidth=2)
            ax.annotate(method, (x, y), xytext=(10, 10),
                       textcoords='offset points', fontsize=11,
                       bbox=dict(boxstyle='round,pad=0.5',
                                facecolor=colors[idx], alpha=0.3))

    ax.set_xlabel('Model Size (MB)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Embedding Similarity', fontsize=13, fontweight='bold')
    ax.set_title('Performance vs Model Size\n(Efficiency Analysis)',
                fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 0.9])

    plt.tight_layout()
    plt.savefig('results/figures/efficiency_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: results/figures/efficiency_comparison.png")
    plt.close()


def print_summary(results):
    """Print text summary of results"""

    print("\n" + "="*80)
    print("COMPREHENSIVE METHOD COMPARISON")
    print("="*80)

    for method, metrics in results.items():
        print(f"\n{method}:")
        print("-" * 40)
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric:30s}: {value:.4f}")

    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    # Find best methods for each metric
    key_metrics = ["embedding_similarity", "device_precision", "param_f1", "numerical_precision"]

    for metric in key_metrics:
        values = {m: r[metric] for m, r in results.items() if metric in r}
        if values:
            best_method = max(values, key=values.get)
            best_value = values[best_method]
            print(f"\nBest {metric}: {best_method} ({best_value:.4f})")


def main():
    # Create output directory
    Path("results/figures").mkdir(parents=True, exist_ok=True)

    print("Loading results from all methods...")
    results = load_results()

    if not results:
        print("No results found! Make sure you've run the experiments.")
        return

    print(f"Found results for {len(results)} methods:")
    for method in results.keys():
        print(f"  - {method}")

    print("\nCreating comparison visualizations...")
    create_comparison_plots(results)
    create_parameter_efficiency_plot(results)

    print_summary(results)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print("\nGenerated plots:")
    print("  1. results/figures/method_comparison.png - Bar charts for each metric")
    print("  2. results/figures/comparison_table.png - Summary table")
    print("  3. results/figures/radar_comparison.png - Radar chart")
    print("  4. results/figures/efficiency_comparison.png - Performance vs size")


if __name__ == "__main__":
    main()
