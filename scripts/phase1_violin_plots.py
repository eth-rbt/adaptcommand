"""
Generate violin plots for Phase 1 baseline adaptation methods
Shows distribution of metrics across personas for each method
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11

def load_results():
    """Load all Phase 1 results"""
    results_dir = Path("results")

    data = []

    # 1. Unified LoRA - per-persona results
    unified_per_persona_path = results_dir / "unified" / "per_persona_results.json"
    if unified_per_persona_path.exists():
        with open(unified_per_persona_path) as f:
            unified_per_persona = json.load(f)
            for persona_id, persona_data in unified_per_persona.items():
                metrics = persona_data['metrics']
                data.append({
                    'method': 'Unified LoRA',
                    'embedding_similarity': metrics['embedding_similarity'],
                    'device_precision': metrics['device_precision'],
                    'param_precision': metrics['param_precision'],
                    'numerical_precision': metrics['numerical_precision']
                })

    # 2. Baseline (no adaptation) - per-persona results
    baseline_per_persona_path = results_dir / "baseline" / "per_persona_results.json"
    if baseline_per_persona_path.exists():
        with open(baseline_per_persona_path) as f:
            baseline_per_persona = json.load(f)
            for persona_id, persona_data in baseline_per_persona.items():
                metrics = persona_data['metrics']
                data.append({
                    'method': 'Baseline\n(No Adaptation)',
                    'embedding_similarity': metrics['embedding_similarity'],
                    'device_precision': metrics['device_precision'],
                    'param_precision': metrics['param_precision'],
                    'numerical_precision': metrics['numerical_precision']
                })

    # 3. Per-Persona LoRA - per-persona distributions
    personalized_path = results_dir / "personalized" / "personalized_summary.json"
    if personalized_path.exists():
        with open(personalized_path) as f:
            personalized = json.load(f)
            for persona_metrics in personalized['per_persona_metrics']:
                data.append({
                    'method': 'Per-Persona LoRA',
                    'embedding_similarity': persona_metrics['embedding_similarity'],
                    'device_precision': persona_metrics['device_precision'],
                    'param_precision': persona_metrics['param_precision'],
                    'numerical_precision': persona_metrics['numerical_precision']
                })

    # 4. Cluster 4 LoRA - per-persona in cluster 4
    cluster4_path = results_dir / "cluster_lora" / "cluster_lora_results.json"
    if cluster4_path.exists():
        with open(cluster4_path) as f:
            cluster_data = json.load(f)
            for persona_result in cluster_data['per_persona_results']:
                data.append({
                    'method': 'Cluster 4 LoRA\n(72 personas)',
                    'embedding_similarity': persona_result['embedding_similarity'],
                    'device_precision': persona_result['device_precision'],
                    'param_precision': None,  # Not available
                    'numerical_precision': persona_result['numerical_precision']
                })

    # 5. MoE Sparse - comprehensive results
    moe_path = results_dir / "moe_sparse" / "moe_sparse_comprehensive_results.json"
    if moe_path.exists():
        with open(moe_path) as f:
            moe = json.load(f)
            for persona_result in moe['per_persona_results']:
                data.append({
                    'method': 'MoE Sparse (K=5)',
                    'embedding_similarity': persona_result['embedding_similarity'],
                    'device_precision': persona_result['device_precision'],
                    'param_precision': None,  # Not available
                    'numerical_precision': persona_result['numerical_precision']
                })

    # 6. Weighted Merge - comprehensive results
    weighted_path = results_dir / "weighted_merge" / "weighted_merge_comprehensive.json"
    if weighted_path.exists():
        with open(weighted_path) as f:
            weighted = json.load(f)
            for persona_result in weighted['per_persona_results']:
                data.append({
                    'method': 'Weighted Merge',
                    'embedding_similarity': persona_result['embedding_similarity'],
                    'device_precision': persona_result['device_precision'],
                    'param_precision': None,  # Not available
                    'numerical_precision': persona_result['numerical_precision']
                })

    return pd.DataFrame(data)

def create_violin_plots(df):
    """Create violin plots for Phase 1 metrics"""

    # Define method order (best to worst by embedding similarity)
    method_order = [
        'Unified LoRA',
        'Cluster 4 LoRA\n(72 personas)',
        'Per-Persona LoRA',
        'MoE Sparse (K=5)',
        'Baseline\n(No Adaptation)',
        'Weighted Merge'
    ]

    # Filter to only methods we have
    method_order = [m for m in method_order if m in df['method'].unique()]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Phase 1: Baseline Adaptation Methods - Metric Distributions',
                 fontsize=18, fontweight='bold', y=0.995)

    metrics = [
        ('embedding_similarity', 'Embedding Similarity', axes[0, 0]),
        ('device_precision', 'Device Precision', axes[0, 1]),
        ('param_precision', 'Parameter Precision', axes[1, 0]),
        ('numerical_precision', 'Numerical Precision', axes[1, 1])
    ]

    colors = {
        'Unified LoRA': '#2ecc71',  # Green (winner)
        'Cluster 4 LoRA\n(72 personas)': '#3498db',  # Blue
        'Per-Persona LoRA': '#e74c3c',  # Red (overfitting)
        'MoE Sparse (K=5)': '#f39c12',  # Orange
        'Baseline\n(No Adaptation)': '#95a5a6',  # Gray
        'Weighted Merge': '#8e44ad'  # Purple (worst)
    }

    for metric_col, metric_name, ax in metrics:
        # Filter out None values for this metric
        plot_df = df[df[metric_col].notna()].copy()

        if len(plot_df) == 0:
            ax.text(0.5, 0.5, f'No data for {metric_name}',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            continue

        # Create violin plot
        parts = ax.violinplot(
            [plot_df[plot_df['method'] == m][metric_col].values
             for m in method_order if m in plot_df['method'].unique()],
            positions=range(len([m for m in method_order if m in plot_df['method'].unique()])),
            showmeans=True,
            showextrema=True,
            widths=0.7
        )

        # Color the violins
        available_methods = [m for m in method_order if m in plot_df['method'].unique()]
        for i, pc in enumerate(parts['bodies']):
            method = available_methods[i]
            pc.set_facecolor(colors.get(method, '#95a5a6'))
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)

        # Style mean lines
        parts['cmeans'].set_color('black')
        parts['cmeans'].set_linewidth(2)

        # Add mean value annotations
        for i, method in enumerate(available_methods):
            method_data = plot_df[plot_df['method'] == method][metric_col]
            mean_val = method_data.mean()
            ax.text(i, mean_val, f'{mean_val:.2%}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='black', alpha=0.8))

        # Set labels and title
        ax.set_ylabel(metric_name, fontsize=13, fontweight='bold')
        ax.set_xticks(range(len(available_methods)))
        ax.set_xticklabels(available_methods, rotation=15, ha='right', fontsize=10)
        ax.set_title(f'{metric_name} Distribution', fontsize=14, fontweight='bold', pad=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(bottom=0, top=1.05)

        # Add horizontal line at unified performance for reference
        if 'Unified LoRA' in available_methods:
            unified_val = plot_df[plot_df['method'] == 'Unified LoRA'][metric_col].mean()
            ax.axhline(unified_val, color='green', linestyle='--', alpha=0.5, linewidth=2,
                      label=f'Unified: {unified_val:.2%}')
            ax.legend(loc='lower right', fontsize=9)

    plt.tight_layout()

    # Save
    output_path = Path("results/figures/phase1_violin_plots.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[DONE] Saved violin plots to {output_path}")

    return fig

def print_statistics(df):
    """Print summary statistics for each method"""
    print("\n" + "="*80)
    print("PHASE 1 STATISTICS SUMMARY")
    print("="*80)

    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        print(f"\n{method}:")
        print(f"  N = {len(method_df)}")

        for metric in ['embedding_similarity', 'device_precision', 'param_precision', 'numerical_precision']:
            values = method_df[metric].dropna()
            if len(values) > 0:
                print(f"  {metric}:")
                print(f"    Mean: {values.mean():.4f}")
                if len(values) > 1:
                    print(f"    Std:  {values.std():.4f}")
                    print(f"    Min:  {values.min():.4f}")
                    print(f"    Max:  {values.max():.4f}")

def main():
    print("Loading Phase 1 results...")
    df = load_results()

    print(f"Loaded {len(df)} data points across {df['method'].nunique()} methods")
    print(f"Methods: {', '.join(df['method'].unique())}")

    print("\nCreating violin plots...")
    create_violin_plots(df)

    print_statistics(df)

    print("\n[SUCCESS] Done!")

if __name__ == "__main__":
    main()
