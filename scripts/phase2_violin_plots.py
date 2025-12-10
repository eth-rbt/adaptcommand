"""
Generate violin plots for Phase 2 lightweight adaptation methods
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
    """Load all Phase 2 results"""
    results_dir = Path("results")

    data = []

    # 1. Unified LoRA - baseline for comparison
    unified_path = results_dir / "unified" / "unified_results.json"
    if unified_path.exists():
        with open(unified_path) as f:
            unified = json.load(f)
            # Replicate the single score for all 200 personas to show as baseline
            for i in range(200):
                data.append({
                    'method': 'Unified LoRA\n(Baseline)',
                    'embedding_similarity': unified['metrics']['embedding_similarity'],
                    'device_precision': unified['metrics']['device_precision'],
                    'param_precision': unified['metrics']['param_precision'],
                    'numerical_precision': unified['metrics']['numerical_precision']
                })

    # 2. Hybrid LoRA - cluster + persona training
    hybrid_path = results_dir / "hybrid" / "hybrid_summary.json"
    if hybrid_path.exists():
        with open(hybrid_path) as f:
            hybrid = json.load(f)
            for persona_metrics in hybrid['per_persona_metrics']:
                data.append({
                    'method': 'Hybrid LoRA',
                    'embedding_similarity': persona_metrics['embedding_similarity'],
                    'device_precision': persona_metrics['device_precision'],
                    'param_precision': persona_metrics['param_precision'],
                    'numerical_precision': persona_metrics['numerical_precision']
                })

    # 3. Prefix Per-User - lightweight personalization
    prefix_path = results_dir / "prefix_per_user" / "prefix_per_user_summary.json"
    if prefix_path.exists():
        with open(prefix_path) as f:
            prefix = json.load(f)
            for persona_metrics in prefix['per_persona_metrics']:
                data.append({
                    'method': 'Prefix Per-User',
                    'embedding_similarity': persona_metrics['embedding_similarity'],
                    'device_precision': persona_metrics['device_precision'],
                    'param_precision': persona_metrics['param_precision'],
                    'numerical_precision': persona_metrics['numerical_precision']
                })

    # 4. Selective Routing - best model per persona
    routing_path = results_dir / "selective_routing" / "routing_decisions.json"
    if routing_path.exists():
        with open(routing_path) as f:
            routing = json.load(f)
            for decision in routing['routing_decisions']:
                # Extract metrics from the best model chosen
                best_model = decision['best_model']

                # Get the score for the best model
                if best_model == 'unified':
                    score_key = 'unified_score'
                elif best_model == 'hybrid':
                    score_key = 'hybrid_score'
                else:  # personalized
                    score_key = 'personalized_score'

                data.append({
                    'method': 'Selective Routing',
                    'embedding_similarity': decision['best_score'],
                    'device_precision': None,  # Not available in routing decisions
                    'param_precision': None,
                    'numerical_precision': None
                })

    return pd.DataFrame(data)

def create_violin_plots(df):
    """Create violin plots for Phase 2 metrics"""

    # Define method order
    method_order = [
        'Unified LoRA\n(Baseline)',
        'Selective Routing',
        'Hybrid LoRA',
        'Prefix Per-User'
    ]

    # Filter to only methods we have
    method_order = [m for m in method_order if m in df['method'].unique()]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Phase 2: Lightweight Adaptation Methods - Metric Distributions',
                 fontsize=18, fontweight='bold', y=0.995)

    metrics = [
        ('embedding_similarity', 'Embedding Similarity', axes[0, 0]),
        ('device_precision', 'Device Precision', axes[0, 1]),
        ('param_precision', 'Parameter Precision', axes[1, 0]),
        ('numerical_precision', 'Numerical Precision', axes[1, 1])
    ]

    colors = {
        'Unified LoRA\n(Baseline)': '#2ecc71',  # Green
        'Selective Routing': '#9b59b6',  # Purple (best)
        'Hybrid LoRA': '#3498db',  # Blue
        'Prefix Per-User': '#e67e22'  # Orange
    }

    for metric_col, metric_name, ax in metrics:
        # Filter out None values for this metric
        plot_df = df[df[metric_col].notna()].copy()

        if len(plot_df) == 0:
            ax.text(0.5, 0.5, f'No data for {metric_name}',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_xticks([])
            continue

        available_methods = [m for m in method_order if m in plot_df['method'].unique()]

        # Create violin plot
        parts = ax.violinplot(
            [plot_df[plot_df['method'] == m][metric_col].values
             for m in available_methods],
            positions=range(len(available_methods)),
            showmeans=True,
            showextrema=True,
            widths=0.7
        )

        # Color the violins
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
            std_val = method_data.std()

            # Position annotation
            y_pos = mean_val
            label = f'{mean_val:.2%}'
            if std_val > 0.001:  # If there's variance, show it
                label += f'\n(±{std_val:.2%})'

            ax.text(i, y_pos, label,
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
        if 'Unified LoRA\n(Baseline)' in available_methods:
            unified_val = plot_df[plot_df['method'] == 'Unified LoRA\n(Baseline)'][metric_col].mean()
            ax.axhline(unified_val, color='green', linestyle='--', alpha=0.5, linewidth=2,
                      label=f'Unified: {unified_val:.2%}')

        # Add selective routing line if available
        if 'Selective Routing' in available_methods and metric_col == 'embedding_similarity':
            routing_val = plot_df[plot_df['method'] == 'Selective Routing'][metric_col].mean()
            ax.axhline(routing_val, color='purple', linestyle='--', alpha=0.5, linewidth=2,
                      label=f'Routing: {routing_val:.2%}')
            ax.legend(loc='lower right', fontsize=9)

    plt.tight_layout()

    # Save
    output_path = Path("results/figures/phase2_violin_plots.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[DONE] Saved violin plots to {output_path}")

    return fig

def print_statistics(df):
    """Print summary statistics for each method"""
    print("\n" + "="*80)
    print("PHASE 2 STATISTICS SUMMARY")
    print("="*80)

    for method in ['Unified LoRA\n(Baseline)', 'Selective Routing', 'Hybrid LoRA', 'Prefix Per-User']:
        if method not in df['method'].unique():
            continue

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

    # Calculate improvements over unified
    print("\n" + "="*80)
    print("IMPROVEMENTS OVER UNIFIED BASELINE")
    print("="*80)

    unified_emb = df[df['method'] == 'Unified LoRA\n(Baseline)']['embedding_similarity'].mean()

    for method in ['Selective Routing', 'Hybrid LoRA', 'Prefix Per-User']:
        if method not in df['method'].unique():
            continue
        method_df = df[df['method'] == method]
        method_emb = method_df['embedding_similarity'].dropna().mean()

        if method_emb:
            abs_improvement = method_emb - unified_emb
            rel_improvement = (abs_improvement / unified_emb) * 100
            print(f"\n{method}:")
            print(f"  Embedding Similarity: {method_emb:.4f}")
            print(f"  Absolute improvement: {abs_improvement:+.4f} ({abs_improvement*100:+.2f}%)")
            print(f"  Relative improvement: {rel_improvement:+.2f}%")

def main():
    print("Loading Phase 2 results...")
    df = load_results()

    print(f"Loaded {len(df)} data points across {df['method'].nunique()} methods")
    print(f"Methods: {', '.join(df['method'].unique())}")

    print("\nCreating violin plots...")
    create_violin_plots(df)

    print_statistics(df)

    print("\n[SUCCESS] Done!")

if __name__ == "__main__":
    main()
