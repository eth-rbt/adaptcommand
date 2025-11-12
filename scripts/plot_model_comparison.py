"""
Model Comparison Plotting Script

Creates comprehensive comparison plots between baseline, personalized, and unified models.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def load_baseline_results(baseline_dir: Path):
    """Load baseline results from CSV."""
    csv_file = baseline_dir / "per_persona_summary.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"Baseline CSV not found: {csv_file}")

    df = pd.read_csv(csv_file)
    df['model'] = 'Baseline'
    return df


def load_personalized_results(personalized_dir: Path):
    """Load personalized results from JSON and convert to DataFrame."""
    json_file = personalized_dir / "personalized_summary.json"
    if not json_file.exists():
        raise FileNotFoundError(f"Personalized JSON not found: {json_file}")

    with open(json_file, 'r') as f:
        data = json.load(f)

    # Convert to DataFrame
    personas = data['per_persona_metrics']
    df = pd.DataFrame(personas)

    # Add missing columns if needed (to match baseline format)
    if 'num_predictions' not in df.columns:
        df['num_predictions'] = 10  # Placeholder, adjust if you know the actual count

    # Add categorical metrics if missing
    if 'categorical_precision' not in df.columns:
        df['categorical_precision'] = 0.0
    if 'categorical_recall' not in df.columns:
        df['categorical_recall'] = 0.0
    if 'categorical_f1' not in df.columns:
        df['categorical_f1'] = 0.0

    # Add numerical_f1 if missing
    if 'numerical_f1' not in df.columns:
        df['numerical_f1'] = 2 * (df['numerical_precision'] * df['numerical_recall']) / \
                             (df['numerical_precision'] + df['numerical_recall'])
        df['numerical_f1'] = df['numerical_f1'].fillna(0)

    df['model'] = 'Personalized'
    return df


def load_unified_results(unified_dir: Path):
    """Load unified results from CSV."""
    csv_file = unified_dir / "per_persona_summary.csv"
    if not csv_file.exists():
        print(f"Warning: Unified CSV not found: {csv_file}")
        return None

    df = pd.read_csv(csv_file)
    df['model'] = 'Unified'
    return df


def create_overall_comparison(baseline_df, personalized_df, unified_df, output_dir: Path):
    """Create bar plot comparing overall average metrics across models."""
    # Key metrics to compare
    metrics = [
        'embedding_similarity',
        'device_precision',
        'device_recall',
        'param_precision',
        'param_recall',
        'param_f1'
    ]

    # Compute averages for each model
    models = []

    baseline_avg = {metric: baseline_df[metric].mean() for metric in metrics}
    baseline_avg['model'] = 'Baseline'
    models.append(baseline_avg)

    personalized_avg = {metric: personalized_df[metric].mean() for metric in metrics}
    personalized_avg['model'] = 'Personalized'
    models.append(personalized_avg)

    if unified_df is not None:
        unified_avg = {metric: unified_df[metric].mean() for metric in metrics}
        unified_avg['model'] = 'Unified'
        models.append(unified_avg)

    # Create DataFrame
    comparison_df = pd.DataFrame(models)

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(metrics))
    width = 0.25

    colors = ['#3498db', '#e74c3c', '#2ecc71']

    for i, model_name in enumerate(comparison_df['model']):
        values = [comparison_df.loc[comparison_df['model'] == model_name, metric].values[0]
                  for metric in metrics]
        offset = (i - 1) * width
        ax.bar(x + offset, values, width, label=model_name, color=colors[i], alpha=0.8)

    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison - Average Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(output_dir / 'overall_comparison.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved overall comparison to {output_dir / 'overall_comparison.png'}")
    plt.close()

    return comparison_df


def create_distribution_plots(baseline_df, personalized_df, unified_df, output_dir: Path):
    """Create distribution plots for key metrics."""
    metrics = [
        ('embedding_similarity', 'Embedding Similarity'),
        ('device_precision', 'Device Precision'),
        ('param_f1', 'Parameter F1 Score')
    ]

    # Combine dataframes
    dfs = [baseline_df, personalized_df]
    if unified_df is not None:
        dfs.append(unified_df)
    combined_df = pd.concat(dfs, ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx]

        # Create violin plot
        sns.violinplot(data=combined_df, x='model', y=metric, ax=ax,
                       palette=['#3498db', '#e74c3c', '#2ecc71'][:len(dfs)])

        # Overlay box plot
        sns.boxplot(data=combined_df, x='model', y=metric, ax=ax,
                   width=0.3, palette=['#3498db', '#e74c3c', '#2ecc71'][:len(dfs)],
                   showcaps=False, boxprops={'facecolor':'None'},
                   showfliers=False, whiskerprops={'linewidth':0})

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Model', fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(output_dir / 'distribution_comparison.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved distribution comparison to {output_dir / 'distribution_comparison.png'}")
    plt.close()


def create_per_persona_heatmap(baseline_df, personalized_df, unified_df, output_dir: Path):
    """Create heatmap showing improvement per persona."""
    # Calculate improvement from baseline
    metrics = ['embedding_similarity', 'device_precision', 'device_recall',
               'param_precision', 'param_recall', 'param_f1']

    # Merge dataframes on persona_id
    baseline_subset = baseline_df[['persona_id'] + metrics].copy()
    personalized_subset = personalized_df[['persona_id'] + metrics].copy()

    # Calculate improvement
    merged = baseline_subset.merge(personalized_subset, on='persona_id', suffixes=('_baseline', '_personalized'))

    improvements = []
    for metric in metrics:
        improvement = ((merged[f'{metric}_personalized'] - merged[f'{metric}_baseline']) /
                      (merged[f'{metric}_baseline'] + 1e-10) * 100)
        improvements.append(improvement)

    improvement_df = pd.DataFrame(improvements,
                                  index=[m.replace('_', ' ').title() for m in metrics],
                                  columns=merged['persona_id'])

    # Plot first 50 personas for readability
    fig, ax = plt.subplots(figsize=(20, 6))
    sns.heatmap(improvement_df.iloc[:, :50], cmap='RdYlGn', center=0,
                annot=False, fmt='.1f', cbar_kws={'label': 'Improvement (%)'}, ax=ax)
    ax.set_title('Personalized vs Baseline: Per-Persona Improvement (%) - First 50 Personas',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Persona ID', fontsize=12)
    ax.set_ylabel('Metric', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / 'persona_improvement_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved improvement heatmap to {output_dir / 'persona_improvement_heatmap.png'}")
    plt.close()


def create_scatter_comparison(baseline_df, personalized_df, unified_df, output_dir: Path):
    """Create scatter plots comparing models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Personalized vs Baseline
    ax = axes[0]
    merged = baseline_df[['persona_id', 'embedding_similarity', 'param_f1']].merge(
        personalized_df[['persona_id', 'embedding_similarity', 'param_f1']],
        on='persona_id', suffixes=('_baseline', '_personalized')
    )

    ax.scatter(merged['embedding_similarity_baseline'],
               merged['embedding_similarity_personalized'],
               alpha=0.6, s=50, c='#e74c3c', label='Embedding Similarity')
    ax.scatter(merged['param_f1_baseline'],
               merged['param_f1_personalized'],
               alpha=0.6, s=50, c='#3498db', label='Parameter F1')

    # Add diagonal line
    lims = [0, 1.0]
    ax.plot(lims, lims, 'k--', alpha=0.3, zorder=0, linewidth=2)

    ax.set_xlabel('Baseline Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Personalized Score', fontsize=12, fontweight='bold')
    ax.set_title('Personalized vs Baseline - Per Persona', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)

    # Add text showing % above diagonal
    above_diag_emb = (merged['embedding_similarity_personalized'] > merged['embedding_similarity_baseline']).sum()
    above_diag_param = (merged['param_f1_personalized'] > merged['param_f1_baseline']).sum()
    total = len(merged)

    ax.text(0.05, 0.95, f'Above diagonal:\nEmbedding: {above_diag_emb}/{total} ({100*above_diag_emb/total:.1f}%)\n'
            f'Param F1: {above_diag_param}/{total} ({100*above_diag_param/total:.1f}%)',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Unified vs Baseline (if available)
    if unified_df is not None:
        ax = axes[1]
        merged = baseline_df[['persona_id', 'embedding_similarity', 'param_f1']].merge(
            unified_df[['persona_id', 'embedding_similarity', 'param_f1']],
            on='persona_id', suffixes=('_baseline', '_unified')
        )

        ax.scatter(merged['embedding_similarity_baseline'],
                   merged['embedding_similarity_unified'],
                   alpha=0.6, s=50, c='#2ecc71', label='Embedding Similarity')
        ax.scatter(merged['param_f1_baseline'],
                   merged['param_f1_unified'],
                   alpha=0.6, s=50, c='#9b59b6', label='Parameter F1')

        # Add diagonal line
        ax.plot(lims, lims, 'k--', alpha=0.3, zorder=0, linewidth=2)

        ax.set_xlabel('Baseline Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Unified Score', fontsize=12, fontweight='bold')
        ax.set_title('Unified vs Baseline - Per Persona', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)

        # Add text showing % above diagonal
        above_diag_emb = (merged['embedding_similarity_unified'] > merged['embedding_similarity_baseline']).sum()
        above_diag_param = (merged['param_f1_unified'] > merged['param_f1_baseline']).sum()
        total = len(merged)

        ax.text(0.05, 0.95, f'Above diagonal:\nEmbedding: {above_diag_emb}/{total} ({100*above_diag_emb/total:.1f}%)\n'
                f'Param F1: {above_diag_param}/{total} ({100*above_diag_param/total:.1f}%)',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax = axes[1]
        ax.text(0.5, 0.5, 'Unified results not available yet',
                transform=ax.transAxes, ha='center', va='center', fontsize=14)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_comparison.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved scatter comparison to {output_dir / 'scatter_comparison.png'}")
    plt.close()


def create_summary_table(baseline_df, personalized_df, unified_df, output_dir: Path):
    """Create and save summary statistics table."""
    metrics = [
        'embedding_similarity',
        'device_precision',
        'device_recall',
        'param_precision',
        'param_recall',
        'param_f1',
        'numerical_precision',
        'numerical_recall'
    ]

    summary_data = []

    for metric in metrics:
        row = {'Metric': metric.replace('_', ' ').title()}

        # Baseline
        row['Baseline Mean'] = f"{baseline_df[metric].mean():.4f}"
        row['Baseline Std'] = f"{baseline_df[metric].std():.4f}"

        # Personalized
        row['Personalized Mean'] = f"{personalized_df[metric].mean():.4f}"
        row['Personalized Std'] = f"{personalized_df[metric].std():.4f}"

        # Improvement
        improvement = ((personalized_df[metric].mean() - baseline_df[metric].mean()) /
                      (baseline_df[metric].mean() + 1e-10) * 100)
        row['Improvement (%)'] = f"{improvement:+.2f}%"

        # Unified (if available)
        if unified_df is not None:
            row['Unified Mean'] = f"{unified_df[metric].mean():.4f}"
            row['Unified Std'] = f"{unified_df[metric].std():.4f}"

            improvement_unified = ((unified_df[metric].mean() - baseline_df[metric].mean()) /
                                  (baseline_df[metric].mean() + 1e-10) * 100)
            row['Unified Improvement (%)'] = f"{improvement_unified:+.2f}%"

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)

    # Save to CSV
    csv_file = output_dir / 'summary_statistics.csv'
    summary_df.to_csv(csv_file, index=False)
    print(f"[OK] Saved summary statistics to {csv_file}")

    # Create pretty table plot
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns,
                     cellLoc='center', loc='center',
                     colWidths=[0.15, 0.12, 0.12, 0.12, 0.12, 0.12] +
                               ([0.12, 0.12, 0.12] if unified_df is not None else []))

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color improvement cells
    for i in range(1, len(summary_df) + 1):
        # Personalized improvement
        imp_col = summary_df.columns.get_loc('Improvement (%)')
        cell_val = summary_df.iloc[i-1, imp_col]
        if '+' in cell_val:
            table[(i, imp_col)].set_facecolor('#d5f4e6')
        else:
            table[(i, imp_col)].set_facecolor('#fadbd8')

        # Unified improvement (if exists)
        if 'Unified Improvement (%)' in summary_df.columns:
            imp_col = summary_df.columns.get_loc('Unified Improvement (%)')
            cell_val = summary_df.iloc[i-1, imp_col]
            if '+' in cell_val:
                table[(i, imp_col)].set_facecolor('#d5f4e6')
            else:
                table[(i, imp_col)].set_facecolor('#fadbd8')

    plt.title('Model Performance Summary Statistics', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'summary_table.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved summary table to {output_dir / 'summary_table.png'}")
    plt.close()

    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Create comparison plots between models")
    parser.add_argument(
        "--baseline_dir",
        type=str,
        default="results/baseline",
        help="Directory containing baseline results"
    )
    parser.add_argument(
        "--personalized_dir",
        type=str,
        default="results/personalized",
        help="Directory containing personalized results"
    )
    parser.add_argument(
        "--unified_dir",
        type=str,
        default="results/unified",
        help="Directory containing unified results"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/comparison",
        help="Output directory for plots"
    )

    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    personalized_dir = Path(args.personalized_dir)
    unified_dir = Path(args.unified_dir)
    output_dir = Path(args.output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("MODEL COMPARISON PLOTS")
    print("="*60)

    # Load data
    print("\nLoading baseline results...")
    baseline_df = load_baseline_results(baseline_dir)
    print(f"[OK] Loaded {len(baseline_df)} personas from baseline")

    print("\nLoading personalized results...")
    personalized_df = load_personalized_results(personalized_dir)
    print(f"[OK] Loaded {len(personalized_df)} personas from personalized")

    print("\nLoading unified results...")
    unified_df = load_unified_results(unified_dir)
    if unified_df is not None:
        print(f"[OK] Loaded {len(unified_df)} personas from unified")
    else:
        print("[INFO] Unified results not available - will create plots without it")

    # Create plots
    print("\nCreating comparison plots...")

    print("\n1. Overall comparison...")
    comparison_summary = create_overall_comparison(baseline_df, personalized_df, unified_df, output_dir)

    print("\n2. Distribution plots...")
    create_distribution_plots(baseline_df, personalized_df, unified_df, output_dir)

    print("\n3. Per-persona improvement heatmap...")
    create_per_persona_heatmap(baseline_df, personalized_df, unified_df, output_dir)

    print("\n4. Scatter comparison plots...")
    create_scatter_comparison(baseline_df, personalized_df, unified_df, output_dir)

    print("\n5. Summary statistics table...")
    summary_df = create_summary_table(baseline_df, personalized_df, unified_df, output_dir)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nAll plots saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - overall_comparison.png")
    print("  - distribution_comparison.png")
    print("  - persona_improvement_heatmap.png")
    print("  - scatter_comparison.png")
    print("  - summary_table.png")
    print("  - summary_statistics.csv")

    print("\nKey Findings:")
    print(comparison_summary.to_string(index=False))


if __name__ == "__main__":
    main()
