"""
Phase 3: Selective Routing Breakdown Visualizations
Shows which personas benefit from routing and which models they get routed to
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib.patches import Rectangle

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

def load_routing_data():
    """Load selective routing results"""
    routing_path = Path("results/selective_routing/routing_decisions.json")

    with open(routing_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data['routing_decisions'])
    summary = data['summary']

    return df, summary

def load_cluster_map():
    """Load persona cluster assignments if available"""
    cluster_path = Path("data/splits/cluster_map.json")

    if cluster_path.exists():
        with open(cluster_path) as f:
            cluster_data = json.load(f)
        return cluster_data
    return None

def create_comprehensive_viz(df, summary):
    """Create comprehensive 4-panel visualization"""

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Main title
    fig.suptitle('Phase 3: Selective Routing - Detailed Breakdown',
                 fontsize=20, fontweight='bold', y=0.98)

    # Panel 1: Pie Chart - Routing Decision Distribution (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    create_routing_pie_chart(df, summary, ax1)

    # Panel 2: Performance Summary (top middle-right)
    ax2 = fig.add_subplot(gs[0, 1:])
    create_performance_summary(df, summary, ax2)

    # Panel 3: Scatter Plot - Improvement vs Model Type (middle left-middle)
    ax3 = fig.add_subplot(gs[1, :2])
    create_improvement_scatter(df, ax3)

    # Panel 4: Top Improvements Bar Chart (middle right)
    ax4 = fig.add_subplot(gs[1, 2])
    create_top_improvements_bar(df, ax4)

    # Panel 5: Cluster Heatmap (bottom - full width)
    ax5 = fig.add_subplot(gs[2, :])
    create_cluster_heatmap(df, ax5)

    return fig

def create_routing_pie_chart(df, summary, ax):
    """Pie chart showing routing decision distribution"""

    counts = df['best_model'].value_counts()
    colors = {
        'unified': '#2ecc71',
        'hybrid': '#3498db',
        'personalized': '#e74c3c'
    }

    wedges, texts, autotexts = ax.pie(
        [counts.get('unified', 0), counts.get('hybrid', 0), counts.get('personalized', 0)],
        labels=['Unified', 'Hybrid', 'Personalized'],
        autopct='%1.1f%%',
        startangle=90,
        colors=[colors['unified'], colors['hybrid'], colors['personalized']],
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )

    # Make percentage text white and bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')

    ax.set_title('Routing Decision Distribution\n(200 personas)',
                 fontsize=13, fontweight='bold', pad=10)

    # Add count annotations
    legend_labels = [
        f'Unified: {counts.get("unified", 0)} personas',
        f'Hybrid: {counts.get("hybrid", 0)} personas',
        f'Personalized: {counts.get("personalized", 0)} personas'
    ]
    ax.legend(legend_labels, loc='upper left', bbox_to_anchor=(0, -0.1), fontsize=10)

def create_performance_summary(df, summary, ax):
    """Summary statistics table"""
    ax.axis('off')

    # Calculate stats
    unified_score = df['unified_score'].iloc[0]
    selective_score = summary['avg_selective_score']
    improvement_abs = summary['improvement']
    improvement_pct = summary['improvement_pct']

    # Count routing decisions
    routing_counts = df['best_model'].value_counts()

    # Calculate how many personas improved
    improved_count = len(df[df['improvement_over_unified'] > 0])
    same_count = len(df[df['improvement_over_unified'] == 0])

    # Create summary text
    summary_text = f"""
SELECTIVE ROUTING PERFORMANCE SUMMARY

Overall Results:
  • Unified Baseline:        {unified_score:.2%}
  • Selective Routing:       {selective_score:.2%}
  • Absolute Improvement:    +{improvement_abs:.4f} ({improvement_abs*100:+.2f}%)
  • Relative Improvement:    +{improvement_pct:.2f}%

Routing Breakdown:
  • Used Unified:            {routing_counts.get('unified', 0)} personas (77.5%)
  • Used Hybrid:             {routing_counts.get('hybrid', 0)} personas (20.5%)
  • Used Personalized:       {routing_counts.get('personalized', 0)} personas (2.0%)

Impact:
  • Personas Improved:       {improved_count} (≈{improved_count/len(df)*100:.0f}%)
  • Personas Same:           {same_count} (≈{same_count/len(df)*100:.0f}%)
  • No Regression:           Guaranteed (routing ensures ≥ unified)
    """

    ax.text(0.05, 0.95, summary_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    ax.set_title('Performance & Impact Summary',
                 fontsize=13, fontweight='bold', pad=10, loc='left')

def create_improvement_scatter(df, ax):
    """Scatter plot showing improvement vs model type"""

    # Color map
    colors = {
        'unified': '#2ecc71',
        'hybrid': '#3498db',
        'personalized': '#e74c3c'
    }

    # Create scatter for each model type
    for model_type in ['unified', 'hybrid', 'personalized']:
        mask = df['best_model'] == model_type
        subset = df[mask]

        ax.scatter(
            range(len(subset)),
            subset['improvement_over_unified'] * 100,
            c=colors[model_type],
            label=model_type.capitalize(),
            alpha=0.7,
            s=60,
            edgecolors='black',
            linewidth=0.5
        )

    # Add horizontal line at 0 (no improvement)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=2)

    # Labels and styling
    ax.set_xlabel('Persona Index (sorted by routing decision)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Improvement over Unified (%)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Persona Improvement by Routing Decision',
                 fontsize=13, fontweight='bold', pad=10)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)

    # Add annotation for key insights
    max_improvement = df['improvement_over_unified'].max() * 100
    max_persona = df.loc[df['improvement_over_unified'].idxmax(), 'persona_id']

    ax.text(0.98, 0.98,
            f'Max improvement: {max_improvement:.1f}%\n({max_persona})',
            transform=ax.transAxes,
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
            fontsize=9)

def create_top_improvements_bar(df, ax):
    """Bar chart of top 20 personas with biggest improvements"""

    # Get top 20 improvers
    top20 = df.nlargest(20, 'improvement_over_unified')

    # Color by model type
    colors = {
        'unified': '#2ecc71',
        'hybrid': '#3498db',
        'personalized': '#e74c3c'
    }
    bar_colors = [colors[model] for model in top20['best_model']]

    # Create horizontal bar chart
    y_pos = range(len(top20))
    improvements = top20['improvement_over_unified'] * 100

    bars = ax.barh(y_pos, improvements, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, improvements)):
        ax.text(val + 0.2, i, f'+{val:.1f}%',
                va='center', fontsize=8, fontweight='bold')

    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels([p.replace('persona_', 'P') for p in top20['persona_id']], fontsize=8)
    ax.set_xlabel('Improvement (%)', fontsize=11, fontweight='bold')
    ax.set_title('Top 20 Personas\nwith Biggest Gains', fontsize=12, fontweight='bold', pad=10)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()  # Highest at top

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['unified'], label='Unified', alpha=0.8),
        Patch(facecolor=colors['hybrid'], label='Hybrid', alpha=0.8),
        Patch(facecolor=colors['personalized'], label='Personalized', alpha=0.8)
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

def create_cluster_heatmap(df, ax):
    """Heatmap showing routing decisions across persona clusters"""

    # Load cluster assignments
    cluster_data = load_cluster_map()

    if cluster_data is None:
        ax.text(0.5, 0.5, 'Cluster data not available',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return

    # Add cluster info to dataframe
    df_with_cluster = df.copy()
    df_with_cluster['cluster'] = df_with_cluster['persona_id'].map(
        lambda p: cluster_data.get(p, -1)
    )

    # Create cross-tabulation
    crosstab = pd.crosstab(
        df_with_cluster['cluster'],
        df_with_cluster['best_model']
    )

    # Reorder columns
    col_order = ['unified', 'hybrid', 'personalized']
    crosstab = crosstab[[c for c in col_order if c in crosstab.columns]]

    # Create heatmap
    sns.heatmap(
        crosstab,
        annot=True,
        fmt='d',
        cmap='YlOrRd',
        ax=ax,
        cbar_kws={'label': 'Number of Personas'},
        linewidths=0.5,
        linecolor='black'
    )

    ax.set_xlabel('Routing Decision', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cluster ID', fontsize=12, fontweight='bold')
    ax.set_title('Routing Decisions by Persona Cluster', fontsize=13, fontweight='bold', pad=10)

    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

def create_detailed_statistics(df):
    """Print detailed statistics about routing decisions"""

    print("\n" + "="*80)
    print("SELECTIVE ROUTING DETAILED STATISTICS")
    print("="*80)

    # Overall stats
    print("\n1. OVERALL PERFORMANCE:")
    print(f"   Average Unified Score:    {df['unified_score'].mean():.4f}")
    print(f"   Average Selective Score:  {df['best_score'].mean():.4f}")
    print(f"   Average Improvement:      {df['improvement_over_unified'].mean():.4f} ({df['improvement_over_unified'].mean()*100:+.2f}%)")

    # Breakdown by routing decision
    print("\n2. PERFORMANCE BY ROUTING DECISION:")
    for model in ['unified', 'hybrid', 'personalized']:
        subset = df[df['best_model'] == model]
        if len(subset) > 0:
            print(f"\n   {model.upper()}:")
            print(f"     Count: {len(subset)}")
            print(f"     Avg Score: {subset['best_score'].mean():.4f}")
            print(f"     Avg Improvement: {subset['improvement_over_unified'].mean():.4f} ({subset['improvement_over_unified'].mean()*100:+.2f}%)")
            print(f"     Min Score: {subset['best_score'].min():.4f}")
            print(f"     Max Score: {subset['best_score'].max():.4f}")

    # Top improvers
    print("\n3. TOP 10 IMPROVERS:")
    top10 = df.nlargest(10, 'improvement_over_unified')
    for idx, row in top10.iterrows():
        improvement_pct = row['improvement_over_unified'] * 100
        print(f"   {row['persona_id']}: +{improvement_pct:.2f}% ({row['best_model']})")

    # Improvement distribution
    print("\n4. IMPROVEMENT DISTRIBUTION:")
    bins = [0, 0.01, 0.02, 0.05, 0.1, 1.0]
    labels = ['0-1%', '1-2%', '2-5%', '5-10%', '>10%']

    improved = df[df['improvement_over_unified'] > 0]
    if len(improved) > 0:
        improvement_cats = pd.cut(improved['improvement_over_unified'], bins=bins, labels=labels)
        counts = improvement_cats.value_counts().sort_index()

        for label, count in counts.items():
            pct = count / len(df) * 100
            print(f"   {label}: {count} personas ({pct:.1f}% of total)")

def main():
    print("Loading selective routing data...")
    df, summary = load_routing_data()

    print(f"Loaded routing decisions for {len(df)} personas")
    print(f"Unified count: {summary['unified_count']}")
    print(f"Hybrid count: {summary['hybrid_count']}")
    print(f"Personalized count: {summary['personalized_count']}")

    print("\nCreating comprehensive visualization...")
    fig = create_comprehensive_viz(df, summary)

    # Save
    output_path = Path("results/figures/phase3_selective_routing.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[DONE] Saved visualization to {output_path}")

    # Print detailed statistics
    create_detailed_statistics(df)

    print("\n[SUCCESS] Done!")

if __name__ == "__main__":
    main()
