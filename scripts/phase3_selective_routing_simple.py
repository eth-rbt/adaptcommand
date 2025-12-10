"""
Phase 3: Selective Routing - Simplified Visualization
Shows routing distribution and top improvements only
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")

def load_routing_data():
    """Load selective routing results"""
    routing_path = Path("results/selective_routing/routing_decisions.json")

    with open(routing_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data['routing_decisions'])
    summary = data['summary']

    return df, summary

def create_simplified_viz(df, summary):
    """Create simplified 2-panel visualization"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Main title
    fig.suptitle('Phase 3: Selective Routing - Distribution & Top Improvements',
                 fontsize=20, fontweight='bold', y=0.98)

    # Panel 1: Pie Chart - Routing Decision Distribution
    create_routing_pie_chart(df, summary, ax1)

    # Panel 2: Top 20 Improvements Bar Chart
    create_top_improvements_bar(df, ax2)

    plt.tight_layout()

    return fig

def create_routing_pie_chart(df, summary, ax):
    """Pie chart showing routing decision distribution"""

    counts = df['best_model'].value_counts()
    colors = {
        'unified': '#2ecc71',
        'hybrid': '#3498db',
        'personalized': '#e74c3c'
    }

    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        [counts.get('unified', 0), counts.get('hybrid', 0), counts.get('personalized', 0)],
        labels=['Unified', 'Hybrid', 'Personalized'],
        autopct='%1.1f%%',
        startangle=90,
        colors=[colors['unified'], colors['hybrid'], colors['personalized']],
        textprops={'fontsize': 14, 'fontweight': 'bold'},
        explode=(0.05, 0.05, 0.05)  # Slight separation
    )

    # Make percentage text white and bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(16)
        autotext.set_fontweight('bold')

    # Make labels larger
    for text in texts:
        text.set_fontsize(14)
        text.set_fontweight('bold')

    ax.set_title('Routing Decision Distribution\n(200 personas)',
                 fontsize=16, fontweight='bold', pad=20)

    # Add detailed count annotations
    legend_labels = [
        f'Unified: {counts.get("unified", 0)} personas (77.5%)',
        f'Hybrid: {counts.get("hybrid", 0)} personas (20.5%)',
        f'Personalized: {counts.get("personalized", 0)} personas (2.0%)'
    ]
    ax.legend(legend_labels, loc='upper left', bbox_to_anchor=(-0.1, -0.05),
              fontsize=12, frameon=True, fancybox=True, shadow=True)

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

    bars = ax.barh(y_pos, improvements, color=bar_colors, alpha=0.85,
                   edgecolor='black', linewidth=1)

    # Add value labels on bars
    for i, (bar, val, model) in enumerate(zip(bars, improvements, top20['best_model'])):
        # Position label at end of bar
        ax.text(val + 0.3, i, f'+{val:.1f}%',
                va='center', fontsize=10, fontweight='bold')

    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels([p.replace('persona_', 'P') for p in top20['persona_id']],
                       fontsize=11, fontweight='bold')
    ax.set_xlabel('Improvement over Unified Baseline (%)',
                  fontsize=14, fontweight='bold')
    ax.set_title('Top 20 Personas with Biggest Gains',
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linewidth=1)
    ax.invert_yaxis()  # Highest at top

    # Set x-axis limit to give space for labels
    ax.set_xlim(0, max(improvements) * 1.15)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['personalized'], label='Personalized', alpha=0.85, edgecolor='black'),
        Patch(facecolor=colors['hybrid'], label='Hybrid', alpha=0.85, edgecolor='black'),
        Patch(facecolor=colors['unified'], label='Unified', alpha=0.85, edgecolor='black')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11,
              frameon=True, fancybox=True, shadow=True)

def main():
    print("Loading selective routing data...")
    df, summary = load_routing_data()

    print(f"Loaded routing decisions for {len(df)} personas")
    print(f"Creating simplified visualization...")

    fig = create_simplified_viz(df, summary)

    # Save
    output_path = Path("results/figures/phase3_selective_routing_simple.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[DONE] Saved visualization to {output_path}")

    # Print key stats
    print("\n" + "="*60)
    print("KEY STATISTICS")
    print("="*60)
    print(f"\nRouting Distribution:")
    print(f"  Unified:      {summary['unified_count']:3d} personas (77.5%)")
    print(f"  Hybrid:       {summary['hybrid_count']:3d} personas (20.5%)")
    print(f"  Personalized: {summary['personalized_count']:3d} personas ( 2.0%)")

    print(f"\nPerformance:")
    print(f"  Unified Baseline:     {summary['unified_score']:.4f}")
    print(f"  Selective Routing:    {summary['avg_selective_score']:.4f}")
    print(f"  Absolute Improvement: +{summary['improvement']:.4f}")
    print(f"  Relative Improvement: +{summary['improvement_pct']:.2f}%")

    print(f"\nTop 5 Improvers:")
    top5 = df.nlargest(5, 'improvement_over_unified')
    for idx, row in top5.iterrows():
        improvement_pct = row['improvement_over_unified'] * 100
        print(f"  {row['persona_id']}: +{improvement_pct:.2f}% ({row['best_model']})")

    print("\n[SUCCESS] Done!")

if __name__ == "__main__":
    main()
