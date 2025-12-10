"""
Create Publication-Quality Graphs for Final Report

Generates clean, professional visualizations suitable for academic papers:
1. Complete methods comparison (all 7 methods)
2. Hybrid methods deep dive
3. Phase-by-phase progression
4. Routing analysis with percentages
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

sns.set_style("whitegrid")
sns.set_palette("Set2")


def load_all_results():
    """Load all experimental results"""
    results = {}

    # Baseline
    with open('results/baseline/baseline_results.json') as f:
        data = json.load(f)
        results['baseline'] = data['metrics']['embedding_similarity']

    # Unified LoRA
    with open('results/unified/unified_results.json') as f:
        data = json.load(f)
        results['unified'] = data['metrics']['embedding_similarity']

    # Per-Persona LoRA
    with open('results/personalized/personalized_summary.json') as f:
        data = json.load(f)
        per_persona_metrics = data['per_persona_metrics']
        results['per_persona'] = np.mean([p['embedding_similarity'] for p in per_persona_metrics])

    # Sparse MoE
    with open('results/moe_sparse/moe_sparse_results.json') as f:
        data = json.load(f)
        results['moe_sparse'] = data['summary']['embedding_similarity_mean']

    # Cluster LoRA
    with open('results/cluster_lora/cluster_lora_results.json') as f:
        data = json.load(f)
        results['cluster_lora'] = data['summary']['embedding_similarity_mean']

    # Hybrid LoRA
    with open('results/hybrid/hybrid_summary.json') as f:
        data = json.load(f)
        per_persona_metrics = data['per_persona_metrics']
        results['hybrid_lora'] = np.mean([p['embedding_similarity'] for p in per_persona_metrics])

    # Weighted Merge
    with open('results/weighted_merge/weighted_merge_cluster4_results.json') as f:
        data = json.load(f)
        results['weighted_merge'] = data['embedding_similarity']

    # Selective Routing
    with open('results/selective_routing/routing_decisions.json') as f:
        data = json.load(f)
        results['routing'] = data['summary']['avg_selective_score']
        results['routing_breakdown'] = {
            'unified': data['summary']['unified_count'],
            'personalized': data['summary']['personalized_count'],
            'hybrid': data['summary']['hybrid_count']
        }

    return results


def create_complete_methods_comparison(results):
    """
    Figure 1: Complete comparison of all methods
    Professional bar chart with annotations
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = [
        'Baseline\n(No Adapt)',
        'Unified\nLoRA',
        'Per-Persona\nLoRA',
        'Sparse\nMoE',
        'Cluster\nLoRA',
        'Hybrid\nLoRA',
        'Weighted\nMerge',
        'Selective\nRouting'
    ]

    scores = [
        results['baseline'],
        results['unified'],
        results['per_persona'],
        results['moe_sparse'],
        results['cluster_lora'],
        results['hybrid_lora'],
        results['weighted_merge'],
        results['routing']
    ]

    # Color scheme: baseline=gray, unified=green, failed methods=red, routing=blue
    colors = ['#7f7f7f', '#2ca02c', '#d62728', '#d62728',
              '#d62728', '#ff7f0e', '#d62728', '#1f77b4']

    bars = ax.barh(methods, scores, color=colors, alpha=0.85,
                   edgecolor='black', linewidth=1.2)

    # Add value labels
    unified_score = results['unified']
    for i, (bar, score) in enumerate(zip(bars, scores)):
        width = bar.get_width()

        # Score label
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.1%}',
                ha='left', va='center', fontsize=9, fontweight='bold')

        # Improvement label (skip baseline and unified)
        if i > 1:
            improvement = ((score - unified_score) / unified_score) * 100
            color = 'darkgreen' if improvement > 0 else 'darkred'
            sign = '+' if improvement > 0 else ''
            ax.text(width/2, bar.get_y() + bar.get_height()/2,
                   f'{sign}{improvement:.1f}%',
                   ha='center', va='center', fontsize=8,
                   color=color, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3',
                            facecolor='white', alpha=0.8, edgecolor=color, linewidth=1))

    # Highlight unified baseline
    bars[1].set_linewidth(2.5)
    bars[1].set_edgecolor('#2ca02c')

    # Unified baseline line
    ax.axvline(unified_score, color='#2ca02c', linestyle='--',
               linewidth=1.5, alpha=0.6, label=f'Unified Baseline ({unified_score:.1%})')

    ax.set_xlabel('Embedding Similarity', fontsize=11, fontweight='bold')
    ax.set_title('Complete Method Comparison: All Approaches Tested',
                 fontsize=12, fontweight='bold', pad=15)
    ax.set_xlim(0.6, 0.88)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='x', alpha=0.3, linewidth=0.8)

    # Add category labels
    ax.text(0.61, 7.5, 'Reference', fontsize=8, style='italic', color='gray')
    ax.text(0.61, 6.5, 'Winner', fontsize=8, style='italic', color='#2ca02c', fontweight='bold')
    ax.text(0.61, 0.5, 'Best Alternative', fontsize=8, style='italic', color='#1f77b4')

    plt.tight_layout()
    plt.savefig('results/figures/final_all_methods_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: results/figures/final_all_methods_comparison.png")
    plt.close()


def create_hybrid_methods_comparison(results):
    """
    Figure 2: Focused comparison of hybrid/cluster approaches
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Bar comparison
    methods = ['Unified\nLoRA', 'Hybrid\nLoRA', 'Cluster\nLoRA', 'Weighted\nMerge']
    scores = [results['unified'], results['hybrid_lora'],
              results['cluster_lora'], results['weighted_merge']]
    colors = ['#2ca02c', '#ff7f0e', '#1f77b4', '#9467bd']

    bars = ax1.bar(methods, scores, color=colors, alpha=0.85,
                   edgecolor='black', linewidth=1.5, width=0.6)

    # Add value labels
    unified_score = results['unified']
    for i, (bar, score) in enumerate(zip(bars, scores)):
        height = bar.get_height()

        # Score on top
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{score:.1%}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Improvement inside (skip unified)
        if i > 0:
            improvement = ((score - unified_score) / unified_score) * 100
            color = 'darkgreen' if improvement > 0 else 'darkred'
            sign = '+' if improvement > 0 else ''
            ax1.text(bar.get_x() + bar.get_width()/2., score/2,
                    f'{sign}{improvement:.1f}%',
                    ha='center', va='center', fontsize=9,
                    color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3',
                             facecolor='white', alpha=0.85))

    # Winner highlight
    bars[0].set_linewidth(2.5)
    bars[0].set_edgecolor('gold')

    ax1.axhline(unified_score, color='#2ca02c', linestyle='--',
                linewidth=1.5, alpha=0.5)
    ax1.set_ylabel('Embedding Similarity', fontsize=11, fontweight='bold')
    ax1.set_title('Hybrid Approaches: Performance Comparison',
                  fontsize=11, fontweight='bold', pad=10)
    ax1.set_ylim(0.65, 0.85)
    ax1.grid(axis='y', alpha=0.3, linewidth=0.8)

    # Right: Training efficiency (performance per GPU hour)
    training_times = {
        'Unified\nLoRA': 2,
        'Hybrid\nLoRA': 52,
        'Cluster\nLoRA': 2,
        'Weighted\nMerge': 0.03
    }

    efficiency = []
    for method in methods:
        time = training_times[method]
        score = scores[methods.index(method)]
        efficiency.append(score / time if time > 0 else 0)

    bars2 = ax2.bar(methods, efficiency, color=colors, alpha=0.85,
                    edgecolor='black', linewidth=1.5, width=0.6)

    for bar, eff in zip(bars2, efficiency):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{eff:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Winner highlight
    bars2[0].set_linewidth(2.5)
    bars2[0].set_edgecolor('gold')

    ax2.set_ylabel('Performance / GPU Hour', fontsize=11, fontweight='bold')
    ax2.set_title('Training Efficiency Comparison',
                  fontsize=11, fontweight='bold', pad=10)
    ax2.grid(axis='y', alpha=0.3, linewidth=0.8)

    plt.tight_layout()
    plt.savefig('results/figures/final_hybrid_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: results/figures/final_hybrid_comparison.png")
    plt.close()


def create_three_phase_progression():
    """
    Figure 3: Three-phase experimental progression
    Clean, simple visualization of the journey
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    results = load_all_results()
    unified_score = results['unified']

    # Phase 1: Baseline attempts
    ax = axes[0]
    methods = ['Baseline', 'Unified\nLoRA', 'Per-Persona\nLoRA', 'Sparse\nMoE']
    scores = [results['baseline'], results['unified'],
              results['per_persona'], results['moe_sparse']]
    colors = ['#7f7f7f', '#2ca02c', '#d62728', '#d62728']

    bars = ax.bar(range(len(methods)), scores, color=colors,
                  alpha=0.85, edgecolor='black', linewidth=1.2)

    # Highlight winner
    bars[1].set_linewidth(2.5)
    bars[1].set_edgecolor('gold')

    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(i, score + 0.02, f'{score:.0%}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylabel('Embedding Similarity', fontsize=10, fontweight='bold')
    ax.set_title('Phase 1: Baseline Attempts\n(Unified Wins)',
                 fontsize=10, fontweight='bold', pad=10)
    ax.set_ylim(0.6, 0.9)
    ax.grid(axis='y', alpha=0.3, linewidth=0.8)
    ax.text(0.5, 0.88, 'WINNER', ha='center', fontsize=8,
            fontweight='bold', color='gold',
            bbox=dict(boxstyle='round', facecolor='white',
                     edgecolor='gold', linewidth=2))

    # Phase 2: Building on unified
    ax = axes[1]
    methods = ['Unified\nLoRA', 'Hybrid\nLoRA', 'Cluster\nLoRA']
    scores = [results['unified'], results['hybrid_lora'], results['cluster_lora']]
    colors = ['#2ca02c', '#ff7f0e', '#1f77b4']

    bars = ax.bar(range(len(methods)), scores, color=colors,
                  alpha=0.85, edgecolor='black', linewidth=1.2)

    # Highlight winner
    bars[0].set_linewidth(2.5)
    bars[0].set_edgecolor('gold')

    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(i, score + 0.01, f'{score:.0%}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

        if i > 0:
            diff = ((score - unified_score) / unified_score) * 100
            ax.text(i, score - 0.03, f'{diff:.1f}%',
                   ha='center', va='top', fontsize=8,
                   color='darkred', fontweight='bold')

    ax.axhline(unified_score, color='#2ca02c', linestyle='--',
              linewidth=1.2, alpha=0.5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylabel('Embedding Similarity', fontsize=10, fontweight='bold')
    ax.set_title('Phase 2: Build on Unified\n(Still Loses)',
                 fontsize=10, fontweight='bold', pad=10)
    ax.set_ylim(0.7, 0.86)
    ax.grid(axis='y', alpha=0.3, linewidth=0.8)

    # Phase 3: Routing
    ax = axes[2]

    # Pie chart of routing decisions
    sizes = [results['routing_breakdown']['unified'],
             results['routing_breakdown']['hybrid'] + results['routing_breakdown']['personalized']]
    labels = [f'Unified\n{sizes[0]}/200\n({sizes[0]/2:.0f}%)',
              f'Personalized\n{sizes[1]}/200\n({sizes[1]/2:.0f}%)']
    colors_pie = ['#2ca02c', '#ff7f0e']
    explode = (0.05, 0)

    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pie,
                                       autopct='', startangle=90, explode=explode,
                                       textprops={'fontsize': 8, 'fontweight': 'bold'},
                                       wedgeprops={'linewidth': 1.5, 'edgecolor': 'black'})

    improvement = ((results['routing'] - unified_score) / unified_score) * 100
    ax.set_title(f'Phase 3: Selective Routing\n({results["routing"]:.1%}, +{improvement:.1f}%)',
                fontsize=10, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig('results/figures/final_three_phases.png', dpi=300, bbox_inches='tight')
    print("Saved: results/figures/final_three_phases.png")
    plt.close()


def create_routing_details():
    """
    Figure 4: Detailed routing analysis
    """
    with open('results/selective_routing/routing_decisions.json') as f:
        data = json.load(f)
        routing_decisions = data['routing_decisions']
        routing_breakdown = data['summary']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Improvement distribution
    improvements = []
    best_models = []
    for decision in routing_decisions:
        imp = decision['improvement_over_unified'] * 100
        improvements.append(imp)
        best_models.append(decision['best_model'])

    # Separate by model type
    unified_imps = [imp for imp, model in zip(improvements, best_models) if model == 'unified']
    hybrid_imps = [imp for imp, model in zip(improvements, best_models) if model == 'hybrid']
    pers_imps = [imp for imp, model in zip(improvements, best_models) if model == 'personalized']

    bins = np.arange(-5, 13, 0.5)
    ax1.hist([unified_imps, hybrid_imps, pers_imps], bins=bins,
             label=[f'Unified ({len(unified_imps)})',
                    f'Hybrid ({len(hybrid_imps)})',
                    f'Personalized ({len(pers_imps)})'],
             color=['#2ca02c', '#ff7f0e', '#d62728'],
             alpha=0.7, edgecolor='black', linewidth=0.8, stacked=False)

    ax1.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax1.set_xlabel('Improvement over Unified (%)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Personas', fontsize=11, fontweight='bold')
    ax1.set_title('Distribution of Improvements by Model Type',
                  fontsize=11, fontweight='bold', pad=10)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(axis='y', alpha=0.3, linewidth=0.8)

    # Right: Summary statistics
    ax2.axis('off')

    # Create summary table
    summary_text = f"""
    SELECTIVE ROUTING SUMMARY
    {'='*40}

    Overall Performance:
      • Unified (all):      82.14%
      • Selective Routing:  82.99%
      • Improvement:        +0.85% (+1.03%)

    Routing Decisions (200 personas):
      • Unified:       {routing_breakdown['unified_count']:3d} ({routing_breakdown['unified_count']/2:.1f}%)
      • Hybrid:        {routing_breakdown['hybrid_count']:3d} ({routing_breakdown['hybrid_count']/2:.1f}%)
      • Personalized:  {routing_breakdown['personalized_count']:3d} ({routing_breakdown['personalized_count']/2:.1f}%)

    Key Finding:
      77.5% of personas get best results
      from simple unified model

    Conclusion:
      Personalization helps minority (<25%)
      Most users benefit from unified approach
    """

    ax2.text(0.1, 0.95, summary_text, transform=ax2.transAxes,
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig('results/figures/final_routing_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: results/figures/final_routing_analysis.png")
    plt.close()


def create_summary_figure():
    """
    Figure 5: Single comprehensive summary
    """
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    results = load_all_results()
    unified_score = results['unified']

    # Top: All methods comparison
    ax1 = fig.add_subplot(gs[0, :])

    methods = ['Baseline', 'Unified\nLoRA', 'Per-Persona', 'MoE',
               'Cluster', 'Hybrid', 'Weighted', 'Routing']
    scores = [results['baseline'], results['unified'], results['per_persona'],
              results['moe_sparse'], results['cluster_lora'], results['hybrid_lora'],
              results['weighted_merge'], results['routing']]
    colors = ['#7f7f7f', '#2ca02c', '#d62728', '#d62728',
              '#d62728', '#ff7f0e', '#d62728', '#1f77b4']

    bars = ax1.bar(methods, scores, color=colors, alpha=0.85,
                   edgecolor='black', linewidth=1.2)

    bars[1].set_linewidth(2.5)
    bars[1].set_edgecolor('gold')

    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax1.text(i, score + 0.01, f'{score:.0%}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.axhline(unified_score, color='#2ca02c', linestyle='--',
               linewidth=1.5, alpha=0.5)
    ax1.set_ylabel('Embedding Similarity', fontsize=11, fontweight='bold')
    ax1.set_title('Complete Method Comparison (7 Approaches Tested)',
                  fontsize=12, fontweight='bold', pad=15)
    ax1.set_ylim(0.6, 0.88)
    ax1.grid(axis='y', alpha=0.3, linewidth=0.8)

    # Middle left: Phase progression
    ax2 = fig.add_subplot(gs[1, 0])

    phases = ['Phase 1:\nBaseline', 'Phase 2:\nHybrid', 'Phase 3:\nRouting']
    phase_winners = [unified_score, max(results['hybrid_lora'], results['cluster_lora']),
                     results['routing']]
    phase_colors = ['#2ca02c', '#ff7f0e', '#1f77b4']

    bars = ax2.bar(phases, phase_winners, color=phase_colors,
                   alpha=0.85, edgecolor='black', linewidth=1.5)

    for bar, score in zip(bars, phase_winners):
        ax2.text(bar.get_x() + bar.get_width()/2, score + 0.01,
                f'{score:.1%}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    ax2.axhline(unified_score, color='#2ca02c', linestyle='--',
               linewidth=1.5, alpha=0.5, label='Unified Baseline')
    ax2.set_ylabel('Best Score per Phase', fontsize=10, fontweight='bold')
    ax2.set_title('Three-Phase Progression', fontsize=10, fontweight='bold')
    ax2.set_ylim(0.72, 0.85)
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3, linewidth=0.8)

    # Middle right: Routing breakdown
    ax3 = fig.add_subplot(gs[1, 1])

    sizes = [results['routing_breakdown']['unified'],
             results['routing_breakdown']['hybrid'],
             results['routing_breakdown']['personalized']]
    labels = ['Unified\n(77.5%)', 'Hybrid\n(20.5%)', 'Personalized\n(2.0%)']
    colors_pie = ['#2ca02c', '#ff7f0e', '#d62728']

    wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors_pie,
                                        autopct='%d', startangle=90,
                                        textprops={'fontsize': 9, 'fontweight': 'bold'},
                                        wedgeprops={'linewidth': 1.5, 'edgecolor': 'black'},
                                        explode=(0.05, 0, 0))

    ax3.set_title('Routing Decisions\n(200 Personas)',
                  fontsize=10, fontweight='bold')

    # Bottom: Key insights
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    insights_text = """
    KEY FINDINGS
    ═══════════════════════════════════════════════════════════════════════════════════════

    1. UNIFIED LORA WINS: 82.14% embedding similarity beats all personalization attempts
       • Better than per-persona (68.28%, -13.9%)
       • Better than clustering (74.14%, -8.0%)
       • Better than hybrid (75.91%, -7.6%)
       • Better than weighted merge (67.00%, -18.4%)

    2. DATA QUANTITY > ALGORITHMIC SOPHISTICATION
       • 6,000 unified examples beats 20-2,160 personalized examples
       • More data trumps clever personalization for small models (0.5B params)

    3. PERSONALIZATION HELPS ONLY 22.5% OF PERSONAS
       • 77.5% prefer unified model (domain knowledge matters more)
       • Selective routing: +1.03% improvement at high complexity cost

    4. RECOMMENDATION: Use Unified LoRA
       • Best single model (82.14%)
       • Simplest deployment (one model)
       • Fastest training (2 hours vs 2-52 hours)
       • Most consistent (0% variance)
    """

    ax4.text(0.05, 0.95, insights_text, transform=ax4.transAxes,
             fontsize=8.5, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8,
                      edgecolor='black', linewidth=1.5))

    plt.suptitle('Smart Home Personalization: Complete Experimental Results',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.savefig('results/figures/final_complete_summary.png', dpi=300, bbox_inches='tight')
    print("Saved: results/figures/final_complete_summary.png")
    plt.close()


def main():
    """Generate all publication-quality figures"""

    print("="*80)
    print("CREATING PUBLICATION-QUALITY FIGURES FOR FINAL REPORT")
    print("="*80)
    print()

    print("Loading all results...")
    results = load_all_results()

    print("\nGenerating figures...")

    print("\n1. Complete Methods Comparison...")
    create_complete_methods_comparison(results)

    print("\n2. Hybrid Methods Comparison...")
    create_hybrid_methods_comparison(results)

    print("\n3. Three-Phase Progression...")
    create_three_phase_progression()

    print("\n4. Routing Analysis...")
    create_routing_details()

    print("\n5. Complete Summary Figure...")
    create_summary_figure()

    print("\n" + "="*80)
    print("ALL FIGURES GENERATED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated files (300 DPI, publication quality):")
    print("  1. final_all_methods_comparison.png  - All 7 methods compared")
    print("  2. final_hybrid_comparison.png        - Hybrid methods deep dive")
    print("  3. final_three_phases.png             - Three-phase progression")
    print("  4. final_routing_analysis.png         - Routing details")
    print("  5. final_complete_summary.png         - Complete summary")
    print()


if __name__ == '__main__':
    main()
