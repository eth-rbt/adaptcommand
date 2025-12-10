"""
Generate graphs for final report restructuring

Creates three key visualizations:
1. Baseline Adaptation Comparison (Unified vs Baseline vs Single LoRA vs MoE)
2. Building on Unified (Single LoRA vs Cluster LoRA)
3. Final Routing Comparison (Percentages of which method works best for which persona)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")

def load_results():
    """Load all necessary results"""
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

    # Selective Routing
    with open('results/selective_routing/routing_decisions.json') as f:
        data = json.load(f)
        results['routing'] = data['summary']['avg_selective_score']
        results['routing_breakdown'] = {
            'unified': data['summary']['unified_count'],
            'personalized': data['summary']['personalized_count'],
            'hybrid': data['summary']['hybrid_count']
        }
        results['routing_decisions'] = data['routing_decisions']

    return results


def create_baseline_comparison(results):
    """
    Graph 1: Baseline Adaptation Attempts
    Compares: Baseline, Unified, Per-Persona, MoE -> Unified wins
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    methods = ['Baseline\n(No Adaptation)', 'Unified LoRA\n(6000 examples)',
               'Per-Persona LoRA\n(20 examples each)', 'Sparse MoE\n(Merged)']
    scores = [results['baseline'], results['unified'],
              results['per_persona'], results['moe_sparse']]
    colors = ['#808080', '#2ca02c', '#d62728', '#9467bd']

    bars = ax.bar(methods, scores, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=2)

    # Add value labels
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.2%}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Add improvement labels
    baseline_score = results['unified']
    for i, (bar, score) in enumerate(zip(bars, scores)):
        if i > 0:  # Skip baseline
            improvement = ((score - baseline_score) / baseline_score) * 100
            color = 'green' if improvement > 0 else 'red'
            sign = '+' if improvement > 0 else ''
            ax.text(bar.get_x() + bar.get_width()/2., score/2,
                   f'{sign}{improvement:.1f}%',
                   ha='center', va='center', fontsize=12, fontweight='bold',
                   color=color, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Highlight winner
    ax.patches[1].set_linewidth(4)
    ax.patches[1].set_edgecolor('gold')

    ax.set_ylabel('Embedding Similarity', fontsize=14, fontweight='bold')
    ax.set_title('Phase 1: Baseline Adaptation Attempts\n(Unified LoRA Wins)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0.6, 0.9)
    ax.grid(axis='y', alpha=0.3)

    # Add annotation
    ax.annotate('WINNER', xy=(1, results['unified']), xytext=(1.5, 0.85),
                arrowprops=dict(arrowstyle='->', lw=3, color='gold'),
                fontsize=14, fontweight='bold', color='gold')

    plt.tight_layout()
    Path('results/figures').mkdir(exist_ok=True, parents=True)
    plt.savefig('results/figures/1_baseline_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: results/figures/1_baseline_comparison.png")
    plt.close()


def create_unified_plus_adaptation(results):
    """
    Graph 2: Building on Unified
    Compares: Unified + Per-Persona adaptation vs Unified + Cluster adaptation
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Bar comparison
    methods = ['Unified LoRA\n(Baseline)', 'Unified + Cluster LoRA\n(5 clusters)']
    scores = [results['unified'], results['cluster_lora']]
    colors = ['#2ca02c', '#ff7f0e']

    bars = ax1.bar(methods, scores, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=2)

    # Add value labels
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{score:.2%}',
                 ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Add improvement
    improvement = ((results['cluster_lora'] - results['unified']) / results['unified']) * 100
    color = 'green' if improvement > 0 else 'red'
    sign = '+' if improvement > 0 else ''
    ax1.text(1, scores[1]/2,
             f'{sign}{improvement:.1f}%',
             ha='center', va='center', fontsize=12, fontweight='bold',
             color=color, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax1.set_ylabel('Embedding Similarity', fontsize=14, fontweight='bold')
    ax1.set_title('Phase 2: Adapting on Top of Unified',
                  fontsize=15, fontweight='bold', pad=15)
    ax1.set_ylim(0.7, 0.85)
    ax1.grid(axis='y', alpha=0.3)

    # Right: Breakdown by cluster
    cluster_data = {}
    with open('results/cluster_lora/cluster_lora_results.json') as f:
        data = json.load(f)
        # Aggregate by cluster
        for persona_result in data['per_persona_results']:
            cluster_id = persona_result['cluster_id']
            if cluster_id not in cluster_data:
                cluster_data[cluster_id] = {
                    'scores': [],
                    'count': 0
                }
            cluster_data[cluster_id]['scores'].append(persona_result['embedding_similarity'])
            cluster_data[cluster_id]['count'] += 1

        # Calculate averages
        cluster_list = []
        for cluster_id, data_dict in sorted(cluster_data.items()):
            cluster_list.append({
                'cluster': f"Cluster {cluster_id}",
                'score': np.mean(data_dict['scores']),
                'count': data_dict['count']
            })
        cluster_data = cluster_list

    cluster_names = [c['cluster'] for c in cluster_data]
    cluster_scores = [c['score'] for c in cluster_data]
    cluster_counts = [c['count'] for c in cluster_data]

    bars2 = ax2.barh(cluster_names, cluster_scores, color='#ff7f0e',
                     alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels and persona counts
    for i, (bar, score, count) in enumerate(zip(bars2, cluster_scores, cluster_counts)):
        width = bar.get_width()
        ax2.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                 f'{score:.2%} ({count} personas)',
                 ha='left', va='center', fontsize=11, fontweight='bold')

    # Add unified baseline line
    ax2.axvline(results['unified'], color='green', linestyle='--',
                linewidth=2, alpha=0.7, label=f'Unified Baseline ({results["unified"]:.2%})')

    ax2.set_xlabel('Embedding Similarity', fontsize=12, fontweight='bold')
    ax2.set_title('Cluster LoRA Performance by Cluster',
                  fontsize=15, fontweight='bold', pad=15)
    ax2.set_xlim(0.7, 0.8)
    ax2.legend(fontsize=10)
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/2_unified_plus_adaptation.png', dpi=150, bbox_inches='tight')
    print("Saved: results/figures/2_unified_plus_adaptation.png")
    plt.close()


def create_routing_comparison(results):
    """
    Graph 3: Final Routing Comparison
    Shows percentages of which method works best for which persona
    """
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Extract routing data
    routing_breakdown = results['routing_breakdown']
    total_personas = 200

    # Top left: Pie chart of routing decisions
    ax1 = fig.add_subplot(gs[0, 0])
    sizes = [routing_breakdown['unified'], routing_breakdown['hybrid'],
             routing_breakdown['personalized']]
    labels = [f'Unified\n({routing_breakdown["unified"]}/200 = {routing_breakdown["unified"]/2:.1f}%)',
              f'Hybrid\n({routing_breakdown["hybrid"]}/200 = {routing_breakdown["hybrid"]/2:.1f}%)',
              f'Personalized\n({routing_breakdown["personalized"]}/200 = {routing_breakdown["personalized"]/2:.1f}%)']
    colors = ['#2ca02c', '#ff7f0e', '#d62728']
    explode = (0.1, 0, 0)  # Explode unified (winner)

    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='',
                                        explode=explode, shadow=True, startangle=90,
                                        textprops={'fontsize': 12, 'fontweight': 'bold'})

    ax1.set_title('Routing Decisions: Which Method Works Best?',
                  fontsize=14, fontweight='bold', pad=20)

    # Top right: Bar chart comparison
    ax2 = fig.add_subplot(gs[0, 1])
    methods = ['Unified LoRA\n(All personas)', 'Selective Routing\n(Best per persona)']
    scores = [results['unified'], results['routing']]
    colors_bar = ['#2ca02c', '#1f77b4']

    bars = ax2.bar(methods, scores, color=colors_bar, alpha=0.8,
                   edgecolor='black', linewidth=2)

    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{score:.2%}',
                 ha='center', va='bottom', fontsize=14, fontweight='bold')

    improvement = ((results['routing'] - results['unified']) / results['unified']) * 100
    ax2.text(1, scores[1]/2,
             f'+{improvement:.2f}%',
             ha='center', va='center', fontsize=12, fontweight='bold',
             color='green', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax2.set_ylabel('Embedding Similarity', fontsize=13, fontweight='bold')
    ax2.set_title('Selective Routing Performance', fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylim(0.81, 0.84)
    ax2.grid(axis='y', alpha=0.3)

    # Bottom: Distribution of improvements
    ax3 = fig.add_subplot(gs[1, :])

    # Calculate improvement for each persona
    improvements = []
    best_models = []
    for decision in results['routing_decisions']:
        imp = decision['improvement_over_unified']
        improvements.append(imp * 100)
        best_models.append(decision['best_model'])

    # Create bins
    bins = np.arange(-5, 15, 1)
    hist_data = {'unified': [], 'hybrid': [], 'personalized': []}

    for imp, model in zip(improvements, best_models):
        hist_data[model].append(imp)

    # Stack histograms
    ax3.hist([hist_data['unified'], hist_data['hybrid'], hist_data['personalized']],
             bins=bins, label=['Unified (155)', 'Hybrid (41)', 'Personalized (4)'],
             color=['#2ca02c', '#ff7f0e', '#d62728'], alpha=0.7,
             edgecolor='black', linewidth=1, stacked=False)

    ax3.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.5,
                label='Unified Baseline')
    ax3.set_xlabel('Improvement over Unified (%)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Number of Personas', fontsize=13, fontweight='bold')
    ax3.set_title('Distribution of Improvements: Which Personas Benefit from Personalization?',
                  fontsize=14, fontweight='bold', pad=15)
    ax3.legend(fontsize=11)
    ax3.grid(axis='y', alpha=0.3)

    # Add statistics text
    stats_text = f"""
    Summary Statistics:
    • 77.5% of personas: Unified model is best
    • 20.5% of personas: Hybrid model is best
    • 2.0% of personas: Personalized model is best

    Overall Improvement: +{improvement:.2f}% (+{improvement/100*results['unified']:.4f} absolute)
    """
    ax3.text(0.98, 0.97, stats_text, transform=ax3.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('results/figures/3_routing_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: results/figures/3_routing_comparison.png")
    plt.close()


def create_summary_figure():
    """Create a single summary figure showing all three phases"""
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    results = load_results()

    # Phase 1: Baseline Comparison
    ax = axes[0]
    methods = ['Baseline', 'Unified', 'Per-Persona', 'MoE']
    scores = [results['baseline'], results['unified'],
              results['per_persona'], results['moe_sparse']]
    colors = ['#808080', '#2ca02c', '#d62728', '#9467bd']

    bars = ax.bar(methods, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2., score + 0.01,
                f'{score:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.patches[1].set_linewidth(4)
    ax.patches[1].set_edgecolor('gold')
    ax.set_ylabel('Embedding Similarity', fontsize=12, fontweight='bold')
    ax.set_title('Phase 1: Baseline Attempts\n(Unified Wins)', fontsize=13, fontweight='bold')
    ax.set_ylim(0.6, 0.9)
    ax.grid(axis='y', alpha=0.3)

    # Phase 2: Building on Unified
    ax = axes[1]
    methods = ['Unified', 'Cluster\nLoRA']
    scores = [results['unified'], results['cluster_lora']]
    colors = ['#2ca02c', '#ff7f0e']

    bars = ax.bar(methods, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2., score + 0.005,
                f'{score:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    improvement = ((results['cluster_lora'] - results['unified']) / results['unified']) * 100
    ax.text(1, scores[1]/2, f'{improvement:.1f}%', ha='center', va='center',
            fontsize=11, fontweight='bold', color='red',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_ylabel('Embedding Similarity', fontsize=12, fontweight='bold')
    ax.set_title('Phase 2: Adapt on Unified\n(Still Loses)', fontsize=13, fontweight='bold')
    ax.set_ylim(0.7, 0.85)
    ax.grid(axis='y', alpha=0.3)

    # Phase 3: Routing
    ax = axes[2]

    # Pie chart
    sizes = [results['routing_breakdown']['unified'],
             results['routing_breakdown']['hybrid'] + results['routing_breakdown']['personalized']]
    labels = [f"Unified\n{sizes[0]}/200\n({sizes[0]/2:.0f}%)",
              f"Personalized\n{sizes[1]}/200\n({sizes[1]/2:.0f}%)"]
    colors_pie = ['#2ca02c', '#ff7f0e']

    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pie,
                                       autopct='', startangle=90, explode=(0.1, 0),
                                       textprops={'fontsize': 10, 'fontweight': 'bold'})

    ax.set_title(f'Phase 3: Selective Routing\n({results["routing"]:.1%}, +{((results["routing"]-results["unified"])/results["unified"]*100):.1f}%)',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/figures/0_summary_all_phases.png', dpi=150, bbox_inches='tight')
    print("Saved: results/figures/0_summary_all_phases.png")
    plt.close()


def main():
    """Generate all report graphs"""
    print("=" * 80)
    print("Generating Final Report Graphs")
    print("=" * 80)

    print("\nLoading results...")
    results = load_results()

    print("\nGenerating graphs...")
    print("\n1. Baseline Adaptation Comparison...")
    create_baseline_comparison(results)

    print("\n2. Building on Unified...")
    create_unified_plus_adaptation(results)

    print("\n3. Routing Comparison...")
    create_routing_comparison(results)

    print("\n4. Summary Figure (All Phases)...")
    create_summary_figure()

    print("\n" + "=" * 80)
    print("All graphs generated successfully!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - results/figures/0_summary_all_phases.png")
    print("  - results/figures/1_baseline_comparison.png")
    print("  - results/figures/2_unified_plus_adaptation.png")
    print("  - results/figures/3_routing_comparison.png")
    print("\n")


if __name__ == '__main__':
    main()
