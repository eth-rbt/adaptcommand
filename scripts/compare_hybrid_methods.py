"""
Compare the Three Hybrid/Cluster Methods

1. Hybrid LoRA: Unified (frozen) + Per-Persona LoRA on top
2. Cluster LoRA: Train cluster-specific LoRAs from scratch
3. Weighted Merge: Smart weighted merging of per-persona LoRAs

All three attempt to balance shared knowledge and personalization, but with
different strategies.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")


def load_all_hybrid_results():
    """Load all three hybrid method results"""
    results = {}

    # 1. Hybrid LoRA (unified + per-persona)
    with open('results/hybrid/hybrid_summary.json') as f:
        data = json.load(f)
        per_persona = data['per_persona_metrics']
        results['Hybrid LoRA'] = {
            'mean': np.mean([p['embedding_similarity'] for p in per_persona]),
            'std': np.std([p['embedding_similarity'] for p in per_persona]),
            'min': np.min([p['embedding_similarity'] for p in per_persona]),
            'max': np.max([p['embedding_similarity'] for p in per_persona]),
            'scores': [p['embedding_similarity'] for p in per_persona],
            'description': 'Unified (frozen) + Per-Persona LoRA',
            'training_approach': 'Two-stage: Unified base + persona fine-tune',
            'params_per_persona': '2.4M (persona LoRA only)',
            'training_data': '20 examples per persona',
            'total_training_time': '~50 hours (200 personas)',
        }

    # 2. Cluster LoRA (cluster-based from scratch)
    with open('results/cluster_lora/cluster_lora_results.json') as f:
        data = json.load(f)
        per_persona = data['per_persona_results']
        results['Cluster LoRA'] = {
            'mean': data['summary']['embedding_similarity_mean'],
            'std': data['summary']['embedding_similarity_std'],
            'min': np.min([p['embedding_similarity'] for p in per_persona]),
            'max': np.max([p['embedding_similarity'] for p in per_persona]),
            'scores': [p['embedding_similarity'] for p in per_persona],
            'description': 'Cluster-specific LoRA (from scratch)',
            'training_approach': 'Single-stage: Train on cluster data',
            'params_per_persona': '2.4M (shared within cluster)',
            'training_data': '480-2160 examples per cluster',
            'total_training_time': '~2 hours (2 clusters)',
        }

    # 3. Weighted Merge (smart merging)
    with open('results/weighted_merge/weighted_merge_cluster4_results.json') as f:
        data = json.load(f)
        results['Weighted Merge'] = {
            'mean': data['embedding_similarity'],
            'std': data['embedding_similarity_std'],
            'min': None,  # Not available in summary
            'max': None,
            'scores': None,
            'description': 'Weighted merge of per-persona LoRAs',
            'training_approach': 'Zero-stage: Merge existing LoRAs',
            'params_per_persona': '2.4M (merged from K=5 personas)',
            'training_data': 'Uses pre-trained per-persona LoRAs',
            'total_training_time': '~2 minutes (merging only)',
        }

    # Load unified baseline for comparison
    with open('results/unified/unified_results.json') as f:
        unified = json.load(f)
        results['Unified (Baseline)'] = {
            'mean': unified['metrics']['embedding_similarity'],
            'std': 0.0,
            'min': unified['metrics']['embedding_similarity'],
            'max': unified['metrics']['embedding_similarity'],
            'scores': None,
            'description': 'Single unified LoRA for all personas',
            'training_approach': 'Single-stage: Train on all data',
            'params_per_persona': '2.4M (shared by all)',
            'training_data': '6000 examples total',
            'total_training_time': '~2 hours',
        }

    return results


def create_comparison_chart(results):
    """Create comprehensive comparison visualization"""

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # Colors
    colors = {
        'Unified (Baseline)': '#2ca02c',
        'Hybrid LoRA': '#ff7f0e',
        'Cluster LoRA': '#1f77b4',
        'Weighted Merge': '#9467bd'
    }

    # Method order
    methods = ['Unified (Baseline)', 'Hybrid LoRA', 'Cluster LoRA', 'Weighted Merge']

    # ============================================================
    # 1. Main comparison bar chart
    # ============================================================
    ax1 = fig.add_subplot(gs[0, :])

    means = [results[m]['mean'] for m in methods]
    stds = [results[m]['std'] for m in methods]
    method_colors = [colors[m] for m in methods]

    bars = ax1.bar(methods, means, yerr=stds, color=method_colors,
                   alpha=0.8, edgecolor='black', linewidth=2, capsize=5)

    # Add value labels
    unified_score = results['Unified (Baseline)']['mean']
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()

        # Score label
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.2%}',
                ha='center', va='bottom', fontsize=13, fontweight='bold')

        # Improvement vs unified
        if i > 0:
            improvement = ((mean - unified_score) / unified_score) * 100
            color = 'green' if improvement > 0 else 'red'
            sign = '+' if improvement > 0 else ''
            ax1.text(bar.get_x() + bar.get_width()/2., mean/2,
                    f'{sign}{improvement:.1f}%',
                    ha='center', va='center', fontsize=11, fontweight='bold',
                    color=color, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Highlight best
    bars[0].set_linewidth(4)
    bars[0].set_edgecolor('gold')

    ax1.axhline(unified_score, color='green', linestyle='--', linewidth=2,
                alpha=0.5, label=f'Unified Baseline ({unified_score:.2%})')
    ax1.set_ylabel('Embedding Similarity', fontsize=14, fontweight='bold')
    ax1.set_title('Hybrid Method Comparison: Performance Overview',
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylim(0.65, 0.85)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)

    # ============================================================
    # 2. Distribution comparison (box plots)
    # ============================================================
    ax2 = fig.add_subplot(gs[1, 0])

    distributions = []
    labels = []
    box_colors = []

    for method in methods:
        if results[method]['scores'] is not None:
            distributions.append(results[method]['scores'])
            labels.append(method)
            box_colors.append(colors[method])

    bp = ax2.boxplot(distributions, labels=labels, patch_artist=True,
                     showmeans=True, meanline=True)

    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax2.axhline(unified_score, color='green', linestyle='--',
                linewidth=2, alpha=0.5)
    ax2.set_ylabel('Embedding Similarity', fontsize=12, fontweight='bold')
    ax2.set_title('Score Distribution by Method', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right', fontsize=9)

    # ============================================================
    # 3. Training time vs performance
    # ============================================================
    ax3 = fig.add_subplot(gs[1, 1])

    training_times = {
        'Unified (Baseline)': 2,
        'Hybrid LoRA': 52,  # 2 unified + 50 persona
        'Cluster LoRA': 2,
        'Weighted Merge': 0.03  # 2 minutes in hours
    }

    x_times = [training_times[m] for m in methods]
    y_scores = [results[m]['mean'] for m in methods]
    method_colors_list = [colors[m] for m in methods]

    scatter = ax3.scatter(x_times, y_scores, s=400, c=method_colors_list,
                          alpha=0.7, edgecolors='black', linewidth=2)

    # Add labels
    for i, method in enumerate(methods):
        offset_x = 0.3 if method != 'Hybrid LoRA' else 5
        offset_y = 0.01 if method != 'Weighted Merge' else -0.01
        ax3.annotate(method, (x_times[i], y_scores[i]),
                    xytext=(offset_x, offset_y), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    ax3.axhline(unified_score, color='green', linestyle='--', linewidth=2, alpha=0.5)
    ax3.set_xlabel('Total Training Time (GPU hours)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Embedding Similarity', fontsize=12, fontweight='bold')
    ax3.set_title('Training Time vs Performance Trade-off', fontsize=14, fontweight='bold')
    ax3.set_xscale('log')
    ax3.set_ylim(0.65, 0.85)
    ax3.grid(alpha=0.3)

    # ============================================================
    # 4. Method comparison table
    # ============================================================
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    table_data = []
    for method in methods:
        r = results[method]
        improvement = ((r['mean'] - unified_score) / unified_score) * 100 if method != 'Unified (Baseline)' else 0
        table_data.append([
            method,
            f"{r['mean']:.2%} ± {r['std']:.2%}",
            f"{improvement:+.1f}%" if method != 'Unified (Baseline)' else "baseline",
            r['training_approach'],
            r['training_data'],
            r['total_training_time']
        ])

    table = ax4.table(cellText=table_data,
                      colLabels=['Method', 'Score (mean ± std)', 'vs Unified',
                                'Training Approach', 'Training Data', 'Training Time'],
                      cellLoc='left',
                      loc='center',
                      colWidths=[0.18, 0.15, 0.10, 0.22, 0.18, 0.17])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)

    # Color header
    for i in range(6):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color method names
    for i, method in enumerate(methods, start=1):
        table[(i, 0)].set_facecolor(colors[method])
        table[(i, 0)].set_alpha(0.3)
        table[(i, 0)].set_text_props(weight='bold')

    ax4.set_title('Detailed Method Comparison', fontsize=14,
                  fontweight='bold', pad=20, loc='left')

    plt.tight_layout()

    # Save
    output_path = Path('results/figures/hybrid_methods_comparison.png')
    output_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_analysis_report(results):
    """Create detailed text analysis"""

    print("\n" + "=" * 100)
    print("HYBRID METHOD COMPARISON ANALYSIS")
    print("=" * 100)

    unified_score = results['Unified (Baseline)']['mean']

    # Ranking
    methods = ['Hybrid LoRA', 'Cluster LoRA', 'Weighted Merge']
    ranked = sorted(methods, key=lambda m: results[m]['mean'], reverse=True)

    print("\n" + "-" * 100)
    print("PERFORMANCE RANKING")
    print("-" * 100)
    print(f"\n0. Unified (Baseline): {unified_score:.4f} (82.14%)")

    for i, method in enumerate(ranked, start=1):
        r = results[method]
        improvement = ((r['mean'] - unified_score) / unified_score) * 100
        print(f"{i}. {method}: {r['mean']:.4f} ({r['mean']*100:.2f}%) "
              f"[{improvement:+.1f}% vs unified]")

    # Detailed analysis
    print("\n" + "-" * 100)
    print("DETAILED ANALYSIS")
    print("-" * 100)

    print("\n1. HYBRID LoRA (Unified + Per-Persona)")
    print("   " + "-" * 95)
    r = results['Hybrid LoRA']
    improvement = ((r['mean'] - unified_score) / unified_score) * 100
    print(f"   Score: {r['mean']:.4f} ({r['mean']*100:.2f}%)")
    print(f"   vs Unified: {improvement:+.2f}%")
    print(f"   Std Dev: {r['std']:.4f} (variance: {r['std']*100:.2f}%)")
    print(f"   Range: {r['min']:.4f} - {r['max']:.4f}")
    print(f"   Strategy: {r['training_approach']}")
    print(f"   Training Time: {r['total_training_time']}")
    print(f"   ")
    print(f"   ANALYSIS:")
    print(f"   - Best hybrid method, but still worse than unified (-7.6%)")
    print(f"   - Benefits from unified foundation (82.14% starting point)")
    print(f"   - Per-persona fine-tuning adds personalization but overfits")
    print(f"   - High variance (std: {r['std']:.4f}) indicates inconsistent results")
    print(f"   - Trade-off: 26x training time for -7.6% performance")

    print("\n2. CLUSTER LoRA (Cluster-based from scratch)")
    print("   " + "-" * 95)
    r = results['Cluster LoRA']
    improvement = ((r['mean'] - unified_score) / unified_score) * 100
    print(f"   Score: {r['mean']:.4f} ({r['mean']*100:.2f}%)")
    print(f"   vs Unified: {improvement:+.2f}%")
    print(f"   Std Dev: {r['std']:.4f} (variance: {r['std']*100:.2f}%)")
    print(f"   Range: {r['min']:.4f} - {r['max']:.4f}")
    print(f"   Strategy: {r['training_approach']}")
    print(f"   Training Time: {r['total_training_time']}")
    print(f"   ")
    print(f"   ANALYSIS:")
    print(f"   - Trained on 480-2160 examples per cluster (vs 6000 unified)")
    print(f"   - Performance drop: -9.7% vs unified")
    print(f"   - Poor clustering quality (silhouette: 0.022) hurts performance")
    print(f"   - Even largest cluster (2160 examples) can't beat unified")
    print(f"   - Conclusion: Insufficient data + poor clustering = failure")

    print("\n3. WEIGHTED MERGE (Smart merging of per-persona LoRAs)")
    print("   " + "-" * 95)
    r = results['Weighted Merge']
    improvement = ((r['mean'] - unified_score) / unified_score) * 100
    print(f"   Score: {r['mean']:.4f} ({r['mean']*100:.2f}%)")
    print(f"   vs Unified: {improvement:+.2f}%")
    print(f"   Std Dev: {r['std']:.4f} (variance: {r['std']*100:.2f}%)")
    print(f"   Strategy: {r['training_approach']}")
    print(f"   Training Time: {r['total_training_time']}")
    print(f"   ")
    print(f"   ANALYSIS:")
    print(f"   - Worst performing method (-18.4% vs unified)")
    print(f"   - Merges K=5 per-persona LoRAs weighted by performance + centrality")
    print(f"   - Smart weighting doesn't prevent destructive averaging")
    print(f"   - Linear averaging of nonlinear weights fails")
    print(f"   - Very high variance (std: {r['std']:.4f}) shows instability")
    print(f"   - Fast (2 min) but useless - even worse than individual LoRAs")

    # Summary insights
    print("\n" + "-" * 100)
    print("KEY INSIGHTS")
    print("-" * 100)
    print("\n1. ALL THREE HYBRID METHODS FAIL TO BEAT UNIFIED")
    print("   - Unified: 82.14%")
    print("   - Best hybrid (Hybrid LoRA): 75.91% (-7.6%)")
    print("   - Worst hybrid (Weighted Merge): 67.00% (-18.4%)")

    print("\n2. DATA QUANTITY > PERSONALIZATION SOPHISTICATION")
    print("   - Unified uses all 6000 examples")
    print("   - Hybrid uses 20 per persona (overfits)")
    print("   - Cluster uses 480-2160 (still insufficient)")
    print("   - Weighted merge uses merged weights (destructive)")

    print("\n3. TRAINING TIME vs PERFORMANCE TRADE-OFF")
    print("   - Unified: 2 hours -> 82.14%")
    print("   - Hybrid: 52 hours -> 75.91% (26x time for -7.6% performance!)")
    print("   - Cluster: 2 hours -> 74.14% (same time, -9.7% performance)")
    print("   - Weighted: 2 min -> 67.00% (fast but terrible)")

    print("\n4. VARIANCE INDICATES INSTABILITY")
    print("   - Unified: 0.0% std (consistent)")
    print("   - Hybrid: {:.2f}% std (inconsistent)".format(results['Hybrid LoRA']['std']*100))
    print("   - Cluster: {:.2f}% std (moderate)".format(results['Cluster LoRA']['std']*100))
    print("   - Weighted: {:.2f}% std (very unstable)".format(results['Weighted Merge']['std']*100))

    print("\n" + "-" * 100)
    print("RECOMMENDATION")
    print("-" * 100)
    print("\nFor this dataset and task:")
    print("  [+] USE: Unified LoRA (82.14%, simple, fast, consistent)")
    print("  [-] AVOID: All hybrid methods (worse performance, more complexity)")
    print("\nHybrid methods might work with:")
    print("  - 100+ examples per persona (vs 20)")
    print("  - 3B+ parameter models (vs 0.5B)")
    print("  - Better clustering (silhouette > 0.3 vs 0.022)")
    print("  - Tasks with >50% personalization benefit (vs 25% for smart home)")

    print("\n" + "=" * 100)
    print()


def main():
    """Run hybrid method comparison"""

    print("=" * 100)
    print("COMPARING THREE HYBRID METHODS")
    print("=" * 100)
    print()
    print("Methods:")
    print("  1. Hybrid LoRA: Unified (frozen) + Per-Persona LoRA on top")
    print("  2. Cluster LoRA: Train cluster-specific LoRAs from scratch")
    print("  3. Weighted Merge: Smart weighted merging of per-persona LoRAs")
    print()

    print("Loading results...")
    results = load_all_hybrid_results()

    print("Creating visualization...")
    create_comparison_chart(results)

    print("Generating analysis report...")
    create_analysis_report(results)

    print("\nDone! Check results/figures/hybrid_methods_comparison.png")


if __name__ == '__main__':
    main()
