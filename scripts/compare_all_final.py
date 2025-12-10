"""
Comprehensive Comparison of All Personalization Methods

Compares:
1. Baseline (no LoRA)
2. Unified LoRA
3. Per-Persona LoRA
4. Hybrid LoRA
5. Prefix Per-User
6. Selective Routing
7. Simple Merged LoRA
8. Cluster Merged LoRA
9. Cluster LoRA (trained from scratch)
10. Sparse MoE (Mixture of Experts merging)

Creates comprehensive visualizations for the final report.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


def load_all_results():
    """Load results from all methods"""
    results = {}

    # 1. Baseline
    try:
        with open('results/baseline/baseline_results.json') as f:
            data = json.load(f)
            results['Baseline'] = {
                'embedding_similarity': data['metrics']['embedding_similarity'],
                'device_precision': data['metrics']['device_precision'],
                'param_f1': data['metrics']['param_f1'],
                'numerical_precision': data['metrics']['numerical_precision'],
            }
    except:
        print("Warning: Baseline results not found")

    # 2. Unified LoRA
    try:
        with open('results/unified/unified_results.json') as f:
            data = json.load(f)
            results['Unified LoRA'] = {
                'embedding_similarity': data['metrics']['embedding_similarity'],
                'device_precision': data['metrics']['device_precision'],
                'param_f1': data['metrics']['param_f1'],
                'numerical_precision': data['metrics']['numerical_precision'],
            }
    except:
        print("Warning: Unified LoRA results not found")

    # 3. Per-Persona LoRA
    try:
        with open('results/personalized/personalized_summary.json') as f:
            data = json.load(f)
            per_persona_metrics = data['per_persona_metrics']
            results['Per-Persona LoRA'] = {
                'embedding_similarity': np.mean([p['embedding_similarity'] for p in per_persona_metrics]),
                'device_precision': np.mean([p['device_precision'] for p in per_persona_metrics]),
                'param_f1': np.mean([p['param_f1'] for p in per_persona_metrics]),
                'numerical_precision': np.mean([p['numerical_precision'] for p in per_persona_metrics]),
            }
    except:
        print("Warning: Per-Persona LoRA results not found")

    # 4. Hybrid LoRA
    try:
        with open('results/hybrid/hybrid_summary.json') as f:
            data = json.load(f)
            hybrid_metrics = data['per_persona_metrics']
            results['Hybrid LoRA'] = {
                'embedding_similarity': np.mean([p['embedding_similarity'] for p in hybrid_metrics]),
                'device_precision': np.mean([p['device_precision'] for p in hybrid_metrics]),
                'param_f1': np.mean([p['param_f1'] for p in hybrid_metrics]),
                'numerical_precision': np.mean([p['numerical_precision'] for p in hybrid_metrics]),
            }
    except:
        print("Warning: Hybrid LoRA results not found")

    # 5. Prefix Per-User
    try:
        with open('results/prefix_per_user/prefix_per_user_summary.json') as f:
            data = json.load(f)
            prefix_metrics = data['per_persona_metrics']
            results['Prefix Per-User'] = {
                'embedding_similarity': np.mean([p['embedding_similarity'] for p in prefix_metrics]),
                'device_precision': np.mean([p['device_precision'] for p in prefix_metrics]),
                'param_f1': np.mean([p['param_f1'] for p in prefix_metrics]),
                'numerical_precision': np.mean([p['numerical_precision'] for p in prefix_metrics]),
            }
    except:
        print("Warning: Prefix Per-User results not found")

    # 6. Selective Routing
    try:
        with open('results/selective_routing/routing_decisions.json') as f:
            data = json.load(f)
            results['Selective Routing'] = {
                'embedding_similarity': data['summary']['avg_selective_score'],
                'device_precision': None,  # Not computed
                'param_f1': None,
                'numerical_precision': None,
            }
    except:
        print("Warning: Selective Routing results not found")

    # 7. Simple Merged LoRA
    try:
        with open('results/merged_simple/results.json') as f:
            data = json.load(f)
            results['Simple Merge'] = {
                'embedding_similarity': data['embedding_similarity'],
                'device_precision': data.get('device_precision'),
                'param_f1': data.get('param_f1'),
                'numerical_precision': data.get('numerical_precision'),
            }
    except:
        print("Warning: Simple Merge results not found")

    # 8. Cluster Merged LoRA
    try:
        with open('results/cluster_merged/results.json') as f:
            data = json.load(f)
            results['Cluster Merge'] = {
                'embedding_similarity': data['embedding_similarity'],
                'device_precision': data.get('device_precision'),
                'param_f1': data.get('param_f1'),
                'numerical_precision': data.get('numerical_precision'),
            }
    except:
        print("Warning: Cluster Merge results not found")

    # 9. Cluster LoRA (trained)
    try:
        with open('results/cluster_lora/cluster_lora_results.json') as f:
            data = json.load(f)
            results['Cluster LoRA'] = {
                'embedding_similarity': data['summary']['embedding_similarity_mean'],
                'device_precision': data['summary']['device_precision_mean'],
                'param_f1': data['summary']['param_f1_mean'],
                'numerical_precision': data['summary']['numerical_precision_mean'],
            }
    except:
        print("Warning: Cluster LoRA results not found")

    # 10. Sparse MoE
    try:
        with open('results/moe_sparse/moe_sparse_results.json') as f:
            data = json.load(f)
            results['Sparse MoE'] = {
                'embedding_similarity': data['summary']['embedding_similarity_mean'],
                'device_precision': None,  # Not computed
                'param_f1': None,
                'numerical_precision': None,
            }
    except:
        print("Warning: Sparse MoE results not found")

    return results


def create_comprehensive_comparison(results):
    """Create comprehensive comparison visualizations"""

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Extract data
    methods = list(results.keys())
    emb_sim = [results[m]['embedding_similarity'] for m in methods]
    dev_prec = [results[m]['device_precision'] if results[m]['device_precision'] is not None else 0 for m in methods]
    param_f1 = [results[m]['param_f1'] if results[m]['param_f1'] is not None else 0 for m in methods]
    num_prec = [results[m]['numerical_precision'] if results[m]['numerical_precision'] is not None else 0 for m in methods]

    # Get unified baseline for comparisons
    unified_score = results.get('Unified LoRA', {}).get('embedding_similarity', 0.8214)

    # Colors
    colors = []
    for method in methods:
        if 'Baseline' in method:
            colors.append('#808080')  # Gray
        elif 'Unified' in method:
            colors.append('#1f77b4')  # Blue (baseline)
        elif 'Per-Persona' in method:
            colors.append('#d62728')  # Red (overfits)
        elif 'Cluster' in method:
            colors.append('#2ca02c')  # Green (good!)
        elif 'MoE' in method:
            colors.append('#17becf')  # Cyan (best!)
        elif 'Merge' in method:
            colors.append('#9467bd')  # Purple (good!)
        elif 'Routing' in method:
            colors.append('#ff7f0e')  # Orange
        else:
            colors.append('#8c564b')  # Brown

    # 1. Main comparison: Embedding Similarity
    ax1 = fig.add_subplot(gs[0, :2])
    bars = ax1.barh(methods, emb_sim, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add values on bars
    for i, (bar, score) in enumerate(zip(bars, emb_sim)):
        width = bar.get_width()
        improvement = ((score - unified_score) / unified_score) * 100

        # Value label
        ax1.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                f'{score:.4f}',
                ha='left', va='center', fontweight='bold', fontsize=10)

        # Improvement label (if not unified)
        if methods[i] != 'Unified LoRA' and methods[i] != 'Baseline':
            color = 'green' if improvement > 0 else 'red'
            ax1.text(width/2, bar.get_y() + bar.get_height()/2,
                    f'{improvement:+.1f}%',
                    ha='center', va='center', fontweight='bold',
                    fontsize=9, color=color)

    ax1.axvline(unified_score, color='blue', linestyle='--', linewidth=2, alpha=0.5,
               label=f'Unified Baseline ({unified_score:.4f})')
    ax1.set_xlabel('Embedding Similarity', fontsize=13, fontweight='bold')
    ax1.set_title('Model Comparison: Embedding Similarity', fontsize=15, fontweight='bold')
    ax1.set_xlim(0.6, max(emb_sim) * 1.1)
    ax1.legend(fontsize=10)
    ax1.grid(axis='x', alpha=0.3)

    # 2. Radar chart of all metrics
    ax2 = fig.add_subplot(gs[0, 2], projection='polar')

    # Select top 5 methods by embedding similarity
    top_methods_idx = np.argsort(emb_sim)[-5:]
    top_methods = [methods[i] for i in top_methods_idx]

    categories = ['Emb Sim', 'Dev Prec', 'Param F1', 'Num Prec']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    for method in top_methods:
        values = [
            results[method]['embedding_similarity'],
            results[method]['device_precision'] if results[method]['device_precision'] else 0,
            results[method]['param_f1'] if results[method]['param_f1'] else 0,
            results[method]['numerical_precision'] if results[method]['numerical_precision'] else 0
        ]
        values += values[:1]

        ax2.plot(angles, values, 'o-', linewidth=2, label=method)
        ax2.fill(angles, values, alpha=0.1)

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories)
    ax2.set_ylim(0, 1)
    ax2.set_title('Top 5 Methods: Multi-Metric Comparison', fontsize=12, fontweight='bold', pad=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    ax2.grid(True)

    # 3. Improvement over Unified
    ax3 = fig.add_subplot(gs[1, :])
    improvements = [((s - unified_score) / unified_score) * 100 for s in emb_sim]

    bar_colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax3.bar(methods, improvements, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height,
                f'{imp:+.2f}%',
                ha='center', va='bottom' if height > 0 else 'top',
                fontweight='bold', fontsize=10)

    ax3.axhline(0, color='blue', linestyle='--', linewidth=2, alpha=0.5, label='Unified Baseline')
    ax3.set_ylabel('Improvement over Unified (%)', fontsize=13, fontweight='bold')
    ax3.set_title('Improvement over Unified LoRA Baseline', fontsize=15, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 4. Device Precision comparison
    ax4 = fig.add_subplot(gs[2, 0])
    valid_dev = [(m, d) for m, d in zip(methods, dev_prec) if d > 0]
    if valid_dev:
        m_dev, d_dev = zip(*valid_dev)
        ax4.bar(range(len(m_dev)), d_dev, color=colors[:len(m_dev)], alpha=0.7, edgecolor='black')
        ax4.set_xticks(range(len(m_dev)))
        ax4.set_xticklabels(m_dev, rotation=45, ha='right', fontsize=8)
        ax4.set_ylabel('Device Precision')
        ax4.set_title('Device Precision Comparison', fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)

    # 5. Param F1 comparison
    ax5 = fig.add_subplot(gs[2, 1])
    valid_param = [(m, p) for m, p in zip(methods, param_f1) if p > 0]
    if valid_param:
        m_param, p_param = zip(*valid_param)
        ax5.bar(range(len(m_param)), p_param, color=colors[:len(m_param)], alpha=0.7, edgecolor='black')
        ax5.set_xticks(range(len(m_param)))
        ax5.set_xticklabels(m_param, rotation=45, ha='right', fontsize=8)
        ax5.set_ylabel('Parameter F1')
        ax5.set_title('Parameter F1 Comparison', fontweight='bold')
        ax5.grid(axis='y', alpha=0.3)

    # 6. Numerical Precision comparison
    ax6 = fig.add_subplot(gs[2, 2])
    valid_num = [(m, n) for m, n in zip(methods, num_prec) if n > 0]
    if valid_num:
        m_num, n_num = zip(*valid_num)
        ax6.bar(range(len(m_num)), n_num, color=colors[:len(m_num)], alpha=0.7, edgecolor='black')
        ax6.set_xticks(range(len(m_num)))
        ax6.set_xticklabels(m_num, rotation=45, ha='right', fontsize=8)
        ax6.set_ylabel('Numerical Precision')
        ax6.set_title('Numerical Precision Comparison', fontweight='bold')
        ax6.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_path = Path('results/figures/comprehensive_comparison.png')
    output_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comprehensive comparison to {output_path}")
    plt.close()


def create_summary_table(results):
    """Create summary table"""

    print("\n" + "=" * 100)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 100)

    # Create DataFrame
    data = []
    for method, metrics in results.items():
        data.append({
            'Method': method,
            'Emb Sim': metrics['embedding_similarity'],
            'Dev Prec': metrics['device_precision'] if metrics['device_precision'] else np.nan,
            'Param F1': metrics['param_f1'] if metrics['param_f1'] else np.nan,
            'Num Prec': metrics['numerical_precision'] if metrics['numerical_precision'] else np.nan,
        })

    df = pd.DataFrame(data)

    # Add improvement column
    unified_score = results.get('Unified LoRA', {}).get('embedding_similarity', 0.8214)
    df['vs Unified'] = ((df['Emb Sim'] - unified_score) / unified_score * 100).round(2)

    # Sort by embedding similarity
    df = df.sort_values('Emb Sim', ascending=False)

    print(df.to_string(index=False))

    # Save to file
    output_path = Path('results/summary_table.txt')
    with open(output_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("COMPREHENSIVE RESULTS SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        f.write(df.to_string(index=False))

    print(f"\nSaved summary table to {output_path}")

    # Also save as CSV
    df.to_csv('results/summary_table.csv', index=False)
    print(f"Saved CSV to results/summary_table.csv")

    return df


def main():
    """Run comprehensive comparison"""

    print("Loading all results...")
    results = load_all_results()

    if not results:
        print("ERROR: No results found!")
        print("Make sure you've run the experiments first.")
        return

    print(f"\nFound {len(results)} methods")
    for method in results.keys():
        print(f"  - {method}")

    print("\nCreating visualizations...")
    create_comprehensive_comparison(results)

    print("\nCreating summary table...")
    df = create_summary_table(results)

    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)
    print("\nGenerated files:")
    print("  - results/figures/comprehensive_comparison.png")
    print("  - results/summary_table.txt")
    print("  - results/summary_table.csv")


if __name__ == '__main__':
    main()
