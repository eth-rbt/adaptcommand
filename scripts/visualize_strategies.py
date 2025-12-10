"""
Visualize how Selective Routing vs RAG work
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_strategy_diagrams():
    """Create side-by-side visual diagrams of both strategies"""

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # SELECTIVE ROUTING
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Selective Routing', fontsize=16, fontweight='bold')

    # Query box
    query_box = FancyBboxPatch((1, 8), 3, 1, boxstyle="round,pad=0.1",
                               facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(query_box)
    ax.text(2.5, 8.5, 'User Query\n(persona_091)', ha='center', va='center', fontsize=10, fontweight='bold')

    # Routing table
    route_box = FancyBboxPatch((1, 5.5), 3, 1.5, boxstyle="round,pad=0.1",
                               facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(route_box)
    ax.text(2.5, 6.5, 'Routing Table', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(2.5, 6.1, 'persona_091 → hybrid', ha='center', va='center', fontsize=8)

    # Arrow from query to routing
    arrow1 = FancyArrowPatch((2.5, 8), (2.5, 7), arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow1)

    # Models
    y_positions = [4, 2.5, 1]
    model_names = ['Unified\n(155/200)', 'Hybrid\n(41/200)', 'Personalized\n(4/200)']
    colors = ['lightgray', 'lightgreen', 'lightcoral']

    for i, (y, name, color) in enumerate(zip(y_positions, model_names, colors)):
        model_box = FancyBboxPatch((5.5, y-0.4), 2.5, 0.8, boxstyle="round,pad=0.1",
                                   facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(model_box)
        ax.text(6.75, y, name, ha='center', va='center', fontsize=9, fontweight='bold')

        # Arrow from routing to each model (highlight hybrid)
        if i == 1:  # Hybrid
            arrow = FancyArrowPatch((4, 6), (5.5, y), arrowstyle='->', lw=3, color='green')
        else:
            arrow = FancyArrowPatch((4, 6), (5.5, y), arrowstyle='->', lw=1, color='gray', alpha=0.3)
        ax.add_patch(arrow)

    # Output
    output_box = FancyBboxPatch((5.5, 0), 2.5, 0.6, boxstyle="round,pad=0.1",
                               facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(6.75, 0.3, 'Response\n(93.16%)', ha='center', va='center', fontsize=9, fontweight='bold')

    arrow_to_output = FancyArrowPatch((6.75, 2.1), (6.75, 0.6), arrowstyle='->', lw=3, color='green')
    ax.add_patch(arrow_to_output)

    # RAG
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Retrieval-Augmented Generation (RAG)', fontsize=16, fontweight='bold')

    # Query box
    query_box = FancyBboxPatch((1, 8), 3, 1, boxstyle="round,pad=0.1",
                               facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(query_box)
    ax.text(2.5, 8.5, 'User Query\n(persona_091)', ha='center', va='center', fontsize=10, fontweight='bold')

    # Memory index
    memory_box = FancyBboxPatch((5.5, 7), 3.5, 2, boxstyle="round,pad=0.1",
                                facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(memory_box)
    ax.text(7.25, 8.5, 'User Memory Index', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(7.25, 8, '1. "Turn lights on" → ...', ha='center', va='center', fontsize=7)
    ax.text(7.25, 7.6, '2. "Play workout music" → ...', ha='center', va='center', fontsize=7)
    ax.text(7.25, 7.2, '3. "Lights on please" → ...', ha='center', va='center', fontsize=7)

    # Arrow from query to memory
    arrow1 = FancyArrowPatch((4, 8.5), (5.5, 8.5), arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow1)
    ax.text(4.75, 8.8, 'Search', ha='center', va='center', fontsize=8)

    # Retrieved examples
    retrieved_box = FancyBboxPatch((1, 5), 7, 1.5, boxstyle="round,pad=0.1",
                                   facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(retrieved_box)
    ax.text(4.5, 6.2, 'Augmented Context', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(4.5, 5.8, 'Query + Top-3 similar past interactions', ha='center', va='center', fontsize=8)
    ax.text(4.5, 5.4, '(full brightness, high-energy, volume 80)', ha='center', va='center', fontsize=7, style='italic')

    # Arrow from memory to retrieved
    arrow2 = FancyArrowPatch((7.25, 7), (6, 6.5), arrowstyle='->', lw=2, color='black')
    ax.add_patch(arrow2)

    # Unified model (single)
    model_box = FancyBboxPatch((2.5, 2.5), 3, 1, boxstyle="round,pad=0.1",
                               facecolor='lightcoral', edgecolor='black', linewidth=2)
    ax.add_patch(model_box)
    ax.text(4, 3, 'Unified Model', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(4, 2.7, '(ONE model for all)', ha='center', va='center', fontsize=8)

    # Arrow from augmented to model
    arrow3 = FancyArrowPatch((4.5, 5), (4, 3.5), arrowstyle='->', lw=3, color='green')
    ax.add_patch(arrow3)

    # Output
    output_box = FancyBboxPatch((2.5, 0.5), 3, 1, boxstyle="round,pad=0.1",
                               facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(4, 1, 'Response', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(4, 0.7, '(~90-95% estimated)', ha='center', va='center', fontsize=8)

    # Arrow from model to output
    arrow4 = FancyArrowPatch((4, 2.5), (4, 1.5), arrowstyle='->', lw=3, color='green')
    ax.add_patch(arrow4)

    plt.tight_layout()
    plt.savefig('results/figures/strategy_comparison_diagram.png', dpi=150, bbox_inches='tight')
    print("Saved diagram to results/figures/strategy_comparison_diagram.png")
    plt.close()


def create_performance_comparison():
    """Create bar chart comparing all approaches"""

    methods = ['Baseline', 'Unified\nLoRA', 'Per-Persona\nLoRA', 'Hybrid\nLoRA',
               'Selective\nRouting', 'RAG\n(estimated)', 'Both\nCombined']

    scores = [0.6379, 0.8214, 0.6828, 0.7591, 0.8299, 0.85, 0.865]

    colors = ['gray', 'blue', 'red', 'orange', 'green', 'purple', 'darkgreen']

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(methods, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{score:.4f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add improvement percentages
    baseline_score = scores[1]  # Unified LoRA
    for i, (bar, score) in enumerate(zip(bars, scores)):
        if i > 1:  # Skip baseline and unified
            improvement = ((score - baseline_score) / baseline_score) * 100
            if improvement > 0:
                color = 'green'
                sign = '+'
            else:
                color = 'red'
                sign = ''

            ax.text(bar.get_x() + bar.get_width()/2., score/2,
                   f'{sign}{improvement:.1f}%',
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   color=color)

    # Horizontal line at unified baseline
    ax.axhline(baseline_score, color='blue', linestyle='--', linewidth=2, alpha=0.5,
              label=f'Unified Baseline ({baseline_score:.4f})')

    ax.set_ylabel('Embedding Similarity', fontsize=13, fontweight='bold')
    ax.set_title('Performance Comparison: Different Personalization Strategies', fontsize=14, fontweight='bold')
    ax.set_ylim(0.6, 0.9)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig('results/figures/performance_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved performance comparison to results/figures/performance_comparison.png")
    plt.close()


if __name__ == '__main__':
    print("Creating strategy visualization diagrams...")
    create_strategy_diagrams()
    create_performance_comparison()
    print("\nDone! Check results/figures/ for visualizations")
