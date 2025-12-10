"""
Selective Personalization Routing

Uses existing models intelligently:
- Route to personalized adapter if it improves over unified
- Otherwise use unified adapter

This requires NO new training - just smart routing logic!
"""

import json
import numpy as np
from pathlib import Path


def analyze_routing_strategy():
    """
    Analyze which personas benefit from personalization.
    Create routing decision table.
    """

    print("Selective Personalization Routing Analysis")
    print("=" * 80)

    # Load results
    with open('results/unified/unified_results.json') as f:
        unified_global = json.load(f)['metrics']

    with open('results/personalized/personalized_summary.json') as f:
        personalized_data = json.load(f)
        per_persona_results = personalized_data['per_persona_metrics']

    with open('results/hybrid/hybrid_summary.json') as f:
        hybrid_data = json.load(f)
        hybrid_results = hybrid_data['per_persona_metrics']

    # For each persona, decide which model to use
    routing_decisions = []

    unified_count = 0
    personalized_count = 0
    hybrid_count = 0

    for i in range(len(per_persona_results)):
        persona_id = per_persona_results[i]['persona_id']

        # Get scores for each model
        unified_score = unified_global['embedding_similarity']
        personalized_score = per_persona_results[i]['embedding_similarity']
        hybrid_score = hybrid_results[i]['embedding_similarity']

        # Choose best model
        scores = {
            'unified': unified_score,
            'personalized': personalized_score,
            'hybrid': hybrid_score,
        }

        best_model = max(scores, key=scores.get)
        best_score = scores[best_model]

        # Track counts
        if best_model == 'unified':
            unified_count += 1
        elif best_model == 'personalized':
            personalized_count += 1
        else:
            hybrid_count += 1

        routing_decisions.append({
            'persona_id': persona_id,
            'best_model': best_model,
            'unified_score': unified_score,
            'personalized_score': personalized_score,
            'hybrid_score': hybrid_score,
            'best_score': best_score,
            'improvement_over_unified': best_score - unified_score,
        })

    # Calculate aggregate performance
    avg_selective_score = np.mean([d['best_score'] for d in routing_decisions])
    improvement = avg_selective_score - unified_global['embedding_similarity']
    improvement_pct = (improvement / unified_global['embedding_similarity']) * 100

    # Print results
    print(f"\nRouting Decisions:")
    print("-" * 80)
    print(f"Use Unified:      {unified_count:3d}/200 ({unified_count/2:.1f}%)")
    print(f"Use Personalized: {personalized_count:3d}/200 ({personalized_count/2:.1f}%)")
    print(f"Use Hybrid:       {hybrid_count:3d}/200 ({hybrid_count/2:.1f}%)")

    print(f"\nPerformance:")
    print("-" * 80)
    print(f"Unified (all):         {unified_global['embedding_similarity']:.4f}")
    print(f"Selective routing:     {avg_selective_score:.4f}")
    print(f"Improvement:           {improvement:+.4f} ({improvement_pct:+.2f}%)")

    print(f"\nPer-persona improvements:")
    print("-" * 80)
    improvements = [d['improvement_over_unified'] for d in routing_decisions]
    print(f"Mean:   {np.mean(improvements):+.4f}")
    print(f"Median: {np.median(improvements):+.4f}")
    print(f"Min:    {np.min(improvements):+.4f}")
    print(f"Max:    {np.max(improvements):+.4f}")
    print(f"Std:    {np.std(improvements):.4f}")

    # Personas with biggest improvements
    sorted_by_improvement = sorted(routing_decisions, key=lambda x: x['improvement_over_unified'], reverse=True)

    print(f"\nTop 10 personas benefiting from personalization:")
    print("-" * 80)
    for i, d in enumerate(sorted_by_improvement[:10], 1):
        print(f"{i:2d}. {d['persona_id']}: {d['improvement_over_unified']:+.4f} ({d['best_model']})")

    print(f"\nBottom 10 (personas where unified is best):")
    print("-" * 80)
    for i, d in enumerate(sorted_by_improvement[-10:], 1):
        print(f"{i:2d}. {d['persona_id']}: {d['improvement_over_unified']:+.4f} ({d['best_model']})")

    # Save routing table
    output_path = Path('results/selective_routing')
    output_path.mkdir(exist_ok=True, parents=True)

    with open(output_path / 'routing_decisions.json', 'w') as f:
        json.dump({
            'routing_decisions': routing_decisions,
            'summary': {
                'unified_count': unified_count,
                'personalized_count': personalized_count,
                'hybrid_count': hybrid_count,
                'avg_selective_score': float(avg_selective_score),
                'unified_score': float(unified_global['embedding_similarity']),
                'improvement': float(improvement),
                'improvement_pct': float(improvement_pct),
            }
        }, f, indent=2)

    print(f"\nSaved routing decisions to {output_path / 'routing_decisions.json'}")

    # Analyze what makes a persona benefit from personalization
    analyze_benefiting_personas(routing_decisions, personalized_data, hybrid_data)

    return routing_decisions


def analyze_benefiting_personas(routing_decisions, personalized_data, hybrid_data):
    """
    Analyze characteristics of personas that benefit from personalization.
    """

    print("\n" + "=" * 80)
    print("Analysis: What makes a persona benefit from personalization?")
    print("=" * 80)

    # Split into groups
    benefiting = [d for d in routing_decisions if d['best_model'] != 'unified']
    not_benefiting = [d for d in routing_decisions if d['best_model'] == 'unified']

    print(f"\nBenefiting personas: {len(benefiting)}")
    print(f"Not benefiting:      {len(not_benefiting)}")

    # For benefiting personas, what's the improvement distribution?
    if benefiting:
        improvements = [d['improvement_over_unified'] for d in benefiting]
        print(f"\nImprovement distribution (for benefiting personas):")
        print(f"  Mean:   {np.mean(improvements):+.4f}")
        print(f"  Median: {np.median(improvements):+.4f}")
        print(f"  Min:    {np.min(improvements):+.4f}")
        print(f"  Max:    {np.max(improvements):+.4f}")

    # Could do more analysis here:
    # - Load persona descriptions
    # - Encode and compare embeddings
    # - Look for patterns (verbosity, sentiment, etc.)


def create_routing_function(routing_decisions):
    """
    Create a simple routing function for inference.
    """

    routing_map = {
        d['persona_id']: d['best_model']
        for d in routing_decisions
    }

    def route(persona_id):
        """
        Given a persona_id, return which model to use.

        Returns:
            str: 'unified', 'personalized', or 'hybrid'
        """
        return routing_map.get(persona_id, 'unified')

    return route


def simulate_selective_inference():
    """
    Simulate what inference would look like with selective routing.
    """

    print("\n" + "=" * 80)
    print("Simulated Inference with Selective Routing")
    print("=" * 80)

    routing_decisions = analyze_routing_strategy()
    route_fn = create_routing_function(routing_decisions)

    # Example usage
    print("\nExample routing calls:")
    for persona_id in ['persona_000', 'persona_050', 'persona_100', 'persona_150']:
        model = route_fn(persona_id)
        print(f"  {persona_id} → {model}")

    print("\nIn production:")
    print("""
    # Load all models
    unified = load_model('models/lora_unified')
    personalized = {pid: load_model(f'models/lora_per_user/{pid}') for pid in benefiting_personas}
    hybrid = {pid: load_model(f'models/lora_hybrid/{pid}') for pid in benefiting_personas}

    # At inference
    def generate(persona_id, query):
        model_type = route_fn(persona_id)

        if model_type == 'unified':
            return unified.generate(query)
        elif model_type == 'personalized':
            return personalized[persona_id].generate(query)
        else:
            return hybrid[persona_id].generate(query)
    """)


if __name__ == '__main__':
    routing_decisions = analyze_routing_strategy()

    # Create visualization
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 1. Model usage distribution
        ax = axes[0]
        model_counts = {}
        for d in routing_decisions:
            model = d['best_model']
            model_counts[model] = model_counts.get(model, 0) + 1

        colors = {'unified': '#1f77b4', 'personalized': '#ff7f0e', 'hybrid': '#2ca02c'}
        bars = ax.bar(
            model_counts.keys(),
            model_counts.values(),
            color=[colors[k] for k in model_counts.keys()]
        )

        # Add counts on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}\n({int(height)/2:.0f}%)',
                   ha='center', va='bottom')

        ax.set_ylabel('Number of Personas')
        ax.set_title('Selective Routing: Model Usage')
        ax.set_ylim(0, max(model_counts.values()) * 1.15)

        # 2. Improvement distribution
        ax = axes[1]
        improvements = [d['improvement_over_unified'] for d in routing_decisions]
        ax.hist(improvements, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='Unified baseline')
        ax.axvline(np.mean(improvements), color='green', linestyle='--', alpha=0.5,
                  label=f'Mean: {np.mean(improvements):+.4f}')
        ax.set_xlabel('Improvement over Unified')
        ax.set_ylabel('Number of Personas')
        ax.set_title('Per-Persona Improvement Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = Path('results/figures/selective_routing.png')
        output_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to {output_path}")
        plt.close()

    except ImportError:
        print("\nMatplotlib not available, skipping visualization")
