import json

# Load all results
with open('results/baseline/baseline_results.json') as f:
    baseline = json.load(f)['metrics']

with open('results/unified/unified_results.json') as f:
    unified = json.load(f)['metrics']

with open('results/personalized/personalized_summary.json') as f:
    data = json.load(f)
    per_persona_metrics = data['per_persona_metrics']

    # Calculate means
    per_persona = {
        'embedding_similarity': sum(p['embedding_similarity'] for p in per_persona_metrics) / len(per_persona_metrics),
        'device_precision': sum(p['device_precision'] for p in per_persona_metrics) / len(per_persona_metrics),
        'param_f1': sum(p['param_f1'] for p in per_persona_metrics) / len(per_persona_metrics),
        'numerical_precision': sum(p['numerical_precision'] for p in per_persona_metrics) / len(per_persona_metrics),
    }

with open('results/hybrid/hybrid_summary.json') as f:
    data = json.load(f)
    hybrid_metrics = data['per_persona_metrics']

    hybrid = {
        'embedding_similarity': sum(p['embedding_similarity'] for p in hybrid_metrics) / len(hybrid_metrics),
        'device_precision': sum(p['device_precision'] for p in hybrid_metrics) / len(hybrid_metrics),
        'param_f1': sum(p['param_f1'] for p in hybrid_metrics) / len(hybrid_metrics),
        'numerical_precision': sum(p['numerical_precision'] for p in hybrid_metrics) / len(hybrid_metrics),
    }

with open('results/prefix_per_user/prefix_per_user_summary.json') as f:
    data = json.load(f)
    prefix_metrics = data['per_persona_metrics']

    prefix = {
        'embedding_similarity': sum(p['embedding_similarity'] for p in prefix_metrics) / len(prefix_metrics),
        'device_precision': sum(p['device_precision'] for p in prefix_metrics) / len(prefix_metrics),
        'param_f1': sum(p['param_f1'] for p in prefix_metrics) / len(prefix_metrics),
        'numerical_precision': sum(p['numerical_precision'] for p in prefix_metrics) / len(prefix_metrics),
    }

print('Model Comparison')
print('=' * 80)
print(f"{'Model':<20} {'Emb Sim':>12} {'Dev Prec':>12} {'Param F1':>12} {'Num Prec':>12}")
print('-' * 80)
print(f"{'Baseline':<20} {baseline['embedding_similarity']:>12.4f} {baseline['device_precision']:>12.4f} {baseline['param_f1']:>12.4f} {baseline['numerical_precision']:>12.4f}")
print(f"{'Unified LoRA':<20} {unified['embedding_similarity']:>12.4f} {unified['device_precision']:>12.4f} {unified['param_f1']:>12.4f} {unified['numerical_precision']:>12.4f}")
print(f"{'Per-Persona LoRA':<20} {per_persona['embedding_similarity']:>12.4f} {per_persona['device_precision']:>12.4f} {per_persona['param_f1']:>12.4f} {per_persona['numerical_precision']:>12.4f}")
print(f"{'Hybrid LoRA':<20} {hybrid['embedding_similarity']:>12.4f} {hybrid['device_precision']:>12.4f} {hybrid['param_f1']:>12.4f} {hybrid['numerical_precision']:>12.4f}")
print(f"{'Prefix Per-User':<20} {prefix['embedding_similarity']:>12.4f} {prefix['device_precision']:>12.4f} {prefix['param_f1']:>12.4f} {prefix['numerical_precision']:>12.4f}")
print()
print('Improvement over Unified LoRA:')
print('-' * 80)
print(f"{'Per-Persona LoRA':<20} {per_persona['embedding_similarity'] - unified['embedding_similarity']:>12.4f} {per_persona['device_precision'] - unified['device_precision']:>12.4f} {per_persona['param_f1'] - unified['param_f1']:>12.4f} {per_persona['numerical_precision'] - unified['numerical_precision']:>12.4f}")
print(f"{'Hybrid LoRA':<20} {hybrid['embedding_similarity'] - unified['embedding_similarity']:>12.4f} {hybrid['device_precision'] - unified['device_precision']:>12.4f} {hybrid['param_f1'] - unified['param_f1']:>12.4f} {hybrid['numerical_precision'] - unified['numerical_precision']:>12.4f}")
print(f"{'Prefix Per-User':<20} {prefix['embedding_similarity'] - unified['embedding_similarity']:>12.4f} {prefix['device_precision'] - unified['device_precision']:>12.4f} {prefix['param_f1'] - unified['param_f1']:>12.4f} {prefix['numerical_precision'] - unified['numerical_precision']:>12.4f}")
print()

# Calculate how many personas benefit from personalization
print('Per-Persona Analysis:')
print('-' * 80)
per_better = 0
hybrid_better = 0
prefix_better = 0

for i in range(200):
    unified_score = unified['embedding_similarity']
    per_score = per_persona_metrics[i]['embedding_similarity']
    hyb_score = hybrid_metrics[i]['embedding_similarity']
    pre_score = prefix_metrics[i]['embedding_similarity']

    if per_score > unified_score:
        per_better += 1
    if hyb_score > unified_score:
        hybrid_better += 1
    if pre_score > unified_score:
        prefix_better += 1

print(f"Personas improved by Per-Persona LoRA: {per_better}/200 ({per_better/200*100:.1f}%)")
print(f"Personas improved by Hybrid LoRA: {hybrid_better}/200 ({hybrid_better/200*100:.1f}%)")
print(f"Personas improved by Prefix Per-User: {prefix_better}/200 ({prefix_better/200*100:.1f}%)")
