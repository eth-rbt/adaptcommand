"""
Prefix Tuning Proof of Concept for persona_000

4-way comparison:
1. Unified (no prefix) - baseline
2. Static text prefix (character description)
3. Learned persona prefix (train on persona_000's 30 examples)
4. Learned cluster prefix (train on cluster 1's 1560 examples)
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from torch.utils.data import Dataset
import argparse


class PrefixTuningModel(nn.Module):
    """Adds learnable prefix to frozen unified LoRA model"""

    def __init__(self, base_model, prefix_length=10):
        super().__init__()
        self.base_model = base_model
        self.prefix_length = prefix_length

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Learnable prefix embeddings
        hidden_size = base_model.config.hidden_size
        self.prefix_embeddings = nn.Parameter(
            torch.randn(prefix_length, hidden_size) * 0.01
        )

        print(f"Prefix params: {self.prefix_embeddings.numel():,}")
        print(f"Total base params: {sum(p.numel() for p in base_model.parameters()):,}")

    def forward(self, input_ids, attention_mask, labels=None):
        batch_size = input_ids.shape[0]

        # Get input embeddings
        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)

        # Expand prefix for batch
        prefix = self.prefix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate prefix with input
        combined_embeds = torch.cat([prefix, inputs_embeds], dim=1)

        # Extend attention mask for prefix
        prefix_mask = torch.ones(
            batch_size, self.prefix_length,
            device=attention_mask.device,
            dtype=attention_mask.dtype
        )
        combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # Adjust labels for prefix
        if labels is not None:
            prefix_labels = torch.full(
                (batch_size, self.prefix_length),
                -100,
                device=labels.device,
                dtype=labels.dtype
            )
            combined_labels = torch.cat([prefix_labels, labels], dim=1)
        else:
            combined_labels = None

        # Forward through base model
        outputs = self.base_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            labels=combined_labels
        )

        return outputs


class DialogueDataset(Dataset):
    """Dataset for prefix tuning"""

    def __init__(self, dialogues, tokenizer, max_length=512):
        self.dialogues = dialogues
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        dialogue = self.dialogues[idx]

        # Build system message
        system_parts = [
            "You are a helpful smart home assistant.",
            f"\nUser Profile: {dialogue['character']}"
        ]

        if 'meta' in dialogue and dialogue['meta']:
            context_items = []
            for k, v in dialogue['meta'].items():
                if k != 'routines' and v:
                    context_items.append(f"{k}: {v}")
            if context_items:
                system_parts.append(f"\nContext: {', '.join(context_items)}")

        system_message = "\n".join(system_parts)

        # Format conversation
        messages = [{"role": "system", "content": system_message}]
        for msg in dialogue['messages']:
            messages.append({
                "role": msg['role'],
                "content": msg['text']
            })

        # Tokenize
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }


def load_training_data(persona_id=None, cluster_id=None):
    """Load training dialogues for persona or cluster"""

    with open('data/cleaned/dialogs_clean.jsonl') as f:
        all_dialogues = [json.loads(line) for line in f]

    with open('data/splits/edgesplits.json') as f:
        splits = json.load(f)

    if cluster_id is not None:
        with open('data/splits/cluster_map.json') as f:
            cluster_data = json.load(f)
        cluster_personas = [p for p, c in cluster_data['cluster_map'].items()
                           if c == cluster_id]
        persona_ids = cluster_personas
    elif persona_id is not None:
        persona_ids = [persona_id]
    else:
        raise ValueError("Must specify persona_id or cluster_id")

    train_dialogues = []
    for idx, dialogue in enumerate(all_dialogues):
        if dialogue['persona_id'] in persona_ids:
            if idx in splits[dialogue['persona_id']]['train']:
                train_dialogues.append(dialogue)

    return train_dialogues


def load_test_data(persona_id):
    """Load test dialogues for persona"""

    with open('data/cleaned/dialogs_clean.jsonl') as f:
        all_dialogues = [json.loads(line) for line in f]

    with open('data/splits/edgesplits.json') as f:
        splits = json.load(f)

    test_dialogues = []
    for idx, dialogue in enumerate(all_dialogues):
        if dialogue['persona_id'] == persona_id:
            if idx in splits[persona_id]['test']:
                test_dialogues.append(dialogue)

    return test_dialogues


def train_prefix(model, train_dataset, output_dir, epochs=10, lr=1e-3):
    """Train prefix embeddings"""

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        learning_rate=lr,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy='no',
        report_to='none',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()

    return model


def evaluate_model(model, tokenizer, test_dialogues, use_prefix=True):
    """Evaluate model on test set"""

    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    similarities = []

    model.eval()

    for dialogue in tqdm(test_dialogues, desc="Evaluating"):
        # Build input
        system_parts = [
            "You are a helpful smart home assistant.",
            f"\nUser Profile: {dialogue['character']}"
        ]

        if 'meta' in dialogue and dialogue['meta']:
            context_items = []
            for k, v in dialogue['meta'].items():
                if k != 'routines' and v:
                    context_items.append(f"{k}: {v}")
            if context_items:
                system_parts.append(f"\nContext: {', '.join(context_items)}")

        system_message = "\n".join(system_parts)

        messages = [{"role": "system", "content": system_message}]
        for msg in dialogue['messages'][:-1]:
            messages.append({
                "role": msg['role'],
                "content": msg['text']
            })

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Generate
        inputs = tokenizer(text, return_tensors='pt').to(model.device if hasattr(model, 'device') else 'cpu')

        with torch.no_grad():
            if use_prefix and hasattr(model, 'prefix_embeddings'):
                # Add prefix
                inputs_embeds = model.base_model.get_input_embeddings()(inputs['input_ids'])
                prefix = model.prefix_embeddings.unsqueeze(0)
                combined_embeds = torch.cat([prefix, inputs_embeds], dim=1)

                prefix_mask = torch.ones(1, model.prefix_length, device=inputs['attention_mask'].device)
                combined_mask = torch.cat([prefix_mask, inputs['attention_mask']], dim=1)

                outputs = model.base_model.generate(
                    inputs_embeds=combined_embeds,
                    attention_mask=combined_mask,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                # Skip prefix tokens in output
                prediction = tokenizer.decode(outputs[0][model.prefix_length + inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                prediction = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Get ground truth
        target = dialogue['messages'][-1]['text']

        # Compute similarity
        pred_emb = encoder.encode([prediction], show_progress_bar=False)[0]
        target_emb = encoder.encode([target], show_progress_bar=False)[0]
        similarity = np.dot(pred_emb, target_emb) / (
            np.linalg.norm(pred_emb) * np.linalg.norm(target_emb)
        )
        similarities.append(float(similarity))

    return np.mean(similarities), np.std(similarities)


def evaluate_with_static_prefix(base_model, tokenizer, test_dialogues, persona_desc):
    """Evaluate with static text prefix"""

    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    similarities = []

    base_model.eval()

    for dialogue in tqdm(test_dialogues, desc="Eval (static prefix)"):
        # Build input with explicit prefix
        prefix_text = f"[Persona: {persona_desc}]\n\n"

        system_parts = [
            "You are a helpful smart home assistant.",
            f"\nUser Profile: {dialogue['character']}"
        ]

        if 'meta' in dialogue and dialogue['meta']:
            context_items = []
            for k, v in dialogue['meta'].items():
                if k != 'routines' and v:
                    context_items.append(f"{k}: {v}")
            if context_items:
                system_parts.append(f"\nContext: {', '.join(context_items)}")

        system_message = prefix_text + "\n".join(system_parts)

        messages = [{"role": "system", "content": system_message}]
        for msg in dialogue['messages'][:-1]:
            messages.append({
                "role": msg['role'],
                "content": msg['text']
            })

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Generate
        inputs = tokenizer(text, return_tensors='pt')

        with torch.no_grad():
            outputs = base_model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        prediction = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        target = dialogue['messages'][-1]['text']

        # Compute similarity
        pred_emb = encoder.encode([prediction], show_progress_bar=False)[0]
        target_emb = encoder.encode([target], show_progress_bar=False)[0]
        similarity = np.dot(pred_emb, target_emb) / (
            np.linalg.norm(pred_emb) * np.linalg.norm(target_emb)
        )
        similarities.append(float(similarity))

    return np.mean(similarities), np.std(similarities)


def main():
    print("="*80)
    print("PREFIX TUNING PROOF OF CONCEPT - persona_000")
    print("="*80)

    # Load unified model
    print("\nLoading unified LoRA model...")
    base_model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
    unified_path = 'models/lora_unified'

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    unified_model = PeftModel.from_pretrained(base_model, unified_path)
    unified_model = unified_model.merge_and_unload()  # Merge for easier handling

    # Load test data
    test_dialogues = load_test_data('persona_000')
    persona_desc = test_dialogues[0]['character']

    print(f"\nPersona: {persona_desc}")
    print(f"Test examples: {len(test_dialogues)}")

    # Get cluster info
    with open('data/splits/cluster_map.json') as f:
        cluster_data = json.load(f)
    cluster_id = cluster_data['cluster_map']['persona_000']
    print(f"Cluster: {cluster_id}")

    results = {}

    # 1. Unified baseline (no prefix)
    print("\n" + "="*80)
    print("1. UNIFIED BASELINE (no prefix)")
    print("="*80)
    mean_sim, std_sim = evaluate_model(unified_model, tokenizer, test_dialogues, use_prefix=False)
    results['unified'] = {'mean': mean_sim, 'std': std_sim}
    print(f"Result: {mean_sim:.4f} ± {std_sim:.4f}")

    # 2. Static text prefix
    print("\n" + "="*80)
    print("2. STATIC TEXT PREFIX (character description)")
    print("="*80)
    mean_sim, std_sim = evaluate_with_static_prefix(unified_model, tokenizer, test_dialogues, persona_desc)
    results['static_prefix'] = {'mean': mean_sim, 'std': std_sim}
    print(f"Result: {mean_sim:.4f} ± {std_sim:.4f}")

    # 3. Learned persona prefix
    print("\n" + "="*80)
    print("3. LEARNED PERSONA PREFIX (30 examples)")
    print("="*80)
    train_persona = load_training_data(persona_id='persona_000')
    print(f"Training examples: {len(train_persona)}")

    train_dataset = DialogueDataset(train_persona, tokenizer)

    persona_prefix_model = PrefixTuningModel(unified_model, prefix_length=10)
    print("Training prefix...")
    persona_prefix_model = train_prefix(
        persona_prefix_model,
        train_dataset,
        'models/prefix_poc/persona_000',
        epochs=10,
        lr=1e-3
    )

    # Save prefix
    output_dir = Path('models/prefix_poc/persona_000')
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(persona_prefix_model.prefix_embeddings, output_dir / 'prefix.pt')

    mean_sim, std_sim = evaluate_model(persona_prefix_model, tokenizer, test_dialogues, use_prefix=True)
    results['learned_persona'] = {'mean': mean_sim, 'std': std_sim}
    print(f"Result: {mean_sim:.4f} ± {std_sim:.4f}")

    # 4. Learned cluster prefix
    print("\n" + "="*80)
    print(f"4. LEARNED CLUSTER PREFIX (cluster {cluster_id}, ~1560 examples)")
    print("="*80)
    train_cluster = load_training_data(cluster_id=cluster_id)
    print(f"Training examples: {len(train_cluster)}")

    train_dataset_cluster = DialogueDataset(train_cluster, tokenizer)

    cluster_prefix_model = PrefixTuningModel(unified_model, prefix_length=10)
    print("Training prefix...")
    cluster_prefix_model = train_prefix(
        cluster_prefix_model,
        train_dataset_cluster,
        f'models/prefix_poc/cluster_{cluster_id}',
        epochs=10,
        lr=1e-3
    )

    # Save prefix
    output_dir = Path(f'models/prefix_poc/cluster_{cluster_id}')
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(cluster_prefix_model.prefix_embeddings, output_dir / 'prefix.pt')

    mean_sim, std_sim = evaluate_model(cluster_prefix_model, tokenizer, test_dialogues, use_prefix=True)
    results['learned_cluster'] = {'mean': mean_sim, 'std': std_sim}
    print(f"Result: {mean_sim:.4f} ± {std_sim:.4f}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY - persona_000")
    print("="*80)
    print(f"1. Unified (no prefix):       {results['unified']['mean']:.4f} ± {results['unified']['std']:.4f}")
    print(f"2. Static text prefix:        {results['static_prefix']['mean']:.4f} ± {results['static_prefix']['std']:.4f}")
    print(f"3. Learned persona prefix:    {results['learned_persona']['mean']:.4f} ± {results['learned_persona']['std']:.4f}")
    print(f"4. Learned cluster prefix:    {results['learned_cluster']['mean']:.4f} ± {results['learned_cluster']['std']:.4f}")

    # Save results
    output_dir = Path('results/prefix_poc')
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'persona_000_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'persona_000_results.json'}")

    # Best method
    best_method = max(results.items(), key=lambda x: x[1]['mean'])
    print(f"\nBEST: {best_method[0]} with {best_method[1]['mean']:.4f}")

    return results


if __name__ == '__main__':
    main()
