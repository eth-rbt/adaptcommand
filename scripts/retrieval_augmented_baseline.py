"""
Strategy 2: Retrieval-Augmented Personalization

Instead of personalized adapters, use the unified model but augment with:
- User's most similar past interactions (retrieved via embedding similarity)
- User's learned preferences/routines
- Recent conversation history

This keeps the strong unified model while adding personalization through context.
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from tqdm import tqdm

class RetrievalAugmentedPersonalization:
    def __init__(self, base_model_name, lora_path, embedding_model_name='sentence-transformers/all-MiniLM-L6-v2'):
        """Initialize with unified LoRA model + retrieval system"""

        print("Loading unified LoRA model...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.model = PeftModel.from_pretrained(base_model, lora_path)

        print("Loading embedding model for retrieval...")
        self.encoder = SentenceTransformer(embedding_model_name)

        self.user_memories = {}  # persona_id -> list of (query, response, embedding)

    def index_user_history(self, dialogues, splits):
        """Build retrieval index from training data"""

        print("Building retrieval index from training data...")

        for dialogue in tqdm(dialogues):
            persona_id = dialogue['persona_id']
            session_id = dialogue['session_id']

            # Only index training data
            if session_id not in splits[persona_id]['train']:
                continue

            if persona_id not in self.user_memories:
                self.user_memories[persona_id] = []

            # Extract user queries and assistant responses
            messages = dialogue['messages']
            for i in range(0, len(messages) - 1, 2):
                if messages[i]['role'] == 'user' and messages[i+1]['role'] == 'assistant':
                    query = messages[i]['text']
                    response = messages[i+1]['text']

                    # Encode query for retrieval
                    embedding = self.encoder.encode(query, convert_to_numpy=True)

                    self.user_memories[persona_id].append({
                        'query': query,
                        'response': response,
                        'embedding': embedding,
                        'context': dialogue.get('meta', {})
                    })

        print(f"Indexed {sum(len(v) for v in self.user_memories.values())} interactions across {len(self.user_memories)} personas")

    def retrieve_similar_interactions(self, persona_id, current_query, top_k=3):
        """Retrieve top-k most similar past interactions for this user"""

        if persona_id not in self.user_memories:
            return []

        # Encode current query
        query_embedding = self.encoder.encode(current_query, convert_to_numpy=True)

        # Calculate similarities
        memories = self.user_memories[persona_id]
        similarities = [
            np.dot(query_embedding, mem['embedding']) / (np.linalg.norm(query_embedding) * np.linalg.norm(mem['embedding']))
            for mem in memories
        ]

        # Get top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [memories[i] for i in top_indices]

    def create_augmented_prompt(self, persona_id, current_query, character, context, retrieved_k=3):
        """Create prompt with retrieved user history"""

        # Get similar past interactions
        similar = self.retrieve_similar_interactions(persona_id, current_query, top_k=retrieved_k)

        # Build system prompt with user history
        system_parts = [
            "You are a helpful smart home assistant.",
            f"\nUser Profile: {character}",
        ]

        if similar:
            system_parts.append("\nRelevant past interactions:")
            for i, mem in enumerate(similar, 1):
                system_parts.append(f"\n{i}. User: {mem['query']}")
                system_parts.append(f"   You: {mem['response']}")

        # Add current context
        if context:
            context_str = ", ".join(f"{k}: {v}" for k, v in context.items() if k != 'routines')
            system_parts.append(f"\nCurrent context: {context_str}")

        system_prompt = "\n".join(system_parts)

        # Format with chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": current_query}
        ]

        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate(self, prompt, max_new_tokens=256, temperature=0.7):
        """Generate response"""

        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1
            )

        response = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        return response.strip()


def run_retrieval_augmented_benchmark():
    """
    Evaluate retrieval-augmented unified model.

    Hypothesis: This should beat unified model by leveraging user-specific
    history without overfitting to small per-user datasets.
    """

    # Load data
    with open('data/cleaned/dialogs_clean.jsonl') as f:
        dialogues = [json.loads(line) for line in f]

    with open('data/splits/edgesplits.json') as f:
        splits = json.load(f)

    # Initialize
    rag = RetrievalAugmentedPersonalization(
        base_model_name='Qwen/Qwen2.5-0.5B-Instruct',
        lora_path='models/lora_unified'
    )

    # Build retrieval index
    rag.index_user_history(dialogues, splits)

    # Evaluate on test set
    print("\nEvaluating on test set...")

    from scripts.action_metrics import ActionExtractor, ActionMetrics
    from sentence_transformers import SentenceTransformer

    extractor = ActionExtractor()
    similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    results = []
    for dialogue in tqdm(dialogues[:100]):  # Start with 100 for quick test
        persona_id = dialogue['persona_id']
        session_id = dialogue['session_id']

        # Only test set
        if session_id not in splits[persona_id]['test']:
            continue

        messages = dialogue['messages']
        for i in range(0, len(messages) - 1, 2):
            if messages[i]['role'] == 'user' and messages[i+1]['role'] == 'assistant':
                query = messages[i]['text']
                reference = messages[i+1]['text']

                # Generate with retrieval augmentation (varying k)
                for k in [0, 1, 3, 5]:  # Compare different retrieval amounts
                    prompt = rag.create_augmented_prompt(
                        persona_id,
                        query,
                        dialogue['character'],
                        dialogue.get('meta', {}),
                        retrieved_k=k
                    )

                    prediction = rag.generate(prompt)

                    # Calculate metrics
                    emb_sim = float(
                        np.dot(
                            similarity_model.encode(prediction),
                            similarity_model.encode(reference)
                        ) / (
                            np.linalg.norm(similarity_model.encode(prediction)) *
                            np.linalg.norm(similarity_model.encode(reference))
                        )
                    )

                    pred_actions = extractor.extract_actions(prediction)
                    ref_actions = extractor.extract_actions(reference)
                    action_metrics = ActionMetrics.compare_actions(pred_actions, ref_actions)

                    results.append({
                        'persona_id': persona_id,
                        'k_retrieved': k,
                        'embedding_similarity': emb_sim,
                        **action_metrics
                    })

    # Aggregate by k
    print("\n" + "=" * 80)
    print("Results by retrieval k:")
    print("-" * 80)

    for k in [0, 1, 3, 5]:
        k_results = [r for r in results if r['k_retrieved'] == k]
        if not k_results:
            continue

        avg_sim = np.mean([r['embedding_similarity'] for r in k_results])
        avg_dev = np.mean([r['device_precision'] for r in k_results])

        print(f"k={k}: Emb Sim={avg_sim:.4f}, Device Prec={avg_dev:.4f}")

    # Save results
    output_path = Path('results/retrieval_augmented')
    output_path.mkdir(exist_ok=True, parents=True)

    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {output_path / 'results.json'}")

if __name__ == '__main__':
    run_retrieval_augmented_benchmark()
