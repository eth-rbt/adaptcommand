"""
Retrieval-Augmented Generation (RAG) - Detailed Example

Shows exactly how RAG personalizes responses using context instead of weights.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer

class RAGExplainer:
    """
    Walk through a concrete example of how RAG works for personalization
    """

    def __init__(self):
        # We'll use a small dataset for the example
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def example_scenario(self):
        """
        Show a complete example with a real persona
        """

        print("=" * 100)
        print("RAG FOR PERSONALIZATION - COMPLETE EXAMPLE")
        print("=" * 100)
        print()

        # PERSONA: Emma (from your dataset)
        persona_character = "Emma is an energetic fitness coach who loves morning workouts. She's cheerful and direct in her communication."

        print("PERSONA: Emma (persona_042)")
        print(f"Character: {persona_character}")
        print()

        # STEP 1: Emma's training history (what we learned about her)
        print("=" * 100)
        print("STEP 1: INDEX EMMA'S TRAINING HISTORY")
        print("=" * 100)
        print()

        emma_history = [
            {
                'query': 'Good morning! Turn on the lights',
                'response': 'Good morning Emma! Turning the lights to full brightness for your morning workout!',
                'context': {'time': 'morning', 'weather': 'sunny'}
            },
            {
                'query': 'I need music for my workout',
                'response': 'You got it! Playing your high-energy playlist on the speaker at volume 80.',
                'context': {'time': 'morning', 'weather': 'rainy'}
            },
            {
                'query': 'Set the AC cooler please',
                'response': 'Setting AC to 68 degrees in cool mode. Perfect for your workout!',
                'context': {'time': 'morning', 'weather': 'hot', 'temperature': 75}
            },
            {
                'query': 'Turn off everything, I\'m done',
                'response': 'Great workout Emma! Turning off lights and speaker. AC staying on at 72.',
                'context': {'time': 'noon', 'weather': 'sunny'}
            },
        ]

        print("Emma's training interactions:")
        for i, interaction in enumerate(emma_history, 1):
            print(f"\n{i}. User: {interaction['query']}")
            print(f"   Assistant: {interaction['response']}")
            print(f"   Context: {interaction['context']}")

        # Encode all queries for retrieval
        print("\n→ Encoding all queries with sentence-transformers...")
        for interaction in emma_history:
            interaction['embedding'] = self.encoder.encode(interaction['query'])

        print("→ Emma's history is now searchable!")
        print()

        # STEP 2: New query comes in
        print("=" * 100)
        print("STEP 2: NEW QUERY FROM EMMA")
        print("=" * 100)
        print()

        new_query = "Turn on the lights and start some music"
        new_context = {'time': 'morning', 'weather': 'cloudy'}

        print(f"Query: {new_query}")
        print(f"Context: {new_context}")
        print()

        # STEP 3: Retrieve similar past interactions
        print("=" * 100)
        print("STEP 3: RETRIEVE SIMILAR PAST INTERACTIONS")
        print("=" * 100)
        print()

        # Encode new query
        new_embedding = self.encoder.encode(new_query)

        # Calculate similarities
        print("Calculating similarity to each past interaction:")
        print()

        similarities = []
        for i, interaction in enumerate(emma_history, 1):
            similarity = np.dot(new_embedding, interaction['embedding']) / (
                np.linalg.norm(new_embedding) * np.linalg.norm(interaction['embedding'])
            )
            similarities.append((i, interaction, similarity))
            print(f"{i}. '{interaction['query'][:50]}...'")
            print(f"   Similarity: {similarity:.4f}")
            print()

        # Get top-k most similar
        k = 2
        top_k = sorted(similarities, key=lambda x: x[2], reverse=True)[:k]

        print(f"→ Retrieving top-{k} most similar interactions:")
        for rank, (idx, interaction, sim) in enumerate(top_k, 1):
            print(f"\n{rank}. User: {interaction['query']}")
            print(f"   You: {interaction['response']}")
            print(f"   Similarity: {sim:.4f}")

        print()

        # STEP 4: Build augmented prompt
        print("=" * 100)
        print("STEP 4: BUILD AUGMENTED PROMPT")
        print("=" * 100)
        print()

        # WITHOUT RAG (baseline unified model)
        print("WITHOUT RAG (Unified Model):")
        print("-" * 100)
        baseline_prompt = f"""System: You are a helpful smart home assistant.

User Profile: {persona_character}

Current context: time: morning, weather: cloudy

User: {new_query}
Assistant:"""
        print(baseline_prompt)
        print()

        # WITH RAG (augmented with history)
        print("WITH RAG (Unified Model + Retrieved History):")
        print("-" * 100)
        rag_prompt = f"""System: You are a helpful smart home assistant.

User Profile: {persona_character}

Relevant past interactions with this user:
1. User: {top_k[0][1]['query']}
   You: {top_k[0][1]['response']}

2. User: {top_k[1][1]['query']}
   You: {top_k[1][1]['response']}

Current context: time: morning, weather: cloudy

User: {new_query}
Assistant:"""
        print(rag_prompt)
        print()

        # STEP 5: Generate responses
        print("=" * 100)
        print("STEP 5: COMPARE RESPONSES")
        print("=" * 100)
        print()

        print("EXPECTED BASELINE RESPONSE (without RAG):")
        print("-" * 100)
        print("I'll turn on the lights and start playing music for you.")
        print()
        print("→ Generic, doesn't capture Emma's preferences")
        print()

        print("EXPECTED RAG RESPONSE (with retrieved history):")
        print("-" * 100)
        print("Good morning Emma! Turning the lights to full brightness and playing your high-energy playlist at volume 80. Perfect for your morning workout!")
        print()
        print("→ Captures Emma's preferences:")
        print("  • Full brightness (learned from interaction 1)")
        print("  • High-energy playlist (learned from interaction 2)")
        print("  • Volume 80 (learned from interaction 2)")
        print("  • Enthusiastic tone matching Emma's energy")
        print()


    def show_why_it_works(self):
        """
        Explain why RAG might beat personalized LoRA
        """

        print("=" * 100)
        print("WHY RAG BEATS PERSONALIZED LORA")
        print("=" * 100)
        print()

        comparison = [
            ("Training data", "Per-Persona LoRA", "RAG"),
            ("Examples per user", "30 (training set only)", "30 (can use all of them!)"),
            ("Overfitting risk", "HIGH (updating weights)", "NONE (no weight updates)"),
            ("Adaptation speed", "Slow (need to retrain)", "Instant (just add to index)"),
            ("Transparency", "Black box (can't see why)", "Transparent (see what's retrieved)"),
            ("Model size", "Need 200 adapters", "ONE unified model"),
            ("Update cost", "Retrain ($$$)", "Add to index (free)"),
            ("Context window", "Limited by model", "Can retrieve ANY amount"),
        ]

        print(f"{'Aspect':<25} | {'Per-Persona LoRA':<30} | {'RAG':<40}")
        print("-" * 100)
        for row in comparison:
            if len(row) == 3:
                print(f"{row[0]:<25} | {row[1]:<30} | {row[2]:<40}")

        print()
        print("KEY ADVANTAGES:")
        print("-" * 100)
        print("1. NO OVERFITTING: RAG doesn't change weights, so can't overfit to 30 examples")
        print("2. STRONG BASE: Uses your best model (unified LoRA at 82.1%)")
        print("3. MORE DATA: Can use all training examples as context, not just for training")
        print("4. DYNAMIC: Can retrieve different amounts (k=1,3,5) based on query")
        print("5. SCALABLE: As user gets more data, just add to index (no retraining)")
        print()


    def show_implementation_details(self):
        """
        Show key implementation details
        """

        print("=" * 100)
        print("IMPLEMENTATION DETAILS")
        print("=" * 100)
        print()

        print("1. INDEXING (done once during setup)")
        print("-" * 100)
        print("""
# For each persona, encode their training interactions
user_memories = {}

for dialogue in training_dialogues:
    persona_id = dialogue['persona_id']

    for user_msg, assistant_msg in dialogue['messages']:
        # Encode user query for retrieval
        embedding = encoder.encode(user_msg)

        # Store in memory
        user_memories[persona_id].append({
            'query': user_msg,
            'response': assistant_msg,
            'embedding': embedding,
            'context': dialogue['meta']
        })

# Result: user_memories['persona_042'] has ~200 indexed interactions
        """)
        print()

        print("2. RETRIEVAL (done for each query)")
        print("-" * 100)
        print("""
def retrieve_similar(persona_id, new_query, k=3):
    # Encode new query
    query_emb = encoder.encode(new_query)

    # Get this user's memories
    memories = user_memories[persona_id]

    # Calculate cosine similarities
    similarities = [
        cosine_sim(query_emb, mem['embedding'])
        for mem in memories
    ]

    # Get top-k indices
    top_k_indices = argsort(similarities)[-k:]

    # Return top-k memories
    return [memories[i] for i in top_k_indices]
        """)
        print()

        print("3. GENERATION (same model, augmented context)")
        print("-" * 100)
        print("""
def generate_with_rag(persona_id, query, context, k=3):
    # Retrieve similar past interactions
    similar = retrieve_similar(persona_id, query, k=k)

    # Build prompt with retrieved examples
    prompt = build_prompt(
        character=persona_info[persona_id],
        similar_interactions=similar,
        current_query=query,
        current_context=context
    )

    # Generate with unified model (NO personalized weights!)
    response = unified_model.generate(prompt)

    return response
        """)
        print()

        print("4. HYPERPARAMETERS TO TUNE")
        print("-" * 100)
        print("• k (retrieval count): Try k=0,1,3,5,10")
        print("  - k=0: No RAG (baseline)")
        print("  - k=1: Minimal context")
        print("  - k=3: Good balance (RECOMMENDED)")
        print("  - k=5: More context")
        print("  - k=10: Risk of too much noise")
        print()
        print("• Retrieval method:")
        print("  - Semantic similarity (CURRENT)")
        print("  - Recency (recent interactions)")
        print("  - Hybrid (weighted combination)")
        print()
        print("• Context format:")
        print("  - Full examples (CURRENT)")
        print("  - Summarized preferences")
        print("  - Just the responses")
        print()


if __name__ == '__main__':
    explainer = RAGExplainer()

    print("\n\n")
    explainer.example_scenario()

    print("\n\n")
    explainer.show_why_it_works()

    print("\n\n")
    explainer.show_implementation_details()

    print("\n\n")
    print("=" * 100)
    print("EXPECTED RESULTS")
    print("=" * 100)
    print()
    print("Based on similar work in literature, RAG typically provides:")
    print("• +2-6% improvement in semantic similarity")
    print("• +5-10% improvement in persona consistency")
    print("• Better performance on edge cases (unusual queries)")
    print()
    print("For your dataset:")
    print("• Baseline unified: 82.1%")
    print("• Expected with RAG (k=3): 84-87%")
    print("• Improvement: +2-5%")
    print()
    print("To run: python scripts/retrieval_augmented_baseline.py")
    print()
