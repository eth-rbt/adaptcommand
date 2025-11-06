# Quick Start — Running Benchmarks

Your data is now prepared and ready for benchmarking! Here's how to run a quick evaluation.

## Data Preparation Complete ✓

You have:
- **10,000 dialogues** from **200 personas**
- **Train/Val/Test splits** (60/20/20 per persona)
- **Cold-start test set** for evaluation

## Running Your First Benchmark

### 1. Activate Environment

```bash
source venv/bin/activate
pip install -r requirements.txt  # if not already installed
```

### 2. Quick Test (50 examples, ~2-5 minutes)

```bash
python scripts/run_baseline_benchmark.py \
  --config configs/baseline_v1.0.json \
  --max_examples 50 \
  --output_dir results/baseline_quick
```

This will:
- Load a small 0.5B model (Qwen2.5-0.5B-Instruct)
- Evaluate on 50 test examples
- Compute ROUGE scores and embedding similarity
- Save results to `results/baseline_quick/`

### 3. Full Evaluation (2000 test examples, ~30-60 minutes)

```bash
python scripts/run_baseline_benchmark.py \
  --config configs/baseline_v1.0.json \
  --output_dir results/baseline
```

## Expected Metrics

The benchmark computes **two key metrics**:

### 1. Semantic Similarity (Embedding-based)
- Measures how semantically similar the response is to the reference
- Uses sentence-transformers to encode both texts
- Cosine similarity between embeddings
- **Range**: 0-1 (higher is better)

### 2. Action Accuracy (Task-specific)
- Extracts device actions from assistant responses
- Compares predicted actions vs reference actions
- **Metrics computed**:
  - `device_precision`: % of predicted devices that are correct
  - `device_recall`: % of reference devices that were predicted
  - `param_precision`: % of predicted parameters that are correct
  - `param_recall`: % of reference parameters that were predicted
  - `param_f1`: F1 score for parameter accuracy
- **Range**: 0-1 (higher is better)

**Example:**
- Reference: "Setting the AC to 22 degrees in heat mode"
- Good prediction: "I'll set the AC to 22 in heat mode" ✓
  - High semantic similarity
  - High action accuracy (temperature=22, mode=heat)
- Bad prediction: "Sure, I can help with that" ✗
  - Medium semantic similarity
  - Low action accuracy (no actions extracted)

## Model Configuration

Edit `configs/baseline_v1.0.json` to:
- Change the model (try different alternatives listed)
- Adjust generation parameters (temperature, top_p, etc.)
- Modify prompt template
- Toggle metrics

### Alternative Small Models

Quick models for iteration (edit in config):
```json
{
  "model": {
    "name": "Qwen/Qwen2.5-0.5B-Instruct",  // Fastest (0.5B)
    "alternatives": [
      "Qwen/Qwen2.5-1.5B-Instruct",         // Better quality (1.5B)
      "TinyLlama/TinyLlama-1.1B-Chat-v1.0", // Alternative (1.1B)
      "microsoft/phi-2"                      // Good quality (2.7B)
    ]
  }
}
```

## Results

After running, check:
- `results/baseline/baseline_results.json` — Full metrics
- `results/baseline/sample_outputs.jsonl` — 20 example predictions

## Next Steps

Once you have baseline results:

1. **Phase C**: Train global adapters (LoRA/Prefix)
   - See `plan.md` Phase C for details

2. **Phase D**: Train per-persona adapters
   - Personalize for each of the 200 users

3. **Analysis**: Compare personalized vs baseline
   - Which personas benefit most?
   - How much improvement per user?

## Data Structure

Each dialogue has:
```json
{
  "persona_id": "persona_042",
  "session_id": 7,
  "character": "Emma is an energetic fitness coach...",
  "routines": [...],  // Smart home preferences
  "meta": {           // Context
    "time_of_day": "morning",
    "weather": "sunny",
    ...
  },
  "messages": [       // Conversation
    {"role": "user", "text": "..."},
    {"role": "assistant", "text": "..."}
  ]
}
```

The model predicts assistant responses given:
- User message
- Context (time, weather, etc.)
- Conversation history

## Troubleshooting

### Out of Memory
- Reduce `--max_examples`
- Use smaller model (0.5B)
- Reduce `max_new_tokens` in config

### Slow Evaluation
- Use GPU if available (CUDA/MPS)
- Reduce `--max_examples` for testing
- Use batch processing (future enhancement)

### Model Download Issues
- Check internet connection
- Ensure you have ~1-2GB free disk space
- Models download from HuggingFace automatically

## Questions?

See:
- `plan.md` — Full implementation roadmap
- `design.md` — Project design and goals
- `scripts/README_PHASE_A.md` — Data preparation details
