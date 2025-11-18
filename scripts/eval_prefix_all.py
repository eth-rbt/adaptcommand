"""Quick script to evaluate the trained global prefix on test set"""
import json
import sys
sys.path.append('.')

from pathlib import Path
from train_prefix_all import evaluate_with_generation, load_all_split_data
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "models/prefix_all/checkpoint-73320"
base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"Loading model from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map=None,
    attn_implementation="eager"
)
base_model = base_model.to(device)

# Load prefix adapter
model = PeftModel.from_pretrained(base_model, model_path)
print("[OK] Model loaded")

# Load test data
print("\nLoading test data...")
test_dialogues = load_all_split_data(
    Path("data/cleaned/dialogs_clean.jsonl"),
    Path("data/splits/edgesplits.json"),
    "test"
)
print(f"[OK] Loaded {len(test_dialogues)} test dialogues")

# Evaluate
print("\nEvaluating on test set...")
system_prompt = "You are a helpful and personalized smart home assistant."
test_metrics = evaluate_with_generation(
    model=model,
    tokenizer=tokenizer,
    dialogues=test_dialogues,
    system_prompt=system_prompt,
    device=device,
    include_persona=True,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9
)

print("\n" + "="*60)
print("TEST METRICS")
print("="*60)
for key, value in test_metrics.items():
    print(f"  {key:30s}: {value:.4f}")

# Save results
output_dir = Path("models/prefix_all")
with open(output_dir / "test_metrics.json", "w") as f:
    json.dump(test_metrics, f, indent=2)

print(f"\n[OK] Test metrics saved to {output_dir / 'test_metrics.json'}")
