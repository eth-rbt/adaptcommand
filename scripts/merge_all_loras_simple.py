"""
Simple Average LoRA Merging

Merge ALL 200 per-persona LoRAs into ONE model.

Simplest possible merging strategy:
merged_weights = (LoRA_000 + LoRA_001 + ... + LoRA_199) / 200

Expected: +1-3% over unified baseline
Time: 10-15 minutes
"""

import json
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import copy


def merge_all_loras_simple():
    """
    Merge all 200 per-persona LoRAs into one averaged model.
    """

    print("=" * 80)
    print("SIMPLE AVERAGE: MERGING ALL 200 LORA ADAPTERS")
    print("=" * 80)

    # Find all persona LoRAs
    lora_base = Path('models/lora_adapters')
    lora_paths = sorted([p for p in lora_base.iterdir() if p.is_dir() and p.name.startswith('persona_')])

    print(f"\nFound {len(lora_paths)} persona LoRA adapters")

    if len(lora_paths) == 0:
        print("ERROR: No LoRA adapters found in models/lora_adapters/")
        print("Make sure you've trained per-persona LoRAs first!")
        return

    # Load base model
    print("\nLoading base model...")
    base_model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

    # Load first LoRA to get structure
    print(f"\nLoading first LoRA: {lora_paths[0]}")
    merged_model = PeftModel.from_pretrained(base_model, lora_paths[0])

    # Get adapter weights
    merged_state = merged_model.state_dict()

    # Find LoRA parameters
    lora_keys = [k for k in merged_state.keys() if 'lora' in k.lower()]
    print(f"Found {len(lora_keys)} LoRA parameters")

    # Initialize averaged weights with zeros
    avg_state = {}
    for key in lora_keys:
        avg_state[key] = torch.zeros_like(merged_state[key])

    # Accumulate all LoRA weights
    print(f"\nAccumulating weights from {len(lora_paths)} adapters...")
    successful_count = 0

    for lora_path in tqdm(lora_paths):
        try:
            # Load this LoRA
            model = PeftModel.from_pretrained(copy.deepcopy(base_model), lora_path)
            state = model.state_dict()

            # Add to average
            for key in lora_keys:
                if key in state:
                    avg_state[key] += state[key].float()

            successful_count += 1

            # Clean up
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            print(f"\nError loading {lora_path.name}: {e}")
            continue

    # Divide by count to get average
    print(f"\nComputing average of {successful_count} adapters...")
    for key in lora_keys:
        avg_state[key] = avg_state[key] / successful_count

    # Update merged model with averaged weights
    print("Updating model with merged weights...")
    merged_state_dict = merged_model.state_dict()
    for key in lora_keys:
        merged_state_dict[key] = avg_state[key]

    merged_model.load_state_dict(merged_state_dict)

    # Save
    output_path = Path('models/lora_merged_simple')
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving merged adapter to {output_path}")
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Save merge info
    with open(output_path / 'merge_info.json', 'w') as f:
        json.dump({
            'merge_method': 'simple_average',
            'num_merged': successful_count,
            'total_attempted': len(lora_paths),
            'description': 'Simple average of all 200 per-persona LoRA adapters'
        }, f, indent=2)

    print(f"\n{'=' * 80}")
    print("MERGING COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nMerged {successful_count} adapters into one model")
    print(f"Saved to: {output_path}")
    print("\nNext steps:")
    print("  python scripts/eval_merged_simple.py")
    print("\nExpected performance: 83-85% (+1-3% over unified)")


if __name__ == '__main__':
    merge_all_loras_simple()
