"""
Detailed Analysis of Numerical Parameter Errors

Runs evaluation and saves examples of numerical errors.
"""

import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import sys

# Add scripts directory to path
sys.path.append("scripts")
from action_metrics import ActionExtractor, ActionMetrics

def load_model(model_path):
    """Load model and tokenizer"""
    base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    print(f"Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print(f"Loading LoRA adapter from {model_path}...")
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()

    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=256):
    """Generate a response from the model"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

def format_prompt(dialogue_history, context):
    """Format the prompt for the model"""
    system_msg = "You are a helpful smart home assistant. Help the user control their devices by understanding their preferences and the current context."

    prompt = system_msg + "\n\n"

    # Add context if available
    if context:
        prompt += "Current device states:\n"
        for device, state in context.items():
            prompt += f"- {device}: {state}\n"
        prompt += "\n"

    # Add conversation history
    prompt += "Conversation:\n"
    for turn in dialogue_history:
        prompt += f"User: {turn['user']}\n"
        if 'assistant' in turn:
            prompt += f"Assistant: {turn['assistant']}\n"

    return prompt

def main():
    # Load unified LoRA model
    model_path = "models/lora_unified"
    model, tokenizer = load_model(model_path)

    # Load test data
    print("\nLoading test data...")
    dataset = load_dataset("json", data_files="data/cleaned/dialogs_clean.jsonl")["train"]

    with open("data/splits/edgesplits.json") as f:
        splits = json.load(f)

    test_indices = splits["test"]
    test_data = dataset.select(test_indices)

    # Action extractor
    extractor = ActionExtractor()

    # Collect error examples
    numerical_errors = []
    missing_numerical = []
    hallucinated_numerical = []

    print(f"\nAnalyzing {len(test_data)} test examples...")

    for idx, example in enumerate(test_data):
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{len(test_data)}")

        # Get dialogue and reference
        dialogue = example["dialogue"]
        if not dialogue:
            continue

        # Build context and history
        history = []
        context = example.get("context", {})

        for turn_idx, turn in enumerate(dialogue[:-1]):
            history.append(turn)

        # Last turn is what we're predicting
        user_input = dialogue[-1]["user"]
        reference = dialogue[-1]["assistant"]

        # Generate prediction
        prompt = format_prompt(history, context)
        prompt += f"User: {user_input}\nAssistant:"

        try:
            prediction = generate_response(model, tokenizer, prompt)
        except Exception as e:
            print(f"Error generating for example {idx}: {e}")
            continue

        # Extract actions
        pred_actions = extractor.extract_actions(prediction)
        ref_actions = extractor.extract_actions(reference)

        # Find numerical parameter errors
        numerical_params = {"temperature", "volume", "brightness", "fan_speed", "alarm_volume"}

        for device in ref_actions.keys() | pred_actions.keys():
            ref_params = ref_actions.get(device, {})
            pred_params = pred_actions.get(device, {})

            for param in ref_params.keys() | pred_params.keys():
                if param not in numerical_params:
                    continue

                ref_val = ref_params.get(param)
                pred_val = pred_params.get(param)

                if ref_val is not None and pred_val is not None:
                    # Check if incorrect
                    if abs(ref_val - pred_val) > 1:
                        numerical_errors.append({
                            'example_idx': idx,
                            'persona_id': example.get('persona_id', 'unknown'),
                            'device': device,
                            'param': param,
                            'ref_value': ref_val,
                            'pred_value': pred_val,
                            'error': abs(ref_val - pred_val),
                            'user_input': user_input,
                            'reference': reference,
                            'prediction': prediction
                        })
                elif ref_val is not None and pred_val is None:
                    # Missing numerical parameter
                    missing_numerical.append({
                        'example_idx': idx,
                        'persona_id': example.get('persona_id', 'unknown'),
                        'device': device,
                        'param': param,
                        'ref_value': ref_val,
                        'user_input': user_input,
                        'reference': reference,
                        'prediction': prediction
                    })
                elif ref_val is None and pred_val is not None:
                    # Hallucinated numerical parameter
                    hallucinated_numerical.append({
                        'example_idx': idx,
                        'persona_id': example.get('persona_id', 'unknown'),
                        'device': device,
                        'param': param,
                        'pred_value': pred_val,
                        'user_input': user_input,
                        'reference': reference,
                        'prediction': prediction
                    })

        # Stop after 500 examples for now (full test set is 5936)
        if idx >= 500:
            break

    # Save results
    output_dir = Path("results/unified")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "numerical_errors_analysis.json", "w") as f:
        json.dump({
            "numerical_errors": numerical_errors,
            "missing_numerical": missing_numerical,
            "hallucinated_numerical": hallucinated_numerical
        }, f, indent=2)

    # Print summary
    print("\n" + "="*80)
    print("NUMERICAL ERROR ANALYSIS SUMMARY")
    print("="*80)
    print(f"Total examples analyzed: 500")
    print(f"Incorrect numerical values: {len(numerical_errors)}")
    print(f"Missing numerical parameters: {len(missing_numerical)}")
    print(f"Hallucinated numerical parameters: {len(hallucinated_numerical)}")

    if numerical_errors:
        print(f"\n{'='*80}")
        print("TOP 10 WORST NUMERICAL ERRORS")
        print('='*80)

        sorted_errors = sorted(numerical_errors, key=lambda x: x['error'], reverse=True)

        for i, err in enumerate(sorted_errors[:10], 1):
            print(f"\n{i}. Error: {err['error']} | {err['device']}.{err['param']}")
            print(f"   Expected: {err['ref_value']}, Predicted: {err['pred_value']}")
            print(f"   User: {err['user_input']}")
            print(f"   Ref:  {err['reference'][:100]}...")
            print(f"   Pred: {err['prediction'][:100]}...")

    if missing_numerical:
        print(f"\n{'='*80}")
        print("EXAMPLES OF MISSING NUMERICAL PARAMETERS")
        print('='*80)

        for i, err in enumerate(missing_numerical[:5], 1):
            print(f"\n{i}. Missing: {err['device']}.{err['param']} = {err['ref_value']}")
            print(f"   User: {err['user_input']}")
            print(f"   Ref:  {err['reference']}")
            print(f"   Pred: {err['prediction']}")

    print(f"\nDetailed results saved to: {output_dir / 'numerical_errors_analysis.json'}")

if __name__ == "__main__":
    main()
