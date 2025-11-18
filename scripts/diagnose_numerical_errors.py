"""
Diagnose Numerical Parameter Errors in Unified LoRA

Analyzes where numerical parameter predictions go wrong.
"""

import json
import re
from pathlib import Path
from collections import defaultdict

def extract_numbers(text):
    """Extract all numbers from text"""
    # Find all numbers (integers and decimals)
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    return [float(n) for n in numbers]

def extract_parameters(text):
    """Extract device parameters from text"""
    params = {}

    # Temperature (AC)
    temp_match = re.search(r'(\d+)\s*degree', text, re.IGNORECASE)
    if temp_match:
        params['temperature'] = int(temp_match.group(1))

    # Volume (TV, speaker)
    vol_match = re.search(r'volume\s*(?:to|at|is)?\s*(\d+)', text, re.IGNORECASE)
    if vol_match:
        params['volume'] = int(vol_match.group(1))

    # Brightness (TV, lights)
    bright_match = re.search(r'brightness\s*(?:to|at|is)?\s*(\d+)', text, re.IGNORECASE)
    if bright_match:
        params['brightness'] = int(bright_match.group(1))

    # Fan speed
    fan_match = re.search(r'fan\s*speed\s*(?:to|at|is)?\s*(\d+)', text, re.IGNORECASE)
    if fan_match:
        params['fan_speed'] = int(fan_match.group(1))

    return params

def analyze_errors():
    """Analyze numerical parameter errors"""

    # Load sample outputs
    samples_path = Path("results/unified/sample_outputs.jsonl")

    if not samples_path.exists():
        print(f"Error: {samples_path} not found")
        return

    print("="*80)
    print("NUMERICAL PARAMETER ERROR ANALYSIS")
    print("="*80)

    errors = []
    correct = []

    with open(samples_path) as f:
        for line_num, line in enumerate(f):
            data = json.loads(line)
            pred = data['prediction']
            ref = data['reference']

            pred_params = extract_parameters(pred)
            ref_params = extract_parameters(ref)

            # Compare parameters
            for param_name in set(list(pred_params.keys()) + list(ref_params.keys())):
                pred_val = pred_params.get(param_name)
                ref_val = ref_params.get(param_name)

                if ref_val is not None:  # Only check if reference has this parameter
                    if pred_val is None:
                        # Missing parameter
                        errors.append({
                            'index': data.get('index', line_num),
                            'persona': data.get('persona_id', 'unknown'),
                            'param': param_name,
                            'ref_val': ref_val,
                            'pred_val': None,
                            'error': 'MISSING',
                            'prediction': pred,
                            'reference': ref
                        })
                    elif pred_val != ref_val:
                        # Incorrect value
                        errors.append({
                            'index': data.get('index', line_num),
                            'persona': data.get('persona_id', 'unknown'),
                            'param': param_name,
                            'ref_val': ref_val,
                            'pred_val': pred_val,
                            'error': abs(pred_val - ref_val),
                            'prediction': pred,
                            'reference': ref
                        })
                    else:
                        # Correct
                        correct.append({
                            'param': param_name,
                            'value': ref_val
                        })

    # Print summary
    print(f"\nTotal parameters checked: {len(errors) + len(correct)}")
    print(f"Correct: {len(correct)} ({len(correct)*100/(len(errors)+len(correct)+0.001):.1f}%)")
    print(f"Errors: {len(errors)} ({len(errors)*100/(len(errors)+len(correct)+0.001):.1f}%)")

    # Error breakdown by type
    missing = [e for e in errors if e['error'] == 'MISSING']
    value_errors = [e for e in errors if e['error'] != 'MISSING']

    print(f"\nError Breakdown:")
    print(f"  Missing parameters: {len(missing)}")
    print(f"  Incorrect values: {len(value_errors)}")

    # Parameter-wise breakdown
    param_errors = defaultdict(list)
    for e in errors:
        param_errors[e['param']].append(e)

    print(f"\nErrors by Parameter Type:")
    for param, errs in sorted(param_errors.items()):
        print(f"  {param}: {len(errs)} errors")

    # Show worst examples
    print(f"\n{'='*80}")
    print("WORST ERROR EXAMPLES")
    print('='*80)

    # Sort by error magnitude
    value_errors_sorted = sorted([e for e in value_errors if isinstance(e['error'], (int, float))],
                                  key=lambda x: x['error'], reverse=True)

    for i, err in enumerate(value_errors_sorted[:10], 1):
        print(f"\nExample {i}:")
        print(f"  Index: {err['index']}, Persona: {err['persona']}")
        print(f"  Parameter: {err['param']}")
        print(f"  Expected: {err['ref_val']}, Predicted: {err['pred_val']}, Error: {err['error']}")
        print(f"  Reference: {err['reference']}")
        print(f"  Prediction: {err['prediction']}")

    # Show missing parameter examples
    if missing:
        print(f"\n{'='*80}")
        print("MISSING PARAMETER EXAMPLES")
        print('='*80)

        for i, err in enumerate(missing[:5], 1):
            print(f"\nExample {i}:")
            print(f"  Index: {err['index']}, Persona: {err['persona']}")
            print(f"  Missing parameter: {err['param']} (expected: {err['ref_val']})")
            print(f"  Reference: {err['reference']}")
            print(f"  Prediction: {err['prediction']}")

    # Error magnitude distribution
    print(f"\n{'='*80}")
    print("ERROR MAGNITUDE DISTRIBUTION")
    print('='*80)

    if value_errors:
        error_mags = [e['error'] for e in value_errors if isinstance(e['error'], (int, float))]
        if error_mags:
            print(f"  Mean error: {sum(error_mags)/len(error_mags):.2f}")
            print(f"  Max error: {max(error_mags)}")
            print(f"  Min error: {min(error_mags)}")

            # Histogram
            bins = [1, 5, 10, 20, 50, 100]
            for i in range(len(bins)):
                if i == 0:
                    count = sum(1 for e in error_mags if e <= bins[i])
                    print(f"  Error <= {bins[i]}: {count}")
                else:
                    count = sum(1 for e in error_mags if bins[i-1] < e <= bins[i])
                    print(f"  Error {bins[i-1]}-{bins[i]}: {count}")
            count = sum(1 for e in error_mags if e > bins[-1])
            print(f"  Error > {bins[-1]}: {count}")

if __name__ == "__main__":
    analyze_errors()
