"""
Action Extraction and Comparison Metrics for Smart Home Assistant

Extracts device actions from assistant responses and compares them.
"""

import re
from typing import Dict, List, Set, Tuple
import json


class ActionExtractor:
    """Extract device actions from smart home assistant responses."""

    # Device types and their parameters
    DEVICES = {
        "tv": ["volume", "brightness", "input_source", "input source"],
        "ac": ["temperature", "mode", "fan_speed", "fan speed"],
        "lights": ["brightness", "color", "mode"],
        "speaker": ["volume", "equalizer"],
        "security": ["armed", "alarm_volume", "alarm volume"]
    }

    # Common action patterns
    PATTERNS = {
        # Setting actions
        "set": r"(?:set|setting|adjust|adjusting|change|changing)\s+(?:the\s+)?(\w+)",
        "to": r"(\w+)\s+(?:to|at)\s+([0-9]+|[a-z\s]+)",

        # Turning on/off
        "turn_on": r"turn(?:ing)?\s+on\s+(?:the\s+)?(\w+)",
        "turn_off": r"turn(?:ing)?\s+off\s+(?:the\s+)?(\w+)",

        # Numbers and values
        "number": r"([0-9]+)\s*(%|degrees?|percent)?",
        "mode": r"(heat|cool|auto|static|dynamic)\s+mode",
    }

    def extract_actions(self, text: str) -> Dict[str, Dict[str, any]]:
        """
        Extract device actions from text.

        Returns:
            Dict mapping device -> {parameter: value}
        """
        text = text.lower()
        actions = {}

        # Extract TV actions
        tv_actions = self._extract_tv(text)
        if tv_actions:
            actions["tv"] = tv_actions

        # Extract AC actions
        ac_actions = self._extract_ac(text)
        if ac_actions:
            actions["ac"] = ac_actions

        # Extract lights actions
        lights_actions = self._extract_lights(text)
        if lights_actions:
            actions["lights"] = lights_actions

        # Extract speaker actions
        speaker_actions = self._extract_speaker(text)
        if speaker_actions:
            actions["speaker"] = speaker_actions

        # Extract security actions
        security_actions = self._extract_security(text)
        if security_actions:
            actions["security"] = security_actions

        return actions

    def _extract_tv(self, text: str) -> Dict[str, any]:
        """Extract TV-related actions."""
        actions = {}

        # Check for TV mention
        if "tv" not in text and "television" not in text:
            return actions

        # Volume
        vol_match = re.search(r"(?:tv\s+)?volume\s+(?:to|at)\s+([0-9]+)", text)
        if vol_match:
            actions["volume"] = int(vol_match.group(1))

        # Brightness
        bright_match = re.search(r"(?:tv\s+)?brightness\s+(?:to|at)\s+([0-9]+)", text)
        if bright_match:
            actions["brightness"] = int(bright_match.group(1))

        # Input source
        input_match = re.search(r"input\s+source\s+(?:to|is)\s+(\w+)", text)
        if input_match:
            actions["input_source"] = input_match.group(1).capitalize()

        # Turn on/off
        if re.search(r"turn(?:ing)?\s+on\s+(?:the\s+)?tv", text):
            actions["power"] = "on"
        elif re.search(r"turn(?:ing|ed)?\s+off\s+(?:the\s+)?tv", text):
            actions["power"] = "off"

        return actions

    def _extract_ac(self, text: str) -> Dict[str, any]:
        """Extract AC-related actions."""
        actions = {}

        # Check for AC mention
        if "ac" not in text and "air condition" not in text and "heat" not in text and "cool" not in text:
            return actions

        # Temperature
        temp_match = re.search(r"(?:temperature\s+(?:to|at)\s+)?([0-9]+)\s*degrees?", text)
        if temp_match:
            actions["temperature"] = int(temp_match.group(1))

        # Mode
        for mode in ["heat", "cool", "auto"]:
            if f"{mode} mode" in text or f"mode to {mode}" in text or f"set to {mode}" in text:
                actions["mode"] = mode
                break

        # Fan speed
        fan_match = re.search(r"fan\s+speed\s+(?:to|at)\s+([0-9]+)", text)
        if fan_match:
            actions["fan_speed"] = int(fan_match.group(1))

        # Turn on/off
        if re.search(r"turn(?:ing)?\s+on\s+(?:the\s+)?ac", text):
            actions["power"] = "on"
        elif re.search(r"turn(?:ing|ed)?\s+off\s+(?:the\s+)?ac", text):
            actions["power"] = "off"

        return actions

    def _extract_lights(self, text: str) -> Dict[str, any]:
        """Extract lights-related actions."""
        actions = {}

        # Check for lights mention
        if "light" not in text and "lamp" not in text:
            return actions

        # Brightness
        bright_match = re.search(r"(?:brightness\s+(?:to|at)\s+)?([0-9]+)\s*(?:%|percent)", text)
        if bright_match:
            actions["brightness"] = int(bright_match.group(1))

        # Color
        for color in ["warm", "cool", "neutral", "white", "red", "blue", "green"]:
            if f"{color} color" in text or f"color to {color}" in text:
                actions["color"] = color
                break

        # Mode
        for mode in ["static", "dynamic"]:
            if f"{mode} mode" in text or f"mode to {mode}" in text:
                actions["mode"] = mode
                break

        # Turn on/off
        if re.search(r"turn(?:ing)?\s+on\s+(?:the\s+)?light", text):
            actions["power"] = "on"
        elif re.search(r"turn(?:ing|ed)?\s+off\s+(?:the\s+)?light", text):
            actions["power"] = "off"

        return actions

    def _extract_speaker(self, text: str) -> Dict[str, any]:
        """Extract speaker-related actions."""
        actions = {}

        # Check for speaker mention
        if "speaker" not in text and "music" not in text and "audio" not in text:
            return actions

        # Volume
        vol_match = re.search(r"(?:speaker\s+)?volume\s+(?:to|at)\s+([0-9]+)", text)
        if vol_match:
            actions["volume"] = int(vol_match.group(1))

        # Equalizer
        for eq in ["balanced", "bass boost", "treble boost", "bass", "treble"]:
            if eq in text:
                actions["equalizer"] = eq
                break

        # Turn on/off
        if re.search(r"turn(?:ing)?\s+on\s+(?:the\s+)?speaker", text):
            actions["power"] = "on"
        elif re.search(r"turn(?:ing|ed)?\s+off\s+(?:the\s+)?speaker", text):
            actions["power"] = "off"

        return actions

    def _extract_security(self, text: str) -> Dict[str, any]:
        """Extract security-related actions."""
        actions = {}

        # Check for security mention
        if "security" not in text and "alarm" not in text and "arm" not in text:
            return actions

        # Armed status
        if "arm" in text and "disarm" not in text:
            actions["armed"] = True
        elif "disarm" in text:
            actions["armed"] = False

        # Alarm volume
        alarm_match = re.search(r"alarm\s+volume\s+(?:to|at)\s+([0-9]+)", text)
        if alarm_match:
            actions["alarm_volume"] = int(alarm_match.group(1))

        return actions


class ActionMetrics:
    """Compute action-based metrics."""

    @staticmethod
    def compare_actions(pred_actions: Dict, ref_actions: Dict) -> Dict[str, float]:
        """
        Compare predicted actions against reference actions.

        Returns metrics:
        - action_precision: % of predicted actions that are correct
        - action_recall: % of reference actions that were predicted
        - action_f1: F1 score
        - device_accuracy: % of devices correctly identified
        - parameter_accuracy: % of parameters correctly set
        - numerical_* metrics: Confusion matrix for numerical parameters
        """
        # Get all devices mentioned
        pred_devices = set(pred_actions.keys())
        ref_devices = set(ref_actions.keys())

        # Device-level metrics
        if len(ref_devices) > 0:
            device_recall = len(pred_devices & ref_devices) / len(ref_devices)
        else:
            device_recall = 1.0 if len(pred_devices) == 0 else 0.0

        if len(pred_devices) > 0:
            device_precision = len(pred_devices & ref_devices) / len(pred_devices)
        else:
            device_precision = 1.0 if len(ref_devices) == 0 else 0.0

        # Parameter-level metrics with detailed tracking
        total_ref_params = 0
        correct_params = 0
        total_pred_params = 0

        # Track numerical vs categorical parameters separately
        numerical_tp = 0  # Correct numerical value
        numerical_fp = 0  # Predicted numerical value incorrectly
        numerical_fn = 0  # Missing numerical value
        numerical_tn = 0  # Correctly didn't predict

        categorical_tp = 0
        categorical_fp = 0
        categorical_fn = 0
        categorical_tn = 0

        # Track numerical errors
        numerical_errors = []  # List of (ref_value, pred_value, error)

        # Define which parameters are numerical
        numerical_params = {
            "temperature", "volume", "brightness", "fan_speed",
            "alarm_volume", "speed", "level"
        }

        for device in ref_devices | pred_devices:
            ref_params = ref_actions.get(device, {})
            pred_params = pred_actions.get(device, {})

            total_ref_params += len(ref_params)
            total_pred_params += len(pred_params)

            # Check all possible parameters for this device
            all_params = set(ref_params.keys()) | set(pred_params.keys())

            for param in all_params:
                is_numerical = param in numerical_params
                in_ref = param in ref_params
                in_pred = param in pred_params

                if in_ref and in_pred:
                    ref_value = ref_params[param]
                    pred_value = pred_params[param]

                    if ActionMetrics._values_match(ref_value, pred_value):
                        correct_params += 1
                        if is_numerical:
                            numerical_tp += 1
                        else:
                            categorical_tp += 1
                    else:
                        if is_numerical:
                            numerical_fp += 1
                            # Track the error
                            if isinstance(ref_value, (int, float)) and isinstance(pred_value, (int, float)):
                                error = abs(ref_value - pred_value)
                                numerical_errors.append((ref_value, pred_value, error))
                        else:
                            categorical_fp += 1

                elif in_ref and not in_pred:
                    # Missing parameter (FN)
                    if is_numerical:
                        numerical_fn += 1
                    else:
                        categorical_fn += 1

                elif not in_ref and in_pred:
                    # Hallucinated parameter (already counted in FP above if matched badly)
                    # This case is when we predict something that shouldn't exist
                    if is_numerical:
                        numerical_fp += 1
                    else:
                        categorical_fp += 1

        # Compute metrics
        if total_ref_params > 0:
            param_recall = correct_params / total_ref_params
        else:
            param_recall = 1.0 if total_pred_params == 0 else 0.0

        if total_pred_params > 0:
            param_precision = correct_params / total_pred_params
        else:
            param_precision = 1.0 if total_ref_params == 0 else 0.0

        if param_precision + param_recall > 0:
            param_f1 = 2 * param_precision * param_recall / (param_precision + param_recall)
        else:
            param_f1 = 0.0

        # Numerical parameter metrics
        numerical_precision = 0.0
        numerical_recall = 0.0
        numerical_f1 = 0.0

        if numerical_tp + numerical_fp > 0:
            numerical_precision = numerical_tp / (numerical_tp + numerical_fp)

        if numerical_tp + numerical_fn > 0:
            numerical_recall = numerical_tp / (numerical_tp + numerical_fn)

        if numerical_precision + numerical_recall > 0:
            numerical_f1 = 2 * numerical_precision * numerical_recall / (numerical_precision + numerical_recall)

        # Categorical parameter metrics
        categorical_precision = 0.0
        categorical_recall = 0.0
        categorical_f1 = 0.0

        if categorical_tp + categorical_fp > 0:
            categorical_precision = categorical_tp / (categorical_tp + categorical_fp)

        if categorical_tp + categorical_fn > 0:
            categorical_recall = categorical_tp / (categorical_tp + categorical_fn)

        if categorical_precision + categorical_recall > 0:
            categorical_f1 = 2 * categorical_precision * categorical_recall / (categorical_precision + categorical_recall)

        # Mean absolute error for numerical parameters
        mae = sum(err for _, _, err in numerical_errors) / len(numerical_errors) if numerical_errors else 0.0

        return {
            "device_precision": device_precision,
            "device_recall": device_recall,
            "param_precision": param_precision,
            "param_recall": param_recall,
            "param_f1": param_f1,
            # Numerical parameter metrics
            "numerical_tp": numerical_tp,
            "numerical_fp": numerical_fp,
            "numerical_fn": numerical_fn,
            "numerical_precision": numerical_precision,
            "numerical_recall": numerical_recall,
            "numerical_f1": numerical_f1,
            "numerical_mae": mae,  # Mean absolute error
            # Categorical parameter metrics
            "categorical_tp": categorical_tp,
            "categorical_fp": categorical_fp,
            "categorical_fn": categorical_fn,
            "categorical_precision": categorical_precision,
            "categorical_recall": categorical_recall,
            "categorical_f1": categorical_f1,
        }

    @staticmethod
    def _values_match(ref_value, pred_value) -> bool:
        """Check if two parameter values match (with tolerance)."""
        # Convert to same type for comparison
        if isinstance(ref_value, (int, float)) and isinstance(pred_value, (int, float)):
            # Numeric: allow small tolerance
            return abs(ref_value - pred_value) <= 1
        elif isinstance(ref_value, str) and isinstance(pred_value, str):
            # String: case-insensitive match
            return ref_value.lower().strip() == pred_value.lower().strip()
        elif isinstance(ref_value, bool) and isinstance(pred_value, bool):
            # Boolean: exact match
            return ref_value == pred_value
        else:
            # Try string comparison as fallback
            return str(ref_value).lower() == str(pred_value).lower()


# Example usage and testing
if __name__ == "__main__":
    extractor = ActionExtractor()

    # Test examples
    examples = [
        "Setting the AC to 22 degrees in heat mode with fan speed 1.",
        "I'll turn on the lights at 50% brightness with warm color.",
        "Adjusting the speaker volume to 30 with balanced equalizer.",
        "Setting the TV volume to 25 and brightness to 40.",
    ]

    print("="*60)
    print("ACTION EXTRACTION EXAMPLES")
    print("="*60)

    for text in examples:
        print(f"\nText: {text}")
        actions = extractor.extract_actions(text)
        print(f"Actions: {json.dumps(actions, indent=2)}")

    # Test comparison
    print("\n" + "="*60)
    print("ACTION COMPARISON EXAMPLE")
    print("="*60)

    ref_actions = {
        "ac": {"temperature": 22, "mode": "heat", "fan_speed": 1},
        "lights": {"brightness": 50, "color": "warm"}
    }

    pred_actions = {
        "ac": {"temperature": 22, "mode": "heat"},  # Missing fan_speed
        "lights": {"brightness": 50, "color": "warm"}
    }

    metrics = ActionMetrics.compare_actions(pred_actions, ref_actions)
    print(f"\nReference: {ref_actions}")
    print(f"Prediction: {pred_actions}")
    print(f"\nMetrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")
