# Device Precision and Numerical Precision Examples

## Example 1: Perfect Match

**User Query**: "Set the AC to 22 degrees and turn on the lights to 50% brightness"

**Reference**: "Setting the AC to 22 degrees in heat mode and adjusting the lights to 50% brightness with a warm color."

**Prediction**: "I'll set the AC to 22 degrees and the lights to 50 percent brightness."

### Parsing Process

#### 1. Device Extraction

**Reference devices**:
- AC ✓
- lights ✓

**Predicted devices**:
- AC ✓
- lights ✓

**Device Precision Calculation**:
```
Device Precision = (Correct devices mentioned) / (Total devices mentioned)
                 = 2 / 2
                 = 1.00 (100%)
```

#### 2. Numerical Value Extraction

**Reference numbers**:
- 22 (degrees for AC)
- 50 (brightness for lights)

**Predicted numbers**:
- 22 (degrees for AC) ✓
- 50 (brightness for lights) ✓

**Numerical Precision Calculation**:
```
Numerical Precision = (Correct numbers) / (Total numbers in reference)
                    = 2 / 2
                    = 1.00 (100%)
```

**Result**: Perfect match on both metrics!

---

## Example 2: Wrong Number

**User Query**: "Set the AC to 22 degrees"

**Reference**: "Setting the AC to 22 degrees in heat mode."

**Prediction**: "I'll set the AC to 20 degrees in heat mode."

### Parsing Process

#### 1. Device Extraction

**Reference devices**:
- AC ✓

**Predicted devices**:
- AC ✓

**Device Precision**: 1 / 1 = **100%** ✓

#### 2. Numerical Value Extraction

**Reference numbers**:
- 22 (degrees for AC)

**Predicted numbers**:
- 20 (degrees for AC) ✗ WRONG!

**Numerical Precision**: 0 / 1 = **0%** ✗

**Embedding Similarity**: Still high (~78%) because the semantic meaning is close, but the numerical precision catches the error!

---

## Example 3: Missing Device

**User Query**: "Turn on the lights and AC"

**Reference**: "Setting the lights to 70% brightness and the AC to 22 degrees in heat mode."

**Prediction**: "I'll turn on the lights to 70% brightness."

### Parsing Process

#### 1. Device Extraction

**Reference devices**:
- lights ✓
- AC ✓

**Predicted devices**:
- lights ✓
- (AC missing) ✗

**Device Precision**: 1 / 1 = **100%**
(Only 1 device mentioned in prediction, and it's correct)

**Device Recall**: 1 / 2 = **50%**
(Only mentioned 1 out of 2 required devices)

**Note**: Device Precision is still 100% because we only count mentioned devices. Device Recall would catch this issue.

#### 2. Numerical Value Extraction

**Reference numbers**:
- 70 (brightness for lights)
- 22 (degrees for AC)

**Predicted numbers**:
- 70 (brightness for lights) ✓

**Numerical Precision**: 1 / 2 = **50%**
(Got the brightness right, but completely missed the AC temperature)

---

## Example 4: Hallucinated Device

**User Query**: "Turn on the lights"

**Reference**: "Setting the lights to 50% brightness with a warm color."

**Prediction**: "I'll set the lights to 50% brightness, turn on the TV to volume 30, and adjust the curtains."

### Parsing Process

#### 1. Device Extraction

**Reference devices**:
- lights ✓

**Predicted devices**:
- lights ✓
- TV ✗ (hallucinated - not in reference!)
- curtains ✗ (hallucinated - not in reference AND not a valid device!)

**Device Precision**: 1 / 3 = **33.3%** ✗
(Mentioned 3 devices, only 1 is correct)

This is BAD - the model is hallucinating devices!

#### 2. Numerical Value Extraction

**Reference numbers**:
- 50 (brightness for lights)

**Predicted numbers**:
- 50 (brightness for lights) ✓
- 30 (TV volume) - not in reference

**Numerical Precision**: 1 / 1 = **100%**
(We only count numbers that appear in reference, and it got the brightness right)

**Note**: Even though numerical precision is 100%, the Device Precision (33%) reveals the hallucination problem!

---

## Example 5: Multiple Parameters per Device

**User Query**: "Set up the bedroom for sleep"

**Reference**: "Setting the bedroom lights to 30% brightness with warm color, AC to 20 degrees in heat mode with fan speed 2, and arming the security system at volume 50."

**Prediction**: "I'll set the lights to 30 percent warm, AC to 20 degrees heat mode fan speed 1, and arm security at volume 50."

### Parsing Process

#### 1. Device Extraction

**Reference devices**:
- lights ✓
- AC ✓
- security ✓

**Predicted devices**:
- lights ✓
- AC ✓
- security ✓

**Device Precision**: 3 / 3 = **100%** ✓

#### 2. Numerical Value Extraction

**Reference numbers**:
- 30 (lights brightness)
- 20 (AC temperature)
- 2 (AC fan speed)
- 50 (security alarm volume)

**Predicted numbers**:
- 30 (lights brightness) ✓
- 20 (AC temperature) ✓
- 1 (AC fan speed) ✗ WRONG (predicted 1, should be 2)
- 50 (security volume) ✓

**Numerical Precision**: 3 / 4 = **75%**

**Analysis**: Got most parameters right, but fan speed is wrong!

---

## Example 6: Paraphrasing vs Numerical Precision

**Reference**: "Setting the AC to 22 degrees"

**Prediction A**: "Adjusting the air conditioner to 22 degrees"
- **Embedding Similarity**: 97.2% (paraphrasing captured!)
- **Device Precision**: 100% (AC = air conditioner)
- **Numerical Precision**: 100% (22 = 22)
- **Result**: Perfect!

**Prediction B**: "Adjusting the air conditioner to 20 degrees"
- **Embedding Similarity**: 78.4% (still reasonably high)
- **Device Precision**: 100% (AC = air conditioner)
- **Numerical Precision**: 0% (20 ≠ 22) ✗
- **Result**: Numerical precision catches the error that embedding similarity might miss!

---

## Parsing Implementation (Pseudocode)

```python
def extract_devices(text):
    """Extract mentioned devices from text"""
    devices = []
    device_keywords = {
        'lights': ['light', 'lights', 'lighting', 'lamp'],
        'AC': ['ac', 'air conditioner', 'temperature', 'heating', 'cooling'],
        'TV': ['tv', 'television', 'screen'],
        'speaker': ['speaker', 'audio', 'music', 'sound'],
        'security': ['security', 'alarm', 'armed']
    }

    text_lower = text.lower()
    for device, keywords in device_keywords.items():
        if any(kw in text_lower for kw in keywords):
            devices.append(device)

    return devices

def extract_numbers(text):
    """Extract numerical values from text"""
    import re
    # Match patterns like "22", "22 degrees", "50%", "50 percent"
    pattern = r'\b(\d+)\s*(?:degrees?|percent|%|°C|°F)?\b'
    matches = re.findall(pattern, text)
    return [int(m) for m in matches]

def calculate_device_precision(reference, prediction):
    """Calculate device precision"""
    ref_devices = set(extract_devices(reference))
    pred_devices = extract_devices(prediction)

    if len(pred_devices) == 0:
        return 0.0

    correct = sum(1 for d in pred_devices if d in ref_devices)
    return correct / len(pred_devices)

def calculate_numerical_precision(reference, prediction):
    """Calculate numerical precision"""
    ref_numbers = extract_numbers(reference)
    pred_numbers = extract_numbers(prediction)

    if len(ref_numbers) == 0:
        return 1.0  # No numbers to check

    correct = sum(1 for rn in ref_numbers if rn in pred_numbers)
    return correct / len(ref_numbers)
```

---

## Real Example from Unified LoRA Results

**Actual Results**:
- Embedding Similarity: 82.14%
- Device Precision: 93.80%
- Numerical Precision: 91.24%

**Interpretation**:
- **82.14% embedding similarity**: On average, predictions are semantically very close to references
- **93.80% device precision**: Model mentions the correct devices 93.8% of the time
- **91.24% numerical precision**: Model gets exact numbers right 91.2% of the time

**What this means**:
- ~6% of the time, the model mentions wrong or hallucinated devices
- ~9% of the time, the model gets numbers wrong (e.g., 20 instead of 22 degrees)
- Embedding similarity alone would miss these errors!

---

## Why Both Metrics Matter

### Example: High Embedding, Low Numerical

**Reference**: "Setting AC to 22 degrees"
**Prediction**: "Setting AC to 20 degrees"

- **Embedding Similarity**: 78% (semantically similar)
- **Numerical Precision**: 0% (wrong number!)

**Problem**: User asked for 22°C but got 20°C - uncomfortable!

### Example: High Embedding, Low Device

**Reference**: "Turn on the lights"
**Prediction**: "Turn on the lights and TV"

- **Embedding Similarity**: 85% (similar action)
- **Device Precision**: 50% (hallucinated TV)

**Problem**: User didn't ask for TV, but it turned on anyway!

### Example: All Metrics High

**Reference**: "Set AC to 22 degrees and lights to 50%"
**Prediction**: "Adjusting AC to 22 degrees and lights to 50 percent brightness"

- **Embedding Similarity**: 97%
- **Device Precision**: 100%
- **Numerical Precision**: 100%

**Result**: Perfect understanding and execution!

---

## Summary

**Device Precision**: Ensures the model doesn't:
- Hallucinate devices (turn on TV when not asked)
- Miss devices (forget to turn on AC)
- Mention wrong devices (turn on speaker instead of TV)

**Numerical Precision**: Ensures the model gets exact values right:
- Temperature (22°C not 20°C)
- Brightness (50% not 60%)
- Volume (30 not 40)

**Embedding Similarity**: Captures overall semantic meaning
- Robust to paraphrasing
- Measures response quality
- But can miss specific errors!

**All three together** give a complete picture of model performance.
