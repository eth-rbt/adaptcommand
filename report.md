# AdaptCommand: Personalized Smart Home Assistant

## Project Overview

AdaptCommand is a research project focused on building and evaluating personalized smart home assistants that can adapt to diverse user personas. The project explores how well language models can learn and respond to individual user preferences across various contexts (weather, time of day, etc.) when controlling smart home devices.

## Problem Statement

Traditional smart home assistants often provide generic responses without considering individual user preferences, behavioral patterns, or personality traits. This project investigates:

1. Can small language models learn to personalize their responses based on rich user personas?
2. How does model performance vary across different types of users?
3. What aspects of personalization (device selection, numerical parameters, categorical choices) are most challenging?

## Dataset

### Personas
- **200 unique personas** with detailed character descriptions
- Each persona includes:
  - Personality traits and speaking style
  - Lifestyle preferences and hobbies
  - Home environment characteristics
  - Context-specific routines (preferences based on weather, time, temperature, etc.)

**Example persona:**
> "Ethan is a reserved librarian with a penchant for mystery novels. He speaks softly, often pausing to choose his words carefully. At home, he enjoys brewing herbal tea and solving crossword puzzles by the fireplace, surrounded by shelves of well-loved books."

### Dialogues
- **10,000 total dialogues** (50 per persona)
- Multi-turn conversations between user and smart home assistant
- Average conversation length: ~7-14 turns
- Users request control of smart home devices based on their context and preferences

### Smart Home Devices
The system controls 5 device types with various parameters:
1. **TV**: volume, brightness, input source
2. **AC**: temperature, mode (heat/cool), fan speed
3. **Lights**: brightness, color, mode
4. **Speaker**: volume, equalizer settings
5. **Security**: armed status, alarm volume

### Data Splits
- **Train**: 6,000 dialogues (60%)
- **Validation**: 2,000 dialogues (20%)
- **Test**: 2,000 dialogues (20%)
- All splits maintain persona-level stratification (30/10/10 dialogues per persona)

## Approach

### Model Configuration
**Baseline Model**: Qwen/Qwen2.5-0.5B-Instruct
- Small (0.5B parameters) for fast iteration
- Generation parameters:
  - Max new tokens: 256
  - Temperature: 0.7
  - Top-p: 0.9
  - Repetition penalty: 1.1

### System Prompt
The model is given a system prompt:
> "You are a helpful smart home assistant. Help the user control their devices by understanding their preferences and the current context."

Each dialogue includes:
- The persona's character description
- Current environmental context (time, weather, temperature, etc.)
- Conversation history
- The persona's learned routines/preferences

### Evaluation Metrics

The evaluation framework measures multiple aspects of model performance:

#### 1. Embedding Similarity
- Compares semantic similarity between predicted and reference responses
- Uses sentence-transformers (all-MiniLM-L6-v2)
- **Baseline mean**: 0.638
- Range: 0.0 to 1.0 (higher is better)

#### 2. Device-Level Metrics
- **Precision**: 0.785 (Are activated devices correct?)
- **Recall**: 0.791 (Are all needed devices activated?)
- **Accuracy**: 0.941 (Overall device selection correctness)
- Per-device breakdown for TV, AC, lights, speaker, security

#### 3. Parameter-Level Metrics

**Numerical Parameters** (e.g., temperature, volume, brightness):
- **Precision**: 0.202 (% of predicted numbers that exactly match reference)
- **Recall**: 0.208
- **MAE**: 27.93 (Mean absolute error for numerical values)
- This is particularly challenging - suggests models struggle with exact numerical values

**Categorical Parameters** (e.g., AC mode, light color, input source):
- **Precision**: 0.386
- **Recall**: 0.375
- More challenging than device selection but easier than numerical parameters

## Current Findings

### 1. Per-Persona Performance Analysis

We analyzed performance variation across all 200 personas to understand how model effectiveness differs by user type.

#### Embedding Similarity Distribution
- **Mean**: 0.6364
- **Std Dev**: 0.0523
- **Range**: 0.517 to 0.812
- **Best performer**: Persona 96 (Remy) - 0.812
- **Worst performer**: Persona 150 (Walter) - 0.517

**Key insight**: The relatively narrow standard deviation (0.052) suggests the model achieves reasonably consistent semantic quality across diverse personas. However, the 0.3 range indicates some personas are significantly more challenging than others.

#### Numerical Precision Distribution
- **Mean**: 0.2089
- **Std Dev**: 0.0985
- **Range**: 0.0 to 0.604

**Key insight**: Much higher variance (std ~0.1) and overall low performance indicate numerical parameter prediction is the most challenging aspect. Some personas show nearly 0% exact-match numerical accuracy, while others reach 60%.

#### Comparison to Baseline
- Per-persona embedding similarity mean (0.6364) is slightly below aggregate baseline (0.6379)
- Per-persona numerical precision mean (0.2089) is slightly above aggregate baseline (0.2020)
- This suggests aggregation slightly smooths out per-persona variations

### 2. Persona Diversity Analysis

We embedded all 200 persona descriptions using the same sentence-transformer model and performed PCA to understand the semantic structure of our persona set.

#### PCA Results
- **Original dimension**: 384 (embedding model output)
- **PC1 variance**: 6.29%
- **PC2 variance**: 3.26%
- **Top 2 components**: 9.55% total variance
- **50% variance threshold**: Requires 24 components
- **All 50 components**: Only 74.63% of variance

**Key insight**: The extremely low variance explained by the first few components indicates that personas are **highly diverse and multi-dimensional**. They cannot be reduced to a few dominant archetypes or traits. This is a positive characteristic for evaluating generalization.

#### Component Importance
The variance is distributed relatively evenly across many components rather than concentrated in a few:
- PC1: 6.29%
- PC2: 3.26%
- PC3: 2.93%
- PC4: 2.77%
- PC5: 2.55%
- ... gradually decreasing

This smooth distribution confirms rich, multi-faceted diversity rather than clustering around a few persona types.

### 3. Performance vs. Persona Characteristics

When visualizing the 2D PCA projection colored by embedding similarity performance:

**Key finding**: High and low performers are **scattered throughout the embedding space** with no clear clustering pattern. This suggests:
- Model difficulty is not correlated with the primary semantic dimensions of personas
- No single "type" of persona is systematically easier or harder
- Performance variations stem from complex, multi-dimensional factors
- The model doesn't just struggle with one particular personality type or lifestyle

### 4. Visualization Gallery

Three key visualizations were generated:

#### A. Per-Persona Histogram Comparison
- Dual histogram showing distribution of embedding similarity and numerical precision
- Compares per-persona distributions to aggregate baseline
- Reveals that numerical precision has much wider variation than embedding similarity

#### B. 2D Persona Embedding Space
- PCA projection of all 200 personas
- Color-coded by model performance (embedding similarity)
- Shows diverse coverage of semantic space
- Best/worst performers annotated

#### C. PCA Component Importance
- Bar chart of explained variance per component (top 20)
- Cumulative variance curve showing components needed for 50%, 80%, 95% thresholds
- Demonstrates the high-dimensional nature of persona diversity

## Key Challenges Identified

### 1. Numerical Precision
- **Current**: 20.2% exact-match accuracy
- The model struggles to predict exact numerical values (temperature, volume, brightness)
- High MAE (27.93) suggests predictions can be quite far from target
- This is the weakest performance area

### 2. Categorical Parameter Selection
- **Current**: 38.6% precision, 37.5% recall
- Better than numerical but still challenging
- Includes choices like AC mode, light color, equalizer settings

### 3. Persona-Specific Variation
- 30% spread in embedding similarity (0.517 to 0.812)
- Some personas are inherently more difficult to model
- Factors causing difficulty are multi-dimensional and not easily categorized

## Strengths of Current Approach

### 1. Device Selection
- 94.1% accuracy in choosing which devices to activate
- 78.5% precision, 79.1% recall
- The model has learned the general device-interaction patterns well

### 2. Semantic Quality
- 63.8% embedding similarity shows reasonable response quality
- Responses are generally coherent and contextually appropriate
- Speaking style and tone reasonably match personas

### 3. Dataset Quality
- 200 truly diverse personas spanning high-dimensional semantic space
- Rich contextual information (weather, time, routines)
- Balanced splits with good coverage

## Future Directions

### Model Improvements
1. **Numerical value prediction**: Explore regression heads or specialized numerical reasoning
2. **Larger models**: Test if scaling improves numerical precision and categorical accuracy
3. **Fine-tuning strategies**: Per-user adaptation, few-shot learning, or LoRA approaches
4. **Structured outputs**: Enforce JSON/structured format to improve parameter accuracy

### Evaluation Enhancements
1. **Soft numerical metrics**: Allow tolerance ranges (e.g., ±2 degrees) rather than exact match
2. **User preference alignment**: Measure adherence to persona routines and preferences
3. **Multi-turn consistency**: Track if model maintains persona characteristics across conversation
4. **Contextual appropriateness**: Evaluate if suggestions match environmental context

### Data Augmentation
1. **Error analysis**: Deep dive into worst-performing personas to identify patterns
2. **Contrastive examples**: Generate similar contexts with different expected outcomes
3. **Synthetic expansion**: Create additional dialogues for challenging scenarios

## Conclusion

The AdaptCommand project demonstrates that small language models (0.5B parameters) can learn general smart home interaction patterns with reasonable accuracy (94% device selection). However, **precise personalization remains challenging**, particularly for:
- Exact numerical values (20% accuracy)
- Categorical parameter selection (38% precision)
- Consistent performance across diverse personas (30% variation)

The **high-dimensional diversity** of personas (requiring 24+ components for 50% variance) suggests the personalization problem is inherently complex. The scattered distribution of high/low performers in embedding space indicates no simple categorization of "easy" vs "hard" personas.

These findings motivate future research into:
- Specialized architectures for numerical reasoning
- Adaptive learning strategies that personalize beyond the generic system prompt
- Hybrid approaches combining retrieval of user preferences with generation

---

**Generated**: 2025-11-09
**Model**: Qwen/Qwen2.5-0.5B-Instruct
**Dataset**: 200 personas, 10,000 dialogues
**Test Set Size**: 2,000 dialogues (20%)
