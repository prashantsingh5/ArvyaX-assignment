# ArvyaX Emotional State Prediction System

**From Understanding Humans → To Guiding Them**

A machine learning system that goes beyond prediction to understand emotional state, reason under uncertainty, and provide meaningful guidance.

## Overview

This system analyzes user reflections from immersive sessions (forest, ocean, rain, mountain, café) combined with contextual metadata to:

1. **Understand** emotional state and intensity
2. **Decide** meaningful next actions
3. **Guide** users toward better mental states
4. **Reason** under uncertainty with confidence awareness

## Key Features

- **Emotional State Classification**: Predicts 6 emotional states (calm, focused, mixed, neutral, overwhelmed, restless)
- **Intensity Estimation**: Predicts emotional intensity on a scale of 1-5
- **Decision Engine**: Recommends specific actions (breathing, journaling, deep work, etc.) and optimal timing
- **Uncertainty Modeling**: Provides confidence scores and uncertainty flags
- **Robustness**: Handles messy, short, and contradictory inputs
- **Edge-Ready**: Lightweight model suitable for on-device deployment

## Project Structure

```
assignmennt/
├── Sample_arvyax_reflective_dataset.xlsx    # Training data (1200 samples)
├── arvyax_test_inputs_120.xlsx              # Test data (120 samples)
├── src/
│   ├── train_model.py                       # Main training pipeline
│   ├── ablation_study.py                    # Text vs metadata analysis
│   ├── error_analysis.py                    # Failure case analysis
│   ├── feature_importance.py                # Feature importance analysis
│   └── explore_data.py                      # Data exploration
├── models/                                  # Trained model files
├── outputs/
│   ├── predictions.csv                      # Test predictions
│   ├── ablation_study.csv                   # Ablation results
│   └── feature_importance.csv               # Feature importance scores
├── docs/
│   ├── ERROR_ANALYSIS.md                    # Detailed error analysis
│   └── EDGE_PLAN.md                         # Edge deployment plan
└── README.md                                # This file
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- 2GB RAM minimum
- Windows/Linux/Mac

### Installation

```bash
# Clone or navigate to project directory
cd assignmennt

# Install required packages
pip install pandas numpy scikit-learn xgboost openpyxl joblib nltk

# Download NLTK data (automatic on first run)
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

## How to Run

### 1. Train Models & Generate Predictions

```bash
python src/train_model.py
```

**Output:**
- Trained models saved to `models/`
- Test predictions saved to `outputs/predictions.csv`

**Predictions include:**
- `predicted_state`: Emotional state (calm, focused, mixed, etc.)
- `predicted_intensity`: Intensity level (1-5)
- `confidence`: Prediction confidence (0-1)
- `uncertain_flag`: Binary uncertainty indicator
- `what_to_do`: Recommended action (breathing, journaling, deep_work, etc.)
- `when_to_do`: Optimal timing (now, within_15_min, later_today, tonight, tomorrow_morning)

### 2. Run Ablation Study

```bash
python src/ablation_study.py
```

**Output:** Compares performance of:
- Text-only model
- Metadata-only model
- Combined (Text + Metadata) model

Results saved to `outputs/ablation_study.csv`

### 3. Run Error Analysis

```bash
python src/error_analysis.py
```

**Output:**
- Analyzes 10+ failure cases
- Identifies patterns in errors
- Provides improvement recommendations
- Report saved to `docs/ERROR_ANALYSIS.md`

### 4. Run Feature Importance Analysis

```bash
python src/feature_importance.py
```

**Output:**
- Feature importance scores
- Text vs metadata contribution
- Results saved to `outputs/feature_importance.csv`

## Approach

### 1. Feature Engineering

#### Text Features (TF-IDF)
- Extracted from journal reflections
- Max 100 features (lightweight for edge deployment)
- Unigrams + bigrams
- Captures emotional language patterns

#### Metadata Features
- **Numerical**: sleep_hours, energy_level, stress_level, duration_min
- **Categorical**: ambience_type, time_of_day, previous_day_mood, face_emotion_hint, reflection_quality
- **Engineered**:
  - `sleep_energy_ratio`: Sleep divided by energy
  - `stress_energy_gap`: Difference between stress and energy
  - `session_intensity`: Duration × energy
  - `well_rested`: Binary flag (sleep ≥ 7 hours)
  - `exhausted`: Binary flag (low energy + high stress)
  - `text_length`, `word_count`, `is_very_short`: Text engagement indicators

### 2. Model Choice

**XGBoost** was chosen for both classification and regression tasks because:
- ✓ Handles mixed feature types (text + metadata)
- ✓ Built-in feature importance
- ✓ Robust to missing values
- ✓ Efficient for edge deployment
- ✓ Strong performance on small datasets
- ✓ Handles noisy labels well

**Alternative Considered**: Random Forest (similar performance but XGBoost is faster)

### 3. Intensity Prediction: Classification or Regression?

**Approach Used**: **Regression (with rounding to integers)**

**Reasoning:**
- Intensity has natural ordering (1 < 2 < 3 < 4 < 5)
- Regression preserves ordinal relationships
- Can predict fractional values (e.g., 3.7) which get rounded
- More appropriate than treating as independent classes

**Alternative**: Ordinal classification could be explored for future improvements

### 4. Decision Engine Logic

The decision engine uses rule-based logic considering:

- **Emotional State**: Current predicted state
- **Intensity**: Severity level
- **Stress Level**: High stress triggers calming actions
- **Energy Level**: Low energy suggests rest/movement
- **Time of Day**: Morning → planning, Evening → rest
- **Confidence**: Low confidence → gentler actions

**Example Rules:**
- High stress + Low energy + Evening → `rest`, `now`
- Positive state + High energy + Morning → `deep_work`, `now`
- Negative state + High intensity → `grounding`, `now`
- Low confidence → `journaling`, `within_15_min`

### 5. Uncertainty Modeling

**Confidence Score**:
- Maximum probability from state prediction
- Range: 0-1 (higher is more confident)

**Uncertain Flag**:
- Set to 1 if:
  - Confidence < 0.6 OR
  - Input text ≤ 3 words
- Indicates when the system is unsure
- Used to decide when to ask clarifying questions

## Performance

### Model Metrics (Full Training Data)

- **State Classification Accuracy**: 90.7%
- **Intensity Prediction RMSE**: 1.093

### Ablation Study Results

| Configuration | State Accuracy | Intensity RMSE |
|--------------|----------------|----------------|
| Text Only | 51.7% | 1.50 |
| Metadata Only | 19.6% | 1.52 |
| Text + Metadata | 49.6% | 1.49 |

**Key Finding**: Text features dominate (91.5% importance for state, 84.3% for intensity), but metadata still contributes, especially for intensity prediction.

### Feature Importance

**For State Prediction:**
- Text: 91.5%
- Metadata: 8.5%

**For Intensity Prediction:**
- Text: 84.3%
- Metadata: 15.7%

Most important metadata features:
- `stress_energy_gap`
- `session_intensity`
- `text_length`

## Robustness

### Handling Edge Cases

**1. Very Short Text ("ok", "fine")**
- System relies more heavily on metadata
- Uncertainty flag set to 1
- Lower confidence scores
- Gentler actions recommended (e.g., journaling to gather more info)

**2. Missing Values**
- Numerical features: Filled with median
- Categorical features: Replaced with "unknown" category
- Graceful handling of unseen categories in test data

**3. Contradictory Inputs**
- Example: "Great session!" but high stress, low energy
- Decision engine considers ALL signals
- Stress and energy override positive text when severe
- Uncertainty flag helps identify conflicting cases

## Error Analysis Summary

**Main Failure Patterns:**

1. **Short Text Problem** (3 cases)
   - Inputs with ≤3 words lack sufficient signal
   - Model forced to over-rely on metadata

2. **Conflicting Signals** (multiple cases)
   - Text and metadata contradict each other
   - Model struggles to weight appropriately

3. **Low Confidence Cases** (15 cases)
   - Often correlated with actual errors
   - Uncertainty flag is predictive of failures

4. **Intensity Calibration** (15 cases)
   - Fine-grained 1-5 scale is subjective
   - Ordinal regression could improve

5. **Label Noise Suspected**
   - Some labels may be incorrect or debatable
   - Emotional state labeling is inherently subjective

**See [docs/ERROR_ANALYSIS.md](docs/ERROR_ANALYSIS.md) for detailed analysis of 10+ failure cases.**

## Improvement Recommendations

### Immediate Actions
1. Add minimum text length requirement
2. Implement calibrated uncertainty estimation
3. Add text-metadata interaction features
4. Use ordinal regression for intensity

### Data Collection
1. Collect more short-but-meaningful examples
2. Add confidence ratings to labels
3. Gather more examples of rare states
4. Label conflicting-signal cases explicitly

### Model Improvements
1. Ensemble methods for robustness
2. Multi-task learning (joint state + intensity)
3. Attention mechanisms for dynamic weighting
4. Label smoothing for noisy labels

## Edge Deployment

The system is designed for on-device deployment. See [docs/EDGE_PLAN.md](docs/EDGE_PLAN.md) for:
- Model size optimization
- Latency considerations
- Mobile deployment strategy
- Quantization approaches
- On-device vs cloud tradeoffs

**Current Model Size**: ~5MB (lightweight enough for mobile)

## Team

**ArvyaX · RevoltronX**
Building AI systems that truly understand and guide humans.

## License

This project is part of an internship assignment for ArvyaX.

---

**Note**: This system prioritizes real-world robustness over perfect accuracy. It knows when it's uncertain and makes safe, helpful decisions even with noisy, incomplete data.
