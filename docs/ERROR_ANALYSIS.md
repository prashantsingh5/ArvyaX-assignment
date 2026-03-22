# Error Analysis Report

## Overview

This document analyzes failure cases to understand model limitations and improvement opportunities.

## Error Category Summary

- **Ambiguous Text**: 4 cases
- **Very Short Input**: 3 cases
- **Low Confidence**: 15 cases
- **Intensity Mismatch**: 15 cases
- **State Confusion**: 15 cases

## Detailed Error Analysis (10+ Cases)

### Case 1

**ID**: 144

**Journal Text**: "For some reason I could finally concentrate so I sat with that feeling."

**True State**: focused (Intensity: 1)

**Predicted State**: neutral (Intensity: 3)

**Confidence**: 0.340

**Context**:
- Sleep: 4.0h
- Stress: 3/5
- Energy: 3/5
- Time: night
- Ambience: mountain

**What Went Wrong**:
- Model had low confidence, indicating uncertainty
- State misclassification: predicted neutral instead of focused
- Large intensity error (2)

**Why It Failed**:

**How to Improve**:
- Use ensemble methods or additional models for low-confidence cases
- Consider asking clarifying questions when confidence is low
- Improve intensity calibration with more training data
- Use ordinal regression instead of standard regression
- Collect more training examples similar to this case
- Add feature engineering to capture text-metadata interactions

---

### Case 2

**ID**: 485

**Journal Text**: "I guess mind was all over the place."

**True State**: neutral (Intensity: 1)

**Predicted State**: restless (Intensity: 3)

**Confidence**: 0.257

**Context**:
- Sleep: 4.0h
- Stress: 3/5
- Energy: 1/5
- Time: night
- Ambience: mountain

**What Went Wrong**:
- Model had low confidence, indicating uncertainty
- State misclassification: predicted restless instead of neutral
- Large intensity error (2)

**Why It Failed**:

**How to Improve**:
- Use ensemble methods or additional models for low-confidence cases
- Consider asking clarifying questions when confidence is low
- Improve intensity calibration with more training data
- Use ordinal regression instead of standard regression
- Collect more training examples similar to this case
- Add feature engineering to capture text-metadata interactions

---

### Case 3

**ID**: 525

**Journal Text**: "At first that helped a little."

**True State**: calm (Intensity: 1)

**Predicted State**: restless (Intensity: 3)

**Confidence**: 0.284

**Context**:
- Sleep: 6.0h
- Stress: 1/5
- Energy: 3/5
- Time: night
- Ambience: ocean

**What Went Wrong**:
- Model had low confidence, indicating uncertainty
- State misclassification: predicted restless instead of calm
- Large intensity error (2)

**Why It Failed**:

**How to Improve**:
- Use ensemble methods or additional models for low-confidence cases
- Consider asking clarifying questions when confidence is low
- Improve intensity calibration with more training data
- Use ordinal regression instead of standard regression
- Collect more training examples similar to this case
- Add feature engineering to capture text-metadata interactions

---

### Case 4

**ID**: 564

**Journal Text**: "kinda calm now"

**True State**: overwhelmed (Intensity: 1)

**Predicted State**: focused (Intensity: 3)

**Confidence**: 0.210

**Context**:
- Sleep: 5.0h
- Stress: 4/5
- Energy: 3/5
- Time: afternoon
- Ambience: rain

**What Went Wrong**:
- Very short input (≤3 words) provides limited signal
- Model had low confidence, indicating uncertainty
- State misclassification: predicted focused instead of overwhelmed
- Large intensity error (2)

**Why It Failed**:
- Insufficient text information forces model to rely heavily on metadata

**How to Improve**:
- Prompt users for more detailed reflections
- Add special handling for short inputs with higher uncertainty
- Use ensemble methods or additional models for low-confidence cases
- Consider asking clarifying questions when confidence is low
- Improve intensity calibration with more training data
- Use ordinal regression instead of standard regression
- Collect more training examples similar to this case
- Add feature engineering to capture text-metadata interactions

---

### Case 5

**ID**: 570

**Journal Text**: "mind was all over the place ..."

**True State**: calm (Intensity: 5)

**Predicted State**: restless (Intensity: 3)

**Confidence**: 0.229

**Context**:
- Sleep: 5.0h
- Stress: 2/5
- Energy: 2/5
- Time: night
- Ambience: forest

**What Went Wrong**:
- Model had low confidence, indicating uncertainty
- State misclassification: predicted restless instead of calm
- Large intensity error (2)

**Why It Failed**:

**How to Improve**:
- Use ensemble methods or additional models for low-confidence cases
- Consider asking clarifying questions when confidence is low
- Improve intensity calibration with more training data
- Use ordinal regression instead of standard regression
- Collect more training examples similar to this case
- Add feature engineering to capture text-metadata interactions

---

### Case 6

**ID**: 587

**Journal Text**: "Honestly got distracted again. ..."

**True State**: calm (Intensity: 5)

**Predicted State**: focused (Intensity: 3)

**Confidence**: 0.284

**Context**:
- Sleep: 5.0h
- Stress: 4/5
- Energy: 5/5
- Time: morning
- Ambience: ocean

**What Went Wrong**:
- Model had low confidence, indicating uncertainty
- State misclassification: predicted focused instead of calm
- Large intensity error (2)

**Why It Failed**:

**How to Improve**:
- Use ensemble methods or additional models for low-confidence cases
- Consider asking clarifying questions when confidence is low
- Improve intensity calibration with more training data
- Use ordinal regression instead of standard regression
- Collect more training examples similar to this case
- Add feature engineering to capture text-metadata interactions

---

### Case 7

**ID**: 600

**Journal Text**: "during the session helped me plan my day. later it changed felt good for a moment."

**True State**: restless (Intensity: 1)

**Predicted State**: focused (Intensity: 3)

**Confidence**: 0.311

**Context**:
- Sleep: 4.0h
- Stress: 1/5
- Energy: 2/5
- Time: evening
- Ambience: forest

**What Went Wrong**:
- Model had low confidence, indicating uncertainty
- State misclassification: predicted focused instead of restless
- Large intensity error (2)

**Why It Failed**:
- Ambiguous language ('ok', 'fine') can indicate various emotional states

**How to Improve**:
- Use ensemble methods or additional models for low-confidence cases
- Consider asking clarifying questions when confidence is low
- Improve intensity calibration with more training data
- Use ordinal regression instead of standard regression
- Collect more training examples similar to this case
- Add feature engineering to capture text-metadata interactions

---

### Case 8

**ID**: 609

**Journal Text**: "by the end felt good for a moment."

**True State**: focused (Intensity: 5)

**Predicted State**: restless (Intensity: 3)

**Confidence**: 0.262

**Context**:
- Sleep: 7.0h
- Stress: 2/5
- Energy: 3/5
- Time: morning
- Ambience: ocean

**What Went Wrong**:
- Model had low confidence, indicating uncertainty
- State misclassification: predicted restless instead of focused
- Large intensity error (2)

**Why It Failed**:
- Ambiguous language ('ok', 'fine') can indicate various emotional states

**How to Improve**:
- Use ensemble methods or additional models for low-confidence cases
- Consider asking clarifying questions when confidence is low
- Improve intensity calibration with more training data
- Use ordinal regression instead of standard regression
- Collect more training examples similar to this case
- Add feature engineering to capture text-metadata interactions

---

### Case 9

**ID**: 611

**Journal Text**: "that helped a little"

**True State**: overwhelmed (Intensity: 1)

**Predicted State**: calm (Intensity: 3)

**Confidence**: 0.326

**Context**:
- Sleep: 5.0h
- Stress: 3/5
- Energy: 1/5
- Time: afternoon
- Ambience: rain

**What Went Wrong**:
- Model had low confidence, indicating uncertainty
- State misclassification: predicted calm instead of overwhelmed
- Large intensity error (2)

**Why It Failed**:

**How to Improve**:
- Use ensemble methods or additional models for low-confidence cases
- Consider asking clarifying questions when confidence is low
- Improve intensity calibration with more training data
- Use ordinal regression instead of standard regression
- Collect more training examples similar to this case
- Add feature engineering to capture text-metadata interactions

---

### Case 10

**ID**: 614

**Journal Text**: "Honestly felt lighter, not fully though. ..."

**True State**: mixed (Intensity: 5)

**Predicted State**: restless (Intensity: 3)

**Confidence**: 0.308

**Context**:
- Sleep: 6.0h
- Stress: 2/5
- Energy: 5/5
- Time: night
- Ambience: cafe

**What Went Wrong**:
- Model had low confidence, indicating uncertainty
- State misclassification: predicted restless instead of mixed
- Large intensity error (2)

**Why It Failed**:
- Large stress-energy gap (3) creates conflicting signals

**How to Improve**:
- Use ensemble methods or additional models for low-confidence cases
- Consider asking clarifying questions when confidence is low
- Improve intensity calibration with more training data
- Use ordinal regression instead of standard regression
- Collect more training examples similar to this case
- Add feature engineering to capture text-metadata interactions

---

### Case 11

**ID**: 673

**Journal Text**: "it was fine"

**True State**: restless (Intensity: 1)

**Predicted State**: focused (Intensity: 3)

**Confidence**: 0.259

**Context**:
- Sleep: 5.0h
- Stress: 4/5
- Energy: 2/5
- Time: night
- Ambience: rain

**What Went Wrong**:
- Very short input (≤3 words) provides limited signal
- Model had low confidence, indicating uncertainty
- State misclassification: predicted focused instead of restless
- Large intensity error (2)
- Conflicting signals: high stress + low energy

**Why It Failed**:
- Insufficient text information forces model to rely heavily on metadata
- Ambiguous language ('ok', 'fine') can indicate various emotional states

**How to Improve**:
- Prompt users for more detailed reflections
- Add special handling for short inputs with higher uncertainty
- Use ensemble methods or additional models for low-confidence cases
- Consider asking clarifying questions when confidence is low
- Improve intensity calibration with more training data
- Use ordinal regression instead of standard regression
- Collect more training examples similar to this case
- Add feature engineering to capture text-metadata interactions

---

### Case 12

**ID**: 705

**Journal Text**: "back to normal after"

**True State**: calm (Intensity: 1)

**Predicted State**: restless (Intensity: 3)

**Confidence**: 0.310

**Context**:
- Sleep: 4.0h
- Stress: 4/5
- Energy: 3/5
- Time: morning
- Ambience: ocean

**What Went Wrong**:
- Model had low confidence, indicating uncertainty
- State misclassification: predicted restless instead of calm
- Large intensity error (2)

**Why It Failed**:

**How to Improve**:
- Use ensemble methods or additional models for low-confidence cases
- Consider asking clarifying questions when confidence is low
- Improve intensity calibration with more training data
- Use ordinal regression instead of standard regression
- Collect more training examples similar to this case
- Add feature engineering to capture text-metadata interactions

---

## Key Insights

### Main Failure Patterns

1. **Short Text Problem**: Inputs with ≤3 words provide insufficient signal
   - Model forced to over-rely on metadata
   - Ambiguous words like 'ok', 'fine' can mean many things

2. **Conflicting Signals**: When text and metadata contradict
   - Example: User writes 'great session' but has high stress, low energy
   - Model struggles to weight text vs metadata appropriately

3. **Low Confidence Cases**: Model uncertainty is informative
   - Low confidence often correlates with actual errors
   - Uncertainty flag is critical for downstream decisions

4. **Intensity Calibration**: Harder than state classification
   - Fine-grained intensity (1-5) is subjective and noisy
   - May benefit from ordinal regression or better loss function

5. **Label Noise**: Some labels may be incorrect or debatable
   - Human emotional state labeling is inherently subjective
   - Inter-annotator agreement would be valuable

## Improvement Recommendations

### Immediate Actions
1. Add minimum text length requirement or prompt for details
2. Implement calibrated uncertainty estimation
3. Add text-metadata interaction features
4. Use ordinal regression for intensity prediction

### Data Collection
1. Collect more examples of:
   - Short but meaningful reflections
   - Cases with conflicting signals
   - Rare emotional states
2. Add confidence ratings to labels (how sure is the annotator?)

### Model Improvements
1. Ensemble methods for robustness
2. Multi-task learning (joint state + intensity)
3. Attention mechanisms to weight text vs metadata dynamically
4. Label smoothing for noisy labels

### Product Improvements
1. When confidence < 0.5, ask clarifying questions
2. Show uncertainty to users ("I'm not quite sure...")
3. Allow users to correct predictions (active learning)
4. Use uncertainty to decide when to escalate to human review

