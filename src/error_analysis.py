"""
Error Analysis: Identify and analyze failure cases
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_model import EmotionalStatePredictor


def analyze_failures(train_df, predictor, top_n=15):
    """Analyze the worst performing predictions"""

    print("="*80)
    print("ERROR ANALYSIS: Identifying Failure Cases")
    print("="*80)
    print()

    # Make predictions on training data to analyze errors
    predictions, _ = predictor.predict_with_uncertainty(train_df)

    # Calculate errors
    train_df = train_df.copy()
    train_df['predicted_state'] = predictions['predicted_state'].values
    train_df['predicted_intensity'] = predictions['predicted_intensity'].values
    train_df['confidence'] = predictions['confidence'].values
    train_df['uncertain_flag'] = predictions['uncertain_flag'].values

    # State prediction errors
    train_df['state_correct'] = (
        train_df['emotional_state'] == train_df['predicted_state']
    ).astype(int)

    # Intensity prediction errors
    train_df['intensity_error'] = np.abs(
        train_df['intensity'] - train_df['predicted_intensity']
    )

    # Combined error score (for ranking worst cases)
    train_df['error_score'] = (
        (1 - train_df['state_correct']) * 2 +  # State errors weighted more
        train_df['intensity_error']
    )

    # Get worst cases
    worst_cases = train_df.nlargest(top_n, 'error_score')

    print(f"Found {top_n} worst cases")
    print()

    # Categorize errors
    error_categories = {
        'ambiguous_text': [],
        'very_short_input': [],
        'conflicting_signals': [],
        'low_confidence': [],
        'label_noise_suspected': [],
        'context_ignored': [],
        'intensity_mismatch': [],
        'state_confusion': []
    }

    for idx, row in worst_cases.iterrows():
        # Categorize error
        text_len = len(str(row['journal_text']).split())

        if text_len <= 3:
            error_categories['very_short_input'].append(idx)

        if row['confidence'] < 0.4:
            error_categories['low_confidence'].append(idx)

        if row['intensity_error'] >= 2:
            error_categories['intensity_mismatch'].append(idx)

        if not row['state_correct']:
            error_categories['state_confusion'].append(idx)

        # Conflicting signals: high stress but calm state predicted
        if row['stress_level'] >= 4 and row['predicted_state'] in ['calm', 'peaceful', 'content']:
            error_categories['conflicting_signals'].append(idx)

        # Ambiguous text
        ambiguous_words = ['ok', 'fine', 'alright', 'good', 'not sure', 'maybe', 'idk']
        if any(word in str(row['journal_text']).lower() for word in ambiguous_words):
            error_categories['ambiguous_text'].append(idx)

    # Print categorization summary
    print("Error Categories:")
    for category, indices in error_categories.items():
        if len(indices) > 0:
            print(f"  {category}: {len(indices)} cases")
    print()

    return worst_cases, error_categories


def generate_error_report(worst_cases, error_categories, output_path='docs/ERROR_ANALYSIS.md'):
    """Generate comprehensive error analysis report"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Error Analysis Report\n\n")
        f.write("## Overview\n\n")
        f.write("This document analyzes failure cases to understand model limitations and improvement opportunities.\n\n")

        f.write("## Error Category Summary\n\n")
        for category, indices in error_categories.items():
            if len(indices) > 0:
                f.write(f"- **{category.replace('_', ' ').title()}**: {len(indices)} cases\n")

        f.write("\n## Detailed Error Analysis (10+ Cases)\n\n")

        # Analyze top 10-12 diverse errors
        # Take the worst cases by error score
        analyzed_cases = worst_cases.head(12).index.tolist()

        case_num = 1
        for idx in analyzed_cases:
            row = worst_cases.loc[idx]

            f.write(f"### Case {case_num}\n\n")
            f.write(f"**ID**: {row['id']}\n\n")
            f.write(f"**Journal Text**: \"{row['journal_text']}\"\n\n")
            f.write(f"**True State**: {row['emotional_state']} (Intensity: {row['intensity']})\n\n")
            f.write(f"**Predicted State**: {row['predicted_state']} (Intensity: {row['predicted_intensity']})\n\n")
            f.write(f"**Confidence**: {row['confidence']:.3f}\n\n")

            f.write("**Context**:\n")
            f.write(f"- Sleep: {row['sleep_hours']}h\n")
            f.write(f"- Stress: {row['stress_level']}/5\n")
            f.write(f"- Energy: {row['energy_level']}/5\n")
            f.write(f"- Time: {row['time_of_day']}\n")
            f.write(f"- Ambience: {row['ambience_type']}\n\n")

            # Diagnose what went wrong
            f.write("**What Went Wrong**:\n")

            text_len = len(str(row['journal_text']).split())
            if text_len <= 3:
                f.write("- Very short input (≤3 words) provides limited signal\n")

            if row['confidence'] < 0.4:
                f.write("- Model had low confidence, indicating uncertainty\n")

            if row['state_correct'] == 0:
                f.write(f"- State misclassification: predicted {row['predicted_state']} instead of {row['emotional_state']}\n")

            if row['intensity_error'] >= 2:
                f.write(f"- Large intensity error ({row['intensity_error']})\n")

            # Check for conflicting signals
            if row['stress_level'] >= 4 and row['energy_level'] <= 2:
                f.write("- Conflicting signals: high stress + low energy\n")

            if row['energy_level'] >= 4 and row['emotional_state'] in ['tired', 'exhausted']:
                f.write("- Metadata contradicts emotional state label\n")

            f.write("\n**Why It Failed**:\n")

            # Explain root cause
            if text_len <= 3:
                f.write("- Insufficient text information forces model to rely heavily on metadata\n")

            ambiguous_words = ['ok', 'fine', 'alright', 'good']
            if any(word in str(row['journal_text']).lower() for word in ambiguous_words):
                f.write("- Ambiguous language ('ok', 'fine') can indicate various emotional states\n")

            if row['stress_level'] != row['energy_level']:
                gap = abs(row['stress_level'] - row['energy_level'])
                if gap >= 3:
                    f.write(f"- Large stress-energy gap ({gap}) creates conflicting signals\n")

            f.write("\n**How to Improve**:\n")

            if text_len <= 3:
                f.write("- Prompt users for more detailed reflections\n")
                f.write("- Add special handling for short inputs with higher uncertainty\n")

            if row['confidence'] < 0.4:
                f.write("- Use ensemble methods or additional models for low-confidence cases\n")
                f.write("- Consider asking clarifying questions when confidence is low\n")

            if row['intensity_error'] >= 2:
                f.write("- Improve intensity calibration with more training data\n")
                f.write("- Use ordinal regression instead of standard regression\n")

            f.write("- Collect more training examples similar to this case\n")
            f.write("- Add feature engineering to capture text-metadata interactions\n")

            f.write("\n---\n\n")
            case_num += 1

        # Key insights section
        f.write("## Key Insights\n\n")

        f.write("### Main Failure Patterns\n\n")
        f.write("1. **Short Text Problem**: Inputs with ≤3 words provide insufficient signal\n")
        f.write("   - Model forced to over-rely on metadata\n")
        f.write("   - Ambiguous words like 'ok', 'fine' can mean many things\n\n")

        f.write("2. **Conflicting Signals**: When text and metadata contradict\n")
        f.write("   - Example: User writes 'great session' but has high stress, low energy\n")
        f.write("   - Model struggles to weight text vs metadata appropriately\n\n")

        f.write("3. **Low Confidence Cases**: Model uncertainty is informative\n")
        f.write("   - Low confidence often correlates with actual errors\n")
        f.write("   - Uncertainty flag is critical for downstream decisions\n\n")

        f.write("4. **Intensity Calibration**: Harder than state classification\n")
        f.write("   - Fine-grained intensity (1-5) is subjective and noisy\n")
        f.write("   - May benefit from ordinal regression or better loss function\n\n")

        f.write("5. **Label Noise**: Some labels may be incorrect or debatable\n")
        f.write("   - Human emotional state labeling is inherently subjective\n")
        f.write("   - Inter-annotator agreement would be valuable\n\n")

        f.write("## Improvement Recommendations\n\n")

        f.write("### Immediate Actions\n")
        f.write("1. Add minimum text length requirement or prompt for details\n")
        f.write("2. Implement calibrated uncertainty estimation\n")
        f.write("3. Add text-metadata interaction features\n")
        f.write("4. Use ordinal regression for intensity prediction\n\n")

        f.write("### Data Collection\n")
        f.write("1. Collect more examples of:\n")
        f.write("   - Short but meaningful reflections\n")
        f.write("   - Cases with conflicting signals\n")
        f.write("   - Rare emotional states\n")
        f.write("2. Add confidence ratings to labels (how sure is the annotator?)\n\n")

        f.write("### Model Improvements\n")
        f.write("1. Ensemble methods for robustness\n")
        f.write("2. Multi-task learning (joint state + intensity)\n")
        f.write("3. Attention mechanisms to weight text vs metadata dynamically\n")
        f.write("4. Label smoothing for noisy labels\n\n")

        f.write("### Product Improvements\n")
        f.write("1. When confidence < 0.5, ask clarifying questions\n")
        f.write("2. Show uncertainty to users (\"I'm not quite sure...\")\n")
        f.write("3. Allow users to correct predictions (active learning)\n")
        f.write("4. Use uncertainty to decide when to escalate to human review\n\n")

    print(f"Error analysis report saved to {output_path}")


def main():
    """Run error analysis"""

    # Load data
    train_df = pd.read_excel('Sample_arvyax_reflective_dataset.xlsx')

    # Load trained model
    predictor = EmotionalStatePredictor()

    print("Training model for error analysis...")
    predictor.train(train_df)

    # Analyze failures
    worst_cases, error_categories = analyze_failures(train_df, predictor, top_n=15)

    # Generate report
    generate_error_report(worst_cases, error_categories)

    print("\nError analysis complete!")


if __name__ == '__main__':
    main()
