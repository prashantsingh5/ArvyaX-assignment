"""
Ablation Study: Text-only vs Text+Metadata
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_model import EmotionalStatePredictor


def evaluate_model(predictor, train_df, val_df, use_text, use_metadata):
    """Train and evaluate model with specific feature configuration"""

    print(f"\n{'='*80}")
    print(f"Configuration: use_text={use_text}, use_metadata={use_metadata}")
    print(f"{'='*80}")

    # Train
    predictor.train(train_df, use_text=use_text, use_metadata=use_metadata)

    # Predict on validation
    predictions, _ = predictor.predict_with_uncertainty(
        val_df, use_text=use_text, use_metadata=use_metadata
    )

    # Encode true labels
    y_true_state = predictor.state_encoder.transform(val_df['emotional_state'])
    y_pred_state = predictor.state_encoder.transform(predictions['predicted_state'])

    y_true_intensity = val_df['intensity'].values
    y_pred_intensity = predictions['predicted_intensity'].values

    # Calculate metrics
    state_acc = accuracy_score(y_true_state, y_pred_state)
    state_f1 = f1_score(y_true_state, y_pred_state, average='weighted')

    intensity_rmse = np.sqrt(mean_squared_error(y_true_intensity, y_pred_intensity))
    intensity_mae = np.mean(np.abs(y_true_intensity - y_pred_intensity))

    # Average confidence
    avg_confidence = predictions['confidence'].mean()
    uncertain_pct = predictions['uncertain_flag'].mean() * 100

    results = {
        'use_text': use_text,
        'use_metadata': use_metadata,
        'state_accuracy': state_acc,
        'state_f1': state_f1,
        'intensity_rmse': intensity_rmse,
        'intensity_mae': intensity_mae,
        'avg_confidence': avg_confidence,
        'uncertain_pct': uncertain_pct
    }

    print(f"State Accuracy: {state_acc:.4f}")
    print(f"State F1 Score: {state_f1:.4f}")
    print(f"Intensity RMSE: {intensity_rmse:.4f}")
    print(f"Intensity MAE: {intensity_mae:.4f}")
    print(f"Avg Confidence: {avg_confidence:.4f}")
    print(f"Uncertain %: {uncertain_pct:.2f}%")

    return results


def main():
    """Run ablation study"""
    print("="*80)
    print("ABLATION STUDY: TEXT vs METADATA vs COMBINED")
    print("="*80)
    print()

    # Load data
    train_df = pd.read_excel('Sample_arvyax_reflective_dataset.xlsx')

    # Split into train and validation
    from sklearn.model_selection import train_test_split
    train_subset, val_subset = train_test_split(
        train_df, test_size=0.2, random_state=42, stratify=train_df['emotional_state']
    )

    print(f"Train samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")
    print()

    results_list = []

    # Configuration 1: Text only
    print("\n### CONFIGURATION 1: TEXT ONLY ###")
    predictor1 = EmotionalStatePredictor()
    results1 = evaluate_model(predictor1, train_subset, val_subset, use_text=True, use_metadata=False)
    results_list.append(results1)

    # Configuration 2: Metadata only
    print("\n### CONFIGURATION 2: METADATA ONLY ###")
    predictor2 = EmotionalStatePredictor()
    results2 = evaluate_model(predictor2, train_subset, val_subset, use_text=False, use_metadata=True)
    results_list.append(results2)

    # Configuration 3: Text + Metadata (Combined)
    print("\n### CONFIGURATION 3: TEXT + METADATA ###")
    predictor3 = EmotionalStatePredictor()
    results3 = evaluate_model(predictor3, train_subset, val_subset, use_text=True, use_metadata=True)
    results_list.append(results3)

    # Create comparison table
    results_df = pd.DataFrame(results_list)

    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    print()
    print(results_df.to_string(index=False))
    print()

    # Save results
    results_df.to_csv('outputs/ablation_study.csv', index=False)
    print("Ablation results saved to outputs/ablation_study.csv")

    # Analysis
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    best_state_idx = results_df['state_accuracy'].idxmax()
    best_intensity_idx = results_df['intensity_rmse'].idxmin()

    print(f"\nBest State Prediction: {results_df.loc[best_state_idx, 'use_text']} text, "
          f"{results_df.loc[best_state_idx, 'use_metadata']} metadata "
          f"(Accuracy: {results_df.loc[best_state_idx, 'state_accuracy']:.4f})")

    print(f"Best Intensity Prediction: {results_df.loc[best_intensity_idx, 'use_text']} text, "
          f"{results_df.loc[best_intensity_idx, 'use_metadata']} metadata "
          f"(RMSE: {results_df.loc[best_intensity_idx, 'intensity_rmse']:.4f})")

    # Calculate improvement
    text_only_acc = results_df[results_df['use_text'] & ~results_df['use_metadata']]['state_accuracy'].values[0]
    combined_acc = results_df[results_df['use_text'] & results_df['use_metadata']]['state_accuracy'].values[0]
    improvement = (combined_acc - text_only_acc) / text_only_acc * 100

    print(f"\nImprovement from Text-only to Combined: {improvement:.2f}%")

    print("\nConclusion:")
    if combined_acc > text_only_acc and combined_acc > results_df[~results_df['use_text'] & results_df['use_metadata']]['state_accuracy'].values[0]:
        print("Success - Combined (Text + Metadata) provides the best performance")
        print("Success - Both text and contextual signals are important for understanding emotional state")
    else:
        print("WARNING - Further investigation needed")


if __name__ == '__main__':
    main()
