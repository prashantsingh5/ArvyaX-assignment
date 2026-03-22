"""
Feature Importance Analysis
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_model import EmotionalStatePredictor


def analyze_feature_importance(predictor, X_train, feature_names):
    """Analyze feature importance from trained models"""

    print("="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    print()

    # Get feature importance from XGBoost models
    state_importance = predictor.state_model.feature_importances_
    intensity_importance = predictor.intensity_model.feature_importances_

    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'state_importance': state_importance,
        'intensity_importance': intensity_importance
    })

    # Add feature type
    importance_df['feature_type'] = importance_df['feature'].apply(
        lambda x: 'text' if x.startswith('text_') else 'metadata'
    )

    # Sort by state importance
    importance_df = importance_df.sort_values('state_importance', ascending=False)

    print("Top 20 features for STATE prediction:")
    print(importance_df[['feature', 'state_importance', 'feature_type']].head(20).to_string(index=False))
    print()

    print("Top 20 features for INTENSITY prediction:")
    print(importance_df.sort_values('intensity_importance', ascending=False)[
        ['feature', 'intensity_importance', 'feature_type']
    ].head(20).to_string(index=False))
    print()

    # Aggregate by feature type
    type_importance = importance_df.groupby('feature_type').agg({
        'state_importance': 'sum',
        'intensity_importance': 'sum'
    }).reset_index()

    print("Aggregate importance by feature type:")
    print(type_importance.to_string(index=False))
    print()

    # Calculate percentages
    state_text_pct = type_importance[type_importance['feature_type'] == 'text']['state_importance'].values[0] / type_importance['state_importance'].sum() * 100
    state_meta_pct = type_importance[type_importance['feature_type'] == 'metadata']['state_importance'].values[0] / type_importance['state_importance'].sum() * 100

    print(f"For STATE prediction:")
    print(f"  Text features: {state_text_pct:.1f}%")
    print(f"  Metadata features: {state_meta_pct:.1f}%")
    print()

    intensity_text_pct = type_importance[type_importance['feature_type'] == 'text']['intensity_importance'].values[0] / type_importance['intensity_importance'].sum() * 100
    intensity_meta_pct = type_importance[type_importance['feature_type'] == 'metadata']['intensity_importance'].values[0] / type_importance['intensity_importance'].sum() * 100

    print(f"For INTENSITY prediction:")
    print(f"  Text features: {intensity_text_pct:.1f}%")
    print(f"  Metadata features: {intensity_meta_pct:.1f}%")
    print()

    # Save results
    importance_df.to_csv('outputs/feature_importance.csv', index=False)
    print("Feature importance saved to outputs/feature_importance.csv")

    return importance_df


def main():
    """Run feature importance analysis"""

    # Load data
    train_df = pd.read_excel('Sample_arvyax_reflective_dataset.xlsx')

    # Train model
    predictor = EmotionalStatePredictor()
    print("Training model for feature importance analysis...")
    X_train, y_state, y_intensity = predictor.train(train_df)

    # Get feature names
    feature_names = X_train.columns.tolist()

    # Analyze
    importance_df = analyze_feature_importance(predictor, X_train, feature_names)


if __name__ == '__main__':
    main()
