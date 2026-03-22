"""Quick data exploration script"""
import pandas as pd
import sys

try:
    # Load training data
    train_df = pd.read_excel('Sample_arvyax_reflective_dataset.xlsx')
    print("=" * 80)
    print("TRAINING DATA")
    print("=" * 80)
    print(f"\nShape: {train_df.shape}")
    print(f"\nColumns: {list(train_df.columns)}")
    print(f"\nFirst few rows:")
    print(train_df.head())
    print(f"\nData types:")
    print(train_df.dtypes)
    print(f"\nMissing values:")
    print(train_df.isnull().sum())
    print(f"\nEmotional states distribution:")
    print(train_df['emotional_state'].value_counts())
    print(f"\nIntensity distribution:")
    print(train_df['intensity'].value_counts().sort_index())

    # Load test data
    test_df = pd.read_excel('arvyax_test_inputs_120.xlsx')
    print("\n" + "=" * 80)
    print("TEST DATA")
    print("=" * 80)
    print(f"\nShape: {test_df.shape}")
    print(f"\nColumns: {list(test_df.columns)}")
    print(f"\nFirst few rows:")
    print(test_df.head())
    print(f"\nMissing values:")
    print(test_df.isnull().sum())

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
