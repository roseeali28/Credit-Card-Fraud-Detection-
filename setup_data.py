"""
Helper script to download or prepare sample data for testing.

This script helps set up the dataset for the credit card fraud detection project.
"""

import os
import sys
import pandas as pd
import numpy as np


def create_sample_data(n_samples=10000, output_path='data/sample_creditcard.csv'):
    """
    Create a sample dataset with similar structure to the credit card fraud dataset.
    
    This is useful for testing when the actual dataset is not available.
    Note: This is synthetic data and should not be used for actual model training.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate.
    output_path : str
        Path to save the sample dataset.
    """
    print(f"Creating sample dataset with {n_samples} samples...")
    
    np.random.seed(42)
    
    # Create synthetic features
    data = {
        'Time': np.random.uniform(0, 172792, n_samples),
        'Amount': np.random.exponential(88, n_samples),
    }
    
    # Create V1-V28 features (simulating PCA features)
    for i in range(1, 29):
        data[f'V{i}'] = np.random.randn(n_samples)
    
    # Create imbalanced target (similar to real dataset)
    fraud_ratio = 0.0017  # ~0.17% fraud cases
    n_fraud = int(n_samples * fraud_ratio)
    n_not_fraud = n_samples - n_fraud
    
    classes = [0] * n_not_fraud + [1] * n_fraud
    np.random.shuffle(classes)
    data['Class'] = classes
    
    df = pd.DataFrame(data)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Sample dataset saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Class distribution:")
    print(df['Class'].value_counts())
    print(f"\nNote: This is synthetic data for testing purposes only.")
    print(f"For actual training, please download the real dataset from Kaggle.")


def check_dataset(file_path='data/creditcard.csv'):
    """
    Check if the dataset exists and display basic information.
    
    Parameters:
    -----------
    file_path : str
        Path to the dataset.
    """
    if os.path.exists(file_path):
        print(f"✓ Dataset found at {file_path}")
        try:
            df = pd.read_csv(file_path, nrows=5)
            print(f"✓ Dataset is readable")
            print(f"  Columns: {df.columns.tolist()}")
            print(f"  Shape (first 5 rows): {df.shape}")
            
            # Check full dataset
            df_full = pd.read_csv(file_path)
            print(f"  Full dataset shape: {df_full.shape}")
            print(f"  Class distribution:")
            print(df_full['Class'].value_counts())
            print("\n✓ Dataset is ready to use!")
        except Exception as e:
            print(f"✗ Error reading dataset: {e}")
    else:
        print(f"✗ Dataset not found at {file_path}")
        print("\nOptions:")
        print("1. Download from Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("2. Create sample data for testing: python setup_data.py --create-sample")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup data for Credit Card Fraud Detection')
    parser.add_argument('--check', action='store_true',
                       help='Check if dataset exists')
    parser.add_argument('--create-sample', action='store_true',
                       help='Create a sample dataset for testing')
    parser.add_argument('--n-samples', type=int, default=10000,
                       help='Number of samples for synthetic data (default: 10000)')
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_data(n_samples=args.n_samples)
    elif args.check:
        check_dataset()
    else:
        # Default: check dataset
        check_dataset()
        print("\n" + "="*60)
        print("To create sample data for testing:")
        print("  python setup_data.py --create-sample")
        print("="*60)


if __name__ == '__main__':
    main()


