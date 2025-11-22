"""
Prediction script for Credit Card Fraud Detection.

This script loads a trained model and makes predictions on new data.
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import os
import sys
from src.preprocessing import Preprocessor
from src.models import ModelTrainer
from src.explainability import ModelExplainer


def load_model_and_preprocessor(model_path):
    """
    Load model and attempt to load preprocessor if available.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model.
        
    Returns:
    --------
    tuple
        (model, preprocessor, feature_names)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = joblib.load(model_path)
    
    # Try to load preprocessor if it exists
    preprocessor_path = model_path.replace('.pkl', '_preprocessor.pkl')
    if os.path.exists(preprocessor_path):
        preprocessor = joblib.load(preprocessor_path)
    else:
        preprocessor = None
        print("Warning: Preprocessor not found. Make sure to preprocess data manually.")
    
    # Try to load feature names
    feature_names_path = model_path.replace('.pkl', '_features.txt')
    feature_names = None
    if os.path.exists(feature_names_path):
        with open(feature_names_path, 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
    
    return model, preprocessor, feature_names


def predict_from_csv(model_path, data_path, output_path=None):
    """
    Make predictions from a CSV file.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model.
    data_path : str
        Path to the CSV file with transaction data.
    output_path : str, optional
        Path to save predictions.
    """
    print(f"Loading model from {model_path}...")
    model, preprocessor, feature_names = load_model_and_preprocessor(model_path)
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Check if Class column exists (for evaluation)
    has_target = 'Class' in df.columns
    if has_target:
        X = df.drop(columns=['Class'])
        y_true = df['Class']
    else:
        X = df
        y_true = None
    
    # Preprocess if preprocessor is available
    if preprocessor is not None:
        X_processed = preprocessor.transform(X)
    else:
        print("Warning: No preprocessor found. Using raw features.")
        X_processed = X
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X_processed)
    probabilities = model.predict_proba(X_processed)
    
    # Create results dataframe
    results = df.copy()
    results['Predicted_Class'] = predictions
    results['Probability_Not_Fraud'] = probabilities[:, 0]
    results['Probability_Fraud'] = probabilities[:, 1]
    results['Prediction'] = results['Predicted_Class'].map({0: 'Not Fraud', 1: 'Fraud'})
    
    if y_true is not None:
        results['True_Class'] = y_true
        results['Correct'] = (predictions == y_true)
        accuracy = results['Correct'].mean()
        print(f"\nAccuracy: {accuracy:.4f}")
    
    # Print summary
    print("\nPrediction Summary:")
    print(f"Total transactions: {len(results)}")
    print(f"Predicted Fraud: {results['Predicted_Class'].sum()}")
    print(f"Predicted Not Fraud: {len(results) - results['Predicted_Class'].sum()}")
    
    # Save results
    if output_path:
        results.to_csv(output_path, index=False)
        print(f"\nPredictions saved to {output_path}")
    else:
        print("\nPredictions (first 10 rows):")
        print(results[['Predicted_Class', 'Probability_Fraud', 'Prediction']].head(10))
    
    return results


def predict_single(model_path, transaction_data):
    """
    Make prediction for a single transaction.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model.
    transaction_data : dict or pd.Series
        Transaction features.
        
    Returns:
    --------
    dict
        Prediction results with explanation.
    """
    model, preprocessor, feature_names = load_model_and_preprocessor(model_path)
    
    # Convert to DataFrame if needed
    if isinstance(transaction_data, dict):
        df = pd.DataFrame([transaction_data])
    elif isinstance(transaction_data, pd.Series):
        df = pd.DataFrame([transaction_data])
    else:
        df = transaction_data
    
    # Preprocess
    if preprocessor is not None:
        X_processed = preprocessor.transform(df)
    else:
        X_processed = df
    
    # Predict
    prediction = model.predict(X_processed)[0]
    probability = model.predict_proba(X_processed)[0]
    
    result = {
        'prediction': int(prediction),
        'prediction_label': 'Fraud' if prediction == 1 else 'Not Fraud',
        'probability_fraud': float(probability[1]),
        'probability_not_fraud': float(probability[0])
    }
    
    # Try to get explanation
    try:
        explainer = ModelExplainer(model, feature_names=feature_names)
        explanation = explainer.explain_prediction(X_processed.iloc[0:1], feature_names=feature_names)
        if explanation:
            result['explanation'] = explanation
    except Exception as e:
        print(f"Could not generate explanation: {e}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Make predictions using trained model')
    parser.add_argument('--model', type=str, default='models/random_forest.pkl',
                       help='Path to the saved model')
    parser.add_argument('--data', type=str,
                       help='Path to CSV file with transaction data')
    parser.add_argument('--output', type=str,
                       help='Path to save predictions (CSV file)')
    
    args = parser.parse_args()
    
    if args.data:
        # Batch prediction from CSV
        predict_from_csv(args.model, args.data, args.output)
    else:
        print("Please provide --data argument with path to CSV file")
        print("\nExample usage:")
        print("  python predict.py --model models/random_forest.pkl --data data/test_transactions.csv --output predictions.csv")


if __name__ == '__main__':
    main()


