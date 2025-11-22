"""
Main training script for Credit Card Fraud Detection.

This script:
1. Loads and preprocesses the data
2. Trains multiple models
3. Evaluates models
4. Saves the best model
"""

import argparse
import os
import sys
from src.data_loader import load_data, check_missing_values, clean_data, split_data
from src.preprocessing import Preprocessor
from src.models import ModelTrainer
from src.evaluation import ModelEvaluator
from src.explainability import ModelExplainer


def main():
    parser = argparse.ArgumentParser(description='Train Credit Card Fraud Detection Models')
    parser.add_argument('--data', type=str, default='data/creditcard.csv',
                       help='Path to the dataset CSV file')
    parser.add_argument('--model', type=str, default='random_forest',
                       choices=['logistic_regression', 'random_forest', 'xgboost', 'lightgbm'],
                       help='Type of model to train')
    parser.add_argument('--use-smote', action='store_true',
                       help='Use SMOTE for oversampling')
    parser.add_argument('--use-class-weights', action='store_true',
                       help='Use class weights for imbalanced data')
    parser.add_argument('--tune-hyperparameters', action='store_true',
                       help='Perform hyperparameter tuning')
    parser.add_argument('--scaler', type=str, default='robust',
                       choices=['standard', 'robust'],
                       help='Type of scaler to use')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Credit Card Fraud Detection - Model Training")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"SMOTE: {args.use_smote}")
    print(f"Class Weights: {args.use_class_weights}")
    print(f"Hyperparameter Tuning: {args.tune_hyperparameters}")
    print(f"Scaler: {args.scaler}")
    print("="*70)
    
    # 1. Load data
    print("\n[1/6] Loading data...")
    df = load_data(args.data)
    
    # 2. Check and clean data
    print("\n[2/6] Checking and cleaning data...")
    check_missing_values(df)
    df = clean_data(df)
    
    # 3. Split data
    print("\n[3/6] Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    
    # 4. Preprocessing
    print("\n[4/6] Preprocessing data...")
    preprocessor = Preprocessor(
        use_smote=args.use_smote,
        use_class_weights=args.use_class_weights,
        scaler_type=args.scaler
    )
    
    X_train_processed, y_train_processed = preprocessor.fit_transform(
        X_train, y_train, apply_smote=args.use_smote
    )
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    class_weights = preprocessor.get_class_weights()
    
    # 5. Train model
    print(f"\n[5/6] Training {args.model} model...")
    trainer = ModelTrainer(model_type=args.model, random_state=42)
    model = trainer.train(
        X_train_processed, y_train_processed,
        class_weights=class_weights,
        tune_hyperparameters=args.tune_hyperparameters
    )
    
    # 6. Evaluate model
    print("\n[6/6] Evaluating model...")
    
    # Evaluate on validation set
    print("\n--- Validation Set Evaluation ---")
    evaluator_val = ModelEvaluator(model, f"{args.model}_validation")
    metrics_val = evaluator_val.evaluate(X_val_processed, y_val)
    evaluator_val.generate_all_plots(X_val_processed, y_val)
    
    # Evaluate on test set
    print("\n--- Test Set Evaluation ---")
    evaluator_test = ModelEvaluator(model, f"{args.model}_test")
    metrics_test = evaluator_test.evaluate(X_test_processed, y_test)
    evaluator_test.generate_all_plots(X_test_processed, y_test)
    
    # 7. Explainability
    print("\n[7/7] Generating explainability plots...")
    explainer = ModelExplainer(model, feature_names=X_train.columns.tolist())
    explainer.plot_feature_importance(top_n=20)
    
    # Try to create SHAP explainer (may fail for some models)
    try:
        explainer.create_shap_explainer(
            X_train_processed.sample(min(100, len(X_train_processed)), random_state=42),
            explainer_type='tree' if args.model in ['random_forest', 'xgboost', 'lightgbm'] else 'linear'
        )
        explainer.plot_shap_summary(X_val_processed.sample(min(500, len(X_val_processed)), random_state=42))
    except Exception as e:
        print(f"SHAP explanation skipped: {e}")
    
    # 8. Save model, preprocessor, and feature names
    model_name = f"{args.model}"
    if args.use_smote:
        model_name += "_smote"
    if args.use_class_weights:
        model_name += "_classweights"
    
    model_path = f"models/{model_name}.pkl"
    trainer.save_model(model_path)
    
    # Save preprocessor
    import joblib
    preprocessor_path = model_path.replace('.pkl', '_preprocessor.pkl')
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Preprocessor saved to {preprocessor_path}")
    
    # Save feature names
    feature_names_path = model_path.replace('.pkl', '_features.txt')
    with open(feature_names_path, 'w') as f:
        for feature in X_train.columns.tolist():
            f.write(f"{feature}\n")
    print(f"Feature names saved to {feature_names_path}")
    
    print("\n" + "="*70)
    print("Training completed successfully!")
    print(f"Model saved to: {model_path}")
    print(f"Test Set ROC-AUC: {metrics_test['roc_auc']:.4f}")
    print(f"Test Set PR-AUC: {metrics_test['pr_auc']:.4f}")
    print("="*70)


if __name__ == '__main__':
    main()

