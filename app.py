"""
Streamlit web application for Credit Card Fraud Detection.

This app provides an interactive interface to make fraud predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from src.models import ModelTrainer
from src.preprocessing import Preprocessor
from src.explainability import ModelExplainer


@st.cache_resource
def load_model(model_path):
    """Load model and preprocessor with caching."""
    if not os.path.exists(model_path):
        return None, None, None
    
    model = joblib.load(model_path)
    
    # Try to load preprocessor
    preprocessor_path = model_path.replace('.pkl', '_preprocessor.pkl')
    preprocessor = None
    if os.path.exists(preprocessor_path):
        preprocessor = joblib.load(preprocessor_path)
    
    # Try to load feature names
    feature_names_path = model_path.replace('.pkl', '_features.txt')
    feature_names = None
    if os.path.exists(feature_names_path):
        with open(feature_names_path, 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
    
    return model, preprocessor, feature_names


def main():
    st.set_page_config(
        page_title="Credit Card Fraud Detection",
        page_icon="💳",
        layout="wide"
    )
    
    st.title("💳 Credit Card Fraud Detection System")
    st.markdown("---")
    
    # Sidebar for model selection
    st.sidebar.header("Model Selection")
    model_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
    
    if not model_files:
        st.error("No trained models found in the 'models' directory.")
        st.info("Please train a model first using: `python main.py`")
        return
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        model_files,
        index=0
    )
    
    model_path = os.path.join('models', selected_model)
    model, preprocessor, feature_names = load_model(model_path)
    
    if model is None:
        st.error(f"Could not load model from {model_path}")
        return
    
    st.sidebar.success(f"Model loaded: {selected_model}")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "About"])
    
    with tab1:
        st.header("Single Transaction Prediction")
        st.markdown("Enter transaction details to predict if it's fraudulent.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Transaction Details")
            
            # Create input fields for key features
            # For the credit card dataset, we have V1-V28, Time, and Amount
            amount = st.number_input("Amount", min_value=0.0, value=0.0, step=0.01)
            time = st.number_input("Time (seconds)", min_value=0.0, value=0.0, step=1.0)
            
            # For V1-V28, we'll use sliders or number inputs
            # In a real app, you'd want to load the actual feature ranges
            st.markdown("**Anonymized Features (V1-V28)**")
            v_features = {}
            for i in range(1, 29):
                v_features[f'V{i}'] = st.number_input(
                    f"V{i}", 
                    value=0.0, 
                    step=0.01,
                    key=f'v{i}'
                )
        
        with col2:
            st.subheader("Prediction")
            
            if st.button("Predict", type="primary"):
                # Create feature vector
                features = {'Time': time, 'Amount': amount}
                features.update(v_features)
                
                df = pd.DataFrame([features])
                
                # Ensure correct column order
                if feature_names:
                    # Reorder columns to match training data
                    missing_cols = set(feature_names) - set(df.columns)
                    for col in missing_cols:
                        df[col] = 0.0
                    df = df[feature_names]
                
                # Preprocess
                if preprocessor is not None:
                    X_processed = preprocessor.transform(df)
                else:
                    X_processed = df.values
                
                # Predict
                prediction = model.predict(X_processed)[0]
                probability = model.predict_proba(X_processed)[0]
                
                # Display results
                if prediction == 1:
                    st.error(f"🚨 **FRAUD DETECTED**")
                else:
                    st.success(f"✅ **NOT FRAUD**")
                
                st.metric("Fraud Probability", f"{probability[1]:.4f}")
                st.metric("Not Fraud Probability", f"{probability[0]:.4f}")
                
                # Progress bar
                st.progress(probability[1])
                
                # Try to show explanation
                try:
                    explainer = ModelExplainer(model, feature_names=feature_names)
                    explanation = explainer.explain_prediction(X_processed, feature_names=feature_names)
                    if explanation and 'feature_contributions' in explanation:
                        st.subheader("Feature Contributions (SHAP)")
                        contribs = explanation['feature_contributions']
                        contrib_df = pd.DataFrame(
                            list(contribs.items()),
                            columns=['Feature', 'Contribution']
                        ).sort_values('Contribution', key=abs, ascending=False)
                        st.dataframe(contrib_df.head(10))
                except Exception as e:
                    st.info(f"Explanation not available: {e}")
    
    with tab2:
        st.header("Batch Prediction")
        st.markdown("Upload a CSV file with multiple transactions to predict.")
        
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="CSV file should contain columns: Time, Amount, V1-V28"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"File loaded: {df.shape[0]} transactions")
                
                # Check if Class column exists (for evaluation)
                has_target = 'Class' in df.columns
                if has_target:
                    X = df.drop(columns=['Class'])
                    y_true = df['Class']
                else:
                    X = df
                    y_true = None
                
                # Preprocess
                if preprocessor is not None:
                    X_processed = preprocessor.transform(X)
                else:
                    X_processed = X
                
                # Predict
                predictions = model.predict(X_processed)
                probabilities = model.predict_proba(X_processed)
                
                # Create results
                results = df.copy()
                results['Predicted_Class'] = predictions
                results['Probability_Fraud'] = probabilities[:, 1]
                results['Prediction'] = results['Predicted_Class'].map({0: 'Not Fraud', 1: 'Fraud'})
                
                if y_true is not None:
                    results['True_Class'] = y_true
                    results['Correct'] = (predictions == y_true)
                    accuracy = results['Correct'].mean()
                    st.metric("Accuracy", f"{accuracy:.4f}")
                
                # Display summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Transactions", len(results))
                with col2:
                    st.metric("Predicted Fraud", int(results['Predicted_Class'].sum()))
                with col3:
                    st.metric("Predicted Not Fraud", int(len(results) - results['Predicted_Class'].sum()))
                
                # Display results table
                st.subheader("Predictions")
                display_cols = ['Predicted_Class', 'Probability_Fraud', 'Prediction']
                if has_target:
                    display_cols.extend(['True_Class', 'Correct'])
                st.dataframe(results[display_cols])
                
                # Download button
                csv = results.to_csv(index=False)
                st.download_button(
                    label="Download Predictions",
                    data=csv,
                    file_name="fraud_predictions.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    with tab3:
        st.header("About")
        st.markdown("""
        ## Credit Card Fraud Detection System
        
        This application uses machine learning to detect fraudulent credit card transactions.
        
        ### Features:
        - **Single Transaction Prediction**: Enter transaction details to get instant fraud prediction
        - **Batch Prediction**: Upload CSV file to predict multiple transactions at once
        - **Model Explainability**: View feature contributions using SHAP values
        
        ### Models Supported:
        - Logistic Regression
        - Random Forest
        - XGBoost
        - LightGBM
        
        ### Metrics:
        - Precision, Recall, F1-Score
        - ROC-AUC, PR-AUC
        - Confusion Matrix
        
        ### How to Use:
        1. Train a model using: `python main.py`
        2. Select the model from the sidebar
        3. Use Single Prediction tab for individual transactions
        4. Use Batch Prediction tab for CSV files
        """)


if __name__ == '__main__':
    main()


