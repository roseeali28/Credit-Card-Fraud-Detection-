# Credit Card Fraud Detection
Rose Ali
A complete, production-ready machine learning project for detecting fraudulent credit card transactions using Python.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Training](#model-training)
- [Making Predictions](#making-predictions)
- [Web Application](#web-application)
- [Evaluation Metrics](#evaluation-metrics)
- [Explainability](#explainability)
- [Testing](#testing)
- [Project Structure Details](#project-structure-details)

## 🎯 Overview

This project implements a comprehensive credit card fraud detection system using various machine learning algorithms. It includes:

- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, LightGBM
- **Class Imbalance Handling**: SMOTE oversampling and class weights
- **Comprehensive Evaluation**: Multiple metrics including ROC-AUC, PR-AUC, Precision, Recall, F1-Score
- **Model Explainability**: Feature importance and SHAP values
- **Interactive Web Interface**: Streamlit-based UI for predictions
- **Production-Ready Code**: Modular structure, unit tests, and documentation

## ✨ Features

- **Data Preprocessing**: Robust scaling, missing value handling, data cleaning
- **Exploratory Data Analysis**: Comprehensive EDA notebook with visualizations
- **Model Training**: Support for multiple algorithms with hyperparameter tuning
- **Evaluation**: Comprehensive metrics and visualization plots
- **Explainability**: Feature importance plots and SHAP explanations
- **Web Interface**: Streamlit app for interactive predictions
- **Batch Prediction**: CLI tool for processing CSV files
- **Unit Tests**: pytest-based test suite

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── data/                      # Dataset directory
│   └── creditcard.csv         # Credit card fraud dataset
│
├── notebooks/                  # Jupyter notebooks
│   └── eda.ipynb              # Exploratory Data Analysis
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── data_loader.py         # Data loading and splitting
│   ├── preprocessing.py       # Preprocessing pipeline
│   ├── models.py              # Model training utilities
│   ├── evaluation.py          # Model evaluation metrics
│   └── explainability.py      # Model explainability (SHAP, feature importance)
│
├── models/                    # Saved trained models
│
├── reports/                   # Generated reports and plots
│   ├── *.png                  # Visualization plots
│   └── *_metrics.json         # Evaluation metrics
│
├── tests/                     # Unit tests
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   └── test_models.py
│
├── main.py                    # Main training script
├── predict.py                 # Prediction script
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone or download the project**:
   ```bash
   cd Y:\projects
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset

This project uses the **Credit Card Fraud Detection** dataset, which contains anonymized credit card transactions. The dataset includes:

- **Features**: V1-V28 (anonymized features from PCA), Time, Amount
- **Target**: Class (0 = Not Fraud, 1 = Fraud)
- **Class Imbalance**: Highly imbalanced (~0.17% fraud cases)

### Downloading the Dataset

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Place the `creditcard.csv` file in the `data/` directory

Alternatively, you can use any CSV file with the same structure (V1-V28, Time, Amount, Class columns).

## 💻 Usage

### Model Training

Train a model using the main script:

```bash
# Basic training with Random Forest
python main.py

# Train with SMOTE oversampling
python main.py --use-smote

# Train with class weights
python main.py --use-class-weights

# Train XGBoost with hyperparameter tuning
python main.py --model xgboost --tune-hyperparameters

# Train with both SMOTE and class weights
python main.py --use-smote --use-class-weights

# Use different scaler
python main.py --scaler standard
```

**Available models**: `logistic_regression`, `random_forest`, `xgboost`, `lightgbm`

**Options**:
- `--data`: Path to dataset (default: `data/creditcard.csv`)
- `--model`: Model type (default: `random_forest`)
- `--use-smote`: Enable SMOTE oversampling
- `--use-class-weights`: Enable class weights
- `--tune-hyperparameters`: Perform hyperparameter tuning
- `--scaler`: Scaler type (`standard` or `robust`, default: `robust`)

### Making Predictions

#### Single Prediction (CLI)

Use the prediction script to make predictions on a CSV file:

```bash
# Predict on a CSV file
python predict.py --model models/random_forest.pkl --data data/test_transactions.csv --output predictions.csv
```

#### Batch Prediction

The script will:
- Load the trained model
- Preprocess the input data
- Make predictions
- Save results to a CSV file

### Web Application

Launch the Streamlit web application:

```bash
streamlit run app.py
```

The app will open in your browser (typically at `http://localhost:8501`).

**Features**:
- **Single Prediction**: Enter transaction details manually
- **Batch Prediction**: Upload CSV file for multiple predictions
- **Model Selection**: Choose from available trained models
- **Feature Contributions**: View SHAP-based feature importance

### Exploratory Data Analysis

Run the EDA notebook:

```bash
jupyter notebook notebooks/eda.ipynb
```

Or use JupyterLab:

```bash
jupyter lab notebooks/eda.ipynb
```

## 📈 Evaluation Metrics

The project evaluates models using multiple metrics suitable for imbalanced classification:

- **Accuracy**: Overall classification accuracy
- **Precision**: Precision for fraud class
- **Recall**: Recall (sensitivity) for fraud class
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **PR-AUC**: Area under the Precision-Recall curve
- **Confusion Matrix**: True/False Positives and Negatives

All metrics are saved to JSON files in the `reports/` directory, and visualization plots (ROC curve, PR curve, confusion matrix) are saved as PNG files.

## 🔍 Explainability

The project includes model explainability features:

1. **Feature Importance**: 
   - Tree-based models: Feature importances from the model
   - Linear models: Absolute coefficient values
   - Saved as plots in `reports/feature_importance.png`

2. **SHAP Values**:
   - SHAP summary plots for global interpretability
   - Individual prediction explanations
   - Feature contribution analysis

## 🧪 Testing

Run the unit tests:

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v
```

## 📂 Project Structure Details

### Source Code Modules

- **`data_loader.py`**: Functions for loading, cleaning, and splitting data
- **`preprocessing.py`**: Preprocessing pipeline with scaling, SMOTE, and class weights
- **`models.py`**: Model training, hyperparameter tuning, and model persistence
- **`evaluation.py`**: Comprehensive evaluation metrics and visualization
- **`explainability.py`**: Feature importance and SHAP-based explanations

### Key Scripts

- **`main.py`**: Main training pipeline
- **`predict.py`**: Command-line prediction tool
- **`app.py`**: Streamlit web application

## 🎓 Example Workflow

1. **Prepare Data**:
   ```bash
   # Place creditcard.csv in data/ directory
   ```

2. **Explore Data**:
   ```bash
   jupyter notebook notebooks/eda.ipynb
   ```

3. **Train Model**:
   ```bash
   python main.py --model random_forest --use-class-weights --tune-hyperparameters
   ```

4. **Evaluate Results**:
   - Check `reports/` for metrics and plots
   - Review saved model in `models/`

5. **Make Predictions**:
   ```bash
   python predict.py --model models/random_forest_classweights.pkl --data data/test.csv
   ```

6. **Use Web Interface**:
   ```bash
   streamlit run app.py
   ```

## 🔧 Configuration

### Toggle Class Imbalance Handling

In `main.py`, you can easily toggle between different approaches:

- **SMOTE**: `--use-smote` flag
- **Class Weights**: `--use-class-weights` flag
- **Both**: Use both flags together

### Model Selection

Choose from:
- `logistic_regression`: Fast, interpretable
- `random_forest`: Robust, good default choice
- `xgboost`: High performance, gradient boosting
- `lightgbm`: Fast gradient boosting

## 📝 Notes

- The dataset is highly imbalanced. Always use appropriate techniques (SMOTE, class weights) or evaluation metrics (ROC-AUC, PR-AUC).
- Hyperparameter tuning can be time-consuming. Use `--tune-hyperparameters` only when needed.
- SHAP explanations may take time for large datasets. The code uses sampling to speed up computation.
- For production use, consider model versioning and monitoring.

## 🤝 Contributing

This is a portfolio/academic project. Feel free to:
- Report issues
- Suggest improvements
- Fork and extend

## 📄 License

This project is provided as-is for educational and portfolio purposes.

## 🙏 Acknowledgments

- Dataset: [Credit Card Fraud Detection on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Libraries: scikit-learn, XGBoost, LightGBM, SHAP, Streamlit

---

**Happy Fraud Detecting! 🚀**



