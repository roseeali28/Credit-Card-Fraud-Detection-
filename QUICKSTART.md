# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Download Dataset

1. Download the Credit Card Fraud Detection dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Place `creditcard.csv` in the `data/` directory

**Alternative**: If you have the dataset elsewhere, you can specify the path when running training.

### Step 3: Train Your First Model

```bash
# Train a Random Forest model with class weights
python main.py --model random_forest --use-class-weights
```

This will:
- Load and preprocess the data
- Train a Random Forest model
- Evaluate on validation and test sets
- Save the model to `models/`
- Generate evaluation plots in `reports/`

### Step 4: Make Predictions

```bash
# Predict on a CSV file
python predict.py --model models/random_forest_classweights.pkl --data data/creditcard.csv --output predictions.csv
```

### Step 5: Launch Web Interface

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` and start making predictions!

## 📊 Common Use Cases

### Train with SMOTE Oversampling

```bash
python main.py --model random_forest --use-smote
```

### Train XGBoost with Hyperparameter Tuning

```bash
python main.py --model xgboost --tune-hyperparameters --use-class-weights
```

### Compare Multiple Models

```bash
# Train different models
python main.py --model logistic_regression --use-class-weights
python main.py --model random_forest --use-class-weights
python main.py --model xgboost --use-class-weights
python main.py --model lightgbm --use-class-weights

# Compare results in reports/ directory
```

## 🔍 View Results

After training, check:
- **Metrics**: `reports/*_metrics.json`
- **Plots**: `reports/*.png` (ROC curves, PR curves, confusion matrices)
- **Feature Importance**: `reports/feature_importance.png`
- **SHAP Plots**: `reports/shap_summary.png` (if available)

## 🧪 Run Tests

```bash
pytest tests/ -v
```

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore the [EDA notebook](notebooks/eda.ipynb) for data analysis
- Customize models and parameters in `main.py`
- Extend the Streamlit app in `app.py`

## ❓ Troubleshooting

**Issue**: "Dataset not found"
- **Solution**: Make sure `creditcard.csv` is in the `data/` directory

**Issue**: "No module named 'src'"
- **Solution**: Run commands from the project root directory

**Issue**: SHAP errors
- **Solution**: SHAP may not work with all models. The code will skip it gracefully.

**Issue**: Memory errors with large datasets
- **Solution**: Use smaller sample sizes or reduce hyperparameter tuning iterations

---

Happy training! 🎉


