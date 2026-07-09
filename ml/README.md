# Machine Learning Stage - Bank Customer Churn

This stage trains and compares churn prediction models using the cleaned dataset.

## Input

- `data/processed/churn_clean.csv`

## Script

- `src/train_churn_model.py`

## Models Compared

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

## Outputs

- `models/best_churn_model.pkl`
- `reports/model_comparison.csv`
- `reports/test_predictions.csv`
- `reports/feature_importance.csv` when the selected model supports feature importance

## Run

```powershell
python src/train_churn_model.py
```

