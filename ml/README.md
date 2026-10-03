# Machine Learning Stage - Bank Customer Churn

This stage trains and compares churn prediction models using the cleaned,
split train/test datasets produced by `src/clean_churn_data.py`.

## Inputs

- `data/processed/churn_clean_train.csv`
- `data/processed/churn_clean_test.csv`

## Scripts

- `src/clean_churn_data.py` - cleaning, train/test split, imputation, feature engineering
- `src/features.py` - shared feature engineering used by training and the app
- `src/train_churn_model.py` - model training, threshold tuning, evaluation

## Models Compared

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Each model is evaluated with 5-fold cross-validation; the probability threshold is
tuned to maximize F1 on out-of-fold predictions. Two variants are trained: one with
Geography and one geography-free (used when the user selects "Not specified").

## Outputs

- `models/best_churn_model.pkl`
- `models/best_churn_model_no_geo.pkl`
- `reports/model_comparison.csv`
- `reports/test_predictions.csv`
- `reports/feature_importance.csv`
- `reports/no_geo_model_comparison.csv`
- `reports/no_geo_test_predictions.csv`
- `reports/no_geo_feature_importance.csv`

## Run

```powershell
python src/clean_churn_data.py
python src/train_churn_model.py
```
