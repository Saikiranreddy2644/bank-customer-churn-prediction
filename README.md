# Bank Customer Churn Analysis and Prediction

End-to-end data science project for analyzing and predicting bank customer churn.

## Project Stages

1. Raw data collection
2. Issue injection for dirty data simulation (`inject_issues.ipynb`)
3. Data audit (`data_audit.ipynb`)
4. Data cleaning + split-first imputation (`src/clean_churn_data.py`)
5. Feature engineering (`src/features.py`, shared across training and app)
6. SQL analysis with MySQL (`sql/`)
7. Machine learning model training (`src/train_churn_model.py`)
8. Model evaluation (`reports/`)
9. Power BI dashboard (`Bank_Churn_dashboard.pbix`)
10. Streamlit prediction app (`app.py`)

## Dataset

Bank customer churn dataset (`data/raw/churn.csv`): credit score, geography, gender,
age, tenure, balance, products, credit card ownership, active membership, estimated
salary, and the churn target (`Exited`).

## Main Outputs

- Cleaned datasets: `data/processed/churn_clean.csv`, `churn_clean_train.csv`, `churn_clean_test.csv`
- SQL scripts: `sql/`
- Trained models: `models/best_churn_model.pkl`, `models/best_churn_model_no_geo.pkl`
- ML reports: `reports/`
- Power BI dashboard: `Bank_Churn_dashboard.pbix`
- Streamlit app: `app.py`

## How to Run

```bash
pip install -r requirements.txt
python src/clean_churn_data.py      # clean + split (leakage-safe)
python src/train_churn_model.py     # train, tune threshold, save models/reports
streamlit run app.py
```

## App Notes

- Choose Geography = "Not specified" to predict with the geography-free model
  variant (`models/best_churn_model_no_geo.pkl`).

## Best Model

Selected by cross-validated ROC-AUC with an F1-tuned probability threshold
(see `reports/model_comparison.csv`).
