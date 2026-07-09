# Bank Customer Churn Analysis and Prediction

End-to-end data science project for analyzing and predicting bank customer churn.

## Project Stages

1. Raw data collection
2. Issue injection for dirty data simulation
3. Data audit
4. Data cleaning
5. Feature engineering
6. SQL analysis with MySQL
7. Machine learning model training
8. Model evaluation
9. Power BI dashboard
10. Streamlit prediction app

## Main Outputs

- Cleaned dataset: `data/processed/churn_clean.csv`
- SQL scripts: `sql/`
- Best trained model: `models/best_churn_model.pkl`
- ML reports: `reports/`
- Power BI dashboard: `Bank_Churn_dashboard.pbix`
- Streamlit app: `app.py`

## Run Streamlit App Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Best Model

The selected model is Random Forest because it provided the best balance for churn prediction, especially recall and F1-score.

