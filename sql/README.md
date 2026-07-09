# SQL Stage - Bank Customer Churn

This folder contains the MySQL scripts for the SQL analysis stage.

## Files

- `01_create_database_table.sql` creates the MySQL database and `customer_churn` table.
- `02_analysis_queries.sql` contains business analysis queries and a Power BI summary view.

## Recommended Workflow

1. Run `01_create_database_table.sql` in MySQL Workbench.
2. Import `data/processed/churn_clean.csv` into the existing `customer_churn` table using Table Data Import Wizard.
3. Run the count check:

```sql
SELECT COUNT(*) FROM customer_churn;
```

4. Run `02_analysis_queries.sql` for SQL analysis.
5. Connect Power BI to `bank_churn_db` and use `customer_churn` or `churn_dashboard_summary`.

