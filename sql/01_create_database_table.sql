CREATE DATABASE IF NOT EXISTS bank_churn_db;
USE bank_churn_db;

CREATE TABLE IF NOT EXISTS customer_churn (
    CustomerId BIGINT,
    CreditScore INT,
    Geography VARCHAR(50),
    Gender VARCHAR(20),
    Age INT,
    Tenure INT,
    Balance DECIMAL(12,2),
    NumOfProducts INT,
    HasCrCard INT,
    IsActiveMember INT,
    EstimatedSalary DECIMAL(12,2),
    Exited INT,
    zero_balance_flag INT,
    age_group VARCHAR(50),
    credit_score_band VARCHAR(50),
    products_per_tenure DECIMAL(10,3),
    salary_to_balance_ratio DECIMAL(12,4)
);

SELECT COUNT(*) AS total_rows FROM customer_churn;
