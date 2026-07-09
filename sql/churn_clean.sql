CREATE DATABASE IF NOT EXISTS bank_churn_db;
USE bank_churn_db;

CREATE TABLE customers_churn (
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

select * from churn_clean limit 10;

DROP TABLE customers_churn;
RENAME TABLE churn_clean TO customer_churn;
SELECT count(*) FROM customer_churn;
