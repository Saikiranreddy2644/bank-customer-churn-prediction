USE bank_churn_db;

-- Check data
SELECT COUNT(*) AS total_customers
FROM customer_churn;

SELECT *
FROM customer_churn
LIMIT 10;

-- Overall churn rate
SELECT 
    ROUND(AVG(Exited) * 100, 2) AS churn_rate_percentage
FROM customer_churn;

-- Churn count
SELECT 
    Exited,
    COUNT(*) AS customer_count
FROM customer_churn
GROUP BY Exited;

-- Churn by geography
SELECT
    Geography,
    COUNT(*) AS total_customers,
    SUM(Exited) AS churned_customers,
    ROUND(AVG(Exited) * 100, 2) AS churn_rate_percentage
FROM customer_churn
GROUP BY Geography
ORDER BY churn_rate_percentage DESC;

-- Churn by gender
SELECT
    Gender,
    COUNT(*) AS total_customers,
    SUM(Exited) AS churned_customers,
    ROUND(AVG(Exited) * 100, 2) AS churn_rate_percentage
FROM customer_churn
GROUP BY Gender
ORDER BY churn_rate_percentage DESC;

-- Churn by age group
SELECT
    age_group,
    COUNT(*) AS total_customers,
    SUM(Exited) AS churned_customers,
    ROUND(AVG(Exited) * 100, 2) AS churn_rate_percentage
FROM customer_churn
GROUP BY age_group
ORDER BY churn_rate_percentage DESC;

-- Churn by credit score band
SELECT
    credit_score_band,
    COUNT(*) AS total_customers,
    SUM(Exited) AS churned_customers,
    ROUND(AVG(Exited) * 100, 2) AS churn_rate_percentage
FROM customer_churn
GROUP BY credit_score_band
ORDER BY churn_rate_percentage DESC;

-- Churn by active member status
SELECT
    IsActiveMember,
    COUNT(*) AS total_customers,
    SUM(Exited) AS churned_customers,
    ROUND(AVG(Exited) * 100, 2) AS churn_rate_percentage
FROM customer_churn
GROUP BY IsActiveMember
ORDER BY churn_rate_percentage DESC;

-- Churn by number of products
SELECT
    NumOfProducts,
    COUNT(*) AS total_customers,
    SUM(Exited) AS churned_customers,
    ROUND(AVG(Exited) * 100, 2) AS churn_rate_percentage
FROM customer_churn
GROUP BY NumOfProducts
ORDER BY NumOfProducts;

-- Average values by churn status
SELECT
    Exited,
    ROUND(AVG(Age), 2) AS avg_age,
    ROUND(AVG(CreditScore), 2) AS avg_credit_score,
    ROUND(AVG(Balance), 2) AS avg_balance,
    ROUND(AVG(EstimatedSalary), 2) AS avg_salary
FROM customer_churn
GROUP BY Exited;

-- View for Power BI dashboard
CREATE OR REPLACE VIEW churn_dashboard_summary AS
SELECT
    Geography,
    Gender,
    age_group,
    credit_score_band,
    IsActiveMember,
    NumOfProducts,
    COUNT(*) AS total_customers,
    SUM(Exited) AS churned_customers,
    ROUND(AVG(Exited) * 100, 2) AS churn_rate_percentage,
    ROUND(AVG(Balance), 2) AS avg_balance,
    ROUND(AVG(EstimatedSalary), 2) AS avg_salary,
    ROUND(AVG(CreditScore), 2) AS avg_credit_score
FROM customer_churn
GROUP BY
    Geography,
    Gender,
    age_group,
    credit_score_band,
    IsActiveMember,
    NumOfProducts;
    
select * from churn_dashboard_summary;