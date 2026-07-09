from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_churn_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def get_age_group(age: int) -> str:
    if age <= 30:
        return "Young (18-30)"
    if age <= 50:
        return "Mid (31-50)"
    return "Senior (51+)"


def get_credit_score_band(credit_score: int) -> str:
    if credit_score <= 579:
        return "Poor"
    if credit_score <= 669:
        return "Fair"
    if credit_score <= 739:
        return "Good"
    return "Excellent"


def build_customer_record(inputs: dict) -> pd.DataFrame:
    balance = inputs["Balance"]
    salary = inputs["EstimatedSalary"]
    tenure = inputs["Tenure"]

    record = {
        "CreditScore": inputs["CreditScore"],
        "Geography": inputs["Geography"],
        "Gender": inputs["Gender"],
        "Age": inputs["Age"],
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": inputs["NumOfProducts"],
        "HasCrCard": inputs["HasCrCard"],
        "IsActiveMember": inputs["IsActiveMember"],
        "EstimatedSalary": salary,
        "zero_balance_flag": int(balance == 0),
        "age_group": get_age_group(inputs["Age"]),
        "credit_score_band": get_credit_score_band(inputs["CreditScore"]),
        "products_per_tenure": round(inputs["NumOfProducts"] / (tenure + 1), 3),
        "salary_to_balance_ratio": round(balance / (salary + 1), 4),
    }

    return pd.DataFrame([record])


def explain_prediction(inputs: dict, probability: float) -> list[str]:
    reasons = []

    if inputs["IsActiveMember"] == 0:
        reasons.append("Customer is not an active member, which increases churn risk.")
    else:
        reasons.append("Customer is active, which helps reduce churn risk.")

    if inputs["CreditScore"] < 580:
        reasons.append("Customer has a poor credit score band.")
    elif inputs["CreditScore"] < 670:
        reasons.append("Customer has a fair credit score, so risk is moderate.")
    else:
        reasons.append("Customer has a good credit profile.")

    if inputs["Age"] > 50:
        reasons.append("Customer belongs to the senior age group, which showed higher churn tendency.")
    elif inputs["Age"] <= 30:
        reasons.append("Customer is in the younger age group.")

    if inputs["Balance"] > 100000:
        reasons.append("Customer has a high account balance, so losing this customer may have business impact.")
    elif inputs["Balance"] == 0:
        reasons.append("Customer has zero account balance.")

    if inputs["NumOfProducts"] >= 3:
        reasons.append("Customer uses many products, which can indicate possible dissatisfaction in this dataset.")
    elif inputs["NumOfProducts"] == 1:
        reasons.append("Customer uses only one product, so cross-selling may improve retention.")

    if inputs["Geography"] == "Germany":
        reasons.append("Germany showed higher churn tendency in the dataset analysis.")
    elif inputs["Geography"] == "Other":
        reasons.append("This region was not present in the training data, so the geography effect is treated as unknown.")

    if probability >= 0.7:
        reasons.append("The model confidence for churn is high.")
    elif probability >= 0.5:
        reasons.append("The model predicts churn, but confidence is moderate.")
    else:
        reasons.append("The model predicts lower churn risk for this customer.")

    return reasons


st.set_page_config(page_title="Bank Churn Prediction", layout="wide")

st.title("Bank Customer Churn Prediction")
st.caption("Enter customer details to predict churn risk using the trained Random Forest model.")

model = load_model()

with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain", "Other"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.number_input("Age", min_value=18, max_value=100, value=40)

    with col2:
        tenure = st.number_input("Tenure", min_value=0, max_value=10, value=3)
        balance = st.number_input("Balance", min_value=0.0, value=75000.0, step=1000.0)
        products = st.selectbox("Number of Products", [1, 2, 3, 4])

    with col3:
        has_card = st.selectbox("Has Credit Card", ["Yes", "No"])
        active_member = st.selectbox("Is Active Member", ["Yes", "No"])
        salary = st.number_input("Estimated Salary", min_value=0.0, value=100000.0, step=1000.0)

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    user_inputs = {
        "CreditScore": int(credit_score),
        "Geography": geography,
        "Gender": gender,
        "Age": int(age),
        "Tenure": int(tenure),
        "Balance": float(balance),
        "NumOfProducts": int(products),
        "HasCrCard": 1 if has_card == "Yes" else 0,
        "IsActiveMember": 1 if active_member == "Yes" else 0,
        "EstimatedSalary": float(salary),
    }

    customer = build_customer_record(user_inputs)
    prediction = int(model.predict(customer)[0])
    probability = float(model.predict_proba(customer)[0][1])

    st.divider()

    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Churn Probability", f"{probability * 100:.2f}%")
    metric_col2.metric("Prediction", "Likely to Churn" if prediction == 1 else "Not Likely to Churn")

    if prediction == 1:
        st.error("This customer is predicted as high churn risk.")
        st.subheader("Possible Reasons")
    else:
        st.success("This customer is predicted as lower churn risk.")
        st.subheader("Customer Risk Notes")

    if geography == "Other":
        st.warning(
            "The model was trained only on France, Germany, and Spain. "
            "For other regions, the prediction can still run, but it should be treated as an estimate."
        )

    for reason in explain_prediction(user_inputs, probability):
        st.write(f"- {reason}")

    st.subheader("Engineered Features Used Internally")
    st.dataframe(customer, use_container_width=True)
