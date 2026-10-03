import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from features import add_features, get_age_group, get_credit_score_band

MODEL_PATH = ROOT / "models" / "best_churn_model.pkl"
NO_GEO_MODEL_PATH = ROOT / "models" / "best_churn_model_no_geo.pkl"


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


def build_customer_record(inputs: dict) -> pd.DataFrame:
    record = {
        "CreditScore": inputs["CreditScore"],
        "Geography": inputs["Geography"],
        "Gender": inputs["Gender"],
        "Age": inputs["Age"],
        "Tenure": inputs["Tenure"],
        "Balance": inputs["Balance"],
        "NumOfProducts": inputs["NumOfProducts"],
        "HasCrCard": inputs["HasCrCard"],
        "IsActiveMember": inputs["IsActiveMember"],
        "EstimatedSalary": inputs["EstimatedSalary"],
    }
    return add_features(pd.DataFrame([record]))


def explain_prediction(inputs: dict, probability: float, no_geography: bool) -> list[str]:
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

    if not no_geography and inputs["Geography"] == "Germany":
        reasons.append("Germany showed higher churn tendency in the dataset analysis.")

    if probability >= 0.7:
        reasons.append("The model confidence for churn is high.")
    elif probability >= 0.5:
        reasons.append("The model predicts churn, but confidence is moderate.")
    else:
        reasons.append("The model predicts lower churn risk for this customer.")

    return reasons


st.set_page_config(page_title="Bank Churn Prediction", layout="wide", page_icon="🏦")

st.markdown(
    """
    <style>
    .main { background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%); }
    .hero {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        padding: 28px 32px; border-radius: 16px; color: white;
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.25); margin-bottom: 24px;
    }
    .hero h1 { color: white; margin: 0; font-size: 2rem; }
    .hero p { color: #dbeafe; margin: 6px 0 0 0; }
    .card {
        background: white; padding: 20px 24px; border-radius: 14px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08); margin-bottom: 16px;
        border: 1px solid #e2e8f0;
    }
    .metric-card {
        background: white; border-radius: 14px; padding: 18px 22px;
        border-left: 5px solid #2563eb; box-shadow: 0 4px 14px rgba(15,23,42,0.06);
    }
    .metric-card h2 { margin: 0; font-size: 1.8rem; color: #0f172a; }
    .metric-card p { margin: 0; color: #64748b; font-weight: 600; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; }
    .risk-high { border-left-color: #dc2626; }
    .risk-low { border-left-color: #16a34a; }
    div[data-testid="stForm"] { background: white; border-radius: 14px; padding: 20px; box-shadow: 0 4px 14px rgba(15,23,42,0.06); border: 1px solid #e2e8f0; }
    </style>
    <div class="hero">
        <h1>Bank Customer Churn Prediction</h1>
        <p>Enter customer details below to estimate churn risk using the trained ML model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

model_bundle = load_model(str(MODEL_PATH))
model = model_bundle["pipeline"]
threshold = model_bundle["threshold"]

no_geo_bundle = load_model(str(NO_GEO_MODEL_PATH))
no_geo_model = no_geo_bundle["pipeline"]
no_geo_threshold = no_geo_bundle["threshold"]

st.subheader("📋 Customer Details")

with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
        geography_options = ["France", "Germany", "Spain", "Not specified"]
        geography = st.selectbox("Geography", geography_options)
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
    no_geography = geography == "Not specified"
    user_inputs = {
        "CreditScore": int(credit_score),
        "Geography": None if no_geography else geography,
        "Gender": gender,
        "Age": int(age),
        "Tenure": int(tenure),
        "Balance": float(balance),
        "NumOfProducts": int(products),
        "HasCrCard": 1 if has_card == "Yes" else 0,
        "IsActiveMember": 1 if active_member == "Yes" else 0,
        "EstimatedSalary": float(salary),
    }

    customer = build_customer_record({**user_inputs, "Geography": "France" if no_geography else geography})

    if no_geography:
        model_input = customer.drop(columns=["Geography"])
        probability = float(no_geo_model.predict_proba(model_input)[0][1])
        prediction = int(probability >= no_geo_threshold)
        st.info("No geography provided — using the geography-free model variant.")
    else:
        probability = float(model.predict_proba(customer)[0][1])
        prediction = int(probability >= threshold)

    st.divider()

    risk_class = "risk-high" if prediction == 1 else "risk-low"
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.markdown(
            f"<div class='metric-card {risk_class}'><p>Churn Probability</p><h2>{probability * 100:.2f}%</h2></div>",
            unsafe_allow_html=True,
        )
    with metric_col2:
        label = "Likely to Churn" if prediction == 1 else "Not Likely to Churn"
        st.markdown(
            f"<div class='metric-card {risk_class}'><p>Prediction</p><h2>{label}</h2></div>",
            unsafe_allow_html=True,
        )
    st.write("")

    if prediction == 1:
        st.error("This customer is predicted as high churn risk.")
    else:
        st.success("This customer is predicted as lower churn risk.")

    st.subheader("Possible Reasons" if prediction == 1 else "Customer Risk Notes")
    for reason in explain_prediction(user_inputs, probability, no_geography):
        st.markdown(f"<div class='card'>{reason}</div>", unsafe_allow_html=True)

    with st.expander("🔧 Engineered features used internally"):
        st.dataframe(customer.drop(columns=["Geography"] if no_geography else []), use_container_width=True)


@st.cache_data
def load_clean_data() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "processed" / "churn_clean.csv")


def churn_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        df.groupby(column)["Exited"]
        .mean()
        .mul(100)
        .round(2)
        .rename("churn_rate_%")
        .to_frame()
    )


st.divider()
st.header("📊 Churn Insights from the Dataset")

df_clean = load_clean_data()

overall = round(df_clean["Exited"].mean() * 100, 2)
st.metric("Overall churn rate", f"{overall}%")

insight_col1, insight_col2 = st.columns(2)
with insight_col1:
    st.subheader("Churn rate by Geography")
    st.bar_chart(churn_rate_by(df_clean, "Geography"))
    st.subheader("Churn rate by Gender")
    st.bar_chart(churn_rate_by(df_clean, "Gender"))
    st.subheader("Churn rate by Active Membership")
    st.bar_chart(churn_rate_by(df_clean, "IsActiveMember"))
with insight_col2:
    st.subheader("Churn rate by Age Group")
    st.bar_chart(churn_rate_by(df_clean, "age_group"))
    st.subheader("Churn rate by Credit Score Band")
    st.bar_chart(churn_rate_by(df_clean, "credit_score_band"))
    st.subheader("Churn rate by Number of Products")
    st.bar_chart(churn_rate_by(df_clean, "NumOfProducts"))
