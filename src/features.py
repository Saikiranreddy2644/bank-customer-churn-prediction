import pandas as pd

BASE_COLUMNS = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

ENGINEERED_COLUMNS = [
    "zero_balance_flag",
    "age_group",
    "credit_score_band",
    "products_per_tenure",
    "salary_to_balance_ratio",
]


def get_age_group(age) -> str:
    if age is None or (isinstance(age, float) and pd.isna(age)):
        return "Unknown"
    age = int(age)
    if age <= 30:
        return "Young (18-30)"
    if age <= 50:
        return "Mid (31-50)"
    return "Senior (51+)"


def get_credit_score_band(credit_score) -> str:
    if credit_score is None or (isinstance(credit_score, float) and pd.isna(credit_score)):
        return "Unknown"
    score = float(credit_score)
    if score <= 579:
        return "Poor"
    if score <= 669:
        return "Fair"
    if score <= 739:
        return "Good"
    return "Excellent"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["zero_balance_flag"] = (out["Balance"] == 0).astype(int)
    out["age_group"] = pd.cut(
        out["Age"],
        bins=[17, 30, 50, 100],
        labels=["Young (18-30)", "Mid (31-50)", "Senior (51+)"],
    ).astype("object").fillna("Unknown")
    out["credit_score_band"] = pd.cut(
        out["CreditScore"],
        bins=[299, 579, 669, 739, 850],
        labels=["Poor", "Fair", "Good", "Excellent"],
    ).astype("object").fillna("Unknown")
    out["products_per_tenure"] = (out["NumOfProducts"] / (out["Tenure"] + 1)).round(3)
    out["salary_to_balance_ratio"] = (out["Balance"] / (out["EstimatedSalary"] + 1)).round(4)
    return out
