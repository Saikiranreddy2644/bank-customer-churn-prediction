from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from features import add_features

ROOT = Path(__file__).resolve().parents[1]
DIRTY_PATH = ROOT / "data" / "raw" / "churn_dirty.csv"
OUT_DIR = ROOT / "data" / "processed"


def clean() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(DIRTY_PATH)

    for col in ["RowNumber", "Surname"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = df.drop_duplicates()

    df["Geography"] = df["Geography"].astype(str).str.strip().str.title()
    df["Gender"] = df["Gender"].astype(str).str.strip().str.title()
    df["Gender"] = df["Gender"].replace({"M": "Male", "F": "Female"})

    df = df[(df["Age"].isna()) | ((df["Age"] >= 18) & (df["Age"] <= 100))]
    df = df[(df["CreditScore"].isna()) | ((df["CreditScore"] >= 300) & (df["CreditScore"] <= 850))]
    df = df[df["EstimatedSalary"] >= 0]

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["Exited"]
    )

    credit_median = train_df["CreditScore"].median()
    age_median = train_df["Age"].median()
    balance_median_by_geo = train_df.groupby("Geography")["Balance"].median()
    global_balance_median = train_df["Balance"].median()

    def impute(part: pd.DataFrame) -> pd.DataFrame:
        part = part.copy()
        part["CreditScore"] = part["CreditScore"].fillna(credit_median)
        part["Age"] = part["Age"].fillna(age_median)
        part["Balance"] = part["Balance"].fillna(
            part["Geography"].map(balance_median_by_geo).fillna(global_balance_median)
        )
        part["HasCrCard"] = part["HasCrCard"].fillna(0)
        part["IsActiveMember"] = part["IsActiveMember"].fillna(0)
        for col in ["Age", "CreditScore", "HasCrCard", "IsActiveMember", "Exited"]:
            part[col] = part[col].astype(int)
        return part

    train_clean = add_features(impute(train_df))
    test_clean = add_features(impute(test_df))

    full_clean = add_features(impute(df))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_clean.to_csv(OUT_DIR / "churn_clean_train.csv", index=False)
    test_clean.to_csv(OUT_DIR / "churn_clean_test.csv", index=False)
    full_clean.to_csv(OUT_DIR / "churn_clean.csv", index=False)

    print(f"train: {train_clean.shape}, test: {test_clean.shape}, full: {full_clean.shape}")
    return train_clean, test_clean


if __name__ == "__main__":
    clean()
