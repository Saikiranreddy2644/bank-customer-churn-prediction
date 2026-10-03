from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "processed" / "churn_clean_train.csv"
TEST_PATH = ROOT / "data" / "processed" / "churn_clean_test.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

TARGET = "Exited"
ID_COLUMN = "CustomerId"


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical_columns = features.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_columns = [c for c in features.columns if c not in categorical_columns]
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )


def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced", random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


def tune_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.arange(0.1, 0.91, 0.05):
        preds = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_threshold, best_f1 = float(threshold), float(score)
    return best_threshold


def evaluate(name, pipeline, x_test, y_test, threshold):
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    return {
        "model": name,
        "threshold": round(threshold, 2),
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(precision_score(y_test, predictions, zero_division=0), 4),
        "recall": round(recall_score(y_test, predictions, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, predictions, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, probabilities), 4),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }, probabilities, predictions


def save_feature_importance(best_model_name, best_pipeline, filename):
    classifier = best_pipeline.named_steps["classifier"]
    preprocessor = best_pipeline.named_steps["preprocessor"]
    if not hasattr(classifier, "feature_importances_"):
        return
    importance = pd.DataFrame(
        {
            "feature": preprocessor.get_feature_names_out(),
            "importance": classifier.feature_importances_,
            "model": best_model_name,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(REPORTS_DIR / filename, index=False)


def train_variant(train_df, test_df, drop_columns, model_filename, report_prefix):
    x_train = train_df.drop(columns=[TARGET, ID_COLUMN] + drop_columns, errors="ignore")
    y_train = train_df[TARGET]
    x_test = test_df.drop(columns=[TARGET, ID_COLUMN] + drop_columns, errors="ignore")
    y_test = test_df[TARGET]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    best = None

    for name, classifier in get_models().items():
        pipeline = Pipeline(
            steps=[("preprocessor", build_preprocessor(x_train)), ("classifier", classifier)]
        )
        cv_auc = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="roc_auc").mean()
        oof_probs = cross_val_predict(pipeline, x_train, y_train, cv=cv, method="predict_proba")[:, 1]
        threshold = tune_threshold(y_train, oof_probs)

        pipeline.fit(x_train, y_train)
        row, probs, preds = evaluate(name, pipeline, x_test, y_test, threshold)
        row["cv_roc_auc"] = round(float(cv_auc), 4)
        results.append(row)

        if best is None or (row["cv_roc_auc"], row["f1_score"]) > (best[0]["cv_roc_auc"], best[0]["f1_score"]):
            best = (row, pipeline, probs, preds, threshold)

    prefix = f"{report_prefix}_" if report_prefix else ""
    results_df = pd.DataFrame(results).sort_values(["cv_roc_auc", "f1_score"], ascending=False)
    results_df.to_csv(REPORTS_DIR / f"{prefix}model_comparison.csv", index=False)

    best_row, best_pipeline, probs, preds, threshold = best
    joblib.dump({"pipeline": best_pipeline, "threshold": threshold}, MODELS_DIR / model_filename)

    prediction_report = test_df[[ID_COLUMN]].copy()
    prediction_report["actual_churn"] = y_test.values
    prediction_report["predicted_churn"] = preds
    prediction_report["churn_probability"] = probs.round(4)
    prediction_report.to_csv(REPORTS_DIR / f"{prefix}test_predictions.csv", index=False)

    save_feature_importance(best_row["model"], best_pipeline, f"{prefix}feature_importance.csv")

    print(f"[{report_prefix or 'full'}] Best: {best_row['model']} (cv_auc={best_row['cv_roc_auc']}, test_auc={best_row['roc_auc']}, threshold={threshold})")
    return results_df


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_variant(train_df, test_df, drop_columns=[], model_filename="best_churn_model.pkl", report_prefix="")
    train_variant(train_df, test_df, drop_columns=["Geography"], model_filename="best_churn_model_no_geo.pkl", report_prefix="no_geo")

    print("Machine learning training completed.")


if __name__ == "__main__":
    main()
