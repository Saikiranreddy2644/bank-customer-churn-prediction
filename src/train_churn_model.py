from pathlib import Path

import joblib
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "churn_clean.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

TARGET = "Exited"
ID_COLUMN = "CustomerId"


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical_columns = features.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_columns = [column for column in features.columns if column not in categorical_columns]

    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )


def evaluate_model(name: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(precision_score(y_test, predictions), 4),
        "recall": round(recall_score(y_test, predictions), 4),
        "f1_score": round(f1_score(y_test, predictions), 4),
        "roc_auc": round(roc_auc_score(y_test, probabilities), 4),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def save_feature_importance(best_model_name: str, best_pipeline: Pipeline) -> None:
    classifier = best_pipeline.named_steps["classifier"]
    preprocessor = best_pipeline.named_steps["preprocessor"]

    if not hasattr(classifier, "feature_importances_"):
        return

    feature_names = preprocessor.get_feature_names_out()
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": classifier.feature_importances_,
            "model": best_model_name,
        }
    ).sort_values("importance", ascending=False)

    importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    x = df.drop(columns=[TARGET, ID_COLUMN])
    y = df[TARGET]

    x_train, x_test, y_train, y_test, train_index, test_index = train_test_split(
        x,
        y,
        df.index,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    results = []
    trained_pipelines = {}

    for name, classifier in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(x_train)),
                ("classifier", classifier),
            ]
        )
        pipeline.fit(x_train, y_train)
        trained_pipelines[name] = pipeline
        results.append(evaluate_model(name, pipeline, x_test, y_test))

    results_df = pd.DataFrame(results).sort_values(["roc_auc", "f1_score"], ascending=False)
    best_model_name = results_df.iloc[0]["model"]
    best_pipeline = trained_pipelines[best_model_name]

    results_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    joblib.dump(best_pipeline, MODELS_DIR / "best_churn_model.pkl")

    test_predictions = best_pipeline.predict(x_test)
    test_probabilities = best_pipeline.predict_proba(x_test)[:, 1]
    prediction_report = df.loc[test_index, [ID_COLUMN]].copy()
    prediction_report["actual_churn"] = y_test.values
    prediction_report["predicted_churn"] = test_predictions
    prediction_report["churn_probability"] = test_probabilities.round(4)
    prediction_report.to_csv(REPORTS_DIR / "test_predictions.csv", index=False)

    save_feature_importance(best_model_name, best_pipeline)

    print("Machine learning training completed.")
    print(f"Best model: {best_model_name}")
    print(results_df.to_string(index=False))
    print(f"Saved model: {MODELS_DIR / 'best_churn_model.pkl'}")
    print(f"Saved reports: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
