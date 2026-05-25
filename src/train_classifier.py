"""
Train deal quality classification models for DealScore.

Models trained:
    1. Logistic Regression baseline
    2. Random Forest Classifier
    3. XGBoost Classifier

Best model is saved to:
    models/deal_classifier.pkl
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from preprocess import PROCESSED_DATA_PATH, preprocess_data
from train_regression import FEATURE_COLUMNS, MODEL_PATH as REGRESSION_MODEL_PATH


ROOT_DIR = Path(__file__).resolve().parents[1]
CLASSIFIER_MODEL_PATH = ROOT_DIR / "models" / "deal_classifier.pkl"
RANDOM_STATE = 42

CATEGORICAL_FEATURES = ["brand", "model", "fuel_type", "transmission", "owner_type", "seller_type"]
NUMERIC_FEATURES = [
    "year",
    "km_driven",
    "mileage",
    "engine",
    "seats",
    "car_age",
    "km_per_year",
    "engine_per_seat",
    "listed_price",
    "predicted_fair_price",
    "price_difference_percent",
]
CLASSIFIER_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def calculate_deal_score(listed_price: float, fair_price: float) -> int:
    """Return a 1-10 score where cheaper-than-fair listings score higher."""
    if fair_price <= 0:
        return 5
    discount_percent = ((fair_price - listed_price) / fair_price) * 100
    raw_score = 5 + (discount_percent / 5)
    return int(np.clip(round(raw_score), 1, 10))


def classify_deal(listed_price: float, fair_price: float) -> str:
    """Convert fair-price comparison into a readable deal category."""
    if fair_price <= 0:
        return "Fair Deal"
    difference_percent = ((listed_price - fair_price) / fair_price) * 100
    if difference_percent <= -10:
        return "Great Deal"
    if difference_percent <= 10:
        return "Fair Deal"
    return "Bad Deal"


def load_cleaned_data() -> pd.DataFrame:
    """Load cleaned data, creating it from raw data if needed."""
    if not PROCESSED_DATA_PATH.exists():
        preprocess_data()
    return pd.read_csv(PROCESSED_DATA_PATH)


def create_classification_dataset() -> pd.DataFrame:
    """Create labels using the trained fair-price model and listed price."""
    if not REGRESSION_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Regression model not found. Run `python src/train_regression.py` before training the classifier."
        )

    df = load_cleaned_data()
    fair_price_model = joblib.load(REGRESSION_MODEL_PATH)
    df["listed_price"] = df["price"]
    df["predicted_fair_price"] = fair_price_model.predict(df[FEATURE_COLUMNS])
    df["price_difference_percent"] = (
        (df["listed_price"] - df["predicted_fair_price"]) / df["predicted_fair_price"]
    ) * 100
    df["deal_score"] = df.apply(
        lambda row: calculate_deal_score(row["listed_price"], row["predicted_fair_price"]),
        axis=1,
    )
    df["deal_label"] = df.apply(
        lambda row: classify_deal(row["listed_price"], row["predicted_fair_price"]),
        axis=1,
    )
    return df


def build_preprocessor() -> ColumnTransformer:
    """Prepare categorical and numeric features for classifiers."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def get_models() -> dict[str, object]:
    """Create candidate classification models."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=250,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost Classifier": XGBClassifier(
            n_estimators=250,
            learning_rate=0.06,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
        ),
    }


def train_classifier_model() -> tuple[Pipeline, pd.DataFrame]:
    """Train all classifier models and save the best one by accuracy."""
    df = create_classification_dataset()
    X = df[CLASSIFIER_FEATURES]
    y = df["deal_label"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_counts = pd.Series(y_encoded).value_counts()
    if len(class_counts) < 2:
        raise ValueError(
            "Classifier training needs at least two deal categories. "
            "Try a larger dataset or adjust the deal thresholds after reviewing residuals."
        )

    stratify_values = y_encoded if class_counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify_values,
    )

    results = []
    best_model = None
    best_accuracy = -1.0
    best_name = ""

    for model_name, model in get_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
        recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
        results.append(
            {
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
            }
        )

        print(f"\n{model_name}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, predictions))
        print("Classification Report:")
        print(classification_report(y_test, predictions, target_names=label_encoder.classes_, zero_division=0))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = pipeline
            best_name = model_name

    model_package = {
        "pipeline": best_model,
        "label_encoder": label_encoder,
        "feature_columns": CLASSIFIER_FEATURES,
    }
    CLASSIFIER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_package, CLASSIFIER_MODEL_PATH)

    results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
    print("\nClassifier model comparison:")
    print(results_df.to_string(index=False))
    print(f"\nBest model: {best_name}")
    print(f"Saved model to: {CLASSIFIER_MODEL_PATH}")
    return best_model, results_df


if __name__ == "__main__":
    train_classifier_model()
