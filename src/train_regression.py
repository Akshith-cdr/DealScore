"""
Train fair price regression models for DealScore.

Models trained:
    1. Linear Regression baseline
    2. Random Forest Regressor
    3. XGBoost Regressor

Best model is saved to:
    models/fair_price_model.pkl
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from preprocess import PROCESSED_DATA_PATH, preprocess_data


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "fair_price_model.pkl"
RANDOM_STATE = 42

TARGET_COLUMN = "price"
CATEGORICAL_FEATURES = ["brand", "model", "fuel_type", "transmission", "owner_type", "seller_type"]
NUMERIC_FEATURES = ["year", "km_driven", "mileage", "engine", "seats", "car_age", "km_per_year", "engine_per_seat"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def load_training_data() -> pd.DataFrame:
    """Load cleaned data, creating it from raw data if needed."""
    if not PROCESSED_DATA_PATH.exists():
        preprocess_data()
    return pd.read_csv(PROCESSED_DATA_PATH)


def build_preprocessor() -> ColumnTransformer:
    """Encode categorical features and scale numeric features."""
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
    """Create candidate regression models."""
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=250,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost Regressor": XGBRegressor(
            n_estimators=300,
            learning_rate=0.06,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate beginner-friendly regression metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def train_regression_model() -> tuple[Pipeline, pd.DataFrame]:
    """Train all regression models and save the best one by lowest RMSE."""
    df = load_training_data()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    results = []
    best_model = None
    best_rmse = float("inf")
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
        metrics = evaluate_regression(y_test, predictions)
        results.append({"Model": model_name, **metrics})

        if metrics["RMSE"] < best_rmse:
            best_rmse = metrics["RMSE"]
            best_model = pipeline
            best_name = model_name

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    results_df = pd.DataFrame(results).sort_values("RMSE")
    print("\nRegression model comparison:")
    print(results_df.to_string(index=False))
    print(f"\nBest model: {best_name}")
    print(f"Saved model to: {MODEL_PATH}")
    return best_model, results_df


if __name__ == "__main__":
    train_regression_model()
