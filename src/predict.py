"""
Prediction pipeline for DealScore.

This script accepts one car listing, predicts fair market price, compares it
with the listed price, generates a DealScore, and returns a deal category.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
FAIR_PRICE_MODEL_PATH = ROOT_DIR / "models" / "fair_price_model.pkl"
DEAL_CLASSIFIER_PATH = ROOT_DIR / "models" / "deal_classifier.pkl"
CURRENT_YEAR = 2026

REGRESSION_FEATURES = [
    "brand",
    "model",
    "fuel_type",
    "transmission",
    "owner_type",
    "seller_type",
    "year",
    "km_driven",
    "mileage",
    "engine",
    "seats",
    "car_age",
    "km_per_year",
    "engine_per_seat",
]


def calculate_deal_score(listed_price: float, fair_price: float) -> int:
    """Return a 1-10 DealScore where stronger discounts receive higher scores."""
    if fair_price <= 0:
        return 5
    discount_percent = ((fair_price - listed_price) / fair_price) * 100
    raw_score = 5 + (discount_percent / 5)
    return int(np.clip(round(raw_score), 1, 10))


def rule_based_deal_category(listed_price: float, fair_price: float) -> str:
    """Fallback deal category rule based on listed price versus fair price."""
    if fair_price <= 0:
        return "Fair Deal"
    difference_percent = ((listed_price - fair_price) / fair_price) * 100
    if difference_percent <= -10:
        return "Great Deal"
    if difference_percent <= 10:
        return "Fair Deal"
    return "Bad Deal"


def prepare_input(car_details: dict[str, Any]) -> pd.DataFrame:
    """Convert user input into the same feature format used during training."""
    row = car_details.copy()
    row["car_age"] = CURRENT_YEAR - int(row["year"])
    row["km_per_year"] = float(row["km_driven"]) / max(row["car_age"], 1)
    row["engine_per_seat"] = float(row["engine"]) / max(float(row["seats"]), 1)

    df = pd.DataFrame([row])
    return df


def predict_deal(car_details: dict[str, Any]) -> dict[str, Any]:
    """Predict fair price, DealScore, and deal category for one car listing."""
    if not FAIR_PRICE_MODEL_PATH.exists():
        raise FileNotFoundError("Fair price model not found. Run `python src/train_regression.py` first.")

    fair_price_model = joblib.load(FAIR_PRICE_MODEL_PATH)
    input_df = prepare_input(car_details)

    fair_price = float(fair_price_model.predict(input_df[REGRESSION_FEATURES])[0])
    listed_price = float(car_details["listed_price"])
    deal_score = calculate_deal_score(listed_price, fair_price)

    category = rule_based_deal_category(listed_price, fair_price)
    if DEAL_CLASSIFIER_PATH.exists():
        classifier_package = joblib.load(DEAL_CLASSIFIER_PATH)
        classifier_input = input_df.copy()
        classifier_input["listed_price"] = listed_price
        classifier_input["predicted_fair_price"] = fair_price
        classifier_input["price_difference_percent"] = ((listed_price - fair_price) / fair_price) * 100

        encoded_prediction = classifier_package["pipeline"].predict(
            classifier_input[classifier_package["feature_columns"]]
        )[0]
        category = classifier_package["label_encoder"].inverse_transform([encoded_prediction])[0]

    savings = fair_price - listed_price
    return {
        "predicted_fair_price": round(fair_price, 2),
        "listed_price": round(listed_price, 2),
        "estimated_savings": round(savings, 2),
        "deal_score": deal_score,
        "deal_category": category,
    }


if __name__ == "__main__":
    sample_car = {
        "brand": "Maruti",
        "model": "Swift",
        "year": 2018,
        "fuel_type": "Petrol",
        "transmission": "Manual",
        "owner_type": "First Owner",
        "km_driven": 45000,
        "seller_type": "Individual",
        "mileage": 21.4,
        "engine": 1197,
        "seats": 5,
        "listed_price": 450000,
    }
    print(predict_deal(sample_car))
