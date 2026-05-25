"""
Streamlit app for DealScore.

Run with:
    streamlit run app/app.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.append(str(SRC_DIR))

from predict import FAIR_PRICE_MODEL_PATH, predict_deal  # noqa: E402


st.set_page_config(page_title="DealScore", page_icon="DS", layout="centered")

st.title("DealScore")
st.subheader("Fair Price Prediction for Indian Used Cars")

if not FAIR_PRICE_MODEL_PATH.exists():
    st.warning("Train the models first: run `python src/train_regression.py` and `python src/train_classifier.py`.")

with st.form("car_input_form"):
    brand = st.text_input("Brand", value="Maruti")
    model = st.text_input("Model", value="Swift")

    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("Year", min_value=1990, max_value=2026, value=2018)
        km_driven = st.number_input("KM Driven", min_value=0, max_value=500000, value=45000, step=1000)
        mileage = st.number_input("Mileage (kmpl)", min_value=0.0, max_value=50.0, value=21.4, step=0.1)
        listed_price = st.number_input("Listed Price (INR)", min_value=10000, value=450000, step=10000)

    with col2:
        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Cng", "Lpg", "Electric"])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
        owner_type = st.selectbox("Owner Type", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"])
        seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
        engine = st.number_input("Engine (CC)", min_value=500, max_value=5000, value=1197, step=50)
        seats = st.number_input("Seats", min_value=2, max_value=10, value=5)

    submitted = st.form_submit_button("Predict DealScore")

if submitted:
    car_details = {
        "brand": brand,
        "model": model,
        "year": year,
        "fuel_type": fuel_type,
        "transmission": transmission,
        "owner_type": owner_type,
        "km_driven": km_driven,
        "seller_type": seller_type,
        "mileage": mileage,
        "engine": engine,
        "seats": seats,
        "listed_price": listed_price,
    }

    try:
        result = predict_deal(car_details)
        st.divider()
        st.metric("Predicted Fair Price", f"INR {result['predicted_fair_price']:,.0f}")
        st.metric("Listed Price", f"INR {result['listed_price']:,.0f}")
        st.metric("Estimated Savings", f"INR {result['estimated_savings']:,.0f}")
        st.metric("DealScore", f"{result['deal_score']} / 10")
        st.success(f"Deal Category: {result['deal_category']}")
    except Exception as error:
        st.error(str(error))
