

import streamlit as st
import requests

st.set_page_config(
    page_title="Ahmedabad House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Ahmedabad House Price Predictor")
st.write("Predict house prices in Ahmedabad using a trained XGBoost model.")


with st.form(key="house_form"):
    total_sqft = st.number_input("Total Square Feet", min_value=100, max_value=10000, value=1200)
    bhk = st.selectbox("BHK", options=[1, 2, 3, 4, 5], index=2)
    location = st.text_input("Location", value="Navrangpura")
    floor_num = st.number_input("Floor Number", min_value=0, max_value=50, value=1)
    price_sqft = st.number_input("Price per Sqft (INR)", min_value=0, value=3765)

    submit_button = st.form_submit_button("Predict Price")



API_URL = "http://127.0.0.1:8000/predict"  # Update if hosted elsewhere

if submit_button:
    input_data = {
        "total_sqft": total_sqft,
        "bhk": bhk,
        "location": location,
        "floor_num": floor_num,
        "price_sqft": price_sqft
    }

    response = requests.post(API_URL, json=input_data)
    if response.status_code == 200:
        result = response.json()
        st.success(f"💰 Predicted Total House Price: {result['predicted_price_lakhs']:.2f} Lakhs")
    else:
        st.error(f"❌ Error: {response.json()['detail']}")
