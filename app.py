import streamlit as st
import joblib
import pandas as pd

model = joblib.load("churn_model.joblib")
defaults = joblib.load("feature_defaults.joblib")
THRESH = 0.4   # same threshold T you used in the notebook

st.title("Customer Churn Predictor")

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly = st.slider("Monthly charges", 18.0, 120.0, 70.0)
two_year = st.checkbox("Two-year contract")
fiber = st.checkbox("Fiber optic internet")

row = pd.DataFrame([defaults])
row["Tenure Months"] = tenure
row["Monthly Charges"] = monthly
row["Total Charges"] = tenure * monthly
row["Contract_Two year"] = float(two_year)
row["Internet Service_Fiber optic"] = float(fiber)

prob = model.predict_proba(row)[0, 1]
st.metric("Churn probability", f"{prob:.0%}")
st.write("At risk" if prob >= THRESH else "Likely to stay")
