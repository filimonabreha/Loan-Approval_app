
# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import pandas as pd
import sklearn

# Load the trained model
# --- Put the Model in Drive First---
with open("my_model.pkl", "rb") as file:
    model = pickle.load(file)

  # App Title
st.markdown(
    """
    <h1 style='text-align:center; color:white; background-color:#2a4d69; padding:12px; border-radius:8px;'>
        🔍 Loan Approval Prediction App
    </h1>
    <p style='text-align:center; font-size:18px;'>
        Enter applicant information below to estimate approval probability
    </p>
    """,
    unsafe_allow_html=True)
st.header("📌 Applicant Information")
  # Numeric Inputs
fico = st.slider("FICO Score", min_value=300, max_value=850, value=700)
req_amt = st.number_input("Requested Loan Amount", min_value=0, value=10000)
grant_amt = st.number_input("Granted Loan Amount", min_value=0, value=10000)
income = st.number_input("Monthly Gross Income", min_value=0, value=5000)
housing = st.number_input("Monthly Housing Payment", min_value=0, value=1500)

# Categorical Inputs

reason = st.selectbox("Reason for Loan",
                       ["Debt Consolidation", "Home Improvement", "Car Purchase", "Other"])

fico_group = st.selectbox("FICO Score Group",
                           ["Poor", "Fair", "Good", "Very Good", "Excellent"])

emp_status = st.selectbox("Employment Status",
                           ["Employed", "Self-employed", "Unemployed", "Student", "Retired"])

emp_sector = st.selectbox("Employment Sector",
                           ["Private", "Public", "Self-employed", "Student", "Retired", "Unknown"])

bk_flag = st.selectbox("Ever Bankrupt or Foreclosed?", ["No", "Yes"])

lender_choice = st.selectbox("Choose Lender (A/B/C)", ["A", "B", "C"])

# Create Input Row
input_df = pd.DataFrame({
    "Reason": [reason],
    "Requested_Loan_Amount": [req_amt],
    "Granted_Loan_Amount": [grant_amt],
    "FICO_score": [fico],
    "FICO Score Group": [fico_group],
    "Employment_Status": [emp_status],
    "Employment_Sector": [emp_sector],
    "Monthly_Gross_Income": [income],
    "Monthly_Housing_Payment": [housing],
    "Ever_Bankrupt_or_Foreclose": [1 if bk_flag == "Yes" else 0],
    "Lender": [lender_choice]})

# One-hot encode user input
encoded = pd.get_dummies(input_df)

# Add missing columns that model expects
model_cols = model.feature_names_in_
for col in model_cols:
    if col not in encoded.columns:
        encoded[col] = 0

# Reorder to match training data
encoded = encoded[model_cols]

# Predict Button
if st.button("🚀 Predict Approval"):
    prob = model.predict_proba(encoded)[0][1]
    result = model.predict(encoded)[0]

    st.subheader("📊 Prediction Result")

    if result == 1:
        st.success(f"**Approved** ✔ (Probability: {prob:.2%})")
    else:
        st.error(f"**Denied** ❌ (Probability: {prob:.2%})")

    # Add a small explanation box
    st.info(
        "This prediction is based on your trained logistic regression / decision tree model "
        "from the BUS 458 Loan Analysis project.")
