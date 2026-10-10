import streamlit as st
import pandas as pd
import joblib




# Set page configuration
st.set_page_config(page_title="Loan Default Predictor", layout="wide")

# Loading trained pipeline
@st.cache_resource
def load_model():
    return joblib.load('loan_default_pipeline.pkl')

pipeline = load_model()


st.title("Credit Risk & Loan Default Predictor")
st.write("Enter the borrower's financial profile below to evaluate default risk.")

# Create a clean layout with two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Loan Details")
    loan_amount = st.number_input("Current Loan Amount (K)", min_value=0.0, value=150000.0, step=1000.0)
    term = st.selectbox("Loan Term", ["Short Term", "Long Term"])
    purpose = st.selectbox("Home Ownership", ["Home Mortgage", "Rent", "Own Home"])
    
    st.subheader("Borrower Financials")
    annual_income = st.number_input("Annual Income (K)", min_value=0.0, value=65000.0, step=1000.0)
    monthly_debt = st.number_input("Monthly Debt (K)", min_value=0.0, value=1200.0, step=100.0)
    years_in_job = st.number_input("Years in Current Job", min_value=0.0, max_value=10.0, value=5.0, step=1.0)
    
with col2:
    st.subheader("Credit History")
    credit_score = st.number_input("Credit Score", min_value=300.0, max_value=850.0, value=720.0, step=1.0)
    years_credit_history = st.number_input("Years of Credit History", min_value=0.0, value=15.0, step=1.0)
    credit_balance = st.number_input("Current Credit Balance (K)", min_value=0.0, value=12000.0, step=500.0)
    max_open_credit = st.number_input("Maximum Open Credit (K)", min_value=0.0, value=25000.0, step=500.0)
    open_accounts = st.number_input("Number of Open Accounts", min_value=0.0, value=8.0, step=1.0)
    
    st.subheader("Risk Indicators")
    credit_problems = st.number_input("Number of Credit Problems", min_value=0.0, value=0.0, step=1.0)
    bankruptcies = st.number_input("Bankruptcies", min_value=0.0, value=0.0, step=1.0)
    tax_liens = st.number_input("Tax Liens", min_value=0.0, value=0.0, step=1.0)
    nvr_delinquent = st.selectbox("Has the borrower previously been delinquent?", ["No (Never)", "Yes"])

# Predict Button
st.markdown("---")
if st.button("Predict Default Risk", type="primary"):
    
    # Map binary categorical input back to the integer format the model expects
    nvr_delinquent_val = 0 if nvr_delinquent == "No (Never)" else 1
    
    # Construct a DataFrame matching your model's exact feature names
    input_data = pd.DataFrame({
        'Current Loan Amount': [loan_amount],
        'Term': [term],
        'Credit Score': [credit_score],
        'Annual Income': [annual_income],
        'Years in current job': [years_in_job],
        'Home Ownership': [purpose],
        'Monthly Debt': [monthly_debt],
        'Years of Credit History': [years_credit_history],
        'Number of Open Accounts': [open_accounts],
        'Number of Credit Problems': [credit_problems],
        'Current Credit Balance': [credit_balance],
        'Maximum Open Credit': [max_open_credit],
        'Bankruptcies': [bankruptcies],
        'Tax Liens': [tax_liens],
        'nvr_delinquent': [nvr_delinquent_val]
    })
    
    # Execute the prediction through the pipeline
    prediction = pipeline.predict(input_data)[0]
    probabilities = pipeline.predict_proba(input_data)[0]
    
    # Display Results
    if prediction == 1:
        st.error(f"⚠️ **HIGH RISK:** The model predicts a Loan Default (Charged Off).")
        st.write(f"Confidence: {probabilities[1]:.1%} probability of default.")
    else:
        st.success(f" **LOW RISK:** The model predicts the loan will be Fully Paid.")
        st.write(f"Confidence: {probabilities[0]:.1%} probability of successful repayment.")