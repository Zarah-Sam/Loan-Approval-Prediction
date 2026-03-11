import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder

# Page config
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="wide"
)

# Styling
st.markdown("""
<style>
.main-header{
font-size:3rem;
text-align:center;
color:#2c7be5;
}

.success-box{
background:#d4edda;
padding:20px;
border-radius:10px;
}

.fail-box{
background:#f8d7da;
padding:20px;
border-radius:10px;
}
</style>
""",unsafe_allow_html=True)

# Load model
model = joblib.load("model.pkl")
#scaler = joblib.load("scaler.pkl")

# Title
st.markdown('<h1 class="main-header">💰 Loan Approval Prediction System</h1>',unsafe_allow_html=True)

st.markdown("---")

# Sidebar Inputs
st.sidebar.header("Applicant Details")

gender = st.sidebar.selectbox("Gender",["Male","Female"])

married = st.sidebar.selectbox("Married",["Yes","No"])

dependents = st.sidebar.slider("Dependents",0,3,0)

education = st.sidebar.selectbox("Education",["Graduate","Not Graduate"])

self_emp = st.sidebar.selectbox("Self Employed",["Yes","No"])

income = st.sidebar.slider("Applicant Income",1000,2000000,100)

co_income = st.sidebar.slider("Coapplicant Income",0,2000000,0)

loan_amount = st.sidebar.slider("Loan Amount",5000,5000000,100)

loan_term = st.sidebar.selectbox("Loan Term",[360,180,120])

credit = st.sidebar.selectbox("Credit History",[1,0])

property_area = st.sidebar.selectbox("Property Area",["Urban","Semiurban","Rural"])

# Encoding
gender = 1 if gender=="Male" else 0
married = 1 if married=="Yes" else 0
education = 1 if education=="Graduate" else 0
self_emp = 1 if self_emp=="Yes" else 0
property_area = {"Urban":2,"Semiurban":1,"Rural":0}[property_area]

features = np.array([[
gender,married,dependents,education,self_emp,
income,co_income,loan_amount,loan_term,credit,property_area
]])

features_scaled = features

# Prediction Button
if st.button("Predict Loan Approval"):

    prediction = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0]

    st.markdown("---")

    st.subheader("Prediction Result")

    if prediction==1:

        st.markdown(f"""
        <div class="success-box">
        <h2>✅ Loan Approved</h2>
        <p>Approval Probability: {prob[1]*100:.2f}%</p>
        </div>
        """,unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="fail-box">
        <h2>❌ Loan Rejected</h2>
        <p>Rejection Probability: {prob[0]*100:.2f}%</p>
        </div>
        """,unsafe_allow_html=True)

    st.subheader("Approval Probability Chart")

    chart_data = pd.DataFrame({
        "Outcome":["Rejected","Approved"],
        "Probability":[prob[0]*100,prob[1]*100]
    })

    st.bar_chart(chart_data.set_index("Outcome"))

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### About Model")

st.sidebar.write("""
This Loan Prediction System uses:

- Random Forest Classifier
- Pandas for data processing
- Scikit-learn for machine learning
- Streamlit for web interface
""")

# Footer
st.markdown("---")
st.markdown(
"<div style='text-align:center'>Built using Python, Pandas, Scikit-learn and Streamlit</div>",
unsafe_allow_html=True
)