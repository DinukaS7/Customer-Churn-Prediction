import streamlit as st
import joblib
import numpy as np
import time

st.set_page_config(
    page_title="Customer Retention AI",
    page_icon="churn.ico",
    layout="wide"
)

@st.cache_resource
def load_data():
    try:
        s = joblib.load("scaler.pkl")
        m = joblib.load("model.pkl")
        return s, m
    except FileNotFoundError:
        return None, None

scaler, model = load_data()

with st.sidebar:
    st.header("📝 Customer Profile")
    st.write("Adjust the values below:")
    
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    age = st.slider("Age (Years)", 10, 100, 30)
    tenure = st.slider("Tenure (Months)", 0, 130, 10)
    monthlycharge = st.number_input("Monthly Charge ($)", min_value=30.0, max_value=150.0, value=50.0)
    
    st.markdown("---")
    
    predict_btn = st.button("Run Analysis", type="primary")

st.title("🛡️ Churn Prediction Dashboard")
st.markdown("### Current Configuration")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Selected Gender", gender)
col2.metric("Customer Age", f"{age} yrs")
col3.metric("Tenure Length", f"{tenure} months")
col4.metric("Monthly Bill", f"${monthlycharge}")

st.divider()

if predict_btn:
    if scaler and model:
        with st.spinner("Processing customer data..."):
            time.sleep(0.5) 
            
            gender_selected = 1 if gender == "Female" else 0
            X = [age, gender_selected, tenure, monthlycharge]
            X_array = scaler.transform([np.array(X)])
            
            prediction = model.predict(X_array)[0]
            
            st.subheader("Analysis Results")
            
            if prediction == 1:
                st.error("⚠️ Prediction: **Churn Detected**")
                st.markdown("""
                    <div style="background-color: #ffe6e6; padding: 10px; border-radius: 5px; color: #cc0000;">
                        <strong>Insight:</strong> This customer is at high risk of leaving. <br>
                        <strong>Action:</strong> Consider offering a loyalty discount or scheduling a support call.
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.success("✅ Prediction: **Retained**")
                st.markdown("""
                    <div style="background-color: #e6ffe6; padding: 10px; border-radius: 5px; color: #006600;">
                        <strong>Insight:</strong> This customer is likely to stay. <br>
                        <strong>Action:</strong> No immediate action required. Maintain standard service.
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.error("🚨 Error: Could not load 'scaler.pkl' or 'model.pkl'. Check your file path.")
else:
    st.info("👈 Please adjust the customer details in the sidebar and click 'Run Analysis'.")