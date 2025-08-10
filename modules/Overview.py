import streamlit as st

def show():
    st.title("📋 Overview")
    st.markdown("""
    **Churn** adalah kondisi ketika pelanggan berhenti menggunakan layanan.  
    Mengetahui kemungkinan churn sangat penting untuk strategi retensi pelanggan.
    
    Model ini dilatih dengan data pelanggan yang telah melalui:
    - **Preprocessing**: Penanganan missing value, transformasi log, standarisasi, encoding ordinal.
    - **Resampling**: SMOTEENN untuk mengatasi imbalance.
    - **Modeling**: XGBoost dengan tuning hyperparameter.
    """)
