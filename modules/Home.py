import streamlit as st

def show():
    st.title("📊 Customer Churn Prediction App")
    st.image("assets/churn_image.png", use_column_width=True)
    st.markdown("""
    Selamat datang di aplikasi prediksi churn pelanggan!  
    Aplikasi ini menggunakan model machine learning **XGBoost** 
    untuk memprediksi kemungkinan pelanggan akan berhenti menggunakan layanan.
    """)
