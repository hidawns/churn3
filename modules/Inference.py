import streamlit as st
import pandas as pd
import numpy as np
import pickle
import math

# ============================
# Load Model dan Preprocessing
# ============================
@st.cache_resource
def load_model_and_tools():
    with open("final_churn_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler_churn.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("encoder_churn.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    return model, scaler, encoder, feature_columns

model, scaler, encoder, feature_columns = load_model_and_tools()

# ============================
# Halaman Streamlit
# ============================
def show():
    st.title("🔮 Churn Prediction Inference")
    st.markdown("Masukkan data customer untuk memprediksi apakah mereka akan **Churn** atau **Tidak Churn**.")

    # Form input
    with st.form("prediction_form"):
        st.subheader("📋 Input Data Customer")

        # Contoh input numerik
        tenure_months = st.number_input("Tenure (bulan)", min_value=0, value=12)
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
        total_revenue = st.number_input("Total Revenue", min_value=0.0, value=600.0)
        avg_monthly_gb = st.number_input("Avg Monthly GB Download", min_value=0.0, value=20.0)
        total_long_distance = st.number_input("Total Long Distance Charges", min_value=0.0, value=10.0)
        number_of_referrals = st.number_input("Number of Referrals", min_value=0, value=0)

        # Contoh input kategorikal
        offer = st.selectbox("Offer", ["None", "Offer A", "Offer B", "Offer C", "Offer D", "Unknown"])
        internet_type = st.selectbox("Internet Type", ["Fiber Optic", "DSL", "Cable", "Unknown"])
        contract = st.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"])
        gender = st.selectbox("Gender", ["Male", "Female"])

        # Boolean flags (hasil feature engineering di training)
        was_refunded = st.selectbox("Pernah Refund?", ["Tidak", "Ya"])
        had_extra_data_charge = st.selectbox("Pernah Extra Data Charge?", ["Tidak", "Ya"])
        has_dependents = st.selectbox("Punya Dependents?", ["Tidak", "Ya"])

        submitted = st.form_submit_button("Prediksi Churn")

    if submitted:
        # ============================
        # Proses input jadi DataFrame
        # ============================
        input_data = pd.DataFrame([{
            "Tenure in Months": tenure_months,
            "Monthly Charges": monthly_charges,
            "Total Revenue": total_revenue,
            "Avg Monthly GB Download": avg_monthly_gb,
            "Total Long Distance Charges": total_long_distance,
            "Number of Referrals": number_of_referrals,
            "Offer": offer,
            "Internet Type": internet_type,
            "Contract": contract,
            "Gender": gender,
            "Was_Refunded": 1 if was_refunded == "Ya" else 0,
            "Had_Extra_Data_Charge": 1 if had_extra_data_charge == "Ya" else 0,
            "Has_Dependents": 1 if has_dependents == "Ya" else 0
        }])

        # ============================
        # Log Transform sesuai training
        # ============================
        for col in ["Avg Monthly GB Download", "Total Long Distance Charges", "Total Revenue", "Number of Referrals"]:
            input_data[col] = np.log1p(input_data[col])

        # ============================
        # Scaling kolom numerik kontinu
        # ============================
        numeric_features = input_data.select_dtypes(include='number').columns.tolist()
        exclude_cols = ["Has_Dependents", "Was_Refunded", "Had_Extra_Data_Charge"]
        fitur_standarisasi = [col for col in numeric_features if col not in exclude_cols]
        input_data[fitur_standarisasi] = scaler.transform(input_data[fitur_standarisasi])

        # ============================
        # Encoding kolom kategorikal
        # ============================
        categorical_cols = input_data.select_dtypes(include='object').columns
        if len(categorical_cols) > 0:
            input_data[categorical_cols] = encoder.transform(input_data[categorical_cols]).astype(int)

        # ============================
        # Pastikan urutan kolom sesuai training
        # ============================
        for col in feature_columns:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[feature_columns]

        # ============================
        # Prediksi
        # ============================
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0][prediction]

        if prediction == 1:
            st.error(f"🚨 Customer kemungkinan **CHURN** ({prediction_proba*100:.2f}% confidence)")
        else:
            st.success(f"✅ Customer kemungkinan **TIDAK CHURN** ({prediction_proba*100:.2f}% confidence)")

