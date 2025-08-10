import streamlit as st
import pandas as pd
import numpy as np
import pickle

def show():
    st.title("🔍 Prediksi Churn Pelanggan")
    st.markdown("Lengkapi formulir di bawah ini untuk memprediksi apakah seorang pelanggan berpotensi untuk churn atau tidak.")
    st.markdown("---")

    # Load model & preprocessing tools
    with open("final_churn_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler_churn.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("encoder_churn.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)

    # Input form sesuai kolom akhir
    col1, col2 = st.columns(2)
    with col1:
        Gender = st.selectbox("Gender", ["Female", "Male"])
        Age = st.number_input("Age", min_value=0)
        Under30 = st.selectbox("Under 30", ["No", "Yes"])
        SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        Married = st.selectbox("Married", ["No", "Yes"])
        Dependents = st.selectbox("Dependents", ["No", "Yes"])
        ReferredAFriend = st.selectbox("Referred a Friend", ["No", "Yes"])
        NumberOfReferrals = st.number_input("Number of Referrals", min_value=0)
        TenureInMonths = st.number_input("Tenure in Months", min_value=0)
        Offer = st.selectbox("Offer", ["Unknown", "Offer A", "Offer B", "Offer C", "Offer D", "None"])
        PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
    with col2:
        AvgMonthlyLongDistanceCharges = st.number_input("Avg Monthly Long Distance Charges", min_value=0.0)
        MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes"])
        InternetService = st.selectbox("Internet Service", ["No", "Yes"])
        InternetType = st.selectbox("Internet Type", ["Unknown", "DSL", "Fiber Optic", "Cable"])
        AvgMonthlyGBDownload = st.number_input("Avg Monthly GB Download", min_value=0.0)
        OnlineSecurity = st.selectbox("Online Security", ["No", "Yes"])
        OnlineBackup = st.selectbox("Online Backup", ["No", "Yes"])
        DeviceProtectionPlan = st.selectbox("Device Protection Plan", ["No", "Yes"])
        PremiumTechSupport = st.selectbox("Premium Tech Support", ["No", "Yes"])
        StreamingTV = st.selectbox("Streaming TV", ["No", "Yes"])
        StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes"])
        StreamingMusic = st.selectbox("Streaming Music", ["No", "Yes"])
        UnlimitedData = st.selectbox("Unlimited Data", ["No", "Yes"])
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])
        PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        MonthlyCharge = st.number_input("Monthly Charge", min_value=0.0)
        TotalCharges = st.number_input("Total Charges", min_value=0.0)
        TotalLongDistanceCharges = st.number_input("Total Long Distance Charges", min_value=0.0)
        TotalRevenue = st.number_input("Total Revenue", min_value=0.0)
        SatisfactionScore = st.number_input("Satisfaction Score", min_value=0)
        CLTV = st.number_input("CLTV", min_value=0.0)
        Was_Refunded = st.selectbox("Was Refunded", ["No", "Yes"])
        Had_Extra_Data_Charge = st.selectbox("Had Extra Data Charge", ["No", "Yes"])
        Has_Dependents = st.selectbox("Has Dependents", ["No", "Yes"])

    # Bangun DataFrame dari input
    input_data = pd.DataFrame([[Gender, Age, Under30, SeniorCitizen, Married, Dependents, 
                                 ReferredAFriend, NumberOfReferrals, TenureInMonths, Offer, PhoneService,
                                 AvgMonthlyLongDistanceCharges, MultipleLines, InternetService, InternetType, AvgMonthlyGBDownload,
                                 OnlineSecurity, OnlineBackup, DeviceProtectionPlan, PremiumTechSupport, StreamingTV, StreamingMovies,
                                 StreamingMusic, UnlimitedData, Contract, PaperlessBilling, PaymentMethod, MonthlyCharge,
                                 TotalCharges, TotalLongDistanceCharges, TotalRevenue, SatisfactionScore, CLTV,
                                 Was_Refunded, Had_Extra_Data_Charge, Has_Dependents]],
                               columns=feature_columns)

    # Preprocessing sama seperti training
    # Transformasi log
    for col in ['AvgMonthlyGBDownload', 'TotalLongDistanceCharges', 'TotalRevenue', 'NumberOfReferrals']:
        if col in input_data.columns:
            input_data[col] = np.log1p(input_data[col])

    # Standarisasi fitur kontinu
    continuous_cols = scaler.feature_names_in_
    input_data[continuous_cols] = scaler.transform(input_data[continuous_cols])

    # Encoding kategori
    categorical_cols = encoder.feature_names_in_
    input_data[categorical_cols] = encoder.transform(input_data[categorical_cols]).astype(int)

    # Prediksi
    if st.button("Predict Churn"):
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
        if prediction == 1:
            st.error(f"❌ Pelanggan berpotensi churn (Prob: {prob:.2f})")
        else:
            st.success(f"✅ Pelanggan diprediksi loyal (Prob: {prob:.2f})")
