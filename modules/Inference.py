# modules/Inference.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def show():
    st.header("🔍 Prediksi Churn Pelanggan")

    # Load model dan preprocessing tools
    model = load_pickle("final_churn_model.pkl")
    scaler = load_pickle("scaler_churn.pkl")
    encoder = load_pickle("encoder_churn.pkl")
    feature_columns = load_pickle("feature_columns.pkl")
    categorical_columns = load_pickle("categorical_columns.pkl")  # kolom kategori dari training

    st.write("Lengkapi formulir di bawah ini untuk memprediksi apakah seorang pelanggan berpotensi untuk churn atau tidak.")

    # ===== Form Input =====
    with st.form("input_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            Gender = st.selectbox("Gender", ["Female", "Male"])
            Age = st.number_input("Age", min_value=18, max_value=100, value=30)
            Under30 = st.selectbox("Under 30", ["No", "Yes"])
            SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            Married = st.selectbox("Married", ["No", "Yes"])
            Dependents = st.selectbox("Dependents", ["No", "Yes"])
            ReferredFriend = st.selectbox("Referred a Friend", ["No", "Yes"])
            NumReferrals = st.number_input("Number of Referrals", min_value=0, value=0)
            TenureMonths = st.number_input("Tenure in Months", min_value=0, value=12)
            Offer = st.selectbox("Offer", ["None", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E", "Unknown"])
            PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
            AvgMonthlyLongDist = st.number_input("Avg Monthly Long Distance Charges", min_value=0.0, value=10.0)
                    
        with col2:
            MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes"])
            InternetService = st.selectbox("Internet Service", ["No", "Yes"])
            InternetType = st.selectbox("Internet Type", ["Cable", "DSL", "Fiber Optic", "Unknown"])
            AvgMonthlyGB = st.number_input("Avg Monthly GB Download", min_value=0.0, value=10.0)
            OnlineSecurity = st.selectbox("Online Security", ["No", "Yes"])
            OnlineBackup = st.selectbox("Online Backup", ["No", "Yes"])
            DeviceProtection = st.selectbox("Device Protection Plan", ["No", "Yes"])
            PremiumTechSupport = st.selectbox("Premium Tech Support", ["No", "Yes"])
            StreamingTV = st.selectbox("Streaming TV", ["No", "Yes"])
            StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes"])
            StreamingMusic = st.selectbox("Streaming Music", ["No", "Yes"])
            UnlimitedData = st.selectbox("Unlimited Data", ["No", "Yes"])
            
        with col3:
            Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])
            PaymentMethod = st.selectbox("Payment Method", ["Bank Withdrawal", "Credit card", "Mailed Check"])
            MonthlyCharge = st.number_input("Monthly Charge", min_value=0.0, value=50.0)
            TotalCharges = st.number_input("Total Charges", min_value=0.0, value=100.0)
            TotalLongDistCharges = st.number_input("Total Long Distance Charges", min_value=0.0, value=20.0)
            TotalRevenue = st.number_input("Total Revenue", min_value=0.0, value=500.0)
            TotalRefunds = st.number_input("Total Refunds", min_value=0.0, value=0.0)
            TotalExtraDataCharges = st.number_input("Total Extra Data Charges", min_value=0.0, value=0.0)
            NumDependents = st.number_input("Number of Dependents", min_value=0, value=0)
            SatisfactionScore = st.number_input("Satisfaction Score", min_value=1, max_value=5, value=3)
            CLTV = st.number_input("CLTV", min_value=0.0, value=1000.0)

        submitted = st.form_submit_button("Prediksi")

    if submitted:
        # ===== DataFrame awal =====
        df_input = pd.DataFrame({
            'Gender': [Gender],
            'Age': [Age],
            'Under 30': [Under30],
            'Senior Citizen': [SeniorCitizen],
            'Married': [Married],
            'Dependents': [Dependents],
            'Referred a Friend': [ReferredFriend],
            'Number of Referrals': [NumReferrals],
            'Tenure in Months': [TenureMonths],
            'Offer': [Offer],
            'Phone Service': [PhoneService],
            'Avg Monthly Long Distance Charges': [AvgMonthlyLongDist],
            'Multiple Lines': [MultipleLines],
            'Internet Service': [InternetService],
            'Internet Type': [InternetType],
            'Avg Monthly GB Download': [AvgMonthlyGB],
            'Online Security': [OnlineSecurity],
            'Online Backup': [OnlineBackup],
            'Device Protection Plan': [DeviceProtection],
            'Premium Tech Support': [PremiumTechSupport],
            'Streaming TV': [StreamingTV],
            'Streaming Movies': [StreamingMovies],
            'Streaming Music': [StreamingMusic],
            'Unlimited Data': [UnlimitedData],
            'Contract': [Contract],
            'Paperless Billing': [PaperlessBilling],
            'Payment Method': [PaymentMethod],
            'Monthly Charge': [MonthlyCharge],
            'Total Charges': [TotalCharges],
            'Total Long Distance Charges': [TotalLongDistCharges],
            'Total Revenue': [TotalRevenue],
            'Total Refunds': [TotalRefunds],
            'Total Extra Data Charges': [TotalExtraDataCharges],
            'Number of Dependents': [NumDependents],
            'Satisfaction Score': [SatisfactionScore],
            'CLTV': [CLTV]
        })

        # ===== Preprocessing sama persis dengan training =====
        df_input['Offer'] = df_input['Offer'].fillna('Unknown')
        df_input['Internet Type'] = df_input['Internet Type'].fillna('Unknown')

        # Feature engineering
        df_input['Was_Refunded'] = (df_input['Total Refunds'] > 0).astype(int)
        df_input['Had_Extra_Data_Charge'] = (df_input['Total Extra Data Charges'] > 0).astype(int)
        df_input['Has_Dependents'] = (df_input['Number of Dependents'] > 0).astype(int)
        df_input.drop(columns=['Total Refunds', 'Total Extra Data Charges', 'Number of Dependents'], inplace=True)

        # Log transform
        for col in ['Avg Monthly GB Download', 'Total Long Distance Charges', 'Total Revenue', 'Number of Referrals']:
            df_input[col] = np.log1p(df_input[col])

        # Scaling numeric continuous
        numeric_features = df_input.select_dtypes(include='number').columns.tolist()
        exclude_cols = ['Has_Dependents', 'Was_Refunded', 'Had_Extra_Data_Charge', 'Satisfaction Score']
        fitur_standarisasi = [col for col in numeric_features if col not in exclude_cols]
        df_input[fitur_standarisasi] = scaler.transform(df_input[fitur_standarisasi])

        # Pastikan semua kolom kategori dari training ada
        for col in categorical_columns:
            if col not in df_input.columns:
                df_input[col] = "Unknown"

        # Pastikan semua nilai kategori valid
        for idx, col in enumerate(categorical_columns):
            allowed_cats = list(encoder.categories_[idx])
            df_input[col] = df_input[col].apply(lambda x: x if x in allowed_cats else allowed_cats[0])

        # Encoding kategorikal sesuai urutan training
        df_input[categorical_columns] = encoder.transform(df_input[categorical_columns]).astype(int)

        # Pastikan urutan kolom sesuai model
        df_input = df_input.reindex(columns=feature_columns, fill_value=0)

        # ===== Prediksi =====
        pred = model.predict(df_input)[0]
        st.subheader("Hasil Prediksi")
        
        if pred == 1:
            st.error(f"❌ **Pelanggan tersebut berpotensi untuk churn**\n"
                     "**Tindakan yang direkomendasikan:**\n"
                     "- Lakukan pendekatan untuk memahami kebutuhan serta ketidakpuasan pelanggan.\n"
                     "- Pertimbangkan untuk menawarkan benefit seperti diskon eksklusif maupun upgrade layanan.\n"
                     "- Tinjau kembali riwayat langganan pelanggan untuk mengidentifikasi gangguan atau masalah pada layanan.")
        else:
            st.info(f"✅ **Pelanggan tersebut diprediksi akan tetap loyal (non-churn).**\n"
                    "**Insight:**\n"
                    "- Pelanggan tersebut tidak menunjukkan kecenderungan untuk churn.\n"
                    "- Pertahankan loyalitas pelanggan tersebut dengan memberi reward maupun penawaran yang menarik.\n"
                    "- Terus berikan pengalaman layanan yang konsisten dan memuaskan pada pelanggan yang loyal.")

