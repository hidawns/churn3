import streamlit as st
import pandas as pd
import numpy as np
import pickle

def show():
    st.title("🔮 Churn Prediction Inference")

    # Load model & preprocessing tools
    with open("final_churn_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler_churn.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("encoder_churn.pkl", "rb") as f:
        encoder = pickle.load(f)

    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)

    # === Form Input ===
    st.subheader("Masukkan Data Pelanggan (Mentah)")

    with st.form("churn_form"):
        # Numeric inputs
        age = st.number_input("Age", min_value=0, step=1)
        num_referrals = st.number_input("Number of Referrals", min_value=0.0)
        tenure = st.number_input("Tenure in Months", min_value=0, step=1)
        avg_long_dist = st.number_input("Avg Monthly Long Distance Charges", min_value=0.0)
        avg_gb_download = st.number_input("Avg Monthly GB Download", min_value=0.0)
        monthly_charge = st.number_input("Monthly Charge", min_value=0.0)
        total_charges = st.number_input("Total Charges", min_value=0.0)
        total_long_dist = st.number_input("Total Long Distance Charges", min_value=0.0)
        total_revenue = st.number_input("Total Revenue", min_value=0.0)
        cltv = st.number_input("CLTV", min_value=0.0)
        total_refunds = st.number_input("Total Refunds", min_value=0.0)
        total_extra_data = st.number_input("Total Extra Data Charges", min_value=0.0)
        num_dependents = st.number_input("Number of Dependents", min_value=0, step=1)
        satisfaction_score = st.number_input("Satisfaction Score", min_value=0, max_value=5, step=1)

        # Categorical inputs
        gender = st.selectbox("Gender", ["Female", "Male"])
        under_30 = st.selectbox("Under 30", ["No", "Yes"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        married = st.selectbox("Married", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        referred_friend = st.selectbox("Referred a Friend", ["No", "Yes"])
        offer = st.selectbox("Offer", ["None", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E", "Unknown"])
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
        internet_service = st.selectbox("Internet Service", ["No", "Yes"])
        internet_type = st.selectbox("Internet Type", ["Cable", "DSL", "Fiber Optic", "Unknown"])
        online_security = st.selectbox("Online Security", ["No", "Yes"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes"])
        device_protection = st.selectbox("Device Protection Plan", ["No", "Yes"])
        premium_support = st.selectbox("Premium Tech Support", ["No", "Yes"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
        streaming_music = st.selectbox("Streaming Music", ["No", "Yes"])
        unlimited_data = st.selectbox("Unlimited Data", ["No", "Yes"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", ["Bank Withdrawal", "Credit card", "Mailed Check"])

        submit = st.form_submit_button("Predict Churn")

    if submit:
        # Buat DataFrame mentah
        raw_data = pd.DataFrame([{
            "Age": age,
            "Number of Referrals": num_referrals,
            "Tenure in Months": tenure,
            "Avg Monthly Long Distance Charges": avg_long_dist,
            "Avg Monthly GB Download": avg_gb_download,
            "Monthly Charge": monthly_charge,
            "Total Charges": total_charges,
            "Total Long Distance Charges": total_long_dist,
            "Total Revenue": total_revenue,
            "CLTV": cltv,
            "Total Refunds": total_refunds,
            "Total Extra Data Charges": total_extra_data,
            "Number of Dependents": num_dependents,
            "Satisfaction Score": satisfaction_score,
            "Gender": gender,
            "Under 30": under_30,
            "Senior Citizen": senior_citizen,
            "Married": married,
            "Dependents": dependents,
            "Referred a Friend": referred_friend,
            "Offer": offer,
            "Phone Service": phone_service,
            "Multiple Lines": multiple_lines,
            "Internet Service": internet_service,
            "Internet Type": internet_type,
            "Online Security": online_security,
            "Online Backup": online_backup,
            "Device Protection Plan": device_protection,
            "Premium Tech Support": premium_support,
            "Streaming TV": streaming_tv,
            "Streaming Movies": streaming_movies,
            "Streaming Music": streaming_music,
            "Unlimited Data": unlimited_data,
            "Contract": contract,
            "Paperless Billing": paperless_billing,
            "Payment Method": payment_method
        }])

        # === Preprocessing sesuai training ===
        # Handle missing
        raw_data['Offer'] = raw_data['Offer'].fillna('Unknown')
        raw_data['Internet Type'] = raw_data['Internet Type'].fillna('Unknown')

        # Feature engineering
        raw_data['Was_Refunded'] = (raw_data['Total Refunds'] > 0).astype(int)
        raw_data['Had_Extra_Data_Charge'] = (raw_data['Total Extra Data Charges'] > 0).astype(int)
        raw_data['Has_Dependents'] = (raw_data['Number of Dependents'] > 0).astype(int)

        raw_data.drop(columns=['Total Refunds', 'Total Extra Data Charges', 'Number of Dependents'], inplace=True)

        # Log transform
        for col in ['Avg Monthly GB Download', 'Total Long Distance Charges', 'Total Revenue', 'Number of Referrals']:
            raw_data[col] = np.log1p(raw_data[col])

        # Scaling numeric features
        numeric_features = raw_data.select_dtypes(include='number').columns.tolist()
        exclude_cols = ['Has_Dependents', 'Was_Refunded', 'Had_Extra_Data_Charge', 'Satisfaction Score']
        fitur_standarisasi = [col for col in numeric_features if col not in exclude_cols]
        raw_data[fitur_standarisasi] = scaler.transform(raw_data[fitur_standarisasi])

        # Encoding categorical features
        categorical_cols = raw_data.select_dtypes(include='object').columns
        raw_data[categorical_cols] = encoder.transform(raw_data[categorical_cols]).astype(int)

        # Reorder columns sesuai training
        raw_data = raw_data.reindex(columns=feature_columns)

        # Predict
        pred = model.predict(raw_data)[0]
        prob = model.predict_proba(raw_data)[0][1]

        # Output
        if pred == 1:
            st.error(f"🚨 Pelanggan berpotensi CHURN dengan probabilitas {prob:.2%}")
        else:
            st.success(f"✅ Pelanggan diprediksi TIDAK CHURN dengan probabilitas {1-prob:.2%}")
