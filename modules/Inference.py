# Inference.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# === Load Model & Preprocessing Tools ===
with open("final_churn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler_churn.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("encoder_churn.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

st.title("Customer Churn Prediction")
st.write("Masukkan data customer untuk memprediksi kemungkinan churn.")

# === Input Form ===
with st.form("churn_form"):
    # Numerical inputs
    Age = st.number_input("Age", min_value=0, max_value=120)
    Number_of_Referrals = st.number_input("Number of Referrals", min_value=0)
    Tenure_in_Months = st.number_input("Tenure in Months", min_value=0)
    Avg_Monthly_Long_Distance_Charges = st.number_input("Avg Monthly Long Distance Charges", min_value=0.0)
    Avg_Monthly_GB_Download = st.number_input("Avg Monthly GB Download", min_value=0.0)
    Monthly_Charge = st.number_input("Monthly Charge", min_value=0.0)
    Total_Charges = st.number_input("Total Charges", min_value=0.0)
    Total_Long_Distance_Charges = st.number_input("Total Long Distance Charges", min_value=0.0)
    Total_Revenue = st.number_input("Total Revenue", min_value=0.0)
    Satisfaction_Score = st.number_input("Satisfaction Score", min_value=1, max_value=5)
    CLTV = st.number_input("CLTV", min_value=0.0)

    # Binary categorical
    Gender = st.selectbox("Gender", ["Female", "Male"])
    Under_30 = st.selectbox("Under 30", ["No", "Yes"])
    Senior_Citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    Married = st.selectbox("Married", ["No", "Yes"])
    Dependents = st.selectbox("Dependents", ["No", "Yes"])
    Referred_a_Friend = st.selectbox("Referred a Friend", ["No", "Yes"])
    Phone_Service = st.selectbox("Phone Service", ["No", "Yes"])
    Multiple_Lines = st.selectbox("Multiple Lines", ["No", "Yes"])
    Internet_Service = st.selectbox("Internet Service", ["No", "Yes"])
    Online_Security = st.selectbox("Online Security", ["No", "Yes"])
    Online_Backup = st.selectbox("Online Backup", ["No", "Yes"])
    Device_Protection_Plan = st.selectbox("Device Protection Plan", ["No", "Yes"])
    Premium_Tech_Support = st.selectbox("Premium Tech Support", ["No", "Yes"])
    Streaming_TV = st.selectbox("Streaming TV", ["No", "Yes"])
    Streaming_Movies = st.selectbox("Streaming Movies", ["No", "Yes"])
    Streaming_Music = st.selectbox("Streaming Music", ["No", "Yes"])
    Unlimited_Data = st.selectbox("Unlimited Data", ["No", "Yes"])
    Paperless_Billing = st.selectbox("Paperless Billing", ["No", "Yes"])

    # Multi-category
    Offer = st.selectbox("Offer", ["None", "Offer A", "Offer B", "Offer C", "Offer D", "Unknown"])
    Internet_Type = st.selectbox("Internet Type", ["Cable", "DSL", "Fiber Optic", "Unknown"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    Payment_Method = st.selectbox("Payment Method", ["Bank Withdrawal", "Credit card", "Mailed Check"])

    # Additional numeric for feature engineering
    Total_Refunds = st.number_input("Total Refunds", min_value=0.0)
    Total_Extra_Data_Charges = st.number_input("Total Extra Data Charges", min_value=0.0)
    Number_of_Dependents = st.number_input("Number of Dependents", min_value=0)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Buat DataFrame dari input
    input_data = pd.DataFrame([{
        "Gender": Gender,
        "Age": Age,
        "Under 30": Under_30,
        "Senior Citizen": Senior_Citizen,
        "Married": Married,
        "Dependents": Dependents,
        "Referred a Friend": Referred_a_Friend,
        "Number of Referrals": Number_of_Referrals,
        "Tenure in Months": Tenure_in_Months,
        "Offer": Offer,
        "Phone Service": Phone_Service,
        "Avg Monthly Long Distance Charges": Avg_Monthly_Long_Distance_Charges,
        "Multiple Lines": Multiple_Lines,
        "Internet Service": Internet_Service,
        "Internet Type": Internet_Type,
        "Avg Monthly GB Download": Avg_Monthly_GB_Download,
        "Online Security": Online_Security,
        "Online Backup": Online_Backup,
        "Device Protection Plan": Device_Protection_Plan,
        "Premium Tech Support": Premium_Tech_Support,
        "Streaming TV": Streaming_TV,
        "Streaming Movies": Streaming_Movies,
        "Streaming Music": Streaming_Music,
        "Unlimited Data": Unlimited_Data,
        "Contract": Contract,
        "Paperless Billing": Paperless_Billing,
        "Payment Method": Payment_Method,
        "Monthly Charge": Monthly_Charge,
        "Total Charges": Total_Charges,
        "Total Long Distance Charges": Total_Long_Distance_Charges,
        "Total Revenue": Total_Revenue,
        "Satisfaction Score": Satisfaction_Score,
        "CLTV": CLTV,
        "Total Refunds": Total_Refunds,
        "Total Extra Data Charges": Total_Extra_Data_Charges,
        "Number of Dependents": Number_of_Dependents
    }])

    # === REPLICATE PREPROCESSING ===
    # Handle missing values
    input_data["Offer"] = input_data["Offer"].fillna("Unknown")
    input_data["Internet Type"] = input_data["Internet Type"].fillna("Unknown")

    # Feature engineering
    input_data["Was_Refunded"] = (input_data["Total Refunds"] > 0).astype(int)
    input_data["Had_Extra_Data_Charge"] = (input_data["Total Extra Data Charges"] > 0).astype(int)
    input_data["Has_Dependents"] = (input_data["Number of Dependents"] > 0).astype(int)
    input_data.drop(columns=["Total Refunds", "Total Extra Data Charges", "Number of Dependents"], inplace=True)

    # Log transform
    for col in ["Avg Monthly GB Download", "Total Long Distance Charges", "Total Revenue", "Number of Referrals"]:
        input_data[col] = np.log1p(input_data[col])

    # Scaling numeric continuous
    numeric_features = input_data.select_dtypes(include="number").columns.tolist()
    exclude_cols = ["Has_Dependents", "Was_Refunded", "Had_Extra_Data_Charge", "Satisfaction Score"]
    fitur_standarisasi = [col for col in numeric_features if col not in exclude_cols]
    input_data[fitur_standarisasi] = scaler.transform(input_data[fitur_standarisasi])

    # Encoding categorical
    categorical_cols = input_data.select_dtypes(include="object").columns
    input_data[categorical_cols] = encoder.transform(input_data[categorical_cols]).astype(int)

    # Reorder columns sesuai training
    input_data = input_data[feature_columns]

    # Predict
    prediction = model.predict(input_data)[0]
    pred_proba = model.predict_proba(input_data)[0][1]

    st.subheader("Hasil Prediksi")
    st.write(f"Prediksi Churn: **{'Ya' if prediction == 1 else 'Tidak'}**")
    st.write(f"Probabilitas Churn: **{pred_proba:.2%}**")
