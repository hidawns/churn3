import streamlit as st
import pandas as pd
import numpy as np
import pickle
import math

def show():
    st.title("🔍 Prediksi Churn Pelanggan (Versi Baru)")
    st.markdown("Lengkapi formulir di bawah ini untuk memprediksi apakah seorang pelanggan berpotensi churn. Form dibuat dinamis berdasarkan `merged_dataset.csv` sehingga mengikuti struktur dataset training.")

    # === Load model & artifacts ===
    try:
        with open("final_churn_model.pkl", "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        st.error("Model `final_churn_model.pkl` tidak ditemukan. Pastikan model sudah disimpan di root repository.")
        return

    try:
        with open("scaler_churn.pkl", "rb") as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        st.error("Scaler `scaler_churn.pkl` tidak ditemukan.")
        return

    try:
        with open("encoder_churn.pkl", "rb") as f:
            encoder = pickle.load(f)
    except FileNotFoundError:
        st.error("Encoder `encoder_churn.pkl` tidak ditemukan.")
        return

    try:
        with open("feature_columns.pkl", "rb") as f:
            feature_columns = pickle.load(f)
    except FileNotFoundError:
        st.error("Feature list `feature_columns.pkl` tidak ditemukan.")
        return

    # === Load original dataset to infer fields and types ===
    try:
        raw_df = pd.read_csv("merged_dataset.csv")
    except FileNotFoundError:
        st.error("File `merged_dataset.csv` tidak ditemukan di root repository.")
        return

    # Prepare list of original input columns (mimic notebook's initial drops)
    tmp = raw_df.copy()
    # drop Customer ID if present
    if 'Customer ID' in tmp.columns:
        tmp.drop(columns=['Customer ID'], inplace=True)
    # drop status columns except keep 'Churn Value' if present
    status_cols = ['Customer Status', 'Churn Label', 'Churn Score', 'Churn Category', 'Churn Reason']
    for c in status_cols:
        if c in tmp.columns:
            tmp.drop(columns=c, inplace=True)

    # We'll present form fields for remaining original columns that were input features before engineering/dropping.
    # But we also need to present fields required to compute engineered features:
    # (Total Refunds, Total Extra Data Charges, Number of Dependents) -> used to compute Was_Refunded, Had_Extra_Data_Charge, Has_Dependents
    # If these three columns are NOT present in raw_df (because you pre-cleaned earlier), we will not ask and default engineered values to 0.
    request_engineer_inputs = []
    for col in ['Total Refunds', 'Total Extra Data Charges', 'Number of Dependents']:
        if col in raw_df.columns:
            request_engineer_inputs.append(col)

    # For display, we will use a subset of columns: first separate categorical vs numeric
    cat_cols = tmp.select_dtypes(include=['object']).columns.tolist()
    num_cols = tmp.select_dtypes(include=['int64', 'float64']).columns.tolist()

    st.markdown("Isi data pelanggan di bawah (jika ada field tidak tampil, isi nilai default di form).")

    # build form
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        user_input = {}

        # fill categorical
        with col1:
            st.subheader("Kategori")
            for c in cat_cols:
                # ignore target if present
                if c == 'Churn Value':
                    continue
                # gather unique values (limit to top 30)
                vals = tmp[c].dropna().unique().tolist()
                if len(vals) == 0:
                    # fallback to text input
                    user_input[c] = st.text_input(c, value="")
                else:
                    # convert to string for selectbox options
                    str_vals = [str(v) for v in vals[:100]]
                    user_input[c] = st.selectbox(c, options=str_vals)

            # engineered input sources (if present)
            for c in request_engineer_inputs:
                user_input[c] = st.number_input(c, value=0.0)

        with col2:
            st.subheader("Numerik")
            for n in num_cols:
                if n == 'Churn Value':
                    continue
                # if these numeric columns are ones used for engineered calculations, still present
                default_val = 0.0
                # try get a reasonable min/max from data
                try:
                    mins = float(tmp[n].min())
                    maxs = float(tmp[n].max())
                    default_val = float(tmp[n].median()) if not math.isnan(float(tmp[n].median())) else 0.0
                except Exception:
                    pass
                # use number_input with no strict bounds
                user_input[n] = st.number_input(n, value=default_val)

        submitted = st.form_submit_button("Predict Churn")

    if not submitted:
        st.info("Isi formulir lalu klik **Predict Churn**.")
        return

    # === Build a row that mimics the preprocessing in the notebook ===
    row = {}

    # start from tmp columns
    for c in tmp.columns:
        if c == 'Churn Value':
            continue
        if c in user_input:
            # attempt to convert numeric-like inputs back to original dtype
            row[c] = user_input[c]
        else:
            # if user didn't provide, fallback to first non-null or default
            if tmp[c].dropna().shape[0] > 0:
                row[c] = tmp[c].dropna().iloc[0]
            else:
                # default
                row[c] = 0 if c in num_cols else ""

    # Compute engineered features as in notebook
    # Was_Refunded = (Total Refunds > 0).astype(int)
    if 'Was_Refunded' in feature_columns:
        if 'Total Refunds' in user_input:
            row['Was_Refunded'] = 1 if float(user_input.get('Total Refunds', 0)) > 0 else 0
        else:
            # if original dataset didn't have Total Refunds, try to infer if column present
            row['Was_Refunded'] = 0

    if 'Had_Extra_Data_Charge' in feature_columns:
        if 'Total Extra Data Charges' in user_input:
            row['Had_Extra_Data_Charge'] = 1 if float(user_input.get('Total Extra Data Charges', 0)) > 0 else 0
        else:
            row['Had_Extra_Data_Charge'] = 0

    if 'Has_Dependents' in feature_columns:
        if 'Number of Dependents' in user_input:
            row['Has_Dependents'] = 1 if float(user_input.get('Number of Dependents', 0)) > 0 else 0
        else:
            row['Has_Dependents'] = 0

    # Fillna for Offer and Internet Type (as in notebook)
    if 'Offer' in row and (row['Offer'] is None or str(row['Offer']).strip() == ""):
        row['Offer'] = 'Unknown'
    if 'Internet Type' in row and (row['Internet Type'] is None or str(row['Internet Type']).strip() == ""):
        row['Internet Type'] = 'Unknown'

    # apply log1p transforms for fields that were transformed in notebook
    for c in ['Avg Monthly GB Download', 'Total Long Distance Charges', 'Total Revenue', 'Number of Referrals']:
        if c in row:
            try:
                val = float(row[c])
                # ensure non-negative before log1p
                if val < 0:
                    # fallback to absolute or 0
                    val = abs(val)
                row[c] = np.log1p(val)
            except Exception:
                # if can't convert, set to 0
                row[c] = 0.0

    # Build DataFrame with the processed features (before scaling/encoding)
    proc_df = pd.DataFrame([row])

    # Standardize numeric continuous features: we must mirror the training selection:
    # numeric_features = df.select_dtypes(include='number').columns.tolist()
    # exclude_cols = ['Churn Value', 'Has_Dependents', 'Was_Refunded',
    #                 'Had_Extra_Data_Charge', 'Satisfaction Score']
    numeric_features = proc_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    exclude_cols = ['Churn Value', 'Has_Dependents', 'Was_Refunded', 'Had_Extra_Data_Charge', 'Satisfaction Score']
    fitur_standarisasi = [col for col in numeric_features if col not in exclude_cols]

    # Apply scaler to those fitur_standarisasi if present
    try:
        if len(fitur_standarisasi) > 0:
            proc_df[fitur_standarisasi] = scaler.transform(proc_df[fitur_standarisasi])
    except Exception as e:
        st.warning("Terjadi masalah saat scaling numeric features: " + str(e))

    # Encode categorical columns using encoder (encoder was fit on original df.select_dtypes(object))
    # Get categorical column names from raw_df as was done in training
    cat_cols_training = raw_df.select_dtypes(include=['object']).columns.tolist()
    if len(cat_cols_training) > 0:
        try:
            # gather values in the right order
            cat_values = []
            used_cat_cols = []
            for c in cat_cols_training:
                if c in proc_df.columns:
                    cat_values.append([proc_df.at[0, c]])
                    used_cat_cols.append(c)
            if len(cat_values) > 0:
                cat_arr = np.array(cat_values).T  # shape (1, n_cat)
                cat_enc = encoder.transform(cat_arr)
                # encoder outputs floats; convert to int
                for idx, c in enumerate(used_cat_cols):
                    proc_df[c] = int(cat_enc[0, idx])
        except Exception as e:
            st.warning("Terjadi masalah saat encoding kategori: " + str(e))

    # Now build final input_df with columns = feature_columns (the same used in training)
    input_df = pd.DataFrame(columns=feature_columns)
    input_df.loc[0] = 0  # default zeros

    # Fill values: there may be overlap between proc_df columns and feature_columns
    for c in proc_df.columns:
        if c in input_df.columns:
            input_df.at[0, c] = proc_df.at[0, c]

    # For engineered binary features (ensure present)
    for engineered in ['Was_Refunded', 'Had_Extra_Data_Charge', 'Has_Dependents']:
        if engineered in input_df.columns and engineered in proc_df.columns:
            input_df.at[0, engineered] = proc_df.at[0, engineered]

    # Safety: ensure all numeric columns are numeric dtype
    for col in input_df.columns:
        try:
            input_df[col] = pd.to_numeric(input_df[col])
        except Exception:
            # leave as is for categorical/encoded columns
            pass

    # === Predict ===
    try:
        prediction = model.predict(input_df)[0]
        # if predict_proba exists
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_df)[0]
            # try get prob for positive class (assume class 1 is churn)
            prob_churn = probability[1] if len(probability) > 1 else probability[0]
        else:
            prob_churn = None
    except Exception as e:
        st.error("Terjadi kesalahan saat memprediksi: " + str(e))
        return

    if prediction == 1 or (prob_churn is not None and prob_churn >= 0.5):
        st.error(f"❌ **Pelanggan tersebut berpotensi untuk churn.**")
        if prob_churn is not None:
            st.markdown(f"**Probabilitas churn:** `{prob_churn:.2f}`")
        st.error("""
        **Tindakan yang direkomendasikan:**
        - Lakukan pendekatan untuk memahami kebutuhan serta ketidakpuasan pelanggan.
        - Pertimbangkan untuk menawarkan benefit seperti diskon eksklusif maupun upgrade layanan.
        - Tinjau kembali riwayat langganan pelanggan untuk mengidentifikasi gangguan atau masalah pada layanan.
        """)
    else:
        st.info(f"✅ **Pelanggan tersebut diprediksi akan tetap loyal (non-churn).**")
        if prob_churn is not None:
            st.markdown(f"**Probabilitas churn:** `{prob_churn:.2f}`")
        st.info("""
        **Insight:**
        - Pelanggan tersebut tidak menunjukkan kecenderungan untuk churn.
        - Pertahankan loyalitas pelanggan tersebut dengan memberi reward maupun penawaran yang menarik.
        - Terus berikan pengalaman layanan yang konsisten dan memuaskan pada pelanggan yang loyal.
        """)
