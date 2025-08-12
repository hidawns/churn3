import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

def show():
    st.title("📁 Dataset: IBM Telco Customer Churn")
    # === Deskripsi Dataset ===
    st.markdown("""Dataset yang digunakan pada penelitian ini adalah IBM Telco Customer Churn yang disediakan oleh IBM Congnos Analytics (2019).
    Dataset ini terdiri dari 7043 observasi dengan 45 fitur yang memuat informasi pelanggan dari perusahaan telekomunikasi yang mencakup data demografi, jenis layanan yang digunakan, metode pembayaran, serta status churn pelanggan.""")
    
    df = pd.read_csv("merged_dataset.csv")
  
    # === Statistik Dataset ===
    with st.expander("📊 Statistik Dataset", expanded=True):
        total_rows = df.shape[0]
        total_columns = df.shape[1]
        target_col = 'Churn'
        feature_cols = df.drop(columns=[target_col]).shape[1] if target_col in df.columns else total_columns
        target_count = 1 if target_col in df.columns else 0
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Jumlah Baris**\n\n{total_rows}")
        with col2:
            st.info(f"**Jumlah Kolom**\n\n{total_columns}")
        with col3:
            st.info("**Proporsi Kelas Target**\n\n73.5%  :  24.5%")

    # === Sampel Dataset ===
    with st.expander("📄 Sampel Dataset"):
        st.dataframe(df.head(), use_container_width=True)

    # === Tipe Data Setiap Fitur ===
    with st.expander("🔎 Tipe Data Setiap Fitur"):
        info_df = pd.DataFrame({
            'Kolom': df.columns,
            'Tipe Data': df.dtypes.astype(str),
            'Non-Null Count': df.notnull().sum()
        }).reset_index(drop=True)
        st.dataframe(info_df, use_container_width=True)

    # === Statistik Deskriptif Fitur Numerik ===
    with st.expander("📈 Statistik Deskriptif Fitur Numerik"):
        st.dataframe(df.describe(), use_container_width=True)
