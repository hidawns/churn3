import streamlit as st
import pandas as pd

def show():
    st.title("📁 Dataset Baru")

    st.markdown("""
    Dataset terbaru ini digunakan untuk memprediksi churn pelanggan dengan fitur-fitur yang relevan.
    Dataset memiliki X baris dan Y kolom (ubah sesuai data kamu).
    """)

    df = pd.read_csv("nama_dataset_baru.csv")  # ganti sesuai file kamu

    # Contoh: ubah tipe data atau transformasi sesuai datasetmu
    # df['BeberapaKolom'] = df['BeberapaKolom'].astype('category')

    with st.expander("📊 Statistik Dataset", expanded=True):
        st.write(df.describe())
    
    with st.expander("📄 Sampel Dataset"):
        st.dataframe(df.head(), use_container_width=True)

    with st.expander("🔎 Tipe Data Setiap Fitur"):
        info_df = pd.DataFrame({
            'Kolom': df.columns,
            'Tipe Data': df.dtypes.astype(str),
            'Non-Null Count': df.notnull().sum()
        }).reset_index(drop=True)
        st.dataframe(info_df, use_container_width=True)
