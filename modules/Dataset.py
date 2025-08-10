import streamlit as st
import pandas as pd

def show():
    st.title("📁 Dataset: Merged / Updated Churn Dataset")
    st.markdown("""Dataset yang digunakan pada versi terbaru ini adalah `merged_dataset.csv`. Halaman ini menampilkan ringkasan, sampel, dan tipe data dari dataset terbaru yang dipakai untuk pelatihan model.""")
    
    try:
        df = pd.read_csv("merged_dataset.csv")
    except FileNotFoundError:
        st.error("File `merged_dataset.csv` tidak ditemukan di root repository. Mohon letakkan file dataset di root.")
        return

    # meniru preprocessing ringan baca untuk tampilan
    # konversi beberapa kolom bila perlu
    if 'Total Revenue' in df.columns:
        # tampilkan ringkasan dasar
        pass

    # === Statistik Dataset ===
    with st.expander("📊 Statistik Dataset", expanded=True):
        total_rows = df.shape[0]
        total_columns = df.shape[1]
        target_col = 'Churn Value'
        feature_cols = df.drop(columns=[target_col]).shape[1] if target_col in df.columns else total_columns
        target_count = 1 if target_col in df.columns else 0
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Jumlah Baris**\n\n{total_rows}")
        with col2:
            st.info(f"**Jumlah Kolom**\n\n{total_columns}")
        with col3:
            if target_count:
                prop = df[target_col].value_counts(normalize=True).to_dict()
                st.info(f"**Proporsi Kelas Target**\n\n{prop}")
            else:
                st.info("**Proporsi Kelas Target**\n\nTarget tidak ditemukan.")

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
        try:
            st.dataframe(df.describe(), use_container_width=True)
        except Exception as e:
            st.write("Gagal menampilkan statistik deskriptif:", e)
