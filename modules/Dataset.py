import streamlit as st
import pandas as pd

def show():
    st.title("📂 Dataset")
    st.markdown("Berikut adalah cuplikan dataset yang digunakan pada model terbaru:")

    df = pd.read_csv("merged_dataset.csv")
    st.dataframe(df.head())

    st.markdown(f"**Jumlah baris:** {df.shape[0]}")
    st.markdown(f"**Jumlah kolom:** {df.shape[1]}")
