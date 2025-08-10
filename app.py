import streamlit as st
import modules.Home as Home
import modules.Overview as Overview
import modules.Dataset as Dataset
import modules.EDA as EDA
import modules.Inference as Inference

st.set_page_config(page_title="Churn Prediction App", layout="wide", page_icon="📊")

# === Sidebar Navigasi ===
with st.sidebar:
    st.markdown("## Navigasi")

    # Navigasi menggunakan tombol biasa
    if st.button("▶ Home"):
        st.session_state.page = "Home"
    if st.button("▶ Churn Overview"):
        st.session_state.page = "Overview"
    if st.button("▶ Dataset"):
        st.session_state.page = "Dataset"
    if st.button("▶ EDA"):
        st.session_state.page = "EDA"
    if st.button("▶ Prediksi"):
        st.session_state.page = "Inference"

    st.markdown("---")
    st.caption("© 2025 | Churn Prediction App")

# === Set default halaman jika belum ada ===
if "page" not in st.session_state:
    st.session_state.page = "Home"

# === Routing Halaman ===
page = st.session_state.page
if page == "Home":
    Home.show()
elif page == "Overview":
    Overview.show()
elif page == "Dataset":
    Dataset.show()
elif page == "EDA":
    EDA.show()
elif page == "Inference":
    Inference.show()
