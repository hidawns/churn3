import streamlit as st
from modules import Home, Overview, Dataset, EDA, Inference

# Sidebar Menu
st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio("Pilih Halaman:", (
    "Home",
    "Overview",
    "Dataset",
    "EDA",
    "Inference"
))

# Routing
if menu == "Home":
    Home.show()
elif menu == "Overview":
    Overview.show()
elif menu == "Dataset":
    Dataset.show()
elif menu == "EDA":
    EDA.show()
elif menu == "Inference":
    Inference.show()
