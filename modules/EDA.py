import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def show():
    st.title("📈 Exploratory Data Analysis")

    df = pd.read_csv("merged_dataset.csv")

    st.subheader("Distribusi Target (Churn Value)")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="Churn Value", ax=ax)
    st.pyplot(fig)

    st.subheader("Distribusi Umur")
    fig, ax = plt.subplots()
    sns.histplot(df["Age"], kde=True, ax=ax)
    st.pyplot(fig)
