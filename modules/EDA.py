import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show():
    sns.set_theme(style="whitegrid", palette="pastel")
    st.title("📈 Exploratory Data Analysis (EDA)")

    try:
        df = pd.read_csv("merged_dataset.csv")
    except FileNotFoundError:
        st.error("File `merged_dataset.csv` tidak ditemukan di root repository. Mohon letakkan file dataset di root.")
        return

    # some light cleanup for plotting readability
    if 'Customer ID' in df.columns:
        # jangan tunjukkan ID
        pass

    st.markdown("""EDA merupakan proses awal untuk memahami karakteristik dan pola data secara menyeluruh sebelum dilakukan pemodelan.""")

    # pilih kolom numerik yang umum untuk plot (maks 3)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    # cobalah pilih tiga kolom yang relevan jika ada
    chosen_numeric = []
    candidates = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Total Revenue', 'Avg Monthly GB Download', 'Satisfaction Score']
    for c in candidates:
        if c in numeric_cols and len(chosen_numeric) < 3:
            chosen_numeric.append(c)
    # fallback ke tiga numerik pertama
    if len(chosen_numeric) == 0:
        chosen_numeric = numeric_cols[:3]

    if chosen_numeric:
        with st.expander("📊 **Univariate Analysis - Fitur Numerik**", expanded=True):
            st.markdown("Melihat sebaran nilai dari setiap fitur numerik.")
            fig, axs = plt.subplots(1, len(chosen_numeric), figsize=(6 * len(chosen_numeric), 4))
            if len(chosen_numeric) == 1:
                axs = [axs]
            for i, col in enumerate(chosen_numeric):
                sns.histplot(data=df, x=col, kde=True, ax=axs[i])
                axs[i].set_title(f'Distribusi {col}')
            st.pyplot(fig)

    # kategorikal
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    if len(categorical_cols) > 0:
        with st.expander("📊 **Univariate Analysis - Fitur Kategorikal**", expanded=False):
            st.markdown("Melihat sebaran kategori dari beberapa fitur kategorikal.")
            # tampilkan hingga 6 grafik
            cols_to_plot = categorical_cols[:6]
            fig, axs = plt.subplots(len(cols_to_plot), 1, figsize=(10, 4 * len(cols_to_plot)))
            if len(cols_to_plot) == 1:
                axs = [axs]
            for i, col in enumerate(cols_to_plot):
                sns.countplot(data=df, x=col, ax=axs[i])
                axs[i].set_title(col)
                axs[i].tick_params(axis='x', rotation=45)
            st.pyplot(fig)

    # bivariate numeric vs churn jika ada
    if 'Churn Value' in df.columns and len(chosen_numeric) > 0:
        with st.expander("🔎 **Bivariate - Numerik vs Churn**", expanded=False):
            fig, axs = plt.subplots(len(chosen_numeric), 1, figsize=(10, 5 * len(chosen_numeric)))
            if len(chosen_numeric) == 1:
                axs = [axs]
            for i, col in enumerate(chosen_numeric):
                sns.boxplot(data=df, x='Churn Value', y=col, ax=axs[i])
                axs[i].set_title(f'{col} vs Churn')
            st.pyplot(fig)

    # korelasi heatmap untuk numerik
    numcols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numcols) >= 2:
        with st.expander("📊 **Heatmap**", expanded=False):
            corr = df[numcols].corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            ax.set_title('Heatmap Korelasi Variabel Numerik')
            st.pyplot(fig)
