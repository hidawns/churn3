import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

def show():
    sns.set_theme(style="whitegrid", palette="pastel")
    st.title("📈 Exploratory Data Analysis (EDA)")

    # Load data
    df = pd.read_csv("merged_dataset.csv")

    st.markdown("""
    EDA merupakan proses awal untuk memahami karakteristik dan pola data secara menyeluruh sebelum dilakukan pemodelan. 
    Eksplorasi Data dilakukan untuk memperoleh pemahaman awal terkait data yang akan digunakan.
    """)

    # Tentukan fitur numerik & kategorikal
    excluded_columns = [
        'Customer ID', 'Count', 'Quarter', 'Customer Status', 
        'Churn Label', 'Churn Score', 'Churn Category', 'Churn Reason'
    ]
    numeric_features_selected = df.drop(columns=excluded_columns, errors='ignore').select_dtypes(include=['number']).columns.tolist()
    categorical_features_selected = df.drop(columns=excluded_columns, errors='ignore').select_dtypes(include=['object']).columns.tolist()

    # === SECTION: Univariate - Numerical ===
    with st.expander("📊 **Univariate Analysis - Fitur Numerik**", expanded=True):
        st.markdown("Melihat sebaran nilai dari setiap fitur numerik untuk memahami pola distribusinya.")
        len_numeric = len(numeric_features_selected)
        cols = 4
        rows = math.ceil(len_numeric / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
        axes = axes.flatten()
        for i, col in enumerate(numeric_features_selected):
            sns.histplot(data=df, x=col, kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}', fontweight='bold', pad=15)
        for j in range(len_numeric, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

    # === SECTION: Univariate - Categorical ===
    with st.expander("📊 **Univariate Analysis - Fitur Kategorikal**", expanded=False):
        st.markdown("Melihat sebaran kategori yang dimiliki setiap fitur kategorikal untuk memahami pola distribusinya.")
        len_categorical = len(categorical_features_selected)
        cols = 4
        rows = math.ceil(len_categorical / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
        axes = axes.flatten()
        for i, col in enumerate(categorical_features_selected):
            sns.countplot(data=df, x=col, ax=axes[i], order=df[col].value_counts().index)
            axes[i].set_title(f'Countplot of {col}', fontweight='bold')
            axes[i].tick_params(axis='x', rotation=45)
        for j in range(len_categorical, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

    # === SECTION: Bivariate - Numerik vs Churn ===
    with st.expander("🔎 **Bivariate Analysis - Fitur Numerik vs Churn**", expanded=False):
        st.markdown("Menganalisis hubungan setiap fitur numerik terhadap masing-masing kelas dalam variabel target churn.")
        if 'Churn Value' in df.columns:
            df['Churn Value'] = df['Churn Value'].astype(str)
            numeric_features = [col for col in numeric_features_selected if col != 'Churn Value']
            len_numeric = len(numeric_features)
            cols = 4
            rows = math.ceil(len_numeric / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
            axes = axes.flatten()
            for i, col in enumerate(numeric_features):
                sns.violinplot(data=df, x='Churn Value', y=col, hue='Churn Value',
                               palette='pastel', ax=axes[i], legend=False)
                axes[i].set_title(f'{col} vs Churn Value')
            for j in range(len_numeric, len(axes)):
                axes[j].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Kolom 'Churn Value' tidak ditemukan di dataset.")

    # === SECTION: Bivariate - Fitur Kategorikal vs Churn ===
    with st.expander("🔎 **Bivariate Analysis - Fitur Kategorikal vs Churn**", expanded=False):
        st.markdown("Menganalisis hubungan setiap kategori pada fitur kategorikal terhadap masing-masing kelas dalam variabel target churn.")
        if 'Churn Value' in df.columns:
            df['Churn Value'] = df['Churn Value'].astype(str)
            len_categorical = len(categorical_features_selected)
            cols = 4
            rows = math.ceil(len_categorical / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6))
            axes = axes.flatten()
            for i, col in enumerate(categorical_features_selected):
                sns.countplot(data=df, x=col, hue='Churn Value', 
                              order=df[col].value_counts().index, palette='pastel', ax=axes[i])
                axes[i].set_title(f'{col} vs Churn Value', fontweight='bold')
                axes[i].tick_params(axis='x', rotation=45)
            for j in range(len_categorical, len(axes)):
                axes[j].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Kolom 'Churn Value' tidak ditemukan di dataset.")

    # === SECTION: Korelasi Heatmap ===
    with st.expander("📊 **Heatmap**", expanded=False):
        st.markdown("Menganalisis nilai korelasi antar variabel numerik.")
        if 'Churn Value' in df.columns:
            df['Churn Value'] = df['Churn Value'].astype(int)
        num = [col for col in df.select_dtypes(include=['int64', 'float64']).columns if col != 'Count']
        if num:
            corr = df[num].corr()
            fig, ax = plt.subplots(figsize=(15, 10))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            plt.xticks(rotation=45, ha='right')
            plt.title('Heatmap Korelasi (mencakup seluruh variabel numerik)')
            st.pyplot(fig)
        else:
            st.warning("Tidak ada fitur numerik untuk ditampilkan pada heatmap.")
