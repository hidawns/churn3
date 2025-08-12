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

    # Tentukan kolom yang dikecualikan
    excluded_columns = [
        'Customer ID', 'Count', 'Quarter', 'Customer Status',
        'Churn Label', 'Churn Score', 'Churn Category', 'Churn Reason'
    ]

    # Pastikan hanya kolom yang ada di data yang di-drop
    excluded_columns = [col for col in excluded_columns if col in df.columns]

    # Pilih fitur numerik & kategorikal
    numeric_features_selected = df.drop(columns=excluded_columns, errors='ignore') \
                                   .select_dtypes(include=['number']).columns.tolist()
    categorical_features_selected = df.drop(columns=excluded_columns, errors='ignore') \
                                       .select_dtypes(include=['object']).columns.tolist()

    # Tentukan nama kolom churn
    churn_col = None
    for possible_col in ['Churn Value', 'Churn']:
        if possible_col in df.columns:
            churn_col = possible_col
            break

    # === SECTION: Univariate - Numerical ===
    with st.expander("📊 **Univariate Analysis - Fitur Numerik**", expanded=True):
        if len(numeric_features_selected) == 0:
            st.warning("Tidak ada fitur numerik yang ditemukan.")
        else:
            len_numeric = len(numeric_features_selected)
            cols = 4
            rows = math.ceil(len_numeric / cols)

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
            axes = axes.flatten()

            for i, col in enumerate(numeric_features_selected):
                sns.histplot(data=df, x=col, kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {col}', fontweight='bold')

            for j in range(len_numeric, len(axes)):
                axes[j].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # === SECTION: Univariate - Categorical ===
    with st.expander("📊 **Univariate Analysis - Fitur Kategorikal**", expanded=False):
        if len(categorical_features_selected) == 0:
            st.warning("Tidak ada fitur kategorikal yang ditemukan.")
        else:
            len_categorical = len(categorical_features_selected)
            cols = 3
            rows = math.ceil(len_categorical / cols)

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
            axes = axes.flatten()

            for i, col in enumerate(categorical_features_selected):
                sns.countplot(data=df, x=col, ax=axes[i],
                              order=df[col].value_counts().index)
                axes[i].set_title(f'Countplot of {col}', fontweight='bold')
                axes[i].tick_params(axis='x', rotation=45)

            for j in range(len_categorical, len(axes)):
                axes[j].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # === SECTION: Bivariate - Numerik vs Churn ===
    with st.expander("🔎 **Bivariate Analysis - Fitur Numerik vs Churn**", expanded=False):
        if churn_col is None:
            st.warning("Kolom target churn tidak ditemukan di dataset.")
        elif len(numeric_features_selected) == 0:
            st.warning("Tidak ada fitur numerik yang ditemukan.")
        else:
            cols = 5
            rows = math.ceil(len(numeric_features_selected) / cols)

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
            axes = axes.flatten()

            for i, col in enumerate(numeric_features_selected):
                sns.violinplot(data=df, x=churn_col, y=col, hue=churn_col,
                               palette='pastel', ax=axes[i], legend=False)
                axes[i].set_title(f'{col} vs {churn_col}')

            for j in range(len(numeric_features_selected), len(axes)):
                axes[j].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # === SECTION: Bivariate - Kategorikal vs Churn ===
    with st.expander("🔎 **Bivariate Analysis - Fitur Kategorikal vs Churn**", expanded=False):
        if churn_col is None:
            st.warning("Kolom target churn tidak ditemukan di dataset.")
        elif len(categorical_features_selected) == 0:
            st.warning("Tidak ada fitur kategorikal yang ditemukan.")
        else:
            cols = 4
            rows = math.ceil(len(categorical_features_selected) / cols)

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
            axes = axes.flatten()

            for i, col in enumerate(categorical_features_selected):
                sns.countplot(data=df, x=col, hue=churn_col, ax=axes[i],
                              order=df[col].value_counts().index, palette='pastel')
                axes[i].set_title(f'{col} vs {churn_col}', fontweight='bold')
                axes[i].tick_params(axis='x', rotation=45)

            for j in range(len(categorical_features_selected), len(axes)):
                axes[j].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # === SECTION: Korelasi Heatmap ===
    with st.expander("📊 **Heatmap**", expanded=False):
        if df.select_dtypes(include=['number']).shape[1] == 0:
            st.warning("Tidak ada variabel numerik untuk membuat heatmap.")
        else:
            if churn_col and df[churn_col].dtype != 'int':
                df[churn_col] = pd.to_numeric(df[churn_col], errors='coerce').fillna(0).astype(int)

            num_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns if col != 'Count']
            corr = df[num_cols].corr()

            fig, ax = plt.subplots(figsize=(15, 10))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_title('Heatmap Korelasi (mencakup seluruh variabel numerik)')
            st.pyplot(fig)
            plt.close(fig)
