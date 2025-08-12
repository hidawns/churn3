import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

def show():
    sns.set_theme(style="whitegrid", palette="pastel")
    st.title("📈 Exploratory Data Analysis (EDA)")

    # Load dataset
    df = pd.read_csv("IBM Churn.csv")

    st.markdown("""
    EDA merupakan proses awal untuk memahami karakteristik dan pola data secara menyeluruh sebelum dilakukan pemodelan.
    """)

    # Kolom yang dikecualikan
    excluded_columns = [
        'Customer ID', 'Count', 'Quarter', 'Customer Status',
        'Churn Label', 'Churn Score', 'Churn Category', 'Churn Reason'
    ]

    # Filter hanya kolom yang benar-benar ada di dataframe
    drop_cols = [c for c in excluded_columns if c in df.columns]

    # Buat df untuk numerik & kategorikal
    df_num = df.drop(columns=drop_cols) if drop_cols else df.copy()
    df_cat = df.drop(columns=drop_cols) if drop_cols else df.copy()

    # Pilih fitur numerik & kategorikal
    numeric_features_selected = df_num.select_dtypes(include=['number']).columns.tolist()
    categorical_features_selected = df_cat.select_dtypes(include=['object']).columns.tolist()

    # Cari kolom churn
    churn_col = next((c for c in ['Churn Value', 'Churn'] if c in df.columns), None)

    # === Univariate Numerical ===
    with st.expander("📊 **Univariate Analysis - Fitur Numerik**", expanded=True):
        if not numeric_features_selected:
            st.warning("Tidak ada fitur numerik.")
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

    # === Univariate Categorical ===
    with st.expander("📊 **Univariate Analysis - Fitur Kategorikal**", expanded=False):
        if not categorical_features_selected:
            st.warning("Tidak ada fitur kategorikal.")
        else:
            len_categorical = len(categorical_features_selected)
            cols = 3
            rows = math.ceil(len_categorical / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
            axes = axes.flatten()
            for i, col in enumerate(categorical_features_selected):
                sns.countplot(data=df, x=col, order=df[col].value_counts().index, ax=axes[i])
                axes[i].set_title(f'Countplot of {col}', fontweight='bold')
                axes[i].tick_params(axis='x', rotation=45)
            for j in range(len_categorical, len(axes)):
                axes[j].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # === Bivariate Numerical vs Churn ===
    with st.expander("🔎 **Bivariate Analysis - Fitur Numerik vs Churn**", expanded=False):
        if churn_col is None:
            st.warning("Kolom target churn tidak ditemukan.")
        elif not numeric_features_selected:
            st.warning("Tidak ada fitur numerik.")
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

    # === Bivariate Categorical vs Churn ===
    with st.expander("🔎 **Bivariate Analysis - Fitur Kategorikal vs Churn**", expanded=False):
        if churn_col is None:
            st.warning("Kolom target churn tidak ditemukan.")
        elif not categorical_features_selected:
            st.warning("Tidak ada fitur kategorikal.")
        else:
            cols = 4
            rows = math.ceil(len(categorical_features_selected) / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
            axes = axes.flatten()
            for i, col in enumerate(categorical_features_selected):
                sns.countplot(data=df, x=col, hue=churn_col,
                              order=df[col].value_counts().index, palette='pastel', ax=axes[i])
                axes[i].set_title(f'{col} vs {churn_col}', fontweight='bold')
                axes[i].tick_params(axis='x', rotation=45)
            for j in range(len(categorical_features_selected), len(axes)):
                axes[j].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # === Heatmap ===
    with st.expander("📊 **Heatmap**", expanded=False):
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if not numeric_cols:
            st.warning("Tidak ada variabel numerik.")
        else:
            if churn_col and df[churn_col].dtype != 'int':
                df[churn_col] = pd.to_numeric(df[churn_col], errors='coerce').fillna(0).astype(int)
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(15, 10))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_title('Heatmap Korelasi')
            st.pyplot(fig)
            plt.close(fig)
