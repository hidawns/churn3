import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show():
    sns.set_theme(style="whitegrid", palette="pastel")
    st.title("📈 Exploratory Data Analysis (EDA)")

    df = pd.read_csv("merged_dataset.csv")

    st.markdown("""EDA merupakan proses awal untuk memahami karakteristik dan pola data secara menyeluruh sebelum dilakukan pemodelan. 
    Eksplorasi Data dilakukan untuk memperoleh pemahaman awal terkait data yang akan digunakan.""")

    excluded_columns = ['Customer ID', 'Count', 'Quarter', 'Customer Status', 'Churn Label', 'Churn Score', 'Churn Category', 'Churn Reason']
    numeric_features_selected = df.drop(columns=[col for col in excluded_columns if col in df.columns]) \
                                    .select_dtypes(include=['number']).columns.tolist()
    categorical_features_selected = df.drop(columns=[col for col in excluded_columns if col in df.columns]) \
                                      .select_dtypes(include=['object']).columns.tolist()
    churn_col = 'Churn Value' if 'Churn Value' in df.columns else 'Churn'  # Sesuaikan jika beda nama

    # === SECTION: Univariate - Numerical ===
    with st.expander("📊 **Univariate Analysis - Fitur Numerik**", expanded=True):
        st.markdown("Melihat sebaran nilai dari setiap fitur numerik untuk memahami pola distribusinya.")

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

    # === SECTION: Univariate - Categorical ===
    with st.expander("📊 **Univariate Analysis - Fitur Kategorikal**", expanded=False):
        categorical_cols = df.select_dtypes(include='object').drop(columns='customerID').columns
        st.markdown("Melihat sebaran kategori yang dimiliki setiap fitur kategorikal untuk memahami pola distribusinya.")

        len_categorical = len(categorical_features_selected)
        cols = 3
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

        if churn_col not in df.columns:
            st.warning(f"Kolom target churn '{churn_col}' tidak ditemukan di data.")
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

    # === SECTION: Bivariate - Fitur Kategorikal vs Churn ===
    with st.expander("🔎 **Bivariate Analysis - Fitur Kategorikal vs Churn**", expanded=False):
        st.markdown( "Menganalisis hubungan setiap kategori pada fitur kategorikal terhadap masing-masing kelas dalam variabel target churn.")
       
         if churn_col not in df.columns:
            st.warning(f"Kolom target churn '{churn_col}' tidak ditemukan di data.")
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

    # === SECTION: Korelasi Heatmap ===
    with st.expander("📊 **Heatmap**", expanded=False):
        st.markdown("Menganalisis nilai korelasi antar variabel numerik.")

        if churn_col in df.columns:
            df[churn_col] = pd.to_numeric(df[churn_col], errors='coerce').fillna(0).astype(int)
        
        num_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns if col != 'Count']
        corr = df[num_cols].corr()
        
        plt.figure(figsize=(15, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
        plt.xticks(rotation=45, ha='right')
        plt.title('Heatmap Korelasi (mencakup seluruh variabel numerik)')
        st.pyplot(plt.gcf())
        plt.close()
