import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show():
    sns.set_theme(style="whitegrid", palette="pastel")
    st.title("📈 Exploratory Data Analysis (EDA)")

    df = pd.read_csv("merged_dataset.csv")
    
    # Drop columns not used for analysis
    drop_cols = ['Customer ID', 'Customer Status', 'Churn Label', 'Churn Score', 'Churn Category', 'Churn Reason']
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=col)

    st.markdown("""EDA merupakan proses awal untuk memahami karakteristik dan pola data secara menyeluruh sebelum dilakukan pemodelan. 
    Eksplorasi Data dilakukan untuk memperoleh pemahaman awal terkait data yang akan digunakan.""")

    # === SECTION: Univariate - Numerical ===
    with st.expander("📊 **Univariate Analysis - Fitur Numerik**", expanded=True):
        st.markdown("Melihat sebaran nilai dari setiap fitur numerik untuk memahami pola distribusinya.")

        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        numeric_cols.remove('Churn Value')
        selected_num = st.selectbox("Select numeric feature to plot histogram", numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[selected_num].dropna(), kde=True, ax=ax)
        st.pyplot(fig)

    # === SECTION: Univariate - Categorical ===
    with st.expander("📊 **Univariate Analysis - Fitur Kategorikal**", expanded=False):
        categorical_cols = df.select_dtypes(include='object').drop(columns='customerID').columns
        st.markdown("Melihat sebaran kategori yang dimiliki setiap fitur kategorikal untuk memahami pola distribusinya.")

        cat_cols = df.select_dtypes(include='object').columns.tolist()
        if cat_cols:
            selected_cat = st.selectbox("Select categorical feature to plot countplot", cat_cols)
            fig, ax = plt.subplots()
            sns.countplot(x=selected_cat, data=df, order=df[selected_cat].value_counts().index, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.write("No categorical features available.")

    # === SECTION: Bivariate - Numerik vs Churn ===
    with st.expander("🔎 **Bivariate Analysis - Fitur Numerik vs Churn**", expanded=False):
        st.markdown("Menganalisis hubungan setiap fitur numerik terhadap masing-masing kelas dalam variabel target churn.")

        fig1, axs1 = plt.subplots(1, 3, figsize=(18, 5))
        for i, col in enumerate(numeric_cols):
            sns.boxplot(data=df, x='Churn', y=col, ax=axs1[i], palette='pastel')
            axs1[i].set_title(f'{col} vs Churn', fontweight='bold')
        st.pyplot(fig1)

        fig2, axs2 = plt.subplots(1, 3, figsize=(18, 6))
        for i, col in enumerate(numeric_cols):
            sns.violinplot(data=df, x='Churn', y=col, ax=axs2[i], palette='pastel')
            axs2[i].set_title(f'{col} vs Churn', fontweight='bold')
        st.pyplot(fig2)

    # === SECTION: Bivariate - Fitur Kategorikal vs Churn ===
    with st.expander("🔎 **Bivariate Analysis - Fitur Kategorikal vs Churn**", expanded=False):
        st.markdown( "Menganalisis hubungan setiap kategori pada fitur kategorikal terhadap masing-masing kelas dalam variabel target churn.")
       
        fig, axs = plt.subplots(6, 3, figsize=(18, 30))
        axs = axs.ravel()
        for i, col in enumerate(categorical_cols):
            sns.countplot(data=df, x=col, hue='Churn', ax=axs[i], palette='pastel')
            axs[i].set_title(f'{col} vs Churn', fontweight='bold')
            axs[i].tick_params(axis='x', rotation=45)
        for j in range(len(categorical_cols), len(axs)):
            fig.delaxes(axs[j])
        fig.tight_layout(h_pad=3)
        st.pyplot(fig)

    # === SECTION: Korelasi Heatmap ===
    with st.expander("📊 **Heatmap**", expanded=False):
        st.markdown("Menganalisis nilai korelasi antar variabel numerik.")

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
