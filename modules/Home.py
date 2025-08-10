import streamlit as st

def show():
    st.title("Churn Prediction App")
    st.markdown("##### Optimalisasi Model XGBoost Menggunakan Teknik Hybrid Resampling SMOTE-ENN<br>dan Hyperparameter Tuning GridSearchCV dalam Prediksi Churn Pelanggan", unsafe_allow_html=True)
    st.write("Dikembangkan oleh: Hidayati Tri Winasis")

    st.markdown("---")  
    col1, col2 = st.columns([3, 1])
    with col1: 
        st.markdown("""
        <div style='
            background-color: #e8f2fc;
            padding: 16px;
            border-radius: 6px;
            color: #004280;
            line-height: 1.4;
            text-align: justify'>
        Aplikasi ini bertujuan untuk memprediksi kemungkinan seorang pelanggan melakukan churn. Aplikasi dibangun dengan mengimplementasikan machine learning melalui algoritma <b>XGBoost</b> 
        yang dioptimalkan dengan teknik resampling <b>SMOTE-ENN</b> serta hyperparameter tuning <b>GridSearchCV</b>.
        </div>
        """, unsafe_allow_html=True)

        
        st.markdown(" ")
        if st.button("**Pergi ke Halaman Prediksi**"):
            st.session_state.page = "Inference"
            st.rerun()
        
            
    with col2:
        st.image("assets/churn_image4.jpg", use_container_width=True)
        
    st.markdown("---")
