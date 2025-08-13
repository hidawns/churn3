import streamlit as st

def show():
    st.title("Churn Prediction App")
    st.markdown("##### Optimalisasi Model XGBoost dengan Teknik Hybrid Resampling SMOTE-ENN dan Hyperparameter Tuning GridSearchCV dalam Prediksi Churn Pelanggan")
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
        Aplikasi ini memprediksi churn pelanggan menggunakan model terbaru yang dikembangkan dengan dataset dan teknik preprocessing terkini.
        </div>
        """, unsafe_allow_html=True)

        st.markdown(" ")
        if st.button("**Pergi ke Halaman Prediksi**"):
            st.session_state.page = "Inference"
            st.experimental_rerun()

    with col2:
        st.image("assets/churn_image4.jpg", use_container_width=True)

    st.markdown("---")
