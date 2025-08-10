import streamlit as st

def show():
    st.title("Mengenal Customer Churn")

    st.markdown("### Apa Itu Customer Churn?")
    st.markdown("""
- Customer churn adalah kondisi dimana seorang pelanggan berhenti menggunakan layanan dari suatu perusahaan, dan beralih menggunakan layanan dari perusahaan lain.
- Churn merupakan tantangan utama yang kerap dihadapi oleh industri telekomunikasi dengan kondisi pasar yang dipenuhi persaingan antar penyedia layanan.
""")

    st.markdown("---")
    st.markdown("### Penyebab Churn Pelanggan")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Harga Tidak Kompetitif**\n\nHarga yang tidak kompetitif dengan pesaing mendorong pelanggan mencari alternatif lain.")
    
    with col2:
        st.info("**Kualitas Layanan Buruk**\n\nKualitas layanan yang tidak memuaskan dapat memicu pelanggan untuk churn.")
    
    with col3:
        st.info("**Tidak Memenuhi Kebutuhan**\n\nLayanan yang ditawarkan tidak relevan dengan kebutuhan dan preferensi pelanggan.")

    st.markdown("---")

    st.markdown("### Dampak Churn Pelanggan")
    st.markdown("""
- Tingginya tingkat churn pelanggan dapat menyebabkan penurunan pendapatan perusahaan secara signifikan.
- Perusahaan juga harus mengeluarkan biaya yang lebih besar untuk dapat mengakuisisi pelanggan baru.
- Upaya mengakuisisi pelanggan baru umumnya memakan biaya yang jauh lebih besar dibandingkan mempertahankan pelanggan yang sudah ada.
""")

    st.markdown("---")

    st.markdown("### Manfaat prediksi churn")
    st.info("""
- Prediksi churn membantu perusahaan mengidentifikasi secara dini pelanggan yang berisiko berhenti menggunakan produk atau layanan. 
- Dengan informasi ini, perusahaan dapat merancang strategi retensi yang lebih efektif untuk mempertahankan pelanggan yang dimiliki. 
- Upaya ini berkontribusi untuk mengurangi biaya akuisisi pelanggan baru serta menjaga stabilitas bisnis perusahaan.
""")
