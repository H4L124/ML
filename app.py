import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# Fungsi untuk load CSS dari file eksternal
def local_css(file_path):
    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Panggil fungsi untuk memuat CSS
local_css("style.css")

# Title aplikasi
st.title('Dashboard Deteksi Stunting pada Balita')

# Sidebar untuk navigasi halaman
st.sidebar.subheader("Dashboard Deteksi Stunting Balita")
st.sidebar.subheader("Disusun Oleh:")
st.sidebar.write("Kelompok 3 Kelas A")
st.sidebar.subheader("Anggota Kelompok:")
st.sidebar.write("Nur Haliza Rositasari (2043211069)")
st.sidebar.write("Hanna Naza Syajidha (2043211019)")
st.sidebar.write("Bibit Eka Wahyuni (2043211053)")
page = st.sidebar.selectbox('Navigasi Halaman', 
                            ['Karakteristik Stunting Menurut Provinsi di Indonesia', 
                             'Faktor-faktor yang Memengaruhi Kejadian Stunting Balita', 
                             'Prediksi Stunting pada Balita'])

# Halaman 1: Karakteristik Stunting Menurut Provinsi di Indonesia
if page == 'Karakteristik Stunting Menurut Provinsi di Indonesia':
    st.header('Karakteristik Stunting Menurut Provinsi di Indonesia')
    st.write('Tampilkan data, peta, atau visualisasi yang menggambarkan karakteristik stunting berdasarkan provinsi di Indonesia.')

# Halaman 2: Faktor-faktor yang Memengaruhi Kejadian Stunting Balita
if page == 'Faktor-faktor yang Memengaruhi Kejadian Stunting Balita':
    st.header('Faktor-faktor yang Memengaruhi Kejadian Stunting Balita')
    st.write('Jelaskan dan visualisasikan faktor-faktor utama yang berkontribusi terhadap kejadian stunting pada balita, seperti faktor gizi, kesehatan, dan lainnya.')

# Halaman 3: Prediksi Stunting pada Balita
if page == 'Prediksi Stunting pada Balita':
    st.header('Prediksi Stunting pada Balita')

    # Input form untuk memasukkan data
    st.subheader('Input Data')
    umur = st.number_input('Umur (bulan)', min_value=0, max_value=60, value=24)
    jenis_kelamin = st.selectbox('Jenis Kelamin', ('Perempuan', 'Laki-laki'))
    # Konversi menjadi 0 dan 1
    if jenis_kelamin == 'Laki-laki':
        jenis_kelamin = 0
    else:
        jenis_kelamin = 1
    tinggi_badan = st.number_input('Tinggi Badan (cm)', min_value=00.0, max_value=150.0, value=80.0)

    # Load model dan scaler
    model = joblib.load('model_stunting_multinomial.pkl')
    scaler = joblib.load('scaler_stunting.pkl')

    # Fungsi untuk prediksi kategori stunting
    def predict_stunting(umur, jenis_kelamin, tinggi_badan):
        # Preprocess input data
        input_data = np.array([[umur, jenis_kelamin, tinggi_badan]])  # Input dalam format 2D
        input_data_scaled = scaler.transform(input_data)  # Transformasi data menggunakan scaler
        prediction = model.predict(input_data_scaled)  # Lakukan prediksi
        return prediction[0]  # Mengembalikan hasil prediksi

    # Fungsi untuk mengubah angka kategori menjadi label
    def map_hasil(category):
        mapping = {
            0: "Severity Stunting",
            1: "Stunting",
            2: "Normal",
            3: "Tinggi"
        }
        return mapping.get(category, "Unknown")

    # Tombol untuk prediksi
    if st.button('Prediksi Kategori Stunting'):
        hasil = predict_stunting(umur, jenis_kelamin, tinggi_badan)
        hasil_label = map_hasil(hasil)
        st.success(f'Hasil Prediksi: {hasil_label}')

    # SHAP Interpretasi
    st.subheader('Interpretasi SHAP')
    st.write('Visualisasi ini membantu memahami fitur mana yang berpengaruh pada prediksi model.')

    # Simulasi plot SHAP
    st.pyplot()
