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
st.title('Dashboard Prediksi Kategori Stunting')

# Input form untuk memasukkan data
st.header('Input Data')
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

# Tombol untuk prediksi
if st.button('Prediksi Kategori Stunting'):
    hasil = predict_stunting(umur, jenis_kelamin, tinggi_badan)
    st.success(f'Hasil Prediksi: {hasil}')
# Tombol untuk prediksi
if st.button('Prediksi Kategori Stunting'):
    hasil = predict_stunting(umur, jenis_kelamin, tinggi_badan)
    st.success(f'Hasil Prediksi: {hasil}')

# SHAP Interpretasi (tidak diubah dari sebelumnya)
st.header('Interpretasi SHAP')
st.write('Visualisasi ini membantu memahami fitur mana yang berpengaruh pada prediksi model.')

# Simulasi plot (tidak diubah dari sebelumnya)
st.pyplot()
