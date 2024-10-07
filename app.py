import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# Fungsi untuk load CSS dari file eksternal
def local_css(style.css):
    with open(style.css) as f:
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
tinggi_badan = st.number_input('Tinggi Badan (cm)', min_value=30.0, max_value=150.0, value=80.0)

# Fungsi untuk prediksi kategori stunting
def predict_stunting(umur, jenis_kelamin, tinggi_badan):
    # Dummy output untuk contoh
    return 'Stunted'

# Tombol untuk prediksi
if st.button('Prediksi Kategori Stunting'):
    hasil = predict_stunting(umur, jenis_kelamin, tinggi_badan)
    st.success(f'Hasil Prediksi: {hasil}')

# SHAP Interpretasi (tidak diubah dari sebelumnya)
st.header('Interpretasi SHAP')
st.write('Visualisasi ini membantu memahami fitur mana yang berpengaruh pada prediksi model.')

# Simulasi plot (tidak diubah dari sebelumnya)
st.pyplot()
