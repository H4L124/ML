import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import plotly.express as px

# Fungsi untuk load CSS dari file eksternal
def local_css(file_path):
    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Panggil fungsi untuk memuat CSS
local_css("style.css")

# Title aplikasi
st.title('Dashboard Deteksi Stunting pada Balita')
# Sidebar untuk navigasi halaman
page = st.sidebar.selectbox('Navigasi Halaman', 
                            ['Karakteristik Stunting Menurut Provinsi di Indonesia', 
                             'Faktor-faktor yang Memengaruhi Kejadian Stunting Balita', 
                             'Prediksi Stunting pada Balita'])

st.sidebar.subheader("Dashboard Deteksi Stunting Balita")
st.sidebar.subheader("Disusun Oleh:")
st.sidebar.write("Kelompok 3 Kelas A")
st.sidebar.subheader("Anggota Kelompok:")
st.sidebar.write("Nur Haliza Rositasari (2043211069)")
st.sidebar.write("Hanna Naza Syajidha (2043211019)")
st.sidebar.write("Bibit Eka Wahyuni (2043211053)")

# Halaman 1: Karakteristik Stunting Menurut Provinsi di Indonesia
if page == 'Karakteristik Stunting Menurut Provinsi di Indonesia':
    st.header('Karakteristik Stunting Menurut Provinsi di Indonesia')
    
    # Load data from Excel file
    data_path = 'stunting.xlsx'
    df = pd.read_excel(data_path)

    # Load the shapefile (GeoJSON format) from a local file
    geojson_path = 'batas_provinsi.geojson'
    provinces_geo = gpd.read_file(geojson_path)

    # Ensure column names match between the GeoDataFrame and your DataFrame
    provinces_geo = provinces_geo.merge(df, how='left', left_on='Provinsi', right_on='Province')

    # Plot the map
    fig = px.choropleth(provinces_geo,
                        geojson=provinces_geo.geometry,
                        locations=provinces_geo.index,
                        color='Persentase Kasus Stunting (%)',
                        hover_name='Provinsi',
                        hover_data=['Jumlah Balita', 'Stunting', 'Severity Stunting'],
                        title='Peta Persebaran Stunting di Indonesia',
                        color_continuous_scale='YlOrRd')

    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0})

    # Display the map
    st.plotly_chart(fig)
    st.write('Peta prevelensi stunting di Indonesia menunjukkan bahwa persebaran stunting dengan perbedaan warna pada setiap daerah dengan rentang warna kuning hingga merah tua, semakin gelap warna menunjukkan semakin tinggi jumlah kejadian stunting. Pada peta menunjukkan Provinsi Sulawesi Selatan dan NTT memiliki prevelensi stunting yang tinggi, Papua dan Papua Barat memiliki prevelensi sedang hingga tinggi')

# Halaman 3: Prediksi Stunting pada Balita
if page == 'Prediksi Stunting pada Balita':
    st.header('Prediksi Stunting pada Balita')

    # Input form untuk memasukkan data
    st.subheader('Input Data')
    umur = st.number_input('Umur (bulan)', min_value=0, max_value=60, value=24)
    jenis_kelamin = st.selectbox('Jenis Kelamin', ('Perempuan', 'Laki-laki'))
    
    # Konversi menjadi 0 dan 1
    jenis_kelamin = 0 if jenis_kelamin == 'Laki-laki' else 1

    tinggi_badan = st.number_input('Tinggi Badan (cm)', min_value=0.0, max_value=150.0, value=80.0)

    # Load model dan scaler dengan error handling
    try:
        model = joblib.load('model_stunting_multinomial.pkl')
        scaler = joblib.load('scaler_stunting.pkl')
    except FileNotFoundError:
        st.error("Model atau scaler tidak ditemukan. Pastikan file sudah benar.")
        st.stop()

    # Fungsi untuk prediksi kategori stunting
    def predict_stunting(umur, jenis_kelamin, tinggi_badan):
        input_data = np.array([[umur, jenis_kelamin, tinggi_badan]])
        input_data_scaled = scaler.transform(input_data)  # Transformasi data
        prediction = model.predict(input_data_scaled)  # Prediksi
        return prediction[0]

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
        if tinggi_badan <= 0:
            st.error("Tinggi badan harus lebih dari 0 cm.")
        else:
            hasil = predict_stunting(umur, jenis_kelamin, tinggi_badan)
            hasil_label = map_hasil(hasil)

            # Tampilkan hasil prediksi dengan warna dinamis
            if hasil in [0, 1]:  # Jika Severity Stunting atau Stunting
                st.markdown(
                    f"<div style='color: red; font-weight: bold;'>Hasil Prediksi: {hasil_label}</div>",
                    unsafe_allow_html=True
                )
                st.warning("Jika anak teridentifikasi stunting, segera bawa anak ke tenaga medis untuk saran dan pemantauan lebih lanjut.")
            else:
                st.success(f'Hasil Prediksi: {hasil_label}')


