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
    tinggi_badan = st.number_input('Tinggi Badan (cm)', min_value=0.0, max_value=150.0, value=80.0)

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
