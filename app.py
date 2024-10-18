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

    # Konversi jenis kelamin menjadi 0 dan 1
    jenis_kelamin = 0 if jenis_kelamin == 'Laki-laki' else 1

    tinggi_badan = st.number_input('Tinggi Badan (cm)', min_value=0.0, max_value=150.0, value=80.0)
    berat_badan = st.number_input('Berat Badan (kg)', min_value=0.0, max_value=50.0, value=10.0)

    # Load model dan scaler
    model = joblib.load('model_stunting_multinomial.pkl')
    scaler = joblib.load('scaler_stunting.pkl')

    # Fungsi prediksi stunting
    def predict_stunting(umur, jenis_kelamin, tinggi_badan):
        input_data = np.array([[umur, jenis_kelamin, tinggi_badan]])  # Format input 2D
        input_data_scaled = scaler.transform(input_data)  # Transformasi data
        prediction = model.predict(input_data_scaled)  # Prediksi
        return prediction[0]  # Kembalikan hasil prediksi

    # Fungsi untuk konversi hasil prediksi menjadi label
    def map_hasil(category):
        mapping = {
            0: "Severity Stunting",
            1: "Stunting",
            2: "Normal",
            3: "Tinggi"
        }
        return mapping.get(category, "Unknown")

    # Fungsi validasi berat badan ideal berdasarkan umur dan jenis kelamin
    def berat_badan_ideal(umur, jenis_kelamin, berat_badan):
        if umur == 12:  # Usia 1 tahun
            if jenis_kelamin == 0:  # Laki-laki
                return 7.7 <= berat_badan <= 10.8
            else:  # Perempuan
                return 7.0 <= berat_badan <= 10.1
        elif umur == 24:  # Usia 2 tahun
            if jenis_kelamin == 0:  # Laki-laki
                return 9.7 <= berat_badan <= 13.6
            else:  # Perempuan
                return 9.0 <= berat_badan <= 13.0
        return True  # Default: anggap ideal jika tidak ada validasi khusus

    # Tombol untuk prediksi
    if st.button('Prediksi Kategori Stunting'):
        hasil = predict_stunting(umur, jenis_kelamin, tinggi_badan)
        hasil_label = map_hasil(hasil)

        # Tampilkan hasil prediksi
        if hasil in [0, 1]:  # Jika Severity Stunting atau Stunting
            st.markdown(
                f"<div style='color: red; font-weight: bold;'>Hasil Prediksi: {hasil_label}</div>",
                unsafe_allow_html=True
            )
            st.warning("Segera bawa anak ke tenaga medis untuk pemantauan lebih lanjut.")
            
            # Validasi berat badan ideal
            if not berat_badan_ideal(umur, jenis_kelamin, berat_badan):
                st.error("Berat badan anak kurang dari standar ideal.")
                st.subheader("Rekomendasi Makanan untuk Kekurangan Berat Badan:")
                st.markdown("""
                1. **Susu tinggi lemak** atau susu formula khusus untuk anak dengan berat badan rendah.
                2. **Nasi, kentang, pasta, dan roti gandum** yang menyediakan energi dan kalori.
                3. **Alpukat, kacang-kacangan, minyak zaitun, dan minyak kelapa** sebagai sumber energi padat.
                4. **Daging, ikan, ayam, tempe, dan tahu**. Protein penting untuk menambah massa tubuh dan otot.
                5. **Buah seperti pisang, mangga, dan kurma**, memberikan energi serta serat dan vitamin.
                6. **Roti dengan selai kacang, smoothie dengan susu, dan granola** yang mengandung campuran lemak, protein, dan karbohidrat.
                """)

        else:  # Untuk kategori Normal atau Tinggi
            st.success(f'Hasil Prediksi: {hasil_label}')

    # SHAP Interpretasi
    st.subheader('Interpretasi SHAP')
    st.write('Visualisasi ini membantu memahami fitur mana yang berpengaruh pada prediksi model.')

    # Placeholder untuk plot SHAP
    st.pyplot()  # Simulasi plot SHAP
