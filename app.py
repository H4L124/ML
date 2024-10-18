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
                             'Prediksi Stunting', 
                             'Deteksi Stunting Standar WHO'])

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
# Halaman baru: Prediksi Stunting Menurut Indikator
if page == 'Prediksi Stunting':
    st.header('Prediksi Stunting Menurut Indikator')

    # Input form untuk memasukkan data
    st.subheader('Input Data')
     # Input form for indicator variables
    penambah_darah = st.number_input('Jumlah Pil Penambah Darah yang Dikonsumsi Selama Mengandung', min_value=0, max_value=1000, value=0)
    bb_lahir = st.number_input('Berat Badan Lahir (kg)', min_value=0.0, max_value=10.0, value=0.0)
    bb = st.number_input('Berat Badan Lahir (kg)', min_value=0.0, max_value=100.0, value=0.0)
    tb = st.number_input('Berat Badan Lahir (kg)', min_value=0.0, max_value=200.0, value=0.0)
    umur = st.number_input('Umur (bulan)', min_value=0, max_value=60, value=24)
    akses_ventilasi = st.selectbox('Akses Ventilasi Rumah', ('Memadahi', 'Tidak Memadahi'))
    kehidupan_rt = st.selectbox('Bagaimana Kondisi Perekonomian Orang Tua?', ('Kurang Mencukupi', 'Mencukupi Kebutuhan Primer', 'Lebih Dari Cukup', 'Kurang Tahu'))
    makan_anak = st.selectbox('Bagaimana Kecukupan untuk Konsumsi Anak?', ('Kurang Mencukupi', 'Cukup', 'Lebih Dari Cukup', 'Kurang Tahu'))
    kesehatan_anak = st.selectbox('Bagaimana Akses untuk Perawatan Kesehatan Anak?', ('Kurang Mencukupi', 'Cukup ', 'Lebih Dari Cukup', 'Kurang Tahu'))
    jenis_kelamin = st.selectbox('Jenis Kelamin', ('Perempuan', 'Laki-laki'))

    # Konversi menjadi 0 dan 1
    akses_ventilasi = 1 if akses_ventilasi == 'Memadahi' else 3
    kehidupan_rt_mapping = {
        'Kurang Mencukupi': 1,
        'Mencukupi Kebutuhan Primer': 2,
        'Lebih Dari Cukup': 3,
        'Kurang Tahu': 8
    }
    kehidupan_rt = kehidupan_rt_mapping[kehidupan_rt]

    makan_anak_mapping = {
        'Kurang Mencukupi': 1,
        'Cukup': 2,
        'Lebih Dari Cukup': 3,
        'Kurang Tahu': 8
    }
    makan_anak = makan_anak_mapping[makan_anak]

    kesehatan_anak_mapping = {
        'Kurang Mencukupi': 1,
        'Cukup': 2,
        'Lebih Dari Cukup': 3,
        'Kurang Tahu': 8
    }
    kesehatan_anak = kesehatan_anak_mapping[kesehatan_anak]
    jenis_kelamin = 1 if jenis_kelamin == 'Laki-laki' else 3
    # Load model dan scaler dengan error handling
    try:
        model = joblib.load('model_faktorstunting_multinomial.pkl')
        scaler = joblib.load('scaler_faktorstunting.pkl')
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
    if st.button('Prediksi Stunting'):
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

# Halaman 3: Prediksi Stunting pada Balita
if page == 'Deteksi Stunting Standar WHO':
    st.header('Prediksi Stunting pada Balita')
    st.write('Deteksi stunting dilakukan berdasarkan indikator HAZ/Height-for-Age Z-score dari standar WHO. WHO telah menetapkan populasi referensi internasional yang digunakan sebagai standar universal untuk menilai pertumbuhan anak-anak di seluruh dunia, termasuk untuk penghitungan Z-score. Standar WHO digunakan untuk mengevaluasi apakah pertumbuhan seorang anak sesuai dengan potensi pertumbuhan biologis optimal, terlepas dari lokasi geografis.')

    # Input form untuk memasukkan data
    st.subheader('Input Data')
    umur = st.number_input('Umur (bulan)', min_value=0, max_value=60, value=24)
    jenis_kelamin = st.selectbox('Jenis Kelamin', ('Perempuan', 'Laki-laki'))
    
    # Konversi menjadi 0 dan 1
    jenis_kelamin = 0 jika jenis_kelamin == 'Laki-laki' else 1

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

            # Setelah prediksi, input data tambahan terkait berat dan tinggi
            tinggi_badan = st.number_input("Masukkan tinggi badan anak (cm):", min_value=0.0)
            berat_badan = st.number_input("Masukkan berat badan anak (kg):", min_value=0.0)
            jenis_kelamin = st.selectbox("Pilih jenis kelamin anak:", ['Laki-laki', 'Perempuan'])
            usia = st.selectbox("Pilih usia anak (tahun):", [1, 2, 3, 4, 5])

            # Standar tinggi badan berdasarkan usia dan jenis kelamin
            standar_tinggi = {
                'Laki-laki': {1: (71, 82.9), 2: (81.7, 96.3), 3: (88.7, 107.2), 4: (94.9, 115.9), 5: (100.7, 123.9)},
                'Perempuan': {1: (68.9, 81.7), 2: (80, 96.1), 3: (87.4, 106.5), 4: (94.1, 115.7), 5: (99.9, 123.7)}
            }

            # Mengecek apakah tinggi badan anak sesuai dengan standar
            standar_min_tinggi, standar_max_tinggi = standar_tinggi[jenis_kelamin][usia]
            if not (standar_min_tinggi <= tinggi_badan <= standar_max_tinggi):
                st.error(f"Tinggi badan anak Anda tidak sesuai standar untuk usia {usia} tahun.")
                st.warning("Rekomendasi Makanan untuk Kekurangan Tinggi Badan:")
                st.markdown("""
                1. Susu dan produk olahan susu seperti yogurt, keju, susu full cream
                2. Daging ayam
                3. Kacang-Kacangan dan Biji-Bijian
                4. Sayuran Hijau Gelap seperti bayam dan brokoli
                """)
            else:
                st.success(f"Tinggi badan anak Anda sesuai dengan standar untuk usia {usia} tahun.")
