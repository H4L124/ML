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
# Halaman baru: Prediksi Stunting Menurut Indikator
if page == 'Prediksi Stunting':
    st.header('Prediksi Stunting Menurut Indikator')

    # Input form untuk memasukkan data
    st.subheader('Input Data')
     # Input form for indicator variables
    penambah_darah = st.number_input('Jumlah Pil Penambah Darah yang Dikonsumsi Ibu Selama Mengandung', min_value=0, max_value=1000, value=0)
    bb_lahir = st.number_input('Berat Badan Lahir (kg)', min_value=0.0, max_value=10.0, value=0.0)
    bb = st.number_input('Berat Badan Sekarang (kg)', min_value=0.0, max_value=100.0, value=0.0)
    tb = st.number_input('Tinggi Badan (cm)', min_value=0.0, max_value=200.0, value=0.0)
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
    dummy_feature = 0
    # Load model dan scaler dengan error handling
    try:
        model = joblib.load('model_faktorstunting_multinomial.pkl')
        scaler = joblib.load('scaler_faktorstunting.pkl')
    except FileNotFoundError:
        st.error("Model atau scaler tidak ditemukan. Pastikan file sudah benar.")
        st.stop()

    # Fungsi untuk prediksi kategori stunting
    def predict_stunting(numerical_data, categorical_data):
        numerical_data_scaled = scaler.transform([numerical_data])  # Transformasi hanya data numerik

    # Gabungkan data numerik yang telah diskalakan dengan data kategorik
        input_data_combined = np.concatenate([numerical_data_scaled[0], categorical_data])
        input_data_combined = np.append(input_data_combined, dummy_feature) 

    # Prediksi menggunakan model
        prediction = model.predict([input_data_combined])  # Prediksi
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
    if tb <= 0:
        st.error("Tinggi badan harus lebih dari 0 cm.")
    else:
        # Pisahkan variabel numerik dan kategorik
        numerical_data = [penambah_darah, bb_lahir, bb, tb, umur]  # Data numerik
        categorical_data = np.array([akses_ventilasi, kehidupan_rt, makan_anak, kesehatan_anak, jenis_kelamin])  # Data kategorik
        
        # Melakukan prediksi
        hasil = predict_stunting(numerical_data, categorical_data)
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

# Peringatan untuk stunting
        if hasil in [0, 1]:  # Jika Severity Stunting atau Stunting
            
            # Rekomendasi makanan berdasarkan berat badan dan usia
            def rekomendasi_berat_badan(umur, bb, jenis_kelamin):
                # Aturan berat badan
                batas_berat = {
                    (1, 12): (7.0, 12.0) if jenis_kelamin == 3 else (7.7, 12.0),  # Anak perempuan: 7.0 - 11.5 kg; Laki-laki: 7.7 - 12.0 kg
                    (13, 24): (9.0, 14.8) if jenis_kelamin == 3 else (9.7, 15.3),  # Anak perempuan: 9.0 - 14.8 kg; Laki-laki: 9.7 - 15.3 kg
                    (25, 36): (10.8, 18.1) if jenis_kelamin == 3 else (11.3, 18.3),  # Anak perempuan: 10.8 - 18.1 kg; Laki-laki: 11.3 - 18.3 kg
                    (37, 48): (12.3, 21.5) if jenis_kelamin == 3 else (12.7, 21.2),  # Anak perempuan: 12.3 - 21.5 kg; Laki-laki: 12.7 - 21.2 kg
                    (49, 60): (13.7, 24.9) if jenis_kelamin == 3 else (14.1, 21.2)   # Anak perempuan: 13.7 - 24.9 kg; Laki-laki: 14.1 - 21.2 kg
                }

                # Cek batas berat badan berdasarkan umur dan jenis kelamin
                for age_range, (min_weight, max_weight) in batas_berat.items():
                    if umur in range(age_range[0], age_range[1] + 1):
                        if bb < min_weight:
                            return True  # Berat badan kurang dari minimum
                        else:
                            return False  # Berat badan sudah sesuai

                return False  # Jika tidak ada rentang umur yang cocok

            # Cek rekomendasi makanan hanya jika anak teridentifikasi stunting
            if rekomendasi_berat_badan(umur, bb, jenis_kelamin):
                st.warning("Berat badan anak kurang dari nilai minimum. Berikut rekomendasi makanan untuk kekurangan berat badan:")
                st.markdown("""
                1. Susu tinggi lemak atau susu formula khusus untuk anak dengan berat badan rendah.
                2. Nasi, kentang, pasta, dan roti gandum yang menyediakan energi dan kalori.
                3. Alpukat, kacang-kacangan, minyak zaitun, dan minyak kelapa sebagai sumber energi padat.
                4. Daging, ikan, ayam, tempe, dan tahu. Protein penting untuk menambah massa tubuh dan otot.
                5. Buah-buahan seperti pisang, mangga, dan kurma, memberikan energi sekaligus serat dan vitamin.
                6. Roti dengan selai kacang, smoothie dengan susu, dan granola yang mengandung campuran lemak, protein, dan karbohidrat.
                """)
# Rekomendasi makanan berdasarkan tinggi badan dan usia
            def rekomendasi_tinggi_badan(umur, tb, jenis_kelamin):
                # Aturan tinggi badan
                batas_tinggi = {
                    (1, 12): (68.9, 81.7) if jenis_kelamin == 3 else (71, 82.9),  # Anak perempuan: 68.9 - 81.7 cm; Laki-laki: 71 - 82.9 cm
                    (13, 24): (80, 96.1) if jenis_kelamin == 3 else (81.7, 96.3),  # Anak perempuan: 80 - 96.1 cm; Laki-laki: 81.7 - 96.3 cm
                    (25, 36): (87.4, 106.5) if jenis_kelamin == 3 else (88.7, 107.2),  # Anak perempuan: 87.4 - 106.5 cm; Laki-laki: 88.7 - 107.2 cm
                    (37, 48): (94.1, 115.7) if jenis_kelamin == 3 else (94.9, 115.9),  # Anak perempuan: 94.1 - 115.7 cm; Laki-laki: 94.9 - 115.9 cm
                    (49, 60): (99.9, 123.7) if jenis_kelamin == 3 else (100.7, 123.9)  # Anak perempuan: 99.9 - 123.7 cm; Laki-laki: 100.7 - 123.9 cm
                }

                # Cek batas tinggi badan berdasarkan umur dan jenis kelamin
                for age_range, (min_height, max_height) in batas_tinggi.items():
                    if umur in range(age_range[0], age_range[1] + 1):
                        if tb < min_height:
                            return True  # Tinggi badan kurang dari minimum
                        else:
                            return False  # Tinggi badan sudah sesuai

                return False  # Jika tidak ada rentang umur yang cocok

            # Cek rekomendasi makanan hanya jika tinggi badan tidak memenuhi batas minimum
            if rekomendasi_tinggi_badan(umur, tb, jenis_kelamin):
                st.warning("Tinggi badan anak kurang dari nilai minimum. Berikut rekomendasi makanan untuk kekurangan tinggi badan:")
                st.markdown("""
                **Rekomendasi Makanan untuk Kekurangan Tinggi Badan**
                1. Susu dan produk olahan susu seperti yogurt, keju, dan susu full cream.
                2. Daging ayam yang kaya akan protein.
                3. Kacang-kacangan dan biji-bijian untuk membantu pertumbuhan.
                4. Sayuran hijau gelap seperti bayam dan brokoli yang kaya nutrisi.
                """)

        elif hasil == 2:  # Normal
            st.markdown(
                f"<div style='color: green; font-weight: bold;'>Hasil Prediksi: {hasil_label}</div>",
                unsafe_allow_html=True
            )
            st.success("Anak dalam kategori normal, pertahankan pola makan sehat dan pemantauan rutin.")
        
        elif hasil == 3:  # Tinggi
            st.markdown(
                f"<div style='color: blue; font-weight: bold;'>Hasil Prediksi: {hasil_label}</div>",
                unsafe_allow_html=True
            )
            st.success("Anak berada dalam kategori tinggi, terus jaga kesehatan dan pola makan yang baik.")

if page == 'Deteksi Stunting Standar WHO':
    st.header('Prediksi Stunting pada Balita')
    st.write('Deteksi stunting dilakukan berdasarkan indikator HAZ/Height-for-Age Z-score dari standar WHO. WHO telah menetapkan populasi referensi internasional yang digunakan sebagai standar universal untuk menilai pertumbuhan anak-anak di seluruh dunia, termasuk untuk penghitungan Z-score. Standar WHO digunakan untuk mengevaluasi apakah pertumbuhan seorang anak sesuai dengan potensi pertumbuhan biologis optimal, terlepas dari lokasi geografis.')

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
