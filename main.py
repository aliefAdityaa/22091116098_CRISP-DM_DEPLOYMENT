import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import altair as alt
import seaborn as sns
import pickle
import os

@st.cache_data
def load_data():
    df = pd.read_csv("datmin/TravelInsurancePrediction.csv")
    return df

# Fungsi untuk menampilkan halaman utama
def show_about():
    st.title("TravelInsurance Dataset")
    st.write("""
        Welcome to the homepage!
        
        Use the sidebar to navigate to different pages.
    """)
    # Path ke file CSV
    file_path = 'datmin/Data Cleaned.csv'

    # Periksa keberadaan file
    if not os.path.exists(file_path):
        st.error(f"File '{file_path}' tidak ditemukan. Pastikan file berada di lokasi yang benar.")
        st.stop()

    # Muat data CSV
    try:
        df = pd.read_csv(file_path)
        st.write(df)  # Tampilkan data jika berhasil dimuat
    except Exception as e:
        st.error(f"Gagal memuat file CSV: {e}")

# Fungsi untuk menampilkan halaman tentang
def show_Distribusi(df):
# Judul dan deskripsi
    st.title("Distribusi Nilai")
    # st.write("Menu ini menampilkan distribusi dari nilai lead time, stays in weekend nights, dan stays in week nights dalam dataset Hotel Booking Demand.")

    st.title("Visualisasi Data Mining menggunakan Streamlit")

    st.subheader("Distribusi Tipe Umur")
    plt.figure(figsize=(10, 5))
    fig, ax = plt.subplots()
    sns.countplot(x='Age', data=df, palette=['darkturquoise', 'royalblue'])
    plt.title('Distribusi Usia Pelanggan')
    plt.xlabel('Usia')
    plt.ylabel('Frekuensi')
    st.pyplot(fig)

    # # Buat plot pie
    # fig, ax = plt.subplots()
    # s = df['is_canceled'].value_counts()
    # s.plot(kind='pie', autopct='%1.1f%%', startangle=360, labels=['Not Canceled', 'Canceled'], ax=ax)
    # #ax.set_ylabel('')  Hilangkan label sumbu y
    # ax.legend()
    # # Tampilkan plot di Streamlit
    # st.pyplot(fig)

    # # Plot bar chart
    # fig, ax = plt.subplots(figsize=(12,6))
    # sns.barplot(x='arrival_date_year', y='lead_time',hue='is_canceled', data= df, palette='vlag', ax=ax)
    # plt.title('Arriving year, Lead time and Cancelations')
    # st.pyplot(fig)

def show_hubungan(df):
    st.title("Hubungan Nilai")
    # st.write("Menu ini menampilkan hubungan antara nilai math, reading, dan writing dalam dataset Hotel Bookings.")
    plot_option = st.selectbox("Pilih Plot:", ["Age vs Annual Income", "Family Members vs Chronic Diseases", "Frequent Flyer vs Travel Insurance"])

    if plot_option == "Age vs Annual Income":
        st.subheader("Scatter Plot: Age vs Annual Income")
        fig, ax = plt.subplots()
        sns.scatterplot(x='Age', y='AnnualIncome', data=df)
        ax.set_xlabel("Age")
        ax.set_ylabel("Annual Income")
        st.pyplot(fig)

    elif plot_option == "Family Members vs Chronic Diseases":
        st.subheader("Bar Plot: Family Members vs Chronic Diseases")
        sns.barplot(x='FamilyMembers', y='ChronicDiseases', data=df)
        st.pyplot()

    elif plot_option == "Frequent Flyer vs Travel Insurance":
        st.subheader("Count Plot: Frequent Flyer vs Travel Insurance")
        sns.countplot(x='FrequentFlyer', hue='TravelInsurance', data=df)
        st.pyplot()

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Korelasi")
    df_file_corr = df.corr(numeric_only=True)

    # Ambil hanya 10 kolom dan baris pertama
    # df_file_corr_subset = df_file_corr.iloc[:10, :10]

    # Buat heatmap menggunakan seaborn
    fig, ax = plt.subplots(figsize=(30, 30))
    sns.heatmap(df_file_corr, annot=True, cmap='vlag', ax=ax)
    plt.title('Nilai Korelasi')
    plt.show()

    # Tampilkan heatmap menggunakan Streamlit
    st.pyplot(fig)

    # Tambahkan teks penjelasan
    # text = 'Tabel korelasi diatas menunjukkan bahwa terdapat hubungan yang signifikan antara beberapa variabel. Perusahaan dapat menggunakan informasi ini untuk membuat keputusan yang lebih baik tentang produk/layanan perusahaan, strategi marketing, dan program loyalitas pelanggan.'
    # st.markdown(text)
    

# Memuat data
# df = load_data()

# Menampilkan hubungan
# show_relationship(df)


# Fungsi untuk menampilkan halaman perbandingan
def show_Perbandingan(df):
    st.title("Perbandingan")
    st.write("""
        Travel insurance
    """)
    # Membuat subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot pertama: market_segment
    sns.countplot(x='Employment Type', data=df, palette='rocket', ax=axes[0, 0])
    axes[0, 0].set_title('Jenis Jenis Tipe Pekerjaan', fontweight="bold", size=10)

    # Plot kedua: distribution_channel
    sns.countplot(data=df, x='GraduateOrNot', palette='Set1_r', ax=axes[0, 1])
    axes[0, 1].set_title('Lulusan atau Tidak', fontweight="bold", size=20)

    # Plot ketiga: is_repeated_guest
    sns.countplot(data=df, x='FrequentFlyer', ax=axes[1, 0]).set_title('Tamu yang sering terbang', fontsize=20)

    # Plot keempat: customer_type
    sns.countplot(x='TravelInsurance', data=df, ax=axes[1, 1]).set_title('Jenis Asuransi')

    # Menampilkan plot di Streamlit
    st.pyplot(fig)

# Fungsi untuk menampilkan halaman komposisi
def show_Komposisi(df):
    st.title("Komposisi")
    st.write("""
        Travel insurance
    """)

        # Buat pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(df.groupby(by=["FamilyMembers"]).size(), labels=df["FamilyMembers"].unique(), autopct="%0.2f")
    ax.set_title('Komposisi Berdasarkan Jumlah Anggota Keluarga')
    st.pyplot(fig)
# Load model
# with open('model.pkl', 'rb') as file:
#     loaded_model = pickle.load(file)

def predict_cancellation(df):
    st.title("Visualisasi Data Mining menggunakan Streamlit")
    st.title("TravelInsurance")

    # Select features
    feature_columns = [
        'Unnamed: 0', 'Age', 'AnnualIncome', 'FamilyMembers',
        'ChronicDiseases', 'TravelInsurance'
    ]

    selected_features = {}
    for feature in feature_columns:
        selected_features[feature] = st.selectbox(f"{feature.replace('_', ' ').title()}", sorted(df[feature].unique()))

    data = pd.DataFrame(selected_features, index=[0])

    # Button for prediction
    button = st.button('Predict')
    if button:
        with open('datmin/model.pkl', 'rb') as file:
            loaded_model = pickle.load(file)

        # Predict using model
        predicted = loaded_model.predict(data)

        # Display prediction
        if predicted[0] == 0:
            st.write('Dapat Asuransi')
        else:
            st.write('Tidak dapat')




# Memuat data
df = load_data()

# Mengatur sidebar
df2 = pd.read_csv('datmin/Data Cleaned.csv')
nav_options = {
    "About": show_about,
    "Distribution": lambda: show_Distribusi(df),
    "Relations": lambda: show_hubungan(df),
    "Perbandingan": lambda: show_Perbandingan(df),
    "Komposisi": lambda: show_Komposisi(df),
    "Prediction": lambda: predict_cancellation(df2)
}

# Menampilkan sidebar
st.sidebar.title("TravelInsurance")
selected_page = st.sidebar.radio("Menu", list(nav_options.keys()))

# Menampilkan halaman yang dipilih
nav_options[selected_page]()
