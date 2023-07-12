# from sqlalchemy import column
import streamlit as st
import plotly_express as px
import pandas as pd
# from torch import layout
from pathlib import Path
import plotly.graph_objects as go

# configuration
st.set_option('deprecation.showfileUploaderEncoding', False)

# Judul Halaman
st.set_page_config(
    page_title="Regresi Data ISPU K-ELM",
    page_icon="📈",
)

# title of the app
st.title("Website Visualisasi Prediksi dan Forecasting Data ISPU dengan metode Kernel-Extreme Learning Machine")

# Add a sidebar
st.sidebar.subheader("Pengaturan Visualisasi")

# Setup selecting csv file
file_select = st.sidebar.selectbox(
    label="Pilih file",
    options=[

                "Prediksi DKI1 PM10", "Prediksi DKI1 SO2", "Prediksi DKI1 CO", "Prediksi DKI1 O3", "Prediksi DKI1 NO2", "Forecasting DKI1",
                "Prediksi DKI2 PM10", "Prediksi DKI2 SO2", "Prediksi DKI2 CO", "Prediksi DKI2 O3", "Prediksi DKI2 NO2", "Forecasting DKI2",
                "Prediksi DKI3 PM10", "Prediksi DKI3 SO2", "Prediksi DKI3 CO", "Prediksi DKI3 O3", "Prediksi DKI3 NO2", "Forecasting DKI3",
                "Prediksi DKI4 PM10", "Prediksi DKI4 SO2", "Prediksi DKI4 CO", "Prediksi DKI4 O3", "Prediksi DKI4 NO2", "Forecasting DKI4",
                "Prediksi DKI5 PM10", "Prediksi DKI5 SO2", "Prediksi DKI5 CO", "Prediksi DKI5 O3", "Prediksi DKI5 NO2", "Forecasting DKI5",
            ]
)

global df

# Pemilihan file prediksi atau forecasting
if file_select == "Prediksi DKI1 PM10":
    uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-TA/Web/files/prediction/PM10.xlsx'
elif file_select == "Prediksi DKI1 SO2":
    uploaded_file = Path(__file__).parents[2] / ''
elif file_select == "Prediksi DKI1 CO":
    uploaded_file = Path(__file__).parents[2] / ''
elif file_select == "Prediksi DKI1 O3":
    uploaded_file = Path(__file__).parents[2] / ''
elif file_select == "Prediksi DKI1 NO2":
    uploaded_file = Path(__file__).parents[2] / ''
elif file_select == "Forecasting DKI1":
    uploaded_file = Path(__file__).parents[2] / ''
if file_select == "Prediksi DKI2 PM10":
    uploaded_file = Path(__file__).parents[2] / ''
elif file_select == "Prediksi DKI2 SO2":
    uploaded_file = Path(__file__).parents[2] / ''
elif file_select == "Prediksi DKI2 CO":
    uploaded_file = Path(__file__).parents[2] / ''
elif file_select == "Prediksi DKI2 O3":
    uploaded_file = Path(__file__).parents[2] / ''
elif file_select == "Prediksi DKI2 NO2":
    uploaded_file = Path(__file__).parents[2] / ''
elif file_select == "Forecasting DKI2":
    uploaded_file = Path(__file__).parents[2] / ''
if file_select == "Prediksi DKI3 PM10":
    uploaded_file = Path(__file__).parents[2] / ''
elif file_select == "Prediksi DKI3 SO2":
    uploaded_file = Path(__file__).parents[2] / ''
elif file_select == "Prediksi DKI3 CO":
    uploaded_file = Path(__file__).parents[2] / ''
elif file_select == "Prediksi DKI3 O3":
    uploaded_file = Path(__file__).parents[2] / ''
elif file_select == "Prediksi DKI3 NO2":
    uploaded_file = Path(__file__).parents[2] / ''
elif file_select == "Forecasting DKI3":
    uploaded_file = Path(__file__).parents[2] / ''
if file_select == "Prediksi DKI4 PM10":
    uploaded_file = Path(__file__).parents[2] / ''
elif file_select == "Prediksi DKI4 SO2":
    uploaded_file = Path(__file__).parents[2] / ''
elif file_select == "Prediksi DKI4 CO":
    uploaded_file = Path(__file__).parents[2] / ''
elif file_select == "Prediksi DKI4 O3":
    uploaded_file = Path(__file__).parents[2] / ''
elif file_select == "Prediksi DKI4 NO2":
    uploaded_file = Path(__file__).parents[2] / ''
elif file_select == "Forecasting DKI4":
    uploaded_file = Path(__file__).parents[2] / ''
if file_select == "Prediksi DKI5 PM10":
    uploaded_file = Path(__file__).parents[2] / ''
elif file_select == "Prediksi DKI5 SO2":
    uploaded_file = Path(__file__).parents[2] / ''
elif file_select == "Prediksi DKI5 CO":
    uploaded_file = Path(__file__).parents[2] / ''
elif file_select == "Prediksi DKI5 O3":
    uploaded_file = Path(__file__).parents[2] / ''
elif file_select == "Prediksi DKI5 NO2":
    uploaded_file = Path(__file__).parents[2] / ''
elif file_select == "Forecasting DKI5":
    uploaded_file = Path(__file__).parents[2] / ''

try:
    df = pd.read_excel(uploaded_file)
except Exception as e:
    print(e)
    df = pd.read_csv(uploaded_file)

global numeric_columns
global non_numeric_columns
st.write(df)
date_column = list(df.select_dtypes(['datetime']).columns)
numeric_columns = list(df.select_dtypes(['float', 'int']).columns)
non_numeric_columns = list(df.select_dtypes(['object']).columns)
non_numeric_columns.append(None)
# print(non_numeric_columns)

# Plotting
st.sidebar.subheader("Pengaturan Plot Garis")
# Select Box dan Multiselect untuk pemilihan data atau warna yang ingin ditampilkan
x_values = st.sidebar.selectbox('X axis', options=date_column)
y_values = st.sidebar.multiselect('Y axis', options=numeric_columns, default=numeric_columns)
color_values = st.sidebar.multiselect("Warna Kategori ISPU", options=["Baik", "Sedang", "Tidak Sehat", "Sangat Tidak Sehat", "Berbahaya"], default=["Baik", "Sedang"])
    
# Keterangan DKI1 sampai DKI5
st.sidebar.markdown(
    """
        Keterangan:\n
        DKI1 :  Bundaran HI, DKI Jakarta\n
        DKI2 :  Kelapa Gading, DKI Jakarta\n
        DKI3 :  Jagakarsa, DKI Jakarta\n
        DKI4 :  Lubang Buaya, DKI Jakarta\n
        DKI5 :  Kebon Jeruk, DKI Jakarta
    """
)

# Pembuatan Data Frame Area Kategori ISPU
area = {"Tanggal":df["Tanggal"], "Baik":50, "Sedang":50, "Tidak Sehat":100, "Sangat Tidak Sehat":100, "Berbahaya":100}
df_area = pd.DataFrame(area)

# Membuat skema warna 
color_sequence = ["#4FF04F", "#55A6E4", "#FFFF54", "#FF5454", "#322F2F"]
# membuat plot area ISPU
plot1 = px.area(data_frame=df_area, x="Tanggal", y=color_values, color_discrete_sequence=color_sequence)

# Membuat skema warna 
color_plot_sequence = ["#00007D", "#B30000", "#B2B200", "#008E00", "#510077"]
# membuat plot garis
plot2= px.line(data_frame=df, x=x_values, y=y_values, color_discrete_sequence=color_plot_sequence)

# Penggabungan Plot Line dan Area
plot3 = go.Figure(data=plot1.data + plot2.data)
st.plotly_chart(plot3)

data = [
    {
        'title': 'Baik',
        'subtitle': 'Tingkat kualitas udara yang sangat baik, tidak memberikan efek  negatif terhadap manusia, hewan, tumbuhan.',
        'action': 'Sangat baik melakukan kegiatan diluar.',
        'color': 'green'
    },
    {
        'title': 'Sedang',
        'subtitle': 'Tingkat kualitas udara masih dapat diterima pada kesehatan manusia, hewan dan tumbuhan.',
        'action': 'Kelompok sensitif: Kurangi aktivitas fisik yang terlalu lama atau berat.Setiap orang: Masih dapat beraktivitas di luar.',
        'color': 'blue'
    },
    {
        'title': 'Tidak Sehat',
        'subtitle': 'Tingkat kualitas udara yang bersifat merugikan paga manusia, hewan, dan tumbuhan.',
        'action': '',
        'color': 'orange'
    },
    {
        'title': 'Sangat Tidak Sehat',
        'subtitle': 'Tingkat kualitas udara yang dapat meningkatkan risiko kesehatan pada sejumlah segmen populasi yang terpapar.',
        'action': 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
        'color': 'red'
    },
    {
        'title': 'Berbahaya',
        'subtitle': 'Tingkat kualitas udara yang dapat merugikan kesehatan serius pada populasi dan perlu penanganan cepat.',
        'action': 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
        'color': 'black'
    }
]

col1, col2, col3, col4, col5 = st.columns(5)

for i, col in enumerate([col1, col2, col3, col4, col5]):
    with col:
        st.write(f'<h2 style="font-size: 20px; text-align:center; background-color:{data[i]["color"]}">{data[i]["title"]}</h2>', unsafe_allow_html=True)
        st.write(f'<h2 style="font-size: 12px; color:black; text-align:center; background-color:white;">{data[i]["subtitle"]}</h2>', unsafe_allow_html=True)
        st.write('<h2 style="font-size: 16px; font-weight:bold; color:black; text-align:center; background-color:white;">Apa yang harus dilakukan:</h2>', unsafe_allow_html=True)
        st.write(f'<h2 style="font-size: 12px; color:black; text-align:center; background-color:white;">{data[i]["action"]}</h2>', unsafe_allow_html=True)

