import streamlit as st
import re
import pickle
import requests
from io import BytesIO

# ==================== SETUP ====================
st.set_page_config(
    page_title="Klasifikasi Teks",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Klasifikasi Teks")
st.write("Prediksi kategori dari judul skripsi/tesis menggunakan model Logistic Regression.")

# ==================== CLEANING FUNCTION ====================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==================== LOAD PICKLE FROM GITHUB (model & vectorizer) ====================
@st.cache_resource
def load_pickle_from_github(raw_url):
    response = requests.get(raw_url)
    return pickle.load(BytesIO(response.content))

# ==================== LOAD PICKLE FROM GOOGLE DRIVE (SVD) ====================
@st.cache_resource
def load_pickle_from_gdrive(1ivgD2MQX9wbSYq133rihIvfUQpvSmO5F):
    gdrive_url = f"https://drive.google.com/file/d/1ivgD2MQX9wbSYq133rihIvfUQpvSmO5F/view?usp=sharing"
    response = requests.get(gdrive_url)
    return pickle.load(BytesIO(response.content))

# ====== GANTI DENGAN LINK RAW GITHUB DAN ID GOOGLE DRIVE ======

# 👉 Ganti dengan link RAW GitHub (bukan halaman browser GitHub)
MODEL_URL = "https://github.com/delladeldel/Text-Classification/blob/e911b3a4ca92fac2b9aaf86ea0f5221618189e6b/model.pkl"
VECTORIZER_URL = "https://github.com/delladeldel/Text-Classification/blob/e911b3a4ca92fac2b9aaf86ea0f5221618189e6b/vectorizer.pkl"

# 👉 Ganti dengan Google Drive file ID untuk svd.pkl
SVD_ID = "1ivgD2MQX9wbSYq133rihIvfUQpvSmO5F"

# ==================== LOAD ALL FILES ====================
model = load_pickle_from_github(MODEL_URL)
vectorizer = load_pickle_from_github(VECTORIZER_URL)
svd = load_pickle_from_gdrive(SVD_ID)

# ==================== INPUT USER ====================
user_input = st.text_area("📝 Masukkan teks di sini:", height=150, placeholder="Contoh: Analisis Strategi Pemasaran Digital...")

# ==================== PREDIKSI ====================
if st.button("🔍 Prediksi"):
    if user_input.strip() == "":
        st.warning("Silakan masukkan teks terlebih dahulu.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        reduced = svd.transform(vectorized)
        prediction = model.predict(reduced)[0]
        st.success(f"📌 Hasil Prediksi: **{prediction}**")
