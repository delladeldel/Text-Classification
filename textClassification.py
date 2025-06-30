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

# ==================== LOAD PICKLE FROM URL ====================
@st.cache_resource
def load_pickle_from_url(url):
    response = requests.get(url)
    return pickle.load(BytesIO(response.content))

# ==================== GANTI URL BERIKUT SESUAI FILE-MU ====================

MODEL_URL = "https://raw.githubusercontent.com/delladeldel/Text-Classification/main/model.pkl"
VECTORIZER_URL = "https://raw.githubusercontent.com/delladeldel/Text-Classification/main/vectorizer.pkl"

# Contoh: SVD dari Hugging Face atau Dropbox dengan direct download
SVD_URL = "https://huggingface.co/datasets/namakamu/namamodel/resolve/main/svd.pkl"
# atau dari Dropbox: "https://www.dropbox.com/s/abc123xyz/svd.pkl?dl=1"

# ==================== LOAD ALL FILES ====================
model = load_pickle_from_url(MODEL_URL)
vectorizer = load_pickle_from_url(VECTORIZER_URL)
svd = load_pickle_from_url(SVD_URL)

# ==================== INPUT ====================
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
