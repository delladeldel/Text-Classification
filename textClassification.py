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
    response.raise_for_status()
    return pickle.load(BytesIO(response.content))

# ==================== MASUKKAN LINKMU DI SINI ====================
MODEL_URL = "https://huggingface.co/delsdell/text_classification/resolve/main/model.pkl"
VECTORIZER_URL = "https://huggingface.co/delsdell/text_classification/resolve/main/vectorizer.pkl"
SVD_URL = "https://huggingface.co/delsdell/text_classification/resolve/main/svd.pkl"

# ==================== LOAD ALL FILES ====================
with st.spinner("📦 Memuat model dan data..."):
    model = load_pickle_from_url(MODEL_URL)
    vectorizer = load_pickle_from_url(VECTORIZER_URL)
    svd = load_pickle_from_url(SVD_URL)

# ==================== TEXT INPUT ====================
user_input = st.text_area("📝 Masukkan teks di sini:", height=150, placeholder="Contoh: Analisis Strategi Pemasaran Digital...")

# ==================== PREDIKSI ====================
if st.button("🔍 Prediksi"):
    if user_input.strip() == "":
        st.warning("Silakan masukkan teks terlebih dahulu.")
    else:
        with st.spinner("🔍 Memprediksi..."):
            cleaned = clean_text(user_input)
            vectorized = vectorizer.transform([cleaned])
            reduced = svd.transform(vectorized)
            prediction = model.predict(reduced)[0]
        st.success(f"📌 Hasil Prediksi: **{prediction}**")
