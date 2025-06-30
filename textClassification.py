import streamlit as st
import pickle
import re

# ==================== SETUP ====================
st.set_page_config(
    page_title="Klasifikasi Teks",
    page_icon="🧠",
    layout="centered"
)

# ==================== STYLE ====================
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        padding: 0.5em 1em;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== TITLE ====================
st.title("🧠 Klasifikasi Teks")
st.write("Masukkan Judul Skripsi / Tesis, lalu sistem akan memprediksi label/kategorinya.")

# ==================== LOAD MODEL & VECTORIZER ====================
@st.cache_resource
def load_resources():
    with open('model.pkl', 'rb') as f_model:
        model = pickle.load(f_model)
    with open('vectorizer.pkl', 'rb') as f_vec:
        vectorizer = pickle.load(f_vec)
    return model, vectorizer

model, vectorizer = load_resources()

# ==================== CLEANING FUNCTION ====================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)  # hanya huruf dan spasi
    text = re.sub(r'\s+', ' ', text).strip()  # hapus spasi berlebih
    return text

# ==================== TEXT INPUT ====================
user_input = st.text_area("📝 Masukkan teks di sini:", height=150, placeholder="Contoh: Analisis Strategi Pemasaran Digital...")

# ==================== PREDICTION ====================
if st.button("🔍 Prediksi"):
    if user_input.strip() == "":
        st.warning("Silakan masukkan teks terlebih dahulu.")
    else:
        cleaned = clean_text(user_input)
        try:
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            st.success(f"📌 Hasil Prediksi: **{prediction}**")
        except ValueError as e:
            st.error(f"Terjadi kesalahan dimensi fitur.\n\nDetail: {e}")
            st.info("Pastikan model dan vectorizer dilatih dari data yang sama.")
