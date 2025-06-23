import streamlit as st
import pickle
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

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

# ==================== TITLE & HEADER ====================
st.title("🧠 Klasifikasi Teks")
st.write("Masukkan Judul Skripsi / Tesis, lalu sistem akan memprediksi label/kategorinya.")

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# ==================== CLEANING FUNCTION ====================
# Inisialisasi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return stemmer.stem(text)
    
# ==================== TEXT INPUT ====================
user_input = st.text_area("📝 Masukkan teks di sini:", height=150, placeholder="Contoh: Strategi Komunikasi Persu...")

# ==================== PREDICTION ====================
if st.button("🔍 Prediksi"):
    if user_input.strip() == "":
    st.warning("Teks tidak boleh kosong!")

