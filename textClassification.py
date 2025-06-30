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

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, vectorizer, le

model, vectorizer, le = load_model()

# ==================== CLEANING + STEMMING FUNCTION ====================
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stemming_cache = {}

def clean_and_stem(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    stemmed_words = []
    for word in words:
        if word in stemming_cache:
            stemmed_words.append(stemming_cache[word])
        else:
            stemmed = stemmer.stem(word)
            stemming_cache[word] = stemmed
            stemmed_words.append(stemmed)
    return ' '.join(stemmed_words)

# ==================== TEXT INPUT ====================
user_input = st.text_area("📝 Masukkan judul skripsi:", height=150, placeholder="Contoh: Strategi Komunikasi Persuasif Dalam ...")

# ==================== PREDICTION ====================
if st.button("🔍 Prediksi"):
    if user_input.strip() == "":
        st.warning("⚠️ Silakan masukkan teks terlebih dahulu.")
    else:
        cleaned_text = clean_and_stem(user_input)
        vectorized_input = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_input)[0]
        label_output = le.inverse_transform([prediction])[0]
        st.success(f"🎯 Prediksi Kelas: **{label_output}**")
