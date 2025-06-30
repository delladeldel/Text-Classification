import streamlit as st
import pickle
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.preprocessing import LabelEncoder

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

# ==================== LOAD MODEL DAN OBJEK LAIN ====================
@st.cache_resource
def load_all():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('svd.pkl', 'rb') as f:
        svd = pickle.load(f)

    # Inisialisasi LabelEncoder dengan urutan label yang sama seperti saat training
    label_list = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    le = LabelEncoder()
    le.fit(label_list)

    return model, vectorizer, svd, le

model, vectorizer, svd, le = load_all()

# ==================== CLEANING FUNCTION ====================
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return stemmer.stem(text)

def prediksi_judul(judul_baru):
    judul_bersih = clean_text(judul_baru)
    judul_vector = vectorizer.transform([judul_bersih])
    judul_vector_svd = svd.transform(judul_vector)
    pred_encoded = model.predict(judul_vector_svd)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]
    return pred_label

# ==================== TEXT INPUT ====================
user_input = st.text_area("📝 Masukkan teks di sini:", height=150, placeholder="Contoh: Strategi Komunikasi Persu...")

# ==================== PREDICTION ====================
if st.button("🔍 Prediksi"):
    if user_input.strip() == "":
        st.warning("Teks tidak boleh kosong!")
    else:
        hasil_prediksi = prediksi_judul(user_input)
        st.success(f"✅ Prediksi Label: **{hasil_prediksi}**")
