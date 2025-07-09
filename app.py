from flask import Flask, request, render_template
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import re
import string
import os

# --- Inisialisasi Aplikasi Flask ---
app = Flask(__name__)

# --- Fungsi untuk Membersihkan Teks ---
def preprocess_text(text):
    text = text.lower() # Ubah ke huruf kecil
    text = re.sub(r"\d+", "", text) # Hapus angka
    text = text.translate(str.maketrans('', '', string.punctuation)) # Hapus tanda baca
    text = text.strip() # Hapus spasi di awal/akhir
    return text

# --- Fungsi untuk Melatih dan Menyimpan Model ---
def train_and_save_model():
    # Muat dataset
    df = pd.read_csv('dataset_indonesiam_toxic.csv')

    # Gunakan kolom 'Tweet' dan 'Label'
    df['text_cleaned'] = df['Tweet'].apply(preprocess_text)
    X = df['text_cleaned']
    y = df['Label']

    # Buat pipeline yang menggabungkan vectorizer dan classifier
    model_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', MultinomialNB())
    ])

    # Latih model dengan seluruh data
    model_pipeline.fit(X, y)

    # Simpan model ke file
    joblib.dump(model_pipeline, 'sentiment_model_pipeline.pkl')
    print("Model telah dilatih dan disimpan sebagai sentiment_model_pipeline.pkl")
    return model_pipeline

# --- Rute untuk Halaman Utama ---
@app.route('/')
def home():
    return render_template('index.html')

# --- Rute untuk Melakukan Prediksi ---
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        
        # Lakukan prediksi menggunakan model yang dimuat
        prediction = model.predict([message])
        prediction_label = int(prediction[0])

        # Pemetaan label ke teks deskriptif
        label_map = {
            0: "Non-Toxic",
            1: "Weak Toxic",
            2: "Moderate Toxic",
            3: "Strong Toxic"
        }
        
        result = label_map.get(prediction_label, "Tidak Diketahui")

        return render_template('result.html', text=message, prediction=result)

# --- Main Program ---
if __name__ == '__main__':
    # Cek apakah file model sudah ada. Jika tidak, latih model terlebih dahulu.
    if not os.path.exists('sentiment_model_pipeline.pkl'):
        print("File model tidak ditemukan. Melatih model baru...")
        model = train_and_save_model()
    else:
        print("Memuat model yang sudah ada...")
        model = joblib.load('sentiment_model_pipeline.pkl')
    
    # Jalankan aplikasi Flask
    app.run(debug=True)