"""Prediksi playlist musik dari tempo, energy, danceability."""

import os
import sys

import joblib
import numpy as np

MODEL_FILE = "model_jst_musik.pkl"
TEMPO_MIN, TEMPO_MAX = 60.0, 150.0


def normalisasi_fitur(tempo, energy, danceability):
    """
    Menyiapkan satu baris fitur input untuk model JST.

    Logika:
    - Tempo dinormalisasi min-max ke [0, 1] dengan rentang yang sama saat training.
    - Energy dan danceability sudah 0–1, dipakai langsung.
    - Hasil dibentuk matrix 1 x 3 karena model menerima batch input.
    """
    tempo_norm = (tempo - TEMPO_MIN) / (TEMPO_MAX - TEMPO_MIN)
    return np.array([[tempo_norm, energy, danceability]])


def prediksi_playlist(model, tempo, energy, danceability, threshold=0.5):
    """
    Menghitung probabilitas dan menentukan rekomendasi playlist.

    Logika:
    - Fitur dinormalisasi lalu diteruskan ke model yang sudah dilatih.
    - predict_proba mengembalikan probabilitas tiap kelas; indeks [1] = kelas Semangat.
    - Jika prob > threshold (default 0.5) → Olahraga/Semangat, else → Santai/Fokus.
    """
    fitur = normalisasi_fitur(tempo, energy, danceability)
    prob = float(model.predict_proba(fitur)[0][1])
    playlist = "Olahraga / Semangat" if prob > threshold else "Santai / Fokus"
    return prob, playlist


if __name__ == "__main__":
    if not os.path.exists(MODEL_FILE):
        print(f"'{MODEL_FILE}' belum ada. Jalankan dulu: python3 train_model.py")
        sys.exit(1)

    model = joblib.load(MODEL_FILE)

    print("JST Rekomendasi Playlist Musik\n")
    tempo = float(input("Tempo BPM (60-150): ") or "120")
    energy = float(input("Energy (0-1): ") or "0.75")
    dance = float(input("Danceability (0-1): ") or "0.65")

    prob, playlist = prediksi_playlist(model, tempo, energy, dance)

    print(f"\nProbabilitas semangat: {prob:.4f}")
    print(f"Playlist: {playlist}")
