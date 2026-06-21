"""Prediksi playlist musik dari tempo, energy, danceability."""

import os
import sys

import joblib
import numpy as np

MODEL_FILE = "model_jst_musik.pkl"
TEMPO_MIN, TEMPO_MAX = 60.0, 150.0

if not os.path.exists(MODEL_FILE):
    print(f"'{MODEL_FILE}' belum ada. Jalankan dulu: python3 train_model.py")
    sys.exit(1)

model = joblib.load(MODEL_FILE)

print("JST Rekomendasi Playlist Musik\n")
tempo = float(input("Tempo BPM (60-150): ") or "120")
energy = float(input("Energy (0-1): ") or "0.75")
dance = float(input("Danceability (0-1): ") or "0.65")

fitur = np.array([[(tempo - TEMPO_MIN) / (TEMPO_MAX - TEMPO_MIN), energy, dance]])
prob = float(model.predict_proba(fitur)[0][1])
playlist = "Olahraga / Semangat" if prob > 0.5 else "Santai / Fokus"

print(f"\nProbabilitas semangat: {prob:.4f}")
print(f"Playlist: {playlist}")
