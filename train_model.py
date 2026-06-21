"""Latih JST rekomendasi playlist musik."""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

MODEL_FILE = "model_jst_musik.pkl"
DATA_FILE = "data_musik.csv"
TEMPO_MIN, TEMPO_MAX = 60.0, 150.0


def buat_dataset():
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(120):
        rows.append({
            "tempo": round(rng.uniform(60, 98), 1),
            "energy": round(rng.uniform(0.05, 0.48), 2),
            "danceability": round(rng.uniform(0.05, 0.45), 2),
            "label": 0,
        })
    for _ in range(120):
        rows.append({
            "tempo": round(rng.uniform(112, 150), 1),
            "energy": round(rng.uniform(0.62, 0.98), 2),
            "danceability": round(rng.uniform(0.55, 0.95), 2),
            "label": 1,
        })
    pd.DataFrame(rows).sample(frac=1, random_state=42).to_csv(DATA_FILE, index=False)
    print(f"Dataset dibuat: {DATA_FILE} ({len(rows)} lagu)")


def normalisasi(data):
    tempo = (data["tempo"] - TEMPO_MIN) / (TEMPO_MAX - TEMPO_MIN)
    return np.column_stack([tempo, data["energy"], data["danceability"]]).astype("float32")


def buat_model():
    # Input(3) → Hidden 4 ReLU → Output Sigmoid (binary)
    return MLPClassifier(
        hidden_layer_sizes=(4,),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
    )


if __name__ == "__main__":
    if os.path.exists(MODEL_FILE):
        print(f"Model '{MODEL_FILE}' sudah ada. Hapus dulu kalau mau training ulang.")
        exit()

    if not os.path.exists(DATA_FILE):
        buat_dataset()

    data = pd.read_csv(DATA_FILE)
    X = normalisasi(data)
    y = data["label"].values
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Melatih model (500 iterasi)...")
    model = buat_model()
    model.fit(x_train, y_train)

    acc = model.score(x_test, y_test)
    print(f"Akurasi: {acc * 100:.2f}%")

    joblib.dump(model, MODEL_FILE)
    print(f"Selesai. Jalankan: python3 prediksi.py")
