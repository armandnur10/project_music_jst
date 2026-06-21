# JST Rekomendasi Playlist Musik

Klasifikasi: lagu masuk **Santai/Fokus** (0) atau **Olahraga/Semangat** (1).

## Setup

```bash
pip install -r requirements.txt
```

## Jalankan

```bash
python3 train_model.py   # latih model
python3 prediksi.py      # coba prediksi
```

## File

- `train_model.py` — buat dataset + latih JST
- `prediksi.py` — input tempo/energy/danceability → playlist
- `data_musik.csv` — dataset (auto-generated saat training)

## Model

Input(3) → Hidden 4 ReLU → Output Sigmoid | Adam | Binary Crossentropy

Threshold 0.5: prob > 0.5 = Semangat, else Santai.
