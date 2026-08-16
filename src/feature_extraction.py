"""
feature_extraction.py
----------------------
Audio -> MFCC (+ complementary spectral/rhythm features) -> one fixed-length
numerical feature vector per file. This is the "Audio -> MFCC features" step
of the pipeline.

For every clip we compute, per-frame:
  - 13 MFCCs
  - 13 delta-MFCCs (first derivative, captures how timbre changes over time)
  - spectral centroid, spectral bandwidth, spectral rolloff
  - zero-crossing rate, RMS energy
  - 12-bin chroma (pitch-class energy distribution)
and then summarize each with mean + std across frames (except chroma, mean
only), giving a fixed-length vector per clip regardless of duration.
"""
import os
import numpy as np
import pandas as pd
import librosa

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "audio")
META_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "metadata.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "features.csv")

N_MFCC = 13


def extract_features(filepath, sr=22050, n_mfcc=N_MFCC):
    y, sr = librosa.load(filepath, sr=sr)
    y, _ = librosa.effects.trim(y, top_db=40)
    if len(y) < sr // 2:
        y = np.pad(y, (0, sr // 2 - len(y)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    feat = {}
    for i in range(n_mfcc):
        feat[f"mfcc{i+1}_mean"] = float(np.mean(mfcc[i]))
        feat[f"mfcc{i+1}_std"] = float(np.std(mfcc[i]))
    for i in range(n_mfcc):
        feat[f"dmfcc{i+1}_mean"] = float(np.mean(delta_mfcc[i]))
        feat[f"dmfcc{i+1}_std"] = float(np.std(delta_mfcc[i]))
    feat["centroid_mean"] = float(np.mean(centroid))
    feat["centroid_std"] = float(np.std(centroid))
    feat["bandwidth_mean"] = float(np.mean(bandwidth))
    feat["bandwidth_std"] = float(np.std(bandwidth))
    feat["rolloff_mean"] = float(np.mean(rolloff))
    feat["rolloff_std"] = float(np.std(rolloff))
    feat["zcr_mean"] = float(np.mean(zcr))
    feat["zcr_std"] = float(np.std(zcr))
    feat["rms_mean"] = float(np.mean(rms))
    feat["rms_std"] = float(np.std(rms))
    for i in range(12):
        feat[f"chroma{i+1}_mean"] = float(np.mean(chroma[i]))

    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        feat["tempo"] = float(np.atleast_1d(tempo)[0])
    except Exception:
        feat["tempo"] = 0.0

    return feat


def main():
    meta = pd.read_csv(META_PATH)
    records = []
    for _, row in meta.iterrows():
        fpath = os.path.join(AUDIO_DIR, row["filename"])
        feat = extract_features(fpath)
        feat["file_id"] = row["file_id"]
        feat["filename"] = row["filename"]
        feat["genre"] = row["genre"]
        feat["artist"] = row["artist"]
        feat["split"] = row["split"]
        records.append(feat)
        print(f"[{row['file_id']+1}/{len(meta)}] extracted {row['filename']}")

    df = pd.DataFrame(records)
    id_cols = ["file_id", "filename", "genre", "artist", "split"]
    feature_cols = [c for c in df.columns if c not in id_cols]
    df = df[id_cols + feature_cols]
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\n{len(feature_cols)} raw features x {len(df)} clips -> {OUT_PATH}")


if __name__ == "__main__":
    main()
