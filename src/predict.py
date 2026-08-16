"""
predict.py
------------
Classify one new audio file with the trained pipeline.

Usage:
    python3 src/predict.py path/to/audio.wav

Runs the same steps a test clip goes through:
  extract features -> keep GA-selected subset -> scale with training scaler
  -> Euclidean distance to each trained centroid -> nearest wins.
"""
import os
import sys
import json
import numpy as np
import joblib

from feature_extraction import extract_features

BASE = os.path.join(os.path.dirname(__file__), "..")
SELECTED_PATH = os.path.join(BASE, "outputs", "selected_features.json")
MODEL_DIR = os.path.join(BASE, "outputs", "model")


def predict(filepath):
    with open(SELECTED_PATH) as f:
        selected = json.load(f)["selected_features"]
    with open(os.path.join(MODEL_DIR, "cluster_to_genre.json")) as f:
        cluster_to_genre = {int(k): v for k, v in json.load(f).items()}
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    km = joblib.load(os.path.join(MODEL_DIR, "kmeans.joblib"))

    feats = extract_features(filepath)
    x = np.array([[feats[f] for f in selected]])
    x = scaler.transform(x)

    dists = np.linalg.norm(x - km.cluster_centers_, axis=1)
    cluster = int(np.argmin(dists))
    return cluster, cluster_to_genre[cluster], dists


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    filepath = sys.argv[1]
    if not os.path.isfile(filepath):
        print(f"File not found: {filepath}")
        sys.exit(1)

    cluster, genre, dists = predict(filepath)
    print(f"File            : {os.path.basename(filepath)}")
    print(f"Predicted genre : {genre}  (cluster {cluster})")
    print("Distances to centroids:")
    for c, d in enumerate(dists):
        marker = "  <-- nearest" if c == cluster else ""
        print(f"  cluster {c}: {d:.4f}{marker}")


if __name__ == "__main__":
    main()
