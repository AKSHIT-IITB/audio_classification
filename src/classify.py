"""
classify.py
-------------
Least-centroid-distance classification of the 60 held-out "unlabeled" clips.

For each test clip:
  1. take its GA-selected feature subset (already extracted)
  2. scale with the SAME scaler fitted on the training set
  3. compute Euclidean distance to each of the K trained centroids
  4. assign the cluster (and majority-vote genre name) of the nearest one
"""
import os
import json
import numpy as np
import pandas as pd
import joblib

FEATURES_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "features.csv")
SELECTED_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "selected_features.json")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "model")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "test_predictions.csv")


def main():
    df = pd.read_csv(FEATURES_PATH)
    with open(SELECTED_PATH) as f:
        selected = json.load(f)["selected_features"]
    with open(os.path.join(MODEL_DIR, "cluster_to_genre.json")) as f:
        cluster_to_genre = {int(k): v for k, v in json.load(f).items()}

    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    km = joblib.load(os.path.join(MODEL_DIR, "kmeans.joblib"))

    test = df[df["split"] == "test"].reset_index(drop=True)
    X_test = scaler.transform(test[selected].values)
    centroids = km.cluster_centers_  # (k, n_selected_features)

    # distance of every test point to every centroid
    dists = np.linalg.norm(X_test[:, None, :] - centroids[None, :, :], axis=2)  # (n_test, k)
    pred_cluster = np.argmin(dists, axis=1)
    min_dist = dists[np.arange(len(dists)), pred_cluster]
    pred_genre = [cluster_to_genre[int(c)] for c in pred_cluster]

    out = test[["file_id", "filename", "genre", "artist"]].copy()
    out.rename(columns={"genre": "true_genre"}, inplace=True)
    out["predicted_cluster"] = pred_cluster
    out["predicted_genre"] = pred_genre
    out["distance_to_centroid"] = min_dist
    for c in range(centroids.shape[0]):
        out[f"dist_to_cluster{c}"] = dists[:, c]

    correct = (out["true_genre"] == out["predicted_genre"]).sum()
    acc = correct / len(out)

    out.to_csv(OUT_PATH, index=False)
    print(f"Classified {len(out)} unlabeled clips via least-centroid-distance")
    print(f"Agreement with true genre (external check only, not used for training): "
          f"{correct}/{len(out)} = {acc*100:.1f}%")
    print(f"\nSaved -> {OUT_PATH}")
    print("\nSample predictions:")
    print(out[["filename", "true_genre", "predicted_genre", "distance_to_centroid"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
