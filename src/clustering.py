"""
clustering.py
---------------
Trains K-Means on the 60 training clips using only the GA-selected feature
subset. Saves the fitted scaler, centroids, and cluster->genre name mapping
(majority vote, for human-readable reporting only -- the clustering itself
never sees the genre labels).
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

FEATURES_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "features.csv")
SELECTED_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "selected_features.json")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "model")


def main():
    df = pd.read_csv(FEATURES_PATH)
    with open(SELECTED_PATH) as f:
        selected = json.load(f)["selected_features"]

    train = df[df["split"] == "train"].reset_index(drop=True)
    k = train["genre"].nunique()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[selected].values)

    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    train_labels = km.fit_predict(X_train)

    sil = silhouette_score(X_train, train_labels)
    dbi = davies_bouldin_score(X_train, train_labels)
    ch = calinski_harabasz_score(X_train, train_labels)
    print(f"Trained K-Means (k={k}) on {X_train.shape[0]} clips x {len(selected)} selected features")
    print(f"Silhouette Score      : {sil:.4f}  (higher=better, range [-1,1])")
    print(f"Davies-Bouldin Index  : {dbi:.4f}  (lower=better)")
    print(f"Calinski-Harabasz     : {ch:.2f}  (higher=better)")

    # map each cluster id -> majority genre (for readability only)
    cluster_to_genre = {}
    for c in range(k):
        genres_in_cluster = train.loc[train_labels == c, "genre"]
        if len(genres_in_cluster):
            cluster_to_genre[int(c)] = genres_in_cluster.value_counts().idxmax()
        else:
            cluster_to_genre[int(c)] = "unknown"
    print("\nCluster -> majority genre mapping:")
    for c, g in cluster_to_genre.items():
        print(f"  cluster {c}: {g}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(km, os.path.join(MODEL_DIR, "kmeans.joblib"))
    with open(os.path.join(MODEL_DIR, "cluster_to_genre.json"), "w") as f:
        json.dump(cluster_to_genre, f, indent=2)
    with open(os.path.join(MODEL_DIR, "metrics_train.json"), "w") as f:
        json.dump({"silhouette": sil, "davies_bouldin": dbi, "calinski_harabasz": ch, "k": k}, f, indent=2)

    train_out = train[["file_id", "filename", "genre", "artist"]].copy()
    train_out["cluster"] = train_labels
    train_out.to_csv(os.path.join(os.path.dirname(MODEL_DIR), "train_clusters.csv"), index=False)
    print(f"\nModel + metrics saved -> {MODEL_DIR}")


if __name__ == "__main__":
    main()
