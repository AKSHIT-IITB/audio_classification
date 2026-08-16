"""
evaluate.py
-------------
Produces the final evaluation artifacts:
  - clustering metrics (silhouette, Davies-Bouldin, Calinski-Harabasz) on train
  - external metrics (Adjusted Rand Index, Normalized Mutual Info) vs true
    genre labels, for both train clusters and test predictions (used ONLY
    for reporting -- never fed back into GA fitness or K-Means training)
  - confusion matrix (predicted cluster/genre vs true genre) on the test set
  - t-SNE 2D projection of all 120 clips, colored by true genre and by
    predicted cluster, on both the full 75-feature set and the GA-selected
    12-feature set (to visually show the GA's effect)
  - GA convergence curve
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score, confusion_matrix,
)

BASE = os.path.join(os.path.dirname(__file__), "..")
FEATURES_PATH = os.path.join(BASE, "outputs", "features.csv")
SELECTED_PATH = os.path.join(BASE, "outputs", "selected_features.json")
MODEL_DIR = os.path.join(BASE, "outputs", "model")
TEST_PRED_PATH = os.path.join(BASE, "outputs", "test_predictions.csv")
GA_CONV_PATH = os.path.join(BASE, "outputs", "ga_convergence.csv")
FIG_DIR = os.path.join(BASE, "outputs", "figures")
METRICS_OUT = os.path.join(BASE, "outputs", "final_metrics.json")

GENRE_COLORS = {"classical": "#4C72B0", "electronic": "#DD8452", "rock": "#55A868", "jazz": "#C44E52"}


def plot_ga_convergence():
    hist = pd.read_csv(GA_CONV_PATH)
    plt.figure(figsize=(7, 4.5))
    plt.plot(hist["generation"], hist["best_overall"], label="Best fitness so far", linewidth=2, color="#4C72B0")
    plt.plot(hist["generation"], hist["mean_fitness"], label="Population mean fitness", linewidth=1.3, alpha=0.7, color="#DD8452")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (silhouette - feature-count penalty)")
    plt.title("Genetic Algorithm Feature Selection: Convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "ga_convergence.png"), dpi=150)
    plt.close()


def plot_tsne(df, feature_cols, true_labels, pred_labels, title_suffix, fname):
    X = df[feature_cols].values
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    perplexity = min(30, max(5, len(df) // 4))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
    emb = tsne.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for g, color in GENRE_COLORS.items():
        mask = true_labels == g
        ax.scatter(emb[mask, 0], emb[mask, 1], c=color, label=g, s=45, edgecolor="white", linewidth=0.5)
    ax.set_title(f"t-SNE colored by TRUE genre ({title_suffix})")
    ax.legend(fontsize=8)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")

    ax = axes[1]
    cmap = plt.get_cmap("tab10")
    for c in sorted(pd.unique(pred_labels)):
        mask = pred_labels == c
        ax.scatter(emb[mask, 0], emb[mask, 1], c=[cmap(int(c) % 10)], label=f"cluster {c}", s=45, edgecolor="white", linewidth=0.5)
    ax.set_title(f"t-SNE colored by PREDICTED cluster ({title_suffix})")
    ax.legend(fontsize=8)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=150)
    plt.close()


def plot_confusion(true_genre, pred_genre, labels, fname, title):
    cm = confusion_matrix(true_genre, pred_genre, labels=labels)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted genre"); ax.set_ylabel("True genre")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=150)
    plt.close()
    return cm


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    df = pd.read_csv(FEATURES_PATH)
    with open(SELECTED_PATH) as f:
        selected = json.load(f)["selected_features"]
    id_cols = ["file_id", "filename", "genre", "artist", "split"]
    all_raw_features = [c for c in df.columns if c not in id_cols]

    km = joblib.load(os.path.join(MODEL_DIR, "kmeans.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    with open(os.path.join(MODEL_DIR, "cluster_to_genre.json")) as f:
        cluster_to_genre = {int(k): v for k, v in json.load(f).items()}
    with open(os.path.join(MODEL_DIR, "metrics_train.json")) as f:
        train_metrics = json.load(f)

    train = df[df["split"] == "train"].reset_index(drop=True)
    test_pred = pd.read_csv(TEST_PRED_PATH)

    # ---- external metrics (reporting only) ----
    X_train_sel = scaler.transform(train[selected].values)
    train_clusters = km.predict(X_train_sel)
    ari_train = adjusted_rand_score(train["genre"], train_clusters)
    nmi_train = normalized_mutual_info_score(train["genre"], train_clusters)

    ari_test = adjusted_rand_score(test_pred["true_genre"], test_pred["predicted_cluster"])
    nmi_test = normalized_mutual_info_score(test_pred["true_genre"], test_pred["predicted_cluster"])
    test_acc = (test_pred["true_genre"] == test_pred["predicted_genre"]).mean()

    genres_sorted = sorted(df["genre"].unique())
    cm_test = plot_confusion(
        test_pred["true_genre"], test_pred["predicted_genre"], genres_sorted,
        "confusion_matrix_test.png", "Test set (unlabeled clips): True vs Predicted genre"
    )

    # ---- t-SNE: full 75 raw features vs GA-selected 12 features ----
    all_clusters = np.concatenate([train_clusters, test_pred["predicted_cluster"].values])
    all_true_genre = pd.concat([train["genre"], test_pred["true_genre"]]).values
    all_df = pd.concat([train[all_raw_features + ["genre"]], df[df["split"] == "test"][all_raw_features + ["genre"]]], ignore_index=True)

    plot_tsne(all_df, all_raw_features, all_true_genre, all_clusters,
              "all 75 raw features", "tsne_all_features.png")
    plot_tsne(all_df, selected, all_true_genre, all_clusters,
              "12 GA-selected features", "tsne_selected_features.png")

    plot_ga_convergence()

    # ---- final report ----
    metrics = {
        "n_total_clips": int(len(df)),
        "n_train": int(len(train)),
        "n_test": int(len(test_pred)),
        "n_raw_features": int(len(all_raw_features)),
        "n_ga_selected_features": int(len(selected)),
        "ga_selected_features": selected,
        "k_clusters": int(train_metrics["k"]),
        "train_silhouette": train_metrics["silhouette"],
        "train_davies_bouldin": train_metrics["davies_bouldin"],
        "train_calinski_harabasz": train_metrics["calinski_harabasz"],
        "train_ari_vs_true_genre": float(ari_train),
        "train_nmi_vs_true_genre": float(nmi_train),
        "test_ari_vs_true_genre": float(ari_test),
        "test_nmi_vs_true_genre": float(nmi_test),
        "test_nearest_centroid_accuracy": float(test_acc),
        "cluster_to_genre_mapping": cluster_to_genre,
        "confusion_matrix_labels": genres_sorted,
        "confusion_matrix_test": cm_test.tolist(),
    }
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"\nFigures saved to -> {FIG_DIR}")
    print(f"Final metrics saved to -> {METRICS_OUT}")


if __name__ == "__main__":
    main()
