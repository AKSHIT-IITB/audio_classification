"""
genetic_algorithm.py
----------------------
Genetic-algorithm-based feature selection.

We have 75 raw MFCC/spectral/rhythm features per clip -- not all of them
help separate genres in feature space, and some are redundant/noisy. A GA
searches the space of {use feature / drop feature} subsets to find one that
produces well-separated, compact K-Means clusters.

Chromosome : binary vector of length n_features (1 = feature kept)
Fitness    : silhouette score of K-Means(k=n_genres) clustering on the
             standardized training features restricted to the selected
             subset, with a small penalty for using too many features
             (encourages compact, generalizable subsets). Fitness is
             computed WITHOUT using genre/artist labels -- clustering stays
             unsupervised, exactly as in the original pipeline.
Operators  : tournament selection, uniform crossover, bit-flip mutation,
             elitism.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

FEATURES_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "features.csv")
OUT_JSON = os.path.join(os.path.dirname(__file__), "..", "outputs", "selected_features.json")
OUT_CONVERGENCE = os.path.join(os.path.dirname(__file__), "..", "outputs", "ga_convergence.csv")

POP_SIZE = 40
N_GENERATIONS = 60
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.03
TOURNAMENT_SIZE = 3
ELITISM = 2
MIN_FEATURES = 4
FEATURE_PENALTY = 0.15  # weight on (n_selected / n_total) subtracted from silhouette


def fitness(mask, X, k, rng):
    if mask.sum() < MIN_FEATURES:
        return -1.0
    X_sub = X[:, mask.astype(bool)]
    km = KMeans(n_clusters=k, n_init=8, random_state=42)
    labels = km.fit_predict(X_sub)
    if len(set(labels)) < 2:
        return -1.0
    sil = silhouette_score(X_sub, labels)
    penalty = FEATURE_PENALTY * (mask.sum() / len(mask))
    return sil - penalty


def tournament_select(pop, fits, rng):
    idxs = rng.integers(0, len(pop), size=TOURNAMENT_SIZE)
    best = idxs[np.argmax(fits[idxs])]
    return pop[best].copy()


def crossover(p1, p2, rng):
    if rng.random() > CROSSOVER_RATE:
        return p1.copy(), p2.copy()
    mask = rng.random(len(p1)) < 0.5
    c1 = np.where(mask, p1, p2)
    c2 = np.where(mask, p2, p1)
    return c1, c2


def mutate(ind, rng):
    flip = rng.random(len(ind)) < MUTATION_RATE
    ind = ind.copy()
    ind[flip] = 1 - ind[flip]
    return ind


def run_ga(X, k, feature_names, seed=42):
    rng = np.random.default_rng(seed)
    n_features = X.shape[1]

    population = [
        (rng.random(n_features) < 0.5).astype(int) for _ in range(POP_SIZE)
    ]
    # ensure at least MIN_FEATURES genes set in each individual
    for ind in population:
        if ind.sum() < MIN_FEATURES:
            on = rng.choice(n_features, MIN_FEATURES, replace=False)
            ind[on] = 1

    history = []
    best_ind, best_fit = None, -np.inf

    for gen in range(N_GENERATIONS):
        fits = np.array([fitness(ind, X, k, rng) for ind in population])

        gen_best_idx = np.argmax(fits)
        if fits[gen_best_idx] > best_fit:
            best_fit = fits[gen_best_idx]
            best_ind = population[gen_best_idx].copy()

        history.append({
            "generation": gen,
            "best_fitness": float(fits.max()),
            "mean_fitness": float(fits.mean()),
            "best_overall": float(best_fit),
        })

        # elitism
        elite_idx = np.argsort(fits)[-ELITISM:]
        new_population = [population[i].copy() for i in elite_idx]

        while len(new_population) < POP_SIZE:
            p1 = tournament_select(population, fits, rng)
            p2 = tournament_select(population, fits, rng)
            c1, c2 = crossover(p1, p2, rng)
            c1 = mutate(c1, rng)
            c2 = mutate(c2, rng)
            new_population.append(c1)
            if len(new_population) < POP_SIZE:
                new_population.append(c2)

        population = new_population
        print(f"gen {gen:3d} | best={fits.max():.4f} | mean={fits.mean():.4f} | best_overall={best_fit:.4f}")

    selected = [f for f, bit in zip(feature_names, best_ind) if bit == 1]
    return best_ind, best_fit, selected, history


def main():
    df = pd.read_csv(FEATURES_PATH)
    train = df[df["split"] == "train"].reset_index(drop=True)
    id_cols = ["file_id", "filename", "genre", "artist", "split"]
    feature_names = [c for c in df.columns if c not in id_cols]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_names].values)

    k = train["genre"].nunique()
    print(f"Running GA feature selection on {X_train.shape[0]} training clips, "
          f"{X_train.shape[1]} raw features, k={k} genres\n")

    best_mask, best_fit, selected, history = run_ga(X_train, k, feature_names)

    print(f"\nBest fitness (silhouette - size penalty): {best_fit:.4f}")
    print(f"Selected {len(selected)}/{len(feature_names)} features:")
    for f in selected:
        print(f"  - {f}")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({
            "selected_features": selected,
            "best_fitness": best_fit,
            "n_total_features": len(feature_names),
        }, f, indent=2)

    pd.DataFrame(history).to_csv(OUT_CONVERGENCE, index=False)
    print(f"\nSaved -> {OUT_JSON}")
    print(f"Saved -> {OUT_CONVERGENCE}")


if __name__ == "__main__":
    main()
