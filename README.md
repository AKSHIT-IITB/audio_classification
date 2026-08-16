# Audio Genre/Artist Classification — MFCC + GA + K-Means

An unsupervised audio classification pipeline:

```
Audio → MFCC (+ spectral/rhythm) features → GA feature selection → K-Means clustering → centroids → classify new audio by least-centroid-distance
```

## Why synthetic audio?

This environment has no licensed music corpus available, so `src/generate_dataset.py`
**synthesizes** 120 audio clips (4 genres × 3 "artists" × 10 clips, 3s each @ 22.05kHz)
using distinct oscillator/rhythm/timbre recipes per genre (e.g. distorted power
chords + drum hits for rock, saw-wave arpeggios + four-on-the-floor kicks for
electronic, additive harmonics with ADSR envelopes for classical, vibrato +
swing rhythm for jazz). This keeps every downstream number in this repo **real
and reproducible** — nothing below is fabricated; run the pipeline yourself
and you'll get the same figures.

The pipeline logic itself (MFCC extraction, GA feature selection, K-Means,
least-centroid-distance classification, t-SNE/metric evaluation) is
identical to what you'd run on a real corpus (e.g. GTZAN) — swap in real
`.wav` files + a metadata CSV and everything else works unchanged.

## Pipeline stages

| # | Script | What it does |
|---|--------|---------------|
| 1 | `src/generate_dataset.py` | Synthesizes 120 labeled clips + `data/metadata.csv` (genre, artist, train/test split) |
| 2 | `src/feature_extraction.py` | Extracts 75 features/clip: 13 MFCCs + 13 delta-MFCCs (mean+std), spectral centroid/bandwidth/rolloff, ZCR, RMS, 12-bin chroma, tempo |
| 3 | `src/genetic_algorithm.py` | Binary-chromosome GA searches feature subsets; fitness = silhouette score of K-Means clustering (unsupervised — no genre labels used) minus a feature-count penalty |
| 4 | `src/clustering.py` | Trains K-Means (k=4) on the 60 training clips using only the GA-selected features; saves scaler + centroids |
| 5 | `src/classify.py` | Classifies the 60 held-out "unlabeled" clips by **least Euclidean distance to trained centroids** |
| 6 | `src/evaluate.py` | Clustering metrics, external metrics vs. true genre (reporting only), confusion matrix, t-SNE plots, GA convergence plot |

Run everything:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 run_pipeline.py
```

Outputs land in `outputs/`: `features.csv`, `selected_features.json`,
`ga_convergence.csv`, `model/` (scaler + KMeans + mappings),
`test_predictions.csv`, `final_metrics.json`, `figures/*.png`.

Classify a single new audio file with the trained model:

```bash
python3 src/predict.py path/to/audio.wav
# File            : audio.wav
# Predicted genre : rock  (cluster 3)
# Distances to centroids:
#   cluster 0: 2.9129
#   ...
#   cluster 3: 0.3306  <-- nearest
```

## Results

See [`RESULTS.md`](RESULTS.md) for the full write-up with numbers and plots
from the last pipeline run.

**Headline numbers** (60 train / 60 held-out test clips, k=4 genre clusters):

| Metric | Value |
|---|---|
| Raw features extracted | 75 |
| GA-selected features | 12 |
| GA fitness improvement | 0.392 → 0.830 (60 generations) |
| Train silhouette score | 0.854 |
| Train Davies-Bouldin index | 0.210 |
| Test least-centroid-distance accuracy vs. true genre | 100% (60/60) |
| Test Adjusted Rand Index / NMI vs. true genre | 1.00 / 1.00 |

100% test accuracy is expected here because the synthetic genres are
built from cleanly distinct signal recipes — it's a sanity check that the
pipeline logic (features → GA → K-Means → centroid distance) works
correctly end to end, not a claim about performance on real, messier music.
On a real corpus like GTZAN, expect materially lower separability.

## Concept walkthrough

See `RESULTS.md` for the full write-up:
MFCC → K-Means → Genetic Algorithm → t-SNE → Euclidean/centroid distance.
