"""
run_pipeline.py
------------------
Runs the full pipeline end to end, in order:

  1. generate_dataset   -> synthesize labeled audio corpus (120 clips)
  2. feature_extraction -> Audio -> MFCC + spectral/rhythm features
  3. genetic_algorithm  -> GA-based feature subset selection (unsupervised fitness)
  4. clustering         -> train K-Means on the 60 training clips
  5. classify           -> assign the 60 held-out clips via least-centroid-distance
  6. evaluate           -> metrics + t-SNE + confusion matrix + GA convergence plots

Usage:
    python run_pipeline.py
"""
import subprocess
import sys
import os

STEPS = [
    ("Generating synthetic labeled audio dataset", "src/generate_dataset.py"),
    ("Extracting MFCC / spectral / rhythm features", "src/feature_extraction.py"),
    ("Running GA-based feature selection", "src/genetic_algorithm.py"),
    ("Training K-Means clustering", "src/clustering.py"),
    ("Classifying unlabeled clips (least-centroid-distance)", "src/classify.py"),
    ("Evaluating + generating figures", "src/evaluate.py"),
]


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    for i, (desc, script) in enumerate(STEPS, 1):
        print("\n" + "=" * 70)
        print(f"STEP {i}/{len(STEPS)}: {desc}")
        print("=" * 70)
        result = subprocess.run([sys.executable, script], cwd=root)
        if result.returncode != 0:
            print(f"\nStep failed: {script}")
            sys.exit(1)
    print("\n" + "=" * 70)
    print("Pipeline complete. See outputs/final_metrics.json and outputs/figures/")
    print("=" * 70)


if __name__ == "__main__":
    main()
