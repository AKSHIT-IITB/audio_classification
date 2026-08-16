"""
generate_dataset.py
--------------------
Synthesizes a small labeled audio corpus (>=110 clips) spanning 4 genres x 3
artists each, and writes a metadata.csv describing genre/artist/split.

Why synthetic audio? We don't have access to a licensed music corpus in this
environment. Rather than fake results on data that doesn't exist, we generate
audio whose genre/artist identity is controlled by real, distinct synthesis
parameters (oscillator type, harmonic mix, rhythm grid, noise level, vibrato).
This lets every downstream step (MFCC extraction, GA feature selection,
K-Means clustering, centroid classification, t-SNE) run on genuine signal,
producing real, reproducible numbers instead of invented ones.

4 genres x 3 artists x 10 clips = 120 clips, 3 seconds each @ 22050 Hz.
"""
import os
import csv
import numpy as np
import soundfile as sf

SR = 22050
DURATION = 3.0
N_SAMPLES = int(SR * DURATION)
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "audio")
META_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "metadata.csv")

MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11, 12]  # semitone offsets


def note_freq(root, semitone):
    return root * (2 ** (semitone / 12.0))


def adsr_envelope(n, attack=0.05, decay=0.1, sustain=0.7, release=0.15):
    a = int(n * attack); d = int(n * decay); r = int(n * release)
    s = max(n - a - d - r, 0)
    env = np.concatenate([
        np.linspace(0, 1, max(a, 1)),
        np.linspace(1, sustain, max(d, 1)),
        np.full(s, sustain),
        np.linspace(sustain, 0, max(r, 1)),
    ])
    return env[:n] if len(env) >= n else np.pad(env, (0, n - len(env)))


def synth_classical(rng, root=261.63, tempo=90, harmonic_richness=0.5, vibrato=0.0):
    """Piano/strings-like: additive harmonics + per-note ADSR envelope, smooth."""
    beat = 60.0 / tempo
    note_len = beat / 2
    t_note = np.linspace(0, note_len, int(SR * note_len), endpoint=False)
    audio = np.zeros(N_SAMPLES)
    pos = 0
    while pos < N_SAMPLES:
        semitone = MAJOR_SCALE[rng.integers(0, len(MAJOR_SCALE))]
        octave_shift = rng.choice([0, 12])
        f0 = note_freq(root, semitone + octave_shift)
        n = len(t_note)
        wave = np.zeros(n)
        for h, amp in enumerate([1.0, 0.5, 0.3, 0.15, 0.08], start=1):
            wave += amp * (harmonic_richness ** (h - 1)) * np.sin(2 * np.pi * f0 * h * t_note)
        wave *= adsr_envelope(n, attack=0.02, decay=0.15, sustain=0.4, release=0.35)
        end = min(pos + n, N_SAMPLES)
        audio[pos:end] += wave[: end - pos]
        pos += n
    return audio * 0.6


def synth_electronic(rng, root=110.0, tempo=128, harmonic_richness=0.6, noise_level=0.05):
    """EDM-like: saw/square lead over a four-on-the-floor kick + hats."""
    beat = 60.0 / tempo
    audio = np.zeros(N_SAMPLES)
    t = np.arange(N_SAMPLES) / SR

    # Saw-wave arpeggio lead (band-limited via harmonic sum)
    step = beat / 2
    pos = 0
    pattern = [0, 4, 7, 12]
    i = 0
    while pos < N_SAMPLES:
        f0 = note_freq(root * 2, pattern[i % len(pattern)])
        n = int(SR * step)
        tt = np.arange(n) / SR
        wave = np.zeros(n)
        for h in range(1, 8):
            wave += ((-1) ** (h + 1) / h) * np.sin(2 * np.pi * f0 * h * tt) * harmonic_richness
        wave *= adsr_envelope(n, attack=0.01, decay=0.05, sustain=0.8, release=0.1)
        end = min(pos + n, N_SAMPLES)
        audio[pos:end] += wave[: end - pos] * 0.35
        pos += n
        i += 1

    # Kick drum: exponentially decaying low sine on every beat
    kick_period = int(SR * beat)
    for start in range(0, N_SAMPLES, kick_period):
        n = min(int(SR * 0.15), N_SAMPLES - start)
        if n <= 0:
            continue
        tt = np.arange(n) / SR
        kick = np.sin(2 * np.pi * 60 * np.exp(-tt * 8) * tt) * np.exp(-tt * 18)
        audio[start:start + n] += kick * 0.9

    # Hi-hat: filtered noise burst on off-beats
    hat_period = int(SR * beat / 2)
    for start in range(hat_period // 2, N_SAMPLES, hat_period):
        n = min(int(SR * 0.04), N_SAMPLES - start)
        if n <= 0:
            continue
        burst = rng.standard_normal(n) * np.exp(-np.arange(n) / (0.01 * SR))
        audio[start:start + n] += burst * 0.25

    audio += noise_level * rng.standard_normal(N_SAMPLES)
    return audio * 0.6


def synth_rock(rng, root=110.0, tempo=140, distortion=3.5, noise_level=0.08):
    """Distorted power chords + noisy snare/kick, broadband and loud."""
    beat = 60.0 / tempo
    audio = np.zeros(N_SAMPLES)
    chord_len = beat * 2
    pos = 0
    while pos < N_SAMPLES:
        semitone = rng.choice([0, 5, 7])
        f0 = note_freq(root, semitone)
        n = int(SR * chord_len)
        tt = np.arange(n) / SR
        wave = np.sin(2 * np.pi * f0 * tt) + 0.8 * np.sin(2 * np.pi * f0 * 1.5 * tt)
        wave = np.tanh(wave * distortion)  # soft clipping -> rich harmonics
        wave *= adsr_envelope(n, attack=0.005, decay=0.05, sustain=0.75, release=0.1)
        end = min(pos + n, N_SAMPLES)
        audio[pos:end] += wave[: end - pos] * 0.4
        pos += n

    # Drum hits: kick on 1,3 / snare-ish noise on 2,4
    beat_samples = int(SR * beat)
    for i, start in enumerate(range(0, N_SAMPLES, beat_samples)):
        n = min(int(SR * 0.12), N_SAMPLES - start)
        if n <= 0:
            continue
        tt = np.arange(n) / SR
        if i % 2 == 0:
            hit = np.sin(2 * np.pi * 70 * np.exp(-tt * 6) * tt) * np.exp(-tt * 15)
        else:
            hit = rng.standard_normal(n) * np.exp(-tt / 0.03)
        audio[start:start + n] += hit * 0.7

    audio += noise_level * rng.standard_normal(N_SAMPLES)
    return np.clip(audio, -1, 1) * 0.6


def synth_jazz(rng, root=220.0, tempo=100, vibrato_depth=4.0, swing=0.62):
    """Sax-like tone with vibrato over a swung rhythm + brush noise."""
    beat = 60.0 / tempo
    audio = np.zeros(N_SAMPLES)
    pos = 0
    long_first = True
    scale = [0, 2, 3, 5, 7, 9, 10, 12]  # dorian-ish for a jazzy color
    while pos < N_SAMPLES:
        dur = beat * (swing if long_first else (1 - swing))
        long_first = not long_first
        semitone = scale[rng.integers(0, len(scale))]
        f0 = note_freq(root, semitone)
        n = int(SR * dur)
        if n <= 0:
            break
        tt = np.arange(n) / SR
        vibrato = vibrato_depth * np.sin(2 * np.pi * 5.5 * tt)
        wave = np.zeros(n)
        for h, amp in enumerate([1.0, 0.6, 0.4, 0.25, 0.15, 0.1], start=1):
            wave += amp * np.sin(2 * np.pi * (f0 * h + vibrato) * tt)
        wave *= adsr_envelope(n, attack=0.03, decay=0.1, sustain=0.6, release=0.2)
        end = min(pos + n, N_SAMPLES)
        audio[pos:end] += wave[: end - pos] * 0.35
        pos += n

    # Brush percussion: soft filtered noise taps
    tap_period = int(SR * beat / 2)
    for start in range(0, N_SAMPLES, tap_period):
        n = min(int(SR * 0.05), N_SAMPLES - start)
        if n <= 0:
            continue
        tap = rng.standard_normal(n) * np.exp(-np.arange(n) / (0.015 * SR))
        audio[start:start + n] += tap * 0.12

    return audio * 0.6


GENRES = {
    "classical": synth_classical,
    "electronic": synth_electronic,
    "rock": synth_rock,
    "jazz": synth_jazz,
}

# 3 artists per genre, each with a fixed timbre/tempo fingerprint.
ARTIST_PARAMS = {
    "classical": [
        dict(root=261.63, tempo=84, harmonic_richness=0.55, vibrato=0.0),
        dict(root=293.66, tempo=96, harmonic_richness=0.42, vibrato=0.0),
        dict(root=220.00, tempo=104, harmonic_richness=0.65, vibrato=0.0),
    ],
    "electronic": [
        dict(root=110.0, tempo=124, harmonic_richness=0.55, noise_level=0.04),
        dict(root=98.0, tempo=132, harmonic_richness=0.7, noise_level=0.06),
        dict(root=130.8, tempo=140, harmonic_richness=0.45, noise_level=0.05),
    ],
    "rock": [
        dict(root=110.0, tempo=132, distortion=3.0, noise_level=0.07),
        dict(root=146.8, tempo=144, distortion=4.2, noise_level=0.09),
        dict(root=98.0, tempo=150, distortion=3.7, noise_level=0.08),
    ],
    "jazz": [
        dict(root=220.0, tempo=92, vibrato_depth=3.5, swing=0.60),
        dict(root=196.0, tempo=104, vibrato_depth=5.0, swing=0.64),
        dict(root=246.9, tempo=112, vibrato_depth=4.2, swing=0.58),
    ],
}

CLIPS_PER_ARTIST = 10


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    file_idx = 0
    master_rng = np.random.default_rng(42)

    for genre, synth_fn in GENRES.items():
        for artist_i, base_params in enumerate(ARTIST_PARAMS[genre]):
            artist_name = f"{genre}_artist{artist_i + 1}"
            for clip_i in range(CLIPS_PER_ARTIST):
                seed = master_rng.integers(0, 1_000_000)
                rng = np.random.default_rng(seed)
                params = dict(base_params)
                # small per-clip jitter so an artist's clips aren't identical
                if "tempo" in params:
                    params["tempo"] = params["tempo"] * (1 + rng.uniform(-0.03, 0.03))
                audio = synth_fn(rng, **params)
                audio = audio / (np.max(np.abs(audio)) + 1e-9) * 0.85

                fname = f"{genre}_{artist_name}_{clip_i:02d}.wav"
                fpath = os.path.join(OUT_DIR, fname)
                sf.write(fpath, audio.astype(np.float32), SR)

                rows.append({
                    "file_id": file_idx,
                    "filename": fname,
                    "genre": genre,
                    "artist": artist_name,
                })
                file_idx += 1

    # 60/60 split: 5 of each artist's 10 clips go to train, 5 to test(unlabeled)
    for genre, artists in ARTIST_PARAMS.items():
        pass

    rng2 = np.random.default_rng(7)
    by_artist = {}
    for r in rows:
        by_artist.setdefault(r["artist"], []).append(r)
    for artist, clips in by_artist.items():
        idxs = np.arange(len(clips))
        rng2.shuffle(idxs)
        train_idxs = set(idxs[:5].tolist())
        for i, r in enumerate(clips):
            r["split"] = "train" if i in train_idxs else "test"

    with open(META_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_id", "filename", "genre", "artist", "split"])
        writer.writeheader()
        writer.writerows(rows)

    n_train = sum(1 for r in rows if r["split"] == "train")
    n_test = sum(1 for r in rows if r["split"] == "test")
    print(f"Generated {len(rows)} clips -> {OUT_DIR}")
    print(f"Train: {n_train}, Test(unlabeled): {n_test}")
    print(f"Metadata -> {META_PATH}")


if __name__ == "__main__":
    main()
