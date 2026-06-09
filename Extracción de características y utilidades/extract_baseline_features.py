"""
extract_baseline_features.py
Lee un manifest de WAVs (originales + augmentados) y extrae features base.
Entrada: CSV con columna 'wav_path' (por ejemplo augmented_manifest.csv) o carpeta.
Salida: D:/pipeline_SVM/features/features_svm_baseline_augmented.csv
"""
import os, argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import librosa
import soundfile as sf
from scipy.stats import variation

ROOT = Path("D:/pipeline_SVM")
OUT_FEATURES = ROOT / "features" / "features_svm_baseline_augmented.csv"
OUT_FEATURES.parent.mkdir(parents=True, exist_ok=True)

# helper functions
def spectral_entropy(S):
    # S: magnitude spectrogram (frames x bins) or 1D
    p = S / (np.sum(S) + 1e-12)
    p = np.where(p <= 0, 1e-12, p)
    return -np.sum(p * np.log2(p))

def crest_factor(y):
    peak = np.max(np.abs(y)) + 1e-12
    rms = np.sqrt(np.mean(y**2)) + 1e-12
    return peak / rms

def harmonic_percussive_ratio(y):
    y_h, y_p = librosa.effects.hpss(y)
    e_h = np.sum(y_h**2)
    e_p = np.sum(y_p**2) + 1e-12
    return e_h / e_p

def process_file(p, sr=44100):
    y, sr = librosa.load(str(p), sr=sr, mono=True)
    duration_s = len(y)/sr
    # RMS/MEL energy
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_total_energy = float(np.mean(np.sum(mel, axis=0)))
    rms = float(np.mean(librosa.feature.rms(y=y)))
    peak = float(np.max(np.abs(y)))
    # spectral
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    # spectral entropy: compute on power spectral density
    S = np.abs(librosa.stft(y, n_fft=2048))
    spec_entropy = spectral_entropy(np.mean(S, axis=1))
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_rate = 0.0 if duration_s == 0 else len(onsets)/duration_s
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_std = float(np.std(chroma))
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = float(np.mean(contrast))
    hp_ratio = harmonic_percussive_ratio(y)
    cr = crest_factor(y)
    return {
        "filepath": str(p),
        "duration_s": duration_s,
        "mel_total_energy": mel_total_energy,
        "rms": rms,
        "peak": peak,
        "centroid_mean": centroid,
        "zcr": zcr,
        "spectral_flatness_mean": flatness,
        "spectral_entropy_mean": spec_entropy,
        "onset_rate": onset_rate,
        "chroma_std": chroma_std,
        "spectral_contrast_mean": contrast_mean,
        "harmonic_percussive_ratio": hp_ratio,
        "crest_factor": cr
    }

def main():
    # Prefer augmented manifest if exists, otherwise process all wavs under augmented
    aug_manifest = ROOT / "augmented" / "augmented_manifest.csv"
    rows = []
    if aug_manifest.exists():
        dfm = pd.read_csv(aug_manifest)
        wavs = dfm['wav_path'].tolist()
        labels = dfm['label'].tolist()
        idx = 0
        out_rows = []
        for p, lab in zip(wavs, labels):
            p = Path(p)
            if not p.exists():
                print("no existe wav:", p)
                continue
            feats = process_file(p)
            feats['label'] = lab
            feats['orig'] = dfm.iloc[idx].get('orig', "")
            feats['aug_ops'] = dfm.iloc[idx].get('aug_ops', "")
            out_rows.append(feats)
            idx += 1
        if out_rows:
            pd.DataFrame(out_rows).to_csv(OUT_FEATURES, index=False)
            print("Guardado features:", OUT_FEATURES, "n:", len(out_rows))
        else:
            print("No se extrajeron features (rows vacías).")
    else:
        # fallback: process all wavs in augmented folder
        wavs = list((ROOT/"augmented").rglob("*.wav"))
        out_rows = []
        for p in wavs:
            feats = process_file(p)
            # derive label from parent folder name
            feats['label'] = p.parent.name
            out_rows.append(feats)
        pd.DataFrame(out_rows).to_csv(OUT_FEATURES, index=False)
        print("Guardado features (all):", OUT_FEATURES, "n:", len(out_rows))

if __name__ == "__main__":
    main()
