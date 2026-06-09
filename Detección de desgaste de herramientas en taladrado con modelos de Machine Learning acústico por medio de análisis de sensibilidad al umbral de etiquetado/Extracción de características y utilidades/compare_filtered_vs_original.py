#!/usr/bin/env python3
"""
compare_filtered_vs_original.py

Compara el rendimiento del SVM ordinal Frank & Hall con features extraidas
de WAVs originales vs WAVs filtrados con noisereduce.

Pipeline:
  1. Carga features originales (features_curated_splits.csv)
  2. Extrae features de WAVs limpios (D:/dataset/cleaned_wavs/)
  3. Entrena SVM ordinal en ambos conjuntos con los mismos splits
  4. Genera tabla comparativa + graficas para el articulo

Uso:
    python compare_filtered_vs_original.py
    python compare_filtered_vs_original.py --top-k 7   # usar top-7 features
    python compare_filtered_vs_original.py --thresholds 75,85,97
"""

import sys, os, json, argparse, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import librosa
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import classification_report, f1_score, accuracy_score

from ordinal_utils import (
    LABEL_TO_IDX, IDX_TO_LABEL,
    ordinal_decode, ordinal_proba, print_ordinal_report,
    ordinal_mae, adjacent_accuracy
)

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
FEATURES_ORIG = Path("D:/pipeline_SVM/features/features_curated_splits.csv")
CLEAN_DIR     = Path("D:/dataset/cleaned_wavs/combinado")
OUTDIR        = Path("D:/pipeline_SVM/results/comparison_filtered")
MASTER_CSV    = Path("D:/dataset/manifests/master.csv")

SR = 44100  # Mismo que feature extraction y noisereduce

# Top-7 features (ablation demostro que supera top-10)
TOP7_FEATURES = [
    "spectral_contrast_mean", "crest_factor", "chroma_std", "zcr",
    "spectral_entropy_mean", "centroid_mean", "harmonic_percussive_ratio",
]

# Top-10 (baseline original)
TOP10_FEATURES = TOP7_FEATURES + ["onset_rate", "spectral_flatness_mean", "duration_s"]

# All 27 numeric features available in features_curated_splits.csv
ALL_FEATURE_COLS = [
    "duration_s", "rms", "rms_db", "peak", "zcr", "mel_total_energy",
    "centroid_mean", "centroid_std", "rolloff_mean", "rolloff_std",
    "spectral_flatness_mean", "spectral_flatness_std",
    "spectral_bandwidth_mean", "spectral_bandwidth_std",
    "spectral_entropy_mean", "spectral_contrast_mean",
    "mfcc_0_mean", "mfcc_1_mean", "chroma_mean_first", "chroma_mean",
    "chroma_std", "tonnetz_0_mean", "harmonic_percussive_ratio",
    "tempo", "onset_rate", "crest_factor", "wavelet_total_energy",
]


# ── Feature extraction (reutiliza logica de extract_baseline_features.py) ────

def spectral_entropy(S):
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


def extract_features(wav_path: str, sr: int = 44100) -> dict:
    """Extrae las 27 features baseline de un WAV."""
    y, sr = librosa.load(wav_path, sr=sr, mono=True)
    duration_s = len(y) / sr

    # Energy
    rms_val = float(np.mean(librosa.feature.rms(y=y)))
    rms_db = float(20 * np.log10(rms_val + 1e-12))
    peak_val = float(np.max(np.abs(y)))
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_total = float(np.mean(np.sum(mel, axis=0)))

    # Spectral
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    S = np.abs(librosa.stft(y, n_fft=2048))
    spec_ent = spectral_entropy(np.mean(S, axis=1))

    # Chroma / Tonnetz
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Temporal
    zcr_val = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_rate = len(onsets) / duration_s if duration_s > 0 else 0.0
    tempo_val = float(librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0])

    # Wavelet energy (simplified)
    try:
        import pywt
        coeffs = pywt.wavedec(y, 'db4', level=5)
        wavelet_e = float(sum(np.sum(c**2) for c in coeffs))
    except ImportError:
        wavelet_e = float(np.sum(y**2))

    return {
        "duration_s": duration_s,
        "rms": rms_val,
        "rms_db": rms_db,
        "peak": peak_val,
        "zcr": zcr_val,
        "mel_total_energy": mel_total,
        "centroid_mean": float(np.mean(centroid)),
        "centroid_std": float(np.std(centroid)),
        "rolloff_mean": float(np.mean(rolloff)),
        "rolloff_std": float(np.std(rolloff)),
        "spectral_flatness_mean": float(np.mean(flatness)),
        "spectral_flatness_std": float(np.std(flatness)),
        "spectral_bandwidth_mean": float(np.mean(bw)),
        "spectral_bandwidth_std": float(np.std(bw)),
        "spectral_entropy_mean": spec_ent,
        "spectral_contrast_mean": float(np.mean(contrast)),
        "mfcc_0_mean": float(np.mean(mfcc[0])),
        "mfcc_1_mean": float(np.mean(mfcc[1])),
        "chroma_mean_first": float(np.mean(chroma[0])),
        "chroma_mean": float(np.mean(chroma)),
        "chroma_std": float(np.std(chroma)),
        "tonnetz_0_mean": float(np.mean(tonnetz[0])),
        "harmonic_percussive_ratio": harmonic_percussive_ratio(y),
        "tempo": tempo_val,
        "onset_rate": onset_rate,
        "crest_factor": crest_factor(y),
        "wavelet_total_energy": wavelet_e,
    }


# ── SVM Training ─────────────────────────────────────────────────────────────

def build_binary_target(y_int, threshold):
    return (y_int >= threshold).astype(int)


def train_ordinal_svm(X_train, y_train_int, groups_train, n_jobs=4, rs=42):
    """Entrena 2 SVMs binarios con GridSearchCV + StratifiedGroupKFold."""
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=rs)
    param_grid = {
        "svc__C": [1.0, 10.0],
        "svc__gamma": ["scale", 0.1],
    }
    clfs = []
    for threshold in [1, 2]:
        y_bin = build_binary_target(y_train_int, threshold)
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("svc", SVC(probability=True, class_weight="balanced",
                        kernel="rbf", random_state=rs)),
        ])
        gs = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1",
                          n_jobs=n_jobs, verbose=0, refit=True)
        gs.fit(X_train, y_bin, groups=groups_train)
        clfs.append(gs.best_estimator_)
        print(f"    C{threshold}: best={gs.best_params_}, cv_f1={gs.best_score_:.3f}")
    return clfs


def predict_ordinal(clfs, X):
    p1 = clfs[0].predict_proba(X)[:, 1]
    p2 = clfs[1].predict_proba(X)[:, 1]
    p2 = np.minimum(p2, p1)  # monotonicity
    probs = np.stack([p1, p2], axis=1)
    return ordinal_decode(probs), probs


def evaluate(clfs, X, y_true_int, label=""):
    y_pred, probs = predict_ordinal(clfs, X)
    f1 = f1_score(y_true_int, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true_int, y_pred)
    mae = ordinal_mae(y_true_int, y_pred)
    adj = adjacent_accuracy(y_true_int, y_pred)
    if label:
        print(f"  [{label}] F1={f1:.3f}  Acc={acc:.3f}  AdjAcc={adj:.3f}  MAE={mae:.3f}")
    return {
        "macro_f1": f1, "accuracy": acc,
        "adjacent_accuracy": adj, "ordinal_mae": mae,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=7,
                        choices=[7, 10, 0],
                        help="Numero de features: 7=top-7, 10=top-10, 0=all-27")
    parser.add_argument("--thresholds", default="75,85,97",
                        help="Umbrales tau_des separados por coma (para reetiquetado)")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Saltar extraccion, usar features_cleaned.csv existente")
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # Seleccionar features
    if args.top_k == 7:
        feat_cols = TOP7_FEATURES
    elif args.top_k == 10:
        feat_cols = TOP10_FEATURES
    else:
        feat_cols = ALL_FEATURE_COLS

    print("=" * 70)
    print(f"  COMPARATIVA: Original vs Filtrado (noisereduce)")
    print(f"  Features: top-{args.top_k if args.top_k else 'all-27'} ({len(feat_cols)} features)")
    print("=" * 70)

    # ── 1. Cargar features originales ──────────────────────────────────────
    print("\n[1/4] Cargando features originales...")
    df_orig = pd.read_csv(FEATURES_ORIG, low_memory=False)
    # Solo originales (no augmentados) para comparacion justa
    df_orig = df_orig[df_orig["aug_type"] == "original"].copy()
    print(f"  {len(df_orig)} muestras originales")
    print(f"  Splits: {df_orig['split'].value_counts().to_dict()}")

    # ── 2. Extraer features de WAVs limpios ─────────────────────────────────
    features_clean_path = OUTDIR / "features_cleaned.csv"

    if args.skip_extraction and features_clean_path.exists():
        print(f"\n[2/4] Cargando features limpias existentes: {features_clean_path}")
        df_clean = pd.read_csv(features_clean_path, low_memory=False)
    else:
        print(f"\n[2/4] Extrayendo features de WAVs filtrados...")
        if not CLEAN_DIR.exists():
            print(f"ERROR: {CLEAN_DIR} no existe. Ejecuta apply_noise_filter_to_dataset.py primero")
            sys.exit(1)

        rows = []
        errors = []
        t0 = time.time()

        for i, (_, row) in enumerate(df_orig.iterrows()):
            src = row["filepath"]
            # Reconstruir path limpio
            rel = os.path.relpath(src, "D:/") if "D:/" in src.replace("\\", "/") \
                  else os.path.basename(src)
            clean_path = str(CLEAN_DIR / rel)

            if not os.path.exists(clean_path):
                errors.append(clean_path)
                rows.append(None)
                continue

            try:
                feats = extract_features(clean_path, SR)
                rows.append(feats)
            except Exception as e:
                print(f"  [ERROR] {clean_path}: {e}")
                errors.append(clean_path)
                rows.append(None)

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(df_orig) - i - 1) / rate / 60
                print(f"  {i+1}/{len(df_orig)}  {rate:.1f} files/s  ETA {eta:.1f}min")

        # Construir DataFrame limpio
        valid_mask = [r is not None for r in rows]
        df_clean = df_orig[valid_mask].copy().reset_index(drop=True)
        feat_df = pd.DataFrame([r for r in rows if r is not None])
        for col in feat_df.columns:
            df_clean[col] = feat_df[col].values

        df_clean.to_csv(features_clean_path, index=False)
        elapsed = time.time() - t0
        print(f"  Extraidos: {len(df_clean)}/{len(df_orig)} en {elapsed/60:.1f}min")
        print(f"  Errores: {len(errors)}")
        if errors:
            print(f"  Primeros errores: {errors[:5]}")
        print(f"  Guardado: {features_clean_path}")

    # ── 3. Entrenar y evaluar ambos modelos ─────────────────────────────────
    print(f"\n[3/4] Entrenamiento y evaluacion comparativa...")

    # Alinear DataFrames (solo filas presentes en ambos)
    common_fps = set(df_orig["filepath"]) & set(df_clean["filepath"])
    df_orig_aligned = df_orig[df_orig["filepath"].isin(common_fps)].sort_values("filepath").reset_index(drop=True)
    df_clean_aligned = df_clean[df_clean["filepath"].isin(common_fps)].sort_values("filepath").reset_index(drop=True)

    assert len(df_orig_aligned) == len(df_clean_aligned), "Mismatch en alignment"
    print(f"  Muestras alineadas: {len(df_orig_aligned)}")

    # Preparar labels y splits
    y_str = df_orig_aligned["label"].astype(str).str.strip()
    y_int = np.array([LABEL_TO_IDX[l] for l in y_str])
    split = df_orig_aligned["split"].values
    groups = df_orig_aligned["experiment"].fillna("unknown").values

    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"

    results = []

    for source_name, df_src in [("original", df_orig_aligned), ("filtered", df_clean_aligned)]:
        print(f"\n  --- {source_name.upper()} ---")

        # Verificar que feat_cols existen
        missing = [c for c in feat_cols if c not in df_src.columns]
        if missing:
            print(f"  WARN: features faltantes: {missing}")
            available = [c for c in feat_cols if c in df_src.columns]
        else:
            available = feat_cols

        X = df_src[available].values

        X_train = X[train_mask]
        y_train = y_int[train_mask]
        g_train = groups[train_mask]

        print(f"    Train: {X_train.shape[0]}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")

        clfs = train_ordinal_svm(X_train, y_train, g_train,
                                 n_jobs=args.n_jobs)

        # Evaluar en val y test
        for split_name, mask in [("val", val_mask), ("test", test_mask)]:
            if mask.sum() == 0:
                continue
            metrics = evaluate(clfs, X[mask], y_int[mask], f"{source_name}/{split_name}")
            metrics["source"] = source_name
            metrics["split"] = split_name
            metrics["n_features"] = len(available)
            metrics["n_samples"] = int(mask.sum())
            results.append(metrics)

        # Guardar modelos
        model_dir = OUTDIR / f"models_{source_name}"
        model_dir.mkdir(exist_ok=True)
        joblib.dump(clfs[0], model_dir / "svm_C1.joblib")
        joblib.dump(clfs[1], model_dir / "svm_C2.joblib")

    # ── 4. Tabla comparativa ────────────────────────────────────────────────
    print(f"\n[4/4] Generando tabla comparativa...")

    df_results = pd.DataFrame(results)
    results_path = OUTDIR / f"comparison_{ts}.csv"
    df_results.to_csv(results_path, index=False)

    # Tabla bonita
    print("\n" + "=" * 70)
    print("  RESULTADOS COMPARATIVOS")
    print("=" * 70)
    print(f"{'Source':<12} {'Split':<6} {'Macro F1':>9} {'Accuracy':>9} "
          f"{'Adj.Acc':>8} {'Ord.MAE':>8}")
    print("-" * 60)
    for _, row in df_results.iterrows():
        print(f"{row['source']:<12} {row['split']:<6} {row['macro_f1']:>9.3f} "
              f"{row['accuracy']:>9.3f} {row['adjacent_accuracy']:>8.3f} "
              f"{row['ordinal_mae']:>8.3f}")

    # Diferencias
    print("\n  DIFERENCIA (filtered - original):")
    for split_name in ["val", "test"]:
        orig_row = df_results[(df_results["source"] == "original") &
                              (df_results["split"] == split_name)]
        filt_row = df_results[(df_results["source"] == "filtered") &
                              (df_results["split"] == split_name)]
        if len(orig_row) and len(filt_row):
            for metric in ["macro_f1", "accuracy", "adjacent_accuracy"]:
                diff = filt_row[metric].values[0] - orig_row[metric].values[0]
                sign = "+" if diff >= 0 else ""
                print(f"    {split_name} {metric}: {sign}{diff:.3f}")

    print(f"\nResultados guardados: {results_path}")
    print(f"Modelos: {OUTDIR}/models_original/ y models_filtered/")


if __name__ == "__main__":
    main()
