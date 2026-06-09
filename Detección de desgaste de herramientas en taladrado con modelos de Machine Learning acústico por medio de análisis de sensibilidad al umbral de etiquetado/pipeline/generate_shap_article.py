#!/usr/bin/env python3
"""
generate_shap_article.py — Articulo 1, SHAP Beeswarm

Genera SHAP beeswarm plots para el mejor modelo SVM ordinal (umbral original).
Usa KernelExplainer (SVM no es tree-based).

Uso:
    python generate_shap_article.py
"""

import sys, os, warnings, json
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "D:/pipeline_tools_for_improvement/scripts")

import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from ordinal_utils import LABEL_TO_IDX

# ── Config ────────────────────────────────────────────────────────────────────

FEATURES_CSV = Path("D:/pipeline_SVM/features/features_curated_splits.csv")
OUTDIR = Path("D:/pipeline_SVM/results/article1_figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Modelos del umbral original (session 4)
MODEL_C1 = Path("D:/pipeline_SVM/results/svm_ordinal_clean/svm_C1.joblib")
MODEL_C2 = Path("D:/pipeline_SVM/results/svm_ordinal_clean/svm_C2.joblib")

# Si no existen, usar los de v2 (reetiquetado)
if not MODEL_C1.exists():
    MODEL_C1 = Path("D:/pipeline_SVM/results/svm_ordinal_v2/best_model.joblib")
    MODEL_C2 = None  # best_model puede ser un solo objeto
    print("WARN: Usando modelo reetiquetado (v2) — no se encontro svm_ordinal_clean")

BASELINE_10 = [
    "harmonic_percussive_ratio", "centroid_mean", "zcr",
    "spectral_flatness_mean", "spectral_entropy_mean", "onset_rate",
    "duration_s", "crest_factor", "chroma_std", "spectral_contrast_mean",
]

# The session-4 models were trained on ALL 26 numeric features (with SelectKBest inside)
# We must pass the same 26 features in the same order
META_COLS = {'filepath', 'mel_path', 'orig_filepath', 'label', 'split', 'aug_type',
             'mic_type', 'drill_group', 'experiment', 'drill_diameter', 'coolant',
             'gcode_pattern', 'drill_series', 'fixture', 'basename',
             'group', 'label_original', 'is_original', 'orig_experiment'}

N_BACKGROUND = 200
N_EVAL = 300
RANDOM_STATE = 42


def main():
    print("=" * 60)
    print("  SHAP BEESWARM — Articulo 1")
    print("=" * 60)

    # Cargar datos
    df = pd.read_csv(FEATURES_CSV)

    # Usar test set para SHAP evaluation
    df_test = df[(df["split"] == "test") & (df["aug_type"] == "original")]
    df_train = df[df["split"] == "train"]

    # The session-4 model was trained on these exact 26 features (no spectral_entropy_mean)
    feat_cols = [
        "duration_s", "rms", "rms_db", "peak", "zcr", "mel_total_energy",
        "centroid_mean", "centroid_std", "rolloff_mean", "rolloff_std",
        "spectral_flatness_mean", "spectral_flatness_std",
        "spectral_bandwidth_mean", "spectral_bandwidth_std",
        "spectral_contrast_mean", "mfcc_0_mean", "mfcc_1_mean",
        "chroma_mean_first", "chroma_mean", "chroma_std", "tonnetz_0_mean",
        "harmonic_percussive_ratio", "tempo", "onset_rate", "crest_factor",
        "wavelet_total_energy",
    ]
    print(f"Using {len(feat_cols)} features")

    X_test = df_test[feat_cols].values
    X_train = df_train[feat_cols].values

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Background sample
    rng = np.random.RandomState(RANDOM_STATE)
    bg_idx = rng.choice(len(X_train), size=min(N_BACKGROUND, len(X_train)), replace=False)
    X_bg = X_train[bg_idx]

    # Eval sample
    eval_idx = rng.choice(len(X_test), size=min(N_EVAL, len(X_test)), replace=False)
    X_eval = X_test[eval_idx]

    # Models are Pipelines (imputer+scaler+select+svc) — pass raw data
    # SHAP will call predict_proba on raw features directly
    X_bg_proc = X_bg
    X_eval_proc = X_eval

    feature_names_es_map = {
        "harmonic_percussive_ratio": "Ratio arm./perc.", "centroid_mean": "Centroide",
        "zcr": "ZCR", "spectral_flatness_mean": "Planitud espectral",
        "spectral_entropy_mean": "Entropía espectral", "onset_rate": "Tasa de onsets",
        "duration_s": "Duración", "crest_factor": "Factor de cresta",
        "chroma_std": "Chroma std", "spectral_contrast_mean": "Contraste espectral",
        "rms": "RMS", "rms_db": "RMS (dB)", "peak": "Pico", "centroid_std": "Centroide std",
        "rolloff_mean": "Rolloff", "rolloff_std": "Rolloff std",
        "spectral_flatness_std": "Planitud std", "spectral_bandwidth_mean": "Ancho banda",
        "spectral_bandwidth_std": "Ancho banda std", "mfcc_0_mean": "MFCC-0",
        "mfcc_1_mean": "MFCC-1", "chroma_mean_first": "Chroma 1st", "chroma_mean": "Chroma media",
        "tonnetz_0_mean": "Tonnetz-0", "tempo": "Tempo", "mel_total_energy": "Energía Mel",
        "wavelet_total_energy": "Energía Wavelet",
    }
    feature_names_es = [feature_names_es_map.get(f, f) for f in feat_cols]

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    # ── SHAP para C1: P(y>=1) ─────────────────────────────────────────────
    if MODEL_C1.exists():
        print("\nCargando modelo C1...")
        clf_c1 = joblib.load(MODEL_C1)

        # Si es un Pipeline, extraer el SVC
        if hasattr(clf_c1, 'predict_proba'):
            predict_fn = lambda x: clf_c1.predict_proba(x)[:, 1]
        else:
            print("WARN: Modelo C1 no tiene predict_proba")
            return

        print(f"Calculando SHAP (KernelExplainer, {N_BACKGROUND} bg, {N_EVAL} eval)...")
        explainer = shap.KernelExplainer(predict_fn, X_bg_proc)
        shap_values = explainer.shap_values(X_eval_proc, nsamples=100)

        # Beeswarm plot
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.summary_plot(shap_values, X_eval_proc,
                         feature_names=feature_names_es,
                         show=False, plot_size=None)
        plt.title("SHAP — P(desgaste ≥ 1): ¿Hay algún desgaste?", fontsize=11)
        plt.tight_layout()
        fig_path = OUTDIR / "shap_beeswarm_C1.pdf"
        plt.savefig(fig_path)
        plt.savefig(OUTDIR / "shap_beeswarm_C1.png")
        plt.close()
        print(f"Guardado: {fig_path}")

        # Guardar SHAP values como CSV
        shap_df = pd.DataFrame(shap_values, columns=feat_cols)
        shap_mean = shap_df.abs().mean().sort_values(ascending=False)
        shap_mean.to_csv(OUTDIR / "shap_mean_abs_C1.csv")
        print(f"SHAP mean |values| guardado")

    # ── SHAP para C2: P(y>=2) ─────────────────────────────────────────────
    if MODEL_C2 is not None and MODEL_C2.exists():
        print("\nCargando modelo C2...")
        clf_c2 = joblib.load(MODEL_C2)

        if hasattr(clf_c2, 'predict_proba'):
            predict_fn = lambda x: clf_c2.predict_proba(x)[:, 1]
        else:
            print("WARN: Modelo C2 no tiene predict_proba")
            return

        print(f"Calculando SHAP C2...")
        explainer = shap.KernelExplainer(predict_fn, X_bg_proc)
        shap_values = explainer.shap_values(X_eval_proc, nsamples=100)

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.summary_plot(shap_values, X_eval_proc,
                         feature_names=feature_names_es,
                         show=False, plot_size=None)
        plt.title("SHAP — P(desgaste ≥ 2): ¿Desgaste severo?", fontsize=11)
        plt.tight_layout()
        fig_path = OUTDIR / "shap_beeswarm_C2.pdf"
        plt.savefig(fig_path)
        plt.savefig(OUTDIR / "shap_beeswarm_C2.png")
        plt.close()
        print(f"Guardado: {fig_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
