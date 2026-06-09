#!/usr/bin/env python3
"""
format_figures_article.py — Articulo 1

Regenera figuras existentes (PCA, confusion matrices, MI ranking) con estilo
publication-quality para RCTA. Tambien genera la comparativa side-by-side
de confusion matrices para 3 umbrales.

Uso:
    python format_figures_article.py
"""

import sys, os, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "D:/pipeline_tools_for_improvement/scripts")
from ordinal_utils import LABEL_TO_IDX, IDX_TO_LABEL

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix

FEATURES_CSV = Path("D:/pipeline_SVM/features/features_curated_splits.csv")
OUTDIR = Path("D:/pipeline_SVM/results/article1_figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

BASELINE_10 = [
    "harmonic_percussive_ratio", "centroid_mean", "zcr",
    "spectral_flatness_mean", "spectral_entropy_mean", "onset_rate",
    "duration_s", "crest_factor", "chroma_std", "spectral_contrast_mean",
]

CLASS_COLORS = {"sin_desgaste": "#4CAF50", "medianamente_desgastado": "#FF9800", "desgastado": "#F44336"}
CLASS_LABELS_ES = {"sin_desgaste": "Sin desgaste", "medianamente_desgastado": "Med. desgastado", "desgastado": "Desgastado"}

RCPARAMS = {
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def main():
    plt.rcParams.update(RCPARAMS)

    df = pd.read_csv(FEATURES_CSV)
    df_orig = df[df["aug_type"] == "original"]

    print("=" * 60)
    print("  PUBLICATION FIGURES — Articulo 1")
    print("=" * 60)

    # ── Fig 4: PCA 2D ─────────────────────────────────────────────────────
    print("\nGenerando PCA 2D...")
    df_test = df_orig[df_orig["split"] == "test"]
    X = df_test[BASELINE_10].values
    labels = df_test["label"].values

    imp = SimpleImputer(strategy='median')
    sc = StandardScaler()
    X_proc = sc.fit_transform(imp.fit_transform(X))
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_proc)

    fig, ax = plt.subplots(figsize=(7, 5))
    for cls in ["sin_desgaste", "medianamente_desgastado", "desgastado"]:
        mask = labels == cls
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=CLASS_COLORS[cls],
                  label=CLASS_LABELS_ES[cls], alpha=0.6, s=20, edgecolors='none')
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} varianza)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} varianza)")
    ax.set_title("PCA — 10 features acústicas (test set E3)")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.savefig(OUTDIR / "pca_2d.pdf")
    fig.savefig(OUTDIR / "pca_2d.png")
    plt.close(fig)
    print(f"  {OUTDIR / 'pca_2d.pdf'}")

    # ── Fig 7: MI ranking ─────────────────────────────────────────────────
    print("Generando MI ranking...")
    mi_csv = Path("D:/pipeline_SVM/results/svm_ordinal_v2/mutual_information.csv")
    if mi_csv.exists():
        mi = pd.read_csv(mi_csv)
        # Determine column names
        feat_col = mi.columns[0]
        mi_col = mi.columns[1]
        mi = mi.sort_values(mi_col, ascending=True).tail(15)

        feature_names_es = {
            "spectral_bandwidth_std": "Ancho banda (std)",
            "rolloff_std": "Rolloff (std)",
            "spectral_bandwidth_mean": "Ancho banda (media)",
            "mfcc_1_mean": "MFCC-1",
            "zcr": "ZCR",
            "mfcc_0_mean": "MFCC-0",
            "rms": "RMS",
            "rms_db": "RMS (dB)",
            "centroid_mean": "Centroide",
            "spectral_contrast_mean": "Contraste espectral",
            "harmonic_percussive_ratio": "Ratio arm./perc.",
            "crest_factor": "Factor de cresta",
            "chroma_std": "Chroma (std)",
            "spectral_flatness_mean": "Planitud espectral",
            "spectral_entropy_mean": "Entropía espectral",
            "onset_rate": "Tasa de onsets",
            "duration_s": "Duración",
        }

        labels = [feature_names_es.get(f, f) for f in mi[feat_col]]

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["#2196F3" if f in BASELINE_10 else "#90CAF9" for f in mi[feat_col]]
        ax.barh(range(len(mi)), mi[mi_col].values, color=colors)
        ax.set_yticks(range(len(mi)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Información Mutua")
        ax.set_title("Ranking de features por Información Mutua (reetiquetado [15/75])")
        # Legend for baseline vs non-baseline
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor="#2196F3", label="Baseline 10"),
                          Patch(facecolor="#90CAF9", label="Otras features")]
        ax.legend(handles=legend_elements, fontsize=8)
        ax.grid(True, alpha=0.2, axis='x')
        fig.savefig(OUTDIR / "mi_ranking.pdf")
        fig.savefig(OUTDIR / "mi_ranking.png")
        plt.close(fig)
        print(f"  {OUTDIR / 'mi_ranking.pdf'}")

    # ── Fig: Distribucion de clases por experimento ───────────────────────
    print("Generando distribucion por experimento...")
    ct = pd.crosstab(df_orig["experiment"], df_orig["label"])
    ct = ct.reindex(columns=["sin_desgaste", "medianamente_desgastado", "desgastado"], fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ct.plot(kind="bar", stacked=True, ax=ax,
           color=[CLASS_COLORS[c] for c in ct.columns])
    ax.set_xlabel("Experimento")
    ax.set_ylabel("Número de muestras")
    ax.set_title("Distribución de clases por experimento (reetiquetado [15/75])")
    ax.legend([CLASS_LABELS_ES[c] for c in ct.columns], fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    fig.savefig(OUTDIR / "class_distribution.pdf")
    fig.savefig(OUTDIR / "class_distribution.png")
    plt.close(fig)
    print(f"  {OUTDIR / 'class_distribution.pdf'}")

    print("\nDone! Todas las figuras en:", OUTDIR)


if __name__ == "__main__":
    main()
