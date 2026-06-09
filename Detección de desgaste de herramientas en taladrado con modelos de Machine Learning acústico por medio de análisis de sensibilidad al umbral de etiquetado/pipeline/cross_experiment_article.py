#!/usr/bin/env python3
"""
cross_experiment_article.py — Articulo 1

Leave-one-experiment-out: para cada Ei, entrena en el resto, evalua en Ei.
Genera heatmap de generalizacion 7x4 (experimento x metrica).

Uso:
    python cross_experiment_article.py
"""

import sys, os, warnings, time
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "D:/pipeline_tools_for_improvement/scripts")
from ordinal_utils import LABEL_TO_IDX, IDX_TO_LABEL, ordinal_decode, ordinal_mae, adjacent_accuracy

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score

from threshold_sensitivity_article import relabel_df, build_binary_target

FEATURES_CSV = Path("D:/pipeline_SVM/features/features_curated_splits.csv")
OUTDIR = Path("D:/pipeline_SVM/results/article1_figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

BASELINE_10 = [
    "harmonic_percussive_ratio", "centroid_mean", "zcr",
    "spectral_flatness_mean", "spectral_entropy_mean", "onset_rate",
    "duration_s", "crest_factor", "chroma_std", "spectral_contrast_mean",
]

# Usar umbral original (97%) para este analisis
THRESH_DES = 0.97
RANDOM_STATE = 42


def train_predict_svm(X_train, y_train, X_test):
    """Train ordinal SVM, return predictions."""
    clfs = []
    for threshold in [1, 2]:
        y_bin = build_binary_target(y_train, threshold)
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('svc', SVC(probability=True, class_weight='balanced',
                       C=10.0, kernel='rbf', gamma='scale', random_state=RANDOM_STATE)),
        ])
        pipe.fit(X_train, y_bin)
        clfs.append(pipe)

    p1 = clfs[0].predict_proba(X_test)[:, 1]
    p2 = clfs[1].predict_proba(X_test)[:, 1]
    p2 = np.minimum(p2, p1)
    return ordinal_decode(np.stack([p1, p2], axis=1))


def main():
    t0 = time.time()
    print("=" * 60)
    print("  CROSS-EXPERIMENT GENERALIZATION — Articulo 1")
    print("=" * 60)

    df = pd.read_csv(FEATURES_CSV)
    # Reetiquetar con umbral original
    df = relabel_df(df, THRESH_DES)
    # Solo originales
    df = df[df["aug_type"] == "original"]

    experiments = sorted(df["experiment"].unique())
    print(f"Experimentos: {experiments}")

    results = []

    for exp_test in experiments:
        mask_test = df["experiment"] == exp_test
        mask_train = ~mask_test

        df_train = df[mask_train]
        df_test = df[mask_test]

        if len(df_test) < 5:
            print(f"  SKIP {exp_test}: solo {len(df_test)} muestras")
            continue

        X_train = df_train[BASELINE_10].values
        X_test = df_test[BASELINE_10].values
        y_train = np.array([LABEL_TO_IDX[l] for l in df_train["label"]])
        y_test = np.array([LABEL_TO_IDX[l] for l in df_test["label"]])

        # Check class distribution
        classes_train = set(np.unique(y_train))
        classes_test = set(np.unique(y_test))

        y_pred = train_predict_svm(X_train, y_train, X_test)

        m = {
            "experiment": exp_test,
            "n_test": len(y_test),
            "n_train": len(y_train),
            "diameter": df_test["drill_diameter"].iloc[0],
            "mic_type": df_test["mic_type"].iloc[0],
            "classes_test": str(sorted(classes_test)),
            "macro_f1": f1_score(y_test, y_pred, average='macro', zero_division=0),
            "exact_acc": accuracy_score(y_test, y_pred),
            "adj_acc": adjacent_accuracy(y_test, y_pred),
            "ordinal_mae": ordinal_mae(y_test, y_pred),
        }
        results.append(m)
        print(f"  {exp_test} ({m['n_test']} test, {m['diameter']}mm, {m['mic_type']}): "
              f"F1={m['macro_f1']:.3f}, acc={m['exact_acc']:.3f}, adj={m['adj_acc']:.3f}")

    df_r = pd.DataFrame(results)
    csv_path = OUTDIR / "cross_experiment.csv"
    df_r.to_csv(csv_path, index=False)
    print(f"\nResultados: {csv_path}")

    # ── Heatmap ───────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        plt.rcParams.update({
            "font.family": "serif", "font.size": 10,
            "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        })

        metrics = ["macro_f1", "exact_acc", "adj_acc", "ordinal_mae"]
        metric_labels = ["Macro F1", "Exactitud", "Adj. Acc.", "MAE ordinal"]

        data = df_r.set_index("experiment")[metrics].values
        exps = df_r["experiment"].values
        extras = [f"({r['diameter']}mm, {r['mic_type'][:4]})" for _, r in df_r.iterrows()]
        ylabels = [f"{e} {x}" for e, x in zip(exps, extras)]

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metric_labels)
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels, fontsize=9)

        # Annotate
        for i in range(len(exps)):
            for j in range(len(metrics)):
                val = data[i, j]
                color = "white" if val < 0.4 or val > 0.85 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                       fontsize=9, color=color)

        ax.set_title("Generalización leave-one-experiment-out (umbral 97%)")
        fig.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()

        fig.savefig(OUTDIR / "cross_experiment_heatmap.pdf")
        fig.savefig(OUTDIR / "cross_experiment_heatmap.png")
        plt.close(fig)
        print(f"Heatmap: {OUTDIR / 'cross_experiment_heatmap.pdf'}")

    except Exception as e:
        print(f"Error en heatmap: {e}")

    print(f"\nTiempo: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
