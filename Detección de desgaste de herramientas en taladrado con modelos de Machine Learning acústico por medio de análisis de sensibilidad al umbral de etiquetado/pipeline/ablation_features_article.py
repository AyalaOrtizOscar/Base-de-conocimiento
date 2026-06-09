#!/usr/bin/env python3
"""
ablation_features_article.py — Articulo 1

Ablation study: evalua subsets incrementales de features (top-3, 5, 7, 10)
y leave-one-out para cada feature del top-10.
Corre en umbrales clave: 75%, 85%, 97%.

Uso:
    python ablation_features_article.py
"""

import sys, os, warnings, json, time
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
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score

# Reuse relabeling from threshold script
from threshold_sensitivity_article import relabel_df, build_binary_target

FEATURES_CSV = Path("D:/pipeline_SVM/features/features_curated_splits.csv")
OUTDIR = Path("D:/pipeline_SVM/results/article1_figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Features ordered by MI ranking (session 7)
BASELINE_10_RANKED = [
    "spectral_contrast_mean",   # 1 (original top)
    "crest_factor",             # 2
    "chroma_std",               # 3
    "zcr",                      # 4
    "spectral_entropy_mean",    # 5
    "centroid_mean",            # 6
    "harmonic_percussive_ratio",# 7
    "onset_rate",               # 8
    "spectral_flatness_mean",   # 9
    "duration_s",               # 10
]

THRESH_KEY = [0.75, 0.85, 0.97]
RANDOM_STATE = 42
N_JOBS = 4


def train_eval_svm(X_train, y_train, groups, X_test, y_test):
    """Train SVM ordinal fast, return test macro F1."""
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    clfs = []
    for threshold in [1, 2]:
        y_bin = build_binary_target(y_train, threshold)
        n_pos = y_bin.sum()
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('svc', SVC(probability=True, class_weight='balanced',
                       C=10.0, kernel='rbf', gamma='scale', random_state=RANDOM_STATE)),
        ])
        if n_pos < 3:
            pipe.fit(X_train, y_bin)
        else:
            pipe.fit(X_train, y_bin)
        clfs.append(pipe)

    # Predict ordinal
    p1 = clfs[0].predict_proba(X_test)[:, 1]
    p2 = clfs[1].predict_proba(X_test)[:, 1]
    p2 = np.minimum(p2, p1)
    probs = np.stack([p1, p2], axis=1)
    y_pred = ordinal_decode(probs)

    return {
        "macro_f1": f1_score(y_test, y_pred, average='macro', zero_division=0),
        "exact_acc": accuracy_score(y_test, y_pred),
        "adj_acc": adjacent_accuracy(y_test, y_pred),
    }


def main():
    t0 = time.time()
    print("=" * 60)
    print("  ABLATION STUDY — Articulo 1")
    print("=" * 60)

    df = pd.read_csv(FEATURES_CSV)
    results = []

    for thresh in THRESH_KEY:
        print(f"\n{'─'*60}")
        print(f"  Umbral desgastado >= {thresh:.0%}")
        print(f"{'─'*60}")

        df_t = relabel_df(df, thresh)
        df_train = df_t[df_t["split"] == "train"]
        df_test = df_t[(df_t["split"] == "test") & (df_t["aug_type"] == "original")]

        y_train = np.array([LABEL_TO_IDX[l] for l in df_train["label"]])
        y_test = np.array([LABEL_TO_IDX[l] for l in df_test["label"]])
        groups = df_train["experiment"].values

        # ── Incremental subsets ───────────────────────────────────────────
        for k in [3, 5, 7, 10]:
            feats = BASELINE_10_RANKED[:k]
            X_tr = df_train[feats].values
            X_te = df_test[feats].values

            m = train_eval_svm(X_tr, y_train, groups, X_te, y_test)
            print(f"  Top-{k}: F1={m['macro_f1']:.3f}, acc={m['exact_acc']:.3f}")
            results.append({
                "threshold": thresh,
                "ablation_type": "incremental",
                "n_features": k,
                "features": ",".join(feats),
                **m,
            })

        # ── Leave-one-out ─────────────────────────────────────────────────
        print(f"\n  Leave-one-out (removing each from top-10):")
        for i, feat_out in enumerate(BASELINE_10_RANKED):
            feats = [f for f in BASELINE_10_RANKED if f != feat_out]
            X_tr = df_train[feats].values
            X_te = df_test[feats].values

            m = train_eval_svm(X_tr, y_train, groups, X_te, y_test)
            print(f"    -{feat_out}: F1={m['macro_f1']:.3f}")
            results.append({
                "threshold": thresh,
                "ablation_type": "leave_one_out",
                "n_features": 9,
                "removed_feature": feat_out,
                "features": ",".join(feats),
                **m,
            })

    # Guardar
    df_r = pd.DataFrame(results)
    csv_path = OUTDIR / "ablation_results.csv"
    df_r.to_csv(csv_path, index=False)
    print(f"\nResultados: {csv_path}")

    # ── Figura ────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            "font.family": "serif", "font.size": 10,
            "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        })

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Panel A: Incremental
        ax = axes[0]
        for thresh in THRESH_KEY:
            sub = df_r[(df_r["threshold"] == thresh) & (df_r["ablation_type"] == "incremental")]
            ax.plot(sub["n_features"], sub["macro_f1"], "o-",
                   label=f"Umbral {thresh:.0%}", linewidth=2)
        ax.set_xlabel("Numero de features (top-k por MI)")
        ax.set_ylabel("Macro F1 (test)")
        ax.set_title("(a) F1 vs numero de features")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([3, 5, 7, 10])

        # Panel B: LOO at 97% threshold
        ax = axes[1]
        loo = df_r[(df_r["threshold"] == 0.97) & (df_r["ablation_type"] == "leave_one_out")]
        if len(loo) > 0:
            baseline_f1 = df_r[(df_r["threshold"] == 0.97) &
                              (df_r["ablation_type"] == "incremental") &
                              (df_r["n_features"] == 10)]["macro_f1"].values[0]
            delta = loo["macro_f1"].values - baseline_f1
            names = loo["removed_feature"].values
            colors = ["#F44336" if d < 0 else "#4CAF50" for d in delta]
            ax.barh(range(len(names)), delta, color=colors, alpha=0.7)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=8)
            ax.axvline(0, color="black", linewidth=0.5)
            ax.set_xlabel("Cambio en Macro F1 al remover feature")
            ax.set_title("(b) Leave-one-out (umbral 97%)")
            ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        fig.savefig(OUTDIR / "ablation_curve.pdf")
        fig.savefig(OUTDIR / "ablation_curve.png")
        plt.close(fig)
        print(f"Figura: {OUTDIR / 'ablation_curve.pdf'}")

    except Exception as e:
        print(f"Error en figura: {e}")

    print(f"\nTiempo: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
