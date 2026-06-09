#!/usr/bin/env python3
"""
bootstrap_metrics_article.py — Articulo 1

Bootstrap confidence intervals (95%) para metricas ordinales en umbrales clave.
Carga predicciones existentes o re-evalua modelos en test set.

Uso:
    python bootstrap_metrics_article.py
"""

import sys, os, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "D:/pipeline_tools_for_improvement/scripts")
from ordinal_utils import LABEL_TO_IDX, IDX_TO_LABEL, ordinal_mae, adjacent_accuracy
from sklearn.metrics import f1_score, accuracy_score

OUTDIR = Path("D:/pipeline_SVM/results/article1_figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 1000
RANDOM_STATE = 42
CI_LEVEL = 0.95


def bootstrap_ci(y_true, y_pred, n_boot=N_BOOTSTRAP, ci=CI_LEVEL, seed=RANDOM_STATE):
    """Computa bootstrap CIs para metricas ordinales."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    metrics_boot = {k: [] for k in ["macro_f1", "exact_acc", "adj_acc", "ordinal_mae"]}

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        yt, yp = y_true[idx], y_pred[idx]

        # Skip if degenerate sample
        if len(np.unique(yt)) < 2:
            continue

        metrics_boot["macro_f1"].append(f1_score(yt, yp, average='macro', zero_division=0))
        metrics_boot["exact_acc"].append(accuracy_score(yt, yp))
        metrics_boot["adj_acc"].append(adjacent_accuracy(yt, yp))
        metrics_boot["ordinal_mae"].append(ordinal_mae(yt, yp))

    alpha = (1 - ci) / 2
    result = {}
    for k, vals in metrics_boot.items():
        vals = np.array(vals)
        result[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "ci_lo": float(np.percentile(vals, 100 * alpha)),
            "ci_hi": float(np.percentile(vals, 100 * (1 - alpha))),
            "n_boot": len(vals),
        }
    return result


def main():
    print("=" * 60)
    print("  BOOTSTRAP CONFIDENCE INTERVALS — Articulo 1")
    print("=" * 60)

    # Cargar predicciones existentes
    results = {}

    # ── Umbral original (97%+) — Session 4 ────────────────────────────────
    pred_orig = Path("D:/pipeline_SVM/results/svm_ordinal_clean/predictions_test.csv")
    if pred_orig.exists():
        df = pd.read_csv(pred_orig)
        # Buscar columnas de true/pred
        true_col = next((c for c in ["true_idx", "true_label", "label"] if c in df.columns), None)
        pred_col = next((c for c in ["pred_idx", "pred_label", "prediction"] if c in df.columns), None)

        if true_col and pred_col:
            if df[true_col].dtype == object:
                y_true = np.array([LABEL_TO_IDX.get(str(l).strip(), -1) for l in df[true_col]])
            else:
                y_true = df[true_col].values.astype(int)

            if df[pred_col].dtype == object:
                y_pred = np.array([LABEL_TO_IDX.get(str(l).strip(), -1) for l in df[pred_col]])
            else:
                y_pred = df[pred_col].values.astype(int)

            valid = (y_true >= 0) & (y_pred >= 0)
            y_true, y_pred = y_true[valid], y_pred[valid]

            print(f"\nUmbral original (97%+): {len(y_true)} muestras test")
            ci = bootstrap_ci(y_true, y_pred)
            results["original_97pct"] = ci
            for k, v in ci.items():
                print(f"  {k}: {v['mean']:.3f} [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]")
    else:
        print(f"\nWARN: No se encontro {pred_orig}")

    # ── Umbral [15/75] — Session 7 ────────────────────────────────────────
    pred_75 = Path("D:/pipeline_SVM/results/svm_ordinal_v2/predictions_test.csv")
    if pred_75.exists():
        df = pd.read_csv(pred_75)
        true_col = next((c for c in ["true_idx", "true_label", "label"] if c in df.columns), None)
        pred_col = next((c for c in ["pred_idx", "pred_label", "prediction"] if c in df.columns), None)

        if true_col and pred_col:
            if df[true_col].dtype == object:
                y_true = np.array([LABEL_TO_IDX.get(str(l).strip(), -1) for l in df[true_col]])
            else:
                y_true = df[true_col].values.astype(int)

            if df[pred_col].dtype == object:
                y_pred = np.array([LABEL_TO_IDX.get(str(l).strip(), -1) for l in df[pred_col]])
            else:
                y_pred = df[pred_col].values.astype(int)

            valid = (y_true >= 0) & (y_pred >= 0)
            y_true, y_pred = y_true[valid], y_pred[valid]

            print(f"\nUmbral [15/75]: {len(y_true)} muestras test")
            ci = bootstrap_ci(y_true, y_pred)
            results["relabeled_75pct"] = ci
            for k, v in ci.items():
                print(f"  {k}: {v['mean']:.3f} [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]")

    # ── Nota: CIs para otros umbrales se calcularan en threshold_sensitivity ──
    # Este script es para los umbrales que YA tienen modelos entrenados.
    # Para la curva completa, threshold_sensitivity_article.py generara
    # bootstrap CIs inline si se desea.

    # Guardar
    out_path = OUTDIR / "bootstrap_ci.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
