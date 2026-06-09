#!/usr/bin/env python3
"""
threshold_sensitivity_article.py — Articulo 1, Figura principal

Barre umbrales de etiquetado de desgastado (60%-97% de vida util) y entrena
SVM ordinal (Frank & Hall) + RF para cada uno. Genera:
  - threshold_sensitivity.csv: metricas por umbral
  - threshold_sensitivity_curve.pdf: figura principal del articulo

El umbral sin_desgaste se fija en 15%.
Solo se reetiquetan experimentos "Con falla" (E1-E4).
E5-E7 mantienen sus etiquetas acusticas originales.

Test = E3 completo (583 muestras, 6mm, linear, condensador).
Train = E1+E4+E5+E6+E7 (sin augmentados en eval, con augmentados en train).
Val = E2 (128 muestras, 8mm, dinamico).

Uso:
    python threshold_sensitivity_article.py
"""

import sys, os, json, warnings, time
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# Ordinal utils
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "D:/pipeline_tools_for_improvement/scripts")
from ordinal_utils import (
    LABEL_TO_IDX, IDX_TO_LABEL, ordinal_decode, ordinal_mae, adjacent_accuracy
)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# ── Config ────────────────────────────────────────────────────────────────────

FEATURES_CSV = Path("D:/pipeline_SVM/features/features_curated_splits.csv")
MASTER_CSV = Path("D:/dataset/manifests/master.csv")
OUTDIR = Path("D:/pipeline_SVM/results/article1_figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

THRESH_SIN = 0.15  # fijo
THRESH_DES_VALUES = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97]

CON_FALLA = {"E1", "E2", "E3", "E4"}

BASELINE_10 = [
    "harmonic_percussive_ratio", "centroid_mean", "zcr",
    "spectral_flatness_mean", "spectral_entropy_mean", "onset_rate",
    "duration_s", "crest_factor", "chroma_std", "spectral_contrast_mean",
]

RANDOM_STATE = 42
N_JOBS = 4

# ── Reetiquetado in-memory ────────────────────────────────────────────────────

def extract_hole_number(filepath: str) -> int:
    """Extrae numero de agujero del filepath (misma logica que relabel_by_tool_life.py)."""
    import re
    fp = str(filepath).replace("\\", "/")
    name = os.path.basename(fp)
    name_clean = re.sub(r"^limpio_", "", name, flags=re.IGNORECASE)

    m = re.search(r"B0\d(\d{3,4})", name_clean)
    if m: return int(m.group(1))

    m = re.search(r"broc[az]_?\d_?\d_?(\d{3,4})", name_clean, re.IGNORECASE)
    if m: return int(m.group(1))

    m = re.search(r"broca8_\d_(\d{3,4})", name_clean, re.IGNORECASE)
    if m: return int(m.group(1))

    m = re.search(r"(\d{3,4})\.wav$", name_clean, re.IGNORECASE)
    if m: return int(m.group(1))

    return -1


def relabel_df(df: pd.DataFrame, thresh_des: float) -> pd.DataFrame:
    """Reetiqueta in-memory con umbral dado. Retorna copia con labels actualizados."""
    df = df.copy()

    # Extraer hole numbers — use filepath for originals, orig_filepath for augmented
    # (orig_filepath is mostly NaN for originals, so fallback to filepath)
    if "orig_filepath" in df.columns:
        fp_series = df["orig_filepath"].fillna(df["filepath"])
    else:
        fp_series = df["filepath"]
    df["_hole"] = fp_series.apply(extract_hole_number)

    # Max hole por experimento (solo originales)
    orig = df[df["aug_type"] == "original"]
    max_holes = orig.groupby("experiment")["_hole"].max().to_dict()

    # Reetiquetar solo Con falla
    for exp in CON_FALLA:
        mask = df["experiment"] == exp
        max_h = max_holes.get(exp, 0)
        if max_h == 0:
            continue
        for idx in df[mask].index:
            h = df.at[idx, "_hole"]
            if h == -1:
                continue
            pct = h / max_h
            if pct <= THRESH_SIN:
                df.at[idx, "label"] = "sin_desgaste"
            elif pct >= thresh_des:
                df.at[idx, "label"] = "desgastado"
            else:
                df.at[idx, "label"] = "medianamente_desgastado"

    df.drop(columns=["_hole"], inplace=True)
    return df


# ── Training ──────────────────────────────────────────────────────────────────

def build_binary_target(y_int, threshold):
    return (y_int >= threshold).astype(int)


def train_svm_ordinal_fast(X_train, y_train_int, groups_train):
    """Entrena SVM ordinal con grid reducido para velocidad."""
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    clfs = []
    cv_f1s = []
    for threshold in [1, 2]:
        y_bin = build_binary_target(y_train_int, threshold)
        n_pos = y_bin.sum()
        if n_pos < 3:
            # Not enough positives — return dummy
            pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('svc', SVC(probability=True, class_weight='balanced',
                           C=1.0, kernel='rbf', random_state=RANDOM_STATE)),
            ])
            pipe.fit(X_train, y_bin)
            clfs.append(pipe)
            cv_f1s.append(0.0)
            continue

        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('svc', SVC(probability=True, class_weight='balanced',
                       random_state=RANDOM_STATE)),
        ])
        param_grid = {
            'svc__C': [1.0, 10.0],
            'svc__kernel': ['rbf'],
            'svc__gamma': ['scale', 0.1],
        }
        gs = GridSearchCV(pipe, param_grid, cv=cv, scoring='f1',
                         n_jobs=N_JOBS, verbose=0, refit=True)
        gs.fit(X_train, y_bin, groups=groups_train)
        clfs.append(gs.best_estimator_)
        cv_f1s.append(gs.best_score_)

    return clfs, cv_f1s


def train_rf_ordinal(X_train, y_train_int, groups_train):
    """Entrena RF ordinal (Frank & Hall) como comparacion."""
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    clfs = []
    for threshold in [1, 2]:
        y_bin = build_binary_target(y_train_int, threshold)
        n_pos = y_bin.sum()
        if n_pos < 3:
            clf = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                        random_state=RANDOM_STATE)
            clf.fit(X_train, y_bin)
            clfs.append(clf)
            continue

        clf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                    random_state=RANDOM_STATE, n_jobs=N_JOBS)
        clf.fit(X_train, y_bin)
        clfs.append(clf)
    return clfs


def predict_ordinal(clfs, X):
    """Prediccion ordinal con monotonicity enforcement."""
    p1 = clfs[0].predict_proba(X)[:, 1]
    p2 = clfs[1].predict_proba(X)[:, 1]
    p2 = np.minimum(p2, p1)  # monotonicity
    probs = np.stack([p1, p2], axis=1)
    y_pred = ordinal_decode(probs)
    return y_pred, probs


def evaluate(y_true_int, y_pred):
    """Computa metricas ordinales."""
    return {
        "exact_accuracy": float(accuracy_score(y_true_int, y_pred)),
        "macro_f1": float(f1_score(y_true_int, y_pred, average='macro', zero_division=0)),
        "ordinal_mae": float(ordinal_mae(y_true_int, y_pred)),
        "adjacent_accuracy": float(adjacent_accuracy(y_true_int, y_pred)),
    }


def per_class_f1(y_true, y_pred):
    """F1 por clase."""
    report = classification_report(y_true, y_pred, labels=[0, 1, 2],
                                   target_names=["sin", "med", "des"],
                                   output_dict=True, zero_division=0)
    return {
        "f1_sin": report["sin"]["f1-score"],
        "f1_med": report["med"]["f1-score"],
        "f1_des": report["des"]["f1-score"],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 70)
    print("  THRESHOLD SENSITIVITY ANALYSIS — Articulo 1")
    print("=" * 70)

    # Cargar features (ya tienen labels [15/75] del reetiquetado sesion 7)
    df = pd.read_csv(FEATURES_CSV)
    print(f"Features cargados: {len(df)} filas, {df.shape[1]} columnas")

    # Necesitamos re-etiquetar para cada umbral, asi que cargamos tambien el master
    # para tener los filepaths completos con orig_filepath
    # Las features ya tienen filepath, experiment, aug_type, split

    results = []

    for thresh_des in THRESH_DES_VALUES:
        print(f"\n{'-'*70}")
        print(f"  Umbral desgastado >= {thresh_des:.0%} vida util")
        print(f"{'-'*70}")

        # Reetiquetar
        df_t = relabel_df(df, thresh_des)

        # Splits: train = not test/val, test = E3, val = E2
        train_mask = df_t["split"] == "train"
        test_mask = df_t["split"] == "test"
        val_mask = df_t["split"] == "val"

        # No augmentados en test/val (ya deberia estar asi, pero verificar)
        test_mask = test_mask & (df_t["aug_type"] == "original")
        val_mask = val_mask & (df_t["aug_type"] == "original")

        df_train = df_t[train_mask]
        df_test = df_t[test_mask]
        df_val = df_t[val_mask]

        # Features
        X_train = df_train[BASELINE_10].values
        X_test = df_test[BASELINE_10].values
        X_val = df_val[BASELINE_10].values

        y_train = np.array([LABEL_TO_IDX[l] for l in df_train["label"]])
        y_test = np.array([LABEL_TO_IDX[l] for l in df_test["label"]])
        y_val = np.array([LABEL_TO_IDX[l] for l in df_val["label"]])

        groups_train = df_train["experiment"].values

        # Class distribution
        for name, y in [("train", y_train), ("test", y_test), ("val", y_val)]:
            u, c = np.unique(y, return_counts=True)
            dist = {IDX_TO_LABEL[k]: int(v) for k, v in zip(u, c)}
            print(f"  {name}: {dist}")

        n_des_train = int((y_train == 2).sum())
        n_des_test = int((y_test == 2).sum())

        # ── SVM ordinal ───────────────────────────────────────────────────────
        print(f"  Training SVM ordinal...")
        svm_clfs, svm_cv_f1s = train_svm_ordinal_fast(X_train, y_train, groups_train)
        y_pred_svm, _ = predict_ordinal(svm_clfs, X_test)
        y_pred_svm_val, _ = predict_ordinal(svm_clfs, X_val)

        m_svm_test = evaluate(y_test, y_pred_svm)
        m_svm_val = evaluate(y_val, y_pred_svm_val)
        f1_svm = per_class_f1(y_test, y_pred_svm)

        print(f"  SVM test: F1={m_svm_test['macro_f1']:.3f}, "
              f"acc={m_svm_test['exact_accuracy']:.3f}, "
              f"adj={m_svm_test['adjacent_accuracy']:.3f}")

        # ── RF ordinal ────────────────────────────────────────────────────────
        print(f"  Training RF ordinal...")
        rf_clfs = train_rf_ordinal(X_train, y_train, groups_train)
        y_pred_rf, _ = predict_ordinal(rf_clfs, X_test)
        y_pred_rf_val, _ = predict_ordinal(rf_clfs, X_val)

        m_rf_test = evaluate(y_test, y_pred_rf)
        m_rf_val = evaluate(y_val, y_pred_rf_val)
        f1_rf = per_class_f1(y_test, y_pred_rf)

        print(f"  RF  test: F1={m_rf_test['macro_f1']:.3f}, "
              f"acc={m_rf_test['exact_accuracy']:.3f}, "
              f"adj={m_rf_test['adjacent_accuracy']:.3f}")

        # ── Almacenar ─────────────────────────────────────────────────────────
        row = {
            "thresh_des_pct": thresh_des,
            "n_des_train": n_des_train,
            "n_des_test": n_des_test,
            # SVM test
            "svm_test_f1": m_svm_test["macro_f1"],
            "svm_test_acc": m_svm_test["exact_accuracy"],
            "svm_test_adj_acc": m_svm_test["adjacent_accuracy"],
            "svm_test_mae": m_svm_test["ordinal_mae"],
            "svm_test_f1_sin": f1_svm["f1_sin"],
            "svm_test_f1_med": f1_svm["f1_med"],
            "svm_test_f1_des": f1_svm["f1_des"],
            # SVM val
            "svm_val_f1": m_svm_val["macro_f1"],
            "svm_val_adj_acc": m_svm_val["adjacent_accuracy"],
            # SVM CV
            "svm_cv_f1_C1": svm_cv_f1s[0],
            "svm_cv_f1_C2": svm_cv_f1s[1],
            # RF test
            "rf_test_f1": m_rf_test["macro_f1"],
            "rf_test_acc": m_rf_test["exact_accuracy"],
            "rf_test_adj_acc": m_rf_test["adjacent_accuracy"],
            "rf_test_mae": m_rf_test["ordinal_mae"],
            "rf_test_f1_sin": f1_rf["f1_sin"],
            "rf_test_f1_med": f1_rf["f1_med"],
            "rf_test_f1_des": f1_rf["f1_des"],
            # RF val
            "rf_val_f1": m_rf_val["macro_f1"],
            "rf_val_adj_acc": m_rf_val["adjacent_accuracy"],
        }
        results.append(row)

    # ── Guardar CSV ───────────────────────────────────────────────────────────
    df_results = pd.DataFrame(results)
    csv_path = OUTDIR / "threshold_sensitivity.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nResultados guardados: {csv_path}")
    print(df_results.to_string(index=False))

    # ── Generar figura ────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        })

        fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
        x = df_results["thresh_des_pct"] * 100

        # Panel A: Macro F1
        ax = axes[0, 0]
        ax.plot(x, df_results["svm_test_f1"], "o-", color="#2196F3", label="SVM (test)", linewidth=2)
        ax.plot(x, df_results["rf_test_f1"], "s--", color="#FF9800", label="RF (test)", linewidth=2)
        ax.set_ylabel("Macro F1")
        ax.set_title("(a) Macro F1 vs umbral de desgaste")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        # Panel B: Adjacent accuracy
        ax = axes[0, 1]
        ax.plot(x, df_results["svm_test_adj_acc"], "o-", color="#2196F3", label="SVM", linewidth=2)
        ax.plot(x, df_results["rf_test_adj_acc"], "s--", color="#FF9800", label="RF", linewidth=2)
        ax.set_ylabel("Exactitud adyacente")
        ax.set_title(u"(b) Exactitud adyacente vs umbral")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.5, 1.05)

        # Panel C: F1 por clase (SVM)
        ax = axes[1, 0]
        ax.plot(x, df_results["svm_test_f1_sin"], "^-", color="#4CAF50", label="sin_desgaste", linewidth=1.5)
        ax.plot(x, df_results["svm_test_f1_med"], "o-", color="#FF9800", label="med_desgastado", linewidth=1.5)
        ax.plot(x, df_results["svm_test_f1_des"], "v-", color="#F44336", label="desgastado", linewidth=1.5)
        ax.set_xlabel("Umbral de desgaste (% vida util)")
        ax.set_ylabel("F1 por clase (SVM)")
        ax.set_title("(c) F1 por clase — SVM")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        # Panel D: n_desgastado en test + MAE
        ax = axes[1, 1]
        ax2 = ax.twinx()
        bars = ax.bar(x, df_results["n_des_test"], width=3, alpha=0.3, color="#9C27B0",
                      label="n_desgastado (test)")
        line1, = ax2.plot(x, df_results["svm_test_mae"], "o-", color="#2196F3",
                         label="MAE ordinal (SVM)", linewidth=2)
        line2, = ax2.plot(x, df_results["rf_test_mae"], "s--", color="#FF9800",
                         label="MAE ordinal (RF)", linewidth=2)
        ax.set_xlabel("Umbral de desgaste (% vida util)")
        ax.set_ylabel("n muestras desgastado (test)")
        ax2.set_ylabel("MAE ordinal")
        ax.set_title("(d) Muestras de desgaste y MAE ordinal")
        # Combined legend — fuera del area de plot, lado derecho
        handles = [bars, line1, line2]
        labels = [h.get_label() for h in handles]
        ax.legend(
            handles, labels,
            fontsize=7,
            loc="upper left",
            bbox_to_anchor=(0.6, 1.0),
            borderaxespad=0,
            frameon=True,
            framealpha=0.95,
        )
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = OUTDIR / "threshold_sensitivity_curve.pdf"
        fig.savefig(fig_path)
        fig.savefig(OUTDIR / "threshold_sensitivity_curve.png")
        plt.close(fig)
        print(f"\nFigura guardada: {fig_path}")

    except Exception as e:
        print(f"\nError generando figura: {e}")
        import traceback
        traceback.print_exc()

    elapsed = time.time() - t0
    print(f"\nTiempo total: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
