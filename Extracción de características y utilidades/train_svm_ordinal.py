#!/usr/bin/env python3
# train_svm_ordinal.py
#
# SVM para clasificación ORDINAL de desgaste (Frank & Hall decomposition)
#
# Entrena 2 clasificadores binarios independientes:
#   C1: P(y >= 1)  → sin_desgaste  vs  (medianamente + desgastado)
#   C2: P(y >= 2)  → (sin + medianamente)  vs  desgastado
#
# La predicción final respeta la restricción ordinal P(C2) <= P(C1).
#
# Uso:
#   python train_svm_ordinal.py \
#       --input  D:/pipeline_SVM/features/merged_features_raw.csv \
#       --outdir D:/pipeline_SVM/results/svm_ordinal \
#       --group-col basename \
#       --holdout-frac 0.15 \
#       --n-jobs 6

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse, json, time, warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import classification_report, f1_score, accuracy_score

from ordinal_utils import (
    LABEL_TO_IDX, IDX_TO_LABEL, N_ORDINAL,
    ordinal_decode, ordinal_proba, print_ordinal_report,
    ordinal_mae, adjacent_accuracy
)

warnings.filterwarnings("ignore")

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_features(path: str, label_col_candidates=('label_fixed', 'label_clean', 'label')):
    df = pd.read_csv(path, low_memory=False)
    # detectar columna de label
    label_col = next((c for c in label_col_candidates if c in df.columns), None)
    if label_col is None:
        raise ValueError(f"No se encontró columna de label en {path}. "
                         f"Esperadas: {label_col_candidates}")
    # limpiar
    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    df = df[df[label_col].astype(str).str.strip().str.lower() != 'nan'].reset_index(drop=True)
    return df, label_col


def numeric_cols(df, exclude=None):
    _excl = {'filepath', 'basename', 'fp_norm', 'label_fixed', 'label_clean',
             'label', 'mic_type', 'experiment', 'is_augment', 'map_method', 'aug_type'}
    if exclude:
        _excl.update(exclude)
    return [c for c in df.columns
            if c not in _excl and np.issubdtype(df[c].dtype, np.number)]


def build_binary_target(y_int: np.ndarray, threshold: int) -> np.ndarray:
    """Crea target binario para umbral k: 1 si y >= threshold, 0 si no."""
    return (y_int >= threshold).astype(int)


def build_svm_pipeline(k_feat='all', C=10.0, kernel='rbf', random_state=42):
    steps = [
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ]
    if k_feat != 'all':
        steps.append(('select', SelectKBest(mutual_info_classif, k=k_feat)))
    steps.append(('svc', SVC(
        probability=True,
        class_weight='balanced',
        C=C,
        kernel=kernel,
        gamma='scale',
        random_state=random_state,
    )))
    return Pipeline(steps)


# ── Entrenamiento ordinal Frank & Hall ────────────────────────────────────────

def train_ordinal_svm(X_train, y_train_int, groups_train,
                      n_jobs=4, random_state=42):
    """
    Entrena 2 clasificadores binarios con GridSearchCV + StratifiedGroupKFold.
    Retorna lista [clf_C1, clf_C2].
    """
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)

    param_grid = [
        {
            'select': [SelectKBest(mutual_info_classif)],
            'select__k': [10, 20, 50],
            'svc__C': [1.0, 10.0, 50.0],
            'svc__kernel': ['rbf'],
            'svc__gamma': ['scale'],
        },
        {
            'select': ['passthrough'],
            'svc__C': [1.0, 10.0, 50.0],
            'svc__kernel': ['rbf', 'linear'],
            'svc__gamma': ['scale'],
        }
    ]

    clfs = []
    for threshold in [1, 2]:
        label_name = "P(y>=1) - hay desgaste" if threshold == 1 else "P(y>=2) - desgaste severo"
        print(f"\n--- Entrenando C{threshold}: {label_name} ---")

        y_bin = build_binary_target(y_train_int, threshold)
        n_pos = y_bin.sum()
        n_neg = len(y_bin) - n_pos
        print(f"   Positivos: {n_pos} | Negativos: {n_neg}")

        base_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('select', SelectKBest(mutual_info_classif)),
            ('svc', SVC(probability=True, class_weight='balanced', random_state=random_state)),
        ])

        gs = GridSearchCV(
            base_pipe, param_grid, cv=cv,
            scoring='f1', n_jobs=n_jobs,
            verbose=1, refit=True
        )
        gs.fit(X_train, y_bin, groups=groups_train)
        print(f"   Mejores parámetros: {gs.best_params_}")
        clfs.append(gs.best_estimator_)

    return clfs


def predict_ordinal(clfs, X):
    """
    Predice clases ordinales a partir de los 2 clasificadores.
    Aplica restricción de monotonicidad P(C2) <= P(C1).

    Returns:
        y_pred: array int (N,) con índices de clase
        probs:  array float (N, 2) con P(y>=1) y P(y>=2)
    """
    p1 = clfs[0].predict_proba(X)[:, 1]  # P(y >= 1)
    p2 = clfs[1].predict_proba(X)[:, 1]  # P(y >= 2)

    # Monotonicidad: P(y>=2) no puede ser mayor que P(y>=1)
    p2 = np.minimum(p2, p1)

    probs = np.stack([p1, p2], axis=1)
    y_pred = ordinal_decode(probs)
    return y_pred, probs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train ordinal SVM (Frank & Hall)")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--outdir", "-o", default="D:/pipeline_SVM/results/svm_ordinal")
    parser.add_argument("--group-col", default="basename")
    parser.add_argument("--holdout-frac", type=float, default=0.15)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    OUT = Path(args.outdir) / f"ordinal_run_{ts}"
    OUT.mkdir(parents=True, exist_ok=True)

    # ── Cargar datos ──────────────────────────────────────────────────────────
    print(f"Cargando features desde: {args.input}")
    df, label_col = load_features(args.input)
    print(f"  {len(df)} muestras | label_col='{label_col}'")

    # Convertir labels a índices ordinales
    y_str = df[label_col].astype(str).str.strip()
    if not set(y_str).issubset(LABEL_TO_IDX.keys()):
        unknown = set(y_str) - set(LABEL_TO_IDX.keys())
        raise ValueError(f"Labels desconocidas: {unknown}. Esperadas: {set(LABEL_TO_IDX.keys())}")
    y_int = np.array([LABEL_TO_IDX[l] for l in y_str])

    # Distribución de clases
    unique, counts = np.unique(y_int, return_counts=True)
    print("  Distribución:", {IDX_TO_LABEL[k]: v for k, v in zip(unique, counts)})

    # Grupos
    group_col = args.group_col if args.group_col in df.columns else 'basename'
    groups = df[group_col].astype(str).fillna('').values if group_col in df.columns \
        else np.array([str(i) for i in range(len(df))])

    feat_cols = numeric_cols(df)
    if not feat_cols:
        raise RuntimeError("No se encontraron columnas numéricas de features.")
    print(f"  Features numéricas: {len(feat_cols)}")

    X = df[feat_cols].values

    # ── Split holdout por grupo ───────────────────────────────────────────────
    gss = GroupShuffleSplit(n_splits=1, test_size=args.holdout_frac,
                            random_state=args.random_state)
    train_idx, hold_idx = next(gss.split(X, y_int, groups))
    X_train, X_hold = X[train_idx], X[hold_idx]
    y_train, y_hold = y_int[train_idx], y_int[hold_idx]
    g_train = groups[train_idx]

    print(f"\n  Train: {len(X_train)} | Holdout: {len(X_hold)}")
    for split_name, y_split in [("Train", y_train), ("Holdout", y_hold)]:
        u, c = np.unique(y_split, return_counts=True)
        print(f"  {split_name}: {dict(zip([IDX_TO_LABEL[k] for k in u], c))}")

    # ── Entrenar clasificadores ordinales ─────────────────────────────────────
    clfs = train_ordinal_svm(X_train, y_train, g_train,
                             n_jobs=args.n_jobs,
                             random_state=args.random_state)

    # ── Evaluar en holdout ────────────────────────────────────────────────────
    print("\n--- Evaluacion en holdout ---")
    y_pred, probs = predict_ordinal(clfs, X_hold)
    print_ordinal_report(y_hold, y_pred)

    # ── Guardar artefactos ────────────────────────────────────────────────────
    # modelos
    joblib.dump(clfs[0], OUT / "svm_ordinal_C1.joblib")  # P(y>=1)
    joblib.dump(clfs[1], OUT / "svm_ordinal_C2.joblib")  # P(y>=2)

    # predicciones holdout
    p_class = ordinal_proba(probs)
    df_hold = df.iloc[hold_idx].copy().reset_index(drop=True)
    df_hold['true_idx']              = y_hold
    df_hold['true_label']            = [IDX_TO_LABEL[i] for i in y_hold]
    df_hold['pred_idx']              = y_pred
    df_hold['pred_label']            = [IDX_TO_LABEL[i] for i in y_pred]
    df_hold['prob_sin_desgaste']     = p_class[:, 0]
    df_hold['prob_med_desgastado']   = p_class[:, 1]
    df_hold['prob_desgastado']       = p_class[:, 2]
    df_hold['ordinal_error']         = np.abs(y_hold - y_pred)
    df_hold.to_csv(OUT / "holdout_predictions.csv", index=False)

    # métricas
    metrics = {
        "timestamp": ts,
        "n_train": int(len(X_train)),
        "n_holdout": int(len(X_hold)),
        "n_features": len(feat_cols),
        "feature_names": feat_cols,
        "holdout_exact_accuracy": float(accuracy_score(y_hold, y_pred)),
        "holdout_macro_f1": float(f1_score(y_hold, y_pred, average='macro', zero_division=0)),
        "holdout_ordinal_mae": float(ordinal_mae(y_hold, y_pred)),
        "holdout_adjacent_accuracy": float(adjacent_accuracy(y_hold, y_pred)),
        "classification_report": classification_report(
            y_hold, y_pred,
            labels=[0, 1, 2],
            target_names=["sin_desgaste", "med_desgastado", "desgastado"],
            output_dict=True, zero_division=0
        )
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    print(f"\nArtefactos guardados en: {OUT}")
    print("  svm_ordinal_C1.joblib  - P(y>=1)")
    print("  svm_ordinal_C2.joblib  - P(y>=2)")
    print("  holdout_predictions.csv")
    print("  metrics.json")


if __name__ == "__main__":
    main()
