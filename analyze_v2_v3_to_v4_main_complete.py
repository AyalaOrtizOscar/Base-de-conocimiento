#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_v2_v3_to_v4_main_complete_fixed_shap.py

Versión con manejo robusto de SHAP (evita ValueError "Per-column arrays must each be 1-dimensional").
"""
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import traceback
import matplotlib.pyplot as plt
import sys
import unicodedata
import warnings
warnings.filterwarnings("ignore")

# Stats / tests
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

# ML
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix

# Optional SHAP
try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

# -----------------------
# CONFIG (rutas) - ajusta si hace falta
# -----------------------
BASE = Path(r"E:\Ensayos Orejarena")
V2 = BASE / "v2"
V3 = BASE / "v3"
V4_RESULTS = BASE / "v4" / "results"
V4_RESULTS.mkdir(parents=True, exist_ok=True)
MERGED_CSV = V4_RESULTS / "merged_features_raw.csv"
MERGED_ARRAYS_CSV = BASE / "merged_array_columns.csv"  # opcional

# Random / model params
RANDOM_STATE = 42
RF_PARAMS = {"n_estimators": 300, "random_state": RANDOM_STATE, "n_jobs": -1, "class_weight": "balanced"}
MAX_NAN_DROP = 0.90  # drop columns with >90% NaN
MIN_SAMPLES_FOR_ANOVA = 8  # umbral más bajo para ANOVA si es necesario
SHAP_SAMPLE_MAX = 1000  # máximo muestras para explicar con SHAP (reduce memoria)

# RESULTS_DIR: donde se guardan outputs
RESULTS_DIR = V4_RESULTS
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("analyze-main")

# -----------------------
# Utilities
# -----------------------
def normalize_text_label(s):
    if pd.isna(s):
        return s
    try:
        s = str(s).strip().lower()
        s = unicodedata.normalize('NFKD', s)
        s = ''.join(ch for ch in s if not unicodedata.combining(ch))
        s = " ".join(s.split())
        return s
    except Exception:
        return str(s).strip().lower()

def find_features_csvs(root: Path):
    return sorted(root.rglob("features_per_file.csv"))

def load_and_label(files, label):
    dfs=[]
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            df['source'] = label
            if 'filepath' not in df.columns and 'filename' in df.columns:
                df['filepath'] = str(f.parent / df['filename'].astype(str))
            dfs.append(df)
            print(f"[auto-merge] cargado {len(df)} filas desde {f}")
        except Exception as e:
            print("[auto-merge] Error loading", f, e)
    return dfs

def auto_merge_if_missing():
    if MERGED_CSV.exists():
        print(f"[auto-merge] merged CSV ya existe: {MERGED_CSV}")
        return
    print("[auto-merge] merged CSV no encontrado. Buscando features_per_file.csv en v2/v3 ...")
    files_v2 = find_features_csvs(V2)
    files_v3 = find_features_csvs(V3)
    print(f"[auto-merge] encontrados {len(files_v2)} archivos en v2 y {len(files_v3)} en v3")
    all_dfs = []
    all_dfs += load_and_label(files_v2, "v2")
    all_dfs += load_and_label(files_v3, "v3")
    if len(all_dfs) == 0:
        print("[auto-merge] No se encontraron CSVs en v2/v3. Abortando.")
        return
    merged = pd.concat(all_dfs, ignore_index=True, sort=False)
    if 'filepath' in merged.columns:
        merged = merged.drop_duplicates(subset=['filepath'])
    elif 'filename' in merged.columns:
        merged = merged.drop_duplicates(subset=['filename'])
    merged.to_csv(MERGED_CSV, index=False)
    try:
        merged.head(5).to_csv(V4_RESULTS / "merged_preview_head.csv", index=False)
        print("[auto-merge] merged CSV creado y preview guardado.")
    except Exception:
        print("[auto-merge] merged CSV creado.")

def safe_save(df, fname):
    out = RESULTS_DIR / fname
    df.to_csv(out, index=False)
    log.info(f"Guardado {out}")

def detect_id_column(df):
    candidates = ["filepath","file","id","filename"]
    for c in candidates:
        if c in df.columns:
            return c
    nunique = df.nunique(dropna=True)
    cand = nunique[nunique >= 0.9 * len(df)].index.tolist()
    return cand[0] if cand else None

def detect_array_like_columns_by_value(df, sample_n=500):
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    arr_cols = []
    for c in obj_cols:
        s = df[c].dropna().head(sample_n)
        if s.apply(lambda v: isinstance(v, (list, tuple, np.ndarray))).any():
            arr_cols.append(c)
        else:
            if s.apply(lambda v: isinstance(v, str) and v.strip().startswith('[') and v.strip().endswith(']')).any():
                arr_cols.append(c)
    return arr_cols

def safe_parse_possible_list_str(v):
    if isinstance(v, str) and v.strip().startswith('[') and v.strip().endswith(']'):
        try:
            s = v.strip()[1:-1].strip()
            if s == "":
                return []
            parts = [p.strip() for p in s.split(',')]
            return [float(p) for p in parts]
        except Exception:
            return v
    return v

def summarize_array_cols(df, arr_cols):
    for c in arr_cols:
        def safe_len(v):
            try:
                if isinstance(v, str):
                    v2 = safe_parse_possible_list_str(v)
                    return len(v2) if isinstance(v2, (list, tuple, np.ndarray)) else np.nan
                return len(v)
            except Exception:
                return np.nan
        def safe_mean(v):
            try:
                if isinstance(v, str):
                    v2 = safe_parse_possible_list_str(v)
                    if isinstance(v2, (list, tuple, np.ndarray)):
                        a = np.array(v2, dtype=float)
                        return float(np.nanmean(a)) if a.size>0 else np.nan
                    return np.nan
                a = np.array(v)
                if a.size == 0: return np.nan
                return float(np.nanmean(a.astype(float)))
            except Exception:
                return np.nan
        df[c + "_len"] = df[c].apply(safe_len)
        df[c + "_mean"] = df[c].apply(safe_mean)
    df.drop(columns=arr_cols, inplace=True, errors='ignore')
    return df

def coerce_numeric_and_report(df, feature_cols):
    not_convertible = []
    for c in feature_cols:
        coerced = pd.to_numeric(df[c], errors='coerce')
        n_coerced_na = coerced.isna().sum()
        if n_coerced_na > 0.95 * len(coerced):
            not_convertible.append(c)
        df[c] = coerced
    return not_convertible

def quick_column_summary(df):
    rows = []
    for c in df.columns:
        ser = df[c]
        rows.append({
            "column": c,
            "dtype": str(ser.dtype),
            "n_null": int(ser.isna().sum()),
            "pct_null": float(ser.isna().mean()),
            "n_unique": int(ser.nunique(dropna=True))
        })
    return pd.DataFrame(rows)

def save_plot_corr_matrix(df, features, path):
    if len(features) == 0:
        return
    corr = df[features].corr()
    plt.figure(figsize=(9,8))
    plt.matshow(corr, fignum=1)
    plt.colorbar()
    plt.title("Correlation matrix (subset)")
    plt.xticks(range(len(features)), features, rotation=90, fontsize=12)
    plt.yticks(range(len(features)), features, fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def save_pca_scatter(X, labels, path, title="PCA 2D"):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pcs = pca.fit_transform(X)
    plt.figure(figsize=(8,6))
    if labels is not None:
        unique = pd.Series(labels).unique()
        for u in unique:
            mask = (labels == u)
            plt.scatter(pcs[mask,0], pcs[mask,1], label=str(u), s=12, alpha=0.7)
        plt.legend(markerscale=2, fontsize=8)
    else:
        plt.scatter(pcs[:,0], pcs[:,1], s=8, alpha=0.6)
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title(title)
    plt.savefig(path, bbox_inches='tight', dpi=120)
    plt.close()

def save_boxplot_by_group(df, feature, group_col, out_path):
    plt.figure(figsize=(6,4))
    groups = []
    labels = []
    for g in sorted(df[group_col].dropna().unique()):
        vals = df.loc[df[group_col]==g, feature].dropna().values
        if len(vals) > 0:
            groups.append(vals)
            labels.append(str(g))
    if len(groups) == 0:
        return
    plt.boxplot(groups, labels=labels)
    plt.xticks(rotation=45, fontsize=8)
    plt.title(f"{feature} by {group_col}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def cohen_d(x, y):
    nx = len(x); ny = len(y)
    if nx < 2 or ny < 2:
        return np.nan
    dof = nx + ny - 2
    if dof <= 0:
        return np.nan
    pooled = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
    if pooled == 0:
        return np.nan
    return (np.mean(x) - np.mean(y)) / pooled

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    auto_merge_if_missing()
    if not MERGED_CSV.exists():
        raise FileNotFoundError(f"No se encontró {MERGED_CSV}. Revisa v2/v3 o crea merged_features_raw.csv.")

    df = pd.read_csv(MERGED_CSV, low_memory=False)
    merged = df.copy()

    # normalize condition/label
    if 'condition' in merged.columns:
        merged['condition_original'] = merged['condition'].astype(str)
        merged['condition'] = merged['condition'].apply(normalize_text_label)
        log.info(f"Condition values after normalization: {merged['condition'].dropna().unique()[:20]}")
    if 'label' in merged.columns:
        merged['label_original'] = merged['label'].astype(str)
        merged['label'] = merged['label'].apply(normalize_text_label)

    summary = quick_column_summary(merged)
    safe_save(summary, "initial_column_summary.csv")

    id_col = detect_id_column(merged)
    log.info(f"Id column detected: {id_col}")

    array_report_cols = []
    if MERGED_ARRAYS_CSV.exists():
        try:
            arr_df = pd.read_csv(MERGED_ARRAYS_CSV, low_memory=False)
            possible_names = [c for c in arr_df.columns if c.lower().find("column")>=0 or c.lower().find("feature")>=0 or c.lower().find("name")>=0]
            if len(possible_names)>0:
                array_report_cols = arr_df[possible_names[0]].dropna().astype(str).tolist()
            else:
                array_report_cols = arr_df.iloc[:,0].dropna().astype(str).tolist()
            log.info(f"Loaded {len(array_report_cols)} array-like names from merged_array_columns.csv")
        except Exception:
            log.warning("No se pudo leer merged_array_columns.csv; ignorando.")

    array_detected_by_value = detect_array_like_columns_by_value(merged)
    log.info(f"Detected {len(array_detected_by_value)} array-like columns by value inspection.")
    array_like_cols = sorted(set(array_report_cols) | set(array_detected_by_value))
    array_like_cols = [c for c in array_like_cols if c in merged.columns]
    safe_save(pd.DataFrame({"array_like_detected": array_like_cols}), "detected_array_like_columns.csv")

    if array_like_cols:
        merged = summarize_array_cols(merged, array_like_cols)

    summary_after = quick_column_summary(merged)
    safe_save(summary_after, "column_summary_after_array_summarize.csv")

    non_feature_candidates = [id_col, "condition", "label", "source"]
    non_feature_cols = [c for c in non_feature_candidates if c is not None and c in merged.columns]
    feature_candidates = [c for c in merged.columns if c not in non_feature_cols]

    not_convertible = coerce_numeric_and_report(merged, feature_candidates)
    safe_save(pd.DataFrame({"not_convertible_after_coerce": not_convertible}), "not_convertible_after_coerce.csv")
    features = [c for c in feature_candidates if c not in not_convertible]

    nan_pct = merged[features].isna().mean().sort_values(ascending=False)
    to_drop = nan_pct[nan_pct > MAX_NAN_DROP].index.tolist()
    if to_drop:
        safe_save(pd.DataFrame({"dropped_high_nan": to_drop}), "columns_dropped_due_to_nan.csv")
        features = [c for c in features if c not in to_drop]

    log.info(f"Features final count: {len(features)}")
    safe_save(pd.DataFrame({"final_features": features}), "final_features_list.csv")
    merged.to_csv(RESULTS_DIR / "merged_features_cleaned_preview.csv", index=False)

    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    if len(features) == 0:
        log.error("No numeric features available after cleaning. Abort.")
        raise RuntimeError("No numeric features available.")

    X_imp = imp.fit_transform(merged[features])
    X_scaled = scaler.fit_transform(X_imp)
    pd.DataFrame(X_scaled, columns=features).head(200).to_csv(RESULTS_DIR / "rf_X_preview_before_modeling.csv", index=False)

    # ----------------------
    # STATISTICAL ANALYSIS (ANOVA fallback)
    # ----------------------
    log.info("Iniciando ANOVA por feature (factores si hay).")
    factors = [f for f in ["condition","diameter","experiment","mic"] if f in merged.columns]
    anova_rows = []

    for feat in features:
        try:
            # try factorial ANOVA first (all factors)
            if len(factors) > 0:
                tmp = merged[[feat] + factors].dropna()
                if tmp.shape[0] >= MIN_SAMPLES_FOR_ANOVA and tmp[factors].nunique().sum() > len(factors):
                    formula = f"{feat} ~ " + " + ".join([f"C({fac})" for fac in factors])
                    model = smf.ols(formula=formula, data=tmp).fit()
                    aov = anova_lm(model, typ=2)
                    ss_total = aov['sum_sq'].sum()
                    eta = {}
                    for idx, row in aov.iterrows():
                        ss = row['sum_sq']; eta[idx] = float(ss / ss_total) if ss_total>0 else np.nan
                    aov.to_csv(RESULTS_DIR / f"anova_factorial_{feat}.csv")
                    anova_rows.append({"feature": feat, "n_samples": int(tmp.shape[0]), "eta": eta})
                    continue
            # fallback: one-way ANOVA by condition (if condition present)
            if 'condition' in merged.columns:
                tmp2 = merged[[feat,'condition']].dropna()
                if tmp2.shape[0] >= MIN_SAMPLES_FOR_ANOVA and tmp2['condition'].nunique() >= 2:
                    model2 = smf.ols(f"{feat} ~ C(condition)", data=tmp2).fit()
                    aov2 = anova_lm(model2, typ=2)
                    ss_total = aov2['sum_sq'].sum()
                    eta = {}
                    for idx, row in aov2.iterrows():
                        ss = row['sum_sq']; eta[idx] = float(ss / ss_total) if ss_total>0 else np.nan
                    aov2.to_csv(RESULTS_DIR / f"anova_by_condition_{feat}.csv")
                    anova_rows.append({"feature": feat, "n_samples": int(tmp2.shape[0]), "eta": eta})
                    continue
            # else skip
        except Exception as e:
            log.debug(f"ANOVA error for {feat}: {e}")

    # build eta summary
    eta_rows_flat = []
    for r in anova_rows:
        feat = r['feature']; eta = r['eta']; row = {"feature": feat, "n_samples": r['n_samples']}
        for k,v in eta.items():
            keyname = str(k).replace('C(','').replace(')','').replace(':','_x_')
            row[f"eta2_{keyname}"] = v
        eta_rows_flat.append(row)
    eta_df = pd.DataFrame(eta_rows_flat)
    if not eta_df.empty:
        eta_cols = [c for c in eta_df.columns if c.startswith("eta2_")]
        if len(eta_cols) > 0:
            eta_df["eta2_sum"] = eta_df[eta_cols].sum(axis=1)
        eta_df.to_csv(RESULTS_DIR / "eta_squared_summary.csv", index=False)
    else:
        log.info("No ANOVA results produced (eta_df empty).")

    # ----------------------
    # KRUSKAL-WALLIS by condition (non-parametric)
    # ----------------------
    if 'condition' in merged.columns:
        conds = sorted(merged['condition'].dropna().unique())
        if len(conds) >= 2:
            kw_rows = []
            for feat in features:
                groups = [merged.loc[merged['condition']==c, feat].dropna().values for c in conds]
                groups = [g for g in groups if len(g)>0]
                if len(groups) >= 2:
                    try:
                        stat,p = stats.kruskal(*groups)
                        kw_rows.append({'feature': feat, 'kw_stat': float(stat), 'kw_pvalue': float(p)})
                    except Exception:
                        pass
            pd.DataFrame(kw_rows).to_csv(RESULTS_DIR / "kruskal_by_condition.csv", index=False)
        else:
            log.info("No hay suficientes grupos en 'condition' para Kruskal.")

    # ----------------------
    # COHEN'S D con falla vs sin falla (labels normalizadas)
    # ----------------------
    cohen_rows = []
    if 'condition' in merged.columns:
        unique_conds = set(merged['condition'].dropna().unique())
        if ("con falla" in unique_conds) and ("sin falla" in unique_conds):
            for feat in features:
                a = merged.loc[merged['condition']=="con falla", feat].dropna().values
                b = merged.loc[merged['condition']=="sin falla", feat].dropna().values
                d = cohen_d(a,b)
                cohen_rows.append({'feature': feat, 'cohen_d': float(d) if np.isfinite(d) else np.nan,
                                   'n_con_falla': len(a), 'n_sin_falla': len(b)})
            pd.DataFrame(cohen_rows).to_csv(RESULTS_DIR / "cohen_d_con_vs_sin.csv", index=False)
        else:
            log.info("No se detectó el par con falla / sin falla en labels normalizados para Cohen's d.")

    # ----------------------
    # SUPERVISED ML: Random Forest (si hay label/condition)
    # ----------------------
    ranking_components = pd.DataFrame({"feature": features})
    if 'condition' in merged.columns or 'label' in merged.columns:
        lab = 'condition' if 'condition' in merged.columns else 'label'
        log.info(f"Detected label: {lab}. Preparing supervised RF.")
        df_model = merged.dropna(subset=[lab]).copy()
        if df_model.shape[0] < 10:
            log.warning("Pocas filas con label no nulo; salto RF.")
        else:
            y_raw = df_model[lab].astype(str).str.strip()
            le = LabelEncoder(); y_enc = le.fit_transform(y_raw)
            classes = le.classes_.tolist()
            log.info(f"Clases detectadas: {classes} (n={len(classes)})")
            X_model = df_model[features].copy()
            X_model_imp = imp.transform(X_model)
            X_model_scaled = scaler.transform(X_model_imp)

            # Cross-validated predictions + metrics
            clf = RandomForestClassifier(**RF_PARAMS)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            try:
                scoring = 'roc_auc' if len(classes)==2 else 'accuracy'
                scores = cross_val_score(clf, X_model_scaled, y_enc, cv=cv, scoring=scoring, n_jobs=-1)
                pd.DataFrame({f"cv_{scoring}": scores}).to_csv(RESULTS_DIR / "rf_cv_results.csv", index=False)
                log.info(f"RF CV ({scoring}): mean {scores.mean():.4f} ± {scores.std():.4f}")
            except Exception:
                with open(RESULTS_DIR / "rf_cv_error.txt", "w", encoding="utf-8") as f:
                    f.write(traceback.format_exc())
                log.warning("RF CV failed; ver rf_cv_error.txt")

            # cross-validated predictions for confusion matrix and classification report
            try:
                y_pred = cross_val_predict(clf, X_model_scaled, y_enc, cv=cv, n_jobs=-1)
                cls_report = classification_report(y_enc, y_pred, target_names=le.classes_, output_dict=False)
                with open(RESULTS_DIR / "classification_report_cv.txt", "w", encoding="utf-8") as f:
                    f.write("Classification report (CV):\n\n")
                    f.write(cls_report)
                cm = confusion_matrix(y_enc, y_pred)
                pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_csv(RESULTS_DIR / "confusion_matrix_cv.csv")
            except Exception:
                with open(RESULTS_DIR / "rf_cv_pred_error.txt", "w", encoding="utf-8") as f:
                    f.write(traceback.format_exc())
                log.warning("CV predictions (confusion matrix) failed.")

            # fit final model and extract importances
            try:
                clf.fit(X_model_scaled, y_enc)
                fi = pd.DataFrame({"feature": features, "rf_importance": clf.feature_importances_}).sort_values("rf_importance", ascending=False)
                fi.to_csv(RESULTS_DIR / "feature_importances.csv", index=False)
                ranking_components = ranking_components.merge(fi, on='feature', how='left')
            except Exception:
                with open(RESULTS_DIR / "rf_fit_error.txt", "w", encoding="utf-8") as f:
                    f.write(traceback.format_exc())
                log.warning("RF fit failed; ver rf_fit_error.txt")

            # permutation importance
            try:
                perm = permutation_importance(clf, X_model_scaled, y_enc, n_repeats=30, random_state=RANDOM_STATE, n_jobs=-1)
                perm_df = pd.DataFrame({'feature': features, 'perm_mean': perm.importances_mean, 'perm_std': perm.importances_std})
                perm_df.sort_values('perm_mean', ascending=False).to_csv(RESULTS_DIR / "permutation_importances.csv", index=False)
                ranking_components = ranking_components.merge(perm_df[['feature','perm_mean']], on='feature', how='left')
            except Exception:
                log.warning("Permutation importance failed.")

            # mutual info
            try:
                mi = mutual_info_classif(X_model_scaled, y_enc, random_state=RANDOM_STATE)
                mi_df = pd.DataFrame({'feature': features, 'mutual_info': mi})
                mi_df.sort_values('mutual_info', ascending=False).to_csv(RESULTS_DIR / "mutual_info.csv", index=False)
                ranking_components = ranking_components.merge(mi_df, on='feature', how='left')
            except Exception:
                log.warning("Mutual info failed.")

            # ---------- ROBUST SHAP SECTION (safe, no-raising) ----------
                        # ---------- ROBUST SHAP SECTION (auto-detect shapes, safe) ----------
            if _HAS_SHAP and ('clf' in locals()) and ('X_model_scaled' in locals()):
                try:
                    n_total = X_model_scaled.shape[0]
                    n_sample = min(SHAP_SAMPLE_MAX, n_total)
                    if n_sample < n_total:
                        rng = np.random.RandomState(RANDOM_STATE)
                        idx = rng.choice(n_total, n_sample, replace=False)
                        X_shap = X_model_scaled[idx]
                    else:
                        X_shap = X_model_scaled

                    explainer = shap.TreeExplainer(clf)
                    shap_vals = explainer.shap_values(X_shap)

                    diag_lines = []
                    diag_lines.append(f"X_shap shape: {getattr(X_shap,'shape',None)}")

                    mean_abs = None
                    # If shap_vals is list => one element per class (each element likely (n_samples,n_features) or similar)
                    if isinstance(shap_vals, list):
                        diag_lines.append(f"shap_values: list length {len(shap_vals)}")
                        per_class_arrays = []
                        for i, s in enumerate(shap_vals):
                            arr = np.asarray(s)
                            diag_lines.append(f"  class {i} arr.shape = {arr.shape}, ndim={arr.ndim}")
                            # if arr is 2D (samples, features) -> ok
                            if arr.ndim == 2 and arr.shape[1] == len(features):
                                per_class_arrays.append(np.abs(arr).mean(axis=0))
                            elif arr.ndim == 3:
                                # try detect feature axis
                                fa = [ax for ax, d in enumerate(arr.shape) if d == len(features)]
                                if len(fa) == 1:
                                    feat_ax = fa[0]
                                    other_axes = tuple(i for i in range(arr.ndim) if i != feat_ax)
                                    per_class_arrays.append(np.mean(np.abs(arr), axis=other_axes))
                                else:
                                    per_class_arrays.append(None)
                            else:
                                per_class_arrays.append(None)
                        # keep only valid arrays
                        valid = [a for a in per_class_arrays if a is not None]
                        if len(valid) > 0:
                            mean_abs = np.mean(np.vstack(valid), axis=0)
                        else:
                            mean_abs = None
                    else:
                        arr = np.asarray(shap_vals)
                        diag_lines.append(f"shap_values: ndarray ndim={arr.ndim}, shape={arr.shape}")
                        if arr.ndim == 2:
                            # (samples, features) or (features, samples)?
                            if arr.shape[1] == len(features):
                                mean_abs = np.mean(np.abs(arr), axis=0)   # samples x features
                            elif arr.shape[0] == len(features):
                                mean_abs = np.mean(np.abs(arr), axis=1)   # features x samples
                            else:
                                mean_abs = None
                        elif arr.ndim == 3:
                            # unknown ordering: find which axis equals n_features
                            fa = [ax for ax, d in enumerate(arr.shape) if d == len(features)]
                            if len(fa) == 1:
                                feat_ax = fa[0]
                                other_axes = tuple(i for i in range(arr.ndim) if i != feat_ax)
                                mean_abs = np.mean(np.abs(arr), axis=other_axes)
                            else:
                                mean_abs = None
                        else:
                            mean_abs = None

                    # diagnostics + safe behaviour
                    if mean_abs is None:
                        diag_lines.insert(0, "Result: could not compute per-feature mean_abs SHAP (incompatible shapes).")
                        with open(RESULTS_DIR / "shap_error.txt", "w", encoding="utf-8") as f:
                            f.write("SHAP diagnostic (no per-feature summary produced):\n\n")
                            f.write("\n".join(diag_lines))
                        log.warning("SHAP produjo formas incompatibles. Se guardó shap_error.txt y se continúa sin SHAP.")
                    else:
                        mean_abs = np.asarray(mean_abs).ravel()
                        if mean_abs.shape[0] != len(features):
                            diag_lines.append(f"Mismatch: mean_abs length {mean_abs.shape[0]} != n_features {len(features)}")
                            with open(RESULTS_DIR / "shap_error.txt", "w", encoding="utf-8") as f:
                                f.write("SHAP diagnostic (length mismatch):\n\n")
                                f.write("\n".join(diag_lines))
                            log.warning("SHAP mean_abs length mismatch. Se guardó shap_error.txt y se continúa sin SHAP.")
                        else:
                            shap_summary = pd.DataFrame({"feature": features, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
                            shap_summary.to_csv(RESULTS_DIR / "shap_summary.csv", index=False)
                            try:
                                ranking_components = ranking_components.merge(shap_summary, on='feature', how='left')
                            except Exception:
                                log.warning("No se pudo mergear summary SHAP en ranking_components, pero shap_summary fue guardado.")
                except Exception:
                    with open(RESULTS_DIR / "shap_error.txt", "w", encoding="utf-8") as f:
                        f.write("Exception running SHAP:\n\n")
                        f.write(traceback.format_exc())
                    log.warning("SHAP falló con una excepción. Ver shap_error.txt")
            else:
                if not _HAS_SHAP:
                    log.info("SHAP no instalado; salto resumen SHAP.")
                else:
                    log.info("SHAP saltado: modelo/clases no disponibles para explicar (clf/X_model_scaled faltantes).")
            # ---------- end SHAP ----------

    # ----------------------
    # Merge eta & cohen into ranking and compute combined score
    # ----------------------
    if 'eta_df' in locals() and not eta_df.empty:
        ranking_components = ranking_components.merge(eta_df[['feature','eta2_sum']], on='feature', how='left')
    else:
        ranking_components['eta2_sum'] = np.nan

    if (RESULTS_DIR / "cohen_d_con_vs_sin.csv").exists():
        cohen_df = pd.read_csv(RESULTS_DIR / "cohen_d_con_vs_sin.csv")
        ranking_components = ranking_components.merge(cohen_df[['feature','cohen_d']], on='feature', how='left')
    else:
        ranking_components['cohen_d'] = np.nan

    # ensure columns exist
    for col in ['rf_importance','perm_mean','mutual_info','mean_abs_shap']:
        if col not in ranking_components.columns:
            ranking_components[col] = np.nan

    # normalization helper
    def normalize_col(s):
        if s.isna().all():
            return s
        mn = np.nanmin(s)
        mx = np.nanmax(s)
        if np.isfinite(mn) and np.isfinite(mx) and mx - mn > 0:
            return (s - mn) / (mx - mn)
        else:
            return s * 0.0

    ranking_components['eta2_sum_norm'] = normalize_col(ranking_components['eta2_sum'])
    ranking_components['cohen_d_abs_norm'] = normalize_col(ranking_components['cohen_d'].abs())
    ranking_components['rf_importance_norm'] = normalize_col(ranking_components['rf_importance'])
    ranking_components['perm_mean_norm'] = normalize_col(ranking_components['perm_mean'])
    ranking_components['mutual_info_norm'] = normalize_col(ranking_components['mutual_info'])
    ranking_components['shap_norm'] = normalize_col(ranking_components['mean_abs_shap'])

    # weights (adjustable)
    w = {"eta":0.35, "cohen":0.20, "rf":0.20, "perm":0.10, "mi":0.10, "shap":0.05}
    ranking_components['combined_score'] = (
        w['eta'] * ranking_components['eta2_sum_norm'].fillna(0) +
        w['cohen'] * ranking_components['cohen_d_abs_norm'].fillna(0) +
        w['rf'] * ranking_components['rf_importance_norm'].fillna(0) +
        w['perm'] * ranking_components['perm_mean_norm'].fillna(0) +
        w['mi'] * ranking_components['mutual_info_norm'].fillna(0) +
        w['shap'] * ranking_components['shap_norm'].fillna(0)
    )
    ranking_components.sort_values('combined_score', ascending=False, inplace=True)
    ranking_components.to_csv(RESULTS_DIR / "feature_ranking_combined.csv", index=False)

    # Save boxplots for top features by ranking
    top_feats = ranking_components['feature'].head(20).tolist()
    if 'condition' in merged.columns:
        for feat in top_feats:
            try:
                save_boxplot_by_group(merged, feat, "condition", RESULTS_DIR / f"box_{feat}_by_condition.png")
            except Exception:
                log.debug(f"Boxplot failed for {feat}")

    # correlation matrix for top var features
    try:
        top_n = min(40, len(features))
        var = np.nanvar(X_scaled, axis=0)
        top_idx = np.argsort(-var)[:top_n]
        top_feats_byvar = [features[i] for i in top_idx]
        save_plot_corr_matrix(pd.DataFrame(X_scaled, columns=features), top_feats_byvar, RESULTS_DIR / "correlation_matrix_top.png")
    except Exception:
        log.exception("Correlation matrix plotting failed.")

    # PCA colored by label (if any)
    try:
        if 'condition' in merged.columns or 'label' in merged.columns:
            lab = 'condition' if 'condition' in merged.columns else 'label'
            df_model = merged.dropna(subset=[lab]).copy()
            X_model_imp = imp.transform(df_model[features]); X_model_scaled = scaler.transform(X_model_imp)
            save_pca_scatter(X_model_scaled, df_model[lab].astype(str).values, RESULTS_DIR / "pca_by_label.png", title="PCA colored by label")
    except Exception:
        log.exception("PCA by label failed.")

    log.info("Análisis completo. Revisa la carpeta results/ para outputs y diagnósticos.")
