#!/usr/bin/env python3
# split_dataset.py
#
# Genera splits reproducibles train / val / test desde master.csv (v2)
#
# Estrategia (documentable en articulo):
#   TEST  -> Experimento E3 completo (6mm, lineal, condensador, 583 orig)
#            Experimento independiente; permite afirmar
#            "entrenado en E1/E4/E5/E6/E7, evaluado en E3" sin contaminacion.
#   VAL   -> Experimento E2 completo (8mm, spiral, dinamico, 128 orig)
#            Diferente diametro+mic que test -> evalua generalizacion cruzada.
#            Si --val-experiment no se especifica, usa 20% estratificado.
#   TRAIN -> Originales de E1+E4+E5+E6+E7 + augmentados (excl E3/E2 derivados)
#
# Reglas metodologicas (no negociables para publicacion):
#   1. Ninguna muestra augmentada aparece en val ni test.
#   2. Muestras de test/val experiments no aparecen en train.
#   3. Augmentados derivados de test/val experiments excluidos de train.
#   4. RANDOM_STATE=42 para reproducibilidad.
#
# Columna 'experiment' (E1-E7) debe existir en master.csv.
# Si no existe, se calcula via filepath_to_experiment() (backward compat).
#
# Uso:
#   python split_dataset.py [--master PATH] [--outdir PATH]
#   python split_dataset.py --test-experiment E3 --val-experiment E2

import argparse
import re
import os
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

RANDOM_STATE = 42

# Mapeo de experimento global a condiciones experimentales
EXPERIMENT_INFO = {
    "E1": "8mm, mixed gcode, with coolant, dynamic mic",
    "E2": "8mm, spiral gcode, with coolant, dynamic mic",
    "E3": "6mm, linear gcode, with coolant, condenser mic",
    "E4": "6mm, spiral gcode, with coolant, condenser mic",
    "E5": "8mm, linear gcode, with coolant, condenser mic",
    "E6": "8mm, linear gcode, with coolant, condenser mic",
    "E7": "6mm, spiral gcode, with coolant, condenser mic",
}


# ── Extraccion de experimento desde filepath (backward compat) ─────────────────

def filepath_to_experiment(filepath: str) -> str:
    """Determina el experimento global (E1-E7) a partir del filepath."""
    fp = str(filepath).replace("\\", "/")
    name = os.path.splitext(os.path.basename(fp))[0].replace("limpio_", "")

    # 6mm patterns: broca/brocz + diametro + secuencia
    m6 = re.search(r"broc[az]_?(\d+)_(\d+)", name, re.IGNORECASE)
    if m6:
        diam = int(m6.group(1))
        seq = int(m6.group(2))
        if diam == 6:
            return {1: "E3", 2: "E4", 3: "E7"}.get(seq, "unknown")
        elif diam == 8:
            return {1: "E5", 2: "E6"}.get(seq, "unknown")

    # 8mm B0X pattern
    m8 = re.search(r"^B0(\d)", name, re.IGNORECASE)
    if m8:
        return {1: "E1", 2: "E2"}.get(int(m8.group(1)), "unknown")

    # Fallback: folder-based
    for e in ["E7", "E6", "E5", "E4", "E3", "E2", "E1"]:
        if f"/{e}_" in fp or f"\\{e}_" in fp:
            return e

    return "unknown"


# ── Estadisticas de un split ──────────────────────────────────────────────────

def split_stats(df: pd.DataFrame, name: str) -> dict:
    """Genera un diccionario con estadisticas del split para el reporte."""
    label_counts = df["label"].value_counts().to_dict()
    mic_counts   = df["mic_type"].value_counts().to_dict() if "mic_type" in df.columns else {}
    aug_counts   = df["aug_type"].fillna("original").value_counts().to_dict()
    exp_counts   = df["experiment"].value_counts().to_dict() if "experiment" in df.columns else {}

    print(f"\n--- {name} ({len(df)} muestras) ---")
    print(f"  Labels      : {label_counts}")
    print(f"  Mic types   : {mic_counts}")
    print(f"  Aug types   : {aug_counts}")
    print(f"  Experiments : {exp_counts}")
    print(f"  Originales  : {(df['aug_type'].fillna('original') == 'original').sum()}")

    return {
        "n": len(df),
        "labels": label_counts,
        "mic_types": mic_counts,
        "aug_types": aug_counts,
        "experiments": exp_counts,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Genera splits train/val/test desde master.csv"
    )
    parser.add_argument(
        "--master",
        default="D:/dataset/manifests/master.csv",
        help="Ruta al manifest canonico (default: D:/dataset/manifests/master.csv)",
    )
    parser.add_argument(
        "--outdir",
        default="D:/dataset/manifests",
        help="Carpeta de salida para train.csv, val.csv, test.csv",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.20,
        help="Fraccion de originales no-test para val si no se usa --val-experiment",
    )
    parser.add_argument(
        "--test-experiment",
        default="E3",
        help="Experimento reservado como test set (default: E3)",
    )
    parser.add_argument(
        "--val-experiment",
        default="E2",
        help="Experimento reservado como val set (default: E2). Usar 'none' para stratified split.",
    )
    args = parser.parse_args()

    test_exp = args.test_experiment
    val_exp = args.val_experiment if args.val_experiment.lower() != "none" else None
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Cargar master ─────────────────────────────────────────────────────────
    print(f"Cargando {args.master} ...")
    df = pd.read_csv(args.master)
    print(f"  Total filas: {len(df)}")

    # Normalizar aug_type
    df["aug_type"] = df["aug_type"].fillna("original")
    df["is_original"] = df["aug_type"] == "original"

    # Usar columna experiment si existe, sino calcular
    if "experiment" not in df.columns:
        print("  Columna 'experiment' no encontrada, calculando desde filepath...")
        df["experiment"] = df["filepath"].apply(filepath_to_experiment)
    else:
        # Rellenar NaN
        mask = df["experiment"].isna() | (df["experiment"] == "")
        if mask.any():
            df.loc[mask, "experiment"] = df.loc[mask, "filepath"].apply(filepath_to_experiment)

    print(f"\n=== Distribucion por experimento x label ===")
    print(pd.crosstab(df["experiment"], df["label"]).to_string())

    # Advertir sobre unknowns
    n_unknown = (df["experiment"] == "unknown").sum()
    if n_unknown > 0:
        print(f"\nWARN: {n_unknown} muestras con experiment='unknown'.")
        print("   Se incluiran en TRAIN. Revisar filepaths si es inesperado.")

    # Reportar y eliminar filepaths duplicados
    n_dupes = df["filepath"].duplicated().sum()
    if n_dupes > 0:
        print(f"\nINFO: {n_dupes} filepaths duplicados — se conserva la primera ocurrencia.")
        df = df.drop_duplicates(subset=["filepath"], keep="first").reset_index(drop=True)
        print(f"  Filas tras deduplicar: {len(df)}")

    # Separar originales y augmentados
    df_orig = df[df["is_original"]].copy().reset_index(drop=True)
    df_aug  = df[~df["is_original"]].copy().reset_index(drop=True)

    # Excluir augmentados derivados de test y val experiments
    excluded_exps = {test_exp}
    if val_exp:
        excluded_exps.add(val_exp)

    if "orig_filepath" in df_aug.columns:
        df_aug["orig_experiment"] = df_aug["orig_filepath"].apply(filepath_to_experiment)
        n_aug_excl = df_aug["orig_experiment"].isin(excluded_exps).sum()
        if n_aug_excl > 0:
            print(f"INFO: {n_aug_excl} augmentados de {excluded_exps} excluidos de train.")
        df_aug = df_aug[~df_aug["orig_experiment"].isin(excluded_exps)].copy().reset_index(drop=True)

    # ── TEST: todos los originales del test experiment ─────────────────────────
    df_test          = df_orig[df_orig["experiment"] == test_exp].copy()
    df_orig_not_test = df_orig[df_orig["experiment"] != test_exp].copy().reset_index(drop=True)

    if len(df_test) == 0:
        raise RuntimeError(
            f"No se encontraron muestras originales con experiment='{test_exp}'. "
            "Verificar columna 'experiment' en master.csv."
        )
    print(f"\nTEST: {len(df_test)} muestras de {test_exp} ({EXPERIMENT_INFO.get(test_exp, '')})")

    # ── VAL: experimento dedicado o stratified split ──────────────────────────
    if val_exp:
        # Val = experimento completo
        df_val = df_orig_not_test[df_orig_not_test["experiment"] == val_exp].copy()
        df_train_orig = df_orig_not_test[df_orig_not_test["experiment"] != val_exp].copy().reset_index(drop=True)
        if len(df_val) == 0:
            raise RuntimeError(
                f"No se encontraron muestras originales con experiment='{val_exp}'. "
                "Verificar columna 'experiment' en master.csv."
            )
        print(f"VAL:  {len(df_val)} muestras de {val_exp} ({EXPERIMENT_INFO.get(val_exp, '')})")
    else:
        # Fallback: stratified split
        if len(df_orig_not_test) < 10:
            raise RuntimeError(
                f"Muy pocas muestras originales fuera de {test_exp} ({len(df_orig_not_test)}). "
                "Verificar master.csv."
            )
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=args.val_frac,
            random_state=RANDOM_STATE,
        )
        train_idx, val_idx = next(
            splitter.split(df_orig_not_test, df_orig_not_test["label"])
        )
        df_val        = df_orig_not_test.iloc[val_idx].copy()
        df_train_orig = df_orig_not_test.iloc[train_idx].copy()
        print(f"VAL:  {len(df_val)} muestras (stratified {args.val_frac:.0%} of non-test orig)")

    # ── TRAIN: originales de train + TODOS los augmentados ────────────────────
    df_train = pd.concat([df_train_orig, df_aug], ignore_index=True)

    # Quitar duplicados (por si acaso)
    df_train = df_train.drop_duplicates(subset=["filepath"]).reset_index(drop=True)
    df_val   = df_val.drop_duplicates(subset=["filepath"]).reset_index(drop=True)
    df_test  = df_test.drop_duplicates(subset=["filepath"]).reset_index(drop=True)

    # ── Validaciones de integridad ────────────────────────────────────────────
    train_files = set(df_train["filepath"])
    val_files   = set(df_val["filepath"])
    test_files  = set(df_test["filepath"])

    assert len(train_files & test_files) == 0, \
        "ERROR: hay solapamiento entre train y test"
    assert len(train_files & val_files) == 0, \
        "ERROR: hay solapamiento entre train y val"
    assert len(val_files & test_files) == 0, \
        "ERROR: hay solapamiento entre val y test"
    assert df_val["is_original"].all(), \
        "ERROR: val contiene muestras augmentadas"
    assert df_test["is_original"].all(), \
        "ERROR: test contiene muestras augmentadas"
    test_in_train = df_train["experiment"].eq(test_exp).sum()
    assert test_in_train == 0, \
        f"ERROR: {test_in_train} muestras de {test_exp} aparecen en train"
    if val_exp:
        val_in_train = df_train[df_train["aug_type"] == "original"]["experiment"].eq(val_exp).sum()
        assert val_in_train == 0, \
            f"ERROR: {val_in_train} originales de {val_exp} aparecen en train"

    print("\nOK: Todas las validaciones de integridad pasaron.")

    # ── Estadisticas por split ────────────────────────────────────────────────
    stats = {
        "random_state": RANDOM_STATE,
        "val_frac": args.val_frac,
        "test_experiment": test_exp,
        "val_experiment": val_exp,
        "master_md5": hashlib.md5(
            Path(args.master).read_bytes()
        ).hexdigest(),
        "split_strategy": (
            f"TEST={test_exp}(orig only); "
            f"VAL={'%s(orig only)' % val_exp if val_exp else 'stratified %.0f%% of non-%s orig' % (args.val_frac*100, test_exp)}; "
            f"TRAIN=remaining orig + all augmented (excl {excluded_exps} derived)"
        ),
        "train": split_stats(df_train, "TRAIN"),
        "val":   split_stats(df_val,   "VAL"),
        "test":  split_stats(df_test,  "TEST"),
    }

    # ── Guardar splits ────────────────────────────────────────────────────────
    df_train["split"] = "train"
    df_val["split"]   = "val"
    df_test["split"]  = "test"

    df_train.to_csv(outdir / "train.csv", index=False)
    df_val.to_csv(outdir / "val.csv",     index=False)
    df_test.to_csv(outdir / "test.csv",   index=False)

    with open(outdir / "split_metadata.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n--- Archivos generados en {outdir} ---")
    print(f"  train.csv  -> {len(df_train):5d} filas")
    print(f"  val.csv    -> {len(df_val):5d} filas  (solo originales)")
    print(f"  test.csv   -> {len(df_test):5d} filas  (solo originales, {test_exp})")
    print(f"  split_metadata.json  -> reproducibilidad + checksums")
    print()
    print("NOTA para el articulo:")
    if val_exp:
        train_exps = sorted(set(df_train[df_train["aug_type"]=="original"]["experiment"].unique()))
        print(f"  'El modelo fue entrenado con datos de {len(train_exps)} experimentos ({', '.join(train_exps)}),")
        print(f"   validado en {val_exp} ({EXPERIMENT_INFO.get(val_exp, '')}),")
        print(f"   y evaluado en {test_exp} ({EXPERIMENT_INFO.get(test_exp, '')}),")
        print(f"   garantizando independencia experimental completa entre splits.'")


if __name__ == "__main__":
    main()
