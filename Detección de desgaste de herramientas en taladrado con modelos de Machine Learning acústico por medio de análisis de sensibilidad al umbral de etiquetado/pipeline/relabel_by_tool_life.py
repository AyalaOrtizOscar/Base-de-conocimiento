#!/usr/bin/env python3
"""
relabel_by_tool_life.py

Reetiqueta el master.csv usando umbrales basados en porcentaje de vida util
de la herramienta, en lugar de los umbrales originales (que solo marcaban
los ultimos 4-8 agujeros como desgastado, ~1% de la vida util).

Estrategia [15/75]:
  - sin_desgaste:            hole <= 15% de max_hole
  - medianamente_desgastado: 15% < hole < 75% de max_hole
  - desgastado:              hole >= 75% de max_hole

Aplica SOLO a experimentos "Con falla" (E1-E4) donde se conoce el
punto final de la vida util. E5-E7 ("Sin falla") mantienen sus
etiquetas actuales (basadas en analisis acustico).

Justificacion para el articulo:
  - El umbral original (97-99%) solo captura el fallo catastrofico final
  - El umbral [15/75] captura desgaste progresivo como objetivo de
    monitoreo preventivo (deteccion temprana)
  - Consistente con ISO 8688: VB crece de forma no lineal, con aceleracion
    significativa despues del 60-80% de la vida util

Autor: Claude Code + Oscar Ayala
Fecha: 2026-03-22
"""

import os
import re
import shutil
import hashlib
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ── Configuracion ─────────────────────────────────────────────────────────────

MASTER_CSV = Path("D:/dataset/manifests/master.csv")
ARCHIVE_DIR = Path("D:/dataset/manifests/_archive")

# Umbrales de vida util (porcentaje del max_hole)
THRESH_SIN = 0.15   # <= 15% -> sin_desgaste
THRESH_DES = 0.75   # >= 75% -> desgastado
# Entre 15% y 75% -> medianamente_desgastado

# Experimentos "Con falla" (se conoce el endpoint de vida util)
CON_FALLA = {"E1", "E2", "E3", "E4"}

# Experimentos "Sin falla" (no se aplica reetiquetado)
SIN_FALLA = {"E5", "E6", "E7"}

RANDOM_STATE = 42


# ── Extraccion de numero de agujero ──────────────────────────────────────────

def extract_hole_number(filepath: str) -> int:
    """Extrae el numero de agujero del filepath."""
    fp = str(filepath).replace("\\", "/")
    name = os.path.basename(fp)
    # Remove limpio_ prefix
    name_clean = re.sub(r"^limpio_", "", name, flags=re.IGNORECASE)

    # Pattern 1: B0XYYY (E1, E2) - 8mm
    m = re.search(r"B0\d(\d{3,4})", name_clean)
    if m:
        return int(m.group(1))

    # Pattern 2: broca/brocz_D_S_YYY (E3, E4, E7) - 6mm
    m = re.search(r"broc[az]_?\d_?\d_?(\d{3,4})", name_clean, re.IGNORECASE)
    if m:
        return int(m.group(1))

    # Pattern 3: broca8_S_YYY (E5, E6) - 8mm
    m = re.search(r"broca8_\d_(\d{3,4})", name_clean, re.IGNORECASE)
    if m:
        return int(m.group(1))

    # Fallback: last group of 3+ digits before .wav
    m = re.search(r"(\d{3,4})\.wav$", name_clean, re.IGNORECASE)
    if m:
        return int(m.group(1))

    return -1


def assign_label_by_life(hole_num: int, max_hole: int) -> str:
    """Asigna etiqueta basada en porcentaje de vida util."""
    pct = hole_num / max_hole
    if pct <= THRESH_SIN:
        return "sin_desgaste"
    elif pct >= THRESH_DES:
        return "desgastado"
    else:
        return "medianamente_desgastado"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 65)
    print(f"  REETIQUETADO POR VIDA UTIL [{THRESH_SIN:.0%}/{THRESH_DES:.0%}]")
    print("=" * 65)

    # Cargar master
    df = pd.read_csv(MASTER_CSV)
    print(f"\nMaster cargado: {len(df)} filas")
    print(f"Distribucion actual:")
    print(f"  {df['label'].value_counts().to_dict()}")

    # Backup
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = ARCHIVE_DIR / f"master_before_relabel_{timestamp}.csv"
    shutil.copy2(MASTER_CSV, backup_path)
    print(f"\nBackup: {backup_path}")

    # Guardar label original
    df["label_original"] = df["label"].copy()

    # Extraer hole numbers
    df["hole_num"] = df["filepath"].apply(extract_hole_number)

    # Para augmentados, usar orig_filepath si existe
    mask_aug = df["aug_type"] != "original"
    if "orig_filepath" in df.columns:
        aug_holes = df.loc[mask_aug, "orig_filepath"].apply(extract_hole_number)
        df.loc[mask_aug, "hole_num"] = aug_holes

    n_missing = (df["hole_num"] == -1).sum()
    if n_missing > 0:
        print(f"\nWARN: {n_missing} filas sin hole_num extraible:")
        print(df[df["hole_num"] == -1][["filepath", "experiment"]].head(10))

    # Calcular max_hole por experimento (solo originales)
    orig = df[df["aug_type"] == "original"]
    max_holes = orig.groupby("experiment")["hole_num"].max().to_dict()
    print(f"\nMax holes por experimento:")
    for exp in sorted(max_holes.keys()):
        print(f"  {exp}: {max_holes[exp]} agujeros")

    # ── Reetiquetado de E1-E4 ────────────────────────────────────────────────
    print(f"\n--- Reetiquetando {CON_FALLA} con umbrales [{THRESH_SIN:.0%}/{THRESH_DES:.0%}] ---")

    changes = {"upgraded": 0, "downgraded": 0, "unchanged": 0}

    for exp in sorted(CON_FALLA):
        mask_exp = df["experiment"] == exp
        max_h = max_holes.get(exp, 0)
        if max_h == 0:
            print(f"  {exp}: SKIP (sin datos)")
            continue

        # Threshold holes
        t_sin = int(max_h * THRESH_SIN)
        t_des = int(max_h * THRESH_DES)

        # Apply to all rows of this experiment (originals + augmented)
        for idx in df[mask_exp].index:
            h = df.at[idx, "hole_num"]
            if h == -1:
                continue
            new_label = assign_label_by_life(h, max_h)
            old_label = df.at[idx, "label"]

            if new_label != old_label:
                label_order = {"sin_desgaste": 0, "medianamente_desgastado": 1, "desgastado": 2}
                if label_order.get(new_label, 0) > label_order.get(old_label, 0):
                    changes["upgraded"] += 1
                else:
                    changes["downgraded"] += 1
            else:
                changes["unchanged"] += 1

            df.at[idx, "label"] = new_label

        # Stats for this experiment
        sub = df[mask_exp]
        new_dist = sub["label"].value_counts().to_dict()
        old_dist = sub["label_original"].value_counts().to_dict()
        print(f"\n  {exp} (max_hole={max_h}, sin<={t_sin}, des>={t_des}):")
        print(f"    Antes:   {old_dist}")
        print(f"    Despues: {new_dist}")

    # ── E5-E7: sin cambios ───────────────────────────────────────────────────
    print(f"\n--- {SIN_FALLA}: etiquetas sin cambios (vida util desconocida) ---")
    for exp in sorted(SIN_FALLA):
        sub = df[df["experiment"] == exp]
        dist = sub["label"].value_counts().to_dict()
        print(f"  {exp}: {dist}")

    # ── Resumen global ───────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  RESUMEN GLOBAL")
    print(f"{'='*65}")
    print(f"\nCambios: {changes}")
    print(f"\nDistribucion anterior:")
    print(f"  {df['label_original'].value_counts().to_dict()}")
    print(f"\nDistribucion nueva:")
    new_counts = df["label"].value_counts().to_dict()
    print(f"  {new_counts}")

    # Solo originales
    orig_new = df[df["aug_type"] == "original"]
    print(f"\nSolo originales:")
    print(f"  {orig_new['label'].value_counts().to_dict()}")

    # Crosstab experiment x label (originales)
    print(f"\nCrosstab experiment x label (originales):")
    ct = pd.crosstab(orig_new["experiment"], orig_new["label"])
    print(ct.to_string())

    # ── Guardar ──────────────────────────────────────────────────────────────
    # Conservar label_original para referencia, quitar hole_num (columna temporal)
    df_out = df.drop(columns=["hole_num"])

    df_out.to_csv(MASTER_CSV, index=False)
    print(f"\nmaster.csv actualizado: {MASTER_CSV}")
    print(f"  Columna 'label_original' conservada para auditoria")

    # Metadata
    meta = {
        "timestamp": timestamp,
        "thresholds": {"sin_max_pct": THRESH_SIN, "des_min_pct": THRESH_DES},
        "experiments_relabeled": sorted(CON_FALLA),
        "experiments_unchanged": sorted(SIN_FALLA),
        "max_holes": max_holes,
        "changes": changes,
        "distribution_before": df["label_original"].value_counts().to_dict(),
        "distribution_after": new_counts,
        "master_md5": hashlib.md5(MASTER_CSV.read_bytes()).hexdigest(),
    }
    meta_path = ARCHIVE_DIR / f"relabel_metadata_{timestamp}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  Metadata: {meta_path}")

    print(f"\nNOTA: Ejecutar split_dataset.py para regenerar train/val/test.")


if __name__ == "__main__":
    main()
