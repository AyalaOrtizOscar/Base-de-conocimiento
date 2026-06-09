#!/usr/bin/env python3
"""
apply_noise_filter_to_dataset.py — Aplicar filtro de ruido a todo el dataset.

Lee el master.csv, aplica noisereduce con el perfil 'combinado' a cada WAV
original, guarda los WAVs limpios en D:/dataset/cleaned_wavs/ manteniendo
la misma estructura relativa, y genera clean_manifest.csv con los paths
actualizados.

Uso:
    python apply_noise_filter_to_dataset.py
    python apply_noise_filter_to_dataset.py --profile husillo_solo
    python apply_noise_filter_to_dataset.py --dry-run  # solo muestra qué haría
"""

import os, sys, json, warnings, argparse, time
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    import noisereduce as nr
except ImportError:
    print("ERROR: noisereduce no instalado. Ejecuta: pip install noisereduce")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
MASTER_CSV  = Path("D:/dataset/manifests/master.csv")
PROFILE_DIR = Path("D:/pipeline_SVM/noise_profiles")
CLEAN_DIR   = Path("D:/dataset/cleaned_wavs")
OUT_CSV     = Path("D:/dataset/manifests/master_clean.csv")

SR = 44100        # librosa carga a este sr (mismo que feature extraction)
N_FFT = 2048
# noisereduce stationary=True → usa el perfil como referencia fija (más estable)
# stationary=False → adapta ventana a ventana (más agresivo, posible artefactos)
STATIONARY = True
PROP_DECREASE = 0.85  # qué tanto atenuar el ruido (0=nada, 1=completo)


def load_noise_waveform(profile_path: Path) -> np.ndarray:
    """Carga waveform de ruido guardado por build_noise_profiles.py."""
    y_noise = np.load(profile_path)
    if y_noise.ndim != 1 or len(y_noise) < 1000:
        raise ValueError(f"Perfil invalido: shape={y_noise.shape}. "
                         "Regenera con build_noise_profiles.py")
    return y_noise.astype(np.float32)


def clean_wav(src_path: str, dst_path: str, y_noise: np.ndarray, sr: int):
    """Carga src, aplica noisereduce, guarda en dst. Sin normalizacion para
    preservar niveles relativos (importante para features de energia)."""
    y, _ = librosa.load(src_path, sr=sr, mono=True)
    y_clean = nr.reduce_noise(
        y=y,
        sr=sr,
        y_noise=y_noise,
        stationary=STATIONARY,
        prop_decrease=PROP_DECREASE,
        n_fft=N_FFT,
    )
    # Clip suave solo si supera 1.0 (evita distorsion sin cambiar niveles)
    y_clean = np.clip(y_clean, -1.0, 1.0)
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst_path, y_clean, sr, subtype="PCM_32")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="combinado",
                        choices=["combinado", "husillo_solo",
                                 "husillo_refrigerante", "ambiente"],
                        help="Perfil de ruido a usar")
    parser.add_argument("--dry-run", action="store_true",
                        help="No procesar, solo mostrar qué haría")
    parser.add_argument("--only-originals", action="store_true", default=True,
                        help="Solo procesar aug_type==original (default True)")
    args = parser.parse_args()

    profile_path = PROFILE_DIR / f"{args.profile}.npy"
    if not profile_path.exists():
        print(f"ERROR: Perfil no encontrado: {profile_path}")
        print("Ejecuta primero: python build_noise_profiles.py")
        sys.exit(1)

    print("=" * 60)
    print(f"  APPLY NOISE FILTER — perfil: {args.profile}")
    print(f"  stationary={STATIONARY}, prop_decrease={PROP_DECREASE}")
    print("=" * 60)

    df = pd.read_csv(MASTER_CSV)
    if args.only_originals:
        df = df[df["aug_type"] == "original"].copy()
    print(f"Archivos a procesar: {len(df)}")

    # Cargar waveform de ruido (generado por build_noise_profiles.py)
    print(f"Cargando perfil: {profile_path}")
    y_noise_ref = load_noise_waveform(profile_path)
    print(f"  Waveform de ruido: {len(y_noise_ref)} muestras ({len(y_noise_ref)/SR:.1f}s, "
          f"RMS={np.sqrt(np.mean(y_noise_ref**2)):.6f})")

    if args.dry_run:
        print("\n[DRY RUN] Primeros 5 archivos:")
        for _, row in df.head(5).iterrows():
            src = row["filepath"]
            rel = os.path.relpath(src, "D:/dataset") if src.startswith("D:") else os.path.basename(src)
            dst = str(CLEAN_DIR / f"{args.profile}" / rel)
            print(f"  {src} → {dst}")
        return

    # Procesar
    clean_paths = []
    errors = []
    t0 = time.time()

    for i, (_, row) in enumerate(df.iterrows()):
        src = row["filepath"]
        if not os.path.exists(src):
            errors.append(src)
            clean_paths.append(None)
            continue

        try:
            # Ruta de destino: misma estructura bajo cleaned_wavs/{profile}/
            rel = os.path.relpath(src, "D:/") if "D:/" in src.replace("\\", "/") else os.path.basename(src)
            dst = str(CLEAN_DIR / args.profile / rel)
            # Saltar si ya existe
            if os.path.exists(dst):
                clean_paths.append(dst)
                if i % 100 == 0:
                    print(f"  {i}/{len(df)} — skip (ya existe)")
                continue

            clean_wav(src, dst, y_noise_ref, SR)
            clean_paths.append(dst)

            if i % 50 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(df) - i - 1) / rate
                print(f"  {i+1}/{len(df)}  {rate:.1f} files/s  ETA {eta/60:.1f}min")

        except Exception as e:
            print(f"  [ERROR] {src}: {e}")
            errors.append(src)
            clean_paths.append(None)

    # Guardar nuevo manifest
    df_out = df.copy()
    df_out["filepath_clean"] = clean_paths
    df_out["noise_profile"] = args.profile
    df_out.to_csv(OUT_CSV, index=False)

    elapsed = time.time() - t0
    print(f"\nCompletado en {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Procesados: {sum(1 for p in clean_paths if p)}/{len(df)}")
    print(f"Errores: {len(errors)}")
    print(f"Manifest limpio: {OUT_CSV}")


if __name__ == "__main__":
    main()
