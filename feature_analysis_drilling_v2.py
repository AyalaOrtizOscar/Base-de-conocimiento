#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feature_analysis_drilling_fixed.py

Versión corregida del extractor de features:
 - evita carpetas no deseadas (procesa solo Con falla / Sin falla / ruidos)
 - corrige la generación de nombres .png/.npy (no usar with_suffix con '_stft.png')
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pywt
from scipy import stats

# -------------------------
# Helpers I/O y utilidades
# -------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_wavs_recursive(root: Path):
    return sorted([p for p in root.rglob("*.wav")])

def read_mono(path, sr=None):
    y, s = sf.read(path)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr is not None and s != sr:
        y = librosa.resample(y.astype(float), orig_sr=s, target_sr=sr)
        s = sr
    return y.astype(np.float32), s

def parse_metadata_from_path(p: Path, input_root: Path):
    rel = p.relative_to(input_root)
    parts = rel.parts
    meta = {
        "condition": None,
        "diameter": None,
        "experiment": None,
        "mic": None
    }
    if len(parts) >= 1:
        meta["condition"] = parts[0]
    if len(parts) >= 2:
        meta["diameter"] = parts[1]
    if len(parts) >= 3:
        sub = parts[2]
        if "E1" in sub.upper():
            meta["experiment"] = "E1"
        elif "E2" in sub.upper():
            meta["experiment"] = "E2"
        meta["mic"] = sub
    if len(parts) >= 4 and meta["mic"] is None:
        meta["mic"] = parts[3]
    return meta

# -------------------------
# Feature extraction
# -------------------------
def compute_basic_metrics(y, sr):
    eps = 1e-12
    rms = float(np.sqrt(np.mean(y**2) + eps))
    rms_db = 20.0 * np.log10(rms + eps)
    peak = float(np.max(np.abs(y)))
    duration = float(len(y) / sr)
    return {"rms": rms, "rms_db": rms_db, "peak": peak, "duration_s": duration}

def compute_spectral_features(y, sr, n_fft=2048, hop_length=512, n_mels=64, n_mfcc=13):
    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**2
    centroid = librosa.feature.spectral_centroid(S=None, y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    rolloff = librosa.feature.spectral_rolloff(S=None, y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)
    S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, power=2.0)
    S_mel_db = librosa.power_to_db(S_mel, ref=np.max)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S_mel), n_mfcc=n_mfcc)
    return {
        "stft_power": D,
        "spectral_centroid": centroid,
        "spectral_rolloff": rolloff,
        "zcr": zcr,
        "mel_power": S_mel,
        "mel_db": S_mel_db,
        "mfcc": mfcc
    }

def compute_wavelet_features(y, sr, widths=None, wavelet='morl'):
    if widths is None:
        widths = np.logspace(np.log10(1), np.log10(128), num=32).astype(int)
        widths = np.unique(widths)
    coef, freqs = pywt.cwt(y, widths, wavelet, sampling_period=1.0/sr)
    energy_per_scale = np.sum(np.abs(coef)**2, axis=1)
    total_energy = np.sum(energy_per_scale)
    return {
        "cwt_coef": coef,
        "cwt_freq_scale": widths,
        "wavelet_energy_per_scale": energy_per_scale,
        "wavelet_total_energy": total_energy
    }

# -------------------------
# Estadística descriptiva
# -------------------------
def summarize_feature_series(df, groupby_cols=None, feature_cols=None):
    if feature_cols is None:
        feature_cols = ["rms", "rms_db", "peak", "duration_s", "zcr", "centroid_mean", "rolloff_mean"]
    if groupby_cols is None:
        groupby_cols = []
    if len(groupby_cols) == 0:
        stats_out = []
        for feat in feature_cols:
            if feat in df.columns:
                arr = df[feat].dropna().values
                stats_out.append({
                    "feature": feat,
                    "count": int(len(arr)),
                    "mean": float(np.mean(arr)) if len(arr)>0 else np.nan,
                    "std": float(np.std(arr)) if len(arr)>0 else np.nan,
                    "median": float(np.median(arr)) if len(arr)>0 else np.nan,
                    "min": float(np.min(arr)) if len(arr)>0 else np.nan,
                    "max": float(np.max(arr)) if len(arr)>0 else np.nan,
                    "skew": float(stats.skew(arr, bias=False)) if len(arr)>2 else np.nan,
                    "kurtosis": float(stats.kurtosis(arr, bias=False)) if len(arr)>3 else np.nan
                })
        return pd.DataFrame(stats_out)
    else:
        grp = df.groupby(groupby_cols)
        out = []
        for gname, gdf in grp:
            row = {"group": gname}
            for feat in feature_cols:
                if feat in gdf.columns:
                    arr = gdf[feat].dropna().values
                    row.update({
                        f"{feat}_count": int(len(arr)),
                        f"{feat}_mean": float(np.mean(arr)) if len(arr)>0 else np.nan,
                        f"{feat}_std": float(np.std(arr)) if len(arr)>0 else np.nan,
                        f"{feat}_median": float(np.median(arr)) if len(arr)>0 else np.nan,
                        f"{feat}_min": float(np.min(arr)) if len(arr)>0 else np.nan,
                        f"{feat}_max": float(np.max(arr)) if len(arr)>0 else np.nan,
                        f"{feat}_skew": float(stats.skew(arr, bias=False)) if len(arr)>2 else np.nan,
                        f"{feat}_kurtosis": float(stats.kurtosis(arr, bias=False)) if len(arr)>3 else np.nan
                    })
            out.append(row)
        return pd.DataFrame(out)

# -------------------------
# Guardado de data / plots
# -------------------------
def save_array_and_spectrogram(array, npy_path: Path, png_path: Path, title=None, sr=None, hop_length=512, y_axis='linear'):
    ensure_dir(npy_path.parent)
    ensure_dir(png_path.parent)
    np.save(str(npy_path), array)
    try:
        fig, ax = plt.subplots(figsize=(8,4))
        if array.ndim == 2:
            img = librosa.display.specshow(librosa.power_to_db(array+1e-12, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis=y_axis, ax=ax)
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
        else:
            ax.plot(array)
        if title is not None:
            ax.set_title(title)
        plt.tight_layout()
        fig.savefig(str(png_path), dpi=150)
        plt.close(fig)
    except Exception as e:
        print("Warning saving png", png_path, e)

# -------------------------
# Pipeline principal
# -------------------------
def process_all(input_root: Path, export_root: Path, sr=44100, n_fft=2048, hop_length=512, n_mels=64, n_mfcc=13, max_files=None):
    input_root = Path(input_root)
    export_root = Path(export_root)
    ensure_dir(export_root)

    features_dir = export_root / "features"
    plots_dir = export_root / "plots"
    arrays_dir = export_root / "arrays"
    wavelets_dir = export_root / "wavelets"
    summary_dir = export_root / "summary"
    for d in [features_dir, plots_dir, arrays_dir, wavelets_dir, summary_dir]:
        ensure_dir(d)

    wavs = list_wavs_recursive(input_root)
    if max_files:
        wavs = wavs[:max_files]
    if len(wavs) == 0:
        print("No se encontraron WAVs en:", input_root)
        return

    allowed_top = {"con falla", "sin falla", "ruidos"}  # valid first-level folders (lowercase)
    records = []
    processed = 0
    skipped = 0

    for p in tqdm(wavs, desc="Archivos"):
        try:
            # validar primer folder relativo
            rel = p.relative_to(input_root)
            if len(rel.parts) == 0:
                skipped += 1
                continue
            top = rel.parts[0].lower()
            if top not in allowed_top:
                skipped += 1
                continue

            y, s = read_mono(p, sr=sr)
            meta = parse_metadata_from_path(p, input_root)

            basic = compute_basic_metrics(y, sr)
            spec = compute_spectral_features(y, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, n_mfcc=n_mfcc)
            wv = compute_wavelet_features(y, sr)

            centroid = spec["spectral_centroid"]
            rolloff = spec["spectral_rolloff"]
            zcr = spec["zcr"]
            centroid_mean = float(np.mean(centroid)) if centroid.size>0 else np.nan
            centroid_std = float(np.std(centroid)) if centroid.size>0 else np.nan
            rolloff_mean = float(np.mean(rolloff)) if rolloff.size>0 else np.nan
            rolloff_std = float(np.std(rolloff)) if rolloff.size>0 else np.nan
            zcr_mean = float(np.mean(zcr)) if zcr.size>0 else np.nan

            # construir carpetas por metadata
            condition = meta.get('condition') or 'unknown'
            diameter = meta.get('diameter') or 'unknown'
            experiment = meta.get('experiment') or 'unk'
            plot_dir = plots_dir / condition / diameter / experiment
            array_dir = arrays_dir / condition / diameter / experiment
            wave_dir = wavelets_dir / condition / diameter / experiment
            ensure_dir(plot_dir)
            ensure_dir(array_dir)
            ensure_dir(wave_dir)

            # STFT
            stft_npy = array_dir / (p.stem + "_stft.npy")
            stft_png = plot_dir / (p.stem + "_stft.png")
            save_array_and_spectrogram(spec["stft_power"], stft_npy, stft_png, title=f"STFT Power - {p.name}", sr=sr, hop_length=hop_length, y_axis='linear')

            # Mel
            mel_npy = array_dir / (p.stem + "_mel.npy")
            mel_png = plot_dir / (p.stem + "_mel.png")
            save_array_and_spectrogram(spec["mel_power"], mel_npy, mel_png, title=f"Mel Power - {p.name}", sr=sr, hop_length=hop_length, y_axis='mel')

            # MFCC
            mfcc_npy = array_dir / (p.stem + "_mfcc.npy")
            mfcc_png = plot_dir / (p.stem + "_mfcc.png")
            save_array_and_spectrogram(spec["mfcc"], mfcc_npy, mfcc_png, title=f"MFCC - {p.name}", sr=sr, hop_length=hop_length, y_axis='linear')

            # Wavelet coef y energía
            wv_coef_path = wave_dir / (p.stem + "_cwt.npy")
            np.save(str(wv_coef_path), wv["cwt_coef"])
            np.save(str(wv_coef_path.with_name(p.stem + "_energy_per_scale.npy")), wv["wavelet_energy_per_scale"])
            fig, ax = plt.subplots(figsize=(6,3))
            ax.plot(wv["wavelet_energy_per_scale"])
            ax.set_title(f"Wavelet energy per scale - {p.name}")
            ax.set_xlabel("scale index")
            ax.set_ylabel("energy")
            wavelet_png = wave_dir / (p.stem + "_wavelet_energy.png")
            fig.tight_layout()
            fig.savefig(str(wavelet_png), dpi=150)
            plt.close(fig)

            # Waveform
            waveform_npy = array_dir / (p.stem + "_waveform.npy")
            waveform_png = plot_dir / (p.stem + "_waveform.png")
            np.save(str(waveform_npy), y)
            fig, ax = plt.subplots(figsize=(8,2))
            times = np.linspace(0, len(y)/sr, num=len(y))
            ax.plot(times, y, linewidth=0.4)
            ax.set_title(f"Waveform - {p.name}")
            ax.set_xlabel("time (s)")
            fig.tight_layout()
            fig.savefig(str(waveform_png), dpi=150)
            plt.close(fig)

            rec = {
                "filepath": str(p),
                "filename": p.name,
                "condition": meta.get('condition'),
                "diameter": meta.get('diameter'),
                "experiment": meta.get('experiment'),
                "mic": meta.get('mic'),
                "sr": s
            }
            rec.update(basic)
            rec.update({
                "zcr": zcr_mean,
                "centroid_mean": centroid_mean,
                "centroid_std": centroid_std,
                "rolloff_mean": rolloff_mean,
                "rolloff_std": rolloff_std,
                "mfcc_0_mean": float(np.mean(spec["mfcc"][0])) if spec["mfcc"].shape[0] > 0 else np.nan,
                "mfcc_1_mean": float(np.mean(spec["mfcc"][1])) if spec["mfcc"].shape[0] > 1 else np.nan,
                "mel_total_energy": float(np.sum(spec["mel_power"]))
            })
            records.append(rec)
            processed += 1
        except Exception as e:
            print("Error procesando", p, e)
            skipped += 1

    df = pd.DataFrame(records)
    features_csv = export_root / "features" / "features_per_file.csv"
    ensure_dir(features_csv.parent)
    df.to_csv(features_csv, index=False)
    print("Features guardadas en:", features_csv)

    feature_cols = ["rms", "rms_db", "peak", "duration_s", "zcr", "centroid_mean", "rolloff_mean", "mel_total_energy"]
    global_stats = summarize_feature_series(df, groupby_cols=[], feature_cols=feature_cols)
    global_stats.to_csv(export_root / "summary" / "global_stats.csv", index=False)
    per_condition = summarize_feature_series(df, groupby_cols=["condition"], feature_cols=feature_cols)
    per_condition.to_csv(export_root / "summary" / "per_condition_stats.csv", index=False)
    per_diameter = summarize_feature_series(df, groupby_cols=["diameter"], feature_cols=feature_cols)
    per_diameter.to_csv(export_root / "summary" / "per_diameter_stats.csv", index=False)
    per_mic = summarize_feature_series(df, groupby_cols=["mic"], feature_cols=feature_cols)
    per_mic.to_csv(export_root / "summary" / "per_mic_stats.csv", index=False)

    print(f"Proceso terminado. Procesados: {processed}. Saltados: {skipped}. Total WAVs inspeccionados: {len(wavs)}")
    print("Resumen guardado en:", export_root / "summary")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extracción masiva de features (fixed).")
    p.add_argument("--input_root", required=True, help="Carpeta raíz con 'Con falla','Sin falla','ruidos'")
    p.add_argument("--export_root", required=True, help="Carpeta de export (ej: D:/v2)")
    p.add_argument("--sr", type=int, default=44100)
    p.add_argument("--n_fft", type=int, default=2048)
    p.add_argument("--hop_length", type=int, default=512)
    p.add_argument("--n_mels", type=int, default=64)
    p.add_argument("--n_mfcc", type=int, default=13)
    p.add_argument("--max_files", type=int, default=None)
    args = p.parse_args()
    process_all(args.input_root, args.export_root, sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length,
                n_mels=args.n_mels, n_mfcc=args.n_mfcc, max_files=args.max_files)
