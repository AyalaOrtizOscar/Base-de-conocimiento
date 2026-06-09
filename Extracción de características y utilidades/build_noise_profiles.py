#!/usr/bin/env python3
"""
build_noise_profiles.py — Construir perfiles espectrales de ruido
para usar como referencia en noisereduce.

Genera perfiles a partir de grabaciones etiquetadas:
  - husillo_solo:    CNC encendida, sin taladrar, sin refrigerante
  - husillo_refrig:  CNC encendida + refrigerante, sin taladrar
  - ambiente:        CNC apagada / ambiente laboratorio

Salida: D:/pipeline_SVM/noise_profiles/{nombre}.npy
        D:/pipeline_SVM/noise_profiles/profiles_summary.json

Uso:
    python build_noise_profiles.py
"""

import os, json, warnings
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

warnings.filterwarnings("ignore")

OUTDIR = Path("D:/pipeline_SVM/noise_profiles")
OUTDIR.mkdir(parents=True, exist_ok=True)

SR = 44100        # sr al que se cargará (mismo que feature extraction)
N_FFT = 2048
HOP = 512
CHUNK_SEC = 30    # leer en chunks de 30s para no saturar RAM

# ── Definición de fuentes de ruido ────────────────────────────────────────────
# Cada entrada: (label, [lista de archivos WAV], descripcion)
NOISE_SOURCES = {
    "husillo_solo": {
        "desc": "CNC encendida, husillo girando, sin refrigerante, sin taladrado",
        "files": [
            # E:/raw_noise — grabaciones largas etiquetadas como MAQUINA ON
            r"E:\raw_noise\_test182_MAQUINA ON\NI\channel_0.wav",
            r"E:\raw_noise\_test317_machine on\NI\channel_0.wav",
            r"E:\raw_noise\_test321 _machine on\NI\channel_0.wav",
            r"E:\raw_noise\_test175_MAQUINA SUENA BIEN\NI\channel_0.wav",
            r"E:\raw_noise\_test276_mic_on\NI\channel_0.wav",
            # D:/dataset/ruidos
            r"D:\dataset\ruidos\maquina_on\sinrefri_001.wav",
            r"D:\dataset\ruidos\maquina_on\sinrefri_002.wav",
            r"D:\dataset\ruidos\maquina_on\sinrefri_003.wav",
            r"D:\dataset\ruidos\maquina_on\sinrefri_004.wav",
            r"D:\dataset\ruidos\maquina_on\sinrefri_005.wav",
        ]
    },
    "husillo_refrigerante": {
        "desc": "CNC encendida + refrigerante activo, sin taladrado",
        "files": [
            # E:/Datos Generados — ch2 etiquetados explícitamente
            r"E:\Datos Generados\6mm_test13\ch2 maquina, encendido rotacion de husillo.wav",
            r"E:\Datos Generados\6mm_test16\ch2 TALADRINA GIRO DE HUSILLO.wav",
            r"E:\Datos Generados\6mm_test15\ch1 SONIDO TALADRINA.wav",
            # D:/dataset/ruidos
            r"D:\dataset\ruidos\maquina_on_and_refrigerante\maquina6_001.wav",
            r"D:\dataset\ruidos\maquina_on_and_refrigerante\maquina6_002.wav",
            r"D:\dataset\ruidos\maquina_on_and_refrigerante\maquina6_003.wav",
            r"D:\dataset\ruidos\maquina_on_and_refrigerante\refri6_001.wav",
            r"D:\dataset\ruidos\maquina_on_and_refrigerante\refri6_002.wav",
            r"D:\dataset\ruidos\maquina_on_and_refrigerante\refri_001.wav",
            r"D:\dataset\ruidos\maquina_on_and_refrigerante\refri_002.wav",
            r"D:\dataset\ruidos\refrigerante\solorefri_001.wav",
            r"D:\dataset\ruidos\refrigerante\solorefri_002.wav",
            r"D:\dataset\ruidos\refrigerante\solorefri_003.wav",
        ]
    },
    "ambiente": {
        "desc": "Ambiente laboratorio / CNC apagada",
        "files": [
            r"E:\Datos Generados\6mm_test8\ch2 sonido bueno de fondo ambiente del laboratorio.wav",
            r"E:\raw_noise\_test168 _RUIDO_AMBIENTE\NI\channel_0.wav",
            r"D:\dataset\ruidos\maquina_off\m_off_001.wav",
            r"D:\dataset\ruidos\maquina_off\solo_001.wav",
            r"D:\dataset\ruidos\maquina_off\solo_002.wav",
            r"D:\dataset\ruidos\maquina_off\solo_003.wav",
        ]
    },
    "combinado": {
        # Perfil más representativo para aplicar en producción:
        # husillo + refrigerante (condición más frecuente en los ensayos E1-E7)
        "desc": "Perfil combinado husillo+refrigerante — recomendado para filtrado de dataset",
        "files": []  # se llena abajo combinando husillo_solo + husillo_refrigerante
    }
}


def load_noise_waveform(files: list, sr: int, max_seconds: float = 120.0) -> np.ndarray:
    """
    Carga archivos de ruido, toma segmentos centrales (evita transitorios),
    concatena y devuelve waveform crudo para noisereduce.
    """
    segments = []
    loaded_sec = 0.0
    per_file = max_seconds / max(len([f for f in files if os.path.exists(f)]), 1)

    for fpath in files:
        if not os.path.exists(fpath):
            print(f"  [SKIP] No encontrado: {fpath}")
            continue
        if loaded_sec >= max_seconds:
            print(f"  [STOP] Ya tenemos {loaded_sec:.0f}s")
            break
        try:
            dur_total = librosa.get_duration(path=fpath)
            dur_to_use = min(dur_total, per_file, max_seconds - loaded_sec)
            # Tomar segmento central para evitar transitorios inicio/fin
            offset = max(0, (dur_total - dur_to_use) / 2)
            y, _ = librosa.load(fpath, sr=sr, offset=offset,
                                duration=dur_to_use, mono=True)
            segments.append(y)
            loaded_sec += len(y) / sr
            fname = os.path.basename(fpath)
            rms = float(np.sqrt(np.mean(y**2)))
            print(f"  + {fname}  ({len(y)/sr:.1f}s, RMS={rms:.6f})  acum={loaded_sec:.0f}s")
        except Exception as e:
            print(f"  [ERROR] {fpath}: {e}")

    if not segments:
        print("  [WARN] Ningun archivo procesado — perfil vacio")
        return np.zeros(int(sr * 1), dtype=np.float32)

    return np.concatenate(segments).astype(np.float32)


def main():
    print("=" * 60)
    print("  BUILD NOISE PROFILES")
    print("=" * 60)

    # Construir perfil combinado desde husillo + refrigerante
    NOISE_SOURCES["combinado"]["files"] = (
        NOISE_SOURCES["husillo_solo"]["files"] +
        NOISE_SOURCES["husillo_refrigerante"]["files"]
    )

    profiles = {}
    summary = {}

    for name, cfg in NOISE_SOURCES.items():
        print(f"\n--- Perfil: {name} ---")
        print(f"    {cfg['desc']}")
        waveform = load_noise_waveform(cfg["files"], SR, max_seconds=120.0)
        out_path = OUTDIR / f"{name}.npy"
        np.save(out_path, waveform)
        profiles[name] = waveform

        n_files_found = sum(1 for f in cfg["files"] if os.path.exists(f))
        rms = float(np.sqrt(np.mean(waveform**2)))
        summary[name] = {
            "desc": cfg["desc"],
            "n_files_configured": len(cfg["files"]),
            "n_files_found": n_files_found,
            "duration_s": float(len(waveform) / SR),
            "rms": rms,
            "output": str(out_path),
        }
        print(f"  Guardado: {out_path}  ({len(waveform)/SR:.1f}s, RMS={rms:.6f})")

    # Guardar resumen
    summary_path = OUTDIR / "profiles_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResumen: {summary_path}")
    print("\nPerfiles generados (waveforms para noisereduce):")
    for name, info in summary.items():
        print(f"  {name}: {info['n_files_found']}/{info['n_files_configured']} archivos, "
              f"{info['duration_s']:.0f}s, RMS={info['rms']:.6f}")

    print("\nPara usar en filtrado:")
    print(f"  y_noise = np.load('{OUTDIR}/combinado.npy')")
    print("  import noisereduce as nr")
    print("  y_clean = nr.reduce_noise(y=y, sr=sr, y_noise=y_noise, stationary=True)")


if __name__ == "__main__":
    main()
