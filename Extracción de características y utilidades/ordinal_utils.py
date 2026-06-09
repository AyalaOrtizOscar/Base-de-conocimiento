# ordinal_utils.py
# Utilidades compartidas para clasificación ordinal (método Frank & Hall)
#
# Orden de clases:
#   0 = sin_desgaste
#   1 = medianamente_desgastado
#   2 = desgastado
#
# Descomposición Frank & Hall (2 clasificadores binarios):
#   C1: P(y >= 1)  →  "¿hay algún desgaste?"
#   C2: P(y >= 2)  →  "¿el desgaste es severo?"
#
# Codificación de targets:
#   sin_desgaste             → [0, 0]
#   medianamente_desgastado  → [1, 0]
#   desgastado               → [1, 1]
#
# Decodificación de probabilidades:
#   P(y=0) = 1 - P1
#   P(y=1) = P1 - P2        (con P2 ≤ P1 forzado)
#   P(y=2) = P2

import numpy as np

LABEL_TO_IDX = {
    "sin_desgaste": 0,
    "medianamente_desgastado": 1,
    "desgastado": 2,
}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}
N_CLASSES = 3
N_ORDINAL = 2  # número de salidas binarias (K-1)


def ordinal_encode(y_int: np.ndarray) -> np.ndarray:
    """
    Convierte índices de clase (0, 1, 2) a targets binarios ordinales.

    Args:
        y_int: array de enteros (N,) con valores en {0, 1, 2}

    Returns:
        array float32 (N, 2) donde:
            col 0 = P(y >= 1)
            col 1 = P(y >= 2)
    """
    y_int = np.asarray(y_int, dtype=int)
    out = np.zeros((len(y_int), N_ORDINAL), dtype="float32")
    out[y_int >= 1, 0] = 1.0
    out[y_int >= 2, 1] = 1.0
    return out


def ordinal_decode(probs: np.ndarray) -> np.ndarray:
    """
    Convierte probabilidades sigmoid (N, 2) a índices de clase (N,).

    Aplica restricción de monotonicidad: P(y>=2) <= P(y>=1).
    Devuelve la clase con mayor probabilidad estimada.

    Args:
        probs: array float (N, 2) — salidas sigmoid del modelo

    Returns:
        array int (N,) con índices de clase predichos {0, 1, 2}
    """
    probs = np.asarray(probs, dtype="float32")
    p1 = probs[:, 0]
    p2 = np.minimum(probs[:, 1], p1)  # monotonicidad garantizada

    p_class = np.stack([
        1.0 - p1,        # P(y=0)
        p1 - p2,         # P(y=1)
        p2,              # P(y=2)
    ], axis=1)

    return p_class.argmax(axis=1)


def ordinal_proba(probs: np.ndarray) -> np.ndarray:
    """
    Convierte salidas sigmoid (N, 2) a probabilidades por clase (N, 3).

    Args:
        probs: array float (N, 2)

    Returns:
        array float (N, 3) con probabilidades de cada clase
    """
    probs = np.asarray(probs, dtype="float32")
    p1 = probs[:, 0]
    p2 = np.minimum(probs[:, 1], p1)

    return np.stack([1.0 - p1, p1 - p2, p2], axis=1)


# ── Métricas ordinales ────────────────────────────────────────────────────────

def ordinal_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Error absoluto medio en escala ordinal.
    Un error de 2 pasos (sin_desgaste → desgastado) penaliza el doble
    que un error de 1 paso (sin_desgaste → medianamente_desgastado).
    """
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def adjacent_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Fracción de predicciones exactas o a distancia ≤ 1 del valor real.
    Métrica útil en clasificación ordinal donde el error de 1 paso
    puede ser aceptable clínicamente.
    """
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= 1))


def ordinal_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Matriz de confusión 3×3 con filas = true, columnas = pred.
    Las celdas fuera de la diagonal principal representan errores ordinales.
    Los errores en la diagonal ±1 son errores de 1 paso (admisibles).
    Los errores en las esquinas son errores de 2 pasos (críticos).
    """
    from sklearn.metrics import confusion_matrix
    labels = [0, 1, 2]
    return confusion_matrix(y_true, y_pred, labels=labels)


def print_ordinal_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Imprime un resumen completo de métricas ordinales."""
    from sklearn.metrics import classification_report
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    print("\n--- Reporte ordinal -------------------------------------------")
    print(f"  Accuracy exacta  : {(y_true == y_pred).mean():.4f}")
    print(f"  Adjacent accuracy: {adjacent_accuracy(y_true, y_pred):.4f}  (exacta + ±1 paso)")
    print(f"  Ordinal MAE      : {ordinal_mae(y_true, y_pred):.4f}  (0=perfecto, 2=máximo error)")
    print()
    print(classification_report(
        y_true, y_pred,
        labels=[0, 1, 2],
        target_names=["sin_desgaste", "med_desgastado", "desgastado"],
        zero_division=0
    ))
    cm = ordinal_confusion_matrix(y_true, y_pred)
    print("  Confusion matrix (filas=true, cols=pred):")
    print("              sin   med   des")
    labels = ["sin_desgaste", "med_desgas", "desgastado"]
    for i, row in enumerate(cm):
        print(f"  {labels[i]:12s}  {row[0]:4d}  {row[1]:4d}  {row[2]:4d}")
    print()
    two_step_errors = cm[0, 2] + cm[2, 0]
    print(f"  Errores de 2 pasos (críticos): {two_step_errors}")
    print("--------------------------------------------------------------")
