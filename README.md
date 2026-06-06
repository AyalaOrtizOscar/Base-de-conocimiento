# Base de conocimiento

**Clasificacion ordinal del desgaste de brocas de acero rapido cobaltado en taladrado CNC mediante analisis acustico.**

Repositorio acompanante de la tesis de grado de Oscar Ivan Ayala Ortiz, Escuela de Ingenieria Mecanica, Universidad Industrial de Santander (UIS), Bucaramanga, Colombia (2026).

> Director: Jorge Enrique Meneses Florez. Codirector: Nicolas Orejarena Osorio.

---

## Cita sugerida (APA 7)

> Ayala-Ortiz, O. I. (2026). *pipeline_SVM: Clasificacion ordinal del desgaste de brocas CNC mediante analisis acustico* (Version 1.0-sustentacion) [Software]. GitHub. https://github.com/AyalaOrtizOscar/pipeline_SVM

Si cita una version concreta congelada para revision por jurado, use el tag `v1.0-sustentacion`.

---

## Contenido del repositorio

| Carpeta | Que contiene |
|---|---|
| `scripts/` | Scripts Python del pipeline (extraccion de 26 features acusticas, entrenamiento SVM Frank & Hall, SHAP, validacion LOEO, generacion de figuras) |
| `manifests/` | CSV de splits train/val/test, manifestos por experimento (E1-E7) |

## NO incluido (por tamano o privacidad)

- `features/` 
- `augmented/` 
- `noise_profiles/`, `previews/`, `data/`, `inputs/`
- Modelos serializados `*.joblib`, `*.keras`, `*.h5`
- Audios crudos `*.wav`, `*.tdms`

Para acceder a los datos brutos, contactar al autor o al director Jorge Enrique Meneses Florez (Escuela de Ingenieria Mecanica, UIS).

---

## Reproducibilidad

- Python 3.10.0
- Dependencias clave: `scikit-learn`, `librosa`, `soundfile`, `noisereduce`, `shap`, `pandas`, `numpy`, `matplotlib`, `joblib`, `umap-learn`
- `random_state=42` constante en todo el pipeline
- Validacion: Leave-One-Experiment-Out (LOEO) con E3 como holdout fijo
- Etiquetado ordinal `[15/75]` (15 % vida util = sin desgaste, > 75 % = desgastado)

## Resultados principales

- **Adjacent accuracy:** 90.1 % (con filtrado espectral; 84.7 % sin filtrado)
- **Errores nunca exceden un escalon ordinal** (98.6 % adj. accuracy en threshold optimo)
- **SVM RBF, C = 10**, top-15 features por informacion mutua
- **3 075 muestras** del Lote I (Orejarena, 2014); pipeline extendido a Lote II (Ayala, 2025-2026)

## Compilacion del informe (Windows / MiKTeX)

```bash
cd informe_proyecto
CLEAN_PATH="/c/Users/ayala/AppData/Local/Programs/MiKTeX/miktex/bin/x64:/c/Windows/system32:/c/Windows"
PATH="$CLEAN_PATH" pdflatex -interaction=nonstopmode informe_proyecto.tex
PATH="$CLEAN_PATH" bibtex informe_proyecto
PATH="$CLEAN_PATH" pdflatex -interaction=nonstopmode informe_proyecto.tex
PATH="$CLEAN_PATH" pdflatex -interaction=nonstopmode informe_proyecto.tex
```

## Licencia

Codigo bajo MIT. Texto del informe y figuras bajo CC BY 4.0. Datos experimentales sujetos a politica de proteccion de datos UIS.

## Contacto

Oscar Ivan Ayala Ortiz — `ayalaortizoscarivan@gmail.com`
