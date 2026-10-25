# CANTEMIST-NER · Grupo 4 (TAPLN)

Sistema de reconocimiento de menciones de morfología tumoral (`MORFOLOGIA_NEOPLASIA`) en casos clínicos
oncológicos en español. Este repositorio cubre la **subtarea NER**; NORM y CODING se añadirán después.

## Qué hay en cada archivo

```
cantemist_ner/
├── README.md              Este documento.
├── requirements.txt       Librerías de Python necesarias (torch, transformers, numpy).
└── src/
    ├── cantemist_utils.py
    ├── ner_train.py
    └── ner_analyze.py
```

### `src/cantemist_utils.py` · base común del proyecto
Lo importan los otros dos scripts. Contiene:
- **Parsers** de los `.ann` (formato Brat, tanto NER como NORM) y del `.tsv` de coding. Acepta códigos como `8140/3`, `8120/31` o `9064/3/H`.
- **Carga y caché:** lee el corpus una vez y guarda el resultado en `cache/cantemist_parsed.pkl`, para no volver a parsear en cada ejecución.
- **Gold de NER:** los spans correctos de cada documento (unión de los `.ann` de NER y de NORM).
- **Evaluación:** P/R/F1 micro con coincidencia exacta para NER, NORM y CODING.
- **Escritura de `.ann`:** guarda predicciones en el formato oficial de Brat.
- **Comando `audit`:** comprueba la coherencia entre NER, NORM y el TSV de coding.

Se ejecuta directamente solo para la auditoría:
```
python cantemist_utils.py audit              # añadir --refresh para reparsear el corpus
```

### `src/ner_train.py` · entrenamiento y predicción
Entrena un encoder biomédico en español (`PlanTL-GOB-ES/roberta-base-biomedical-clinical-es`) como etiquetador
de secuencias con etiquetas BIO. Hace:
- Tokenizar los documentos y alinear los offsets de carácter con las etiquetas BIO.
- Trocear los documentos largos en **ventanas de 512 tokens con solape de 64**, y promediar las probabilidades en las zonas solapadas al predecir.
- Entrenar, validar en cada época con el F1 a nivel de span, y guardar el mejor modelo en `models/`.
- Imprimir al inicio el **techo de tokenización**, el F1 máximo alcanzable con este esquema (0,9878 en dev-set1).
- Evaluar con el mejor modelo en dev-set1 y dev-set2, y guardar las predicciones.

Dos subcomandos:
- `train`: entrena y evalúa.
- `predict --split X`: aplica un modelo ya entrenado. Para `test-set` exige `--final`, para no tocar el test antes de tiempo.

### `src/ner_analyze.py` · análisis de errores
Recibe las predicciones de un sistema (o genera un baseline de diccionario) y escribe un informe con:
- F1 estricto y F1 relajado (cualquier solape). La diferencia son errores de **frontera**.
- Tipos de falsos negativos y falsos positivos (frontera más corta, más larga, no detectada, inventada).
- Recall por longitud de mención y por mención vista o no vista en train.
- Menciones más frecuentes entre FN y FP, y los códigos eCIE-O con más FN (enlaza con NORM).
- Ejemplos con contexto.

El informe se guarda en `outputs/ner/analysis_*.txt`.

### Carpetas que se crean solas
- `cache/`: corpus parseado.
- `models/`: modelo entrenado.
- `outputs/ner/`: predicciones (`preds_*.json` y carpetas `ann_*`), historial de entrenamiento e informes de análisis.

## Cómo ejecutarlo

### Requisitos
- Python 3.9 o superior.
- **Una GPU NVIDIA con CUDA, o Google Colab con GPU (T4 gratuita).** En CPU el entrenamiento tarda horas por época y no es viable.
- El corpus CANTEMIST con esta estructura:
  ```
  cantemist/
  ├── train-set/     cantemist-ner/ (.txt y .ann), cantemist-norm/ (.ann), cantemist-coding/ (.tsv)
  ├── dev-set1/
  ├── dev-set2/
  └── test-set/
  ```

### Instalación (en una carpeta, con entorno virtual)
```
python -m venv .venv
# Linux/macOS:  source .venv/bin/activate
# Windows:      .venv\Scripts\Activate.ps1

pip install torch --index-url https://download.pytorch.org/whl/cu121   # con GPU NVIDIA (ver pytorch.org para otras versiones de CUDA)
pip install "transformers>=4.40" numpy
python -c "import torch; print(torch.cuda.is_available())"            # debe imprimir True
```
La primera vez que se entrena se descarga el modelo de Hugging Face (unos 500 MB, necesita internet).

### Ejecución (desde `src/`)
Indicad dónde está el corpus (la carpeta que contiene `train-set`, `dev-set1`, etc.):
```
# Linux/macOS
export CANTEMIST_DIR="/ruta/a/cantemist"
# Windows PowerShell
$env:CANTEMIST_DIR="C:\ruta\a\cantemist"
```
(o añadid `--corpus-dir RUTA` a cada comando).

```
python cantemist_utils.py audit                                          # 1) auditoría de datos
python ner_analyze.py --split dev-set2 --system dict                     # 2) baseline de diccionario
python ner_train.py train                                                # 3) entrenar
python ner_analyze.py --split dev-set2 --preds ../outputs/ner/preds_dev-set2_ner_roberta_bio.json   # 4) errores
python ner_train.py predict --split test-set --final                     # 5) SOLO al final del proyecto
```

Opciones útiles de `train`: `--epochs N`, `--batch-size N` (bajarlo si falta memoria de GPU), `--lr`, `--patience`.
Las opciones globales (`--precision`, `--max-len`, `--model`, `--name`) van **antes** de `train`, por ejemplo
`python ner_train.py --precision fp16 train`. En una T4 de Colab usad `--precision fp16`; si la pérdida sale `nan`, repetid con `--precision fp32`.

### Problemas típicos
- `No module named transformers`: entorno virtual sin activar o librerías sin instalar.
- `CUDA out of memory`: `--batch-size 4`.
- Falla la descarga del modelo: `python ner_train.py --model BSC-LT/roberta-base-biomedical-clinical-es train`.
- En Colab, copiad `models/` y `outputs/` a Drive al terminar: lo de `/content` se pierde al cerrar la sesión.

## Protocolo de evaluación (no romper)
- **train-set:** entrenar.
- **dev-set1:** elegir la mejor época.
- **dev-set2:** resultado que se reporta (no se usa para decidir nada).
- **test-set:** una sola vez, al final.

## Estado actual
| Experimento | Split | F1 estricto | F1 relajado |
|---|---|---|---|
| Diccionario de menciones de train | dev-set2 | 0,690 | 0,887 |
| RoBERTa biomédico + BIO | dev-set2 | pendiente | pendiente |

Techo de tokenización con BIO (dev-set1): 0,9878. Las menciones solapadas no se pueden representar en BIO y se descartan (la más corta).

## Notas para el Diario de Trabajo
- Las estadísticas obtenidas leyendo el corpus desde Google Drive en Colab daban datos inconsistentes (documentos sin anotaciones que en local sí las tienen). **Usad solo cifras calculadas con el corpus en local.**
- Hubo un fallo de parser: los códigos de dos dígitos (`8120/31`, grado histológico) se descartaban. Está corregido en `CODE_RE`.
- El modelo base (`roberta-base-biomedical-clinical-es`, BSC, licencia Apache 2.0) es un **baseline**: es un encoder, no un LLM. La parte LLM del proyecto va aparte.
- Antes de citar el modelo en la memoria, comprobad el artículo original: arXiv 2109.03570.