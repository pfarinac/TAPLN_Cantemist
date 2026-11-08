
## Como entrenar varios modelos distintos
#### No he cambiado los codigos, solo he añadido 2 
    - run_neer_experiments.py
    - compare_ner.py

# IMPORTANTE
**Subimos todos los archivos de la carpeta de src a sample_data (apartado de memoria de colab)**
# IMPORTANTE

### para conectarte al drive y rollo eso

from google.colab import drive
drive.mount('/content/drive')
CORPUS = "/content/drive/MyDrive/Cantemist_pro/cantemist"
!pip -q install transformers sentencepiece
!nvidia-smi | head -n 15

### ejecutas el baseline 

!python /content/sample_data/ner_analyze.py --corpus-dir $CORPUS --split dev-set2 --system dict


### ejecutas el archivo con la semilla que quieres
!python /content/sample_data/run_ner_experiments.py --corpus-dir $CORPUS --seeds 2026

### para ver los resultados y graficas

!python /content/sample_data/compare_ner.py --corpus-dir $CORPUS

from IPython.display import Image, display
import glob
for f in sorted(glob.glob('../outputs/ner/figs/*.png')):
    display(Image(f))


### si te quieres descargar la carpeta con la info entrenada+ graficas

import shutil
from google.colab import files
ruta_carpeta = '/content/outputs/ner' 
nombre_archivo_zip = 'outputs_ner'
shutil.make_archive(nombre_archivo_zip, 'zip', ruta_carpeta)
files.download(nombre_archivo_zip + '.zip')