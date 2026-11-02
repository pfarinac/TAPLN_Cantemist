
import argparse
import subprocess
import sys
import time
from pathlib import Path

from cantemist_utils import ROOT



## elige el modelo y descomenta el que quieras usar para los experimentos
# MODELS = {
    
#     'bioclin': 'PlanTL-GOB-ES/roberta-base-biomedical-clinical-es',
#     'xlmr': 'xlm-roberta-base',
#     'beto': 'dccuchile/bert-base-spanish-wwm-cased',
#     'biomedical': 'PlanTL-GOB-ES/roberta-base-biomedical-es',
#     'bsc-bio-ehr': 'PlanTL-GOB-ES/bsc-bio-ehr-es',
#     'xlmr-large': 'xlm-roberta-large',
# }


OUT = ROOT / 'outputs' / 'ner'


def auto_precision():
    """bf16 solo en GPUs Ampere o superiores; fp16 en T4/V100; fp32 sin GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return 'bf16' if torch.cuda.get_device_capability()[0] >= 8 else 'fp16'
    except Exception:
        pass
    return 'fp32'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus-dir', required=True)
    ap.add_argument('--models', nargs='+', default=list(MODELS), choices=list(MODELS))
    ap.add_argument('--seeds', nargs='+', type=int, default=[42])
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--precision', default='auto', choices=['auto', 'bf16', 'fp16', 'fp32'])
    ap.add_argument('--force', action='store_true', help='repetir aunque ya exista el resultado')
    a = ap.parse_args()

    prec = auto_precision() if a.precision == 'auto' else a.precision
    (OUT / 'logs').mkdir(parents=True, exist_ok=True)
    train_py = Path(__file__).resolve().parent / 'ner_train.py'
    print(f'Precisión: {prec} | modelos: {a.models} | semillas: {a.seeds}\n', flush=True)

    failed = []
    for seed in a.seeds:                      # semilla fuera: primero todos los modelos con una semilla
        for alias in a.models:
            name = f'ner_{alias}_s{seed}'
            if (OUT / f'preds_dev-set2_{name}.json').exists() and not a.force:
                print(f'[saltado] {name} (ya terminado)', flush=True)
                continue
            # Las opciones globales van ANTES de "train"
            cmd = [sys.executable, '-u', str(train_py), '--corpus-dir', a.corpus_dir,
                   '--model', MODELS[alias], '--name', name, '--precision', prec,
                   'train', '--seed', str(seed), '--lr', str(a.lr), '--epochs', str(a.epochs),
                   '--patience', str(a.patience), '--batch-size', str(a.batch_size)]
            print(f'\n=== {name} · {MODELS[alias]} ===', flush=True)
            t0 = time.time()
            tail = []
            with open(OUT / 'logs' / f'{name}.log', 'w', encoding='utf-8') as log:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1)
                for line in proc.stdout:
                    log.write(line)
                    tail = (tail + [line])[-15:]
                    if line.lstrip().startswith('ép') and 'paso' in line:
                        continue              # no inundar la pantalla con la barra de progreso
                    print(line, end='', flush=True)
                proc.wait()
            if proc.returncode != 0:
                failed.append(name)
                print(f'\n!!! {name} FALLÓ (código {proc.returncode}). Últimas líneas:\n' + ''.join(tail), flush=True)
            else:
                print(f'--- {name} terminado en {(time.time() - t0) / 60:.1f} min', flush=True)

    print('\nHecho.' + (f' Fallaron: {failed}' if failed else ' Todo correcto.'))


if __name__ == '__main__':
    main()