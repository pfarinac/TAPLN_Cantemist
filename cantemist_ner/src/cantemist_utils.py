"""
cantemist_utils.py
Utilidades comunes del proyecto CANTEMIST: parsers, carga con caché,
gold de NER y evaluación (P/R/F1 micro, coincidencia exacta).

Uso como script (auditoría de datos):
    python cantemist_utils.py --corpus-dir /ruta/a/cantemist audit
"""
import argparse
import os
import pickle
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ['train-set', 'dev-set1', 'dev-set2', 'test-set']
NER_LABEL = 'MORFOLOGIA_NEOPLASIA'
CODE_RE = re.compile(r'\b(\d{4}/\d{1,2}(?:/[A-Za-z0-9]+)?)\b')


# --------------------------------------------------------------------------
# Modelo de datos y parsers
# --------------------------------------------------------------------------
@dataclass
class Entity:
    id: str
    label: str
    spans: List[Tuple[int, int]]
    text: str
    code: Optional[str] = None

    @property
    def start(self):
        return self.spans[0][0]

    @property
    def end(self):
        return self.spans[-1][1]

    @property
    def discontinuous(self):
        return len(self.spans) > 1


def parse_ann(path) -> List[Entity]:
    """Parsea un .ann Brat (NER o NORM). code=None si no hay normalización."""
    entities: Dict[str, Entity] = {}
    notes: Dict[str, str] = {}
    with open(path, encoding='utf-8') as f:
        for raw in f:
            line = raw.rstrip('\n').rstrip('\r')
            if not line.strip():
                continue
            if line.startswith('T'):
                parts = line.split('\t')
                if len(parts) < 3:
                    continue
                tid, meta, text = parts[0], parts[1], '\t'.join(parts[2:])
                label, _, offs = meta.partition(' ')
                spans = []
                for seg in offs.split(';'):
                    nums = seg.split()
                    if len(nums) == 2:
                        spans.append((int(nums[0]), int(nums[1])))
                if spans:
                    entities[tid] = Entity(tid, label, spans, text)
            elif line.startswith('#'):
                m = re.match(r'^#\S+\s+\S+\s+(T\d+)\s*(.*)$', line.replace('\t', ' '))
                if m:
                    cm = CODE_RE.search(m.group(2))
                    if cm:
                        notes[m.group(1)] = cm.group(1)
    for tid, code in notes.items():
        if tid in entities:
            entities[tid].code = code
    return sorted(entities.values(), key=lambda e: (e.start, e.end))


def parse_coding_tsv(path) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').rstrip('\r').split('\t')
            if len(parts) < 2:
                continue
            doc, code = parts[0].strip(), parts[1].strip()
            if not CODE_RE.search(code):  # cabecera u otra línea no válida
                continue
            out.setdefault(doc.replace('.txt', ''), set()).add(code)
    return out


# --------------------------------------------------------------------------
# Carga (con caché local)
# --------------------------------------------------------------------------
def _read_text(corpus_dir: Path, split: str, doc_id: str) -> Optional[str]:
    for sub in ('cantemist-ner', 'cantemist-norm', 'text-files'):
        p = corpus_dir / split / sub / f'{doc_id}.txt'
        if p.exists():
            return p.read_text(encoding='utf-8')
    return None


def load_split(corpus_dir: Path, split: str):
    ner_dir = corpus_dir / split / 'cantemist-ner'
    norm_dir = corpus_dir / split / 'cantemist-norm'
    doc_ids = set()
    for d in (ner_dir, norm_dir):
        if d.exists():
            doc_ids |= {p.stem for p in d.glob('*.ann')} | {p.stem for p in d.glob('*.txt')}
    docs = {}
    for i, doc_id in enumerate(sorted(doc_ids), 1):
        if i % 100 == 0:
            print(f'  {split}: {i}/{len(doc_ids)}', flush=True)
        f_ner, f_norm = ner_dir / f'{doc_id}.ann', norm_dir / f'{doc_id}.ann'
        docs[doc_id] = {
            'text': _read_text(corpus_dir, split, doc_id),
            'ner': parse_ann(f_ner) if f_ner.exists() else [],
            'norm': parse_ann(f_norm) if f_norm.exists() else [],
        }
    coding: Dict[str, Set[str]] = {}
    cdir = corpus_dir / split / 'cantemist-coding'
    if cdir.exists():
        for t in cdir.glob('*.tsv'):
            for k, v in parse_coding_tsv(t).items():
                coding.setdefault(k, set()).update(v)
    return docs, coding


def get_corpus_dir(arg: Optional[str]) -> Path:
    p = arg or os.environ.get('CANTEMIST_DIR')
    if not p:
        raise SystemExit('Indica el corpus con --corpus-dir o con la variable de entorno CANTEMIST_DIR')
    p = Path(p)
    if not p.exists():
        raise SystemExit(f'No existe: {p}')
    return p


def load_all(corpus_dir: Path, refresh: bool = False):
    """Devuelve (DATA, CODING). La primera vez parsea y guarda caché en ROOT/cache."""
    cache = ROOT / 'cache' / 'cantemist_parsed.pkl'
    if cache.exists() and not refresh:
        try:
            with open(cache, 'rb') as f:
                d = pickle.load(f)
            return d['DATA'], d['CODING']
        except Exception as ex:  # caché antigua o corrupta: se vuelve a parsear
            print(f'Caché inválida ({ex!r}); se vuelve a parsear.')
    DATA, CODING = {}, {}
    for s in SPLITS:
        if (corpus_dir / s).exists():
            print(f'Cargando {s}...', flush=True)
            DATA[s], CODING[s] = load_split(corpus_dir, s)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, 'wb') as f:
        pickle.dump({'DATA': DATA, 'CODING': CODING}, f)
    print('Caché guardada en', cache)
    return DATA, CODING


# --------------------------------------------------------------------------
# Gold para NER
# --------------------------------------------------------------------------
def ner_gold_spans(doc: dict) -> List[Tuple[int, int]]:
    """Spans gold de NER de un documento.
    En train hay 36 documentos con anotación en cantemist-norm pero no en cantemist-ner,
    así que usamos la unión de ambos (sin duplicados), que en dev/test coincide con NER."""
    spans = {(e.start, e.end) for e in doc['ner']} | {(e.start, e.end) for e in doc['norm']}
    return sorted(spans)


def ner_gold(DATA, split: str) -> Dict[str, List[Tuple[int, int]]]:
    return {k: ner_gold_spans(d) for k, d in DATA[split].items()}


def is_inconsistent_empty(doc_id: str, doc: dict, coding: Dict[str, Set[str]]) -> bool:
    """Documento sin ninguna entidad en los .ann pero con códigos en el TSV de coding."""
    return not ner_gold_spans(doc) and bool(coding.get(doc_id))


# --------------------------------------------------------------------------
# Evaluación (P/R/F1 micro, coincidencia exacta)
# --------------------------------------------------------------------------
def prf(gold: set, pred: set):
    tp = len(gold & pred)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {'P': round(p, 4), 'R': round(r, 4), 'F1': round(f, 4),
            'TP': tp, 'FP': len(pred) - tp, 'FN': len(gold) - tp}


def _ner_set(preds):
    return {(doc, s, e) for doc, lst in preds.items() for (s, e, *_) in lst}


def _norm_set(preds):
    return {(doc, s, e, c) for doc, lst in preds.items() for (s, e, c) in lst if c}


def _coding_set(preds):
    return {(doc, c) for doc, codes in preds.items() for c in codes}


def evaluate_ner(gold, pred):
    """gold/pred: {doc_id: [(start, end, ...), ...]}"""
    return prf(_ner_set(gold), _ner_set(pred))


def evaluate_norm(gold, pred):
    """gold/pred: {doc_id: [(start, end, code), ...]}"""
    return prf(_norm_set(gold), _norm_set(pred))


def evaluate_coding(gold, pred):
    """gold/pred: {doc_id: {codes}}"""
    return prf(_coding_set(gold), _coding_set(pred))


# --------------------------------------------------------------------------
# Salida en formato Brat
# --------------------------------------------------------------------------
def write_ann_files(out_dir: Path, texts: Dict[str, str], preds: Dict[str, list]):
    out_dir.mkdir(parents=True, exist_ok=True)
    for doc_id, text in texts.items():
        lines = []
        for i, (s, e, *_) in enumerate(sorted(preds.get(doc_id, [])), 1):
            lines.append(f'T{i}\t{NER_LABEL} {s} {e}\t{text[s:e]}')
        (out_dir / f'{doc_id}.ann').write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')


# --------------------------------------------------------------------------
# Auditoría de datos
# --------------------------------------------------------------------------
def audit(corpus_dir: Path, DATA, CODING):
    print('\n=== Auditoría NER vs NORM vs CODING ===')
    for s, docs in DATA.items():
        n_empty = n_empty_inc = 0
        ner_not_in_norm = norm_not_in_ner = 0
        docs_ner_only_in_norm = 0
        for k, d in docs.items():
            a = {(e.start, e.end) for e in d['ner']}
            b = {(e.start, e.end) for e in d['norm']}
            ner_not_in_norm += len(a - b)
            norm_not_in_ner += len(b - a)
            if not a and b:
                docs_ner_only_in_norm += 1
            if not (a | b):
                n_empty += 1
                if CODING[s].get(k):
                    n_empty_inc += 1
        print(f'{s:9s} docs={len(docs):4d} | sin ninguna entidad={n_empty:3d} '
              f'(con códigos en el TSV: {n_empty_inc:3d}) | '
              f'spans NER no están en NORM={ner_not_in_norm} | spans NORM no están en NER={norm_not_in_ner} '
              f'| docs con NORM pero sin NER={docs_ner_only_in_norm}')

    print('\n=== Menciones NORM sin código: líneas crudas del .ann ===')
    shown = 0
    for k, d in DATA['train-set'].items():
        for e in d['norm']:
            if e.code:
                continue
            raw_path = corpus_dir / 'train-set' / 'cantemist-norm' / f'{k}.ann'
            lines = raw_path.read_text(encoding='utf-8').splitlines()
            rel = [l for l in lines if re.search(rf'\b{e.id}\b', l)]
            print(f'{k} {e.id} {e.text!r}')
            for l in rel:
                print('   ', repr(l))
            shown += 1
            break
        if shown >= 3:
            break
    if not shown:
        print('(ninguna)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus-dir')
    ap.add_argument('--refresh', action='store_true', help='reparsear ignorando la caché')
    ap.add_argument('cmd', choices=['audit'])
    a = ap.parse_args()
    corpus = get_corpus_dir(a.corpus_dir)
    DATA, CODING = load_all(corpus, refresh=a.refresh)
    audit(corpus, DATA, CODING)


if __name__ == '__main__':
    # Se reimporta como módulo para que la caché guarde Entity como cantemist_utils.Entity
    # (y no como __main__.Entity, que otros scripts no podrían leer).
    import cantemist_utils
    cantemist_utils.main()
