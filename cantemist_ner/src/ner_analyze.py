"""
ner_analyze.py
Análisis de errores de CANTEMIST-NER.

    python ner_analyze.py --corpus-dir /ruta/cantemist --split dev-set2 --preds ../outputs/ner/preds_dev-set2_ner_roberta_bio.json
    python ner_analyze.py --corpus-dir /ruta/cantemist --split dev-set2 --system dict     # baseline de diccionario
"""
import argparse
import json
import re
from collections import Counter

from cantemist_utils import ROOT, get_corpus_dir, load_all, ner_gold_spans, prf


def dict_baseline(DATA, train_split, split):
    """Diccionario: todas las menciones de train (en minúsculas), emparejando la más larga primero."""
    terms = {e_text.lower().strip()
             for d in DATA[train_split].values()
             for e_text in (d['text'][s:e] for s, e in ner_gold_spans(d))}
    terms = sorted((t for t in terms if len(t) >= 3), key=len, reverse=True)
    pat = re.compile(r'(?<!\w)(' + '|'.join(re.escape(t) for t in terms) + r')(?!\w)', re.IGNORECASE)
    return {k: [(m.start(), m.end()) for m in pat.finditer(d['text'])] for k, d in DATA[split].items()}


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def relation(gold, pred):
    """Cómo es la predicción respecto al gold cuando solapan."""
    if pred[0] <= gold[0] and pred[1] >= gold[1]:
        return 'pred más larga que gold'
    if pred[0] >= gold[0] and pred[1] <= gold[1]:
        return 'pred más corta que gold'
    return 'solape parcial'


def analyze(DATA, split, preds, train_split, out):
    docs = DATA[split]
    # mención -> código (para ver errores por código)
    code_of = {(k, e.start, e.end): e.code for k, d in docs.items() for e in d['norm'] if e.code}
    train_mentions = {d['text'][s:e].lower().strip() for d in DATA[train_split].values()
                      for s, e in ner_gold_spans(d)}

    G = {k: set(ner_gold_spans(d)) for k, d in docs.items()}
    P = {k: set(map(tuple, preds.get(k, []))) for k in docs}

    gold_all = {(k, s, e) for k, v in G.items() for s, e in v}
    pred_all = {(k, s, e) for k, v in P.items() for s, e in v}
    strict = prf(gold_all, pred_all)

    fn = sorted(gold_all - pred_all)
    fp = sorted(pred_all - gold_all)

    # --- relajado: cuenta cualquier solape ---
    g_hit = sum(1 for k, s, e in gold_all if any(overlaps((s, e), p) for p in P[k]))
    p_hit = sum(1 for k, s, e in pred_all if any(overlaps((s, e), g) for g in G[k]))
    rp = p_hit / max(len(pred_all), 1)
    rr = g_hit / max(len(gold_all), 1)
    rf = 2 * rp * rr / (rp + rr) if rp + rr else 0.0

    w = lambda *a: print(*a, file=out) or print(*a)
    w(f'# Análisis NER · {split}')
    w(f'\nEstricto (span exacto): P={strict["P"]:.4f} R={strict["R"]:.4f} F1={strict["F1"]:.4f} '
      f'| TP={strict["TP"]} FP={strict["FP"]} FN={strict["FN"]}')
    w(f'Relajado (cualquier solape): P={rp:.4f} R={rr:.4f} F1={rf:.4f}')
    w('  -> la diferencia entre estricto y relajado son errores de FRONTERA; lo que queda en el relajado son '
      'menciones no detectadas o inventadas.')

    # --- FN ---
    w(f'\n## Falsos negativos ({len(fn)})')
    cats = Counter()
    for k, s, e in fn:
        ov = [p for p in P[k] if overlaps((s, e), p)]
        cats['frontera: ' + relation((s, e), ov[0]) if ov else 'no detectada'] += 1
    for c, n in cats.most_common():
        w(f'  {c:40s} {n:5d} ({100 * n / max(len(fn), 1):.1f}%)')

    # --- FP ---
    w(f'\n## Falsos positivos ({len(fp)})')
    cats = Counter()
    for k, s, e in fp:
        ov = [g for g in G[k] if overlaps((s, e), g)]
        cats['frontera: ' + relation(ov[0], (s, e)) if ov else 'inventada (sin solape con gold)'] += 1
    for c, n in cats.most_common():
        w(f'  {c:40s} {n:5d} ({100 * n / max(len(fp), 1):.1f}%)')

    # --- recall por longitud ---
    w('\n## Recall estricto por longitud de mención (palabras)')
    buckets = {'1': (1, 1), '2': (2, 2), '3-4': (3, 4), '5+': (5, 99)}
    fn_set = set(fn)
    for name, (lo, hi) in buckets.items():
        sel = [(k, s, e) for k, s, e in gold_all if lo <= len(docs[k]['text'][s:e].split()) <= hi]
        tp = sum(1 for x in sel if x not in fn_set)
        w(f'  {name:4s} n={len(sel):5d}  recall={tp / max(len(sel), 1):.4f}')

    # --- visto / no visto en train ---
    w('\n## Recall estricto según si la mención aparece (en minúsculas) en train')
    for name, cond in (('vista en train', True), ('NO vista en train', False)):
        sel = [(k, s, e) for k, s, e in gold_all if (docs[k]['text'][s:e].lower().strip() in train_mentions) == cond]
        tp = sum(1 for x in sel if x not in fn_set)
        w(f'  {name:18s} n={len(sel):5d}  recall={tp / max(len(sel), 1):.4f}')

    # --- top errores ---
    txt = lambda k, s, e: docs[k]['text'][s:e].lower().strip()
    w('\n## Top 15 FN (mención gold no encontrada exactamente)')
    for t, n in Counter(txt(*x) for x in fn).most_common(15):
        w(f'  {n:4d}  {t!r}')
    w('\n## Top 15 FP (predicción que no es gold)')
    for t, n in Counter(txt(*x) for x in fp).most_common(15):
        w(f'  {n:4d}  {t!r}')

    # --- FN por código ---
    w('\n## Códigos eCIE-O con más FN (enlaza con la subtarea NORM)')
    by_code, tot_code = Counter(), Counter()
    for k, s, e in gold_all:
        c = code_of.get((k, s, e))
        if c:
            tot_code[c] += 1
            if (k, s, e) in fn_set:
                by_code[c] += 1
    for c, n in by_code.most_common(10):
        w(f'  {c:10s} FN={n:4d} de {tot_code[c]:4d}  (recall {1 - n / tot_code[c]:.3f})')

    # --- ejemplos con contexto ---
    w('\n## 8 ejemplos de FN con contexto')
    for k, s, e in fn[:8]:
        t = docs[k]['text']
        w(f'  [{k}] ...{t[max(0, s - 40):s]}[[{t[s:e]}]]{t[e:e + 40]}...'.replace('\n', ' '))
    w('\n## 8 ejemplos de FP con contexto')
    for k, s, e in fp[:8]:
        t = docs[k]['text']
        w(f'  [{k}] ...{t[max(0, s - 40):s]}[[{t[s:e]}]]{t[e:e + 40]}...'.replace('\n', ' '))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus-dir')
    ap.add_argument('--split', default='dev-set2')
    ap.add_argument('--train-split', default='train-set')
    ap.add_argument('--preds', help='JSON de predicciones {doc: [[start, end], ...]}')
    ap.add_argument('--system', choices=['dict'], help='analizar un sistema generado al vuelo')
    a = ap.parse_args()
    if not a.preds and not a.system:
        raise SystemExit('Indica --preds o --system dict')

    DATA, _ = load_all(get_corpus_dir(a.corpus_dir))
    if a.system == 'dict':
        preds, name = dict_baseline(DATA, a.train_split, a.split), 'dict'
    else:
        with open(a.preds, encoding='utf-8') as f:
            preds = json.load(f)
        name = re.sub(r'^preds_', '', re.sub(r'\.json$', '', a.preds.replace('\\', '/').split('/')[-1]))
    out_path = ROOT / 'outputs' / 'ner' / f'analysis_{a.split}_{name}.txt'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as out:
        analyze(DATA, a.split, preds, a.train_split, out)
    print(f'\nGuardado en {out_path}')


if __name__ == '__main__':
    main()
