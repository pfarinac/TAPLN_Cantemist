
import argparse
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cantemist_utils import ROOT, get_corpus_dir, load_all, ner_gold_spans, prf
from ner_analyze import dict_baseline, overlaps
from run_ner_experiments import MODELS

OUT = ROOT / 'outputs' / 'ner'
FIGS = OUT / 'figs'
TRAIN, DEV1, DEV2 = 'train-set', 'dev-set1', 'dev-set2'
CMAP = plt.get_cmap('tab10')



def load_preds(split, name):
    p = OUT / f'preds_{split}_{name}.json'
    if not p.exists():
        return None
    with open(p, encoding='utf-8') as f:
        return {k: [tuple(x) for x in v] for k, v in json.load(f).items()}


def sets(DATA, split, preds):
    docs = DATA[split]
    G = {k: set(ner_gold_spans(d)) for k, d in docs.items()}
    P = {k: set(map(tuple, preds.get(k, []))) for k in docs}
    gold = {(k, s, e) for k, v in G.items() for s, e in v}
    pred = {(k, s, e) for k, v in P.items() for s, e in v}
    return G, P, gold, pred


def breakdown(G, P, gold, pred):
    """Reparto de errores: frontera mal (hay solape con gold/pred) vs no detectada / inventada."""
    fn_f = fn_n = fp_f = fp_i = 0
    for k, s, e in gold - pred:
        if any(overlaps((s, e), p) for p in P[k]):
            fn_f += 1
        else:
            fn_n += 1
    for k, s, e in pred - gold:
        if any(overlaps((s, e), g) for g in G[k]):
            fp_f += 1
        else:
            fp_i += 1
    return {'TP': len(gold & pred), 'FN_no_detectada': fn_n, 'FN_frontera': fn_f,
            'FP_frontera': fp_f, 'FP_inventada': fp_i}


def recall_groups(DATA, split, gold, pred, train_mentions):
    docs = DATA[split]
    txt = lambda x: docs[x[0]]['text'][x[1]:x[2]]
    groups = {
        'vista en train': lambda x: txt(x).lower().strip() in train_mentions,
        'no vista en train': lambda x: txt(x).lower().strip() not in train_mentions,
        '1 palabra': lambda x: len(txt(x).split()) == 1,
        '2 palabras': lambda x: len(txt(x).split()) == 2,
        '3-4 palabras': lambda x: 3 <= len(txt(x).split()) <= 4,
        '5+ palabras': lambda x: len(txt(x).split()) >= 5,
    }
    rec, sizes = {}, {}
    for nm, cond in groups.items():
        sel = [x for x in gold if cond(x)]
        rec['rec: ' + nm] = sum(x in pred for x in sel) / max(len(sel), 1)
        sizes[nm] = len(sel)
    return rec, sizes


def evaluate_system(DATA, preds1, preds2, train_mentions):
    row = {}
    G1, P1, gold1, pred1 = sets(DATA, DEV1, preds1)
    m1 = prf(gold1, pred1)
    row.update({f'{k}_dev1': m1[k] for k in ('P', 'R', 'F1')})
    G2, P2, gold2, pred2 = sets(DATA, DEV2, preds2)
    m2 = prf(gold2, pred2)
    row.update({f'{k}_dev2': m2[k] for k in ('P', 'R', 'F1')})
    row.update(breakdown(G2, P2, gold2, pred2))
    rec, sizes = recall_groups(DATA, DEV2, gold2, pred2, train_mentions)
    row.update(rec)
    return row, (G2, P2), sizes


def parse_log(name):
    info = {'ceiling': np.nan, 'time_min': np.nan}
    p = OUT / 'logs' / f'{name}.log'
    if p.exists():
        t = p.read_text(encoding='utf-8', errors='ignore')
        m = re.search(r'Techo de tokenización en \S+: F1=([\d.]+)', t)
        if m:
            info['ceiling'] = float(m.group(1))
        secs = [int(x) for x in re.findall(r'Época \d+: loss=[\d.]+ \((\d+)s\)', t)]
        if secs:
            info['time_min'] = sum(secs) / 60
    return info


def load_history(name):
    p = OUT / f'history_{name}.json'
    if not p.exists():
        return []
    with open(p, encoding='utf-8') as f:
        return json.load(f)

def per_doc(G, P, order):
    arr = []
    for k in order:
        tp = len(G[k] & P[k])
        arr.append((tp, len(P[k]) - tp, len(G[k]) - tp))
    return np.array(arr)


def f1_of(arr):
    tp, fp, fn = arr.sum(0)
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return 2 * p * r / (p + r) if p + r else 0


def bootstrap_diff(a, b, n=2000, seed=0):
    """Diferencia de F1 (a - b) remuestreando documentos. Devuelve media e IC 95 %."""
    rng = np.random.default_rng(seed)
    N = len(a)
    d = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        d.append(f1_of(a[idx]) - f1_of(b[idx]))
    return float(np.mean(d)), np.percentile(d, [2.5, 97.5])

def fig_f1(df, aliases, path):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x, w = np.arange(len(aliases)), 0.38
    for off, tag, lab in ((-w / 2, 'dev1', 'dev1 (aquí se elige la época)'),
                          (w / 2, 'dev2', 'dev2 (no usado para elegir)')):
        g = [df[df.alias == a][f'F1_{tag}'] for a in aliases]
        means = [v.mean() for v in g]
        stds = [np.nan_to_num(v.std(ddof=1)) if len(v) > 1 else 0 for v in g]
        ax.bar(x + off, means, w, yerr=stds, capsize=3, label=lab)
        for xi, v in zip(x + off, g):
            ax.scatter([xi] * len(v), v, color='k', s=9, zorder=3)
        for xi, m in zip(x + off, means):
            ax.text(xi, m + 0.004, f'{100 * m:.1f}', ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(aliases)
    ax.set_ylim(max(0, df[['F1_dev1', 'F1_dev2']].min().min() - 0.06), 1.0)
    ax.set_ylabel('F1 estricto (span exacto)')
    ax.set_title('F1 por modelo (barras = media, puntos = semillas, línea = desv. típica)')
    ax.legend()
    ax.grid(axis='y', alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_pr(df, aliases, path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for i, a in enumerate(aliases):
        s = df[df.alias == a]
        ax.scatter(s.R_dev2, s.P_dev2, color=CMAP(i), s=70, label=a)
    ax.set_xlabel('Recall (dev2)')
    ax.set_ylabel('Precisión (dev2)')
    ax.set_title('Precisión vs recall: ¿qué modelo es más "prudente" o más "generoso"?')
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_curves(hist, aliases, path):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, a in enumerate(aliases):
        hs = hist.get(a, [])
        if not hs:
            continue
        L = max(len(h) for h in hs)
        f1 = np.full((len(hs), L), np.nan)
        ls = np.full((len(hs), L), np.nan)
        for j, h in enumerate(hs):
            f1[j, :len(h)] = [e['F1'] for e in h]
            ls[j, :len(h)] = [e['loss'] for e in h]
        ep = np.arange(1, L + 1)
        axs[0].plot(ep, np.nanmean(f1, 0), marker='o', color=CMAP(i), label=a)
        axs[1].plot(ep, np.nanmean(ls, 0), marker='o', color=CMAP(i), label=a)
    axs[0].set_title('F1 en dev1 por época (media de semillas)')
    axs[1].set_title('Loss de entrenamiento por época')
    for ax in axs:
        ax.set_xlabel('época')
        ax.grid(alpha=.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_errors(df, aliases, path):
    cats = [('FN_no_detectada', 'FN: mención no detectada'), ('FN_frontera', 'FN: frontera mal'),
            ('FP_frontera', 'FP: frontera mal'), ('FP_inventada', 'FP: inventada')]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bottom = np.zeros(len(aliases))
    for c, lab in cats:
        vals = np.array([df[df.alias == a][c].mean() for a in aliases])
        ax.bar(aliases, vals, bottom=bottom, label=lab)
        bottom += vals
    ax.set_ylabel('nº de errores en dev2 (media de semillas)')
    ax.set_title('Tipos de error (cuanto menor la barra, mejor)')
    ax.legend()
    ax.grid(axis='y', alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_recall_groups(df, aliases, sizes, path):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, keys, title in ((axs[0], ['vista en train', 'no vista en train'], 'Recall según si la mención se vio en train'),
                            (axs[1], ['1 palabra', '2 palabras', '3-4 palabras', '5+ palabras'], 'Recall según longitud de la mención')):
        x, w = np.arange(len(keys)), 0.8 / len(aliases)
        for i, a in enumerate(aliases):
            vals = [df[df.alias == a]['rec: ' + k].mean() for k in keys]
            ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=a, color=CMAP(i))
        ax.set_xticks(x)
        ax.set_xticklabels([f'{k}\n(n={sizes[k]})' for k in keys])
        ax.set_ylim(0, 1.02)
        ax.set_title(title)
        ax.grid(axis='y', alpha=.3)
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_bootstrap(rows, ref, path):
    fig, ax = plt.subplots(figsize=(7, 0.7 * len(rows) + 1.8))
    for i, (a, m, lo, hi) in enumerate(rows):
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt='o', capsize=4, color=CMAP(i + 1))
    ax.axvline(0, color='k', lw=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel(f'ΔF1 frente a {ref} en dev2 (IC 95 % bootstrap por documentos)')
    ax.set_title('¿La diferencia es real o ruido? (si el intervalo cruza 0, no se puede afirmar)')
    ax.grid(axis='x', alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def md_table(summary):
    cols = ['Modelo', 'semillas', 'techo tok.', 'P dev2', 'R dev2', 'F1 dev2', 'F1 dev1', 'mejor ép.', 'min/ejec.']
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    pm = lambda m, s: f'{100 * m:.2f}' + (f' ± {100 * s:.2f}' if not np.isnan(s) else '')
    for a, r in summary.iterrows():
        lines.append('| ' + ' | '.join([
            a, str(int(r['n'])),
            '-' if np.isnan(r['ceiling']) else f"{r['ceiling']:.4f}",
            pm(r['P_dev2_m'], r['P_dev2_s']), pm(r['R_dev2_m'], r['R_dev2_s']),
            pm(r['F1_dev2_m'], r['F1_dev2_s']), pm(r['F1_dev1_m'], r['F1_dev1_s']),
            '-' if np.isnan(r['best_epoch']) else f"{r['best_epoch']:.1f}",
            '-' if np.isnan(r['time_min']) else f"{r['time_min']:.1f}"]) + ' |')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus-dir')
    ap.add_argument('--ref', default='bioclin', help='modelo de referencia para el bootstrap')
    a = ap.parse_args()
    DATA, _ = load_all(get_corpus_dir(a.corpus_dir))
    FIGS.mkdir(parents=True, exist_ok=True)
    train_mentions = {d['text'][s:e].lower().strip() for d in DATA[TRAIN].values() for s, e in ner_gold_spans(d)}

    rows, store, hist = [], {}, {}
    
    print('Calculando baseline de diccionario...', flush=True)
    r, st, sizes = evaluate_system(DATA, dict_baseline(DATA, TRAIN, DEV1), dict_baseline(DATA, TRAIN, DEV2), train_mentions)
    rows.append(dict(alias='diccionario', seed=0, ceiling=np.nan, best_epoch=np.nan, time_min=np.nan, **r))
    store['diccionario'] = st

    for alias in MODELS:
        for p in sorted(OUT.glob(f'preds_{DEV2}_ner_{alias}_s*.json')):
            name = p.stem.replace(f'preds_{DEV2}_', '')
            seed = int(re.search(r'_s(\d+)$', name).group(1))
            p1, p2 = load_preds(DEV1, name), load_preds(DEV2, name)
            if p1 is None or p2 is None:
                print(f'(aviso) faltan predicciones de {name}; se ignora')
                continue
            r, st, _ = evaluate_system(DATA, p1, p2, train_mentions)
            h = load_history(name)
            info = parse_log(name)
            best = (int(np.argmax([e['F1'] for e in h])) + 1) if h else np.nan
            rows.append(dict(alias=alias, seed=seed, ceiling=info['ceiling'], best_epoch=best,
                             time_min=info['time_min'], **r))
            store.setdefault(alias, st)       # primera semilla, para el bootstrap
            hist.setdefault(alias, []).append(h)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'results_runs.csv', index=False)
    aliases_all = ['diccionario'] + [m for m in MODELS if m in set(df.alias)]
    aliases_mod = aliases_all[1:]

    agg = {}
    for c in ('P_dev2', 'R_dev2', 'F1_dev2', 'F1_dev1'):
        agg[c + '_m'] = (c, 'mean')
        agg[c + '_s'] = (c, lambda v: v.std(ddof=1) if len(v) > 1 else np.nan)
    summary = df.groupby('alias').agg(n=('seed', 'count'), ceiling=('ceiling', 'mean'),
                                      best_epoch=('best_epoch', 'mean'), time_min=('time_min', 'mean'),
                                      **agg).loc[aliases_all]
    summary.to_csv(OUT / 'results_summary.csv')
    table = md_table(summary)
    (OUT / 'results_table.md').write_text(table, encoding='utf-8')
    print('\n' + table)

    cols = ['TP', 'FN_no_detectada', 'FN_frontera', 'FP_frontera', 'FP_inventada'] + \
           [c for c in df.columns if c.startswith('rec: ')]
    detail = df.groupby('alias')[cols].mean().loc[aliases_all].round(3)
    detail.to_csv(OUT / 'results_detail.csv')
    print('\nDesglose de errores y recall por grupo (dev2, media de semillas):')
    print(detail.to_string())
    print('\nTamaño de los grupos en dev2:', sizes)

    boot = []
    if a.ref in store:
        order = sorted(DATA[DEV2])
        ref_arr = per_doc(*store[a.ref], order)
        print(f'\nBootstrap por documentos, ΔF1 frente a {a.ref} (dev2, primera semilla):')
        for al in aliases_all:
            if al == a.ref:
                continue
            m, (lo, hi) = bootstrap_diff(per_doc(*store[al], order), ref_arr)
            boot.append((al, m, lo, hi))
            veredicto = 'diferencia clara' if (lo > 0 or hi < 0) else 'NO concluyente'
            print(f'  {al:12s} ΔF1={100 * m:+.2f} pts  IC95=[{100 * lo:+.2f}, {100 * hi:+.2f}]  -> {veredicto}')
    else:
        print(f'\n(aviso) no hay resultados de la referencia "{a.ref}": se omite el bootstrap')

    fig_f1(df, aliases_all, FIGS / '1_f1_por_modelo.png')
    fig_pr(df, aliases_all, FIGS / '2_precision_vs_recall.png')
    fig_curves(hist, aliases_mod, FIGS / '3_curvas_entrenamiento.png')
    fig_errors(df, aliases_all, FIGS / '4_tipos_de_error.png')
    fig_recall_groups(df, aliases_all, sizes, FIGS / '5_recall_por_grupos.png')
    if boot:
        fig_bootstrap(boot, a.ref, FIGS / '6_bootstrap_vs_referencia.png')
    print(f'\nFiguras en {FIGS}\nTablas en {OUT} (results_table.md, results_summary.csv, results_detail.csv, results_runs.csv)')


if __name__ == '__main__':
    main()