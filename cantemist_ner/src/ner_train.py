"""
ner_train.py
CANTEMIST-NER como etiquetado de secuencias (BIO) con un encoder biomédico en español.

    python ner_train.py --corpus-dir /ruta/cantemist train
    python ner_train.py --corpus-dir /ruta/cantemist predict --split dev-set2
    python ner_train.py --corpus-dir /ruta/cantemist predict --split test-set --final

Protocolo: entrenar en train-set, elegir época con dev-set1 y reportar en dev-set2 (no usado para elegir).
test-set solo se toca con --final.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from cantemist_utils import (ROOT, evaluate_ner, get_corpus_dir, is_inconsistent_empty, load_all,
                             ner_gold, ner_gold_spans, write_ann_files, prf)

LABELS = ['O', 'B', 'I']
DEFAULT_MODEL = 'PlanTL-GOB-ES/roberta-base-biomedical-clinical-es'


# ==========================================================================
# Lógica sin torch (tokenización, etiquetas BIO, ventanas, decodificación)
# ==========================================================================
def encode_document(tokenizer, text):
    """Tokeniza el documento completo (sin truncar) y devuelve ids y offsets de carácter."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    return list(enc['input_ids']), [tuple(o) for o in enc['offset_mapping']]


def label_tokens(offsets, spans):
    """Etiquetas BIO por token (0=O, 1=B, 2=I). Si dos spans solapan se conserva el más largo."""
    labels = [0] * len(offsets)
    if not offsets:
        return labels, 0
    starts = np.array([o[0] for o in offsets])
    ends = np.array([o[1] for o in offsets])
    skipped = 0
    for s, e in sorted(spans, key=lambda x: -(x[1] - x[0])):
        idx = np.nonzero((starts < e) & (ends > s))[0]
        if idx.size == 0 or any(labels[i] for i in idx):
            skipped += 1
            continue
        labels[idx[0]] = 1
        for i in idx[1:]:
            labels[i] = 2
    return labels, skipped


def decode_spans(offsets, labels, text=None):
    """Convierte etiquetas BIO por token en spans de caracteres (start, end).
    Si se pasa el texto, se recortan espacios en los bordes (algunos tokenizadores BPE incluyen
    el espacio inicial en el offset del token)."""
    spans, cur = [], None
    for (a, b), l in zip(offsets, labels):
        if l == 1 or (l == 2 and cur is None):
            if cur:
                spans.append(tuple(cur))
            cur = [a, b]
        elif l == 2:
            cur[1] = b
        else:
            if cur:
                spans.append(tuple(cur))
                cur = None
    if cur:
        spans.append(tuple(cur))
    if text is not None:
        trimmed = []
        for s, e in spans:
            while s < e and text[s].isspace():
                s += 1
            while e > s and text[e - 1].isspace():
                e -= 1
            if e > s:
                trimmed.append((s, e))
        spans = trimmed
    return spans


def make_windows(n_tokens, max_content, overlap):
    """Ventanas deslizantes (start, end) sobre n_tokens con solapamiento."""
    if n_tokens <= max_content:
        return [(0, n_tokens)]
    step = max_content - overlap
    wins, s = [], 0
    while True:
        e = min(s + max_content, n_tokens)
        wins.append((s, e))
        if e == n_tokens:
            return wins
        s += step


def predict_spans(text, tokenizer, probs_fn, max_len, overlap):
    """Predice spans de un documento. probs_fn(list[list[int]]) -> list[np.ndarray (len, 3)]."""
    ids, offs = encode_document(tokenizer, text)
    if not ids:
        return []
    wins = make_windows(len(ids), max_len - 2, overlap)
    seqs = [[tokenizer.cls_token_id] + ids[s:e] + [tokenizer.sep_token_id] for s, e in wins]
    probs = probs_fn(seqs)
    acc = np.zeros((len(ids), len(LABELS)))
    cnt = np.zeros(len(ids))
    for (s, e), p in zip(wins, probs):
        acc[s:e] += p[1:-1]
        cnt[s:e] += 1
    return decode_spans(offs, (acc / cnt[:, None]).argmax(1), text)


def tokenization_ceiling(tokenizer, docs):
    """F1 máximo alcanzable con este tokenizador: gold -> etiquetas BIO -> spans."""
    gold, rec = {}, {}
    skipped = 0
    for k, d in docs.items():
        g = ner_gold_spans(d)
        ids, offs = encode_document(tokenizer, d['text'])
        labels, sk = label_tokens(offs, g)
        skipped += sk
        gold[k] = g
        rec[k] = decode_spans(offs, labels, d['text'])
    return evaluate_ner(gold, rec), skipped


def build_examples(tokenizer, docs, max_len, overlap, exclude=()):
    """Ejemplos de entrenamiento: ventanas con [CLS] ... [SEP] y etiquetas (-100 en especiales)."""
    examples, skipped_total = [], 0
    for k, d in docs.items():
        if k in exclude:
            continue
        ids, offs = encode_document(tokenizer, d['text'])
        if not ids:
            continue
        labels, sk = label_tokens(offs, ner_gold_spans(d))
        skipped_total += sk
        for s, e in make_windows(len(ids), max_len - 2, overlap):
            examples.append(([tokenizer.cls_token_id] + ids[s:e] + [tokenizer.sep_token_id],
                             [-100] + labels[s:e] + [-100]))
    return examples, skipped_total


# ==========================================================================
# Parte con torch
# ==========================================================================
def pick_device(torch):
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def autocast_ctx(torch, device, precision):
    if device.type != 'cuda' or precision == 'fp32':
        return torch.autocast(device_type='cpu', enabled=False)
    dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
    return torch.autocast(device_type='cuda', dtype=dtype)


def resolve_precision(torch, device, precision):
    if precision != 'auto':
        return precision
    if device.type == 'cuda' and torch.cuda.is_bf16_supported():
        return 'bf16'
    return 'fp32'


def make_probs_fn(torch, model, tokenizer, device, batch_size, precision):
    pad = tokenizer.pad_token_id

    @torch.no_grad()
    def probs_fn(seqs):
        model.eval()
        out = []
        for i in range(0, len(seqs), batch_size):
            chunk = seqs[i:i + batch_size]
            L = max(len(s) for s in chunk)
            ids = torch.full((len(chunk), L), pad, dtype=torch.long)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            for j, s in enumerate(chunk):
                ids[j, :len(s)] = torch.tensor(s)
                att[j, :len(s)] = 1
            with autocast_ctx(torch, device, precision):
                logits = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits
            p = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            out.extend(p[j, :len(s)] for j, s in enumerate(chunk))
        return out
    return probs_fn


def predict_split(torch, model, tokenizer, docs, device, args, precision):
    probs_fn = make_probs_fn(torch, model, tokenizer, device, args.eval_batch_size, precision)
    preds = {}
    for i, (k, d) in enumerate(docs.items(), 1):
        preds[k] = [(s, e) for s, e in predict_spans(d['text'], tokenizer, probs_fn, args.max_len, args.overlap)]
        if i % 50 == 0:
            print(f'    predichos {i}/{len(docs)}', flush=True)
    return preds


def save_predictions(split, name, docs, preds):
    out = ROOT / 'outputs' / 'ner'
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f'preds_{split}_{name}.json', 'w', encoding='utf-8') as f:
        json.dump({k: [list(x) for x in v] for k, v in preds.items()}, f)
    write_ann_files(out / f'ann_{split}_{name}', {k: d['text'] for k, d in docs.items()}, preds)
    print(f'  predicciones guardadas en {out}/preds_{split}_{name}.json y ann_{split}_{name}/')


def report(split, docs, preds):
    gold = ner_gold(DATA_REF['DATA'], split)
    r = evaluate_ner(gold, preds)
    print(f'  {split}: P={r["P"]:.4f} R={r["R"]:.4f} F1={r["F1"]:.4f}  (TP={r["TP"]} FP={r["FP"]} FN={r["FN"]})', flush=True)
    return r


DATA_REF = {}


def cmd_train(args):
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    corpus = get_corpus_dir(args.corpus_dir)
    DATA, CODING = load_all(corpus)
    DATA_REF['DATA'] = DATA
    device = pick_device(torch)
    precision = resolve_precision(torch, device, args.precision)
    print(f'Dispositivo: {device} | precisión: {precision} | modelo: {args.model}')
    if device.type == 'cpu':
        print('AVISO: sin GPU el entrenamiento de un roberta-base será muy lento (horas por época).')

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.model_max_length = 10 ** 9  # los documentos se trocean a mano en ventanas

    # --- datos de entrenamiento ---
    train_docs = {}
    exclude = set()
    for s in args.train_splits:
        for k, d in DATA[s].items():
            key = f'{s}/{k}'
            train_docs[key] = d
            if not args.keep_inconsistent and is_inconsistent_empty(k, d, CODING[s]):
                exclude.add(key)
    print(f'Train: {len(train_docs)} docs; excluidos {len(exclude)} sin entidades pero con códigos en el TSV '
          f'(posibles anotaciones ausentes). Usa --keep-inconsistent para incluirlos.')

    ceil, sk = tokenization_ceiling(tokenizer, DATA[args.val_split])
    print(f'Techo de tokenización en {args.val_split}: F1={ceil["F1"]} (P={ceil["P"]} R={ceil["R"]}), '
          f'spans solapados descartados: {sk}')

    examples, skipped = build_examples(tokenizer, train_docs, args.max_len, args.overlap, exclude)
    print(f'Ventanas de entrenamiento: {len(examples)} | spans solapados descartados: {skipped}')

    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(LABELS),
        id2label={i: l for i, l in enumerate(LABELS)}, label2id={l: i for i, l in enumerate(LABELS)})
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = (len(examples) + args.batch_size - 1) // args.batch_size
    total = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    scaler = torch.amp.GradScaler('cuda') if (precision == 'fp16' and device.type == 'cuda') else None
    pad = tokenizer.pad_token_id

    def collate(batch):
        L = max(len(x[0]) for x in batch)
        ids = torch.full((len(batch), L), pad, dtype=torch.long)
        lab = torch.full((len(batch), L), -100, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for j, (x, y) in enumerate(batch):
            ids[j, :len(x)] = torch.tensor(x)
            lab[j, :len(y)] = torch.tensor(y)
            att[j, :len(x)] = 1
        return ids, att, lab

    loader = torch.utils.data.DataLoader(examples, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    model_dir = ROOT / 'models' / args.name
    best_f1, bad = -1.0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for step, (ids, att, lab) in enumerate(loader, 1):
            ids, att, lab = ids.to(device), att.to(device), lab.to(device)
            with autocast_ctx(torch, device, precision):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            opt.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            run += loss.item()
            if step % 5 == 0:
                print(f'  ép {epoch} paso {step}/{steps_per_epoch} loss={run / step:.4f}', flush=True)
        print(f'Época {epoch}: loss={run / steps_per_epoch:.4f} ({time.time() - t0:.0f}s). Validando en {args.val_split}...')
        preds = predict_split(torch, model, tokenizer, DATA[args.val_split], device, args, precision)
        r = report(args.val_split, DATA[args.val_split], preds)
        history.append({'epoch': epoch, 'loss': run / steps_per_epoch, **r})
        if r['F1'] > best_f1:
            best_f1, bad = r['F1'], 0
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            print(f'  * mejor modelo (F1={best_f1:.4f}) guardado en {model_dir}')
        else:
            bad += 1
            if bad >= args.patience:
                print(f'Early stopping (sin mejora en {args.patience} épocas).')
                break
    (ROOT / 'outputs' / 'ner').mkdir(parents=True, exist_ok=True)
    with open(ROOT / 'outputs' / 'ner' / f'history_{args.name}.json', 'w') as f:
        json.dump(history, f, indent=1)

    # --- evaluación final con el mejor modelo ---
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device)
    print('\n=== Resultados con el mejor modelo ===')
    for split in (args.val_split, args.heldout_split):
        preds = predict_split(torch, model, tokenizer, DATA[split], device, args, precision)
        report(split, DATA[split], preds)
        save_predictions(split, args.name, DATA[split], preds)


def cmd_predict(args):
    if args.split == 'test-set' and not args.final:
        raise SystemExit('test-set solo se evalúa una vez al final: añade --final si ya está todo decidido.')
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    corpus = get_corpus_dir(args.corpus_dir)
    DATA, CODING = load_all(corpus)
    DATA_REF['DATA'] = DATA
    device = pick_device(torch)
    precision = resolve_precision(torch, device, args.precision)
    model_dir = ROOT / 'models' / args.name
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.model_max_length = 10 ** 9
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device)
    preds = predict_split(torch, model, tokenizer, DATA[args.split], device, args, precision)
    report(args.split, DATA[args.split], preds)
    save_predictions(args.split, args.name, DATA[args.split], preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus-dir')
    ap.add_argument('--name', default='ner_roberta_bio', help='nombre de la ejecución (carpeta en models/)')
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--max-len', type=int, default=512)
    ap.add_argument('--overlap', type=int, default=64)
    ap.add_argument('--eval-batch-size', type=int, default=16)
    ap.add_argument('--precision', choices=['auto', 'bf16', 'fp16', 'fp32'], default='auto')
    sub = ap.add_subparsers(dest='cmd', required=True)

    t = sub.add_parser('train')
    t.add_argument('--epochs', type=int, default=10)
    t.add_argument('--batch-size', type=int, default=8)
    t.add_argument('--lr', type=float, default=5e-5)
    t.add_argument('--patience', type=int, default=3)
    t.add_argument('--seed', type=int, default=42)
    t.add_argument('--train-splits', nargs='+', default=['train-set'])
    t.add_argument('--val-split', default='dev-set1')
    t.add_argument('--heldout-split', default='dev-set2')
    t.add_argument('--keep-inconsistent', action='store_true')
    t.set_defaults(fn=cmd_train)

    p = sub.add_parser('predict')
    p.add_argument('--split', required=True)
    p.add_argument('--final', action='store_true')
    p.set_defaults(fn=cmd_predict)

    args = ap.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
