"""Genera consultas de evaluación (JSONL, UTF-8) para `hnz eval`: título de una story -> relevantes =
comentarios de su hilo.
Uso: python make_eval.py data/2024/2024-01.parquet 300 --out eval/2024-01.jsonl
(escribe el fichero directamente en UTF-8: redirigir la salida con '>' en PowerShell 5.1 la convierte a UTF-16)"""
import argparse, json, collections, sys
import numpy as np, pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument('parquet')
ap.add_argument('n', type=int)
ap.add_argument('--out', help='fichero de salida (UTF-8); si se omite, salida estándar')
a = ap.parse_args()
t = pq.read_table(a.parquet, columns=['id', 'type', 'parent', 'title', 'text', 'dead'])
ids, typ, par, title, text, dead = (t[c].to_pylist() for c in ['id', 'type', 'parent', 'title', 'text', 'dead'])
idx = {i: k for k, i in enumerate(ids)}
root = []
for k in range(len(ids)):
    p, cur = par[k], k
    while p and p in idx:
        cur = idx[p]; p = par[cur]
    root.append(ids[cur] if not p else -1)
cnt = collections.Counter(root[k] for k in range(len(ids)) if typ[k] == 2 and not dead[k] and len(text[k]) > 80 and root[k] > 0)
stories = [r for r, c in cnt.items() if 15 <= c <= 60 and title[idx[r]]]
rng = np.random.default_rng(1)
rng.shuffle(stories)
lines = [json.dumps({'q': title[idx[r]], 'story': r, 'n_rel': cnt[r]}) for r in stories[:a.n]]
if a.out:
    with open(a.out, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
else:
    sys.stdout.reconfigure(encoding='utf-8')
    print('\n'.join(lines))
