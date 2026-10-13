"""Genera consultas de evaluación (JSONL) para `hnz eval`: título de una story -> relevantes =
comentarios de su hilo. Uso: python make_eval.py data/2024/2024-01.parquet 300 > eval/2024-01.jsonl"""
import sys, json, collections, numpy as np, pyarrow.parquet as pq

path, n = sys.argv[1], int(sys.argv[2])
t = pq.read_table(path, columns=['id', 'type', 'parent', 'title', 'text', 'dead'])
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
for r in stories[:n]:
    print(json.dumps({'q': title[idx[r]], 'story': r, 'n_rel': cnt[r]}))
