"""E27: smaller vectors + exact reranking of the top-K (re-embedding the decompressed text).

With a budget of ~10 s per query, the 1st stage can be a minimal binary index (128-256 bits/doc)
and the 2nd can decompress the K best candidates and embed them again with EmbeddingGemma 2.
Since decompression is lossless, re-embedding gives the original embedding. Here it is simulated with
the saved float embeddings (emb/g2_*.f16.npy), with no approximations.
Corpus: the 3 months (2024-01, 2010-06, 2015-06), all with the same bits (a single tier).
Queries: eval/{2024-01,2010-06,2015-06}.jsonl (relevant = comments in the thread). Vector only.
"""
import json, os, numpy as np, pyarrow.parquet as pq

import sys
MONTHS = {'2024-01': 2024, '2010-06': 2010, '2015-06': 2015}
# embedding set: g2 = full text (E27a); g2p128 = 128-token LM prefixes (E30a, embed_prefix.py),
# used both for the codes and for the reranking
TAG = sys.argv[1] if len(sys.argv) > 1 else 'g2'
BITS = [768, 512, 384, 256, 192, 128]
KS = [0, 50, 100, 200]


def queries(m):
    qs = [json.loads(l) for l in open(f'eval/{m}.jsonl', encoding='utf-8-sig') if l.strip()]
    Q = np.load(f'emb/q_eval_{m}.npy').astype(np.float32)
    return qs, Q / np.linalg.norm(Q, axis=1, keepdims=True)


def month(m, y):
    E = np.load(f'emb/{TAG}_{m}.f16.npy', mmap_mode='r')
    ids = np.load(f'emb/{TAG}_{m}.ids.npy')
    t = pq.read_table(f'data/{y}/{m}.parquet', columns=['id', 'parent', 'dead'])
    pid, par, dead = (t[c].to_pylist() for c in ['id', 'parent', 'dead'])
    idx = {i: k for k, i in enumerate(pid)}
    root = np.full(len(ids), -1, np.int64)
    alive = np.ones(len(ids), bool)
    for j, i in enumerate(ids):
        k = idx[int(i)]
        alive[j] = not dead[k]
        p, cur = par[k], k
        while p and p in idx:
            cur = idx[p]; p = par[cur]
        root[j] = pid[cur] if not p else -1
    return E, ids, root, alive


data = {m: month(m, y) for m, y in MONTHS.items()}
E = np.concatenate([np.asarray(data[m][0]) for m in MONTHS])           # float16, N x 768
ids = np.concatenate([data[m][1] for m in MONTHS])
root = np.concatenate([data[m][2] for m in MONTHS])
alive = np.concatenate([data[m][3] for m in MONTHS])
En = E.astype(np.float32)
En /= np.linalg.norm(En, axis=1, keepdims=True)
print(f'{len(ids):,} docs with a vector (3 months), embeddings {TAG}', flush=True)


def binary_scores(Q, bits, chunk=60000):
    q = Q[:, :bits] / np.linalg.norm(Q[:, :bits], axis=1, keepdims=True)
    out = np.empty((len(Q), len(En)), np.float32)
    for s in range(0, len(En), chunk):
        S = np.where(E[s:s + chunk, :bits] > 0, 1.0, -1.0).astype(np.float32)
        out[:, s:s + chunk] = q @ S.T / np.sqrt(bits)
    return np.where(alive[None, :], out, -9.0)


def run(qs, Q, bits):
    S = binary_scores(Q, bits)
    res = {}
    for K in KS:
        p10, rr = np.zeros(len(qs)), np.zeros(len(qs))
        for qi, q in enumerate(qs):
            s = np.where(ids == q['story'], -9.0, S[qi])
            if K == 0:
                top = np.argpartition(-s, 10)[:10]
                top = top[np.argsort(-s[top])]
            else:
                cand = np.argpartition(-s, K)[:K]
                exact = En[cand] @ Q[qi]                              # re-embedding = original embedding
                top = cand[np.argsort(-exact)[:10]]
            rel = root[top] == q['story']
            p10[qi] = rel.mean()
            rr[qi] = next((1 / (i + 1) for i, r in enumerate(rel) if r), 0)
        res[K] = (p10, rr)
    return res


def boot_ci(d, B=4000, seed=0):
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (B, len(d)))].mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


evals = {m: queries(m) for m in MONTHS}
per = {}
for bits in BITS:
    for em, (qs, Q) in evals.items():
        for K, v in run(qs, Q, bits).items():
            per[bits, K, em] = v
    print(f'{bits} bits done', flush=True)

base = {em: per[768, 0, em][0] for em in MONTHS}   # today: 768 bits, no reranking
print(f'\nvector P@10 (mean over the 900 queries: 2024 / 2010 / 2015) and difference against 768 bits without reranking [95 % CI]')
print(f'{"bits":>5} {"B/doc":>6} ' + ''.join(f'{"top-" + str(K) if K else "no rerank":>34}' for K in KS))
for bits in BITS:
    cells = []
    for K in KS:
        p = np.concatenate([per[bits, K, em][0] for em in MONTHS])
        d = p - np.concatenate([base[em] for em in MONTHS])
        lo, hi = boot_ci(d)
        cells.append(f'{p.mean():.3f} ({d.mean():+.3f} [{lo:+.3f},{hi:+.3f}])')
    print(f'{bits:>5} {bits // 8:>6} ' + ''.join(f'{c:>34}' for c in cells), flush=True)
print('\nper query set (P@10 / MRR):')
for bits in BITS:
    for K in KS:
        row = ' '.join(f'{em}: {per[bits, K, em][0].mean():.3f}/{per[bits, K, em][1].mean():.3f}' for em in MONTHS)
        print(f'  {bits:>4} bits, {"top-" + str(K) if K else "no rerank":>13}: {row}')
# ceiling: exact float 768 over everything
p = []
for em, (qs, Q) in evals.items():
    S = En @ Q.T
    for qi, q in enumerate(qs):
        s = np.where((ids == q['story']) | ~alive, -9.0, S[:, qi])
        top = np.argpartition(-s, 10)[:10]
        p.append((root[top] == q['story']).mean())
print(f'\nceiling (exact float 768, 3,072 B/doc): P@10 {np.mean(p):.3f}')
