"""E28a: a first stage of ~16-18 B/doc with product quantization (faiss) + exact reranking.

For generating the candidates that get reranked with the exact embedding (= re-embedding the
decompressed text), compares the 256-bit binary code (32 B, measured in E27a) with smaller PQ codes on
the 256-dimension MRL prefix: OPQ+PQ16 (16 B) and IVF4096 + PQ16 of the residual (2 + 16 B), with
exhaustive search (all lists). Same corpus and queries as bench_rescore.py.
"""
import json, numpy as np, pyarrow.parquet as pq, faiss

MONTHS = {'2024-01': 2024, '2010-06': 2010, '2015-06': 2015}
KS = [0, 100, 200]


def queries(m):
    qs = [json.loads(l) for l in open(f'eval/{m}.jsonl', encoding='utf-8-sig') if l.strip()]
    Q = np.load(f'emb/q_eval_{m}.npy').astype(np.float32)
    return qs, Q / np.linalg.norm(Q, axis=1, keepdims=True)


def month(m, y):
    E = np.load(f'emb/g2_{m}.f16.npy', mmap_mode='r')
    ids = np.load(f'emb/g2_{m}.ids.npy')
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
E = np.concatenate([np.asarray(data[m][0]) for m in MONTHS])
ids = np.concatenate([data[m][1] for m in MONTHS])
root = np.concatenate([data[m][2] for m in MONTHS])
alive = np.concatenate([data[m][3] for m in MONTHS])
En = E.astype(np.float32)
En /= np.linalg.norm(En, axis=1, keepdims=True)
D256 = En[:, :256] / np.linalg.norm(En[:, :256], axis=1, keepdims=True)
evals = {m: queries(m) for m in MONTHS}
print(f'{len(ids):,} docs; {sum(len(v[0]) for v in evals.values())} queries', flush=True)
faiss.omp_set_num_threads(16)
rng = np.random.default_rng(0)
train = np.ascontiguousarray(D256[rng.choice(len(D256), 200_000, replace=False)])


def make(spec):
    ix = faiss.index_factory(256, spec, faiss.METRIC_INNER_PRODUCT)
    ix.train(train)
    ix.add(np.ascontiguousarray(D256))
    if 'IVF' in spec:
        faiss.extract_index_ivf(ix).nprobe = faiss.extract_index_ivf(ix).nlist   # exhaustive
    return ix


def binary256():
    """Reference: 256-bit asymmetric binary (what hnz uses), as a home-made index."""
    S = np.where(E[:, :256] > 0, 1.0, -1.0).astype(np.float32)
    class B:
        def search(self, Q, k):
            sc = Q @ S.T
            top = np.argpartition(-sc, k, axis=1)[:, :k]
            order = np.argsort(-np.take_along_axis(sc, top, 1), axis=1)
            return None, np.take_along_axis(top, order, 1)   # like faiss: highest first
    return B()


def evaluate(ix):
    out = {K: [] for K in KS}
    for em, (qs, Q) in evals.items():
        Q256 = np.ascontiguousarray(Q[:, :256] / np.linalg.norm(Q[:, :256], axis=1, keepdims=True))
        _, I = ix.search(Q256, 260)
        for qi, q in enumerate(qs):
            cand = np.array([d for d in I[qi] if d >= 0 and alive[d] and ids[d] != q['story']])
            for K in KS:
                if K == 0:
                    top = cand[:10]                        # order of the 1st stage itself
                else:
                    c = cand[:K]
                    top = c[np.argsort(-(En[c] @ Q[qi]))[:10]]
                out[K].append((root[top] == q['story']).mean())
    return {K: np.mean(v) for K, v in out.items()}


for name, spec, B in [('binary 256 bits (hnz)', None, 32), ('OPQ16 + PQ16 (256d)', 'OPQ16,PQ16', 16),
                      ('IVF4096 + PQ16 residual (256d)', 'IVF4096,PQ16', 18), ('OPQ32 + PQ32 (256d)', 'OPQ32,PQ32', 32)]:
    ix = binary256() if spec is None else make(spec)
    r = evaluate(ix)
    print(f'  {name:<34} {B:>3} B/doc   P@10 no rerank {r[0]:.3f} | top-100 {r[100]:.3f} | top-200 {r[200]:.3f}', flush=True)
