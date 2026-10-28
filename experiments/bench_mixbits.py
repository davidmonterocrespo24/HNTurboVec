"""E26: comparable scores across segments with different bit counts (768 hot / 512 cold).

Simulates hnz's asymmetric binary vector search (score q·sign(d)/sqrt(bits)) over the 3 months with
the real EmbeddingGemma 2 embeddings: 2024-01 hot at 768 bits, 2010-06 and 2015-06 cold at 512 bits
(MRL prefix). Three query sets (make_eval.py): titles from 2024-01, 2010-06 and 2015-06 → relevant =
comments in the thread, which sit in the hot month OR in a cold one. So a correction that only
sinks the cold months shows up (2024 gets better but 2010/2015 get worse).
Variants for the 512-bit segments:
  - all at 768 bits: reference (what the cold months would give with 768 bits)
  - renormalized: cosine in 512 dims (hnz before E26)
  - OLS: E[s768 | s512] = a·s + b (regression; shrinks the extremes)
  - moments: same mean and deviation as s768 in the top (no shrinking)
  - not renormalized: the query keeps its 768-dim norm (biased against the cold months)
The constants are fitted with the 2024-01 queries over the 2024-01 docs (top-200 of each query),
where both scores exist for the same documents.
"""
import json, os, sys, numpy as np, pyarrow.parquet as pq

MONTHS = {'2024-01': 2024, '2010-06': 2010, '2015-06': 2015}
HOT = '2024-01'


def queries(m):
    qs = [json.loads(l) for l in open(f'eval/{m}.jsonl', encoding='utf-8-sig') if l.strip()]
    cache = f'emb/q_eval_{m}.npy'
    if os.path.exists(cache):
        Q = np.load(cache)
    else:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer('google/embeddinggemma-2', device='cpu').float()
        Q = st.encode([{'text': q['q']} for q in qs], prompt_name='query', normalize_embeddings=True).astype(np.float32)
        np.save(cache, Q)
    Q = Q.astype(np.float32)
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


def scores(Q, E, alive, bits, renorm=True, chunk=60000):
    q = Q[:, :bits]
    if renorm:
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
    out = np.empty((len(Q), len(E)), np.float32)
    for s in range(0, len(E), chunk):
        S = np.where(np.asarray(E[s:s + chunk, :bits], np.float32) > 0, 1.0, -1.0).astype(np.float32)
        out[:, s:s + chunk] = q @ S.T / np.sqrt(bits)
    return np.where(alive[None, :], out, -9.0)


data = {m: month(m, y) for m, y in MONTHS.items()}
evals = {m: queries(m) for m in MONTHS}
print(', '.join(f'{m}: {len(d[1]):,} docs' for m, d in data.items()), flush=True)

# --- calibration constants (2024-01 queries over 2024-01, top-200 of each query)
qs, Q = evals[HOT]
E, ids, root, alive = data[HOT]
s768, s512 = scores(Q, E, alive, 768), scores(Q, E, alive, 512)
xs, ys = [], []
for qi in range(len(Q)):
    sel = np.union1d(np.argpartition(-s768[qi], 200)[:200], np.argpartition(-s512[qi], 200)[:200])
    xs.append(s512[qi, sel]); ys.append(s768[qi, sel])
x, y = np.concatenate(xs), np.concatenate(ys)
a_ols, b_ols = np.polyfit(x, y, 1)
a_mom = y.std() / x.std()
b_mom = y.mean() - a_mom * x.mean()
print(f'OLS:      s768 ≈ {a_ols:.4f}·s512 {b_ols:+.4f}   (r = {np.corrcoef(x, y)[0, 1]:.3f})')
print(f'moments:  s768 ≈ {a_mom:.4f}·s512 {b_mom:+.4f}', flush=True)
del s768, s512

VARIANTS = {
    'all at 768 bits (reference)': lambda Q, d: scores(Q, d[0], d[3], 768),
    'renormalized (hnz before E26)': lambda Q, d: scores(Q, d[0], d[3], 512),
    'OLS a·s+b': lambda Q, d: a_ols * scores(Q, d[0], d[3], 512) + b_ols,
    'moments a·s+b': lambda Q, d: a_mom * scores(Q, d[0], d[3], 512) + b_mom,
    'not renormalized': lambda Q, d: scores(Q, d[0], d[3], 512, renorm=False),
}


def evaluate(qs, mats):
    """mats: {month: scores nq x N}; relevant = docs whose thread is the query's story."""
    order = list(mats)
    allroot = np.concatenate([data[m][2] for m in order])
    allid = np.concatenate([data[m][1] for m in order])
    p10, rr = np.zeros(len(qs)), np.zeros(len(qs))
    for qi, q in enumerate(qs):
        s = np.concatenate([mats[m][qi] for m in order])
        s = np.where(allid == q['story'], -9.0, s)
        top = np.argpartition(-s, 10)[:10]
        top = top[np.argsort(-s[top])]
        rel = allroot[top] == q['story']
        p10[qi] = rel.sum() / 10
        rr[qi] = next((1 / (i + 1) for i, r in enumerate(rel) if r), 0)
    return p10, rr   # per query (for paired comparisons)


def boot_ci(d, B=4000, seed=0):
    """95 % CI of the mean of a paired per-query difference (bootstrap over queries)."""
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (B, len(d)))].mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


print()
print(f'{"variant (cold at 512 bits)":<36}' + ''.join(f'{"queries " + m:>24}' for m in MONTHS) + f'{"mean":>16}')
per = {}   # (variant, query set) -> P@10 per query
for name, f in VARIANTS.items():
    row, ps = [], []
    for em, (qs, Q) in evals.items():
        mats = {HOT: scores(Q, data[HOT][0], data[HOT][3], 768)}
        for m in MONTHS:
            if m != HOT:
                mats[m] = f(Q, data[m])
        p, rr = evaluate(qs, mats)
        per[name, em] = p
        row.append(f'P@10 {p.mean():.3f} MRR {rr.mean():.3f}')
        ps.append(p.mean())
    print(f'{name:<36}' + ''.join(f'{r:>24}' for r in row) + f'{"P@10 " + format(np.mean(ps), ".3f"):>16}', flush=True)

# paired differences against hnz's score (renormalized), with a bootstrap 95 % CI
BASE = 'renormalized (hnz before E26)'
print(f'\nP@10 difference against "{BASE}" (95 % CI, paired bootstrap over queries):')
print(f'{"variant":<36}' + ''.join(f'{"queries " + m:>26}' for m in MONTHS) + f'{"all 900":>26}')
for name in VARIANTS:
    if name == BASE:
        continue
    cells, alld = [], []
    for em in MONTHS:
        d = per[name, em] - per[BASE, em]
        alld.append(d)
        lo, hi = boot_ci(d)
        cells.append(f'{d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]')
    d = np.concatenate(alld)
    lo, hi = boot_ci(d)
    print(f'{name:<36}' + ''.join(f'{c:>26}' for c in cells) + f'{d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]'.rjust(26), flush=True)
