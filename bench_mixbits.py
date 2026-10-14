"""E26: puntuaciones comparables entre segmentos con distintos bits (768 caliente / 512 frío).

Simula la búsqueda vectorial binaria asimétrica de hnz (puntuación q·sign(d)/sqrt(bits)) sobre los
3 meses con los embeddings reales de EmbeddingGemma 2: 2024-01 caliente a 768 bits, 2010-06 y 2015-06
fríos a 512 bits (prefijo MRL). Tres conjuntos de consultas (make_eval.py): títulos de 2024-01, 2010-06
y 2015-06 → relevantes = comentarios del hilo, que están en el mes caliente O en uno frío; así una
corrección que solo hunda a los fríos se nota (mejora 2024 pero empeora 2010/2015).
Variantes para los segmentos de 512 bits:
  - todo a 768 bits: referencia (lo que darían los fríos con 768 bits)
  - renormalizada: coseno en 512 dims (hnz antes de E26)
  - OLS: E[s768 | s512] = a·s + b (regresión; contrae los extremos)
  - momentos: misma media y desviación que s768 en el top (sin contracción)
  - sin renormalizar: la consulta conserva su norma de 768 dims (sesgo en contra de los fríos)
Las constantes se ajustan con las consultas de 2024-01 sobre los docs de 2024-01 (top-200 de cada
consulta), donde existen ambas puntuaciones para los mismos documentos.
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

# --- constantes de calibración (consultas de 2024-01 sobre 2024-01, top-200 de cada consulta)
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
print(f'momentos: s768 ≈ {a_mom:.4f}·s512 {b_mom:+.4f}', flush=True)
del s768, s512

VARIANTS = {
    'todo a 768 bits (referencia)': lambda Q, d: scores(Q, d[0], d[3], 768),
    'renormalizada (hnz antes de E26)': lambda Q, d: scores(Q, d[0], d[3], 512),
    'OLS a·s+b': lambda Q, d: a_ols * scores(Q, d[0], d[3], 512) + b_ols,
    'momentos a·s+b': lambda Q, d: a_mom * scores(Q, d[0], d[3], 512) + b_mom,
    'sin renormalizar': lambda Q, d: scores(Q, d[0], d[3], 512, renorm=False),
}


def evaluate(qs, mats):
    """mats: {mes: puntuaciones nq x N}; relevantes = docs cuyo hilo es la story de la consulta."""
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
    return p10, rr   # por consulta (para comparaciones pareadas)


def boot_ci(d, B=4000, seed=0):
    """IC 95 % de la media de una diferencia pareada por consulta (bootstrap de consultas)."""
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (B, len(d)))].mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


print()
print(f'{"variante (frío a 512 bits)":<36}' + ''.join(f'{"consultas " + m:>24}' for m in MONTHS) + f'{"media":>16}')
per = {}   # (variante, conjunto) -> P@10 por consulta
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

# diferencias pareadas frente a la puntuación de hnz (renormalizada), con IC 95 % por bootstrap
BASE = 'renormalizada (hnz antes de E26)'
print(f'\ndiferencia de P@10 frente a "{BASE}" (IC 95 %, bootstrap pareado de consultas):')
print(f'{"variante":<36}' + ''.join(f'{"consultas " + m:>26}' for m in MONTHS) + f'{"las 900":>26}')
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
