"""E30c: calidad de la consulta COMPACTA completa (candidatos + reordenación + fusión) y ajuste de la fusión.

Simula, con datos reales de los 3 meses de muestra y el tokenizador del motor (hnz.terms), la consulta
del perfil compacto:
  1. candidatos por vectores: top-100 con códigos binarios de 256 bits (puntuación asimétrica exacta);
  2. candidatos por palabras: top-20 del índice de términos RAROS (df <= 0,02 %, sin frecuencia);
  3. reordenación de la unión: coseno exacto del embedding del PREFIJO (128 tokens del LM, E30a) +
     w * b / (b + c), con b = BM25 de TODAS las palabras de la consulta sobre el texto del prefijo
     (IDF global, norma de longitud del documento completo, K1 = 1,2, B = 0,75, como el motor).
Rejilla de (w, c). Consultas: eval/{2024-01,2010-06,2015-06}.jsonl (relevantes = comentarios del hilo).
Referencia: la búsqueda híbrida actual (índice completo, texto completo) sobre los mismos 3 meses.
Uso: python experimentos/bench_fusion.py
"""
import collections, html, json, math, re
import numpy as np, pyarrow.parquet as pq
import hnz

MONTHS = {'2024-01': 2024, '2010-06': 2010, '2015-06': 2015}
P, KV, KB, BITS = 128, 100, 20, 256
K1, B = 1.2, 0.75
WS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
CS = [5.0, 10.0, 20.0, 40.0]
from transformers import AutoTokenizer
lmtok = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')
clean = lambda s: re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', s))).strip()


def plain(body):
    """Texto legible del cuerpo en la representación del LM (como experimentos/embed_prefix.py)."""
    if '<a href' in body or '<p>' in body:
        return clean(body)
    body = re.sub(r'</?i>|<pre><code>|</code></pre>', '', body)
    body = re.sub(r'<([^>|]+)\|([^>]*)>', r'\2 (\1)', body)
    body = re.sub(r'<([^>]+)>\d+>', r'\1', body)
    body = re.sub(r'<([^>\s]+)>', r'\1', body)
    return clean(html.escape(body, quote=False))


def norm(n):            # norma de 1 byte del motor (bm25.rs: norm_encode / norm_decode)
    q = min(255, round(16 * math.log2(1 + n)))
    return max(0.0, 2 ** (q / 16) - 1)


# ---- corpus: ids con embedding de prefijo, texto completo (Parquet) y texto del LM (lmdump)
ids, E, root, alive, full_txt, lm_s = [], [], [], [], [], []
for m, y in MONTHS.items():
    e = np.load(f'emb/g2p{P}_{m}.f16.npy')
    mids = np.load(f'emb/g2p{P}_{m}.ids.npy')
    t = pq.read_table(f'data/{y}/{m}.parquet', columns=['id', 'parent', 'dead', 'title', 'url', 'text'])
    pid, par, dead, title, url, text = (t[c].to_pylist() for c in ['id', 'parent', 'dead', 'title', 'url', 'text'])
    idx = {i: k for k, i in enumerate(pid)}
    lms = {}
    with open(f'data_lm/full_{m}.jsonl', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            lms[d['id']] = d['s']
    for j, i in enumerate(mids):
        k = idx[int(i)]
        p, cur = par[k], k
        while p and p in idx:
            cur = idx[p]; p = par[cur]
        ids.append(int(i)); root.append(pid[cur] if not p else -1); alive.append(not dead[k])
        full_txt.append(hnz.index_text(title[k] or '', url[k] or '', text[k] or ''))
        lm_s.append(lms[int(i)])
    E.append(e)
ids, root, alive = np.array(ids), np.array(root), np.array(alive)
E = np.concatenate(E).astype(np.float32)
E /= np.linalg.norm(E, axis=1, keepdims=True)
N = len(ids)
print(f'{N:,} documentos', flush=True)

# ---- df global y longitudes (texto completo, tokenizador del motor)
df = collections.Counter()
dl = np.zeros(N, np.float32)
for d, t in enumerate(full_txt):
    ts = hnz.terms(t)
    dl[d] = norm(len(ts))
    df.update(set(ts))
avgdl = float(np.mean([len(hnz.terms(t)) for t in full_txt[::50]]))   # el motor usa la media exacta; muestra
idf = lambda t: math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1.0)
RARE = math.ceil(0.0002 * N)
post = collections.defaultdict(list)                                 # índice de términos raros (sin tf)
for d, t in enumerate(full_txt):
    for term in set(hnz.terms(t)):
        if df[term] <= RARE:
            post[term].append(d)
print(f'{len(df):,} términos; raros (df <= {RARE}): {len(post):,} con {sum(map(len, post.values())):,} postings', flush=True)

_pref = {}


def prefix_tf(d):
    """Frecuencias de los términos en el texto del PREFIJO (lo que la reordenación descomprime)."""
    if d not in _pref:
        title, url, body = (lm_s[d].split('\n', 2) + ['', ''])[:3]
        head = lmtok(f'{title}\n{url}\n', add_special_tokens=False)['input_ids']
        b = lmtok(body, add_special_tokens=False)['input_ids']
        if len(b) > P or len(head) + len(b) > P + 96:
            body = lmtok.decode(b[:max(0, min(P, P + 96 - len(head)))])
        host = url.split('://')[1].split('/')[0] if '://' in url else ''
        _pref[d] = collections.Counter(hnz.terms(f'{title} {host} {plain(body)}'))
    return _pref[d]


def bm25(qterms, tf, d):
    return sum(idf(t) * tf[t] * (K1 + 1) / (tf[t] + K1 * (1 - B + B * dl[d] / avgdl)) for t in qterms if tf.get(t))


# ---- consultas
evals = {}
for m in MONTHS:
    qs = [json.loads(l) for l in open(f'eval/{m}.jsonl', encoding='utf-8-sig') if l.strip()]
    Q = np.load(f'emb/q_eval_{m}.npy').astype(np.float32)
    evals[m] = (qs, Q / np.linalg.norm(Q, axis=1, keepdims=True))

S = np.where(E[:, :BITS] > 0, 1.0, -1.0).astype(np.float32) / np.sqrt(BITS)
res = {(w, c): {m: [] for m in MONTHS} for w in WS for c in CS}
recall = []
for m, (qs, Q) in evals.items():
    q256 = Q[:, :BITS] / np.linalg.norm(Q[:, :BITS], axis=1, keepdims=True)
    for qi, q in enumerate(qs):
        ok = alive & (ids != q['story'])
        s1 = np.where(ok, S @ q256[qi], -9.0)
        vc = np.argpartition(-s1, KV)[:KV]
        qterms = list(dict.fromkeys(hnz.terms(q['q'])))
        kb = collections.Counter()
        for t in qterms:
            for d in post.get(t, ()):
                if ok[d]:
                    kb[d] += idf(t) * (K1 + 1) / (1 + K1 * (1 - B + B * dl[d] / avgdl))
        cands = list(dict.fromkeys(list(vc) + [d for d, _ in kb.most_common(KB)]))
        cos = E[cands] @ Q[qi]
        b = np.array([bm25(qterms, prefix_tf(d), d) for d in cands])
        rel = root[cands] == q['story']
        recall.append(rel.sum() / max(1, (root[ok] == q['story']).sum()))
        for w in WS:
            for c in CS:
                sc = cos + w * b / (b + c)
                top = np.argsort(-sc)[:10]
                r = rel[top]
                res[w, c][m].append((r.mean(), next((1 / (i + 1) for i, x in enumerate(r) if x), 0.0)))
    print(f'{m} listo', flush=True)

print(f'\nconsulta compacta: top-{KV} vectores ({BITS} bits) + top-{KB} términos raros, reordenación con '
      f'prefijos de {P} tokens; cobertura de relevantes en los candidatos: {np.mean(recall):.3f}')
print(f'{"w":>5} {"c":>5}  ' + '  '.join(f'{m:>15}' for m in MONTHS) + f'  {"media P@10":>11}')
best = None
for w in WS:
    for c in CS:
        row = [np.array(res[w, c][m]) for m in MONTHS]
        mean = np.mean([r[:, 0].mean() for r in row])
        print(f'{w:5.2f} {c:5.0f}  ' + '  '.join(f'{r[:, 0].mean():.3f}/{r[:, 1].mean():.3f}' for r in row) + f'  {mean:11.3f}', flush=True)
        if best is None or mean > best[0]:
            best = (mean, w, c)
print(f'\nmejor fusión: w={best[1]}, c={best[2]} (P@10 media {best[0]:.3f}); referencia actual (hnz eval, '
      f'índice y texto completos, 3 meses): 0,278 / 0,208 / 0,249 (media 0,245)')
