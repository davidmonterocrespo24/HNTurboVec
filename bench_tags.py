"""Idea: ~1000 etiquetas (temas) + jerarquia de niveles.
(A) genera ordenes de almacenamiento por tema para medir compresion (bench_store los usa)
(B) enrutado: buscar solo dentro de las n etiquetas mas cercanas a la consulta -> calidad vs % escaneado
(C) jerarquia: 1000 etiquetas -> 40 areas, nombradas con palabras distintivas (c-TF-IDF)"""
import os, re, html, time, collections, math
import numpy as np, pyarrow.parquet as pq, faiss

NT, NAREA = 1000, 40
t = pq.read_table('data/2024/2024-01.parquet', columns=['id', 'type', 'parent', 'title', 'text', 'dead'])
ids, typ, par, title, text, dead = (t[c].to_pylist() for c in ['id', 'type', 'parent', 'title', 'text', 'dead'])
N = len(ids); idx_of = {i: k for k, i in enumerate(ids)}
root = []
for k in range(N):
    p, cur = par[k], k
    while p and p in idx_of:
        cur = idx_of[p]; p = par[cur]
    root.append(ids[cur] if not p else p)
clean = lambda s: html.unescape(re.sub(r'<[^>]+>', ' ', s)).strip()
docs = [clean((title[k] + '\n' + text[k]).strip()) for k in range(N)]

if os.path.exists('emb/all_potion.npy'):
    E = np.load('emb/all_potion.npy')
else:
    from fastembed import TextEmbedding
    m = TextEmbedding('minishlab/potion-retrieval-32M', max_length=256)
    t0 = time.time()
    E = np.array(list(m.embed([d or '-' for d in docs], batch_size=256)), dtype=np.float32)
    print(f'embeddings {N/(time.time()-t0):.0f} docs/s'); np.save('emb/all_potion.npy', E)
E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
d = E.shape[1]

# ---- etiquetas = k-means esferico (1000) ----
t0 = time.time()
km = faiss.Kmeans(d, NT, niter=20, spherical=True, seed=1, verbose=False); km.train(E)
C = km.centroids / np.linalg.norm(km.centroids, axis=1, keepdims=True)
sims = E @ C.T
top3 = np.argsort(-sims, 1)[:, :3]
tag = top3[:, 0]
print(f'k-means {NT} etiquetas: {time.time()-t0:.0f}s; tamano por etiqueta: mediana {int(np.median(np.bincount(tag, minlength=NT)))}, '
      f'max {np.bincount(tag).max()}')
print(f'coste de guardar etiquetas: 1 etiqueta = 10 bits/doc -> {49.3e6*10/8/1e9:.2f} GB; 3 etiquetas -> {49.3e6*30/8/1e9:.2f} GB')

# ---- (A) ordenes para compresion ----
# por tema: (etiqueta, id); hilo-por-tema: cada hilo entero va al tema mayoritario de sus mensajes
thr_tag = {}
cnt = collections.defaultdict(collections.Counter)
for k in range(N): cnt[root[k]][tag[k]] += 1
for r, c in cnt.items(): thr_tag[r] = c.most_common(1)[0][0]
order_topic = sorted(range(N), key=lambda k: (tag[k], ids[k]))
order_thr_topic = sorted(range(N), key=lambda k: (thr_tag[root[k]], root[k], ids[k]))
np.save('emb/order_topic.npy', np.array(order_topic)); np.save('emb/order_thread_topic.npy', np.array(order_thr_topic))

# ---- (C) jerarquia y nombres ----
TOK = re.compile(r"[a-z][a-z0-9+#\-]{2,}")
STOP = set('the and for that this with you are but not have was they what from can all just like your one its about more there their would will out has been some which when them than also any how who our get only very then these into other much because could even does most did those use way make think people don really thing things doesn isn didn being here should well still want need know see good time work same lot something many over going said too were where why while years year new using used etc http https www com org html'.split())
wc = [collections.Counter() for _ in range(NT)]
for k in range(N):
    wc[tag[k]].update(w for w in set(TOK.findall(docs[k].lower())) if w not in STOP)
dfw = collections.Counter(w for c in wc for w in c)
def name(i, n=4):
    tot = sum(wc[i].values()) or 1
    sc = {w: f / tot * math.log(NT / dfw[w]) for w, f in wc[i].items() if f >= 3}
    return ', '.join(sorted(sc, key=sc.get, reverse=True)[:n])
km2 = faiss.Kmeans(d, NAREA, niter=30, spherical=True, seed=2); km2.train(C)
area = km2.index.search(C, 1)[1][:, 0]
size = np.bincount(tag, minlength=NT)
print(f'\nJerarquia: {NT} etiquetas -> {NAREA} areas. Ejemplos (5 areas mas grandes, 4 etiquetas cada una):')
for a in np.argsort(-np.bincount(area, weights=size, minlength=NAREA))[:5]:
    ts = sorted(np.where(area == a)[0], key=lambda i: -size[i])
    print(f'  AREA {a} ({int(size[ts].sum()):,} docs, {len(ts)} etiquetas)')
    for i in ts[:4]:
        print(f'     - [{size[i]:>5} docs] {name(i)}')

# ---- (B) enrutado: consulta = titulo de story, relevantes = comentarios de su hilo ----
rng = np.random.default_rng(1)
thread = collections.defaultdict(list)
for k in range(N):
    if typ[k] == 2 and not dead[k] and len(text[k]) > 80: thread[root[k]].append(k)
stories = [r for r, v in thread.items() if 15 <= len(v) <= 60 and r in idx_of and title[idx_of[r]]]
rng.shuffle(stories); stories = stories[:300]
from fastembed import TextEmbedding
qm = TextEmbedding('minishlab/potion-retrieval-32M')
Q = np.array(list(qm.query_embed([title[idx_of[r]] for r in stories])), dtype=np.float32)
Q /= np.linalg.norm(Q, axis=1, keepdims=True)
mask = np.zeros(N, bool); mask[[idx_of[r] for r in stories]] = True  # no dejar que la story se encuentre a si misma
rootarr = np.array(root)
def p10(cands_scores):
    return np.mean([np.mean(rootarr[c] == stories[i]) for i, c in enumerate(cands_scores)])
S = Q @ E.T; S[:, mask] = -9
full = [np.argsort(-S[i])[:10] for i in range(len(stories))]
print(f'\n(B) Enrutado sobre {N:,} docs, {len(stories)} consultas (modelo potion, float)')
print(f'  busqueda completa:                         P@10 {p10(full):.3f}   escanea 100%')
QC = Q @ C.T
for multi in (1, 3):
    member = top3[:, :multi]
    for npb in (1, 3, 5, 10, 20, 50):
        res, scanned = [], 0
        for i in range(len(stories)):
            probe = np.argsort(-QC[i])[:npb]
            cand = np.where(np.isin(member, probe).any(1) & ~mask)[0]
            scanned += len(cand)
            res.append(cand[np.argsort(-S[i, cand])[:10]])
        print(f'  {multi} etiqueta(s)/doc, buscar en {npb:>2} etiquetas: P@10 {p10(res):.3f}   escanea {scanned/len(stories)/N:6.2%}')
# solo etiquetas, sin vectores (ordenar por similitud de etiqueta): el limite de usar etiquetas como unico indice
res = []
for i in range(len(stories)):
    s = QC[i][tag].copy(); s[mask] = -9
    res.append(np.argsort(-s, kind='stable')[:10])
print(f'  SOLO etiquetas (sin vectores por doc):     P@10 {p10(res):.3f}')
