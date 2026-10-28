"""Real quality (not just fidelity): query = title of a story, relevant = comments in THAT thread.
Metric: 'same thread' precision@10 and recall@100. Compares float vs compressed vectors vs BM25."""
import sys, os, re, html, time, collections, math
import numpy as np, pyarrow.parquet as pq

MODEL, NSTORY, NDIST = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
t = pq.read_table('data/2024/2024-01.parquet', columns=['id', 'type', 'parent', 'title', 'text', 'dead'])
ids, typ, par, title, text, dead = (t[c].to_pylist() for c in ['id', 'type', 'parent', 'title', 'text', 'dead'])
N = len(ids); idx_of = {i: k for k, i in enumerate(ids)}
root = []
for k in range(N):
    p, cur = par[k], k
    while p and p in idx_of:
        cur = idx_of[p]; p = par[cur]
    root.append(ids[cur] if not p else -1)
clean = lambda s: html.unescape(re.sub(r'<[^>]+>', ' ', s)).strip()
thread = collections.defaultdict(list)
for k in range(N):
    if typ[k] == 2 and not dead[k] and root[k] > 0 and len(text[k]) > 80:
        thread[root[k]].append(k)
rng = np.random.default_rng(1)
stories = [r for r, v in thread.items() if 15 <= len(v) <= 60 and title[idx_of[r]]]
rng.shuffle(stories); stories = stories[:NSTORY]
sset = set(stories)
docs_k = [k for r in stories for k in thread[r]]
pool = [k for k in range(N) if typ[k] == 2 and not dead[k] and len(text[k]) > 80 and root[k] not in sset]
docs_k += list(rng.choice(pool, NDIST, replace=False))
lab = np.array([root[k] for k in docs_k])
docs = [clean(text[k]) for k in docs_k]
queries = [title[idx_of[r]] for r in stories]
print(f'{MODEL}: {len(stories)} queries, {len(docs):,} docs ({len(docs)-NDIST:,} relevant in threads)')

tag = f'q_{MODEL.replace("/", "_").replace(":", "")}_{NSTORY}_{NDIST}'
if os.path.exists(f'emb/{tag}_D.npy'):
    D, Q = np.load(f'emb/{tag}_D.npy'), np.load(f'emb/{tag}_Q.npy')
else:
    order = np.argsort([len(d) for d in docs]); t0 = time.time()
    if MODEL.startswith('st:'):
        import torch; from sentence_transformers import SentenceTransformer
        torch.set_num_threads(os.cpu_count())
        m = SentenceTransformer(MODEL[3:], device='cpu'); m.max_seq_length = 256
        E = m.encode([{'text': docs[i]} for i in order], prompt_name='document', batch_size=32)
        Q = m.encode([{'text': q} for q in queries], prompt_name='query')
    else:
        from fastembed import TextEmbedding
        m = TextEmbedding(MODEL, max_length=256, threads=os.cpu_count())
        E = np.array(list(m.embed([docs[i] for i in order], batch_size=32)))
        Q = np.array(list(m.query_embed(queries)))
    D = np.empty_like(E); D[order] = E
    print(f'embeddings {len(docs)/(time.time()-t0):.1f} docs/s')
    np.save(f'emb/{tag}_D.npy', D.astype(np.float32)); np.save(f'emb/{tag}_Q.npy', np.asarray(Q, np.float32))
D, Q = D.astype(np.float32), np.asarray(Q, np.float32)

def evaluate(scores):
    top = np.argsort(-scores, 1)[:, :100]
    p10 = np.mean([np.mean(lab[top[i, :10]] == stories[i]) for i in range(len(stories))])
    r100 = np.mean([np.sum(lab[top[i]] == stories[i]) / np.sum(lab == stories[i]) for i in range(len(stories))])
    return p10, r100

n = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
print(f'{"variant":<40}{"B/vec":>7}{"P@10":>7}{"R@100":>7}')
full = D.shape[1]
for d in sorted({full, 512, 256, 128} & set(range(1, full + 1)), reverse=True):
    Dd, Qd = n(D[:, :d]), n(Q[:, :d])
    S = Qd @ Dd.T
    print(f'{d}d float32{"":<30}{4*d:>7}' + '{:>7.3f}{:>7.3f}'.format(*evaluate(S)))
    print(f'{d}d binary asymmetric{"":<20}{d//8:>7}' + '{:>7.3f}{:>7.3f}'.format(*evaluate(Qd @ np.sign(Dd).T)))
    # binary + rescore of the top-100 by re-embedding (full vector)
    c = np.argsort(-(Qd @ np.sign(Dd).T), 1)[:, :100]
    S2 = np.full_like(S, -9); np.put_along_axis(S2, c, np.einsum('qd,qrd->qr', n(Q), n(D)[c]), 1)
    print(f'{d}d binary + re-embed top-100{"":<12}{d//8:>7}' + '{:>7.3f}{:>7.3f}'.format(*evaluate(S2)))

# reference BM25 (keywords)
TOK = re.compile(r"[a-z0-9]+")
dt = [collections.Counter(TOK.findall(d.lower())) for d in docs]
dfc = collections.Counter(w for c in dt for w in c)
avgdl = np.mean([sum(c.values()) for c in dt]); Nd = len(docs)
S = np.zeros((len(queries), Nd))
for qi, q in enumerate(queries):
    for w in set(TOK.findall(q.lower())):
        if w not in dfc: continue
        idf = math.log(1 + (Nd - dfc[w] + .5) / (dfc[w] + .5))
        for j, c in enumerate(dt):
            f = c.get(w)
            if f:
                dl = sum(c.values()); S[qi, j] += idf * f * 2.2 / (f + 1.2 * (.25 + .75 * dl / avgdl))
print(f'{"BM25 (keywords)":<40}{"~35":>7}' + '{:>7.3f}{:>7.3f}'.format(*evaluate(S)))
# hybrid: RRF of BM25 + asymmetric binary vector (full dim)
def ranks(s):
    r = np.empty_like(s); r[np.arange(len(s))[:, None], np.argsort(-s, 1)] = np.arange(s.shape[1]); return r
H = 1 / (60 + ranks(S)) + 1 / (60 + ranks(n(Q) @ np.sign(n(D)).T))
print(f'{"hybrid BM25 + binary "+str(full)+"d (RRF)":<40}{full//8:>7}' + '{:>7.3f}{:>7.3f}'.format(*evaluate(H)))
