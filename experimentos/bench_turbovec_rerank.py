"""(1) turbovec (TurboQuant, Google 2504.19874) vs índice binario asimétrico, a igual tamaño.
(2) Reranker cross-encoder (equivalente open source a Jev/System One "Score") sobre el top-30.
Mes completo 2024-01 (embeddings EmbeddingGemma 2 de embed_month.py), 300 consultas de eval/2024-01.jsonl,
relevantes = comentarios del hilo (sin contar la propia story). Métrica: P@10 y MRR."""
import json, re, html, time, collections, sys, os
import numpy as np, pyarrow.parquet as pq

E = np.load('emb/g2_2024-01.f16.npy').astype(np.float32)
IDS = np.load('emb/g2_2024-01.ids.npy')
qs = [json.loads(l) for l in open('eval/2024-01.jsonl', encoding='utf-8')]
t = pq.read_table('data/2024/2024-01.parquet', columns=['id', 'parent', 'title', 'text', 'dead'])
ids, par, title, text, dead = (t[c].to_pylist() for c in ['id', 'parent', 'title', 'text', 'dead'])
idx = {i: k for k, i in enumerate(ids)}
root = {}
for k in range(len(ids)):
    p, cur = par[k], k
    while p and p in idx:
        cur = idx[p]; p = par[cur]
    root[ids[k]] = ids[cur] if not p else -1
clean = lambda s: re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', s))).strip()
alive = np.array([not dead[idx[i]] for i in IDS])
doc_root = np.array([root[i] for i in IDS])
print(f'{len(IDS):,} docs con vector, {len(qs)} consultas', flush=True)

# consultas con EmbeddingGemma 2 (mismo prompt que el sistema)
cache = 'emb/q_eval_2024-01.npy'
if os.path.exists(cache):
    Q = np.load(cache)
else:
    import torch
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer('google/embeddinggemma-2', device='cpu').float()
    Q = st.encode([{'text': q['q']} for q in qs], prompt_name='query', normalize_embeddings=True).astype(np.float32)
    np.save(cache, Q)

def evaluate(ranked_lists, label):
    p10, mrr = 0.0, 0.0
    for q, lst in zip(qs, ranked_lists):
        lst = [d for d in lst if IDS[d] != q['story']][:10]
        rel = [doc_root[d] == q['story'] for d in lst]
        p10 += sum(rel) / 10
        mrr += next((1 / (i + 1) for i, r in enumerate(rel) if r), 0)
    print(f'  {label:<44} P@10 {p10/len(qs):.3f}  MRR {mrr/len(qs):.3f}', flush=True)

def topk(S, k=30):
    S = np.where(alive[None, :], S, -1e9)
    part = np.argpartition(-S, k, axis=1)[:, :k]
    return [p[np.argsort(-S[i, p])] for i, p in enumerate(part)]

def norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

print('(1) índices vectoriales, solo vector:')
res = {}
for bits in (768, 512, 256):
    Dn, Qn = norm(E[:, :bits]), norm(Q[:, :bits])
    if bits == 768:
        res['float'] = topk(Qn @ Dn.T)
        evaluate(res['float'], 'float32 768d (3072 B, techo)')
    S = Qn @ np.sign(Dn).T
    r = topk(S)
    res[f'bin{bits}'] = r
    evaluate(r, f'binario asimétrico {bits} bits ({bits//8} B) [hnz]')

import turbovec
for dim, bw in ((256, 2), (384, 2), (256, 4), (768, 2)):
    Dn, Qn = norm(E[:, :dim]), norm(Q[:, :dim])
    ix = turbovec.TurboQuantIndex(dim, bw)
    rng = np.random.default_rng(0)
    ix.calibrate(np.ascontiguousarray(Dn[rng.choice(len(Dn), 2048, replace=False)]))
    ix.add(np.ascontiguousarray(Dn))
    t0 = time.time()
    out = ix.search(np.ascontiguousarray(Qn), 60)
    dt = (time.time() - t0) / len(qs) * 1000
    I = out[1] if isinstance(out, tuple) else out
    I = np.asarray(I)
    lists = [[d for d in row if d >= 0 and alive[d]][:30] for row in I]
    evaluate(lists, f'turbovec {bw}-bit {dim}d (~{dim*bw//8} B), {dt:.1f} ms/cons.')

print('(2) reranker cross-encoder sobre el top-30 del vector binario 768:', flush=True)
from fastembed.rerank.cross_encoder import TextCrossEncoder
doc_text = lambda d: (title[idx[IDS[d]]] + ' ' + clean(text[idx[IDS[d]]]))[:1500]
for name in ('Xenova/ms-marco-MiniLM-L-6-v2', 'jinaai/jina-reranker-v1-tiny-en', 'BAAI/bge-reranker-v2-m3-int8'):
    try:
        ce = TextCrossEncoder(name, threads=os.cpu_count())
        lists = []
        t0 = time.time()
        for q, cand in zip(qs, res['bin768']):
            sc = np.array(list(ce.rerank(q['q'], [doc_text(d) for d in cand])))
            lists.append([cand[i] for i in np.argsort(-sc)])
        dt = (time.time() - t0) / len(qs) * 1000
        evaluate(lists, f'{name.split("/")[1]} ({dt:.0f} ms/cons.)')
        # fusión: reranker + vector (rangos recíprocos)
    except Exception as ex:
        print(f'  {name}: FALLO {type(ex).__name__}: {str(ex)[:150]}')
