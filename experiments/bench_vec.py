"""Vector index benchmark: bytes/vector vs recall@10 (against exact float32 search with the same
model). Variants: MRL (truncated dims), int8, 4-bit, binary, binary+rescore, PQ, RaBitQ."""
import sys, os, time, re, html
import numpy as np, pyarrow.parquet as pq, faiss
from fastembed import TextEmbedding

MODEL, NDOC = sys.argv[1], int(sys.argv[2])
NQ, K = 300, 10
os.makedirs('emb', exist_ok=True)
tag = MODEL.replace('/', '_') + f'_{NDOC}'

t = pq.read_table('data/2024/2024-01.parquet', columns=['type', 'title', 'text', 'dead'])
typ, title, text, dead = (t[c].to_pylist() for c in ['type', 'title', 'text', 'dead'])
clean = lambda s: html.unescape(re.sub(r'<[^>]+>', ' ', s)).strip()
rng = np.random.default_rng(42)
cand = [i for i in range(len(typ)) if not dead[i] and (text[i] or title[i])]
rng.shuffle(cand)
stories = [i for i in cand if typ[i] == 1 and title[i]]
qids = stories[:NQ]
qset = set(qids)
dids = [i for i in cand if i not in qset][:NDOC]
docs = [clean((title[i] + '\n' + text[i]).strip()) for i in dids]
queries = [title[i] for i in qids]

if os.path.exists(f'emb/{tag}_D.npy'):
    D = np.load(f'emb/{tag}_D.npy'); Q = np.load(f'emb/{tag}_Q.npy')
elif MODEL.startswith('st:'):  # sentence-transformers (e.g. google/embeddinggemma-2)
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(os.cpu_count())
    m = SentenceTransformer(MODEL[3:], device='cpu')
    m.max_seq_length = 256
    order = np.argsort([len(d) for d in docs])
    t0 = time.time()
    E = m.encode([{'text': docs[i]} for i in order], prompt_name='document', batch_size=32, convert_to_numpy=True).astype(np.float32)
    D = np.empty_like(E); D[order] = E
    print(f'embeddings: {NDOC/(time.time()-t0):.1f} docs/s', flush=True)
    Q = m.encode([{'text': q} for q in queries], prompt_name='query', convert_to_numpy=True).astype(np.float32)
    np.save(f'emb/{tag}_D.npy', D); np.save(f'emb/{tag}_Q.npy', Q)
else:
    m = TextEmbedding(MODEL, max_length=256, threads=os.cpu_count())
    order = np.argsort([len(d) for d in docs])  # grouping by length = less padding
    t0 = time.time()
    E = np.array(list(m.embed([docs[i] for i in order], batch_size=32)), dtype=np.float32)
    D = np.empty_like(E); D[order] = E
    print(f'embeddings: {NDOC/(time.time()-t0):.1f} docs/s', flush=True)
    Q = np.array(list(m.query_embed(queries)), dtype=np.float32)
    np.save(f'emb/{tag}_D.npy', D); np.save(f'emb/{tag}_Q.npy', Q)

def norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def topk(scores, k):
    return np.argsort(-scores, axis=1)[:, :k]

full = D.shape[1]
gt = topk(norm(Q) @ norm(D).T, K)  # ground truth: full float32

def recall(found):
    return np.mean([len(set(a) & set(b)) / K for a, b in zip(found, gt)])

dims = [d for d in (1024, 768, 512, 384, 256, 128, 64) if d <= full]
print(f'\n{MODEL}  docs={NDOC:,}  queries={NQ}  dim={full}')
print(f'{"variant":<34}{"B/vector":>9}{"recall@10":>11}{"49.3M":>10}')
for d in dims:
    Dd, Qd = norm(D[:, :d]), norm(Q[:, :d])
    rows = []
    rows.append(('float32', 4 * d, topk(Qd @ Dd.T, K)))
    # scalar int8 / 4 bit (faiss SQ)
    for qt, name, b in [(faiss.ScalarQuantizer.QT_8bit, 'int8', d), (faiss.ScalarQuantizer.QT_4bit, '4-bit', d // 2)]:
        ix = faiss.IndexScalarQuantizer(d, qt, faiss.METRIC_INNER_PRODUCT); ix.train(Dd); ix.add(Dd)
        rows.append((name, b, ix.search(Qd, K)[1]))
    # binary (sign) with Hamming
    Db, Qb = np.packbits(Dd > 0, axis=1), np.packbits(Qd > 0, axis=1)
    ham = np.unpackbits(Qb[:, None, :] ^ Db[None, :, :], axis=2).sum(2)
    rows.append(('binary 1-bit', d // 8, np.argsort(ham, axis=1, kind='stable')[:, :K]))
    # binary + exact rescore of the top-100 (rescore = re-embedding the decompressed text)
    for R in (100, 300):
        c = np.argsort(ham, axis=1, kind='stable')[:, :R]
        s = np.einsum('qd,qrd->qr', Q, D[c])  # re-embedding = full vector
        rows.append((f'binary + rescore top-{R}', d // 8, np.take_along_axis(c, np.argsort(-s, 1)[:, :K], 1)))
    # PQ
    for msub in (8, 16, 32):
        if d % msub == 0 and d // msub >= 2:
            ix = faiss.IndexPQ(d, msub, 8, faiss.METRIC_INNER_PRODUCT); ix.train(Dd); ix.add(Dd)
            rows.append((f'PQ {msub}x8', msub, ix.search(Qd, K)[1]))
    # RaBitQ (1 bit/dim + 2 floats)
    if hasattr(faiss, 'IndexRaBitQ'):
        ix = faiss.IndexRaBitQ(d, faiss.METRIC_INNER_PRODUCT); ix.train(Dd); ix.add(Dd)
        rows.append(('RaBitQ 1-bit', d // 8 + 8, ix.search(Qd, K)[1]))
    for name, b, f in rows:
        print(f'{d:>4}d {name:<29}{b:>9}{recall(f):>11.3f}{b*49.3e6/1e9:>8.2f}GB', flush=True)
