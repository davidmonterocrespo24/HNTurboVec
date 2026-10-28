"""E30f: reranking WITHOUT re-embedding every candidate (for a CPU-only server).

Same corpus, candidates and BM25-over-prefix as E30c (experiments/bench_fusion.py): top-100 by the
256-bit codes (asymmetric score s1) + top-20 from the rare-term index. Re-embedding ~115 prefixes on
CPU costs ~3-4 s (E30b: 28-38 docs/s), so this measures cheaper rerankings:
  R0  no re-embedding: s1 + w * b / (b + c)          (grid over w, c)
  R1  E30c reference: cos(prefix embedding, query) + 0.15 * b / (b + 40), all candidates re-embedded
  CK  cascade: rank by the best R0, re-embed only its top K, rerank those K with R1 (K = 10..50)
P@10 / MRR per month (900 queries). Usage: python experiments/bench_rerank_cpu.py
"""
import collections
import html
import json
import math
import re

import numpy as np
import pyarrow.parquet as pq

import hnz

MONTHS = {'2024-01': 2024, '2010-06': 2010, '2015-06': 2015}
P, KV, KB, BITS = 128, 100, 20, 256
K1, B = 1.2, 0.75
WS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
CS = [5.0, 10.0, 20.0, 40.0]
KS = [10, 20, 30, 50]
from transformers import AutoTokenizer
lmtok = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')
clean = lambda s: re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', s))).strip()


def plain(body):
    if '<a href' in body or '<p>' in body:
        return clean(body)
    body = re.sub(r'</?i>|<pre><code>|</code></pre>', '', body)
    body = re.sub(r'<([^>|]+)\|([^>]*)>', r'\2 (\1)', body)
    body = re.sub(r'<([^>]+)>\d+>', r'\1', body)
    body = re.sub(r'<([^>\s]+)>', r'\1', body)
    return clean(html.escape(body, quote=False))


def norm(n):
    q = min(255, round(16 * math.log2(1 + n)))
    return max(0.0, 2 ** (q / 16) - 1)


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
            cur = idx[p]
            p = par[cur]
        ids.append(int(i))
        root.append(pid[cur] if not p else -1)
        alive.append(not dead[k])
        full_txt.append(hnz.index_text(title[k] or '', url[k] or '', text[k] or ''))
        lm_s.append(lms[int(i)])
    E.append(e)
ids, root, alive = np.array(ids), np.array(root), np.array(alive)
E = np.concatenate(E).astype(np.float32)
E /= np.linalg.norm(E, axis=1, keepdims=True)
N = len(ids)
print(f'{N:,} documents', flush=True)

df = collections.Counter()
dl = np.zeros(N, np.float32)
lens = np.zeros(N, np.float32)
for d, t in enumerate(full_txt):
    ts = hnz.terms(t)
    dl[d] = norm(len(ts))
    lens[d] = len(ts)
    df.update(set(ts))
avgdl = float(lens.mean())
idf = lambda t: math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1.0)
RARE = math.ceil(0.0002 * N)
post = collections.defaultdict(list)
for d, t in enumerate(full_txt):
    for term in set(hnz.terms(t)):
        if df[term] <= RARE:
            post[term].append(d)
print(f'{len(df):,} terms; rare (df <= {RARE}): {len(post):,}', flush=True)

_pref = {}


def prefix_tf(d):
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


S = np.where(E[:, :BITS] > 0, 1.0, -1.0).astype(np.float32) / np.sqrt(BITS)
r0 = {(w, c): {m: [] for m in MONTHS} for w in WS for c in CS}
r1 = {m: [] for m in MONTHS}
rk = {k: {m: [] for m in MONTHS} for k in KS}
best_r0 = (0.1, 20.0)          # replaced below by the grid's best; the cascade uses it


def pr(rel_sorted):
    r = rel_sorted[:10]
    return r.mean(), next((1 / (i + 1) for i, x in enumerate(r) if x), 0.0)


queries = {}
for m in MONTHS:
    qs = [json.loads(l) for l in open(f'eval/{m}.jsonl', encoding='utf-8-sig') if l.strip()]
    Q = np.load(f'emb/q_eval_{m}.npy').astype(np.float32)
    queries[m] = (qs, Q / np.linalg.norm(Q, axis=1, keepdims=True))

# per query: candidates, s1, cos, b, rel (kept to run the cascade with the best R0 afterwards)
cached = []
for m, (qs, Q) in queries.items():
    q256 = Q[:, :BITS] / np.linalg.norm(Q[:, :BITS], axis=1, keepdims=True)
    for qi, q in enumerate(qs):
        ok = alive & (ids != q['story'])
        s1all = np.where(ok, S @ q256[qi], -9.0)
        vc = np.argpartition(-s1all, KV)[:KV]
        qterms = list(dict.fromkeys(hnz.terms(q['q'])))
        kb = collections.Counter()
        for t in qterms:
            for d in post.get(t, ()):
                if ok[d]:
                    kb[d] += idf(t) * (K1 + 1) / (1 + K1 * (1 - B + B * dl[d] / avgdl))
        cands = np.array(list(dict.fromkeys(list(vc) + [d for d, _ in kb.most_common(KB)])))
        s1 = s1all[cands]
        cos = E[cands] @ Q[qi]
        b = np.array([bm25(qterms, prefix_tf(d), d) for d in cands])
        rel = root[cands] == q['story']
        cached.append((m, s1, cos, b, rel))
        for w in WS:
            for c in CS:
                r0[w, c][m].append(pr(rel[np.argsort(-(s1 + w * b / (b + c)))]))
        r1[m].append(pr(rel[np.argsort(-(cos + 0.15 * b / (b + 40.0)))]))
    print(f'{m} done', flush=True)

means = {k: np.mean([np.array(v[m])[:, 0].mean() for m in MONTHS]) for k, v in r0.items()}
best_r0 = max(means, key=means.get)
for m, s1, cos, b, rel in cached:
    w, c = best_r0
    order = np.argsort(-(s1 + w * b / (b + c)))
    for k in KS:
        top = order[:k]
        final = top[np.argsort(-(cos[top] + 0.15 * b[top] / (b[top] + 40.0)))]
        rest = order[k:]
        rk[k][m].append(pr(rel[np.concatenate([final, rest])]))


def row(name, res):
    a = [np.array(res[m]) for m in MONTHS]
    print(f'{name:<34} ' + '  '.join(f'{x[:, 0].mean():.3f}/{x[:, 1].mean():.3f}' for x in a) + f'  {np.mean([x[:, 0].mean() for x in a]):.3f}', flush=True)


print(f'\n{"P@10/MRR":<34} ' + '  '.join(f'{m:>11}' for m in MONTHS) + '   mean')
row('R1 E30c (all re-embedded)', r1)
for (w, c) in sorted(r0, key=lambda k: -means[k])[:6]:
    row(f'R0 no re-embedding w={w} c={c:.0f}', r0[w, c])
row('R0 w=0 (stage-1 order only)', r0[0.0, 5.0])
for k in KS:
    row(f'cascade: re-embed top {k} of R0', rk[k])
