"""E33a: the dynamic-shape int8 IR (scripts/export_gemma2_dyn.py) on CPU, as the reranker uses it.

1. Queries: the 900 eval queries through the static b1_s64 int8 IR and the dynamic int8 IR, both at
   [1, 64] (as `Embedder::embed_query`): cosine and latency.
2. Documents: a sample of v2 document prompts (the token ids `hnz prompts` wrote) through the
   dynamic int8 IR at their canonical shape [1, tokens rounded up to 16] (as `Embedder::embed_docs`),
   against the stored v2 vectors (static fp16 IRs on the iGPU, `scripts/embed_prompts.py`): cosine,
   and sign flips over the first 256 dimensions (the bits stage 1 compares).
3. Repeatability: the same documents again, on 4 streams in any order: bit-identical?

Usage: python experimentos/bench_doc_embed_dyn.py [n_docs_per_month]
"""
import json
import sys
import time

import numpy as np
import openvino as ov
from tokenizers import Tokenizer

N_DOCS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
MONTHS = ('2024-01', '2010-06', '2015-06')
QUERY_PROMPT = 'task: search result | query: '


def pct(v, p):
    return float(np.percentile(np.asarray(v), p))


def main():
    core = ov.Core()
    core.set_property('CPU', {'INFERENCE_PRECISION_HINT': 'f32'})
    dyn = core.compile_model('models/gemma2_text_dyn_int8.xml', 'CPU')
    stat = core.compile_model('models/gemma2_text_b1_s64_int8.xml', 'CPU')
    tok = Tokenizer.from_file('models/gemma2_tokenizer.json')

    # 1. queries
    qs = []
    for m in MONTHS:
        qs += [json.loads(l)['q'] for l in open(f'eval/{m}.jsonl', encoding='utf-8-sig') if l.strip()]
    cos, t_dyn, t_stat = [], [], []
    rd, rs = dyn.create_infer_request(), stat.create_infer_request()
    for q in qs:
        x = tok.encode(QUERY_PROMPT + q).ids[:64]
        a = np.zeros((1, 64), np.int64)
        mk = np.zeros((1, 64), np.int64)
        a[0, :len(x)] = x
        mk[0, :len(x)] = 1
        t0 = time.perf_counter()
        e1 = rd.infer([a, mk])[0][0].copy()
        t1 = time.perf_counter()
        e2 = rs.infer([a, mk])[0][0].copy()
        t2 = time.perf_counter()
        t_dyn.append((t1 - t0) * 1e3)
        t_stat.append((t2 - t1) * 1e3)
        cos.append(float(e1 @ e2 / np.linalg.norm(e1) / np.linalg.norm(e2)))
    print(f'queries: {len(qs)}; cosine dynamic vs static: mean {np.mean(cos):.5f}, min {np.min(cos):.5f}; '
          f'latency p50 dynamic {pct(t_dyn, 50):.1f} ms, static {pct(t_stat, 50):.1f} ms', flush=True)

    # 2. documents
    rng = np.random.default_rng(7)
    docs = []          # (month, token ids, stored vector)
    for m in MONTHS:
        ids = np.load(f'work/p3/prompts/{m}.ids.npy')
        lens = np.load(f'work/p3/prompts/{m}.len.npy').astype(np.int64)
        toks = np.load(f'work/p3/prompts/{m}.tok.npy', mmap_mode='r')
        off = np.concatenate([[0], np.cumsum(lens)])
        sids = np.load(f'emb/g2v2_{m}.ids.npy')
        svec = np.load(f'emb/g2v2_{m}.f16.npy', mmap_mode='r')
        row = {int(d): r for r, d in enumerate(sids)}
        for r in np.sort(rng.choice(len(ids), N_DOCS, replace=False)):
            if lens[r] == 0 or int(ids[r]) not in row:
                continue
            docs.append((m, np.asarray(toks[off[r]:off[r + 1]], np.int64), np.asarray(svec[row[int(ids[r])]], np.float32)))

    def run(req, x):
        n = (len(x) + 15) // 16 * 16
        a = np.zeros((1, n), np.int64)
        mk = np.zeros((1, n), np.int64)
        a[0, :len(x)] = x
        mk[0, :len(x)] = 1
        v = req.infer([a, mk])[0][0].copy()
        return v / np.linalg.norm(v)

    t0 = time.perf_counter()
    vecs = [run(rd, x) for _, x, _ in docs]
    dt = time.perf_counter() - t0
    by = {m: ([], []) for m in MONTHS}
    for (m, x, s), v in zip(docs, vecs):
        s = s / np.linalg.norm(s)
        by[m][0].append(float(v @ s))
        by[m][1].append(int(np.sum((v[:256] > 0) != (s[:256] > 0))))
    allc = [c for m in MONTHS for c in by[m][0]]
    allf = [f for m in MONTHS for f in by[m][1]]
    print(f'documents: {len(docs)} ({dt / len(docs) * 1e3:.0f} ms each, mean {np.mean([len(d[1]) for d in docs]):.0f} tokens); '
          f'cosine vs stored v2: mean {np.mean(allc):.5f}, p1 {pct(allc, 1):.5f}, min {np.min(allc):.5f}; '
          f'sign flips of 256: mean {np.mean(allf):.2f}, p99 {pct(allf, 99):.0f}, max {np.max(allf)}', flush=True)
    for m in MONTHS:
        print(f'  {m}: cosine mean {np.mean(by[m][0]):.5f}, min {np.min(by[m][0]):.5f}; flips mean {np.mean(by[m][1]):.2f}, max {np.max(by[m][1])}')

    # 3. repeatability on 4 streams, in reverse order
    dyn4 = core.compile_model('models/gemma2_text_dyn_int8.xml', 'CPU', {'NUM_STREAMS': '4'})
    reqs = [dyn4.create_infer_request() for _ in range(4)]
    again = [None] * len(docs)
    order = list(range(len(docs)))[::-1]
    for i in range(0, len(order), 4):
        batch = order[i:i + 4]
        for req, j in zip(reqs, batch):
            x = docs[j][1]
            n = (len(x) + 15) // 16 * 16
            a = np.zeros((1, n), np.int64)
            mk = np.zeros((1, n), np.int64)
            a[0, :len(x)] = x
            mk[0, :len(x)] = 1
            req.start_async([a, mk])
        for req, j in zip(reqs, batch):
            req.wait()
            v = req.get_output_tensor(0).data[0].copy()
            again[j] = v / np.linalg.norm(v)
    same = sum(np.array_equal(a, b) for a, b in zip(vecs, again))
    print(f'repeat on 4 streams, reverse order: {same}/{len(docs)} bit-identical; '
          f'max |diff| {max(float(np.max(np.abs(a - b))) for a, b in zip(vecs, again)):.2e}')


if __name__ == '__main__':
    main()
