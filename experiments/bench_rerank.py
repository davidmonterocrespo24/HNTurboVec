"""Cross-encoder reranker (open source equivalent of Jev/System One "Score") over hnz's top-30.

Uses the hnz server (real hybrid search over all of out/, as in `hnz eval`) and reranks its first 30
results with small fastembed cross-encoders (ONNX, CPU). Queries: eval/*.jsonl
(title of a story → relevant = comments in its thread; the story itself does not count).
Measures P@10 and MRR, the gain over hnz with a 95 % CI (paired bootstrap over queries) and the
end-to-end latency against the normal query (k=10, snippets), each with a freshly started server
and an empty L2 cache.
  --full 1: the reranker reads the full text of the candidates (LM-tier ones are fully decompressed)
  --full 0: it reads what hnz returns by default (title + 64-token snippet in the LM tier)
Usage: python experiments/bench_rerank.py [--eval eval/2015-06.jsonl] [--full 0] [--models a,b] [--n 300] [--cand 30]
"""
import argparse, json, os, subprocess, sys, time, urllib.parse, urllib.request
import numpy as np
# huggingface_hub's hf_xet downloader dies mid-download on this Windows machine (exit 127, E26).
# Use the plain HTTP download instead (it can resume too). Subprocesses inherit this.
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # project root
HNZ = os.path.join(ROOT, 'hnz', 'target', 'release', 'hnz.exe' if os.name == 'nt' else 'hnz')
PORT = 8097
# small ones (80-150 MB) and a large one as a reference (BGE-M3 int8, 570 MB)
MODELS = ['Xenova/ms-marco-MiniLM-L-6-v2', 'Xenova/ms-marco-MiniLM-L-12-v2', 'jinaai/jina-reranker-v1-tiny-en',
          'jinaai/jina-reranker-v1-turbo-en', 'BAAI/bge-reranker-v2-m3-int8']


def start():
    """Freshly started server (empty L1 cache) and no L2 cache on disk."""
    l2 = os.path.join(ROOT, 'out', '_cache_l2.bin')
    if os.path.exists(l2):
        os.remove(l2)
    p = subprocess.Popen([HNZ, 'serve', '--root', 'out', '--port', str(PORT)], cwd=ROOT,
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    while True:
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{PORT}/', timeout=1).read()
            return p
        except Exception:
            if p.poll() is not None:
                raise SystemExit('the server exited: ' + p.stderr.read().decode(errors='replace'))
            time.sleep(0.2)


def stop(p):
    p.terminate()
    p.wait()


def search(q, k, full):
    url = f'http://127.0.0.1:{PORT}/search?' + urllib.parse.urlencode({'q': q, 'k': k, 'full': int(full)})
    r = json.loads(urllib.request.urlopen(url, timeout=600).read())
    if 'error' in r:
        raise SystemExit(r['error'])
    return r['results'], r['ms']['total']


def doc_text(r):
    # only the document itself (its title if it has one + plain text), the same as what the
    # embeddings and BM25 see. NOTE: adding the thread title to comments leaks the label (the
    # query IS the story title) and gives a false P@10 of 0.44 (E26)
    return ((r['title'] + '\n' if r['title'] else '') + r['text'])[:1500]


def per_query(qs, lists):
    """P@10 and RR@10 per query (relevant = its thread is the query's story)."""
    p10, rr = np.zeros(len(qs)), np.zeros(len(qs))
    for i, (q, lst) in enumerate(zip(qs, lists)):
        rel = [r.get('story_id') == q['story'] for r in lst[:10]]
        p10[i] = sum(rel) / 10
        rr[i] = next((1 / (j + 1) for j, x in enumerate(rel) if x), 0)
    return p10, rr


def boot_ci(d, B=4000, seed=0):
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (B, len(d)))].mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--cand', type=int, default=30)
    ap.add_argument('--full', type=int, default=1, choices=(0, 1))
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--eval', default=os.path.join('eval', '2024-01.jsonl'))
    a = ap.parse_args()
    qs = [json.loads(l) for l in open(os.path.join(ROOT, a.eval), encoding='utf-8-sig') if l.strip()][:a.n]
    # 1) normal query (k=10, snippets), with an empty cache
    srv = start()
    try:
        base_ms = [search(q['q'], 10, False)[1] for q in qs]
    finally:
        stop(srv)
    # 2) candidates for the reranker, with an empty cache again
    srv = start()
    try:
        cands, cand_ms = [], []
        for q in qs:
            res, ms = search(q['q'], a.cand + 1, a.full)
            cands.append([r for r in res if r['id'] != q['story']][:a.cand])
            cand_ms.append(ms)
    finally:
        stop(srv)
    p0, r0 = per_query(qs, cands)
    print(f'{len(qs)} queries from {a.eval}; top-{a.cand} from hnz (hybrid, all of out/); the reranker reads '
          f'{"the full text" if a.full else "title + snippet (what hnz returns)"}')
    print(f'  normal hnz query (k=10, snippets): median {np.median(base_ms):.0f} ms, p90 {np.percentile(base_ms, 90):.0f} ms')
    print(f'  top-{a.cand} candidates ({"full" if a.full else "snippets"}): median {np.median(cand_ms):.0f} ms, p90 {np.percentile(cand_ms, 90):.0f} ms')
    print(f'  {"hnz hybrid (no reranker)":<32} P@10 {p0.mean():.3f}  MRR {r0.mean():.3f}', flush=True)
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    threads = max(1, (os.cpu_count() or 2) // 2)
    for name in a.models.split(','):
        try:
            ce = TextCrossEncoder(name, threads=threads)
            list(ce.rerank('warm up', ['a', 'b']))
            lists, fused, t = [], [], []
            for q, c in zip(qs, cands):
                t0 = time.perf_counter()
                sc = np.array(list(ce.rerank(q['q'], [doc_text(r) for r in c])))
                t.append((time.perf_counter() - t0) * 1000)
                order = np.argsort(-sc)
                lists.append([c[i] for i in order])
                # reciprocal rank fusion (reranker + hnz order)
                rk = np.empty(len(c)); rk[order] = np.arange(len(c))
                f = 1 / (60 + rk) + 1 / (60 + np.arange(len(c)))
                fused.append([c[i] for i in np.argsort(-f)])
            p1, r1 = per_query(qs, lists)
            p2, r2 = per_query(qs, fused)
            (lo1, hi1), (lo2, hi2) = boot_ci(p1 - p0), boot_ci(p2 - p0)
            tot = np.array(cand_ms) + np.array(t)
            print(f'  {name.split("/")[1]:<32} P@10 {p1.mean():.3f} ({p1.mean() - p0.mean():+.3f} [{lo1:+.3f},{hi1:+.3f}])  MRR {r1.mean():.3f}'
                  f' | RRF P@10 {p2.mean():.3f} ({p2.mean() - p0.mean():+.3f} [{lo2:+.3f},{hi2:+.3f}])  MRR {r2.mean():.3f}'
                  f' | reranker {np.median(t):.0f} ms, total median {np.median(tot):.0f} ms ({threads} threads)', flush=True)
            del ce
        except Exception as ex:
            print(f'  {name}: FAILED {type(ex).__name__}: {str(ex)[:150]}', flush=True)


if __name__ == '__main__':
    main()
