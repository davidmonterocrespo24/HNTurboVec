"""P4 acceptance: latency of get() (an interactive LM decode) while concurrent cold reranked searches
decode their prefixes in the same shared LM service.

Opens the v3 sample block once; `n_rerank` threads run reranked searches with fresh queries (cold
prefixes) in a loop while one thread times get() of records it has not fetched before (cold too).
Prints get()'s p50 / p95 / max alone and under the load, and the reranked searches' p50.
Usage: python experimentos/bench_get_under_rerank.py [n_rerank] [seconds] [block]
"""
import json
import statistics
import sys
import threading
import time

import hnz

sys.stdout.reconfigure(line_buffering=True)
N_RERANK = int(sys.argv[1]) if len(sys.argv) > 1 else 4
SECONDS = float(sys.argv[2]) if len(sys.argv) > 2 else 60
BLOCK = sys.argv[3] if len(sys.argv) > 3 else 'work/p3/hn-sample-v3.hnz'


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]


hn = hnz.open(BLOCK, ram='2GB', preload=True)
qs = []
for m in ('2024-01', '2010-06', '2015-06'):
    qs += [json.loads(l)['q'] for l in open(f'eval/dev-{m}.jsonl', encoding='utf-8') if l.strip()]
# records to get: comments of other days (not among the searches' results, not cached)
ids = [h['id'] for d in ('2015-06-20', '2010-06-20', '2024-01-20')
       for h in hn.search('', k=200, since=d, until=d, types='comment', raw=True)['results']]


def time_gets(ids):
    out = []
    for i in ids:
        t = time.perf_counter()
        hn.get(i)
        out.append((time.perf_counter() - t) * 1e3)
    return out


alone = time_gets(ids[:100])
print(f'get() alone: p50 {pct(alone, 50):.0f} ms, p95 {pct(alone, 95):.0f} ms, max {max(alone):.0f} ms ({len(alone)} records)')

stop = threading.Event()
search_ms = []
lock = threading.Lock()


def reranker(k):
    j = k
    while not stop.is_set():
        t = time.perf_counter()
        hn.search(qs[j % len(qs)], k=10)
        with lock:
            search_ms.append((time.perf_counter() - t) * 1e3)
        j += N_RERANK


ts = [threading.Thread(target=reranker, args=(k,)) for k in range(N_RERANK)]
for t in ts:
    t.start()
time.sleep(3)   # the searches are decoding by now
loaded = []
t_end = time.perf_counter() + SECONDS
rest = iter(ids[100:])
while time.perf_counter() < t_end:
    i = next(rest, None)
    if i is None:
        break
    loaded += time_gets([i])
stop.set()
for t in ts:
    t.join()
print(f'get() with {N_RERANK} cold reranks running: p50 {pct(loaded, 50):.0f} ms, p95 {pct(loaded, 95):.0f} ms, '
      f'max {max(loaded):.0f} ms ({len(loaded)} records)')
print(f'reranked searches meanwhile: {len(search_ms)}, p50 {statistics.median(search_ms):.0f} ms, '
      f'p95 {pct(search_ms, 95):.0f} ms')
hn.close()
