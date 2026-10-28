"""Real latencies of the hnz server (models loaded once, as in daily use).

Each case is measured with a freshly started server (empty L1 cache), no L2 cache on disk and N
DIFFERENT queries (titles from eval/*.jsonl). It reports the median and the p90 (a single query with
a warm cache is not representative, E26). Then the cache scenarios, on purpose: the same full text
the 1st time, repeated (L1) and after restarting the server (L2 on disk).
Usage: python experiments/bench_serve.py [--n 30]   (with out/ built and hnz compiled in hnz/target/release)
"""
import argparse, json, os, subprocess, time, urllib.parse, urllib.request
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # project root
HNZ = os.path.join(ROOT, 'hnz', 'target', 'release', 'hnz.exe' if os.name == 'nt' else 'hnz')
PORT = 8099
L2 = os.path.join(ROOT, 'out', '_cache_l2.bin')


def start(clear_l2=True):
    if clear_l2 and os.path.exists(L2):
        os.remove(L2)
    p = subprocess.Popen([HNZ, 'serve', '--root', 'out', '--port', str(PORT)], cwd=ROOT,
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    t0 = time.time()
    while True:
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{PORT}/', timeout=1).read()
            return p, time.time() - t0
        except Exception:
            if p.poll() is not None:
                raise SystemExit('the server exited: ' + p.stderr.read().decode(errors='replace'))
            time.sleep(0.2)


def stop(p):
    p.terminate()
    p.wait()


def q(params):
    url = f'http://127.0.0.1:{PORT}/search?' + urllib.parse.urlencode(params)
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(url, timeout=600).read())
    wall = (time.time() - t0) * 1000
    if 'error' in r:
        raise SystemExit(r['error'])
    return r['ms'], wall, r['results']


def item(i):
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(f'http://127.0.0.1:{PORT}/item/{i}', timeout=600).read())
    return r.get('ms'), (time.time() - t0) * 1000


def titles(m):
    return [json.loads(l)['q'] for l in open(os.path.join(ROOT, 'eval', f'{m}.jsonl'), encoding='utf-8-sig') if l.strip()]


def row(label, ms, wall, lm_frac):
    tot = [x['total'] for x in ms]
    med = lambda k: np.median([x[k] for x in ms])
    print(f'{label:<50} total median {np.median(tot):7.1f} ms, p90 {np.percentile(tot, 90):7.1f} '
          f'(embedding {med("embedding"):5.1f}, search {med("search"):5.1f}, text {med("fetch"):7.1f}) '
          f'| wall {np.median(wall):7.1f} ms | {lm_frac:.0%} from the LM tier', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=30)
    a = ap.parse_args()
    T = {m: titles(m) for m in ('2024-01', '2010-06', '2015-06')}
    words = ['rust', 'python', 'remote', 'engineer', 'startup', 'data', 'security', 'golang', 'design', 'growth']
    cases = [
        ('hot (2024), hybrid, k=10', [dict(q=t, k=10, **{'from': '2024', 'to': '2024'}) for t in T['2024-01']]),
        ('hot (2024), keywords, k=10', [dict(q=t, k=10, mode='keyword', **{'from': '2024', 'to': '2024'}) for t in T['2024-01']]),
        ('whole archive, hybrid, k=10 (2024 queries)', [dict(q=t, k=10) for t in T['2024-01']]),
        ('whole archive, hybrid, k=10 (2010 queries)', [dict(q=t, k=10) for t in T['2010-06']]),
        ('cold (2010), hybrid, k=10, snippets', [dict(q=t, k=10, **{'from': '2010', 'to': '2010'}) for t in T['2010-06']]),
        ('cold (2015), hybrid, k=10, snippets', [dict(q=t, k=10, **{'from': '2015', 'to': '2015'}) for t in T['2015-06']]),
        ('cold (2010), hybrid, k=10, full text', [dict(q=t, k=10, full=1, **{'from': '2010', 'to': '2010'}) for t in T['2010-06']]),
        ('whole archive, hybrid, k=100, snippets', [dict(q=t, k=100) for t in T['2015-06']]),
        ('type filter job + keywords (exact scan)', [dict(q=w, k=10, type='job', mode='keyword') for w in words]),
        ('author filter, hybrid', [dict(q=t, k=10, by=b) for t, b in zip(T['2015-06'], ['pg', 'tptacek', 'patio11', 'jacquesm', 'dang'] * 10)]),
        ('no text: most recent with filters', [dict(q='', k=10, type=ty, **{'from': y}) for ty in ('story', 'comment', 'job') for y in ('2010', '2015', '2024')]),
    ]
    p, t_start = start()
    print(f'server ready in {t_start:.1f} s (loading segments + models); {a.n} different queries per case')
    stop(p)
    for label, params in cases:
        p, _ = start()
        try:
            q({'q': 'warmup', 'k': 3, 'mode': 'keyword'})
            ms, wall, lm, n = [], [], 0, 0
            for prm in params[:a.n]:
                m, w, res = q(prm)
                ms.append(m); wall.append(w)
                lm += sum(r['tier'] == 'lm' for r in res); n += len(res)
            row(label, ms, wall, lm / max(n, 1))
        finally:
            stop(p)
    # cache scenarios, on purpose: the same queries with full text
    prm = [dict(q=t, k=10, full=1, **{'from': '2010', 'to': '2010'}) for t in T['2010-06'][100:100 + min(a.n, 20)]]
    p, _ = start()
    try:
        first = [q(x) for x in prm]
        again = [q(x) for x in prm]
    finally:
        stop(p)
    p, t_restart = start(clear_l2=False)
    try:
        after = [q(x) for x in prm]
    finally:
        stop(p)
    row('cold (2010), full text, 1st time', [x[0] for x in first], [x[1] for x in first], 1)
    row('  the same query repeated (L1 cache)', [x[0] for x in again], [x[1] for x in again], 1)
    row(f'  after restarting in {t_restart:.1f} s (L2 cache on disk)', [x[0] for x in after], [x[1] for x in after], 1)
    # /item/<id> of uncached cold records (full, on demand)
    p, _ = start()
    try:
        ids = []
        for t in T['2015-06'][200:200 + 10]:
            _, _, res = q(dict(q=t, k=5, **{'from': '2015', 'to': '2015'}))
            ids += [r['id'] for r in res if r['tier'] == 'lm' and r['partial']][:1]
    finally:
        stop(p)
    p, _ = start()   # empty cache: those records had only been read as snippets
    try:
        w = [item(i)[1] for i in ids]
        print(f'{"/item/<id> cold (full, no cache)":<50} median {np.median(w):7.1f} ms, p90 {np.percentile(w, 90):7.1f} ms ({len(w)} items)')
    finally:
        stop(p)


if __name__ == '__main__':
    main()
