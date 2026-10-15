"""Paired cold-latency comparison of two hnz binaries (A/B), interleaved so both see the same machine
state (a laptop's clocks and temperature drift between runs of a whole setting).

For each eval query, `hnz search --rerank` runs once with each binary (in alternating order), each
in a new process; prints the medians of search time, decoding, wait after decoding and documents
embedded, and the median of the paired differences.
Usage: python experimentos/bench_ab_latency.py A B [n_queries] [block]
A and B: an hnz.exe, optionally followed by extra search arguments after a '|'
(e.g. "work/bin/hnz.exe|--rerank-kv 60 --rerank-top 40").
"""
import json
import os
import re
import statistics
import subprocess
import sys

import hnz

sys.stdout.reconfigure(line_buffering=True)

def spec(s):
    exe, _, extra = s.partition('|')
    return os.path.abspath(exe), extra.split()


(A, A_ARGS), (B, B_ARGS) = spec(sys.argv[1]), spec(sys.argv[2])
N = int(sys.argv[3]) if len(sys.argv) > 3 else 30
BLOCK = os.path.abspath(sys.argv[4] if len(sys.argv) > 4 else 'work/p3/hn-sample-v3.hnz')
qs = []
for m in ('2024-01', '2010-06', '2015-06'):
    lines = [json.loads(l) for l in open(f'eval/{m}.jsonl', encoding='utf-8-sig') if l.strip()]
    qs += [l['q'] for l in lines[60:60 + N // 3 + 1]]
qs = qs[:N]
env = dict(os.environ, HNZ_OV_LIBS=hnz._ov_libs() or '')


def run(exe, extra, q):
    r = subprocess.run([exe, 'search', '--root', BLOCK, '--rerank', '--json', '-k', '10', *extra, q], env=env,
                       capture_output=True, text=True, encoding='utf-8')
    m1 = re.search(r'(\d+) re-embebidos; etapa 1 (\d+) ms, decodificación (\d+) ms, embebido y fusión (\d+) ms', r.stderr)
    m2 = re.search(r'búsqueda ([\d.]+) ms', r.stderr)
    if not (m1 and m2):
        raise SystemExit(f'failed: {q}\n{r.stderr[-300:]}')
    return {'search': float(m2.group(1)), 'decode': float(m1.group(3)), 'tail': float(m1.group(4)),
            'embedded': int(m1.group(1)), 'out': r.stdout}


res = {'A': [], 'B': []}
for i, q in enumerate(qs):
    order = ('A', 'B') if i % 2 == 0 else ('B', 'A')
    got = {}
    for w in order:
        got[w] = run(*((A, A_ARGS) if w == 'A' else (B, B_ARGS)), q)
    if got['A']['out'] != got['B']['out']:
        print(f'results differ: {q}')
    for w in 'AB':
        res[w].append(got[w])
print(f'{len(qs)} queries, cold, interleaved; A = {os.path.basename(A)} {" ".join(A_ARGS)}, B = {os.path.basename(B)} {" ".join(B_ARGS)}')
for key in ('search', 'decode', 'tail', 'embedded'):
    a = [r[key] for r in res['A']]
    b = [r[key] for r in res['B']]
    d = [y - x for x, y in zip(a, b)]
    print(f'{key:>9}: A median {statistics.median(a):8.0f}  B median {statistics.median(b):8.0f}  '
          f'median of B - A {statistics.median(d):+8.0f}  (B lower in {sum(x < 0 for x in d)}/{len(d)})')
