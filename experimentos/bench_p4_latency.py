"""E33e step 4: cold latency of a reranked search (P4) on the sample block, per setting.

Each query runs in a fresh `hnz search --rerank` process (empty caches; model loading is not
counted: the CLI reports the search time apart), with the embedding streams and LM threads of the
setting. Prints p50 / p90 of the search time and its parts (stage 1, decoding, waiting for the
re-embedding after the last prefix).
Usage: python experimentos/bench_p4_latency.py [n_queries] [hnz.exe] [block]
"""
import json
import os
import re
import statistics
import subprocess
import sys

import hnz

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
EXE = sys.argv[2] if len(sys.argv) > 2 else 'hnz/target_p3/release/hnz.exe'
BLOCK = sys.argv[3] if len(sys.argv) > 3 else 'work/p3/hn-sample-v3.hnz'
SETTINGS = [  # (embedding streams, LM threads)
    (1, 8), (2, 8), (4, 8), (4, 12), (8, 8),
]
qs = []
for m in ('2024-01', '2010-06', '2015-06'):
    lines = [json.loads(l) for l in open(f'eval/{m}.jsonl', encoding='utf-8-sig') if l.strip()]
    qs += [l['q'] for l in lines[:N // 3 + 1]]
qs = qs[:N]
env0 = dict(os.environ, HNZ_OV_LIBS=hnz._ov_libs() or '')


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]


print(f'{N} queries, cold, {BLOCK}')
print(f'{"streams":>7} {"LM thr":>6} {"search p50":>10} {"p90":>7} {"stage1":>7} {"decode":>7} {"embed tail":>10} {"tokens":>7}')
for streams, lm_threads in SETTINGS:
    env = dict(env0, HNZ_EMBED_STREAMS=str(streams), HNZ_LM_THREADS=str(lm_threads))
    tot, st1, dec, tail, tok = [], [], [], [], []
    for q in qs:
        r = subprocess.run([EXE, 'search', '--root', BLOCK, '--rerank', '--json', '-k', '10', q], env=env,
                           capture_output=True, text=True, encoding='utf-8')
        err = r.stderr
        m1 = re.search(r'etapa 1 (\d+) ms, decodificación (\d+) ms, embebido y fusión (\d+) ms', err)
        m2 = re.search(r'búsqueda ([\d.]+) ms', err)
        m3 = re.search(r'\((\d+) tokens\)', err)
        if not (m1 and m2):
            print('failed:', q, err[-300:])
            continue
        st1.append(float(m1.group(1)))
        dec.append(float(m1.group(2)))
        tail.append(float(m1.group(3)))
        tot.append(float(m2.group(1)))
        tok.append(int(m3.group(1)) if m3 else 0)
    print(f'{streams:>7} {lm_threads:>6} {statistics.median(tot):>10.0f} {pct(tot, 90):>7.0f} {statistics.median(st1):>7.0f} '
          f'{statistics.median(dec):>7.0f} {statistics.median(tail):>10.0f} {statistics.median(tok):>7.0f}', flush=True)
