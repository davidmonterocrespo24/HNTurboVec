"""Acceptance check of the installed `hnz` package on the block of all of HN.

It uses only the public API (what a user of the wheel has): open with a RAM budget, search
(hybrid, reranked, keyword, filters, no query, deadline), get, embed, async, memory, the HTTP
server and the `hnz` command. Each check prints PASS or FAIL; the exit code is the number of
failures. Takes ~2-3 min on a 16-thread laptop.

    python scripts/accept_full.py [--root work/p6/hn-2026-08.hnz] [--ram 2GB] [--verify full]
"""
import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request

import hnz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILS = []


def check(name, cond, detail=''):
    print(f'{"PASS" if cond else "FAIL"}  {name}' + (f'  ({detail})' if detail else ''), flush=True)
    if not cond:
        FAILS.append(name)


def timed(fn, *a, **kw):
    t = time.perf_counter()
    out = fn(*a, **kw)
    return out, (time.perf_counter() - t) * 1e3


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def http(port, path, timeout=300):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}', timeout=timeout) as r:
        return r.status, json.loads(r.read())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--root', default=os.path.join(ROOT, 'work', 'p6', 'hn-2026-08.hnz'))
    ap.add_argument('--ram', default='2GB')
    ap.add_argument('--verify', default='quick', choices=['header', 'quick', 'full'],
                    help='block check before opening (full hashes all 5 GB, a few seconds)')
    a = ap.parse_args()
    print(f'hnz {hnz.__version__} from {os.path.dirname(hnz.__file__)}')

    _, ms = timed(hnz.verify_block, a.root, a.verify)
    check(f'verify_block({a.verify!r})', True, f'{ms:.0f} ms')

    hn, ms = timed(hnz.open, a.root, ram=a.ram, preload=True)
    with hn:
        st = hn.stats()
        check('open with the embedder loaded', st['docs'] == 49_324_111, f'{st["docs"]:,} docs, {ms / 1e3:.1f} s')

        q = 'how do I deal with burnout as a software engineer'
        r, ms = timed(hn.search, q, k=10)
        check('hybrid reranked search: 10 results', len(list(r)) == 10 and r.rerank is not None, f'{ms:.0f} ms cold')
        ids = [h.id for h in r]
        r2, ms = timed(hn.search, q, k=10)
        check('the same search again: same results', [h.id for h in r2] == ids, f'{ms:.0f} ms')
        r, ms = timed(hn.search, q, k=10, rerank=False)
        check('first stage only (rerank=False)', len(list(r)) == 10, f'{ms:.0f} ms')

        r, ms = timed(hn.search, 'sqlite', k=10, mode='keyword', since='2020', types='story')
        hits = list(r)
        check('keyword + since + types filters hold', hits and all(h.kind == 'story' and h.date >= '2020' for h in hits), f'{ms:.0f} ms')
        r, _ = timed(hn.search, 'rust', k=10, since='2015-06', until='2015-06', types='comment')
        hits = list(r)
        check('a one-month window holds', hits and all(h.date.startswith('2015-06') and h.kind == 'comment' for h in hits))
        r = hn.search(author='pg', k=5, types='story')
        check('author filter, no query', all(h.by == 'pg' for h in r) and len(list(r)) > 0)
        r = hn.search(types='job', since='2024', k=5)
        check('no query: the most recent jobs', len(list(r)) == 5 and all(h.kind == 'job' for h in r))

        r, ms = timed(hn.search, 'kubernetes is too complex for small teams', k=10, deadline_ms=1)
        check('deadline_ms=1 still answers (degraded)', len(list(r)) == 10 and r.rerank.get('degraded'), f'{ms:.0f} ms')

        it, ms = timed(hn.get, 1)
        check('get(1): the first HN item', it is not None and it.kind == 'story' and it.by == 'pg', f'{it.title!r}, {ms:.0f} ms')
        it = hn.get(ids[0])
        check('get() of a search result', it is not None and it.id == ids[0] and (it.text or it.title))
        check('get() of an id that does not exist', hn.get(4_000_000_000) is None)
        check('get_many()', [x.id if x else None for x in hn.get_many([1, 2, 3])] == [1, 2, 3])

        v = hn.embed('hello world')
        check('embed(): a 768-dim vector', len(v) == 768)

        async def both():
            return await asyncio.gather(hn.asearch(q, k=10), hn.aget(1))
        ar, ag = asyncio.run(both())
        check('asyncio: asearch / aget', [h.id for h in ar] == ids and ag.id == 1)

        try:
            hn.search(q, k=0)
            check('a bad query raises ValueError', False)
        except ValueError:
            check('a bad query raises ValueError', True)

        m = hn.memory()
        b = m['budget']
        check('the RAM ledger stays within the budget', b['peak'] <= b['limit'],
              f'used {b["used"] / 1e9:.2f} GB, peak {b["peak"] / 1e9:.2f} GB, limit {b["limit"] / 1e9:.2f} GB, '
              f'process private {m["process"]["private"] / 1e9:.2f} GB')

        # the HTTP server on this archive, stopped with an Event
        port, stop = free_port(), threading.Event()
        th = threading.Thread(target=hn.serve, kwargs=dict(port=port, threads=2, stop=stop), daemon=True)
        th.start()
        for _ in range(100):
            try:
                st_code, body = http(port, '/stats', 5)
                break
            except OSError:
                time.sleep(0.1)
        else:
            st_code, body = 0, {}
        check('Archive.serve(): /stats', st_code == 200 and body.get('docs') == st['docs'])
        st_code, body = http(port, '/search?q=' + urllib.parse.quote_plus(q) + '&k=10')
        check('Archive.serve(): /search equals search()', [h['id'] for h in body['results']] == ids)
        st_code, body = http(port, '/item/1')
        check('Archive.serve(): /item/1', body.get('id') == 1)
        stop.set()
        th.join(60)
        check('Archive.serve() stops on the Event', not th.is_alive())

    # the `hnz` console script installed with the wheel
    # (a development install has no hnz.exe: `python -m hnz` is the same command)
    exe = os.path.join(os.path.dirname(sys.executable), 'hnz.exe')
    cmd = [exe] if os.path.exists(exe) else [sys.executable, '-m', 'hnz']
    out = subprocess.run(cmd + ['--version'], capture_output=True, text=True)
    check('`hnz --version`', out.stdout.strip() == f'hnz {hnz.__version__}', ' '.join(cmd))
    port = free_port()
    p = subprocess.Popen(cmd + ['serve', '--root', a.root, '--ram', a.ram, '--port', str(port), '--threads', '2'],
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        banner = p.stderr.readline().strip()
        _, body = http(port, '/search?q=' + urllib.parse.quote_plus('rust vs go') + '&k=3')
        check('`hnz serve`: answers a search', len(body['results']) == 3, banner)
    except Exception as e:
        check('`hnz serve`: answers a search', False, repr(e))
    finally:
        p.terminate()
        p.wait(60)

    print(f'\n{"ALL PASSED" if not FAILS else f"{len(FAILS)} FAILED: " + ", ".join(FAILS)}')
    return len(FAILS)


if __name__ == '__main__':
    sys.exit(main())
