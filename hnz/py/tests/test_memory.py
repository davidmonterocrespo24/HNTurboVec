"""P2 acceptance tests: RAM governance (budget planner, disk reads, capped LM batch, admission).

They need the sample archive (out/), the models (models/) and the installed `hnz` package. From
the project root:
    .venv\\Scripts\\python -m pytest hnz/py/tests -q
"""
import concurrent.futures as cf
import json
import os
import subprocess
import sys

import pytest

import hnz

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
OUT = os.path.join(ROOT, 'out')
WORKER = os.path.join(os.path.dirname(__file__), 'mem_worker.py')
MiB = 2 ** 20
# above the minimum: the LM model's measured size varies by a few MiB between opens; with this
# margin the pools and the cache still take all of it, so nothing is loaded into RAM
TIGHT = 24 * MiB
QUERIES = ['how do I deal with burnout as a software engineer', 'rust async runtime', 'is the iPad going to kill netbooks',
           'startup funding advice', 'postgres vs mysql', 'best text editor for programmers']


@pytest.fixture(scope='module')
def full():
    """Archive without a limit: everything in RAM."""
    a = hnz.open(OUT, ram=None, preload=True)
    yield a
    a.close()


@pytest.fixture(scope='module')
def floor(full):
    return full.memory()['plan']['floor']


@pytest.fixture(scope='module')
def tight(floor):
    """Archive with the minimum budget: nothing loaded (everything read from disk) and the LM batch
    capped at its minimum."""
    a = hnz.open(OUT, ram=floor + TIGHT, preload=True)
    yield a
    a.close()


def test_budget_error_below_minimum(floor):
    with pytest.raises(hnz.BudgetError, match='hacen falta al menos'):
        hnz.open(OUT, ram=floor // 2)
    assert issubclass(hnz.BudgetError, MemoryError) and issubclass(hnz.BusyError, TimeoutError)


def test_tight_budget_reads_everything_from_disk(tight, floor):
    m = tight.memory()
    parts = m['engine']['parts']
    for p in ('vectors', 'postings', 'text'):
        assert parts[p]['resident_bytes'] == 0, (p, parts[p])
    assert m['plan']['lm_pool'] < 64 * MiB
    b = m['budget']
    assert b['used'] <= b['limit'] and b['peak'] <= b['limit']


def test_same_results_on_disk_and_in_ram(full, tight):
    """Disk reads (vectors, BM25 by blocks, metadata, text) and the capped LM batch give exactly the
    same results as everything in RAM."""
    for q in QUERIES:
        for kw in ({'k': 10}, {'k': 10, 'mode': 'keyword'}, {'k': 10, 'mode': 'vector'},
                   {'k': 20, 'since': '2010', 'until': '2015'}, {'k': 40, 'mode': 'keyword', 'text': 'full'},
                   {'k': 15, 'types': 'comment', 'since': '2015-06-10', 'until': '2015-06-12'}):
            a = [(h['id'], h['text'], h['html'], round(h['score'], 6)) for h in full.search(q, raw=True, **kw)['results']]
            b = [(h['id'], h['text'], h['html'], round(h['score'], 6)) for h in tight.search(q, raw=True, **kw)['results']]
            assert a == b, (q, kw)
    for kw in ({'types': 'job', 'k': 20}, {'author': 'pg', 'k': 20}, {'types': 'story', 'since': '2024-01-20', 'k': 50}):
        assert [h.id for h in full.search(**kw)] == [h.id for h in tight.search(**kw)], kw
    for i in (9747338, 1424000, 9700000, 39100000):
        assert full.get(i, raw=True) == tight.get(i, raw=True)


def test_memory_report(tight):
    s = tight.stats()
    m = s['memory']
    assert set(m) >= {'budget', 'plan', 'engine', 'embedder', 'admission', 'process'}
    b = m['budget']
    assert b['limit'] == s['ram_budget'] and b['used'] == b['models'] + b['index'] + b['cache'] + b['working']
    assert m['embedder']['loaded'] and m['embedder']['measured']['steady'] > 100 * MiB
    assert m['admission']['max_concurrent'] == 8 and m['admission']['running'] == 0


def test_busy_error_when_no_turn():
    a = hnz.open(OUT, ram='2GB', preload=True, max_concurrent=1, timeout=0)
    try:
        def heavy(i):
            try:
                a.search(QUERIES[i % len(QUERIES)], k=40, mode='keyword', text='full', since='2010', until='2015')
                return 'ok'
            except hnz.BusyError:
                return 'busy'
        with cf.ThreadPoolExecutor(6) as ex:
            got = list(ex.map(heavy, range(12)))
        assert 'busy' in got and 'ok' in got, got
        assert a.memory()['admission']['rejected'] >= got.count('busy')
        assert len(a.search('rust', k=5)) == 5          # still works afterwards
    finally:
        a.close()


def test_persistent_cache(tmp_path):
    q = dict(k=10, mode='keyword', text='full', since='2010', until='2010')
    a = hnz.open(OUT, ram='2GB', cache_dir=tmp_path)
    r1 = a.search('startup funding advice', raw=True, **q)['results']
    n = a.memory()['engine']['cache']['l2_entries']
    a.close()
    del a
    import gc
    gc.collect()
    assert n and n >= 10
    b = hnz.open(OUT, ram='2GB', cache_dir=tmp_path)
    assert b.memory()['engine']['cache']['l2_entries'] == n
    assert b.search('startup funding advice', raw=True, **q)['results'] == r1
    b.close()


def _worker(ram, threads=8, seconds=60):
    out = subprocess.run([sys.executable, WORKER, OUT, str(ram), str(threads), str(seconds)], cwd=ROOT,
                         capture_output=True, text=True, encoding='utf-8', check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


@pytest.mark.parametrize('extra', [TIGHT, 'all'])
def test_process_memory_stays_within_budget(floor, extra):
    """Acceptance criterion of P2: with 8 concurrent clients, the process memory (working set peak
    as the OS reports it, above the baseline before opening) stays <= 1.03 x ram."""
    ram = floor + extra if extra != 'all' else 2_000_000_000
    r = _worker(ram)
    assert not r['errors'], r['errors']
    assert r['ops'] > 50, r
    grown = r['end']['peak_working_set'] - r['base']['working_set']
    assert grown <= 1.03 * ram, (grown / MiB, ram / MiB, r['budget'])
    assert r['budget']['peak'] <= ram


def test_request_limits(tight):
    """Review fixes: query length cap, BM25 terms capped (long queries still work), absurd timeouts
    and picklable exceptions."""
    import pickle
    with pytest.raises(ValueError, match='demasiado larga'):
        tight.search('x' * 9000)
    with pytest.raises(ValueError):
        tight.embed('y' * 9000)
    # ~1,000 terms (< 8 KB): only the 64 rarest that exist count; unknown words must not crowd
    # out the real ones
    words = ' '.join(f'zq{i}x' for i in range(1000)) + ' postgres mysql'
    assert len(words.encode()) < 8192
    got = tight.search(words, k=10, mode='keyword')
    assert [h.id for h in got] == [h.id for h in tight.search('postgres mysql', k=10, mode='keyword')]
    with pytest.raises(ValueError, match='timeout'):
        hnz.open(OUT, timeout=1e300)
    with pytest.raises(ValueError, match='timeout'):
        hnz.open(OUT, timeout=float('nan'))
    for e in (hnz.BudgetError('a'), hnz.BusyError('b')):
        e2 = pickle.loads(pickle.dumps(e))
        assert type(e2) is type(e) and e2.args == e.args


def test_large_k_with_filters_on_disk(full, tight):
    """k=500 with a type filter: the declared working memory covers it and results match."""
    for q in QUERIES[:2]:
        a = [h.id for h in full.search(q, k=500, types='story')]
        b = [h.id for h in tight.search(q, k=500, types='story')]
        assert a == b and len(a) == 500
    m = tight.memory()['admission']
    assert m['running'] == 0 and m['pool_in_use'] == 0


def test_persistent_cache_survives_small_budget(tmp_path, floor):
    """Reopening with a budget whose cache share is smaller keeps the newest entries (it used to
    empty the cache), and the ledger never goes above the limit while opening."""
    q = dict(k=40, mode='keyword', text='full', since='2010', until='2010')
    a = hnz.open(OUT, ram='2GB', cache_dir=tmp_path)
    r1 = a.search('startup funding advice', raw=True, **q)['results']
    n = a.memory()['engine']['cache']['l2_entries']
    a.close()
    del a
    import gc
    gc.collect()
    b = hnz.open(OUT, ram=floor + TIGHT, cache_dir=tmp_path)
    m = b.memory()
    kept = m['engine']['cache']['l2_entries']
    assert 0 < kept <= n
    assert m['budget']['peak'] <= m['budget']['limit']
    assert b.search('startup funding advice', raw=True, **q)['results'] == r1
    b.close()
