"""P1 acceptance tests (Python library on the current format).

They need the sample archive (out/: 2010-06, 2015-06, 2024-01), the models (models/), the original
Parquet files (data/) and the hnz.exe binary. From the project root:
    .venv\\Scripts\\python -m pytest hnz/py/tests -q
"""
import concurrent.futures as cf
import json
import os
import subprocess

import pyarrow.parquet as pq
import pytest

import hnz

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
OUT = os.path.join(ROOT, 'out')
HNZ = os.path.join(ROOT, 'hnz', 'target', 'release', 'hnz.exe')
QUERIES = ['how do I deal with burnout as a software engineer', 'rust async runtime', 'is the iPad going to kill netbooks',
           'startup funding advice', 'postgres vs mysql', 'best text editor for programmers']


@pytest.fixture(scope='module')
def hn():
    a = hnz.open(OUT, ram='4GB', preload=True)
    yield a
    a.close()


def cli(*args):
    env = dict(os.environ, HNZ_OV_LIBS=hnz._ov_libs() or '')
    out = subprocess.run([HNZ, 'search', '--root', OUT, '--json', *args], cwd=ROOT, env=env,
                         capture_output=True, text=True, encoding='utf-8', check=True).stdout
    return json.loads(out)


def test_open_and_stats(hn):
    s = hn.stats()
    assert s['docs'] == 637046
    assert sorted(x['name'] for x in s['segments']) == ['seg-2010-06', 'seg-2015-06', 'seg-2024-01']
    assert s['ram_budget'] == 4_000_000_000


def same_text(a, b):
    """Same text, except that a record already cached in full (e.g. in the CLI's disk cache) comes
    back complete (`partial=False`) where the other side decoded only the 64-token snippet."""
    if a['partial'] == b['partial']:
        return a['text'] == b['text']
    snip, whole = (a, b) if a['partial'] else (b, a)
    return whole['text'].startswith(snip['text'].rstrip())


@pytest.mark.parametrize('q', QUERIES[:3])
def test_same_results_as_cli(hn, q):
    """The library returns exactly what `hnz search` returns (same query, same filters)."""
    for kw, args in (({}, []), ({'mode': 'keyword'}, ['--mode', 'keyword']),
                     ({'since': '2015', 'until': '2015', 'types': 'comment'}, ['--from', '2015', '--to', '2015', '--type', 'comment'])):
        py = hn.search(q, k=10, raw=True, **kw)['results']
        c = cli('-k', '10', *args, q)
        assert [r['id'] for r in py] == [r['id'] for r in c], (q, kw)
        assert all(same_text(a, b) for a, b in zip(py, c)), (q, kw)


def test_listing_without_text(hn):
    r = hn.search(types='story', since='2024', k=5)
    assert len(r) == 5 and all(h.kind == 'story' for h in r)
    assert [h.time for h in r] == sorted((h.time for h in r), reverse=True)


def test_get_is_lossless(hn):
    """get() returns the exact text, title, url and author of the original Parquet (all 3 tiers)."""
    for month, year in (('2024-01', 2024), ('2015-06', 2015), ('2010-06', 2010)):
        t = pq.read_table(os.path.join(ROOT, 'data', str(year), f'{month}.parquet'),
                          columns=['id', 'by', 'title', 'url', 'text', 'deleted'])
        rows = t.to_pylist()
        for row in rows[::max(1, len(rows) // 25)][:25]:
            it = hn.get(row['id'])
            assert it is not None, row['id']
            if row['deleted']:
                assert it.deleted
                continue
            assert (it.html, it.title, it.url, it.by) == (row['text'] or '', row['title'] or '', row['url'] or '', row['by'] or ''), row['id']
    assert hn.get(1) is None or hn.get(1).id == 1


def test_concurrent_queries_match_sequential(hn):
    """8 threads with 48 mixed queries (hybrid, keyword, full text, get) = the same results as one
    at a time."""
    jobs = []
    for i, q in enumerate(QUERIES * 2):
        jobs.append(('search', q, {'k': 10}))
        jobs.append(('search', q, {'k': 5, 'mode': 'keyword', 'text': 'full' if i % 2 else 'snippet'}))
        jobs.append(('search', q, {'k': 5, 'since': '2010', 'until': '2015'}))
        jobs.append(('get', 9747338, {}))

    def run(j):
        kind, q, kw = j
        if kind == 'get':
            return hn.get(q, raw=True)
        return [(h['id'], h['text']) for h in hn.search(q, raw=True, **kw)['results']]

    seq = [run(j) for j in jobs]
    with cf.ThreadPoolExecutor(8) as ex:
        par = list(ex.map(run, jobs))
    assert par == seq


def test_errors(hn):
    with pytest.raises(ValueError):
        hn.search('x', types='foo')
    with pytest.raises(ValueError):
        hn.search('x', since='2015-13')
    with pytest.raises(ValueError):
        hn.search('x', k=0)
    with pytest.raises(ValueError):
        hn.search('x', mode='nope')
    with pytest.raises(ValueError):
        hnz.open(OUT, ram='muchos')


def test_closed_archive():
    a = hnz.open(OUT)
    a.close()
    assert a.closed
    with pytest.raises(RuntimeError):
        a.search('x')


def test_datetime_bounds_are_exact(hn):
    """A datetime is an exact instant (UTC), not the whole day."""
    import datetime as dt
    it = hn.search(types='comment', since='2015-06-10', until='2015-06-10', k=1)[0]
    t = dt.datetime.fromtimestamp(it.time, dt.timezone.utc)
    same = hn.search(types='comment', since=t, until=t, k=50)
    assert it.id in [h.id for h in same] and all(h.time == it.time for h in same)
    after = hn.search(types='comment', since=t + dt.timedelta(seconds=1), until=t + dt.timedelta(hours=1), k=500)
    assert it.id not in [h.id for h in after] and all(h.time > it.time for h in after)
    naive = hn.search(types='comment', since=t.replace(tzinfo=None), until=t.replace(tzinfo=None), k=50)
    assert [h.id for h in naive] == [h.id for h in same]          # no time zone = UTC


def test_int_bounds_are_years(hn):
    assert len(hn.search(types='story', since=2015, until=2015, k=3)) == 3
    with pytest.raises(ValueError):
        hn.search('x', since=1700000000)                           # a timestamp is not a year
    with pytest.raises(ValueError):
        hn.search('x', since='99999999999999999')


def test_open_empty_folder_fails(tmp_path):
    with pytest.raises(RuntimeError, match='no hay segmentos'):
        hnz.open(tmp_path)


def test_lm_work_does_not_block_hot_tier(hn):
    """A heavy LM-tier query (full text) does not block a get() of the zstd tier."""
    import threading
    import time
    hot = hn.search(types='story', since='2024-01-20', k=1)[0].id
    t0 = time.perf_counter()
    th = threading.Thread(target=lambda: hn.search('the', k=100, since='2010', until='2010', text='full', mode='keyword'))
    th.start()
    time.sleep(0.3)                                                # the LM batch is already running
    t1 = time.perf_counter()
    assert hn.get(hot) is not None
    dt_hot = time.perf_counter() - t1
    th.join()
    dt_lm = time.perf_counter() - t0
    assert dt_hot < 0.5 and dt_lm > 2 * dt_hot, (dt_hot, dt_lm)
