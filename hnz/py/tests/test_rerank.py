"""P4 tests: reranking a compact block's candidates by their prefix.

They need the sample block packed with the dynamic-shape embedder (work/p3/hn-sample-v3.hnz, from
`hnz pack --spec work/p3/pack-v3.json --out work/p3/hn-sample-v3.hnz`) and the installed `hnz`
package. From the project root:
    .venv\\Scripts\\python -m pytest hnz/py/tests/test_rerank.py -q
"""
import concurrent.futures as cf
import os

import pytest

import hnz

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
BLOCK = os.path.join(ROOT, 'work', 'p3', 'hn-sample-v3.hnz')
MiB = 2 ** 20
QUERIES = ['how do I deal with burnout as a software engineer', 'is the iPad going to kill netbooks',
           'startup funding advice', 'best text editor for programmers']

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(BLOCK, 'hn.hnzb')), reason='no v3 sample block (hnz pack)')


@pytest.fixture(scope='module')
def hn():
    a = hnz.open(BLOCK, ram='2GB', preload=True)
    yield a
    a.close()


@pytest.fixture(scope='module')
def floor(hn):
    return hn.memory()['plan']['floor']


def ids_scores(r):
    return [(h['id'], round(h['score'], 5)) for h in r['results']]


def test_rerank_on_by_default(hn):
    r = hn.search(QUERIES[1], k=10, raw=True)
    # 80 vector + up to 20 rare-term candidates; the top 50 by the cheap score are re-embedded,
    # plus a few speculative ones (overlap)
    assert 'rerank' in r and r['rerank']['candidates'] >= 80 and r['rerank']['embedded'] >= 50
    assert len(r['results']) == 10
    p3 = hn.search(QUERIES[1], k=10, raw=True, rerank=False)
    assert 'rerank' not in p3 and len(p3['results']) == 10


def test_embedder_reproduces_the_packed_vectors(hn):
    """At load, the block's short canaries are decoded, re-embedded and compared with their stored
    codes: only a few of 256 signs differ (a broken chain would flip about 128)."""
    m = hn.memory()['embedder']['measured']
    assert m['docs'] and m['check_flips'] is not None
    assert m['check_flips'] <= 32, m


def test_keyword_mode_finds_frequent_words(hn):
    """Today's P3 block returns nothing for these (frequent words have no postings)."""
    for q in ('rust async runtime', 'postgres vs mysql'):
        r = hn.search(q, k=10, mode='keyword', raw=True)
        assert len(r['results']) == 10, q
        assert all(h['bm25'] > 0 for h in r['results'])
        assert 'warning' in r
        assert len(hn.search(q, k=10, mode='keyword', rerank=False)) == 0


def test_deterministic_cold_and_warm(hn):
    a = hn.search(QUERIES[0], k=20, raw=True)       # possibly cold
    b = hn.search(QUERIES[0], k=20, raw=True)       # warm: prefixes cached
    assert ids_scores(a) == ids_scores(b)


def test_same_results_on_disk_and_in_ram(hn, floor):
    tight = hnz.open(BLOCK, ram=floor + 64 * MiB, preload=True)
    try:
        for q in QUERIES:
            for kw in ({'k': 10}, {'k': 10, 'mode': 'vector'}, {'k': 10, 'since': '2015', 'until': '2015'},
                       {'k': 10, 'mode': 'keyword'}):
                assert ids_scores(hn.search(q, raw=True, **kw)) == ids_scores(tight.search(q, raw=True, **kw)), (q, kw)
    finally:
        tight.close()


def test_concurrent_reranks_match_sequential(hn):
    jobs = [(q, kw) for q in QUERIES for kw in ({'k': 10}, {'k': 5, 'mode': 'keyword'}, {'k': 10, 'text': 'full'})]
    run = lambda j: ids_scores(hn.search(j[0], raw=True, **j[1]))
    seq = [run(j) for j in jobs]
    with cf.ThreadPoolExecutor(8) as ex:
        assert list(ex.map(run, jobs)) == seq


def test_results_reuse_the_decoded_prefixes(hn):
    """A reranked result's text is the P3 snippet of the same document, or its whole record when
    the record fits in the prefix (then `partial` is false and `html` is the original)."""
    r = hn.search(QUERIES[2], k=10, raw=True)
    for h in r['results']:
        it = hn.get(h['id'])
        if not h['partial']:
            assert h['html'] == it.html and h['title'] == it.title
        else:
            assert it.text.replace('\n', ' ').startswith(h['text'][:40].replace('\n', ' ')[:20]) or h['text'] == ''


def test_deadline_degrades_but_answers(hn):
    """A deadline is opt-in: missed, the answer still comes (stage 1's, or the cheap score's
    order) and says it is degraded; generous, it changes nothing."""
    q = 'which compiler optimizations matter for a hobby kernel'
    r = hn.search(q, k=10, raw=True, deadline_ms=1)
    assert r['rerank']['degraded'] in ('decode', 'embed')
    assert len(r['results']) == 10 and 'warning' in r
    full = hn.search(q, k=10, raw=True)
    assert full['rerank']['degraded'] is None and 'warning' not in full
    assert ids_scores(hn.search(q, k=10, raw=True, deadline_ms=600_000)) == ids_scores(full)
    # the prefixes are cached now: nothing to decode, so a missed deadline leaves R0's order
    r = hn.search(q, k=10, raw=True, deadline_ms=1)
    assert r['rerank']['degraded'] == 'embed' and len(r['results']) == 10 and 'warning' in r
    res = hn.search(q, k=10, deadline_ms=600_000)
    assert res.rerank is not None and res.rerank['degraded'] is None and res.warning is None
    for bad in (0, -5, float('nan')):
        with pytest.raises(ValueError):
            hn.search(q, deadline_ms=bad)


def test_fusion_parameters_reach_the_reranker(hn):
    """w_bm25 / c_bm25 weigh BM25 in the reranked fusion too (None = its defaults 0.15 / 40)."""
    q = 'postgres replication lag monitoring'
    base = ids_scores(hn.search(q, k=10, raw=True))
    assert ids_scores(hn.search(q, k=10, raw=True, w_bm25=0.15, c_bm25=40.0)) == base
    assert any(ids_scores(hn.search(q, k=10, raw=True, w_bm25=w)) != base for w in (0.0, 2.0))
    with pytest.raises(ValueError):
        hn.search(q, w_bm25=-1.0)


def test_reranking_progresses_under_query_and_get_streams(hn):
    """Streams of query embeddings and of interactive LM decodes (get) slow a reranked search but
    never stall it, and a deadline still bounds it."""
    import threading
    import time
    ids = [h['id'] for h in hn.search('', k=300, since='2010-06-01', until='2010-06-03', raw=True)['results']]
    stop = threading.Event()

    def embeds():
        while not stop.is_set():
            hn.embed('a steady stream of query embeddings')

    def gets():
        for i in ids:
            if stop.is_set():
                return
            hn.get(i)

    ts = [threading.Thread(target=f) for f in (embeds, embeds, gets)]
    for t in ts:
        t.start()
    try:
        t0 = time.perf_counter()
        r = hn.search('database index structures like btrees and lsm trees', k=10, raw=True, deadline_ms=4000)
        assert len(r['results']) == 10
        assert time.perf_counter() - t0 < 6.0
        t0 = time.perf_counter()
        r = hn.search('garbage collectors in managed runtimes', k=10, raw=True)
        assert len(r['results']) == 10 and r['rerank']['degraded'] is None
        assert time.perf_counter() - t0 < 60.0
    finally:
        stop.set()
        for t in ts:
            t.join()


def test_rerank_depth(hn):
    q = QUERIES[2]
    r = hn.search(q, k=10, raw=True, rerank_depth=60)
    assert len(r['results']) == 10 and r['rerank']['candidates'] <= 60 + 20
    assert r['rerank']['embedded'] >= 40
    deep = hn.search(q, k=50, raw=True, rerank_depth=200)
    assert len(deep['results']) == 50 and deep['rerank']['candidates'] > 100
    for bad in (0, 501):
        with pytest.raises(ValueError):
            hn.search(q, rerank_depth=bad)


def test_filters_hold(hn):
    r = hn.search(QUERIES[0], k=20, types='comment', since='2015-06-10', until='2015-06-20')
    assert len(r) > 0
    assert all(h.kind == 'comment' and '2015-06-10' <= h.date[:10] <= '2015-06-20' for h in r)


def test_process_memory_with_reranks(floor):
    """8 clients for 60 s (their searches are reranked) at the floor + 64 MiB: no errors, process
    growth <= 1.03 x ram and the ledger <= ram."""
    import json
    import subprocess
    import sys
    ram = floor + 64 * MiB
    worker = os.path.join(os.path.dirname(__file__), 'mem_worker.py')
    out = subprocess.run([sys.executable, worker, BLOCK, str(ram), '8', '60'], cwd=ROOT,
                         capture_output=True, text=True, encoding='utf-8', check=True).stdout
    r = json.loads(out.strip().splitlines()[-1])
    assert not r['errors'], r['errors']
    assert r['ops'] > 8, r
    assert r['end']['peak_working_set'] - r['base']['working_set'] <= 1.03 * ram, r
    assert r['budget']['peak'] <= ram
