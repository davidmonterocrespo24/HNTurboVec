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
    assert 'rerank' in r and r['rerank']['candidates'] >= 100 and r['rerank']['embedded'] == 50
    assert len(r['results']) == 10
    p3 = hn.search(QUERIES[1], k=10, raw=True, rerank=False)
    assert 'rerank' not in p3 and len(p3['results']) == 10


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


def test_filters_hold(hn):
    r = hn.search(QUERIES[0], k=20, types='comment', since='2015-06-10', until='2015-06-20')
    assert len(r) > 0
    assert all(h.kind == 'comment' and '2015-06-10' <= h.date[:10] <= '2015-06-20' for h in r)
