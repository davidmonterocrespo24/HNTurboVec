"""P3 acceptance tests: the compact block (`hn.hnzb`) from Python.

They need the sample block (work/p3/hn-sample.hnz, from `hnz pack --spec work/p3/pack.json --out
work/p3/hn-sample.hnz`), the original Parquet files (data/), the installed `hnz` package and, for
the CLI comparison and the incompatibility cases, hnz/target_p3/release/hnz.exe and
examples/rewrite_block.exe (`cargo build --release --bin hnz --example rewrite_block` with
CARGO_TARGET_DIR=target_p3). From the project root:
    .venv\\Scripts\\python -m pytest hnz/py/tests/test_block.py -q
"""
import concurrent.futures as cf
import json
import os
import random
import shutil
import subprocess
import sys

import pyarrow.parquet as pq
import pytest

import hnz

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
BLOCK = os.path.join(ROOT, 'work', 'p3', 'hn-sample.hnz')
TARGET = os.path.join(ROOT, 'hnz', 'target_p3', 'release')
HNZ = os.path.join(TARGET, 'hnz.exe')
REWRITE = os.path.join(TARGET, 'examples', 'rewrite_block.exe')
WORKER = os.path.join(os.path.dirname(__file__), 'mem_worker.py')
MONTHS = (('2024-01', 2024), ('2015-06', 2015), ('2010-06', 2010))
KIND = {1: 'story', 2: 'comment', 3: 'poll', 4: 'pollopt', 5: 'job'}
MiB = 2 ** 20
TIGHT = 24 * MiB
QUERIES = ['how do I deal with burnout as a software engineer', 'rust async runtime', 'is the iPad going to kill netbooks',
           'startup funding advice', 'postgres vs mysql', 'best text editor for programmers']
ID = 9747338            # a 2015 comment

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(BLOCK, 'hn.hnzb')), reason='no sample block (hnz pack)')


def parquet(month, year, columns):
    return pq.read_table(os.path.join(ROOT, 'data', str(year), f'{month}.parquet'), columns=columns).to_pylist()


@pytest.fixture(scope='module')
def full():
    """Everything in RAM."""
    a = hnz.open(BLOCK, ram=None, preload=True)
    yield a
    a.close()


@pytest.fixture(scope='module')
def floor(full):
    return full.memory()['plan']['floor']


@pytest.fixture(scope='module')
def tight(floor):
    """The minimum budget: every part read from disk."""
    a = hnz.open(BLOCK, ram=floor + TIGHT, preload=True)
    yield a
    a.close()


@pytest.fixture(scope='module')
def info():
    return hnz.inspect(BLOCK)


def test_open_stats_and_verify(full, info):
    s = full.stats()
    assert s['docs'] == 637046
    b = s['block']
    assert b['parts'] == 18 and len(b['snapshot']) == 32
    r = hnz.verify_block(BLOCK, 'full')
    assert r['ok'] and r['snapshot'] == b['snapshot']
    assert [c['check'] for c in r['checks']] == ['header', 'quick', 'full']
    assert info['n_docs'] == 637046 and info['source']['n_late'] == 40


def test_get_is_lossless(tight):
    """get() of 300 random items per month (deleted rows too) = the Parquet, every column but the
    derived `words` and the order of `kids` (stored, checked by `hnz verify-block --level deep`)."""
    cols = ['id', 'by', 'title', 'url', 'text', 'deleted', 'dead', 'time', 'score', 'parent', 'type', 'poll', 'parts', 'descendants']
    for month, year in MONTHS:
        for row in random.Random(7).sample(parquet(month, year, cols), 300):
            it = tight.get(row['id'])
            assert it is not None, row['id']
            got = (it.html, it.title, it.url, it.by, it.points, it.parent_id or 0, it.time, it.kind, it.dead, it.deleted,
                   it.poll or 0, it.parts, it.descendants)
            want = (row['text'], row['title'], row['url'], row['by'], row['score'], row['parent'], int(row['time'].timestamp()),
                    KIND[row['type']], bool(row['dead']), bool(row['deleted']), row['poll'], row['parts'], row['descendants'])
            assert got == want, row['id']


def test_items_out_of_time_order(tight):
    """Ids whose time is later than that of a higher id (0.63 % of HN) are stored in time order:
    found by id and inside their exact date range."""
    rows = [r for m, y in MONTHS for r in parquet(m, y, ['id', 'time', 'type', 'dead', 'deleted'])]
    rows.sort(key=lambda r: r['id'])
    late, low = [], None
    for r in reversed(rows):
        t = int(r['time'].timestamp())
        if low is not None and t > low:
            late.append((r['id'], t, KIND[r['type']], bool(r['dead'] or r['deleted'])))
        low = t if low is None else min(low, t)
    assert len(late) == 40
    import datetime as dt
    searched = 0
    for i, t, kind, hidden in late:
        assert tight.get(i).time == t
        if not hidden:                                     # searches never return dead or deleted items
            at = dt.datetime.fromtimestamp(t, dt.timezone.utc)
            assert i in [h.id for h in tight.search(types=kind, since=at, until=at, k=50)]
            searched += 1
    assert searched >= 5


def same_text(a, b):
    """Same text, except that a record already cached in full (e.g. by an earlier get()) comes back
    complete (`partial=False`) where the other side decoded only the 64-token snippet."""
    if a['partial'] == b['partial']:
        return (a['text'], a['html']) == (b['text'], b['html'])
    snip, whole = (a, b) if a['partial'] else (b, a)
    return whole['text'].startswith(snip['text'].rstrip())


def test_same_results_on_disk_and_in_ram(full, tight):
    for q in QUERIES:
        for kw in ({'k': 10}, {'k': 10, 'mode': 'keyword'}, {'k': 10, 'mode': 'vector'},
                   {'k': 20, 'since': '2010', 'until': '2015'}, {'k': 40, 'mode': 'keyword', 'text': 'full'},
                   {'k': 15, 'types': 'comment', 'since': '2015-06-10', 'until': '2015-06-12'}):
            a = full.search(q, raw=True, **kw)['results']
            b = tight.search(q, raw=True, **kw)['results']
            assert [(h['id'], round(h['score'], 6)) for h in a] == [(h['id'], round(h['score'], 6)) for h in b], (q, kw)
            assert all(same_text(x, y) for x, y in zip(a, b)), (q, kw)
    for kw in ({'types': 'job', 'k': 20}, {'author': 'pg', 'k': 20}, {'types': 'story', 'since': '2024-01-20', 'k': 50}):
        assert [h.id for h in full.search(**kw)] == [h.id for h in tight.search(**kw)], kw
    for i in (ID, 1424000, 9700000, 39100000):
        assert full.get(i, raw=True) == tight.get(i, raw=True)


def test_concurrent_queries_match_sequential(tight):
    jobs = []
    for i, q in enumerate(QUERIES * 2):
        jobs.append(('search', q, {'k': 10}))
        jobs.append(('search', q, {'k': 5, 'mode': 'keyword', 'text': 'full' if i % 2 else 'snippet'}))
        jobs.append(('search', q, {'k': 5, 'since': '2010', 'until': '2015'}))
        jobs.append(('get', ID, {}))

    def run(j):
        kind, q, kw = j
        if kind == 'get':
            return tight.get(q, raw=True)
        return [(h['id'], h['text']) for h in tight.search(q, raw=True, **kw)['results']]

    seq = [run(j) for j in jobs]
    with cf.ThreadPoolExecutor(8) as ex:
        assert list(ex.map(run, jobs)) == seq


def test_budget_error_below_minimum(floor):
    with pytest.raises(hnz.BudgetError, match='hacen falta al menos'):
        hnz.open(BLOCK, ram=floor // 2)


@pytest.mark.skipif(not os.path.exists(HNZ), reason='no P3 CLI build')
def test_same_results_as_cli(full):
    env = dict(os.environ, HNZ_OV_LIBS=hnz._ov_libs() or '')
    for q in QUERIES[:3]:
        for kw, args in (({}, []), ({'mode': 'keyword'}, ['--mode', 'keyword']),
                         ({'since': '2015', 'until': '2015', 'types': 'comment'}, ['--from', '2015', '--to', '2015', '--type', 'comment'])):
            out = subprocess.run([HNZ, 'search', '--root', BLOCK, '--json', '-k', '10', *args, q], cwd=ROOT, env=env,
                                 capture_output=True, text=True, encoding='utf-8', check=True).stdout
            assert [r['id'] for r in full.search(q, k=10, raw=True, **kw)['results']] == [r['id'] for r in json.loads(out)], (q, kw)


def test_process_memory_stays_within_budget(floor):
    """8 concurrent clients for 60 s at the minimum budget: process growth <= 1.03 x ram."""
    ram = floor + TIGHT
    out = subprocess.run([sys.executable, WORKER, BLOCK, str(ram), '8', '60'], cwd=ROOT,
                         capture_output=True, text=True, encoding='utf-8', check=True).stdout
    r = json.loads(out.strip().splitlines()[-1])
    assert not r['errors'], r['errors']
    assert r['ops'] > 50, r
    assert r['end']['peak_working_set'] - r['base']['working_set'] <= 1.03 * ram, r
    assert r['budget']['peak'] <= ram


# ---------------------------------------------------------------- damaged and incompatible blocks

def copy_block(tmp_path, name='blk', embed_copy=False):
    """A copy of the container (the embedder files hard-linked, or copied with `embed_copy`)."""
    d = tmp_path / name
    d.mkdir()
    shutil.copyfile(os.path.join(BLOCK, 'hn.hnzb'), d / 'hn.hnzb')
    for f in ('embed.xml', 'embed.bin'):
        if embed_copy:
            shutil.copyfile(os.path.join(BLOCK, f), d / f)
        else:
            os.link(os.path.join(BLOCK, f), d / f)
    return d


def flip(path, off, xor=0x40):
    with open(path, 'r+b') as f:
        f.seek(off)
        b = f.read(1)[0]
        f.seek(off)
        f.write(bytes([b ^ xor]))


def part(info, name):
    return next(p for p in info['parts'] if p['name'] == name)


@pytest.mark.parametrize('where', ['superblock', 'toc', 'trailer', 'truncated', 'extended', 'meta/flags', 'model/lm', 'lex/fst', 'text/gsum'])
def test_damage_found_at_open(tmp_path, info, where):
    d = copy_block(tmp_path)
    f = d / 'hn.hnzb'
    size = os.path.getsize(f)
    if where == 'superblock':
        flip(f, 100)
    elif where == 'toc':
        flip(f, info['toc']['off'] + info['toc']['len'] // 2)
    elif where == 'trailer':
        flip(f, size - 20)
    elif where == 'truncated':
        with open(f, 'r+b') as h:
            h.truncate(size - 4096)
    elif where == 'extended':
        with open(f, 'ab') as h:
            h.write(b'\0' * 4096)
    else:
        p = part(info, where)
        flip(f, p['off'] + p['bytes'] // 2)
    with pytest.raises(hnz.CorruptBlockError):
        hnz.open(d)
    with pytest.raises(hnz.CorruptBlockError):
        hnz.verify_block(d, 'header' if where in ('superblock', 'toc', 'trailer', 'truncated', 'extended') else 'quick')


def test_damaged_record_found_when_read(tmp_path, floor):
    loc = hnz.inspect(BLOCK, id=ID)['locate']
    d = copy_block(tmp_path)
    flip(d / 'hn.hnzb', loc['text_group']['off'] + loc['text_group']['len'] // 2)
    a = hnz.open(d, ram=floor + TIGHT)                     # text read from disk: open is fine
    try:
        with pytest.raises(hnz.CorruptBlockError, match='text group'):
            a.get(ID)
        assert a.get(1424000) is not None                  # other records still read
    finally:
        a.close()
    with pytest.raises(hnz.CorruptBlockError):
        hnz.open(d, ram=None)                              # everything loaded: the part hash fails
    with pytest.raises(hnz.CorruptBlockError):
        hnz.verify_block(d, 'full')


def test_damaged_metadata_found_when_read(tmp_path, floor):
    loc = hnz.inspect(BLOCK, id=ID)['locate']
    d = copy_block(tmp_path)
    flip(d / 'hn.hnzb', loc['meta_block']['off'] + loc['meta_block']['len'] // 2)
    a = hnz.open(d, ram=floor + TIGHT)
    try:
        with pytest.raises(hnz.CorruptBlockError, match='meta block'):
            a.get(ID)
        assert a.get(39100000) is not None
    finally:
        a.close()
    with pytest.raises(hnz.CorruptBlockError):
        hnz.verify_block(d, 'full')


def test_damaged_padding_found_by_full(tmp_path, info):
    ps = sorted(info['parts'], key=lambda p: p['off'])
    gap = next(p['off'] + p['bytes'] for p, q in zip(ps, ps[1:]) if q['off'] > p['off'] + p['bytes'])
    d = copy_block(tmp_path)
    flip(d / 'hn.hnzb', gap)
    hnz.open(d).close()
    assert hnz.verify_block(d, 'quick')['ok']
    with pytest.raises(hnz.CorruptBlockError, match='relleno'):
        hnz.verify_block(d, 'full')


def test_damaged_embedder_found_when_loaded(tmp_path):
    d = copy_block(tmp_path, embed_copy=True)
    flip(d / 'embed.bin', os.path.getsize(d / 'embed.bin') // 2)
    a = hnz.open(d)                                        # the embedder loads at the first semantic query
    try:
        assert len(a.search('netbooks', k=5, mode='keyword')) == 5   # a rare word: no embedder needed
        with pytest.raises(hnz.CorruptBlockError, match='embed.bin'):
            a.search('rust async runtime', k=5)
    finally:
        a.close()
    with pytest.raises(hnz.CorruptBlockError):
        hnz.open(d, preload=True)
    with pytest.raises(hnz.CorruptBlockError):
        hnz.verify_block(d, 'full')


def test_damaged_vectors_or_postings_never_panic(tmp_path, info, floor):
    """Vector codes and word postings are read without a hash at query time: damage there gives
    other results or a CorruptBlockError, never a crash; `full` finds it."""
    d = copy_block(tmp_path)
    rnd = random.Random(3)
    for name in ('vec/codes', 'lex/post'):
        p = part(info, name)
        for _ in range(200):
            flip(d / 'hn.hnzb', p['off'] + rnd.randrange(p['bytes']), 1 << rnd.randrange(8))
    a = hnz.open(d, ram=floor + TIGHT)
    try:
        for q in QUERIES + ['kubernetes terraform ansible', 'postgres']:
            for kw in ({'k': 10}, {'k': 10, 'mode': 'keyword'}, {'k': 10, 'mode': 'vector'}):
                try:
                    a.search(q, **kw)
                except hnz.CorruptBlockError:
                    pass
    finally:
        a.close()
    with pytest.raises(hnz.CorruptBlockError):
        hnz.verify_block(d, 'full')


def test_not_a_block(tmp_path):
    (tmp_path / 'x').mkdir()
    (tmp_path / 'x' / 'hn.hnzb').write_bytes(os.urandom(100_000))
    with pytest.raises(hnz.IncompatibleBlockError, match='no es un bloque'):
        hnz.open(tmp_path / 'x')


def test_block_next_to_segments_refused(tmp_path):
    d = copy_block(tmp_path)
    (d / 'seg-2024-01').mkdir()
    (d / 'seg-2024-01' / 'manifest.json').write_text('{}')
    with pytest.raises(RuntimeError, match='una cosa u otra'):
        hnz.open(d)


@pytest.mark.skipif(not os.path.exists(REWRITE), reason='no rewrite_block build')
@pytest.mark.parametrize('change', [
    ['--set', 'text.math_version=1'],
    ['--set', 'meta.row_encoding=3'],
    ['--set', 'vec.sign_rule=2'],
    ['--part', 'model/lm=models/lm-pythia-70m.hnzlm'],
    ['--part', 'future/part=models/lm-hn-70m.tokenizer.json'],
])
def test_intact_but_incompatible(tmp_path, change):
    """Valid hashes everywhere, but a definition, a model or a part this binary does not read."""
    d = tmp_path / 'rw'
    subprocess.run([REWRITE, BLOCK, str(d), *change], cwd=ROOT, check=True, capture_output=True)
    # definitions and parts are checked with the header; the LM model's id when it is loaded
    with pytest.raises(hnz.IncompatibleBlockError):
        hnz.verify_block(d, 'quick' if change[1].startswith('model/lm') else 'header')
    with pytest.raises(hnz.IncompatibleBlockError):
        hnz.open(d)


@pytest.mark.skipif(not os.path.exists(REWRITE), reason='no rewrite_block build')
def test_rewrite_without_changes_is_identical(tmp_path):
    d = tmp_path / 'same'
    snap = subprocess.run([REWRITE, BLOCK, str(d)], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    assert snap == hnz.inspect(BLOCK)['snapshot']
    with open(d / 'hn.hnzb', 'rb') as a, open(os.path.join(BLOCK, 'hn.hnzb'), 'rb') as b:
        assert a.read() == b.read()
