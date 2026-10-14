"""Pruebas de aceptación de la fase P1 (librería Python sobre el formato actual).

Necesitan el archivo de muestra (out/: 2010-06, 2015-06, 2024-01), los modelos (models/), los
Parquet originales (data/) y el binario hnz.exe. Desde la raíz del proyecto:
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


@pytest.mark.parametrize('q', QUERIES[:3])
def test_same_results_as_cli(hn, q):
    """La librería devuelve exactamente lo mismo que `hnz search` (misma consulta, mismos filtros)."""
    for kw, args in (({}, []), ({'mode': 'keyword'}, ['--mode', 'keyword']),
                     ({'since': '2015', 'until': '2015', 'types': 'comment'}, ['--from', '2015', '--to', '2015', '--type', 'comment'])):
        py = hn.search(q, k=10, raw=True, **kw)['results']
        c = cli('-k', '10', *args, q)
        assert [r['id'] for r in py] == [r['id'] for r in c], (q, kw)
        assert [r['text'] for r in py] == [r['text'] for r in c]


def test_listing_without_text(hn):
    r = hn.search(types='story', since='2024', k=5)
    assert len(r) == 5 and all(h.kind == 'story' for h in r)
    assert [h.time for h in r] == sorted((h.time for h in r), reverse=True)


def test_get_is_lossless(hn):
    """get() devuelve el texto, título, url y autor exactos del Parquet original (los 3 niveles)."""
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
    """8 hilos con 48 consultas mezcladas (híbridas, palabras, texto completo, get) = mismo resultado
    que de una en una."""
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
