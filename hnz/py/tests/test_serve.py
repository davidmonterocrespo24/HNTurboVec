"""P6 tests: the HTTP server (`Archive.serve` and the `hnz serve` command) on the sample block.

They need work/p3/hn-sample-v3.hnz (see test_rerank.py) and the installed `hnz` package. From the
project root:
    .venv\\Scripts\\python -m pytest hnz/py/tests/test_serve.py -q
"""
import _thread
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import pytest

import hnz

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
BLOCK = os.path.join(ROOT, 'work', 'p3', 'hn-sample-v3.hnz')
Q = 'startup funding advice'

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(BLOCK, 'hn.hnzb')), reason='no v3 sample block (hnz pack)')


@pytest.fixture(scope='module')
def hn():
    a = hnz.open(BLOCK, ram='2GB', preload=True)
    yield a
    a.close()


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def fetch(port, path, timeout=120):
    """(status, body) of a GET to the local server."""
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}', timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def wait_up(port, limit=120, proc=None):
    """/stats once the server answers (it may still be opening the archive); stops early if `proc`
    (a server process) has exited."""
    t = time.time()
    while True:
        try:
            return fetch(port, '/stats', timeout=10)
        except OSError:
            if time.time() - t > limit or (proc is not None and proc.poll() is not None):
                raise
            time.sleep(0.2)


def raw(port, data, timeout=60):
    """(status, headers, body) of raw bytes sent to the server (it closes every connection)."""
    with socket.create_connection(('127.0.0.1', port), timeout=timeout) as c:
        c.sendall(data)
        out = b''
        while chunk := c.recv(65536):
            out += chunk
    head, _, body = out.partition(b'\r\n\r\n')
    lines = head.decode('latin-1').split('\r\n')
    return int(lines[0].split()[1]), lines[1:], body


def test_serve_answers_like_the_library_and_stops(hn):
    port, stop, err = free_port(), threading.Event(), []

    def run():
        try:
            hn.serve(port=port, threads=2, stop=stop)
        except BaseException as e:
            err.append(e)

    th = threading.Thread(target=run, daemon=True)
    th.start()
    try:
        st, body = wait_up(port)
        assert st == 200 and json.loads(body)['docs'] == hn.stats()['docs']
        st, body = fetch(port, '/search?q=' + urllib.parse.quote_plus(Q) + '&k=5')
        assert st == 200, body
        d = json.loads(body)
        assert [h['id'] for h in d['results']] == [h['id'] for h in hn.search(Q, k=5, raw=True)['results']]
        assert d.get('rerank') is not None
        top = d['results'][0]['id']
        st, body = fetch(port, f'/item/{top}')
        assert st == 200 and json.loads(body)['id'] == top
        st, body = fetch(port, '/search?q=x&type=story&from=2006&k=3&rerank=0&deadline_ms=5000&depth=60')
        assert st == 200 and all(h['kind'] == 'story' for h in json.loads(body)['results'])
        # errors: 400 for a bad query or id, 404 for what is not there
        for path, want in (('/search?q=x&k=0', 400), ('/search?q=x&k=abc', 400), ('/search?q=x&mode=bogus', 400),
                           ('/item/abc', 400), ('/item/4000000000', 404), ('/nope', 404)):
            st, body = fetch(port, path)
            assert st == want, (path, st, body)
            if st == 400:
                assert 'error' in json.loads(body)
        st, body = fetch(port, '/')
        assert st == 200 and b'<html' in body.lower()
        st, body = fetch(port, '/memory')
        assert st == 200 and 'budget' in json.loads(body)
        # raw HTTP: only GET/HEAD, no bodies, a bounded head; HEAD answers without a body
        st, hdrs, body = raw(port, b'HEAD /stats HTTP/1.1\r\nHost: x\r\n\r\n')
        assert st == 200 and body == b'' and any(h.lower().startswith('content-length:') for h in hdrs)
        st, _, body = raw(port, b'POST /search HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\n{}')
        assert st == 405, body
        st, _, body = raw(port, b'GET /stats HTTP/1.1\r\nHost: x\r\nContent-Length: 99999999999999999999\r\n\r\n')
        assert st == 400, body
        st, _, body = raw(port, b'GET /stats HTTP/1.1\r\nHost: x\r\nX-Big: ' + b'a' * 20000 + b'\r\n\r\n')
        assert st == 431, body
        st, _, body = raw(port, b'garbage\r\n\r\n')
        assert st == 400, body
        st, body = fetch(port, '/stats')   # still serving after all that
        assert st == 200
    finally:
        stop.set()
        th.join(timeout=60)
    assert not th.is_alive() and not err, err
    with pytest.raises(OSError):   # stopped: nothing listens there any more
        fetch(port, '/stats', timeout=10)


def test_ctrl_c_stops_the_server(hn):
    """Ctrl+C (SIGINT in the main thread) ends `serve` with KeyboardInterrupt."""
    port, up = free_port(), []

    def interrupt():
        try:
            up.append(wait_up(port)[0])
        finally:   # always, or a server that never came up would block the test run
            _thread.interrupt_main()

    threading.Thread(target=interrupt, daemon=True).start()
    t = time.time()
    with pytest.raises(KeyboardInterrupt):
        hn.serve(port=port, threads=2)
    assert up == [200] and time.time() - t < 120
    with pytest.raises(OSError):
        fetch(port, '/stats', timeout=10)


def test_serve_bad_arguments(hn):
    with pytest.raises(ValueError):
        hn.serve(port=0)
    with pytest.raises(ValueError):
        hn.serve(port=free_port(), threads=0)
    with socket.socket() as s:   # a port already taken
        s.bind(('127.0.0.1', 0))
        s.listen()
        stop = threading.Event()
        stop.set()   # if the bind wrongly succeeded, serve returns at once instead of hanging
        with pytest.raises(RuntimeError):
            hn.serve(port=s.getsockname()[1], stop=stop)


def test_the_hnz_command():
    """`python -m hnz` (the `hnz` command): --version, and `serve` answering searches."""
    out = subprocess.run([sys.executable, '-m', 'hnz', '--version'], capture_output=True, text=True, check=True).stdout
    assert out.strip() == f'hnz {hnz.__version__}'
    port = free_port()
    p = subprocess.Popen([sys.executable, '-m', 'hnz', 'serve', '--root', BLOCK, '--port', str(port), '--threads', '2'],
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    errs = b''
    try:
        st, _ = wait_up(port, proc=p)
        assert st == 200
        st, body = fetch(port, '/search?q=' + urllib.parse.quote_plus(Q) + '&k=3')
        assert st == 200 and len(json.loads(body)['results']) == 3
    except Exception as e:
        p.kill()
        raise AssertionError(f'{e}; server stderr: {p.communicate(timeout=60)[1]!r}') from e
    finally:
        if p.poll() is None:
            p.terminate()
        if p.stderr and not p.stderr.closed:
            errs = p.communicate(timeout=60)[1]
    assert b'docs - http://127.0.0.1:' in errs, errs
