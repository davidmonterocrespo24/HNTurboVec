"""The `hnz` command installed with the package (also `python -m hnz`): the HTTP server.

    hnz serve --root D:\\hn\\hn-2026-08.hnz --ram 2GB --threads 8 --port 8088
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

import hnz


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog='hnz', description='hnz: all of Hacker News, compressed and searchable.')
    ap.add_argument('--version', action='version', version=f'hnz {hnz.__version__}')
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('serve', help='HTTP server: /search, /item/<id>, /stats, /memory and a search page at /',
                       description='Serves an archive over HTTP until Ctrl+C (see Archive.serve).')
    s.add_argument('--root', required=True, help='the block (a folder hn-YYYY-MM.hnz) or a folder of segments')
    s.add_argument('--ram', default='2GB', help='RAM budget for everything hnz holds (default 2GB; "none" = no limit)')
    s.add_argument('--host', default='127.0.0.1',
                   help='127.0.0.1 = this machine only (default); 0.0.0.0 = every interface (a private network; '
                        'on the internet, behind a reverse proxy)')
    s.add_argument('--port', type=int, default=8088)
    s.add_argument('--threads', type=int, default=8, help='requests served at the same time, 1-256 (default 8)')
    s.add_argument('--timeout', type=float, default=30.0,
                   help='seconds a request may wait for its turn before answering 503 (default 30)')
    a = ap.parse_args(argv)
    # checked before the archive's open, which takes seconds
    if not 1 <= a.port <= 65535:
        ap.error(f'invalid --port: {a.port} (1-65535)')
    if not 1 <= a.threads <= 256:
        ap.error(f'invalid --threads: {a.threads} (1-256)')
    ram = None if a.ram.lower() == 'none' else a.ram
    with hnz.open(a.root, ram=ram, preload=True, max_concurrent=a.threads, timeout=a.timeout) as hn:
        print(f'hnz: {hn.stats()["docs"]} docs - http://{a.host}:{a.port}/ (Ctrl+C to stop)', file=sys.stderr, flush=True)
        try:
            hn.serve(host=a.host, port=a.port, threads=a.threads)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
