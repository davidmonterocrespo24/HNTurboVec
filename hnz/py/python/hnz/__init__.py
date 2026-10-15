"""hnz: all of Hacker News compressed losslessly, with hybrid search and verbatim retrieval.

    import hnz
    with hnz.open(r"C:\\path\\out", ram="4GB") as hn:
        r = hn.search("how do I deal with burnout", k=10, since="2015", until="2018-06", types="comment")
        for h in r:
            print(h.rank, h.date, h.by, h.story_title, h.text[:120])
        item = hn.get(9700000)          # verbatim text (item.html = HN's original HTML)
        print(hn.memory()["budget"])    # RAM used by hnz, per class, against the `ram` limit

`Archive` is thread-safe: a backend (e.g. FastAPI with one worker) can serve several queries at
once on the same open archive; the heavy work runs without the GIL. Everything hnz holds in RAM
(models, embedder, loaded indexes, decompression memory, caches, per-query memory) stays within
the `ram` budget given to `open()`; what does not fit is read from disk on demand.
"""
from __future__ import annotations

import datetime as _dt
import importlib.util
import json
import os
import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Literal, Optional, Sequence, Union

from . import _hnz

__version__: str = _hnz.__version__
terms = _hnz.terms                  # the engine's BM25 tokenizer: terms("Rust's async") -> ['rusts', 'async']
index_text = _hnz.index_text        # text indexed for an item: index_text(title, url, html)
MAX_K: int = _hnz.MAX_K
KINDS = ('story', 'comment', 'job', 'poll', 'pollopt')
DateLike = Union[str, int, _dt.date, _dt.datetime]

BudgetError = _hnz.BudgetError      # MemoryError: `ram` below the minimum (the message gives it)
BusyError = _hnz.BusyError          # TimeoutError: no turn within `timeout`; retry or raise the limits
BlockError = _hnz.BlockError        # a compact block cannot be used: one of the two below
CorruptBlockError = _hnz.CorruptBlockError            # damaged bytes (hash, checksum or length)
IncompatibleBlockError = _hnz.IncompatibleBlockError  # intact, but this hnz cannot read it


@dataclass(frozen=True, slots=True)
class Hit:
    """A search result. `text` is plain text (a snippet if `partial`); `html`, the original."""
    rank: int
    id: int
    kind: str
    by: str
    date: str
    time: int
    points: int
    title: str
    url: str
    text: str
    html: str
    partial: bool
    tier: str
    parent_id: Optional[int]
    story_id: Optional[int]
    story_title: Optional[str]
    score: float
    vscore: float
    bm25: float
    link: str


@dataclass(frozen=True, slots=True)
class SearchResult:
    query: str
    hits: list[Hit]
    timings_ms: dict[str, float]

    def __iter__(self) -> Iterator[Hit]:
        return iter(self.hits)

    def __len__(self) -> int:
        return len(self.hits)

    def __getitem__(self, i: int) -> Hit:
        return self.hits[i]


@dataclass(frozen=True, slots=True)
class Item:
    """A full, verbatim item (`html` is HN's original HTML; `text`, the same as plain text)."""
    id: int
    kind: str
    by: str
    time: int
    date: str
    title: str
    url: str
    html: str
    text: str
    points: int
    descendants: int
    parent_id: Optional[int]
    poll: Optional[int]
    parts: list[int]
    dead: bool
    deleted: bool


_KIND = {1: 'story', 2: 'comment', 3: 'poll', 4: 'pollopt', 5: 'job'}


def _date(t: int) -> str:
    try:
        return _dt.datetime.fromtimestamp(t, _dt.timezone.utc).strftime('%Y-%m-%d %H:%M')
    except (OverflowError, OSError, ValueError):     # a time outside Python's dates: keep the item
        return ''


def _item(d: dict) -> Item:
    t = int(d.get('time', 0))
    return Item(
        id=d['id'], kind=_KIND.get(d.get('typ', 0), str(d.get('typ'))), by=d.get('by', ''), time=t,
        date=_date(t),
        title=d.get('title', ''), url=d.get('url', ''), html=d.get('text', ''), text=d.get('text_plain', ''),
        points=d.get('score', 0), descendants=d.get('descendants', 0), parent_id=d.get('parent') or None,
        poll=d.get('poll') or None, parts=list(d.get('parts', [])), dead=bool(d.get('dead')), deleted=bool(d.get('deleted')))


def _size(ram: Union[int, str, None]) -> Optional[int]:
    """'2GB' -> 2e9, '2GiB' -> 2**31, '512MB', or a number of bytes."""
    if ram is None or (isinstance(ram, int) and not isinstance(ram, bool)):
        if isinstance(ram, int) and ram < 0:
            raise ValueError(f'tamaño inválido: {ram!r}')
        return ram
    m = re.fullmatch(r'\s*([\d.]+)\s*([KMGT]?)(i?)B?\s*', str(ram), re.I)
    if not m:
        raise ValueError(f'tamaño inválido: {ram!r} (p.ej. "2GB", "512MB")')
    exp = 'KMGT'.find(m.group(2).upper()) + 1 if m.group(2) else 0
    return int(float(m.group(1)) * (1024 if m.group(3) else 1000) ** exp)


def _bound(d: Optional[DateLike], name: str) -> tuple[Optional[str], Optional[int]]:
    """Date bound -> (text for the engine, or exact instant in epoch seconds UTC).

    - str: "2015", "2015-06", "2015-06-01" (validated by the engine); date: that whole day.
    - datetime: that exact instant (without a time zone it is taken as UTC, HN's zone).
    - int: a year (1970-2100); for a specific instant use datetime.
    """
    if d is None:
        return None, None
    if isinstance(d, _dt.datetime):
        if d.tzinfo is None:
            d = d.replace(tzinfo=_dt.timezone.utc)
        return None, int(d.timestamp() // 1)
    if isinstance(d, _dt.date):
        return d.strftime('%Y-%m-%d'), None
    if isinstance(d, bool):
        raise ValueError(f'{name} inválido: {d!r}')
    if isinstance(d, int):
        if not 1970 <= d <= 2100:
            raise ValueError(f'{name}={d}: un entero se interpreta como año (1970-2100); para un instante use datetime')
        return str(d), None
    if isinstance(d, str):
        return d, None
    raise ValueError(f'{name} inválido: {d!r} (str "2015-06", int año, date o datetime)')


def _ov_libs() -> Optional[str]:
    """Folder with openvino_c.dll: HNZ_OV_LIBS or the one of the installed openvino package (not imported)."""
    env = os.environ.get('HNZ_OV_LIBS')
    if env:
        return env
    spec = importlib.util.find_spec('openvino')
    if spec and spec.origin:
        libs = os.path.join(os.path.dirname(spec.origin), 'libs')
        if os.path.exists(os.path.join(libs, 'openvino_c.dll')):
            return libs
    return None


class Archive:
    """An open hnz archive. Thread-safe; close it with `close()` or use it with `with`."""

    def __init__(self, inner, path: str, ram: Optional[int]):
        self._a = inner
        self.path = path
        self.ram = ram

    def search(self, query: str = '', *, k: int = 10, since: Optional[DateLike] = None,
               until: Optional[DateLike] = None, types: Union[str, Iterable[str], None] = None,
               author: Optional[str] = None, mode: Literal['hybrid', 'vector', 'keyword'] = 'hybrid',
               text: Literal['snippet', 'full'] = 'snippet', include_dead: bool = False, raw: bool = False,
               w_bm25: float = 0.15, c_bm25: float = 20.0, rerank: Optional[bool] = None) -> Union[SearchResult, dict]:
        """Searches all of HN (or the range [since, until], both inclusive: "2015", "2015-06", a date,
        or a datetime for an exact instant in UTC).

        - Without `query`: the most recent items that pass the filters.
        - `types`: "story", "comment", "job", "poll", "pollopt" (one or several).
        - `text="full"`: full text also for items of the LM-compressed tier (slower).
        - `raw=True`: returns the JSON dict as is (the same format as the HTTP server).
        - `rerank` (compact blocks with an embedder of dynamic shapes): rerank the candidates by
          their prefix (P4); None = whenever possible.

        Raises `BusyError` if it found no turn within the archive's `timeout`.
        """
        if isinstance(types, str):
            types = [types]
        if text not in ('snippet', 'full'):
            raise ValueError('text debe ser "snippet" o "full"')
        since_s, since_ts = _bound(since, 'since')
        until_s, until_ts = _bound(until, 'until')
        req = {'query': query or '', 'k': int(k), 'since': since_s, 'until': until_s,
               'since_ts': since_ts, 'until_ts': until_ts,
               'types': list(types or []), 'author': author, 'mode': mode, 'full': text == 'full',
               'include_dead': bool(include_dead), 'w_bm25': float(w_bm25), 'c_bm25': float(c_bm25),
               'rerank': rerank}
        d = json.loads(self._live().search_json(json.dumps(req)))
        if raw:
            return d
        return SearchResult(query=d['query'], hits=[Hit(**h) for h in d['results']], timings_ms=d['ms'])

    def get(self, id: int, *, raw: bool = False) -> Union[Item, dict, None]:
        """An item by HN id, verbatim and complete; None if it is not in the archive."""
        s = self._live().get_json(int(id))
        if s is None:
            return None
        d = json.loads(s)
        return d if raw else _item(d)

    def get_many(self, ids: Sequence[int], *, raw: bool = False) -> list:
        return [self.get(i, raw=raw) for i in ids]

    def embed(self, text: str) -> list[float]:
        """768-dimensional vector (EmbeddingGemma 2) of a query text."""
        return self._live().embed(text)

    def stats(self) -> dict:
        """Archive summary (segments, dates, documents) and `memory()`."""
        return json.loads(self._live().stats_json())

    def memory(self) -> dict:
        """RAM breakdown: `budget` (hnz's own ledger: limit, used, peak, and per class: models, index,
        cache, working), `plan` (what was loaded into RAM and the size of each pool), `engine` (loaded
        parts, caches and their hit rates), `embedder`, `admission` and `process` (as the OS sees it)."""
        return json.loads(self._live().memory_json())

    def close(self) -> None:
        self._a = None

    @property
    def closed(self) -> bool:
        return self._a is None

    def _live(self):
        if self._a is None:
            raise RuntimeError('el archivo está cerrado')
        return self._a

    def __enter__(self) -> 'Archive':
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return f'<hnz.Archive {self.path!r}{" (cerrado)" if self.closed else ""}>'


def open(path: Union[str, os.PathLike], *, ram: Union[int, str, None] = '2GB',
         models: Union[str, os.PathLike, None] = None, ov_libs: Union[str, os.PathLike, None] = None,
         preload: bool = False, max_concurrent: int = 8, timeout: float = 30.0,
         cache_dir: Union[str, os.PathLike, None] = None, cache_dir_size: Union[int, str] = '512MB',
         verify: Literal['header', 'quick', 'full'] = 'quick') -> Archive:
    """Opens an hnz archive: a compact block (a folder with `hn.hnzb`, or the file itself) or a
    folder of segments.

    - `ram`: RAM budget for everything hnz holds ("2GB", "10GB", bytes, or None for no limit). At
      open, a planner loads into RAM what fits by benefit (vectors, metadata, word index, then
      caches and compressed text, newest segments first) and reads the rest from disk on demand.
      Raises `BudgetError` if it is below the minimum (models + fixed structures).
    - `models`: models folder (default `<path>/models` or `<path>/../models`).
    - `ov_libs`: folder with openvino_c.dll (default: the one of the installed `openvino` package).
    - `preload`: load the embedding model now (otherwise at the first semantic search). Until it is
      loaded, its load peak stays reserved in the budget.
    - `max_concurrent`, `timeout`: at most `max_concurrent` searches/gets run at once; the rest wait
      up to `timeout` seconds and then raise `BusyError`.
    - `cache_dir`: folder for an optional persistent cache of decompressed records (survives
      restarts; up to `cache_dir_size` on disk). Off by default.
    - `verify` (blocks): "header" checks the structure; "quick" (default, ~0.1-0.3 s) also the
      always-loaded parts and decodes 4 short records whose text is known; "full" hashes every
      part and the embedder files too. Errors: `CorruptBlockError`, `IncompatibleBlockError`.
    """
    path = os.fspath(path)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 <= timeout < 1e9:
        raise ValueError(f'timeout inválido: {timeout!r} (segundos, >= 0)')
    if isinstance(max_concurrent, bool) or not isinstance(max_concurrent, int) or max_concurrent < 1:
        raise ValueError(f'max_concurrent inválido: {max_concurrent!r} (entero >= 1)')
    opts = {'models': os.fspath(models) if models else None,
            'ov_libs': os.fspath(ov_libs) if ov_libs else _ov_libs(),
            'preload': bool(preload), 'ram': _size(ram), 'max_concurrent': int(max_concurrent),
            'timeout': float(timeout), 'cache_dir': os.fspath(cache_dir) if cache_dir else None,
            'cache_dir_bytes': _size(cache_dir_size), 'verify': verify}
    return Archive(_hnz.open(path, json.dumps(opts)), path, opts['ram'])


def verify_block(path: Union[str, os.PathLike], level: Literal['header', 'quick', 'full'] = 'full') -> dict:
    """Checks a compact block without opening it for queries: "header" (structure and versions),
    "quick" (what `open` checks by default) or "full" (every part hash, the padding, the embedder
    files and every canary: a few seconds per GB). Returns a report; raises `CorruptBlockError` or
    `IncompatibleBlockError` (the deeper check against the sources is `hnz verify-block --level deep`).
    It opens the block on its own, outside any open archive's `ram` budget (quick and full load the
    always-loaded parts and the LM model)."""
    return json.loads(_hnz.verify_block(os.fspath(path), level))


def inspect(path: Union[str, os.PathLike], *, project: Optional[int] = None,
            lex: Union[str, os.PathLike, None] = None, id: Optional[int] = None) -> dict:
    """Describes a compact block: its parts (offset, bytes, bytes per document), TOC and source
    snapshot. `project`: sizes projected to that many documents (`lex`: the exact word-index sizes
    from `hnz lexicon`); `id`: where that item's bytes are (this opens the block on its own,
    outside any `ram` budget)."""
    return json.loads(_hnz.inspect(os.fspath(path), project, os.fspath(lex) if lex else None, id))


__all__ = ['open', 'Archive', 'SearchResult', 'Hit', 'Item', 'BudgetError', 'BusyError', 'BlockError',
           'CorruptBlockError', 'IncompatibleBlockError', 'KINDS', 'MAX_K', 'terms', 'index_text',
           'verify_block', 'inspect', '__version__']
