"""hnz: todo Hacker News comprimido sin pérdida, con búsqueda híbrida y recuperación literal.

    import hnz
    with hnz.open(r"C:\\ruta\\out", ram="4GB") as hn:
        r = hn.search("how do I deal with burnout", k=10, since="2015", until="2018-06", types="comment")
        for h in r:
            print(h.rank, h.date, h.by, h.story_title, h.text[:120])
        item = hn.get(9700000)          # texto literal (item.html = HTML original de HN)

`Archive` es seguro entre hilos: un backend (p.ej. FastAPI con un worker) puede atender varias
consultas a la vez sobre el mismo archivo abierto; el trabajo pesado se hace sin el GIL.
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
MAX_K: int = _hnz.MAX_K
KINDS = ('story', 'comment', 'job', 'poll', 'pollopt')
DateLike = Union[str, int, _dt.date, _dt.datetime]


@dataclass(frozen=True, slots=True)
class Hit:
    """Un resultado de búsqueda. `text` es texto plano (fragmento si `partial`); `html`, el original."""
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
    """Un ítem completo y literal (`html` es el HTML original de HN; `text`, el mismo en texto plano)."""
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


def _item(d: dict) -> Item:
    t = int(d.get('time', 0))
    return Item(
        id=d['id'], kind=_KIND.get(d.get('typ', 0), str(d.get('typ'))), by=d.get('by', ''), time=t,
        date=_dt.datetime.fromtimestamp(t, _dt.timezone.utc).strftime('%Y-%m-%d %H:%M'),
        title=d.get('title', ''), url=d.get('url', ''), html=d.get('text', ''), text=d.get('text_plain', ''),
        points=d.get('score', 0), descendants=d.get('descendants', 0), parent_id=d.get('parent') or None,
        poll=d.get('poll') or None, parts=list(d.get('parts', [])), dead=bool(d.get('dead')), deleted=bool(d.get('deleted')))


def _size(ram: Union[int, str, None]) -> Optional[int]:
    """'2GB' -> 2e9, '2GiB' -> 2**31, '512MB', o un número de bytes."""
    if ram is None or isinstance(ram, int):
        return ram
    m = re.fullmatch(r'\s*([\d.]+)\s*([KMGT]?)(i?)B?\s*', str(ram), re.I)
    if not m:
        raise ValueError(f'ram inválida: {ram!r} (p.ej. "2GB", "512MB")')
    exp = 'KMGT'.find(m.group(2).upper()) + 1 if m.group(2) else 0
    return int(float(m.group(1)) * (1024 if m.group(3) else 1000) ** exp)


def _date(d: Optional[DateLike]) -> Optional[str]:
    if d is None:
        return None
    if isinstance(d, (_dt.date, _dt.datetime)):
        return d.strftime('%Y-%m-%d')
    return str(d)


def _ov_libs() -> Optional[str]:
    """Carpeta con openvino_c.dll: HNZ_OV_LIBS o la del paquete openvino instalado (sin importarlo)."""
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
    """Archivo hnz abierto. Seguro entre hilos; ciérrelo con `close()` o úselo con `with`."""

    def __init__(self, inner, path: str, ram: Optional[int]):
        self._a = inner
        self.path = path
        self.ram = ram

    def search(self, query: str = '', *, k: int = 10, since: Optional[DateLike] = None,
               until: Optional[DateLike] = None, types: Union[str, Iterable[str], None] = None,
               author: Optional[str] = None, mode: Literal['hybrid', 'vector', 'keyword'] = 'hybrid',
               text: Literal['snippet', 'full'] = 'snippet', include_dead: bool = False, raw: bool = False,
               w_bm25: float = 0.15, c_bm25: float = 20.0) -> Union[SearchResult, dict]:
        """Busca en todo HN (o en el rango [since, until], ambos inclusive: "2015", "2015-06", fechas).

        - Sin `query`: lo más reciente que cumple los filtros.
        - `types`: "story", "comment", "job", "poll", "pollopt" (uno o varios).
        - `text="full"`: texto completo también de los ítems del nivel comprimido con el LM (más lento).
        - `raw=True`: devuelve el dict JSON tal cual (el mismo formato que el servidor HTTP).
        """
        if isinstance(types, str):
            types = [types]
        if text not in ('snippet', 'full'):
            raise ValueError('text debe ser "snippet" o "full"')
        req = {'query': query or '', 'k': int(k), 'since': _date(since), 'until': _date(until),
               'types': list(types or []), 'author': author, 'mode': mode, 'full': text == 'full',
               'include_dead': bool(include_dead), 'w_bm25': float(w_bm25), 'c_bm25': float(c_bm25)}
        d = json.loads(self._live().search_json(json.dumps(req)))
        if raw:
            return d
        return SearchResult(query=d['query'], hits=[Hit(**h) for h in d['results']], timings_ms=d['ms'])

    def get(self, id: int, *, raw: bool = False) -> Union[Item, dict, None]:
        """Un ítem por id de HN, literal y completo; None si no está en el archivo."""
        s = self._live().get_json(int(id))
        if s is None:
            return None
        d = json.loads(s)
        return d if raw else _item(d)

    def get_many(self, ids: Sequence[int], *, raw: bool = False) -> list:
        return [self.get(i, raw=raw) for i in ids]

    def embed(self, text: str) -> list[float]:
        """Vector de 768 dimensiones (EmbeddingGemma 2) de un texto de consulta."""
        return self._live().embed(text)

    def stats(self) -> dict:
        return json.loads(self._live().stats_json())

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
         preload: bool = False) -> Archive:
    """Abre un archivo hnz (la carpeta con los segmentos).

    - `ram`: presupuesto de RAM ("2GB", "10GB"…). En esta versión se guarda e informa; el control
      estricto de la memoria llega en una fase posterior.
    - `models`: carpeta de modelos (por defecto `<path>/models` o `<path>/../models`).
    - `ov_libs`: carpeta con openvino_c.dll (por defecto la del paquete `openvino` instalado).
    - `preload`: cargar ya el modelo de embeddings (si no, en la primera búsqueda semántica).
    """
    path = os.fspath(path)
    opts = {'models': os.fspath(models) if models else None,
            'ov_libs': os.fspath(ov_libs) if ov_libs else _ov_libs(),
            'preload': bool(preload), 'ram': _size(ram)}
    return Archive(_hnz.open(path, json.dumps(opts)), path, opts['ram'])


__all__ = ['open', 'Archive', 'SearchResult', 'Hit', 'Item', 'KINDS', 'MAX_K', '__version__']
