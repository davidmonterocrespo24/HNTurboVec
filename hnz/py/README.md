# hnz — all of Hacker News, compressed, searchable from Python

`hnz` opens a compact, read-only block holding every Hacker News item (49.3 M stories, comments,
jobs and polls from 2006-10 to 2026-08), losslessly, in about 5.3 GB, and searches it the way a
search engine would: free text (semantic and keyword, fused), filters (dates, type, author), `k`
results, and the verbatim text of any item. Everything it holds in memory stays within a RAM
budget you choose.

- **Lossless:** every item's text comes back byte for byte (HN's original HTML).
- **Hybrid search:** a 256-bit semantic index (EmbeddingGemma 2) and a word index, then a reranking
  of the best candidates by their first 128 tokens (P@10 0.32 on the evaluation queries within
  their month; see Performance).
- **RAM budget:** `ram="2GB"` or `ram="10GB"`: models, indexes, decompression memory, caches and
  every query's working memory fit in it; what does not fit is read from disk.
- **Thread-safe:** one open archive serves concurrent queries (the work runs without the GIL);
  `asyncio` versions of every call.

## Requirements

- Windows 10 or 11, x86-64 with AVX2 (Intel Haswell / AMD Zen or later).
- Python 3.10 or later.
- The `openvino` package (installed with `hnz`): hnz uses its DLLs to embed queries on the CPU.
- A block: a folder `hn-YYYY-MM.hnz\` with `hn.hnzb`, `embed.xml` and `embed.bin`.

## Install

```bash
pip install hnz-0.1.0-cp310-abi3-win_amd64.whl     # pulls openvino
```

## Quick start

```python
import hnz

with hnz.open(r"D:\hn\hn-2026-08.hnz", ram="2GB") as hn:
    r = hn.search("how do I deal with burnout as a software engineer", k=10,
                  since="2015", until="2018-06", types="comment")
    for h in r:
        print(h.rank, h.date, h.by, h.story_title, h.text[:120])
    print(r.timings_ms, r.rerank)

    item = hn.get(9700000)            # the whole item, verbatim (item.html is HN's HTML)
    print(item.title, item.text[:200])

    jobs = hn.search(types="job", since="2024", k=20)   # no query: the most recent items
```

## API

### `hnz.open(path, *, ram="2GB", preload=False, max_concurrent=8, timeout=30.0, verify="quick", cache_dir=None, cache_dir_size="512MB", models=None, ov_libs=None) -> Archive`

- `ram`: budget for everything hnz holds ("2GB", "10GB", bytes, or `None` for no limit). Below the
  minimum (`memory()["plan"]["floor"]`, about 0.9 GB) it raises `BudgetError`.
- `preload`: load the embedding model at open (otherwise at the first semantic search, ~3 s).
- `max_concurrent`, `timeout`: at most that many searches/gets at once; the rest wait up to
  `timeout` seconds, then `BusyError`.
- `verify`: what is checked at open: "header", "quick" (default: the always-loaded parts and 4
  known records decoded) or "full" (every part's hash).
- `cache_dir`: an optional disk cache of decompressed records (off by default).

### `Archive.search(query="", *, k=10, since=None, until=None, types=None, author=None, mode="hybrid", text="snippet", include_dead=False, rerank=None, deadline_ms=None, rerank_depth=None, w_bm25=None, c_bm25=None, raw=False) -> SearchResult`

- `since`, `until`: "2015", "2015-06", "2015-06-01" (both inclusive) or a `datetime` (an exact UTC
  instant).
- `types`: "story", "comment", "job", "poll", "pollopt" (one or a list).
- `mode`: "hybrid" (default), "vector" (semantic only) or "keyword" (words only; frequent words
  are found among the semantic candidates, with a `warning`).
- `text`: "snippet" (a snippet, or the whole item when it is short) or "full".
- `rerank`: `None` = rerank whenever the block allows it; `False` = the fast first stage only.
- `deadline_ms` (opt-in): answer within this many milliseconds; a reranking not done by then gives
  way to the first stage's order (`result.rerank["degraded"]` says so).
- `rerank_depth`: semantic candidates to rerank (default 80; 60 is faster, more reaches deeper).
- `raw=True`: the response as a dict (the same JSON as the HTTP server).

`SearchResult`: iterate it for `Hit`s; also `timings_ms`, `rerank` (what the reranking did) and
`warning`. A `Hit` has `rank, id, kind, by, date, time, points, title, url, text, html, partial,
parent_id, story_id, story_title, score, vscore, bm25, link`.

### Other calls

| Call | Returns |
|---|---|
| `get(id)` / `get_many(ids)` | the whole item (`Item`: `id, kind, by, time, date, title, url, html, text, …`) or `None` |
| `embed(text)` | the query's 768-dim vector |
| `asearch(…)`, `aget(id)`, `aembed(text)` | the same, for `asyncio` |
| `stats()` | documents, dates, block snapshot |
| `memory()` | the RAM ledger (limit, used, peak, per class), the load plan, caches, the process |
| `serve(host=, port=, threads=, stop=)` | the HTTP server on this archive (see below) |
| `close()` | frees everything (also on leaving a `with` block) |
| `hnz.verify_block(path, level)` | checks a block without opening it ("header", "quick", "full") |

Errors: `ValueError` (a bad query), `BusyError`, `BudgetError`, `CorruptBlockError`,
`IncompatibleBlockError` (both are `BlockError`).

## HTTP server

The `hnz` command, installed with the package (also `python -m hnz`), serves an archive:

```bash
hnz serve --root D:\hn\hn-2026-08.hnz --ram 2GB --threads 8 --port 8088
```
`GET /search?q=…&k=10&from=2015&to=2018&type=comment&by=pg&mode=hybrid&deadline_ms=3000` returns
the same JSON as `search(…, raw=True)` (also `rerank=0`, `depth=60`, `full=1`, `dead=1`, `w_bm25`,
`c_bm25`); `GET /item/<id>`, `/stats`, `/memory`; `/` is a search page. Bad queries answer 400,
`BusyError` 503. It listens on this machine only unless `--host 0.0.0.0`; Ctrl+C stops it.

It is a small server meant for this machine or a private network: GET and HEAD only, one request
per connection, no TLS and no authentication, request heads up to 16 KiB. To expose it on the
internet, put it behind a reverse proxy (nginx, Caddy) that adds TLS, authentication and rate
limits.

`Archive.serve(host="127.0.0.1", port=8088, threads=8, stop=None)` runs the same server on an open
archive, until Ctrl+C or until `stop` (a `threading.Event`) is set.

In a Python web app (FastAPI), share one archive:
```python
import hnz
from fastapi import FastAPI

hn = hnz.open(r"D:\hn\hn-2026-08.hnz", ram="4GB", preload=True)
app = FastAPI()

@app.get("/search")
async def search(q: str, k: int = 10):
    return await hn.asearch(q, k=k, raw=True)
```

## Performance (16-thread laptop CPU, the block of all of HN, `hnz serve --ram 2GB`)

| | |
|---|---|
| Open | ~4 s with the embedder loaded (`preload=True`) |
| Cold search, reranked | ~2-4 s (decoding the candidates' first tokens is most of it) |
| Repeated search | ~1.5-2 s |
| First stage only (`rerank=False`) | ~0.3-0.5 s |
| `get()` | ~80-90 ms |
| Memory | the ledger stays within `ram`; at 2 GB the process uses ~1.4 GB private |

Quality on 300 queries per month whose relevant items are one story's thread: P@10 0.32 within the
query's month (keyword only: 0.26). Across all 20 years it is ~0.05, because threads on the same
subject from other years count as misses in that eval.

See [doc/09-rerank.md](../../doc/09-rerank.md) (search), [doc/07-ram.md](../../doc/07-ram.md) (RAM)
and [doc/08-block.md](../../doc/08-block.md) (the block format).
