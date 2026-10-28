# Testing the Python library (`import hnz`)

How to check, and approve, the `hnz` package. There are three levels.

| Level | What it proves | Data it needs | Time |
|---|---|---|---|
| 1. Acceptance on the full block | the installed wheel works on all of HN, through its public API only | the block `hn-2026-08.hnz` | ~30 s |
| 2. Automated suite (pytest) | each feature, edge cases, damage detection, memory limits | the sample blocks and archive (3 months) | ~19 min |
| 3. Block and quality checks | the block is lossless; search quality | the block + the Parquet sources | 5 s to ~10 h |

All commands run from the project root (`C:\Desarrollo\Odoo\AI\HNTurboVec`) on Windows.

## Setup

```powershell
# a clean environment, as a user would have (Python 3.10 or later)
py -3.12 -m venv C:\tmp\hnz-test
C:\tmp\hnz-test\Scripts\python -m pip install work\wheels\hnz-0.1.0-cp310-abi3-win_amd64.whl
```
The wheel pulls `openvino`. If `pip` hangs while downloading, `uv pip install --python
C:\tmp\hnz-test\Scripts\python.exe <wheel>` does the same.

**`AttributeError: module 'hnz' has no attribute 'open'`** means that Python did not find the
package and imported the project's `hnz\` folder (the Rust sources) instead: it happens with a
Python that does not have `hnz` installed (e.g. the system `python`) run from the project root.
Use the environment where the wheel is installed (`C:\tmp\hnz-test\Scripts\python` or the
project's `.venv\Scripts\python`). If the console shows `�` for dashes or quotes, set
`$env:PYTHONIOENCODING = "utf-8"` first.

For level 2, use the project's `.venv` instead: it has `pytest`, `pyarrow` and the package in
development mode (`hnz/py/python`, with the extension `_hnz.pyd` copied there by the build). To
rebuild the wheel and that extension: `python scripts/p6_run.py --only wheel` (delete
`work/p6/state/wheel.json` first if it already ran).

## 1. Acceptance on the full block

```powershell
C:\tmp\hnz-test\Scripts\python scripts\accept_full.py --root work\p6\hn-2026-08.hnz --ram 2GB
```
Options: `--ram 4GB` (another budget), `--verify full` (hash the whole block before opening it).

Each line prints `PASS` or `FAIL` and the exit code is the number of failures. Expected: 24 checks,
`ALL PASSED`.

| Check | Expected |
|---|---|
| `verify_block`, `open(ram=, preload=True)` | 49,324,111 docs; opening takes ~4 s |
| hybrid reranked search, the same search again, `rerank=False` | 10 results; ~2-4 s cold, ~2 s repeated, ~0.5 s for the first stage |
| filters: keyword + `since` + `types`, a one-month window, `author`, no query | every result matches the filters |
| `deadline_ms=1` | still 10 results, `rerank["degraded"]` set |
| `get(1)`, `get()` of a result, a missing id, `get_many` | item 1 is pg's "Y Combinator" story; a missing id gives `None` |
| `embed()`, `asearch`/`aget`, `ValueError` for `k=0` | a 768-dim vector; async gives the same results |
| RAM ledger | peak ≤ the budget (at 2 GB: ~1.6 GB peak, ~1.45 GB process private memory) |
| `Archive.serve()`: `/stats`, `/search`, `/item/1`, stop on an `Event` | `/search` returns the same ids as `search()` |
| the `hnz` command: `--version`, `hnz serve` | answers a search over HTTP |

Last run (2026-10-26, new wheel in a clean Python 3.12 venv): 24/24 passed in 30 s.

## 2. Automated suite (pytest)

```powershell
.venv\Scripts\python -m pytest hnz/py/tests -q -p no:cacheprovider          # all: 73 tests, ~19 min
.venv\Scripts\python -m pytest hnz/py/tests/test_serve.py -q                 # one file
.venv\Scripts\python -m pytest hnz/py/tests -q -k "lossless or budget"       # by name
```
A file whose data is missing is skipped (`s` in the output), not failed. Expected: `73 passed`.

| File | Tests | Phase | What it checks | Needs |
|---|---|---|---|---|
| `test_hnz.py` | 13 | P1 | open and stats; the same results as the CLI; listings without text; `get` is byte-for-byte lossless against Parquet; concurrent = sequential; errors; closed archive; exact date bounds; the LM tier does not block the fast tier | `out/` (sample archive), `models/`, `data/` (Parquet), `hnz/target/release/hnz.exe` |
| `test_memory.py` | 11 | P2 | the RAM budget: `BudgetError` below the minimum; a tight budget reads from disk with the same results as in RAM; the memory report; `BusyError` when there is no turn; the disk cache; the process stays within the budget; request limits | `out/`, `models/` |
| `test_block.py` | 31 | P3 | the compact block `hn.hnzb`: open/verify; lossless `get`; disk = RAM; concurrency; the same results as the CLI; process memory; **damage** (a changed byte in a record, metadata, padding, the embedder, vectors or postings) found at open or when read, never a crash; files that are not blocks; incompatible versions; a rewrite without changes is identical | `work/p3/hn-sample.hnz`, `data/`; some cases also `hnz/target_p3/release/hnz.exe` and the `rewrite_block` example |
| `test_rerank.py` | 14 | P4 | reranking on by default; the embedder reproduces the packed vectors; keyword mode; determinism cold and warm; disk = RAM; concurrency; reuse of decoded prefixes; `deadline_ms` degrades but answers; fusion parameters; progress under load; `rerank_depth`; filters; process memory; the asyncio API | `work/p3/hn-sample-v3.hnz` |
| `test_serve.py` | 4 | P6 | the HTTP server answers like the library (`/search`, `/item`, `/stats`, `/memory`, `/`), 400/404 errors, raw HTTP (HEAD without a body, 405 for POST, 400 for a body, 431 for a head over 16 KiB, garbage), stops on an `Event` and frees the port; Ctrl+C; bad arguments and a port in use; the `hnz` command | `work/p3/hn-sample-v3.hnz` |

The sample data are built by the earlier phases: `out/` by P1 (`hnz build`), the blocks by
`hnz pack --spec work/p3/pack.json --out work/p3/hn-sample.hnz` and
`hnz pack --spec work/p3/pack-v3.json --out work/p3/hn-sample-v3.hnz`.

Rust tests (the core): `cd hnz; cargo test --release --lib` → 35 passed in < 1 s.

Last run (2026-10-26): Python 73/73 (18.9 min), `test_serve.py` 4/4 after its last change, Rust 35/35.

## 3. Block and quality checks (CLI)

```powershell
work\bin\hnz-p6.exe verify-block work\p6\hn-2026-08.hnz --level full                  # every part's hash: ~5 s
work\bin\hnz-p6.exe verify-block work\p6\hn-2026-08.hnz --level deep --decode 0.005   # against the Parquet: ~10 h
work\bin\hnz-p6.exe eval --root work\p6\hn-2026-08.hnz --file eval\2024-01.jsonl --rerank --emb ov --quick --within-month
```
- `--level full` must print `"ok": true, "differences": 0`; `--level deep` compares every record
  with its Parquet source and decodes 0.5 % of them with the LM (last run: 0 differences, 581 min).
- `eval` prints P@10 and MRR for 300 queries; with `--within-month`, expected ~0.31-0.34 P@10 for
  the hybrid search and ~0.25-0.27 for keyword (the results are in EXPERIMENT-LOG E36).

`python scripts/p6_run.py` runs the build, these checks, the tests and the eval as resumable stages
(`python scripts/p6_run.py status` shows which are done; logs in `logs/p6/`).
