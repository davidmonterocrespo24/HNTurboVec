# RAM governance (P2): `hnz.open(path, ram="2GB")`

**Status:** implemented and measured on 2026-10-14 (log entry E31 in [BITACORA.md](BITACORA.md)).
Design context: [06-libreria.md](06-libreria.md).

## What the budget covers

`ram` is a ceiling for **everything hnz holds in the process**:

| Class (`memory()["budget"]`) | What | How it is known |
|---|---|---|
| `models` | LM models of the cold tier; the query embedder (OpenVINO + tokenizer) | measured with the OS counters while loading (at least the file size); the embedder's *load peak* is reserved until it is loaded |
| `index` | always-loaded structures (BM25 dictionaries and norms, flag column, block indexes, author list) + the segment parts the planner loads | exact (`Vec` sizes; the author list is one buffer + offsets) |
| `working` | the LM decompression batch (`lm_pool`) and the working memory of running queries (`query_pool`) | reserved whole at open |
| `cache` | decompressed records and titles; the RAM index of the optional disk cache | weighed per entry (strings + hash-table slack), bounded |

Not covered: the host application's own memory (e.g. the Python objects built from the results)
and the operating system's file cache (outside the process; it speeds up the disk reads).

## How it works

1. **Open** with nothing loaded (only the fixed structures), load and measure the LM models, and
   reserve the embedder's load peak until it is loaded (estimated as 1.4 × weights + 4 × tokenizer
   + 32 MiB = 519 MiB; the int8 EmbeddingGemma 2 measures +447 MiB, peak and steady).
2. **Plan** (`hnz/src/plan.rs`): if `ram` is below the floor (models + fixed structures + minimum
   pools + minimum cache), `BudgetError` with the minimum. Otherwise, in order of benefit:

   | Step | Takes | Why |
   |---|---|---|
   | 1 | LM batch and query pools: up to 5 % of `ram` each (max 256 MB) | more concurrency without waiting |
   | 2 | cache: 5 % of `ram` (max 128 MB) | repeated results without the LM |
   | 3 | **vectors**, newest segments first | every semantic search scans them whole |
   | 4 | metadata, newest first | every result, date-range edges, author filters |
   | 5 | BM25 postings, newest first | keyword queries |
   | 6 | cache up to 25 % of `ram` (max 1 GB) | each hit saves ~25 ms of LM per record |
   | 7 | compressed text, newest first | on disk it only costs one read |
   | 8 | the rest → cache | |

   Without a limit (`ram=None`) everything is loaded, the pools get 256 MB each and the cache 1 GB.
3. **What is not loaded is read from disk on demand** with positional reads (no mmap: mapped pages
   would enter the process working set uncontrolled): BM25 lists one 128-document block at a time,
   vectors in windows of 8,192 documents per thread, metadata and text one block/record at a time.
   Results are identical to the all-in-RAM mode (tested).
4. **LM batch under a cap** (`lm_pool`): records join the batch while the sum of their bounds fits
   (bound = tokens × KV bytes per token — 24 KiB for pythia-70m — × 7/6 for the moment a layer's
   cache grows, + logits and activations per step; the KV cache can never grow past it) and each
   finished record frees its memory and lets the next one in (continuous batching). The batch also
   reserves the sampling scratch of its threads (vocabulary × 16 B each, ~7 MiB). The minimum pool is
   the longest possible record (2,048 tokens) plus that scratch: ≈ 63 MiB. Same output with any cap:
   each sequence is computed identically in any batch. The LM's floating-point state is forced to
   the default (round to nearest, no flush-to-zero) while it runs: a host library built with
   fast-math cannot break lossless decompression.
5. **Caches**: two generations with a byte limit (≈ LRU): records 7/8, titles 1/8; with `cache_dir`
   the disk cache's RAM index takes 1/8 first (it is opened after planning and, if it holds more
   entries than its share, only the newest stay indexed). After the embedder loads, the unused part
   of its reservation goes to the caches.
6. **Admission control**: at most `max_concurrent` (8) searches/gets at once, served in arrival
   order. A search declares its working memory: 2 MB + k × (8 KB snippet | 64 KB full text) + its
   candidate lists (first round of every segment it touches, as many segments at once as worker
   threads); a get, 1 MB. Without room it waits up to `timeout` (30 s), then `BusyError`. When a
   segment's candidate round has to be widened (×4, for very selective filters), it asks for the
   extra memory without waiting; if there is none, it returns what it has. Internal steps never take
   a second turn.
7. **Request limits**: query text up to 8 KB (`ValueError` beyond; the embedder reads its first 64
   tokens), and BM25 uses at most the 64 rarest query terms that exist in the archive (each term
   opens a cursor per segment).

## API

```python
import hnz
hn = hnz.open(r"D:\hn\out", ram="2GB",            # or "10GB", bytes, None = no limit
              preload=True,                         # load the embedder now (recommended for servers)
              max_concurrent=8, timeout=30.0,       # admission control
              cache_dir=None, cache_dir_size="512MB")   # optional persistent cache of decompressed records
m = hn.memory()
m["budget"]      # {'limit', 'used', 'peak', 'models', 'index', 'cache', 'working'} in bytes
m["plan"]        # floor, what was loaded (segment, part), lm_pool, query_pool, cache
m["engine"]      # loaded/total bytes per part, cache sizes and hit counts
m["embedder"], m["admission"], m["process"]       # embedder measurements, turns/waits/rejections, OS view
```
Errors: `hnz.BudgetError` (a `MemoryError`) when `ram` is below the minimum; `hnz.BusyError` (a
`TimeoutError`) when an operation gets no turn within `timeout`.

## Measurements (sample archive: 3 months, 637,046 items, 150 MB of segments)

Minimum (`floor`): **712 MiB (746 MB)** = models 597 MiB (LM 79 + embedder reservation 519) + fixed
structures 7 + LM batch 63 + queries 29 + cache 16. Process memory with 8 concurrent clients (mixed
hybrid / keyword / full-text / date-filtered / listing / get), working-set peak above the baseline
before opening, as reported by Windows, 90 s per budget:

| `ram` | Loaded into RAM | Process peak | Peak / `ram` | Ledger peak |
|---|---|---|---|---|
| floor + 24 MiB (736 MiB) | nothing (all from disk) | +586 MiB | **0.80** | 705 MiB |
| floor + 120 MiB (832 MiB) | vectors, metadata, BM25 postings (78 MiB) | +664 MiB | 0.80 | 789 MiB |
| 2 GB (1,907 MiB) | everything (146 MiB) | +728 MiB | 0.38 | 933 MiB |
| None | everything | +728 MiB | — | 1,254 MiB |

30 minutes with 8 clients at the minimum budget (acceptance criterion of P2): first run (before the
review fixes, minimum 963 MiB) 29,088 operations, 0 errors, peak **0.84 × `ram`**; final run: running at the time of this commit.

The embedder loads at its steady size (+447 MiB): its tokenizer is parsed before OpenVINO compiles
the model. Parsed afterwards (the first version), the parse transient landed on top of the
compiled model and the load peaked at +746 MB, which is why the first minimum was 955 MiB. In
steady state the process sits at +558 MiB with everything on disk and +680 MiB with everything
loaded. The ledger always stays above what the OS measures (it reserves the pools in full).

## Caveats

- The opaque parts (OpenVINO, tokenizers) are measured with OS counters around their load; with
  other threads working at the same time the measurement is approximate (it never goes below 60 %
  of the estimate). `preload=True` measures them cleanly at open. The LM model's measured size varies
  by a few MiB between opens, so the minimum does too.
- The floor grows with the archive: about 2 bytes per document always stay in RAM (flag column and
  BM25 norms, ~100 MB for all of HN), the per-segment author lists (~90 MiB for all of HN in monthly
  segments) and the BM25 dictionaries, whose size depends on the final format (P3 replaces the
  per-segment dictionaries with one global index of rare terms).
- Disk-mode files are read with plain handles: as before, the archive must not be rewritten in place
  while a process serves it (builds and `retier` swap whole folders, which Windows refuses while
  files are open).
- Error messages are still in Spanish (like the rest of the library).
