# `hnz` library for backends: design (2026-10-14)

**Status:** **P0 and P1 done** (E30: `import hnz` on the current format, reviewed; measurements E30a-d:
compact query −3% P@10 vs. today, 1st stage at 49 M documents in ~40-53 ms).
**P2 done (E31, 2026-10-14):** RAM governance implemented and measured — see [07-ram.md](07-ram.md)
(English). Next: P3.

**Decisions made (2026-10-14):** server possibly **without an iGPU** (everything must work well with
CPU only); **256-bit binary vectors** and reranking of **100 candidates** by re-embedding the first
128 decompressed tokens (~5.5 GB for all of HN, ~4-4.5 s cold on CPU; vector P@10 0.221 vs. 0.229
today, E30a); optional persistent cache and pre-warming (disabled by default); frequent words are
searched for among the semantic candidates.

Design made with a panel of 5 agents (3 specialists: API and packaging, RAM budget, compact format
and query; an architect who integrates it; and an adversarial review against the code and the
log). Full panel output (in English): [anexos/libreria-panel.json](anexos/libreria-panel.json).

## Requirements (decided with the user)
- A **solid block** usable from a backend, importable from **Python** (other languages, optional).
- **As compact as possible on disk** when not in use: compact profile (~5 GB, ~4 s cold).
- When loaded, a **RAM budget** (e.g. 2 GB or 10 GB): it may decompress/cache in RAM to go faster,
  but **everything** (models, embedder, indexes, LM memory, caches) fits in it.
- **Lossless.** Windows x86-64 (like this machine). A read-only snapshot that is rebuilt from time
  to time. One process with several queries at once (e.g. FastAPI with one worker).

## What would be delivered
- **Python package** `pip install hnz`: `abi3` wheel (Python ≥ 3.10, win_amd64) with the core in Rust
  (PyO3). It only depends on the `openvino` package (it uses its DLLs; it does not import it).
- **Data block**: folder `hn-AAAA-MM.hnz\` with `hn.hnzb` (container: data, LM, tokenizers,
  manifest and a hash per part) + `embed.xml`/`embed.bin` (EmbeddingGemma 2 int8). **≈ 5.45 GB**
  with 256-bit vectors, **≈ 5.1 GB** with 192. Nothing else on disk by default.
- `hnz.exe` (CLI + HTTP server) and, optionally, `hnz.dll` + `hnz.h` (C, JSON input and output) for
  other languages.

## Python API
```python
import hnz
with hnz.open(r"D:\hn\hn-2026-10.hnz", ram="4GB") as hn:
    r = hn.search("how do I deal with burnout", k=10, since="2015", until="2018-06", types="comment")
    for h in r:
        print(h.rank, h.date, h.by, h.story_title, (h.text or "")[:120])
    it = hn.get(9700000)                          # verbatim text (original HN HTML)
    jobs = hn.search(types="job", since="2024")   # no text: the most recent
    print(r.timings_ms, hn.stats()["memory"])
```
- Thread-safe, releases the GIL; async `asearch`/`aget`; `raw=True` returns dicts with the same JSON
  as the HTTP server (FastAPI returns them as is).
- `deadline`: if it runs out, it returns what is already reranked and fills the rest in 1st-stage
  order (`degraded=True`) instead of failing. Selectable quality: `fast` / `balanced` / `precise`.

## RAM budget (`ram=`)
A ceiling for everything hnz holds. It is controlled with **its own ledger** (reservations per
component: models, indexes, LM memory, caches, the memory of each query; OpenVINO and the
tokenizers, measured when they are loaded), not with the process "working set", which in a backend
includes the application's own memory (a correction from the review).

| | 1 GB | 2 GB | 4 GB | 10 GB |
|---|---|---|---|---|
| Embedder | CPU | iGPU if it fits (otherwise CPU) | iGPU | iGPU |
| Vectors (1.1-1.5 GB) | read from disk | partly (the most recent) | all in RAM | all in RAM |
| Word index, metadata | disk | disk | in RAM | in RAM + author column |
| Caches (decompressed text, embeddings) | — | ~0.3 GB | ~0.3-0.9 GB | ~5 GB |
| Cold query (median, estimate) | words/listings only | ~4-5 s | ~3.3-3.8 s | ~3.3-3.8 s |
| Query already cached | 0.1-0.3 s | 0.1-0.3 s | 0.08-0.15 s | 0.08-0.15 s |
| Repeated query | < 10 ms | < 10 ms | < 10 ms | < 10 ms |

**Honest limit:** LM-compressed text decompresses at ~30 records/s (8 threads): all of HN would
take ~19 days, so **the whole block cannot be decompressed when loading it**. RAM speeds things up
because the indexes stay resident, the embedder runs on the iGPU and what has already been
decompressed is kept; cold queries remain LM-bound at any level. **Above ~4 GB cold queries barely
improve**, unless background pre-warming of the most recent items (~137,000 docs/hour, optional)
or a persistent cache file (optional) is enabled.

## Query in the compact profile
1. Query embedding (~8 ms on the iGPU, ~20 ms on CPU) and date range → document range.
2. 1st stage: binary vectors (Hamming → asymmetric table) + rare terms → ~115 candidates
   (in RAM: tens of ms; **not validated at full scale**: it could be 0.4-0.6 s).
3. Decompress the first 128 tokens of each candidate that is not cached (~7,000 tokens ≈ 3 s)
   and re-embed them in parallel (iGPU); exact BM25 with all the terms over that text; fusion.
4. Top-k: metadata, snippets (free: they are already decompressed), thread titles; full text
   only if requested (+0.5 s median).

## What the adversarial review corrected
- **RAM:** its own ledger, not "working set minus baseline" (in a backend, the application's growth
  would cause permanent rejections).
- **Embedder on CPU:** re-embedding candidates on CPU runs at ~20 docs/s (measured) → 6-10 s per
  cold query, not 4. Use the iGPU whenever there is one (minimum ~1.7 GB of RAM).
- **LM scheduler:** without "parking" jobs, a `get()` or a snippet would wait for the heavy queries
  in progress (each step with 64 active sequences takes ~28 ms). The active sequences must be
  limited when interactive work is waiting.
- **Truly lossless:** also store the order of `kids` (≤ ~6 MB for all of HN) and check 100% of the
  bytes when packing.
- Freeze the function that builds the text to embed (`prompt.rs`) before days of iGPU work;
  keep ≥ 512 sign bits in the build material in case the width has to change.
- Validate the 1st stage at 49 M documents (target ~5 ns/doc/thread) and the reranking with the real
  tie-break; enforce the processor rounding mode (MXCSR) on the LM threads inside a host process.
- Realistic machine time to build all of HN: **~8 days** (LM on CPU and embeddings on the iGPU
  compete), preferably month by month so it can resume with little loss.

## Phases (panel estimate)
| Phase | Work | Acceptance |
|---|---|---|
| P0 Preliminary measurements | prefix embeddings (E30a), on-the-fly BM25 and fusion (E30c), global index (E30e), 1st stage at scale | confidence intervals; freeze P, bits, K, thresholds |
| P1 Python library on the current format | Cargo workspace, `api.rs`, PyO3, OpenVINO loader, embedding service | installs in a clean venv; same results as `hnz search`; LM bit-for-bit identical |
| P2 RAM governance | ledger, planner, reads without mmap, admission control | memory ≤ 1.03 × `ram` with 8 clients for 30 min |
| P3 Compact block v1 | container, `hnz pack`, global index, `verify-block` | lossless on the sample; projected size ≤ 5.6 GB |
| P4 Reranking | prefix decompression + re-embedding + exact BM25 + fusion | quality ≥ today − 0.01; latencies measured |
| P5 Full build | `scripts/pipeline.py --profile compact` for all of HN | size, lossless (0.5% sample + checks), memory |
| P6 Packaging | wheel, OpenVINO DLLs, server, optional C ABI, tests | clean install on Windows |
| P7 Optional improvements | record headers, cascaded reranking, int4 embedder, persistent cache, zstd tier for the most recent items | each one measured |

## Decisions pending from the user
1. Start P0 (measurements) and P1 (library on the current format), which do not commit to a format.
2. Final LM: the current one (1% of HN) or train it on 5% (~7 h, −2-4% of text).
3. Vector bits: 256 (≈ 5.45 GB) or 192 (≈ 5.1 GB); decide after E30d.
4. Partial response (`degraded`) when the deadline runs out: acceptable?
5. Search by frequent words only: is it enough to find them among the semantic candidates, or an
   index of frequent words (+1.4-1.8 GB)?
6. Will there be an Intel iGPU on the server? (without it, ~0.5-1 s more per cold query, or more).
7. Disk outside the block: is an optional persistent cache or pre-warming allowed?
8. Build tools: can the MSVC Build Tools be installed for the wheels?
