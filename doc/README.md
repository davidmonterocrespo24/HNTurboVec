# HNTurboVec — project documentation

**Goal:** to have *all* of Hacker News (~49.3 M items) in the **minimum disk space** possible,
with **Google/RAG-style search** (free-form query, keywords, number of results, year/date range)
and **verbatim retrieval of the text** of the results. Preparation may take a while; what
matters is daily use: **fast, accurate and compact**. Implementation in **Rust** (`hnz/`).

## How this folder is organized

| Document | Contents |
|---|---|
| [BITACORA.md](BITACORA.md) | **Chronological record of ALL experiments** (what was tried, script, result). Check it before trying something, so as not to repeat it. |
| [01-almacenamiento-texto.md](01-almacenamiento-texto.md) | Text/metadata compression: zstd, blocks, ordering, LM as a compressor. |
| [02-busqueda.md](02-busqueda.md) | Search indexes: vectors (binary, PQ, RaBitQ, turbovec), BM25, real quality, embedding models. |
| [03-etiquetas-jerarquia.md](03-etiquetas-jerarquia.md) | Idea of ~1000 tags + a graph of levels: compression, routing, hierarchy. |
| [04-investigacion-arxiv.md](04-investigacion-arxiv.md) | Papers reviewed (priority to Google/Microsoft), what is useful and what is not. |
| [05-arquitectura.md](05-arquitectura.md) | The system built in Rust: format, query, caches, robustness, sizes. |
| [06-libreria.md](06-libreria.md) | Design of the library for backends (Python, RAM budget, compact block); pending approval. |
| [07-ram.md](07-ram.md) | (English) RAM governance (P2): what `ram=` covers, the planner, disk reads, capped LM batch, admission control, measurements. |
| [08-block.md](08-block.md) | (English) The compact block (P3): byte layout of `hn.hnzb`, parts, manifest, integrity levels, tools (`pack`, `verify-block`, `inspect`), sample measurements. |
| [09-rerank.md](09-rerank.md) | (English) Reranking (P4): candidates, prefixes decoded by the shared LM service, BM25 over the prefix, cascade re-embedding, keyword mode with frequent words, determinism, quality. |

## Repository structure

| Folder | What is in it |
|---|---|
| `hnz/` | Rust engine (library + CLI `hnz.exe`) and Python package in `hnz/py` (`import hnz`) |
| `scripts/` | Tools to build the archive: download, pipeline, models, LM training |
| `experimentos/` | Experiment scripts (`bench_*.py`, `make_eval.py`) cited in the log |
| `resultados/` | Experiment outputs (`results_*.txt`) |
| `eval/` | Evaluation query sets (title → thread comments) |
| `doc/` | This documentation; `doc/investigacion/` = prior research (Hermes) |
| `data/`, `emb/`, `models/`, `out/` | Not in git: dataset, embeddings, models and built archive |

## Current status

- [x] Measurement of the size of HN and of the baseline (prior research in Hermes).
- [x] Benchmarks of storage, vectors, BM25, real quality, LM, tags (month 2024-01).
- [x] EmbeddingGemma 2 on the iGPU (OpenVINO): ~204 docs/s → all of HN in ~2.8 days (E10, E13).
- [x] `hnz` system in Rust: segments with a zstd tier (hot) and an LM tier (cold), BM25,
      binary vectors, hybrid search with filters, CLI + HTTP + web, lossless verification.
- [x] In-house deterministic LM compressor (Pythia int8, AVX-VNNI, arithmetic coder) (E17, E20).
- [x] L1/L2 caches, `retier`, int8 query embedder, measured latencies (E23, E24).
- [x] Adversarial review (141 agents): 60 findings confirmed and fixed (E25).
- [x] 2nd, 3rd and 4th reviews (18 + 4 + 18 findings fixed); sample segments rebuilt with the
      final binary and fully verified (637,046 records identical to the original); same
      byte-for-byte result with two different binaries (E26).
- [x] Cross-encoder reranker (open-source equivalent of Jev/System One) tested (E26): +6-10% in
      P@10 but multiplies latency by 5-7 (it needs the full text) → not used by default.
- [x] Whole dataset downloaded (`scripts/download_hn.py`: 239 months, 12.42 GB, SHA-256 checked).
- [x] LM trained on HN on the iGPU: −9% text at the same cost (E29).
- [x] **Compact profile** plan measured (E27-E29): **~5 GB** for all of HN (5.74 lossless, 5.24
      with −1%) with ~4 s queries; documented in [05-arquitectura.md](05-arquitectura.md), not
      implemented for now.
- [x] Python library `import hnz` (P1, E30) and **RAM governance** (P2, E31): everything hnz holds
      stays within `ram=`; what does not fit is read from disk with identical results; process
      memory <= 1.03 x `ram` with 8 concurrent clients ([07-ram.md](07-ram.md)).
- [x] **Compact block** (P3, E32): one read-only file `hn.hnzb` + the embedder IR, documents in
      (time, id) order, global word index of rare terms, integrity levels and canaries,
      `hnz pack` / `verify-block` / `inspect`, document prompts defined in Rust (v2); sample block
      verified 100 % against its sources; projection to all of HN: **5.32 GB** ([08-block.md](08-block.md)).
- [x] **Reranking** (P4, E33): each candidate's prefix decoded by a shared LM service, BM25 of every
      query word over it, the top 50 of a cheap score re-embedded on the CPU; frequent words found
      among the semantic candidates; opt-in deadline; P@10 0.238 (P3 block: 0.211); cold search
      ~2.8 s in the library on a 16-thread laptop ([09-rerank.md](09-rerank.md)).
- [ ] Full build of the 239 months (P5; days of CPU/iGPU).

## Result with the 3-month sample (637,046 items, E26)

| | Hot (zstd) | Cold (LM) |
|---|---|---|
| Size per item (text + metadata + BM25 + vectors) | 267.6 B | 187-188 B |
| All of HN (~49.3 M; 2 years in the hot tier) | **≈ 10.0 GB** + models 0.38 GB (≈ 10.4 GB) | |
| Hybrid search k=10 (embedding + search + text; median, empty cache) | 26 ms | 280 ms (64-token snippets) |
| Full text of 10 cold results | — | 527 ms; 24 ms cached in RAM, 53 ms cached on disk |
| Opening the archive | 180 ms | |

Across the whole archive, latency depends on how many results come from the cold tier (~25 ms
each): 103 ms median with queries on 2024 (2 of 10 cold) and 264 ms with queries on 2010
(8 of 10). Without text (listings with filters): 4 ms.

Quality (hybrid P@10, title → thread comments, whole archive): 0.278 / 0.208 / 0.249 with
queries from 2024 / 2010 / 2015. With more months this figure drops even though search does not
get worse: similar threads from other years count as misses (E26).

## Environment

- Python: `.venv` created with `uv` (Python 3.12) and installed from [`../requirements-torch.txt`](../requirements-torch.txt) (CPU torch, PyTorch index) and [`../requirements.txt`](../requirements.txt)
  (pinned versions). Models: `.venv\Scripts\python scripts\prepare_models.py` (or with the environment activated; see `hnz/README.md`).
- Data: Hugging Face dataset `open-index/hacker-news`: **239 monthly Parquet files** (2006-10 →
  2026-08) plus hourly files for the last few days (`today/…`) and statistics (628 files in total).
  The whole dataset is downloaded with `scripts/download_hn.py` (~12.4 GB in `data/`): resumable
  byte by byte if interrupted (just run it again), pinned dataset version and SHA-256 of each file.
- Embeddings cached in `emb/` (`*.npy`) so as not to recompute them.
- Hugging Face downloads with `HF_HUB_DISABLE_XET=1`: the `hf_xet` downloader dies mid-download
  on this machine (E26); `scripts/pipeline.py`, `scripts/prepare_models.py` and `experimentos/bench_rerank.py` set it.
- Machine: Intel Core Ultra 7 265H (16 cores, Arc 140T iGPU, NPU), 31 GB RAM, no NVIDIA GPU.
- Rust: toolchain `stable-x86_64-pc-windows-gnu` (user install, without Visual Studio); the
  binary requires an x86-64-v3 CPU (AVX2).
