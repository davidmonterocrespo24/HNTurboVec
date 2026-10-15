# `hnz` system architecture (Rust)

Status: **implemented and verified on 3 sample months** (2010-06 and 2015-06 in the cold tier,
2024-01 in the hot tier). Numerical evidence in [BITACORA.md](BITACORA.md); usage in
[../hnz/README.md](../hnz/README.md). Adversarial review and fixes: E25.

## Principles
1. **Text is what takes up the most space** → tiers by age (like an LSM): zstd for recent data
   (reads in µs) and **LM + arithmetic coding** for old data (half the size). A
   segment can be "cooled down" later with `hnz retier`.
2. **Hybrid search**: EmbeddingGemma 2 binary vectors + compact BM25 (each one fails
   where the other succeeds; DeepMind 2508.21038). Calibrated fusion (E14): cosine + 0.15·bm25/(bm25+20).
3. **Temporal order**: docIDs follow the HN id, so a date range is a contiguous range.
4. **Lossless and verifiable**: `hnz verify` compares every record with the original Parquet (the
   `deleted` rows too, which are stored as metadata so as not to break the threads).
5. **LM tier determinism**: exact int8 matmul + our own math in a fixed order → the same
   binary yields the same bits on any compatible CPU (requires x86-64-v3 / AVX2; AVX-VNNI is
   optional: with or without it the result is identical). Each record carries 1 check byte
   (validated when the record is decompressed in full; snippets do not cover it) and opening
   requires the same model id and `MATH_VERSION`.

## Segments (self-contained, immutable folder; `build` is atomic via `<seg>.building`)
| File | Contents | B/doc hot (2024-01) | B/doc cold (2015-06) |
|---|---|---|---|
| `manifest.json` | id/date range, tier, LM model, `build_id` | — | — |
| `meta.bin/.idx`, `authors.zst` | id/time/parent/type/dead/deleted/score/descendants/author/poll/parts, blocks of 1024, delta varint + zstd | 5.5 | 5.7 |
| `meta.flags.zst` | 1 B/doc (type, dead, deleted) to filter without decompressing; loaded into RAM on open | 0.10 | 0.16 |
| `text.bin/.idx/.dict` (**zstd tier**) | blocks of 16, columnar, HTML→reversible compact form (2 dialects, truncated links), zstd-19 + dictionary | 116.8 | — |
| `lm.bin/.idx/.json` (**LM tier**) | per record: varint header + check byte + arithmetic stream; groups of 32 | — | 64.4 |
| `bm25.fst/.post/.norms/.json` | term FST (df=1 inline), postings in blocks of 128 with BitPacker4x, 1 B norms, MaxScore; no pruning (E25) | 49.2 | 53.9 |
| `vec.bin/.has` | sign of the EmbeddingGemma 2 dims: 768 bits (hot) / 512 bits (cold, E19) | 96.1 | 64.1 |
| **Total** (`hnz stats`, E26) | | **267.6** | **188.2** |

Projection for ~49.3 M items with the last 2 years hot (~9 M) and the rest cold:
≈ 9 M × 267.6 B + 40.3 M × 188 B ≈ **10.0 GB**, plus the runtime models (0.38 GB: int8 embedder, LM and tokenizers) → **≈ 10.4 GB** in total.

## Query
```
query (text, k ≤ 500, from/to, type, author)
  ├─ no text: listing of the most recent items that match the filters (no embedding)
  ├─ embedding: EmbeddingGemma 2 (OpenVINO int8 IR, CPU) from Rust ................. ~20-25 ms
  ├─ per segment overlapping the dates (rayon):
  │    · with author or rare types (job/poll): exact metadata scan + scoring
  │    · otherwise: vectors (Hamming 256 bits → asymmetric) + BM25 (MaxScore with global IDF);
  │      filters with the in-RAM flags column, in score order; if the segment does not reach k
  │      and its candidate list was truncated, only that segment is re-run with ×4 and then ×16
  │    · only docs with signal: some term (keyword), vector (vector), either of the two (hybrid)
  ├─ fusion → top-k ........................................................ ~3-10 ms
  └─ retrieve: zstd (µs) | LM tier: results + thread titles in ONE batch (KV-cache,
     own thread pool); 64-token body snippet by default, full body on demand
```
Measured latencies: see the table in [README.md](README.md) and E26 in the log.

## Layered cache
- **L1 RAM**: already decompressed LM-tier records (20,000) and story titles.
- **L2 disk** (`<root>/_cache_l2.bin`, ≤ 512 MB, not counted in `hnz stats`): already decompressed
  cold records, re-compressed with fast zstd; keyed by the segment's `build_id`, validated
  header, exclusive access (one process).
- **L3**: the store (zstd or LM).

## Robustness
- Atomic `build`; the engine ignores `*.building`, `*.retier`, `*.old` and rejects overlapping segments.
- `retier` rebuilds into `<seg>.retier`, verifies (sample of 2,000 or `--verify-all`) and leaves the
  original as `<seg>.old` (it only deletes it with `--verify-all --delete-old`).
- Changing the LM math increments `MATH_VERSION`: old segments are rejected with a clear
  message and must be rebuilt from the Parquet (`hnz build --tier lm`).
- Several LM models can coexist in one archive (each segment states its own).

## Build (offline)
1. `scripts/prepare_models.py`: EmbeddingGemma 2 IR (documents in buckets, queries in int8), tokenizer and
   `.hnzlm` LM.
2. `scripts/pipeline.py --years 2006-2026`: download → embeddings on the iGPU (~204 docs/s, ~2.8 days for everything)
   → segment per year (zstd for the last 2 years of the dataset, LM for the rest; ~120 rec/s on CPU, ~4 days)
   → vectors (768/512 bits). Resumable; rebuilds the current year if new months appear.
3. `hnz verify` per segment (continuous batching in the LM tier).

## Models (counted in the total system size)
| Model | Use | Size |
|---|---|---|
| EmbeddingGemma 2 (text) int8 IR + tokenizer | query embedding | 276 MB + 32 MB |
| EmbeddingGemma 2 fp16 IR per bucket | build only (iGPU) | 4 × 542 MB (not needed for querying) |
| pythia-70m `.hnzlm` int8 + tokenizer | cold-tier (de)compression | 72 MB + 3.5 MB |

## Compact profile: measured plan, NOT implemented yet (E27-E29)
Requested goal: 4-5 GB for all of HN, accepting up to 10 s per query (with decompression). Idea:
spend the time on reranking, not on storage: a minimal index yields 100-200 candidates, which are
re-scored with their own decompressed text (re-embedding + exact BM25). Decision of 2026-10-14:
document it and not implement it for now.

| Component (all of HN) | Today | Compact | How | Measured? |
|---|---|---|---|---|
| Text | 3.65 GB | **2.84 GB** | everything in the LM tier with the LM trained on HN (`models/lm-hn-70m.hnzlm`, −9.2%) | yes (E29) |
| Vectors | 3.45 GB | **1.18-1.58 GB** | 192-256-bit binary + rerank the top-100/200 by re-embedding the decompressed text | yes, simulated (E27a) |
| Keywords | 2.61 GB | **0.55-0.65 GB** | rare terms only (df ≤ 0.02%) + on-the-fly BM25 of the frequent ones over the candidates | size and hybrid quality yes (E27b); on-the-fly BM25 no |
| Metadata | 0.29 GB | 0.28 GB | same | yes |
| Models | 0.38 GB | 0.25-0.38 GB | int4 embedder (optional) | no |
| **Total** | **10.4 GB** | **5.74 GB** lossless · **5.24 GB** (−1%) · **~4.96 GB** with format improvements | | |

- Planned query: minimal index → quick view (~0.4 s); decompress the first 128 tokens of
  ~100 candidates (~3 s) and re-embed them on the iGPU; exact BM25; full text of the top-10 → ~4 s
  median (estimate, not implemented).
- To implement: reranking stage in the engine; rare-term index + on-the-fly BM25;
  profile selectable in `build`/`scripts/pipeline.py`; shorter record headers; vocabulary shared
  per year; int4 embedder; rebuild all of HN (~5 days of iGPU + CPU, resumable).
- Discarded (E28): 16-18 B PQ codes (worse candidates than the binary ones) and the thread title as
  LM context (no savings).
- Optional: train the HN LM on more data (5% of HN, ~7 h on the iGPU; the 1% one was still improving).
- The LM is part of the archive (like a dictionary): a copy of the exact `.hnzlm` must be kept.

## Pending / improvements
- [ ] Full build of the 239 months (days of compute); the dataset is already downloaded (`scripts/download_hn.py`).
- [x] HN-specific LM: trained on the iGPU in 93 min, −9% of text at the same cost (E29).
- [ ] Compact profile (~5 GB, ~4 s queries): plan above.
- [x] Cross-encoder reranker (open-source equivalent of Jev/System One), tested in E26: +6-10% in
      P@10 (significant) reading the full text of 30 candidates, at 0.6-1.6 s per query on CPU;
      with snippets, no gain in the cold tier → not on by default; the large ones remain to be tested.
- [ ] Topic tag/graph layer (exploration, routing) — optional (E08).
