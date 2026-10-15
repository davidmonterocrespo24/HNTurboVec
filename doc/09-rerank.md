# Reranking (P4): a compact block's candidates reordered by their prefix

**Status:** implemented on 2026-10-15 (log entry E33 in [BITACORA.md](BITACORA.md)); quality and
cold latency measured; the memory run with concurrent clients pending. Design: a panel of three
architects and a judge (E33); the measurements it rests on: E30a (prefix embeddings), E30c
(fusion), E30f (cascade).

## What it does

A cold search of a compact block (hybrid mode, the default):

1. **Stage 1** (as in P3): the query vector against the 256-bit codes gives the top 100 by the
   asymmetric score `s1`; the rare-term index gives the top 20 by BM25. Filters (dates, types,
   author, dead/deleted) apply here, so every reranked document passes them.
2. **Prefixes.** Each candidate's prefix (title, url and the first 128 LM tokens of the body,
   [08-block.md](08-block.md) "Vectors") comes from the prefix cache, from the block's
   `text/prefix_fallback` (records stored without LM tokens), or from the **shared LM service**,
   which decodes the prefixes of every concurrent query in one continuous batch.
3. **BM25 over the prefix:** `b` = BM25 of every query word — frequent words included, which have
   no postings in a block — over the prefix text, with the global idf (the block's df of every
   term) and the document's full-length norm.
4. **Cheap score** `R0 = s1 + 0.3 · b / (b + 20)` orders the candidates.
5. **Cascade:** the top 50 by R0 are re-embedded (their document prompt v2, rebuilt from the
   decoded prefix exactly as `hnz prompts` built it at pack time) and scored
   `cos + 0.15 · b / (b + 40)`; the rest follow in R0 order. Re-embedding only the top 50 gives
   the same quality as re-embedding all ~110 candidates (E30f, E33e).
6. **Results** reuse the decoded prefixes: a record that fits in its prefix (~84 % of them) comes
   back whole, the others as the same 64-token snippet as before.

Other modes:
- **Vector:** the 100 vector candidates, all re-embedded, by cosine.
- **Keyword:** if every query word is rare, the P3 path (exhaustive postings). If one is frequent
  (e.g. "rust async runtime"), candidates = the rare top 50 + the vector top 100, kept when their
  BM25 over the prefix is > 0, ordered by it; the response carries a warning: frequent words are
  only looked for among the documents closest to the query, in their first 128 tokens (the user's
  decision of 2026-10-15: no index of frequent words).
- **Listing** (no text): unchanged.

Decoding and re-embedding **overlap**: each candidate's features are computed as soon as its
prefix is decoded, and a candidate whose R0 is already among the 50 best lower bounds (R0 when
known, `s1` while it is still being decoded) is queued for embedding at once. The embedding worker
takes the queued candidates with the best R0 first and drops those whose R0 has fallen below the
50th best lower bound (that bound only rises, so they can no longer make the top 50). The final
choice uses every R0, so results never depend on timing; a few speculative embeddings may still
be wasted (those already embedded when they drop out).

## Determinism

Each document prompt is embedded alone at its canonical shape `[1, tokens rounded up to 16]`
(≤ 256), with the precision pinned to f32, so its vector never depends on what else is embedded or
on how many streams run; the LM decodes every sequence identically in any batch. Results are
identical with 1 or 4 embedding streams and with or without the overlap (checked byte for byte).

## Deadline (opt-in)

`search(…, deadline_ms=N)` (CLI: `hnz search --rerank --deadline-ms N`) answers within N ms of
the call (the CLI: of the start of the search), the user's decision: no deadline by default.
- The deadline passes while prefixes are still being decoded: the decodes not done yet are
  cancelled (the shared LM service drops them) and the answer is stage 1's, as without reranking;
  `rerank.degraded = "decode"` and a warning.
- It passes while the top 50 are being re-embedded: the candidates keep the cheap score's (R0)
  order; `rerank.degraded = "embed"` and a warning. That order is better than stage 1's (P@10
  0.217 vs 0.211 in E33e).
- The prefixes decoded before the deadline stay in the cache, so a repeated query goes further.
- Fetching the k results' text comes after the deadline (its own cost, ~40-60 ms cold).

## Self-check of the embedding chain

When the library loads the embedder of a block with prompt v2 vectors, it decodes the prefixes of
the block's 4 short canaries (records of 24-64 LM tokens chosen at pack time), rebuilds their
document prompts, embeds them and compares the signs of their first 256 dimensions with the codes
stored in the block. A matching chain (LM, prompt, tokenizer, IR) flips 2.35 of 256 on average and
at most 9 (E33a, 1,200 prompts); a broken one flips about 128, so a mean above 32 makes the load
fail with `IncompatibleBlockError`. The
mean is reported as `memory()["embedder"]["measured"]["check_flips"]`. It costs ~0.1-0.2 s, once.

## Requirements

- A block with **prompt v2** vectors and an **embedder IR with dynamic shapes**
  (`scripts/export_gemma2_dyn.py` → `models/gemma2_text_dyn_int8`, the pack spec's `embed_ir`).
  The same IR embeds the queries (shape `[1, 64]`, as before).
- `hnz.open(…)`: `search(…, rerank=None)` reranks whenever the archive can; `rerank=True` raises
  `ValueError` if it cannot; `rerank=False` gives the P3 results. The response reports
  `rerank: {candidates, decoded, tokens, embedded, ms_stage1, ms_decode, ms_embed}` and, in keyword
  mode with frequent words, `warning`. CLI: `hnz search --rerank`, `hnz eval --rerank`.
- `HNZ_EMBED_STREAMS` (default 1): documents embedded at the same time; the embedder's memory
  reservation grows by 64 MiB per stream. More than 1 was slower on the measured laptop (below).
- `HNZ_EMBED_THREADS`, `HNZ_EMBED_CORES` (`ANY_CORE`, `PCORE_ONLY`, `ECORE_ONLY`): the embedder's
  CPU threads and core type (default: OpenVINO's choice). OpenVINO's results may change slightly
  with the number of threads, so set them the same way on every machine that must agree bit for
  bit.

## Quality (E33e, 300 queries per month, P@10)

| Setting | 2024-01 | 2010-06 | 2015-06 | mean |
|---|---|---|---|---|
| P3 block, hybrid (no reranking) | 0.246 | 0.168 | 0.220 | 0.211 |
| **P4, hybrid, cascade 50** | **0.282** | **0.193** | **0.241** | **0.239** |
| P4, all candidates re-embedded | 0.282 | 0.192 | 0.239 | 0.238 |
| today's full index and text (P0 reference) | 0.278 | 0.208 | 0.249 | 0.245 |
| P3 block, keyword | 0.119 | 0.056 | 0.080 | 0.085 |
| **P4, keyword with frequent words** | **0.229** | **0.152** | **0.187** | **0.189** |

Quality gate (user's decision): mean >= 0.235 and every month >= today − 0.02: **met** (with the
stored document vectors standing in for re-embedding). With the real CPU re-embedding (E33e step
3): 2024-01 0.281 (stored vectors: 0.282), 2010-06 0.194 (0.193); keyword search identical.

## Latency (cold, sample block)

Measured with `experimentos/bench_p4_latency.py` (E33e step 4, `resultados/results_e33e_latency.txt`):
20 eval queries, each in a new `hnz search --rerank` process (hnz's caches empty, the OS file cache
warm, model loading not counted), k = 10, on a Core Ultra 7 265H laptop (6 P + 8 E + 2 LP-E cores,
16 threads) with nothing else running:

| Embedding streams | LM threads | search p50 | p90 | decoding | wait after decoding |
|---|---|---|---|---|---|
| **1 (default)** | **8 (default)** | **4,647 ms** | **5,741 ms** | 3,608 ms | 866 ms |
| 2 | 8 | 4,989 | 6,771 | 3,801 | 752 |
| 4 | 8 | 5,075 | 7,240 | 4,140 | 512 |
| 4 | 12 | 5,282 | 7,044 | 3,987 | 574 |
| 8 | 8 | 6,376 | 7,638 | 4,344 | 743 |

- **Decoding is ~78 % of a cold query:** ~100 prefixes, ~6,350 LM tokens, ~1,760 tokens/s with 8
  threads. Stage 1 takes 5 ms; fetching 10 results 53-59 ms (they reuse the decoded prefixes).
- More embedding streams shorten the wait at the end but take CPU from the LM, which then decodes
  more slowly; 12 LM threads are no faster than 8 on this hybrid CPU (its E-cores hold back every
  step). The defaults are the best setting here.
- The embedding queue with priorities (above) cut the wait after decoding from a median of 571 ms
  to 434 ms on 21 queries, with identical results.
- **Target:** ~4 s needs >= 16 fast threads or an iGPU (user's decision); this 16-thread hybrid
  laptop gives p50 4.6 s, p90 5.7 s. A repeated query reuses the cached prefixes.
