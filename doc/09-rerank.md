# Reranking (P4): a compact block's candidates reordered by their prefix

**Status:** implemented on 2026-10-15 (log entry E33 in [BITACORA.md](BITACORA.md)); quality
measured; latency and memory measurements on a free machine pending. Design: a panel of three
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
known, `s1` while it is still being decoded) is embedded at once. The final choice uses every R0,
so results never depend on timing (a few speculative embeddings may be wasted).

## Determinism

Each document prompt is embedded alone at its canonical shape `[1, tokens rounded up to 16]`
(≤ 256), with the precision pinned to f32, so its vector never depends on what else is embedded or
on how many streams run; the LM decodes every sequence identically in any batch. Results are
identical with 1 or 4 embedding streams and with or without the overlap (checked byte for byte).

## Requirements

- A block with **prompt v2** vectors and an **embedder IR with dynamic shapes**
  (`scripts/export_gemma2_dyn.py` → `models/gemma2_text_dyn_int8`, the pack spec's `embed_ir`).
  The same IR embeds the queries (shape `[1, 64]`, as before).
- `hnz.open(…)`: `search(…, rerank=None)` reranks whenever the archive can; `rerank=True` raises
  `ValueError` if it cannot; `rerank=False` gives the P3 results. The response reports
  `rerank: {candidates, decoded, tokens, embedded, ms_stage1, ms_decode, ms_embed}` and, in keyword
  mode with frequent words, `warning`. CLI: `hnz search --rerank`, `hnz eval --rerank`.
- `HNZ_EMBED_STREAMS` (default 1): documents embedded at the same time; the embedder's memory
  reservation grows by 64 MiB per stream.

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
stored document vectors standing in for re-embedding; the run with the real CPU re-embedding is in
the log).

## Latency (cold, sample block)

To be measured on a free machine (`experimentos/bench_p4_latency.py`). With the CPU shared with
the LM training (16-thread Core Ultra 7 265H): ~100-120 prefixes (~6,000-9,000 LM tokens) decoded
in 4-6 s, the re-embedding of 50 prompts hidden under the decoding; 5-9 s per cold query. Target:
~4 s with >= 16 threads or an iGPU (user's decision).
