# Experiment log

> **Where the cited files are** (reorganization of 2026-10-14): the experiment scripts (`bench_*.py`, `make_eval.py`) in `experimentos/`; their outputs (`results_*.txt`) in `resultados/`; the tools (`pipeline.py`, `download_hn.py`, `prepare_models.py`, `embed_month.py`, `export_*.py`, `compress_embedder.py`, `train_lm.py`) in `scripts/`. They are run from the project root, e.g. `python experimentos/bench_store.py`.

Rule: **every experiment is logged here** (date, script, data, result, conclusion) so as not to
repeat tests. The "49.3 M" numbers are linear extrapolations from the sample month.
Default sample: `data/2024/2024-01.parquet` = **382,681 items** (346,509 comments,
36,093 stories), text 125 MB, ~335 B/item of raw text.

---

### E00 — Prior research (Hermes session, 2026-10-13)
- HN: `maxitem` = 50,064,927; ~96.8% of ids with data → **~49.3 M items**; ~86% comments.
- Raw JSON ≈ 24 GB; Parquet dump of `open-index/hacker-news` = 12.4 GB (252 B/item).
- Baseline: zstd-19 + 64 KB dictionary **per record** = 185 B/rec → **9.1 GB**, 3 µs/read.
- turbovec (TurboQuant) 384d: 2-bit 101 B/vec (4.99 GB), 4-bit 198 B/vec (9.78 GB). 3-bit = same
  cost as 4-bit (padding) → do not use. Needs `calibrate()`. Flat index → it has to be split up.
- An embedding does NOT contain the text (192 B of vector vs 330 B of text; vec2text is lossy).
- Jev / "System One" (TypeSafe): an API model that returns typed decisions (Choice/Score/Noul).
  It does not compress. Useful open source equivalent here: **cross-encoder reranker** (scores candidates).

### E01 — Text storage (`bench_store.py` → `results_store.txt`)
| Variant | B/item | 49.3 M | read 1 block |
|---|---|---|---|
| JSON per record, zstd19+dict (reproduced baseline) | 181.2 | 8.94 GB | 2-3 µs |
| Columnar per record (metadata in delta varints) | 168.4 | 8.30 GB | 3 µs |
| + reversible HTML transformation (94.5% of records) | 166.7 | 8.22 GB | 3 µs |
| Blocks of 16, id order | 125.5 | 6.19 GB | 11-25 µs |
| Blocks of 64, id order | 121.4 | 5.98 GB | 34 µs |
| Blocks of 256, id order | 118.5 | 5.84 GB | 0.13-0.25 ms |
| Blocks of 1024, id order | 115.8 | 5.71 GB | 0.7 ms |
| Blocks of 16 / 64 / 256 / 1024, **order by thread** | 123.9 / 118.0 / 114.4 / 113.1 | 6.11 / 5.82 / 5.64 / 5.58 GB | — |
| zstd-22 + dict, block 256, thread | 114.4 | 5.64 GB | no improvement |
| brotli-11 (16 MB window), block 256, thread | 113.8 | 5.61 GB | 0.54 ms |
| xz-9e, block 4096, thread | 106.6 | 5.26 GB | ms |
| xz-9e ALL together (floor, no random access) | 91.0 | 4.49 GB | — |

**Conclusion:** what pays off most is going from records to **small blocks** (181 → 124 B).
Stronger compressors (zstd-22, brotli, xz) barely contribute. The HTML transformation contributes ~1%.

### E02 — Metadata and text separately (ad hoc command, 2026-10-13)
- Metadata (id, time, parent, type, score, descendants, author id) in delta columns over
  blocks of 1024: **5.9 B/item**. Author dictionary: 0.25 MB (46,084 authors/month).
- Text only (title+body), zstd19+dict per record: 136.6 B/rec (raw 335 B).

### E03 — LM as a compressor (`bench_lm.py`, idea from DeepMind arXiv 2309.10668)
Arithmetic coding bound: bits = Σ −log2 p(token) + 2 closing bytes; each record on its own.
| Model | Records | text B/rec | bits/byte | speed |
|---|---|---|---|---|
| SmolLM2-135M | 300 | 52.9 | 1.26 | (shared CPU) |
| SmolLM2-135M | 100 | 46.3 | 1.28 | 66 ms/rec **encoding** (parallel pass) |
→ with metadata (~6 B) + URL (~3 B): **~55-62 B/item ≈ 2.7-3.1 GB** for all of HN (vs 124 B zstd).
**Pending:** measure the *decompression*, which is sequential token by token (slower), and the
bit-for-bit determinism (encoder and decoder must compute identical probabilities).

### E04 — Word index (BM25) (`bench_lex.py` → `results_lex.txt`)
20.1 M tokens, 184,433 terms, 14.91 M postings (39/doc); the 200 most frequent words = 47%.
| Variant | B/doc | 49.3 M |
|---|---|---|
| All terms + tf, varint+zstd | 51.9 | 2.56 GB |
| All terms + tf, Elias-Fano | 53.6 | 2.64 GB |
| Without 200 stopwords + tf, varint+zstd | **37.0** | **1.82 GB** |
| Without 200 stopwords, boolean, varint+zstd | 31.1 | 1.53 GB |
| Actual tantivy (freq, no positions) | 83.3 | 4.11 GB |

### E05 — Fidelity of compressed vectors (`bench_vec.py` → `results_vec.txt`; Gemma 2 in `results_g2.txt`)
recall@10 against the exact float32 search *of the same model* (300 queries = titles).
Embedding speed on CPU: potion ~800-4450 docs/s; bge-small ~31-35; EmbeddingGemma v1 ~11;
**EmbeddingGemma 2 ~14-17 docs/s** (65 docs/s only with short sentences).
| Model / dim | int8 | 4-bit | binary | bin+rescore100 | bin+rescore300 | PQ 32 B | RaBitQ |
|---|---|---|---|---|---|---|---|
| potion 512d | .994 | .909 | .508 | .876 | .941 | .497 | .676 |
| bge-small 384d | .991 | .857 | .429 | .862 | .954 | .346 | .602 |
| EmbeddingGemma v1 768d | .993 | .919 | .538 | .957 | .996 | .400 | .715 |
| **EmbeddingGemma 2 768d** | .993 | .900 | .479 | .878 | .931 | .408 | .708 |
**Beware:** this metric measures "closeness to float", not quality. Truncating dimensions (MRL)
looks heavily penalized here but barely loses real quality (see E07). PQ of 8-32 B: always bad.

### E06 — Asymmetric scoring and rescore with truncated int8 (ad hoc command)
Query in float, documents in 1 bit (no extra cost): Gemma v1 768d .651, bge 384d .560
(fidelity). Rescore with stored 128-256d int8: limited by the truncation → not worth it.

### E07 — Real QUALITY (`bench_quality.py` → `results_quality_bge.txt`, `results_g2.txt`)
Query = title of a story; relevant = comments of THAT thread. 200 queries, 16,667 docs
(6,667 relevant + 10,000 distractors). P@10 / R@100.
| Variant | potion | bge-small | **EmbeddingGemma 2** |
|---|---|---|---|
| float32 full dim | .474 / .380 | .578 / .433 | **.634 / .496** (3072 B) |
| asymmetric binary full dim | .430 | .538 | **.624 / .497 (96 B)** |
| binary + re-embed top-100 full dim | .471 | .575 | .636 |
| 512d asymmetric binary / + re-embed | — | — | .607 / .639 (64 B) |
| 256d float / asym. binary / + re-embed | .468 / .408 / .470 | .541 / .489 / .573 | .614 / .543 / **.639 (32 B)** |
| 128d float / asym. binary / + re-embed | .432 / .339 / .448 | .475 / .370 / .541 | .502 / .366 / .571 |
| BM25 | .490 / .339 | .490 | .490 / .339 |
| Hybrid RRF (BM25 + binary) | .501 / .388 | **.585** / .449 | .607 / .484 (worse than vector alone) |
**Conclusions:** EmbeddingGemma 2 is the best. Asymmetric binary 768d (96 B) ≈ float (−1.6%).
Re-embed of the top-100 recovers everything even at 256d, but on CPU it costs ~7 s/query → not
viable without a GPU. With Gemma 2, RRF with BM25 is worse: fusion must be weighted, not blind RRF.

### E08 — Tags (1000 k-means topics) + routing + hierarchy (`bench_tags.py` → `results_tags.txt`)
potion embeddings of the 382,681 items (4,450 docs/s). Spherical k-means 1000: median 329 docs
per tag, max. 17,804. Cost: 1 tag/doc = 0.06 GB; 3 = 0.18 GB (49.3 M).
Routing (full corpus of the month, P@10 of the full search = .167):
| 3 tags/doc, search in N | P@10 | % scanned |
|---|---|---|
| N=1 / 3 / 5 / 10 / 20 / 50 | .100 / .124 / .127 / .136 / .144 / .153 | 0.3 / 0.9 / 1.5 / 2.9 / 5.6 / 13.3% |
| Tags only, no vectors | **.004** | — |
Hierarchy 1000→40 areas: there are clear topics (Python/pandas, housing, copyright/DMCA, moderation),
but potion also groups "by tone" ("nice, lovely, glad") → use a better model + names with an LLM.

### E09 — Compression with blocks ordered by topic (`bench_store_topic.py` → `results_topic_*.txt`)
| Order | block 16 | 64 | 256 |
|---|---|---|---|
| id | 125.5 | 121.4 | 118.5 |
| thread | 123.9 | 118.0 | 114.4 |
| topic | 124.5 | 119.6 | 116.0 |
| thread grouped by topic | 123.5 | 117.5 | **113.5** |
**Conclusion:** grouping by topic contributes < 1%. The zstd dictionary already captures the vocabulary.

### E10 — EmbeddingGemma 2 speed per device (`export_gemma2_ov.py`, `bench_embed_ir.py`, `bench_embed_speed.py`)
- The ONNX from `onnx-community/embeddinggemma-2-ONNX` (multimodal) **fails on GPU/NPU**
  (image/video/audio inputs of size 0). Solution: export **only the text part** from PyTorch to
  OpenVINO IR (static shapes, batch 32 × 128 tokens, fp16 weights). Cosine vs PyTorch: 0.99998.
- **Finding:** sentence-transformers loads the model in **bfloat16**, which this CPU emulates → slow.
  You have to force `.float()`.
- Real HN texts (mean 60 tokens, 15.3% truncated to 128):
| Device | docs/s | 49.3 M |
|---|---|---|
| **OpenVINO GPU (Arc 140T iGPU, fp16)** | **168.8** | **81 h (3.4 days)** |
| OpenVINO CPU | 19.9 | 28.6 days |
| PyTorch CPU float32 | 10.7 | 53 days |
| onnxruntime CPU (ONNX fp16) | 4.5 | — |
| NPU (AI Boost) | fails to compile | — |
**Conclusion:** embedding all of HN is viable on the iGPU (~3 days, improvable with length buckets
and int8 weights). The CPU is left free in parallel for the LM compression.

### E11 — Which LM to use as the compressor (`bench_lm2.py` → `results_lm2.txt`)
300 records per month; bits/byte of the text (title+body), B/rec includes 2 B for closing the
encoder; speeds with PyTorch CPU float32 (dominated by framework overhead in small models; a
custom int8 engine will be considerably faster). Tokenizer round-trip: 100% for all of them.
| Model | 2024-01 bpb / B | 2015-06 bpb / B | Decode batch 10 (ms/step) | Encode 49.3 M (PyTorch) |
|---|---|---|---|---|
| pythia-14m | 1.637 / 70.4 | 1.619 / 65.5 | 6.0 | 4.9 days |
| pythia-31m | 1.488 / 64.2 | 1.462 / 59.3 | 6.6 | 6.7 days |
| pythia-70m | 1.401 / 60.6 | 1.374 / 55.9 | 30.0 | 13.7 days |
| pythia-160m | 1.273 / 55.2 | 1.242 / 50.7 | 27.4 | 34.5 days |
| gpt2 (124M) | 1.444 / 62.4 | 1.421 / 57.7 | 51.2 | 42.3 days |
| SmolLM2-135M | 1.217 / 52.9 | 1.187 / 48.6 | 36.7 | 46.4 days |
| SmolLM2-360M | 1.129 / 49.2 | 1.094 / 44.9 | 105.9 | 107.8 days |
**Conclusions:**
- More parameters give better compression, but at an almost linear cost when encoding and decoding.
- Pythia (trained on the Pile, which includes HN up to ~2020) shows no notable memorization in 2015.
- MACs/token calculation (the vocabulary head dominates in small models): pythia-31m ~18 M,
  pythia-70m ~45 M, pythia-160m ~124 M, SmolLM2-135M ~134 M.
- **Initial choice: GPT-NeoX architecture (Pythia family), custom engine in Rust with exact int8
  matmul (deterministic), configurable model.** By default pythia-70m (balance: ~61 B/rec of text,
  estimated decoding ~50-90 ms for a top-10 with the custom engine). Future improvement: train a
  small HN-specific LM (on GPU) that beats SmolLM2-135M with fewer parameters.

### Data notes (2026-10-13)
- Types: 1 story, 2 comment, 3 poll, 4 pollopt, 5 job.
- 2010-06 and 2015-06 have `deleted` rows (no content), non-empty `kids` and raw line breaks
  in `text`; 2024-01 has none of the three → the format must support both cases.
- **Decision:** text, title, url, author, date, score, descendants, parent, poll/parts and `dead`
  are stored losslessly. [Corrected in E25: `deleted` rows ARE stored as metadata, because
  discarding them broke the thread chains.] The `deleted` rows (empty) and the order of `kids` are not stored (the thread
  structure is rebuilt from `parent`; only the order in which HN displayed the replies is lost).

### E12 — Rust core: meta + zstd text (`hnz build/verify/stats/bench`)
1-month segments, blocks of 16, zstd-19 + 112 KB dictionary, metadata in blocks of 1024.
| Month | Docs | B/item | 49.3 M | HTML transformation | Verification |
|---|---|---|---|---|---|
| 2010-06 | 80,363 (+2,366 deleted ones omitted) | 128.0 | 6.31 GB | 35.7% modern + 64.3% old | **100% identical** |
| 2015-06 | 166,390 (+5,246 deleted) | 133.1 | 6.56 GB | 100% modern | **100% identical** |
| 2024-01 | 382,681 | 122.2 | 6.03 GB | 98.2% modern, 1.8% raw | **100% identical** |
- 2024-01 breakdown: text 116.0 B, meta 4.84 B, authors 0.60 B, block index 0.52 B, dictionary 0.30 B.
- Building a month: ~4.5 s (Python: minutes). Uncached random read: ~67 µs/record (1 thread).
- **Two HTML dialects in HN:** old (~2007-2012: `&#38; &#60; &#62;`, quotes/apostrophes/slashes
  not escaped) and modern (`&amp; &#x27; &#x2F; …`). **Truncated links** ("…", 60-95 characters) are
  30-40% of the links → stored as (url, length).

### E13 — Embeddings of a full month on the iGPU (`embed_month.py` → `results_embed_2024-01.txt`)
364,891 items with content (out of 382,681), length buckets of 32/64/128/256 tokens (3.8% truncated to 256):
| bucket | docs | docs/s |
|---|---|---|
| 32 / 64 / 128 / 256 | 110,360 / 105,631 / 91,087 / 57,813 | 565 / 357 / 163 / 81 |
**Total 204 docs/s (29.8 min) → 49.3 M in ~2.8 days** (vs 168.8 docs/s without buckets).

### E14 — Full search in Rust (`hnz search` / `hnz eval` → `results_eval_rust.txt`)
629,434 docs (3 months), query embedding with OpenVINO from Rust (IR fp16, CPU).
Latencies (query "burnout…"): embedding 45 ms, search 6.5 ms, retrieve+decompress 1.0 ms;
model load 3.8 s (once). BM25 only: search 1.3-1.7 ms.
Quality (300 queries = titles from 2024-01, relevant = comments of the thread, k=10):
| Configuration | P@10 | MRR | ms/query |
|---|---|---|---|
| 2-stage vector (Hamming 256 bits → asymmetric 768) | 0.291 | 0.574 | 6.3 |
| exact vector (asymmetric over everything) | 0.293 | 0.577 | 14.5 |
| BM25 | 0.201 | 0.466 | 3.4 |
| **hybrid w=0.15 c=20 (chosen)** | **0.318** | 0.591 | 5.7 |
(full grid in the file; w ∈ 0.05-0.4, c ∈ 5-20: all between 0.304 and 0.318)
Sizes per doc (2024-01): vectors 96 B + bitmap 0.13 B; BM25 49 B (fst 5.2 + postings 42.9 + norms 1.0).

### E15 — Text representation for the LM (`bench_lm_repr.py` → `results_lm_repr.txt`, pythia-70m)
| Representation | 2024-01 B/rec (tokens) | 2010-06 B/rec (tokens) |
|---|---|---|
| a) original HTML | 61.1 (87.1) | 56.8 (77.3) |
| b) compact with control characters | 61.4 (71.4) | 59.6 (72.0) |
| c) "natural" (bound, NOT reversible) | 57.0 (72.2) | 55.3 (72.7) |
| d) reversible: `\n\n` + control markers | 57.7 (72.5) | 56.3 (73.1) |
| **e) reversible: `\n\n` + links as `<url>`** | **57.2 (72.4)** | **55.6 (73.0)** |
**Chosen e):** −6% bytes and −17% tokens (= faster decoding) vs the HTML.

### E16 — LM quantization (`bench_lm_quant.py` → `results_lm_quant.txt`, representation e)
| Model | fp32 | W8A32 | **W8A8** |
|---|---|---|---|
| pythia-70m | 57.21 | 57.42 | **57.94 (+1.3%)** |
| pythia-160m | 52.64 | 52.77 | **53.21 (+1.1%)** |
**Decision:** per-row int8 weights + per-token int8 activations → exact INTEGER matmul: identical
result on any CPU (scalar, AVX2 or AVX-VNNI) = determinism for the arithmetic coder.

### E17 — Custom LM engine in Rust (cold tier) (`hnz lmtest`, model `export_lm.py` → `.hnzlm`)
GPT-NeoX/Pythia W8A8: per-row int8 weights, per-token int8 activations, exact INTEGER matmul;
LayerNorm/softmax/GELU with custom exp/erf and sums in a fixed order; RoPE precomputed in the file.
Binary arithmetic coder (LZMA style, 16-bit probabilities, token = 16 decisions over
the vocabulary tree), minimal termination (zero trimming). Representation "e" (E15).
- pythia-70m exported: 71.7 MB; pythia-160m: 164 MB.
- **Correctness:** 200/200 real records from 2024-01 compressed and decompressed: identical;
  the compressed bytes are the same with the scalar and AVX-VNNI kernels (determinism test).
- **Size:** 60.7 B/record (title+url+text, with header), 1.467 bits/byte, 77.4 tokens/rec
  (zstd in blocks of 16 on the same month: ~116 B/rec of text).
- **Compression speed** (16 threads, one record per thread): scalar 9 rec/s → AVX-VNNI with
  4×32 microkernel: **97 rec/s (7,520 tokens/s)**.
- **Decompression speed** (batch of 10 records in parallel, KV-cache, lockstep):
  | Implementation | ms/batch (median) |
  |---|---|
  | rayon, 16 threads | 1,205-1,865 |
  | custom "spin" pool, 16 threads (oversubscription with rayon) | 17,304 (pathological) |
  | **spin pool, 8 threads (default = half the logical processors)** | **1,033** |
  | spin pool, 6 / 4 threads | 1,083 / 1,164 |
  Per step (10 active): layer matmul 0.9 ms, attention 0.7 ms, vocabulary head 0.9 ms,
  softmax+decoder 0.3 ms. The batch takes as long as its longest record (steps = its tokens).
**Pending:** more efficient kernels (~27 GMAC/s/thread, far from peak), "snippet" mode (decode
only the first ~64 tokens for the result list) and a layered cache.

### E18 — First real COLD segment and more compact BM25 (`hnz build --tier lm`, `hnz rebm25`)
**2010-06 in the LM tier** (pythia-70m): 80,355 records with LM + 8 with fallback zstd, 79.6 tokens/rec,
61.2 B/rec of text; build 1,118 s (72 rec/s, with the CPU shared with the iGPU).
Breakdown per doc: LM text 62.5 B · BM25 56.9 B · metadata 5.4 B → **124.9 B/doc** (without vectors).
Now **BM25 weighs almost as much as the text** → variants on 2024-01 (300 queries):
| BM25 variant | B/doc | P@10 keyword | P@10 hybrid |
|---|---|---|---|
| base (tf, all terms) | 49.1 | 0.223 | 0.316 |
| no tf (boolean) | 39.9 | 0.192 | 0.306 |
| df ≤ 5% | 47.8 | 0.224 | 0.316 |
| **df ≤ 2% (chosen)** | **43.3** | 0.220 | **0.315** |
| no tf + df ≤ 5% | 39.0 | 0.192 | 0.305 |
**Decision (REVERTED in E25):** do not index terms present in > 2% of the documents (−12%, same quality);
term frequency does matter (−3% in hybrid without it).
Embeddings of 2010-06 / 2015-06 on the iGPU: 126-131 docs/s (with the CPU busy with the LM).

### E19 — Vector bits (MRL) on the full month (`hnz index --bits`, `hnz eval`)
With BM25 df ≤ 2%, 300 queries, 382,681 docs:
| Bits | B/doc | P@10 vector | P@10 hybrid | MRR hybrid |
|---|---|---|---|---|
| 768 | 96 | 0.291 | 0.315 | 0.588 |
| **512** | **64** | 0.278 | **0.307 (−2.5%)** | 0.581 |
| 384 | 48 | 0.258 | 0.298 (−5.4%) | 0.572 |
| 256 | 32 | 0.235 | 0.289 (−8.3%) | 0.556 |
**Decision:** 768 bits in the hot tier, **512 in the cold tier**.

### E20 — LM engine optimization (`hnz lmtest`)
- AVX-VNNI kernel 2 rows × 4 tokens with `hadd` reduction: compression +12% (77→86 rec/s),
  decompression −4%; identical bytes (determinism across kernels confirmed).
- Compression profiling (1 thread, 30 records): **attention** was 43% of the time (quadratic on
  long records, ~0.8 GFLOPS). Switch to **transposed keys** [head·dim][position] and blocks of
  32 keys / 64 dims with accumulators in registers → attention 1.0 s → 0.4 s.
  It is a change to the numerical definition → `MATH_VERSION = 2` stored in `lm.json`; a segment with
  another version is rejected with a clear message (it has to be recompressed).
- Result (16 threads): **compression 122 rec/s (9,462 tokens/s)**, +42%; decompression of 10
  records in a batch: **median 531 ms** (previously 656 ms), p90 1,055 ms.
- **Problem detected:** the "spin" pool hogs the CPU when there are OTHER heavy processes (a
  verification took hours at 5 rec/s competing with the compression). Fix: spin only ~20 µs and
  then `yield_now()` for up to 3 ms, then sleep.

### E21 — Final hot segment (2024-01, zstd + BM25 df ≤ 2% + 768 bits)
Verification 100% identical. **261.7 B/doc**: text 116.8 · vectors 96.1 · BM25 43.3 · meta 5.5.

### E22 — turbovec (TurboQuant) vs asymmetric binary index, at equal size (`bench_turbovec_rerank.py`)
Full month 2024-01 (364,891 docs with a vector), 300 queries, vector only (no BM25):
| Index | B/vec | P@10 | MRR |
|---|---|---|---|
| float32 768d (ceiling) | 3072 | 0.294 | 0.563 |
| **asymmetric binary 768 bits (hnz)** | **96** | **0.293** | **0.577** |
| asymmetric binary 512 bits (hnz) | 64 | 0.280 | 0.558 |
| asymmetric binary 256 bits | 32 | 0.235 | 0.514 |
| turbovec 2-bit 256d (calibrated) | ~64 | 0.285 | 0.557 |
| turbovec 2-bit 384d | ~96 | 0.301 | 0.563 |
| turbovec 4-bit 256d | ~128 | 0.277 | 0.543 |
| turbovec 2-bit 768d | ~192 | 0.310 | 0.575 |
**Conclusion:** at equal size turbovec gives +1.8-2.7% P@10 but equal or worse MRR; with 300 queries the
standard error of P@10 is ~0.015 → differences within the noise. The asymmetric binary index (no
training, 2-stage search in Rust) is kept; turbovec remains as a future alternative.
Part (2) of the script (cross-encoder reranker, equivalent to Jev/System One) **never got to run**:
the system stopped the background processes due to lack of RAM.

### E23 — EmbeddingGemma 2 for queries in int8 (`compress_embedder.py`, Intel's NNCF)
Asymmetric per-channel int8 weights: **545 MB → 276 MB**, **~2.5x faster on CPU**, cosine with fp16:
mean 0.99980, minimum 0.99942 (100 queries). `hnz` uses the int8 IR if it exists.

### Incident (2026-10-13 ~06:30): processes stopped due to lack of memory
The system stopped the background v2 recompression of 2010-06/2015-06 (it was at 50k/80k of 2010) and
the reranker benchmark. `out/seg-2010-06` was left incomplete (no manifest: the engine ignores it).
Pending relaunch, one at a time so as not to exhaust the RAM.

### E24 — v2 cold segments fully verified + real server latencies
- v2 recompression (512 bits): 2010-06 in 625 s (129 rec/s, **181.6 B/doc**), 2015-06 in 1,368 s
  (122 rec/s, **183.4 B/doc**: LM text 65.3 · vectors 64.1 · BM25 48.2 · meta 5.8).
- **Full verification with the LM: 80,363 + 166,390 = 246,753 records identical to the original.**
  (verify at 24 rec/s: the batch of 64 waits for its longest record → future improvement: continuous batching).
- Latencies with `hnz serve` (models loaded once; int8 embedder), k=10 (`results_latency.txt`):
| Query | Total | Embedding | Search | Retrieve/decompress |
|---|---|---|---|---|
| hot 2024, hybrid | 49-51 ms | 25-37 | 6-15 | 6-11 |
| hot 2024, keyword | 11-20 ms | — | 2-5 | 9-15 |
| cold 2010 / 2015, snippets | 430 / 326 ms | ~30 | 4-7 | 393 / 284 |
| whole archive, hybrid | 44-49 ms | ~30 | 5-6 | 3-13 |
| cold, full text, 1st time | 601-641 ms | | | 568-603 |
| cold, full text, repeated (L1) | 116 → **46 ms** | | | 74 → **3** |
| cold, full text, after restart (L2) | 655 → **83 ms** | | | 627 → **54** |
| `/item/<id>` cold, uncached | ~200 ms | | | |
- **Bug found and fixed:** the L2 cache was never written on Windows (a file opened in append
  mode does not allow `set_len` → `L2::open` failed and the error was ignored). It is now opened read/
  write and a warning is given if it fails. Also: a cache of story titles and a single LM batch for
  results + thread titles.


### E25 — Adversarial review of the system and fixes (workflow of 141 agents)
6 reviewers in parallel (lossless format, LM/coder, concurrency/unsafe/caches, search,
CLI/server/pipeline, documentation), read-only; 2 verifiers with different approaches tried to
refute each finding (+1 tie-breaker). Result: **65 findings, 60 confirmed, 5 refuted**.
Main fixes (commit "Correcciones de la revisión adversarial" ("Fixes from the adversarial review")):
| Confirmed problem | Fix |
|---|---|
| --by/--type/dead filters applied after trimming candidates (could return 0 results) | applied while generating candidates: exact metadata scan for author and rare types; oversampling + retry for the rest |
| `deleted` rows discarded → broken thread chain (141 comments in 2010-06) | stored as metadata (a bit in meta; 1 byte in the LM tier) |
| BM25 pruning by df in each segment (E18) → terms frequent in one period disappear | no pruning (E18 decision reverted) |
| Tokenizer: "php", "html", "net" as stopwords; "C"/"R" discarded | searchable |
| The docs promised a "per-record CRC" that did not exist | 1 check byte per LM record (`rec_check`) |
| L2 cache: key by name only (another item's text after a rebuild), no header validation or locking | key with `build_id`, header validated, exclusive access; it also did not work on Windows (E24) |
| `retier`: deleted the original after verifying 200 docs; `--seg .` destroyed the segment; `.retier` leftovers served as segments | keeps `<seg>.old`, verifies 2,000 or everything, cleans up if it fails, canonical path; the engine ignores transient folders |
| Overlapping segments (monthly + yearly) → duplicate results and failed `get` | the engine detects them and refuses to open |
| `build` over an existing folder: an interruption leaves a mixed segment | atomic build in `<seg>.building` |
| `k` unbounded over HTTP (exhausts memory) | k ≤ 500 (100 over HTTP) |
| Pool not panic-safe; `HNZ_LM_THREADS=0` divides by zero | `catch_unwind` + re-raise; minimum 1 thread |
| model id with 1/97 byte sampling | full hash (the old id is accepted) |
| A single LM model per archive | one model per segment |
| `pipeline.py`: ~27 GB of RAM when merging a year; 768 bits in the cold tier; year "finished" even with months missing; hot-tier limit relative to the range | streaming merge (memmap) + `hnz index` via mmap; 512 bits in the cold tier; rebuilds if the months change; absolute limit relative to the dataset |
| `make_eval.py` with `>` in PowerShell produces UTF-16 | `--out` in UTF-8; `hnz eval` tolerates BOM |
| Outdated docs ("628 months": there are 239; model tokenizer not generated; x86-64-v3 requirement undocumented) | docs rewritten; `prepare_models.py`; `requirements.txt` |
Also: continuous batching (`decode_stream`) in verify/retier/rebm25 (before, each batch waited for its
longest record: verify at 24 rec/s).
Measurement notes: the E17 and E20 speeds were measured with different samples and CPU loads
(E17 with jobs in parallel); compare only within each experiment.


### E26 — 2nd, 3rd and 4th review, final segments, cross-tier scoring and reranker
**Reviews.** 2nd review of the E25 fixes (same method): 19 findings, **18 confirmed**,
1 refuted. 3rd review (one reviewer on the resulting commit): **4 more**. All fixed:
| Confirmed problem | Fix |
|---|---|
| `build` deleted any existing `--out` folder (e.g. `out/` or `data/`) | refuses if it is not a segment (no `manifest.json`) |
| Non-atomic replacement: a half-finished deletion left a broken segment | the old one is moved aside with a rename (`<seg>.old.<id>`) and deleted at the end |
| Keyword mode + rare author/type filled k with docs that had none of the terms | only docs with signal: bm25 > 0 (keyword), a vector (vector), either of the two (hybrid) |
| The filter decompressed one metadata block per candidate (almost all of `meta.bin` per query) | 1 B/doc flags column in RAM, stored as `meta.flags.zst` (0.1-0.16 B/doc) |
| Global saturation: a segment with < k valid results was not widened if the others added up to k | per-segment retry (×4 up to ×16), only if its list was truncated |
| Without text, hybrid/vector did not return "most recent first" | listing: without text no embedding is computed and it walks from newest to oldest |
| Text with only stop words in keyword mode ("how to") returned the most recent items (3rd) | with text but no usable signal: no results |
| Heap of `cand+1` per chunk of 64K docs (memory with a large k) | bounded by the chunk size |
| `retier/index --bits` not validated: with 1-7 bits `Engine::open` panicked (3rd) | multiple of 64; `vec.bin` is checked against the manifest on open |
| Without a vector, the vector score was -1 in the fusion (3rd) | 0 (neutral), as before E25 |
| `retier` from inside the segment, `.retier.building` not cleaned up, `sources.json` lost, 768 bits in the cold tier | refuses, cleans up, keeps it with the new tier, truncates to 512 bits via MRL (`--bits`) |
| `pipeline.py`: the current month was downloaded incomplete and never updated | compares sizes with the dataset; re-downloads, recomputes embeddings (`.src.json`) and rebuilds the year |
| `pipeline.py` rebuilt (or reverted to zstd) an already cooled-down segment | never because of the tier: `--retier` cools it down; bits according to the actual tier; does not rebuild if already-archived Parquet files are missing |
| `pipeline.py` without network / `IndexError` without Parquet | local Parquet files; the index uses the months in `sources.json` |
| `uv` did not resolve `torch==…+cpu` when mixing indexes | separate `requirements-torch.txt` |
| The docs said the check byte protects snippets and titles | clarified: only full records; for snippets, the model id + `MATH_VERSION` |

**4th review** (workflow of 51 agents: 4 reviewers — pipeline, Rust fixes, validity of the
experiments, documentation — and 2 verifiers per finding + tie-breaker): 20 findings, **18
confirmed**, 2 refuted. Fixes:
| Confirmed problem | Fix |
|---|---|
| `pipeline.py` adopted embeddings without a record, using the size of the freshly re-downloaded Parquet: stale vectors forever | only adopted if they are newer than the Parquet |
| A segment without `sources.json` was rebuilt with whatever local Parquet files remained (losing the archived months) | left untouched; provisional record `seg-YYYY.sources.json` before the build, adopted if the process dies right after |
| A normal run re-downloaded the archived Parquet files and re-embedded (days of iGPU) already complete years | complete year, up to date with the dataset: nothing is downloaded or embedded |
| A network failure silently switched to offline mode (hot/cold boundary based on the local Parquet files) | stops; explicit `--offline` with the last year seen online (`out/pipeline_state.json`) |
| `VecIndex::open` accepted a `vec.bin` of a different width if it was a multiple of the code size: server panic | exact size `n_docs × bytes` and a large enough `vec.has` |
| `vec_cand = Some(0)` panicked; `--c-bm25 0` gave NaN; impossible dates (2015-13) overflowed | candidates ≥ k; `c > 0`, `w ≥ 0`; dates validated |
| Overstated mixbits conclusion (differences on the order of the noise) | paired bootstrap and a more cautious conclusion (below) |
| Reranker: latency without the candidate download and gain without a noise test | end-to-end latency and 95% CI (below) |
| Server latencies from a single query and with a warm cache | medians of 30 distinct queries with a freshly started server and an empty cache (below) |
| Docs: `python` without activating the environment; text 116.8 B/doc (not 116.6); models 0.38 GB (not 0.35); 05 out of date; check byte presented as a guarantee | fixed |

**Final segments** (`results_build_v4.txt`, final binary):
- 2024-01 (zstd): **382,681 records identical** to the original.
- 2010-06 (LM) rebuilt with the final binary: **all files byte-for-byte identical** to the v3
  from the previous binary, already fully verified (**82,729 identical records**, 2,768 s) →
  determinism also across different binaries.
- 2015-06 (LM): 171,636 items (5,246 deleted), 63.0 B/rec of LM text; full verification: **171,636 identical records** (5,971 s, 29 rec/s).
- The two LM months were compressed at the same time (half the threads each): 2010-06 in 1,325 s and
  2015-06 in 2,079 s; one after the other (E24) it would have been ~625 + ~1,370 s, so parallel gains nothing.

**Final measurements** (`results_final.txt`, idle CPU):
- Size (`hnz stats`): hot **267.6 B/doc** (text 116.8 · vectors 96.1 · BM25 49.2 · meta 5.5 · flags 0.1);
  cold **187.0 / 188.2 B/doc** (LM text 64.4 · vectors 64.1 · BM25 53.9 · meta 5.9). The cold BM25
  grows ~5 B/doc from removing the pruning (E25). Projection for all of HN: **≈ 10.0 GB** + 0.38 GB of models (≈ 10.4 GB in total).
- Opening the archive: **180 ms** (previously ~950 ms, building the flags on open).
- Server latencies (`bench_serve.py`): median and p90 of 30 distinct queries per case, with the
  server freshly started and the L2 cache empty (the 1st version of E26 measured a single query with a
  warm cache: the "29 ms" for the whole archive was a cache hit):
| Query (k=10 unless stated) | Median | p90 | Embedding | Search | Text | From the LM tier |
|---|---|---|---|---|---|---|
| hot 2024, hybrid | 25.9 ms | 27.4 | 19.2 | 4.3 | 2.5 | 0% |
| hot 2024, keyword | 3.1 ms | 5.1 | — | 0.6 | 2.4 | 0% |
| whole archive, hybrid (2024 queries) | 103 ms | 158 | 20.1 | 4.1 | 73.5 | 19% |
| whole archive, hybrid (2010 queries) | 264 ms | 311 | 21.7 | 3.9 | 239 | 80% |
| cold 2010 / 2015, hybrid, snippets | 280 / 284 ms | 321 / 310 | 20-24 | 2.8 | 256 / 257 | 100% |
| cold 2010, full text | 525 ms | 1,092 | 20.9 | 2.8 | 502 | 100% |
| whole archive, k=100, snippets | 1,306 ms | 1,846 | 24.3 | 10.0 | 1,271 | 60% |
| type job + keyword (exact scan) | 189 ms | 332 | — | 13.8 | 176 | 77% |
| author, hybrid | 217 ms | 306 | 20.0 | 13.9 | 184 | 73% |
| no text: most recent with filters | 4.0 ms | 6.0 | — | 0.4 | 1.7 | 0% |
| cold full text: 1st time / repeated (L1) / after restart (L2) | 527 / 24 / 53 ms | | | | 503 / 2 / 29 | |
| `/item/<id>` cold, full, no cache | 326 ms | 589 | | | | |
  What is expensive is the LM tier: ~25 ms per cold result in a batch of 10 snippets.

**Quality: why P@10 drops and what NOT to fix.**
- `hnz eval` (2024-01 queries, whole archive): vector 0.255, hybrid 0.278 versus 0.291 / 0.315
  from E14/E19. Control with only the 2024-01 segment: 0.291 / 0.314 → the engine is fine.
- In E14 the cold months had no vectors yet and contributed no distractors. Now they do, and with the
  label "comments of the same thread" a similar thread from 2010 counts as a miss.
- Does mixing 768 bits (hot) and 512 (cold) matter? Simulation with the real embeddings
  (`bench_mixbits.py` → `results_mixbits.txt`), vector P@10 with 3 query sets
  (`make_eval.py`; relevant ones in 2024, 2010 or 2015):
| Scoring of the 512-bit segments | 2024 queries | 2010 | 2015 | mean |
|---|---|---|---|---|
| everything at 768 bits (reference) | 0.265 | 0.197 | 0.234 | 0.232 |
| **cosine renormalized on the MRL prefix (hnz)** | 0.259 | 0.194 | 0.235 | **0.229** |
| calibration by regression (s768 ≈ 0.936·s512 + 0.019) | 0.268 | 0.177 | 0.210 | 0.218 |
| calibration by moments (s768 ≈ s512 − 0.006) | 0.265 | 0.183 | 0.221 | 0.223 |
| without renormalizing the query | 0.292 | 0.065 | 0.070 | 0.142 |
  With a paired bootstrap over queries (95% CI of the P@10 difference versus what hnz does):
  everything at 768 bits +0.003 [−0.001, +0.007] over the 900 queries (not significant); regression
  −0.011 [−0.014, −0.008] and moments −0.006 [−0.008, −0.004] (significantly worse: they gain on the
  2024 queries, +0.009 and +0.006, and lose more on the 2010 and 2015 ones). Conclusion: no linear
  calibration improves on the cosine renormalized per MRL prefix (each prefix is trained as a cosine
  of its own), and having everything at 768 bits would barely change anything. **Lesson: do not
  validate with a single query set whose answers are all in the same tier.**
- Queries on the cold tier with the real system (`hnz eval`, whole archive): 2010-06 vector
  0.193, hybrid 0.208; 2015-06 vector 0.236, hybrid 0.249.
- This metric drops as the archive grows even if search does not get worse: always compare versions
  with the same composition.

**Cross-encoder reranker** (open-source equivalent of Jev/System One; `bench_rerank.py`: reranks the
real hnz top-30 requested from the server):
- 1st run: P@10 0.444 (+59%) → **false**: the text the reranker saw included
  "re: <thread title>" and the query IS that title (label leakage). Fixed: the reranker
  sees only the document itself, like the embeddings and BM25.
- Real result with 95% CI (paired bootstrap; P@10 difference versus hnz) and end-to-end
  latency (median; freshly started server and empty cache):
| Queries | The reranker reads | hnz | MiniLM-L6 (80 MB) | MiniLM-L12 (120 MB) | RRF hnz + L12 | Latency: hnz → with L6 / L12 |
|---|---|---|---|---|---|---|
| 2024-01 | full text | 0.279 | +0.015 [+0.004, +0.027] | +0.020 [+0.008, +0.033] | +0.023 [+0.015, +0.032] | 90 ms → 696 / 1,119 ms |
| 2024-01 | snippets | 0.279 | +0.017 [+0.005, +0.029] | +0.023 [+0.010, +0.035] | +0.024 [+0.016, +0.033] | 91 ms → 595 / 956 ms |
| 2015-06 | full text | 0.249 | +0.017 [+0.005, +0.030] | +0.019 [+0.006, +0.032] | +0.026 [+0.017, +0.035] | 249 ms → 1,212 / 1,616 ms |
| 2015-06 | snippets | 0.249 | −0.011 [−0.025, +0.003] | −0.013 [−0.026, +0.001] | +0.001 [−0.007, +0.010] | 264 ms → 668 / 848 ms |
- Interpretation: with the full text the reranker improves significantly (+6-10% relative; better in
  RRF fusion with hnz's ranking) but multiplies latency by 5-7. With the 64-token snippets
  (the cheap option) it does not improve in the cold tier: the snippet is too short to judge (in 2024
  it makes no difference because the hot tier returns the whole text). **Decision: not by default**;
  an optional "precise" mode would have to decompress the full text of the 30 candidates.
- Larger rerankers (jina v1 tiny/turbo, bge-reranker-v2-m3 int8): the download failed due to network timeouts.

**Incidents:** huggingface_hub's `hf_xet` downloader dies mid-download on this
machine (exit 127) → `HF_HUB_DISABLE_XET=1` in the scripts that download.

### E27 — Compact profile: how much it can shrink with ≤ 10 s per query (`results_e27.txt`)
The user asks for **4-5 GB for all of HN**, accepting up to 10 s per query (decompression included)
and slow compression. Idea: spend the time on **reranking, not on storing**: a minimal index gives
100-200 candidates, which are re-scored with their own decompressed text (re-embedding + exact BM25);
the text has to be stored anyway, so it acts as a free "second index".
(b) Word index with only the rare terms (2024-01, `hnz rebm25 --max-df`, `hnz eval`):
| Indexed terms (max. df) | B/doc | P@10 words only | P@10 hybrid |
|---|---|---|---|
| all (today) | 49.2 | 0.227 | 0.314 |
| all, without frequency (tf) | 40.0 | 0.192 | 0.306 |
| ≤ 2% | 43.4 | 0.223 | 0.312 |
| ≤ 0.5% | 32.6 | 0.212 | 0.324 |
| ≤ 0.1% | 21.1 | 0.174 | 0.322 |
| **≤ 0.02%** | **13.2** | 0.141 | **0.320** |
| ≤ 0.02%, without tf | 11.1 | 0.129 | 0.318 |
| ≤ 0.005% | 9.1 | 0.097 | 0.304 |
  In hybrid search, what the word index contributes is the rare terms (names, identifiers,
  products): with only those (−73% in size) the quality is the same or slightly better. Searching
  only by frequent words gets worse: in a compact profile they would be scored on the fly over the
  decompressed candidates.
(c) Larger LM (200 records from 2024-01, `hnz lmtest`): pythia-160m **56.2 B/rec vs 61.6**
  (−8.8%) but it decompresses 2.3 times slower (10 full records: 1.8 s vs 0.78 s) and
  compresses at 35 rec/s vs 76.
(a) Vectors with fewer bits + reranking the top-K by re-embedding the decompressed text
  (`bench_rescore.py`: 611,640 docs from the 3 months, 900 queries, vector only; float ceiling 0.230):
| Bits/doc | B/doc | No reranking | Top-100 | Top-200 |
|---|---|---|---|---|
| 768 | 96 | 0.232 | 0.231 | 0.230 |
| 512 | 64 | 0.218 | 0.232 | 0.231 |
| 384 | 48 | 0.201 | 0.231 | 0.230 |
| **256** | **32** | 0.183 | 0.225 (−0.008) | **0.228 (−0.004, not significant)** |
| 192 | 24 | 0.151 | 0.217 | 0.221 (−0.011) |
| 128 | 16 | 0.100 | 0.178 | 0.193 (−0.039) |
**Design panel** (workflow of 4 agents: 3 independent designs — text, indexes, query — and a
judge that checks them against the log): central estimate ~4.7 GB (4.2-5.3) with queries at
p50 ~4 s (decompressing the first 128 tokens of ~100 candidates ~3 s + re-embedding on the iGPU),
which depended on 3 hypotheses: 18 B IVF-PQ codes, the thread title as LM context, and an LM
trained on HN.

### E28 — The panel's cheap hypotheses (`results_e28.txt`)
- **Product quantization (faiss) vs binary, to generate candidates** (`bench_ivfpq.py`):
| Code (256-dim MRL prefix) | B/doc | No reranking | Top-100 | Top-200 |
|---|---|---|---|---|
| asymmetric binary 256 bits (hnz) | 32 | 0.183 | 0.225 | **0.228** |
| OPQ16 + PQ16 | 16 | 0.079 | 0.158 | 0.177 |
| IVF4096 + PQ16 of the residual | 18 | 0.120 | 0.186 | 0.202 |
| OPQ32 + PQ32 | 32 | 0.140 | 0.211 | 0.219 |
  **Fails:** with EmbeddingGemma 2 the sign bits are the best per byte (as in E05); 18 B PQ
  falls below the 192-bit binary (0.221 with top-200).
- **Thread context for the LM** (`bench_titlectx.py`, pythia-70m, 300 comments per month): with the
  story title, **+0.4-0.5% (worse)**; with the title and the first 64 tokens of the parent
  comment, −2.2% (2024) and −2.8% (2015), but each candidate would also require decompressing its
  parent (twice the work when reranking). **Not worth it.**
- Estimate from the measurements (all of HN): all LM + binary 256 + rare terms **≈ 6.0 GB** (same
  quality); with binary 192 **≈ 5.6 GB** (−1%); with shorter record headers, vocabulary
  shared per year, int4 embedder and no vector for very short comments **≈ 5.0-5.2 GB** (−1-2%).
  Going below 5 GB depends on an **LM trained on HN** (pending, E29): to train it and evaluate it
  on unseen months, the whole dataset is downloaded (`download_hn.py`: resumable byte by byte,
  pinned dataset version and SHA-256 of each file).

### E29 — LM trained on HN: −9% text at the same cost (`train_lm.py`, `train_lm.log`)
Continued pretraining of pythia-70m on HN text; same architecture, tokenizer and
cost per token → the Rust engine and `export_lm.py` work unchanged and it decompresses just as fast.
- **Data:** `hnz lmdump` (the EXACT text the cold tier encodes: title, url and body in the
  "e" representation) of 1 in 100 items from all months except the evaluation ones and their neighbors
  (2010-05..07, 2015-05..07, 2023-12..2024-02): 456,059 records, **36.8 M tokens** (~1% of HN). The
  full dataset was downloaded earlier with `download_hn.py` (239 months, 12.42 GB, SHA-256 verified).
- **Training:** Arc 140T iGPU with PyTorch XPU (separate environment `.venv-xpu`), mixed fp16 with
  loss scaling; **bf16 breaks pythia** (loss 5.71 vs 4.00 in fp32; fp16 gives the same as
  fp32). lr 2e-4, 200-step warmup and cosine, batches of ~8,000 tokens, 5,000 steps (one
  pass) at ~7,300 tokens/s: **93 minutes**. The CPU trains at ~1,000 tokens/s.
- **Evolution** (bits/byte in float32 over 1,000 records of each month NOT seen):
| Step | 2024-01 | 2015-06 | 2010-06 |
|---|---|---|---|
| original pythia-70m | 1.429 | 1.397 | 1.428 |
| 500 | 1.500 | 1.466 | 1.526 |
| 1,000 | 1.384 | 1.356 | 1.411 |
| 2,500 | 1.338 | 1.313 | 1.370 |
| 4,000 | 1.304 | 1.280 | 1.336 |
| **5,000** | **1.296 (−9.3%)** | **1.272 (−9.0%)** | **1.326 (−7.1%)** |
  At first it gets worse (raising the lr on an already trained model throws it off) and it recovers
  from step 1,000 on; at the end it was still improving.
- **In the real compressor** (int8, `hnz lmtest`, the same 300 records of each month; 0 errors):
| Month | original pythia-70m | HN LM | Savings |
|---|---|---|---|
| 2024-01 | 59.8 B/rec (1.492 bits/byte) | **53.4 B (1.333)** | **−10.7%** |
| 2015-06 | 70.6 B (1.334) | **64.3 B (1.216)** | **−8.9%** |
| 2010-06 | 58.0 B (1.482) | **53.4 B (1.365)** | **−7.9%** |
  All of HN in the LM tier: **3.13 → 2.84 GB**. Decompression just as fast (same model).
- **Beware:** the LM file is part of the archive, like a dictionary: without it (or with another one)
  it cannot be decompressed; each segment's manifest stores its id (hash) and it is checked on open.
- **Compact profile from the measurements (all of HN):** everything in the LM tier with the HN LM +
  256-bit binary vectors with reranking + index of rare terms = **5.74 GB** (today's quality); with
  192 bits and without tf = **5.24 GB** (−1%); with shorter record headers, vocabulary shared per
  year and int4 embedder (not measured) **≈ 4.96 GB**. More training data (5% of HN, ~7 h on the iGPU)
  should bring the text down somewhat further.

### E30 — Library for backends: P1 (`import hnz`) and first P0 measurements
Design in [06-libreria.md](06-libreria.md). User decisions (2026-10-14): Windows; read-only
snapshot; one process with several queries at once; compact profile (~5 GB, ~4 s cold); **the
server may not have an iGPU**; optional persistent cache and prewarming; frequent words are
searched for among the semantic candidates.
- **E30b — embedding documents on CPU only** (`bench_cpu_embed.py` → `results_e30b.txt`; 1,024
  real documents from 2024-01, 64 tokens on average, padded to 128; THROUGHPUT mode):
| IR | 16 threads | 8 threads (what is left free while the LM decompresses) |
|---|---|---|
| fp16 32×128 | 21.3 docs/s | 15.9 docs/s |
| **int8 32×128** (NNCF) | **38.2 docs/s** | **28.2 docs/s** |
  Without an iGPU, re-embedding ~115 candidates would cost ~4 s on top of the LM: the CPU profile
  will need fewer candidates (~50, with 384-bit vectors, +0.8 GB) or grouping by length (not
  measured); to be decided with the E30a/E30d measurements.
- **P1 — Python library on top of the current format:**
  - `hnz::api`: thread-safe facade (checked at compile time that the engine is `Send + Sync`); the
    OpenVINO embedder lives on its own thread (its objects cannot be shared between threads) and
    the LM decompresses one query at a time (its thread pool is global).
  - `hnz-py` crate (PyO3 0.29, `abi3` wheel for Python ≥ 3.10, GNU toolchain without Visual Studio)
    and `hnz` package: `open(path, ram=…)`, `search(…)` with dataclasses or `raw=True`, `get(id)`,
    `embed`, `stats`; releases the GIL; the OpenVINO DLLs are taken from the pip package.
  - Tests (`hnz/py/tests`, 9): same results as the CLI (hybrid, keyword and with filters);
    `get()` identical to the Parquet on 75 records from the 3 months (including the LM tier); 48
    mixed operations on 8 threads = the same result as one at a time; argument errors → `ValueError`.
  - Latency from Python: **33 ms** median (2024, hot; the Python layer adds ~0.2 ms);
    **45 queries/s** with 8 threads (the embedder, one query at a time, is the bottleneck).
- **Adversarial review of P1** (workflow of 28 agents: concurrency and FFI, Python API,
  packaging; 2 verifiers per finding): 11 findings, **9 confirmed** (6 distinct problems), 2
  refuted. Fixed, each with a test (13 tests in total, all passing):
  - the LM lock wrapped the whole `fetch`/`get`: a heavy LM-tier query blocked all the
    others → now it is a global lock only around decompression (a `get()` from the zstd tier
    responds in < 0.5 s while an LM batch runs);
  - `stats()` waited holding the GIL while the embedder was loading → without the GIL and without
    blocking; the embedder is loaded once (OnceLock) and a failure is not retried on every query (30 s);
  - `ov_libs` was not honored (the openvino-rs finder checks INTEL_OPENVINO_DIR and other variables
    first) and the PATH grew on every load → `openvino_c.dll` is loaded by full path once per process;
  - a `datetime` in since/until lost its time and time zone → now it is the exact instant in UTC;
  - an integer was taken as a year without checking (a timestamp gave empty or unfiltered
    results) → only years 1970-2100; the engine rejects years outside 1-9999;
  - opening a folder with no segments "worked" as an empty archive → clear error.
- **E30a — reranking by re-embedding only the BEGINNING of each candidate** (`experimentos/embed_prefix.py`
  + `experimentos/bench_rescore.py g2p128` → `resultados/results_e30a.txt`): embeddings of the prefix of
  128 LM tokens (full title and url + the first 128 tokens of the body; truncates 16% of the
  documents), used both for the binary codes and for reranking, as in the compact
  profile. 611,626 docs, 900 queries, vector only:
| Bits | No reranking | Top-50 | Top-100 | Top-200 | (full text, E27a, top-100) |
|---|---|---|---|---|---|
| 512 | 0.214 | 0.226 | 0.227 | 0.226 | 0.232 |
| 384 | 0.198 | 0.223 | **0.227** | 0.226 | 0.231 |
| **256** | 0.181 | 0.217 | **0.221** | 0.223 | 0.225 |
| 192 | 0.148 | 0.202 | 0.213 | 0.218 | 0.217 |
  Ceiling (exact float): 0.225 with prefixes vs 0.230 with full text. **Using only the prefix
  costs ~0.005 of P@10 (~2%) in every combination**; the bits / number of candidates relationship
  does not change. With CPU alone (E30b: ~28 docs/s re-embedding with 8 threads while the LM uses
  the other 8): 256 bits + top-100 ≈ 4-4.5 s cold (~5.5 GB); 384 bits + top-50 ≈ 2-2.5 s (~6.3 GB).
- **E30d — 1st vector stage at the scale of all of HN** (`hnz/examples/stage1.rs` →
  `resultados/results_e30d.txt`): the real 256-bit codes from the 3 months replicated up to
  **49.7 M documents** (1.59 GB; 8 bits flipped at random in each copy).
  - With the selection at the time (a heap of `cand` candidates per chunk of 65,536 docs, and
    sorting everything at the end): **545-695 ms** with 20,000 candidates (~760 chunks × 20,000
    entries to sort) and 83-111 ms with 2,000. The scan itself was already close to the memory limit.
  - **New selection by histogram of Hamming distances** (integers 0-256): a sample of ~65,000
    docs estimates the cutoff, a single pass collects what falls below it and counts the exact
    histogram, and only if the estimate fell short is there a 2nd pass. **Same set and order** as
    with heaps (checked on the test bench; `hnz eval` gives exactly the same figures).
  - Result: selection **37 ms** (vs 545 ms with 20,000 candidates); full search (stage 1
    + asymmetric scoring of the top-100) **~53 ms with 8 threads and ~40 ms with 16** over the 49.7 M
    documents, with 2,000 or with 20,000 candidates. It also improves the current fast profile.
- **E30c — the full compact query and fusion tuning** (`experimentos/bench_fusion.py` →
  `resultados/results_e30c.txt`; the engine's tokenizer exposed to Python as `hnz.terms` /
  `hnz.index_text`): candidates = top-100 by 256-bit vectors + top-20 from the index of rare
  terms (df ≤ 0.02%, without frequency); reranking = cosine of the 128-token prefix embedding +
  w·b/(b+c), with b = BM25 of **all** the query words over the prefix text (global IDF,
  full-document norm). 900 queries; P@10 / MRR:
| | 2024 | 2010 | 2015 | mean P@10 |
|---|---|---|---|---|
| Today (full index and text, `hnz eval`, 3 months) | 0.278 / 0.557 | 0.208 / 0.482 | 0.249 / 0.516 | **0.245** |
| **Compact, best fusion (w = 0.15, c = 40)** | 0.280 / 0.566 | 0.192 / 0.466 | 0.242 / 0.492 | **0.238** |
| Compact without BM25 when reranking (w = 0) | 0.264 / 0.538 | 0.183 / 0.449 | 0.226 / 0.475 | 0.225 |
  Fusion is flat between w = 0.10 and 0.20 (0.232-0.238). **The compact profile loses ~3% of P@10 on
  average** (2024 the same, 2015 −3%, 2010 −8%): within what the design anticipated (−3% to +1%).
  Knobs to recover quality in P4: more candidates from words (K_b) or from vectors, or tf in the
  index of rare terms.
- **E30e (size of the global index of rare terms):** to be measured for real when built in P3; with
  the global threshold of 0.02% (proportional to the monthly one, E27b), ~0.4-0.5 GB is estimated
  for all of HN.
- **P0 done.**

### E31 — P2: RAM governance (`hnz.open(path, ram=…)`)
*(From this entry on, the log is written in English, as the user asked on 2026-10-14.)*
Full description and API: [07-ram.md](07-ram.md). Everything hnz holds in the process stays within
`ram`; what does not fit is read from disk, with identical results.
- **Own ledger** (`hnz/src/mem.rs`): classes models / index / cache / working, RAII charges, peak.
  Opaque parts (OpenVINO, tokenizers, LM weights) measured with the OS counters while loading.
- **Planner** (`plan.rs`): floor = models + fixed structures + minimum pools + minimum cache
  (`BudgetError` below it); then the pools (≤ 5 % each), a small cache share, vectors → metadata →
  BM25 postings (newest segments first), cache up to 25 %, compressed text, the rest to cache.
- **No more mmap** (`blob.rs`): each file is resident (a `Vec`, charged) or read with positional
  reads (through the OS file cache, outside the process). BM25 cursors read one 128-doc block at a
  time; vectors are scanned in windows of 8,192 docs per thread; metadata/text one block/record.
- **LM batch under a cap** (`decode_records_pool`): continuous batching; each record's bound = tokens
  × 24 KiB × 7/6 + per-step buffers (the KV cache can never grow past it, `KvCache::with_limit`),
  finished records free their memory at once; the batch also reserves its threads' sampling
  scratch. Minimum pool: the longest record (2,048 tokens) + scratch ≈ 63 MiB.
- **Caches** bounded in bytes (two generations ≈ LRU). The L2 disk cache is now **opt-in**
  (`cache_dir=`): opened after planning with its RAM index capped (newest entries kept) and scanned
  header by header (before, the whole file was read into RAM at open).
- **Admission** (FIFO; `max_concurrent`=8; working pool; `timeout`=30 s → `BusyError`): a search
  declares 2 MB + k × (8 KB | 64 KB) + its candidate lists (first round of each segment it touches,
  as many at once as worker threads); widening a round asks for the extra memory without waiting.
- **Limits:** query text ≤ 8 KB; BM25 uses at most the 64 rarest query terms that exist.
- **Embedder:** its load peak stays reserved until it loads, then its measured size is charged and
  the rest goes to the caches. Tokenizer without its per-thread BPE cache, parsed *before* OpenVINO
  compiles the model.
- **E31a — embedder memory** (int8 EmbeddingGemma 2: 272 MB of weights + 32 MB tokenizer): parsed
  after the compile, the tokenizer's transient landed on the compiled model: **+746 MB peak**,
  +470 MB steady. Parsed first: **peak = steady = +447 MiB**. Reservation estimate 1.4 × weights +
  4 × tokenizer + 32 MiB = 519 MiB. The sample archive's minimum went from 955 MiB to **712 MiB**.
- **E31b — process memory vs `ram`** (sample archive, 8 clients with mixed queries, 90 s per budget;
  working-set peak above the baseline before opening, as Windows reports it):

| `ram` | Loaded into RAM | Process peak / `ram` | Ledger peak |
|---|---|---|---|
| floor + 24 MiB (736 MiB) | nothing | +586 MiB / 736 = **0.80** | 705 MiB |
| floor + 120 MiB (832 MiB) | vectors, metadata, postings | +664 / 832 = 0.80 | 789 MiB |
| 2 GB | everything (146 MiB) | +728 / 1,907 = 0.38 | 933 MiB |
| None | everything | +728 MiB | 1,254 MiB |

  **Acceptance (P2): 8 clients for 30 minutes at the minimum budget** — first build (minimum 963
  MiB): 29,088 operations, 0 errors, peak 0.84 × `ram`; final build: 28,024 operations, 0 errors, peak **0.81 × `ram`** (736 MiB).
- **Regression:** the new CLI against the previous binary (mmap, HEAD 3f2888a): `hnz eval` identical
  on the 3 months and **83 searches identical** (ids, text, HTML, scores, thread titles), before and
  after the review fixes. In the library, everything-on-disk with the minimum LM pool gives exactly
  the same results as everything-in-RAM (36 searches, 3 listings, 4 gets, k=500 with a type filter).
- **Adversarial review** (workflow of 21 agents: 5 lenses — ledger completeness, concurrency, disk
  mode, LM batch and planner, API/Python — and a skeptical verifier per finding, several with
  allocator-level measurements): 16 findings, 12 confirmed = **9 distinct problems**, 4 refuted.
  All 9 fixed, with tests:
  - query text unbounded: a 0.8 MB query opened ~250k BM25 cursors (~300 MB) while declaring 2 MB →
    queries ≤ 8 KB and at most the 64 rarest existing terms (unknown words cannot crowd out the
    real ones — found while testing the fix);
  - search working memory under-declared for large k / type filters (candidate lists of every
    segment in parallel) → declared per segment; widening asks for its memory;
  - the embedder tokenizer's per-thread BPE cache kept up to 10,000 past queries (30-85 MB) → off;
  - author lists counted at about half their real size (Vec slack + allocator overhead) → one buffer
    + offsets per segment (exact, and about half the RAM);
  - persistent cache emptied on reopen when it held more entries than the new cap, and its whole
    index loaded before planning (ledger above the limit while opening) → opened after the plan,
    newest entries kept;
  - admission not FIFO (a big request could starve behind small ones) → FIFO;
  - a huge `timeout` panicked → validated (`ValueError`);
  - `BudgetError`/`BusyError` not picklable (module `_hnz`) → module `hnz`.
  Refuted: slots held while waiting for the LM lock (only under overload, the documented contract);
  a cache shrink overshoot (bounded by the original share; hardened anyway); disk-mode files no
  longer blocking in-place rewrites (outside the immutable-snapshot contract — denying writes would
  let a concurrent `rebm25` leave a half-written segment); the LM's sampling scratch (covered by the
  static pool; hardened anyway: counted as batch overhead and released on host threads).
- Also found while writing: a lock-order inversion between the two RAM caches (fixed before the
  review); the LM bound needs 7/6, not 1.05, for the moment a layer's KV cache grows; an MXCSR guard
  — the LM entry points set the default floating-point state (round to nearest, no FTZ/DAZ) and
  restore the caller's, so a host library built with fast-math cannot break lossless decompression;
  `hnz index` no longer loads the segment's text.
- Tests: Rust 21 (GenCache bounds, admission timeouts and FIFO, planner, BM25 on disk, L2 index cap
  and reopen, author list, FP guard…); Python 13 (P1) + 11 (P2: `BudgetError`, disk == RAM, memory
  report, `BusyError`, persistent cache incl. reopening with a small budget, request limits,
  k=500 with filters, ≤ 1.03 × `ram` with 8 clients at two budgets).
- Note: the P1 test comparing the library with the CLI now tolerates a record that the CLI's disk
  cache already holds in full (`partial=False`) where the library decoded a snippet — existing
  behavior, exposed because the library no longer locks the CLI's cache file.
- **P2 done.** Next: P3 (compact block: container, `hnz pack`, global index of rare terms).
