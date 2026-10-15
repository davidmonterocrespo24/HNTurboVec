# Search (indexes)

Numerical details in [BITACORA.md](BITACORA.md) (E04-E07, E10, E13, E14, E18, E19, E22, E25, E26).

## What we have learned

1. **Measure real quality, not just fidelity.** "recall vs float" heavily penalizes quantization and
   truncation; with real labels (title → comments of the same thread) the loss is minimal.
2. **EmbeddingGemma 2 (Google, Oct-2026) is the best model tested**: P@10 .634 in float on the
   small set, versus .578 for bge-small and .474 for potion.
3. **Asymmetric binary** (document at 1 bit/dim, query in float): 768 bits = 96 B/doc, almost the
   same as float (E07, E22). On the full month (E19): 768 bits hybrid P@10 .315; 512 bits (64 B)
   .307; 384 bits .298; 256 bits .289 → **768 in the hot tier, 512 in the cold tier**.
4. **2-stage search** (Hamming over 256 bits → asymmetric): loses 0.7% and is 2.3x faster than the
   full asymmetric one (E14).
5. **PQ (8-32 B) and RaBitQ** (faiss) do not beat asymmetric binary per byte. **turbovec**
   (TurboQuant) at the same size is tied within the noise (E22).
6. **Compact BM25**: FST + BitPacker4x postings + 1 B norms, no positions: ~49 B/doc versus 83 for
   tantivy. **Terms are not pruned by df**: per-segment pruning (E18) made a period's frequent terms
   (e.g. "covid" in 2020) disappear for keyword search (E25).
7. **Fusion** cosine + 0.15·bm25/(bm25+20): +9% P@10 over vectors only (E14).
8. **Embeddings at scale**: text-only EmbeddingGemma 2 exported to OpenVINO on the **Arc iGPU**:
   ~204 docs/s with length buckets → all of HN in ~2.8 days (E13). Queries on CPU with int8 weights:
   ~25-40 ms (E23).
9. **Filters** (author, type, dead, dates) are applied when generating candidates: author and rare
   types scan the metadata of the range (exact); the rest is oversampled and filtered in score
   order with a 1 B/doc flags column in RAM (`meta.flags.zst`), retrying only the segment that
   falls short (E25, E26). Without text: a listing of the most recent items.
10. **Mixing 768- and 512-bit segments needs no calibration**: the cosine renormalized at each MRL
    prefix is +0.003 [−0.001, +0.007] P@10 away from having everything at 768 (not significant);
    calibrating (regression or moments) is significantly worse overall: it gains on the queries
    about 2024 and loses more on those about older years (E26).
11. **Validate with queries whose answers lie in different tiers** (2024, 2010, 2015): with a single
    set, any bias against old content looks like an improvement (E26).
12. **The title → thread metric drops as the archive grows** (similar threads from other years count
    as misses): compare versions with the same archive composition (E26).
13. **Cross-encoder reranker** (MiniLM 80-120 MB, equivalent to Jev/System One): with the full text
    of the 30 candidates, +0.015-0.026 P@10 (significant; better in RRF fusion with hnz's ranking)
    at a cost of 0.6-1.6 s per query; with the 64-token snippets it does not improve in the cold
    tier → not by default. Beware of label leakage: the reranker must see only the document itself,
    not the thread title (E26).

## Pending
- Larger rerankers (jina, bge-reranker-v2-m3 int8): download failed due to the network (E26).
- Evaluate with keyword queries and questions (currently only titles → thread).
