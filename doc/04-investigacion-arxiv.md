# Research on arXiv (Google / Microsoft priority)

arXiv IDs verified against the arXiv API (export.arxiv.org) on 2026-10-13. Figures marked with ≈
are from memory of the papers and were not re-read.

## What IS useful for this project

| Technique | Paper | Who | Idea | Use here |
|---|---|---|---|---|
| TurboQuant | 2504.19874 | Google Research 2025 | Random rotation + optimal scalar quantizer + 1-bit QJL residual; no training | Basis of **turbovec** (2/4-bit index). Evaluated in E00 |
| Matryoshka (MRL) | 2205.13147 | Google + UW 2022 | Truncatable embeddings | EmbeddingGemma 2 comes with it: 768→512→256→128 |
| EmbeddingGemma | 2509.20354 | Google 2025 | 300M model, MRL, robust to quantization | v1 evaluated (E05/E07) |
| **EmbeddingGemma 2** | (model card, 2026-10-06) | Google DeepMind | 270M text (130M transformer + 140M table), MRL, 8K ctx | **Chosen** (best quality, E07) |
| RaBitQ / ext-RaBitQ | 2405.12497 / 2409.09913 | NTU | 1 bit/dim + 2 floats, unbiased estimator | Tested in faiss (fidelity, E05): +7-10% at the same dimension but not better per byte than asymmetric binary; not measured on real quality |
| Language Modeling Is Compression | 2309.10668 | Google DeepMind 2023 | LM + arithmetic coding compresses better than gzip/zstd | **Text cold tier** (E03) |
| Training LLMs over Neurally Compressed Text | 2404.03626 | Google DeepMind 2024 | "Equal-info windows": encoded blocks that can be decoded separately | Random-access design with an LM |
| FineZip / LLMZip | 2409.17141 / 2306.04050 | — | Practical LM compression (batching, fine-tuning) | Batching ideas |
| Inverted index compression | 1908.10598 (Pibiri & Venturini) | — | Partitioned Elias-Fano, BP128 ≈ 2-5 bits/docID | Compact BM25 (E04) |
| SPANN / SPFresh | 2111.08566 / 2410.14452 | Microsoft | Centroids in RAM + lists on disk | Partition-based routing (E08) |
| Theoretical Limitations of Embedding-Based Retrieval | 2508.21038 | Google DeepMind 2025 | Small vectors cannot represent every top-k | Keep BM25 alongside vectors |
| GraphRAG | 2404.16130 | Microsoft 2024 | Hierarchical entity/community graph with per-level summaries | Topic exploration layer (E08) |

## What is NOT useful (and why)

| Technique | Paper | Reason |
|---|---|---|
| DSI / DSI++ (model = index) | 2202.06991 / 2212.09744 (Google) | Does not scale: on 8.8 M passages it falls below dual encoders (2305.11841); does not return verbatim text |
| NCI / MEVI | 2206.02743 / 2309.13335 (Microsoft) | Same: ~320k docs |
| FM-index / r-index / Succinct | 0712.3360 / 1705.10382 | 25-40% of the non-repetitive English text: worse than zstd in blocks |
| ColBERTv2 / PLAID / XTR / MUVERA | 2112.01488 / 2205.09707 / 2304.01982 / 2405.19504 | Multi-vector: ≥ 1 KB/doc |
| SPLADE / uniCOIL | 2109.10086 / 2106.14807 | 2-4× the size of BM25 |
| PQ / Distill-VQ / RepCONC | 2204.00185 / 2110.05789 | 8-32 B PQ measured: very low quality (E05) without training the encoder |
| Gist tokens / ICAE / xRAG / vec2text | 2304.08467 / 2307.06945 / 2405.13792 / 2310.06816 | Lossy compression: does not recover verbatim text |
| BitFunnel (Bing) | SIGIR'17 | Bit-sliced signatures: fast, not smaller |

## Jev / "System One" (TypeSafe AI)
Non-autoregressive API model (early access Sept-2026) that returns typed decisions
(Choice / Score / Noul). **It does not compress and is not an embedding model.** Its equivalent
role in an open source system is a **cross-encoder reranker** (scoring candidate relevance) or a
zero-shot classifier (tagging). See the reranker experiment in the log.
