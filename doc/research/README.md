# Prior research (archive)

`hermes-7a9d842233b2.html` is the saved transcript of the research session that started the
project (October 12-13, 2026). It is kept as the original record and is in Spanish.

What came out of it, and is used throughout the project:

- The size of Hacker News: about 49.3 M items (`maxitem` 50,064,927; about 96.8% of the ids hold
  data; about 86% are comments).
- The starting points: about 24 GB of raw JSON; 12.4 GB for the Parquet dump of
  `open-index/hacker-news`; 9.1 GB for zstd-19 with a 64 KB dictionary per record.
- The first ideas tried later in the experiment log: small compressed blocks, binary and
  quantized vectors (turbovec), BM25, a language model as the compressor, and a cross-encoder
  reranker.

These findings are entry E00 of [the experiment log](../EXPERIMENT-LOG.md), in English, with
everything measured after it.
