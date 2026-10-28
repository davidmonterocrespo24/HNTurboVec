# All of Hacker News in 5.3 GB, searchable on a laptop

*Every story, comment, job and poll ever posted to Hacker News — 49.3 million items, October 2006
to August 2026 — losslessly compressed into one 5.34 GB read-only block, with hybrid semantic +
keyword search, filters by date, type and author, and byte-exact retrieval. It runs on a laptop
CPU within a RAM budget you choose (2 GB is plenty), from Python, over HTTP or from the command
line.*

![All of HN on disk](img/size.svg)

## The goal

The constraints were simple to state:

- **All of HN**, not a sample: ~49.3 M items.
- **The smallest disk footprint possible.** Building can take days; daily use has to be fast.
- **Search like a search engine:** a free-form question, keywords, *k* results, date ranges.
- **Lossless:** every result's original HTML, byte for byte.
- **One ordinary laptop** (Intel Core Ultra 7 265H, 31 GB RAM, an integrated GPU, no NVIDIA).

The starting points: the raw JSON from HN's API is ~24 GB; the
[open-index/hacker-news](https://huggingface.co/datasets/open-index/hacker-news) Parquet dump is
12.4 GB; the best general-purpose baseline with random access, zstd-19 with a trained dictionary
per item, is 9.1 GB — for the text and metadata alone, with no search at all.

The result is 5.34 GB *including* a semantic index, a word index and both models. Here is how,
with the numbers from the [experiment log](BITACORA.md) (36 entries; every claim below has one).

## 1. Records → blocks: the cheap 30 %

Compressing each item on its own wastes the redundancy between neighbours. Grouping items into
small blocks before zstd took the text + metadata from **181 to 124 bytes per item**. Stronger
compressors barely moved it: zstd-22, brotli-11 and xz-9e on blocks all land at 106-114 B, and
even xz over *the whole month in one stream* (no random access at all) only reaches 91 B.

That floor said general-purpose compressors were done. What they do not know is English, HTML and
the way HN people write.

## 2. A language model as the compressor

[Language Modeling Is Compression](https://arxiv.org/abs/2309.10668) (DeepMind, 2023) makes the
point formally: an LM that predicts the next token well, driving an arithmetic coder, is a very
good compressor. A first estimate with SmolLM2-135M gave ~50 B per record. Getting from an
estimate to a compressor took three things.

**A bit-exact engine.** Arithmetic decoding only works if the decoder computes *exactly* the same
probabilities as the encoder, on any machine, at any batch size. Floating-point libraries do not
promise that. So hnz has its own Rust inference engine for GPT-NeoX / Pythia models: int8 weights
and activations with exact integer matmuls, LayerNorm / softmax / GELU with hand-written exp and
erf and sums in a fixed order, and a binary arithmetic coder with 16-bit probabilities. The
AVX-VNNI kernels produce the same bytes as the scalar reference, and the numerical definition is
versioned (`MATH_VERSION`): a block made with another version is refused instead of decoded wrong.

**Speed.** Compression runs one record per thread (122 records/s on 16 threads, after moving
attention to transposed keys: +42 %); decompression decodes many records in lock-step with a KV
cache.

**An LM that knows HN.** pythia-70m, further trained for **93 minutes** on 1 % of HN on the
laptop's integrated GPU, compresses **7-11 % better** than the original (1.33 vs 1.49 bits per
byte on 2024 comments) at the same cost per token. One surprise on the way: **bf16 breaks Pythia**
(loss 5.71 vs 4.00 in fp32); fp16 with loss scaling matches fp32.

Result: **59 bytes of text per item**, against 137 for the same text with zstd-19 and a trained
dictionary. The 71.7 MB model ships inside the block — without it the text cannot be read, like a
dictionary.

## 3. Spend time on reading, not on storing

A search engine normally keeps rich indexes so it never has to touch the documents. Here the
documents *are* compressed with a model that can decode a prefix in milliseconds. That suggests
the opposite trade: keep the indexes tiny and pay at query time.

- **Semantic index: 256 bits per item** (32 B). Each item is embedded with EmbeddingGemma 2
  (on the integrated GPU, ~200 docs/s, ~3 days for all of HN) and reduced to a binary code, scored
  asymmetrically against the full query vector. At 256 bits alone it loses quality, so it only
  has to *find candidates*.
- **Word index: rare terms only.** BM25 postings for terms in ≤ 0.02 % of documents — names,
  identifiers, products, error messages. That is **−73 % in size at the same hybrid quality**:
  in hybrid search, the rare terms are what the word index contributes. 4.2 M terms, 7.4 B per item.
- **Rerank by decoding.** For each query, the 80 best vector candidates (plus 20 from BM25) have
  their **first 128 tokens decoded** by a shared LM service that batches the work of concurrent
  queries; every query word — frequent ones included — is scored with BM25 over that prefix; the
  50 best are **re-embedded on the CPU** and reordered by cosine + BM25.

The stored text becomes a free second index: on the 3-month sample, reranking lifted the 256-bit
block from 0.211 to 0.238 P@10, and with reranking 384 bits already matched 768 bits without it. Everything is deterministic: each prompt is embedded alone at a canonical
shape, so results never depend on timing or on how many queries run at once.

![What is inside the block](img/block.svg)

## 4. One file, time-ordered, paranoid

The block is one read-only container, `hn.hnzb`, plus the query embedder's two files:

- Documents in **(time, id) order**, so a date filter is a contiguous range found exactly from
  the metadata index (309,761 HN ids — 0.63 % — have a time later than a higher id; they are
  listed apart).
- Metadata in delta-coded columns: ~6 bytes per item for time, author, score, parent and kids.
  Deleted items are kept as metadata so no thread breaks.
- Every part hashed (XXH3-128), canary records decoded at open, and four verification levels up
  to `--level deep`, which compares every record with its Parquet source. On the final block:
  **0 differences** in 49.3 M items (9.7 hours).

## 5. A RAM budget instead of a RAM requirement

`hnz.open(path, ram="2GB")` plans what lives in memory — the models, the hottest parts of the
indexes, caches — and reads the rest from disk with **identical results**. Every query's working
memory is admitted against the same ledger, so concurrent queries wait (or get a `BusyError`)
instead of pushing the process over. With 2 GB the process uses ~1.45 GB.

## 6. Building it: a week on one laptop

Two resumable workers ran side by side: LM compression on the CPU (~7 days, sharing the machine)
and embeddings on the integrated GPU (~4.5 days); packing the final block then took about an hour. The machine fought
back:

- **Windows' power throttling** treats a process without a window as background work and parks it
  on efficient cores: 79 records/s instead of 129. Opting out
  (`SetProcessInformation(ProcessPowerThrottling)`) brought it back to 119.
- **Priority starvation:** the GPU feeder at below-normal priority fed the GPU 12 docs/s instead
  of ~200. It needs little CPU, but promptly; it now runs *above* normal.
- **A GPU hang** right after the charger was unplugged and plugged back: the embedder sat for 3 h.
  A watchdog now kills and retries a month that takes 3× its normal time.
- **Modern Standby**: on battery with the lid closed the laptop freezes; one pack paused for
  3.9 hours and simply carried on. Every stage is resumable and restarts at logon.

## Results

![Search quality](img/quality.svg)

Each evaluation query is a story's title; the relevant items are the comments of that story's own
thread (300 queries per set, three months, test and dev sets). Searched within the query's month
among all 49.3 M items, the default hybrid search puts **3.2 relevant comments in the top 10** on
average; keyword search alone 2.6.

An honest caveat about this eval: without the month filter the score drops to ~0.05, because
similar threads from 20 years of HN now count as misses — the eval rewards finding *that* thread,
not finding good answers. The same queries on a 3-month block scored 0.24, so the search did not
get worse with 16× more data. A better eval for the full archive is on the list.

![Latency](img/latency.svg)

A cold reranked search takes ~2.3 s, most of it decoding the candidates' prefixes; keyword search
~0.2 s; fetching an item ~80 ms. For a 5 GB archive of 20 years of discussion, on a laptop, that
is the trade I wanted.

## What did not make it

- **A cross-encoder reranker:** +6-10 % P@10, but it needs the full text of every candidate:
  5-7× the latency.
- **A bigger LM** (pythia-160m): −8.8 % text, but decompression 2.3× slower.
- **TurboQuant-style vector quantization:** at equal size, within the noise of the 768-bit
  asymmetric binary index, and it needs training.
- **Stronger general-purpose compressors:** see section 1.

## Tested

![Acceptance](img/term-accept.svg)

73 Python tests (lossless retrieval against Parquet, disk = RAM, concurrency, a changed byte
anywhere is detected and never crashes, memory limits, the HTTP server), 35 Rust tests (bit-exact
kernels, a corruption matrix), 24 acceptance checks of the installed wheel on the full block, and
four rounds of adversarial review with ~100 findings fixed.

## Try it

```python
import hnz

with hnz.open(r"D:\hn\hn-2026-08.hnz", ram="2GB") as hn:
    for h in hn.search("how do I deal with burnout as a software engineer", k=5,
                       since="2015", until="2018", types="comment"):
        print(h.date, h.by, h.text[:120])
```

```
hnz serve --root D:\hn\hn-2026-08.hnz --ram 2GB     # web page + JSON API on localhost
```

Today it is Windows on x86-64 with AVX2; the block is rebuilt from the public dataset with the
scripts in the repository (about a week of laptop time). Code, design notes and the full experiment
log: see the [README](../README.md).

---

### Show HN text

> **Show HN: All of Hacker News in 5.3 GB, with semantic search on a laptop**
>
> I compressed every HN item (49.3 M stories and comments, 2006–2026) losslessly into a single
> 5.34 GB file — the Parquet dump is 12.4 GB — with hybrid semantic + keyword search and
> byte-exact retrieval, running on a laptop CPU in 2 GB of RAM.
>
> The text is arithmetic-coded with a 70 M-parameter LM fine-tuned on HN (59 bytes per item vs 137
> with zstd + dictionary), run by a bit-exact int8 engine in Rust. Because decoding a prefix is
> cheap, the indexes stay tiny (256-bit vectors, BM25 for rare terms only) and the candidates are
> reranked by decoding their first 128 tokens and re-embedding them. A cold search takes ~2.3 s.
>
> Python (`import hnz`), an HTTP server and a CLI. The experiment log has every number, including
> what did not work.
