# All of Hacker News in 5.3 GB, searchable on a laptop

*This article describes hnz, a Rust library with a Python API that stores every Hacker News item
(49.3 million of them, from October 2006 to August 2026) in a single 5.34 GB file and searches it
with hybrid semantic and keyword search on a laptop CPU. It explains how a small language model
compresses the text, why the search indexes can stay small, and what the results look like.*

## Why I built it

I wanted all of Hacker News on my laptop, searchable the way a search engine is: a question in
plain words, a few keywords, a date range, the top results with their full text. No server, no
GPU card. The public Parquet dump of HN is 12.4 GB and has no search at all, so I set myself a
target of 4 to 5 GB for everything. In exchange I was willing to wait up to 10 seconds for an
answer, and to let the archive take days to build.

The machine is a laptop with an Intel Core Ultra 7 265H, 31 GB of RAM and an integrated GPU.
Everything in this article was built and measured on it, in October 2026.

Before going further, what you need to use it today: hnz runs on Windows on x86-64 with AVX2, and
the archive (I call it the block) is not published for download yet. You rebuild it from the
public dataset with the scripts in the repository, which takes about a week on a laptop like mine.

## Where the numbers start

![All of HN on disk](img/size.svg)

| Format | Size | Search |
|---|---|---|
| Raw JSON from the HN API | ~24 GB | none |
| Parquet dump ([open-index/hacker-news](https://huggingface.co/datasets/open-index/hacker-news)) | 12.4 GB | none |
| zstd-19 with a trained dictionary, item by item | 9.1 GB | none |
| hnz block | 5.34 GB | hybrid, with filters |

The 5.34 GB include the text, the metadata, a semantic index, a word index and the two models
that hnz needs to read them.

## Compressing the text with a language model

My first experiments used ordinary compressors. Grouping items into small blocks before zstd took
text and metadata from 181 to 124 bytes per item. After that, stronger compressors stopped
helping. zstd at level 22, brotli and xz all landed between 106 and 114 bytes, and even xz over a
whole month in one stream, with no random access at all, only reached 91.

The paper [Language Modeling Is Compression](https://arxiv.org/abs/2309.10668) (DeepMind, 2023)
shows that a model that predicts the next token well, driving an arithmetic coder, is a very good
compressor. A quick test with SmolLM2-135M suggested about 50 bytes per record, so I tried it for
real. Three problems came up.

1. **The decoder has to compute exactly the same probabilities as the encoder.** One different
   bit and the rest of the record decodes as garbage. Floating-point inference libraries do not
   promise this across machines or batch sizes, so I wrote a small inference engine in Rust for
   Pythia models: int8 weights and activations, exact integer matrix products, and LayerNorm,
   softmax and GELU with my own exp and erf, summed in a fixed order. The AVX-VNNI kernels produce
   the same bytes as the plain scalar code, and the block records which version of the arithmetic
   it was made with, so a mismatched reader refuses it.
2. **Speed.** Compression runs one record per thread and reaches 122 records per second on 16
   threads. Decompression decodes many records in lock-step, with a KV cache.
3. **A model that knows HN.** I trained pythia-70m for another 93 minutes on 1% of HN, on the
   laptop's integrated GPU. It now compresses HN text 7 to 11% better (1.33 bits per byte instead
   of 1.49 on comments from 2024), at the same cost per token. One surprise along the way: bf16
   broke Pythia completely (loss 5.71, against 4.00 in fp32), while fp16 with loss scaling matched
   fp32.

The text now takes 59 bytes per item. The same text with zstd-19 and a trained dictionary takes
137. The model itself is 72 MB and travels inside the block, the same way a compression
dictionary would.

## Small indexes, with the text as a second index

Decoding the first tokens of a record takes milliseconds. That changes the usual trade of a search
engine, which keeps rich indexes so it never has to touch the documents. Here the indexes can be
small, and the search pays a little at query time instead:

- The **semantic index** keeps 256 bits per item. Every item was embedded with EmbeddingGemma 2 on
  the integrated GPU (about 200 documents per second) and reduced to a binary code. 256 bits on
  their own lose quality, so they are only used to find candidates.
- The **word index** only covers rare terms, the ones in at most 0.02% of the documents: names,
  identifiers, products, error messages. In my tests it was 73% smaller than a full index, with the
  same hybrid quality. Frequent words get scored later, on the decoded text.
- For each query, the 80 best vector candidates and the 20 best keyword candidates get their
  **first 128 tokens decoded** by a language model service shared by all running queries. Every
  word of the query is scored with BM25 over that prefix, and the best 50 candidates are embedded
  again on the CPU and reordered.

![How a query runs](img/diagram-query.svg)

On a three-month sample, this reranking raised P@10 from 0.211 to 0.238, and with it 384 bits
did as well as 768 bits without it. Results are also deterministic: each prompt is embedded alone,
at a fixed shape, so the answer does not depend on timing or on how many queries run at once.

## Inside the block

![Inside the 5.34 GB block](img/diagram-block.svg)

The block is one read-only file, `hn.hnzb`, next to the two files of the query embedder.

- Items are stored in order of time and id, so a date filter is a contiguous range, found
  exactly from the metadata index. About 310,000 HN ids (0.63%) have a timestamp later than a
  higher id, and those are listed separately.
- Metadata lives in delta-coded columns, about 6 bytes per item for the time, author, score,
  parent and children. Deleted items stay as metadata, so threads never break.
- Every part has a hash (XXH3-128), and a few known records are decoded when the block opens. The
  deepest check, `verify-block --level deep`, compares every record with its Parquet source. On the
  final block it found 0 differences in 49.3 million items, after 9.7 hours.

## Memory

`hnz.open(path, ram="2GB")` decides what to keep in memory (the models, the most used parts of
the indexes, caches) and reads the rest from disk, with the same results either way. The working
memory of each query is charged to the same budget, so when it runs out, new queries wait or get a
`BusyError` instead of pushing the process over the limit. With `ram="2GB"` the process used about
1.45 GB.

## Building it

![How the block is built](img/diagram-build.svg)

Two workers ran side by side for about a week. One compressed the text with the language model on
the CPU (about 7 days), the other computed the embeddings on the integrated GPU (about 4.5 days).
Both could resume after a reboot, which turned out to matter. Packing the final block took about
an hour.

Windows gave me most of the trouble:

- Its power throttling treats a process without a window as background work and moves it to the
  efficient cores. Compression dropped from 129 to 79 records per second. Opting out with
  `SetProcessInformation(ProcessPowerThrottling)` brought it back to 119.
- At below-normal priority, the process feeding the GPU starved: 12 documents per second instead
  of about 200. It needs little CPU, but it needs it promptly, so it now runs above normal.
- Once, right after I unplugged and plugged back the charger, the GPU run hung for 3 hours. A
  watchdog now kills and retries any month that takes three times longer than usual.
- On battery with the lid closed, the laptop goes into Modern Standby and everything freezes. One
  packing run paused for 3.9 hours and continued on its own when the laptop woke up.

## Results

![Search quality](img/quality.svg)

To measure quality I used 300 queries per month for three months, in a test set and a dev set.
Each query is a story's title, and the relevant results are the comments in that story's own
thread. Searching within the query's month, among all 49.3 million items, the default hybrid
search puts on average 3.2 relevant comments in the top 10. Keyword search alone puts 2.6.

Without the month filter the score drops to about 0.05, and that number needs some context. This
eval only counts the query's own thread as relevant, so similar threads from other years count as
misses even when they answer the question. The same queries on a three-month block scored 0.24, so
the search itself did not get worse with 16 times more data. A better eval for the whole archive
is on my list.

![Latency](img/latency.svg)

A cold reranked search takes about 2.3 seconds, and most of that is decoding the candidates'
prefixes. A keyword search takes about 0.2 seconds, and fetching an item about 80 ms. That is well
inside the 10 seconds I was ready to accept.

![The built-in search page](img/web-burnout.png)

## Ideas I tried and dropped

| Idea | What it gave | Why I dropped it |
|---|---|---|
| A cross-encoder reranker | +6 to 10% P@10 | it needs the full text of every candidate: 5 to 7 times slower |
| A bigger LM (pythia-160m) | 8.8% less text | decompression 2.3 times slower |
| TurboQuant-style vectors | within the noise at the same size | needs training, no clear gain |
| Stronger general compressors | 106 to 114 bytes per item | far from what the LM reaches |

## Testing

![Acceptance checks](img/term-accept.svg)

The Python suite has 73 tests. They check that every item comes back byte for byte against the
Parquet files, that results are the same from disk and from RAM, concurrency, that a changed byte
anywhere in the block is detected without crashing, the memory limits and the HTTP server. The
Rust core has 35 more, including bit-exact kernels and a corruption matrix. A separate script runs
24 checks of the installed package against the full block in about 30 seconds.

## Trying it

1. Build the block with the steps in the [README](../README.md#build-the-block-yourself)
   (about a week), or copy one you already have.
2. Install the package: `pip install hnz-0.1.0-cp310-abi3-win_amd64.whl`.
3. Search from Python:

   ```python
   import hnz

   with hnz.open(r"D:\hn\hn-2026-08.hnz", ram="2GB") as hn:
       for h in hn.search("how do I deal with burnout as a software engineer", k=5,
                          since="2015", until="2018", types="comment"):
           print(h.date, h.by, h.text[:120])
   ```

4. Or start the web page with `hnz serve --root D:\hn\hn-2026-08.hnz --ram 2GB` and open
   `http://127.0.0.1:8088/?q=sqlite+in+production`.

The code, the design notes and the full experiment log, including the experiments that failed,
are in the repository.

---

## Show HN text

**Show HN: All of Hacker News in 5.3 GB, with semantic search on a laptop**

I wanted all of HN on my laptop with real search, so I compressed every item (49.3M stories and
comments, 2006 to 2026) into one 5.34 GB file. For comparison, the Parquet dump is 12.4 GB and
has no search. Retrieval is lossless, byte for byte.

The text is arithmetic-coded with pythia-70m, which I fine-tuned on 1% of HN. It takes 59 bytes
per item, against 137 with zstd and a dictionary. To make that work I had to write a small int8
inference engine in Rust that gives the same probabilities on every run, because one different
bit breaks the decoding.

Since decoding the start of an item is cheap, the indexes are small: 256-bit vectors and BM25 for
rare terms only. Each query decodes the first 128 tokens of 100 candidates and reranks them. A cold
search takes about 2.3 s on the CPU, inside a 2 GB RAM budget.

It's a Python package (`import hnz`) with a CLI and a small HTTP server, Windows only for now. The
repo has the experiment log with every number, including what didn't work.
