# Text storage (the retrievable information)

Numerical details in [BITACORA.md](BITACORA.md) (E00-E03, E09, E12, E15-E17, E20, E24, E25).

## What we have learned

1. **Text is the heaviest part** (~335 B/item raw). Metadata in delta columns: ~5.5 B/item.
2. **Small blocks > stronger compressor.** Going from 1 record to 16 per block brings it down from
   181 to ~124 B/item; zstd-22, brotli-11 and xz barely improve on zstd-19 + dictionary.
3. **Ordering by thread** helps a little (118.5 → 114.4 B in blocks of 256); by topic < 1%. But an
   order other than by id requires storing a doc→block map (~4 B/doc) → **ordering by id**
   (= chronological order), which also turns a date range into a contiguous range of documents.
4. **The big leap is the LM as a compressor**: with pythia-70m int8 and representation "e" (E15),
   ~61-64 B/record of text versus ~117 B with zstd in blocks of 16. Each record is encoded on its
   own → perfect random access; the "dictionary" is the model (72 MB).
5. **HN's HTML**: two escaping dialects (old ~2007-2012 and modern) and links truncated with
   "…" (30-40% of the links) → reversible transformation verified record by record.
6. **`deleted` rows**: stored as metadata (type, date, parent, dead) so as not to break the chain
   of the threads (in 2010-06, 141 comments hang off a deleted item). The order of `kids` is not
   stored.

## How the LM's risks were resolved

- **Bit-for-bit determinism:** custom engine in Rust with exact int8 matmul (the result does not
  depend on the SIMD kernel or on the threads), custom exp/erf and sums in a fixed order;
  `MATH_VERSION` in `lm.json`.
- **Integrity:** 1 check byte per record, validated when decompressing the **full** record
  (`--full`, `get`, `verify`): a divergence (another binary with the same model and the same
  `MATH_VERSION`) produces an error in ~255 out of every 256 records; the rest are rejected only if
  they also come out as invalid UTF-8 or a malformed record, so it is not an absolute guarantee
  (`hnz verify` is one: it compares against the original). Snippets and titles are cut before the
  end and do not check it; there, the protection is that opening the segment requires the same
  model id and `MATH_VERSION`.
- **Sequential decompression:** a batch with all the results of a query, custom thread pool,
  snippet mode (64 tokens of the body) and L1 (RAM) + L2 (disk) caches.
- **Compression cost:** ~120 rec/s on 16 threads (≈ 4 days for ~40 M cold items), in parallel with
  the embeddings on the iGPU.

## Implemented tiers

| Tier | Typical use | Format | Text B/item (measured) | Reading a top-10 |
|---|---|---|---|---|
| Hot | recent years | zstd-19 + dict, blocks of 16 | ~117 | ~2.5 ms (26 ms with embedding and search) |
| Cold | older years | pythia-70m LM + arithmetic coding | ~61-65 | ~0.26 s snippets; ~0.5 s full (2 ms cached in RAM, 29 ms on disk) |

The zstd block size is configurable (`--block`): blocks of 256 give ~114 B with reads of
~0.2 ms (a possible "warm" tier). A hot segment is cooled down with `hnz retier --tier lm`.
