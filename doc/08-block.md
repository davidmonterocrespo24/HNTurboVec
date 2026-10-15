# The compact block (P3): `hn.hnzb`

**Status:** format v1.0 implemented and measured on the 3-month sample on 2026-10-15, then
reviewed (log entry E32 in [BITACORA.md](BITACORA.md)). Design context:
[06-libreria.md](06-libreria.md); RAM rules: [07-ram.md](07-ram.md). This document is the
reference for the bytes of the container, the metadata, the vectors, the word index and the
kids. Three encodings are only named here: the LM's arithmetic-coded records (they need hnz's LM
decoder for the block's `math_version`), the FST of the `fst` crate and the BitPacker4x blocks of
the `bitpacking` crate.

## What a block is

A **block** is a folder with three files:

| File | What |
|---|---|
| `hn.hnzb` | the container: every document's metadata, text, vector code and word-index entry, the LM model and the tokenizers |
| `embed.xml`, `embed.bin` | the query embedder (EmbeddingGemma 2 int8, OpenVINO IR), read by OpenVINO by path; their size and hash are in the container |

It is read-only. A new HN snapshot means a new block (rebuilt with `hnz pack`), with a new
**snapshot id** (the hash of its table of contents). `hnz.open(path)` accepts the folder or the
`hn.hnzb` file; a folder that holds a block **and** segment folders is refused, and a folder
without `hn.hnzb` is not a block. Folders of segments (the format before P3) are still read.

## Documents

- One block holds documents numbered `0..n_docs` ("docnums"), every HN item of the source files,
  **deleted rows included** (as metadata-only documents, so threads are never broken).
- Documents are in **(time, id) order**. 309,761 HN ids (0.63 %) have a time later than some
  higher id (most in 2013, 2015, 2024, 2025 and 2026), so id order and time order differ. With
  time order a date range is a contiguous range of documents, found exactly from the metadata
  index. The ids out of time order ("late") are listed in `meta/late`.
- Times are whole seconds. The dataset's Parquet files store them as timestamps in milliseconds,
  or in microseconds (the files from 2026-03 on); `hnz pack` refuses an item whose time is outside
  its file's month.
- Every part below is "per document" in this order.

## Container layout

All integers are little-endian. Hashes: XXH3-128 written as `"xxh3-128:<32 hex digits>"` in the
TOC; XXH3-64 for the superblock and the trailer.

```text
0          superblock (4096 bytes)
4096       part 1 (each part starts at a multiple of 4096; the gaps are zero bytes)
...        part n
toc_off    TOC: zstd-compressed JSON (starts at a multiple of 4096)
file_len-64  trailer (64 bytes)
```

**Superblock** (4,096 bytes):

| Offset | Size | Field |
|---|---|---|
| 0 | 8 | magic `48 4E 5A 42 0D 0A 1A 0A` ("HNZB\r\n\x1a\n") |
| 8 | 2 | format major (1) |
| 10 | 2 | format minor (0) |
| 12 | 4 | superblock length (4096) |
| 16 | 8 | incompatible feature bits (a reader refuses any bit it does not know; v1: none) |
| 24 | 8 | compatible feature bits (ignored if unknown; v1: none) |
| 32 | 8 | file length |
| 40 | 8 | TOC offset |
| 48 | 8 | TOC stored length |
| 56 | 8 | TOC decoded length |
| 64 | 4 | TOC codec (1 = zstd) |
| 68 | 4 | part alignment (4096) |
| 72 | 16 | TOC hash (XXH3-128 of the stored TOC bytes) = **snapshot id** |
| 88 | 8 | number of documents |
| 96 | 4 | LM math version (also in the manifest; they must agree) |
| 104 | 64 | writer, e.g. "hnz 0.1.0" (NUL padded) |
| 4088 | 8 | XXH3-64 of bytes 0..4088 |

**Trailer** (last 64 bytes): magic "HNZBEND\0", TOC offset, TOC length, TOC hash (16), file length,
8 reserved bytes, XXH3-64 of the first 56 bytes. It must repeat the superblock's values. A file
whose first 8 bytes are not the magic but whose trailer is valid is a damaged block; anything else
is "not a block".

**TOC** (JSON; the first part and an external file of the sample block):
```json
{"format": {"major": 1, "minor": 0, "incompat": [], "compat": []},
 "parts": [{"name": "meta/index", "ver": 1, "off": 4096, "len": 4944, "codec": "zstd",
            "raw_len": 12468, "hash": "xxh3-128:…", "access": "load", "required": true}, …],
 "external": [{"name": "embed.bin", "len": 272498827, "hash": "xxh3-128:…", "role": "query_ir_weights"}, …],
 "manifest": {…}}
```
- `codec`: `raw` (stored as is) or `zstd` (one frame, level 19; `raw_len` = decoded size).
- `hash`: of the **stored** bytes. `access`: how the engine uses the part: `load` = read whole at
  open; `scan` = scanned by queries; `random` = read at offsets by queries; `cold` = read whole
  when needed (tools, the embedder's load), never by queries.
- Parts are aligned, inside `[4096, toc_off)`, do not overlap, and have unique names.
- A reader refuses (incompatible) a **required** part it does not know, a part version it does not
  read, or a required part with a codec or access class it does not know; it ignores unknown
  optional parts. A TOC whose hash is right but that this reader cannot parse is incompatible
  (written by another hnz), not damaged.
- External names are plain file names in the block folder (no path, drive or `..`), one per role
  (`query_ir`, `query_ir_weights`).

## Parts (v1)

| Part | Codec | Access | Content |
|---|---|---|---|
| `meta/index` | zstd | load | per metadata block: i64 time of its first row, u32 "main id", u64 offset into `meta/blocks`; then the u64 end |
| `meta/blocks` | raw | random | metadata rows, 1,024 per zstd frame (level 19, content size and **content checksum**) |
| `meta/flags` | zstd | load | 1 byte per document: `type | dead<<3 | deleted<<4 | late<<5` (filters without decompressing rows) |
| `meta/late` | zstd | load | (u32 id, u32 doc) of the ids out of time order, sorted by id |
| `meta/authors` | zstd | load | the global author table: names separated by `\n`, by count desc then name asc |
| `meta/kids` | raw | cold | the order of each item's `kids`, zstd frames of 1,024 documents |
| `meta/kids.idx` | zstd | cold | u64 offset of each kids frame, then the end |
| `text/groups` | raw | random | the compressed records, in groups of 32 documents |
| `text/idx` | zstd | load | u64 offset of each group, then the end |
| `text/gsum` | raw | load | u32 per group: low 32 bits of the XXH3-64 of the group's bytes |
| `text/prefix_fallback` | zstd | load | the prompt prefix of each document with a vector whose record is not an LM record: u32 count, then per document u32 doc, u8 text flags, varint length, the prefix (UTF-8) |
| `vec/has` | zstd | load | u64 words: bit `d % 64` of word `d / 64` = document `d` has a vector |
| `vec/codes` | raw | scan | `bits / 8` bytes per document **that has a vector**, in document order |
| `lex/fst` | raw | load | FST: every indexed term → tagged value (below) |
| `lex/post` | raw | random | postings lists of the rare terms |
| `lex/norms` | zstd | load | 1 byte per document: `min(255, round(16 · log2(1 + length)))` |
| `model/lm` | raw | load | the LM model (`.hnzlm`) that compressed the text |
| `model/lm.tok` | zstd | cold | its tokenizer (JSON): needed only to recompress or to recheck records |
| `model/embed.tok` | zstd | cold | the query embedder's tokenizer (JSON), read when the embedder loads |

### Metadata rows (row encoding 2)

A frame decodes to the columns of its rows, one after another (varints are LEB128; `zz` is
zigzag):
1. id: the first raw, then `zz(id - previous id)` (signed: rows are in time order);
2. time (seconds): the first raw, then `zz(time - previous time)`;
3. one byte per row: `type | dead<<3 | deleted<<4` (type 1 story, 2 comment, 3 poll, 4 pollopt, 5 job);
4. parent: 0, or `zz(id - parent)`;
5. author: index into `meta/authors`;
6. `zz(score)`; 7. `zz(descendants)`;
8. poll: 0, or `zz(id - poll)`;
9. parts: count, then `zz(part - id)` each.

**Finding an id:** first `meta/late` (binary search); otherwise the last block whose main id is
<= the id; if the blocks before it share the same main id (a block made only of late rows
inherits the previous main id), the first of that run; then a scan of its rows. The main id of a
block is its first row that is not late (in-order ids increase with the document number).
**Finding a time:** the first block whose first time is >= t, and the rows of the block before it.

### Text records

A group is: the varint length of each of its records, then the records. A record is:

```text
varint header = ntok<<4 | codec<<3 | lmmap<<2 | mode
LM record (codec 0, ntok > 0):   check byte, then the arithmetic-coded tokens
zstd record (codec 1):           zstd(varint title_len, varint url_len, title, url, html)
empty record:                    the header alone, 0x00: items with no title, url or text
```
The LM records are decoded with `model/lm`: int8 weights and f32 activations, computed in a fixed
order so every CPU gives the same bytes (`math_version`). The text is `title\nurl\nbody`, with
`mode` / `lmmap` saying how the HTML was turned into the LM's string (and back, losslessly). The
check byte is the FNV-1a-64 of the decoded string (offset basis `0xcbf29ce484222325`, prime
`0x100000001b3`) folded as `h ^ h>>8 ^ h>>16 ^ h>>24 ^ h>>32`, low 8 bits: a mismatch on a record
whose group checksum is right means **this binary decodes differently** → incompatible, not
corrupt. `ntok` is at most `min(2000, max_pos − 1)` of the model.

Records are **copied byte for byte** from the per-month LM segments (`hnz build --tier lm`);
`hnz pack` checks every copied record against its source item, with the tokenizer the block stores
(the whole record for zstd and empty records, the header and check byte for LM records).

### Vectors

Sign rule 1: bit `j` (bit `j % 8` of byte `j / 8`) is 1 when dimension `j` of the float16 document
embedding is > 0; the first `bits` dimensions (Matryoshka). v1 uses **256 bits** of the 768-dim
EmbeddingGemma 2 embedding of the document's prompt:
- `prompt_version` 2 (`hnz/src/prompt.rs`, the reference): the **prefix** is the LM string up to
  the token that completes its second newline (end of title and url) plus 128 body tokens, which
  is where the LM decoder stops when asked for 128 body tokens; the prompt is
  `title: <title or "none"> | text: <the prefix body as plain text>` (a link with its own text
  becomes "text (url)"; whitespace collapsed), cut at 256 Gemma tokens. The url is not embedded.
  `hnz prompts` writes the Gemma token ids; `scripts/embed_prompts.py` embeds them.
- `prompt_version` 1 (E30a, `experimentos/embed_prefix.py`): the same idea in Python, with the
  title/url and the body tokenized separately and a 224-token cap.

Documents **with a prompt** have a vector: those that are not deleted and whose LM string (title,
url and text) is not blank; deleted rows, blank items and items whose title or url contains a
newline (they have no LM string) have no code. `vec/has` plus a rank table built at open map a
document to its code.

### Word index (value layout 1)

Every indexed term (the engine's tokenizer, `tokenizer_version` 1) is in the FST. Value `v`:
- `v & 3 = 0` FREQUENT: `v >> 2` = df in the block; no postings (global df > 0.02 % of all HN,
  from `hnz globals`);
- `v & 3 = 1` SINGLE: the only document is `v >> 2`;
- `v & 3 = 2` LIST: postings at offset `v >> 2` of `lex/post`.

A postings list: varint df; per full block of 128 documents, a header pair (varint delta of its
last document, varint payload length); then the payloads (byte: bits per document delta; byte:
0; BitPacker4x sorted deltas from the previous block's last document); then the remaining
documents as varint deltas. No term frequencies (E30c). BM25 uses the global IDF of the block and
the norms.

### Kids order

Per document: code 0 = no kids; 1 = its children (the documents whose parent is its id) in id
order; 2 = in reverse id order; 3 + k varints = a permutation of the children; 4 + n + n varints =
an explicit list (zigzag deltas chained from the item's id). Reading it needs the children found
through `parent`, so it is decoded by full passes (`verify-block --level deep`, an export), not by
`get()`.

## Manifest

The TOC's `manifest` records what the bytes depend on and where they came from:

| Key | What |
|---|---|
| `snapshot` | dataset, source files (path, rows, SHA-256, revision), `n_docs`, `n_deleted`, `id_min/max`, `time_min/max`, `n_lm`, `n_fallback`, `n_empty`, `n_with_vector`, `n_late` |
| `meta` | `block_rows` 1024, `row_encoding` 2, `frame_checksum`, author table (`n`, order, source hash) |
| `kids` | `encoding` 1, `block` 1024 |
| `text` | `codec` lm, `group` 32, `framing` 1, `rec_check` 1, `math_version`, `transform_version` 1, `max_tokens`, `model_id`, `lm_records`, `zstd_records`, `tokens`, `gsum` |
| `vec` | `bits`, `source_dims`, `sign_rule` 1, `layout` sparse, `prompt_version`, `prefix_tokens` 128, `max_seq` 256, `doc_prompt`, `query_prompt`, `prefix_fallback` (count) |
| `lex` | `info` (n_docs, avgdl, n_terms, n_postings, layout, max_df…), `tokenizer_version` 1, `value_layout` 1, `class_source` (n_docs, hash of the global df), term class counts |
| `search` | `c1`: minimum stage-1 vector candidates (20,000 for blocks) |
| `canaries` | 7 documents (4 short, 1 long ~1,500 tokens, 1 zstd, 1 empty) with the XXH3-64 of their record bytes and of their text |
| `build` | hnz version and the pack spec (JSON), so the block can be rebuilt and checked |

A reader refuses (**incompatible**) a block whose manifest names a definition version it does
not implement: `text.math_version`, `text.transform_version`, `text.rec_check`, `text.framing`,
`text.group` (32), `meta.row_encoding`, `lex.value_layout`, `lex.tokenizer_version`,
`vec.sign_rule`, `kids.encoding`; a `model/lm` whose id is not `text.model_id`; and intact model or
word-index parts it cannot parse.

## Integrity

| Level | Where | Checks | Sample block |
|---|---|---|---|
| header | every open | magic, superblock and trailer hashes, file length, TOC hash, part bounds / alignment / overlaps, known parts and versions, external names, definition versions | ms |
| quick | `hnz.open` default (`verify="quick"`) | + the `load` parts read whole (so hashed), `model/lm` id = manifest, the 4 short canaries decoded and compared | 0.33 s |
| full | `verify="full"`, `hnz.verify_block()`, CLI | + every part hash, the zero padding, the external files, every canary | 4.5 s (423 MB) |
| deep | `hnz verify-block --level deep` | + everything against the sources of its pack spec (below) | 4.4 min (0.5 % decoded), ≈13 min (2 %) |

While serving, damage is found where it is read: a text group's checksum (`text/gsum`, on every
read), a metadata frame's zstd checksum, the structure of a postings block (sizes, bit widths,
increasing documents < n_docs), a record header (at most `max_tokens`), the external files' hash
when the embedder loads, and any part hash when the planner loads it whole. **Not** checked at
query time: single flipped bits in `vec/codes` and inside a valid-looking postings list (they can
change results silently until `full` runs; per-chunk hashes are planned for P5). A failed read of
codes or postings is an error, not a silent zero.

Errors (Python): `hnz.CorruptBlockError` (damaged bytes: hash, checksum, length, malformed data;
the message names the part or group) and `hnz.IncompatibleBlockError` (intact, but this hnz cannot
read it: not a block, format major, unknown required part, codec or feature, a definition version,
a different LM model, or a canary / check byte that decodes differently), both subclasses of
`hnz.BlockError`.

**Deep check** (`hnz verify-block <block> --level deep [--spec pack.json] [--decode 0.02|all|none]`):
- the metadata: the (time, id) order, unique ids, the flag column, the late flags against the
  times, `meta/late` and `meta/index` against the rows, every id lookup, the time lookup at every
  block start and at a sample;
- the models, the tokenizers and the embedder files against the pack spec's files; the author
  table against `hnz globals`;
- per month of the spec, every item against its document: the metadata row (every column but
  `words`), the record bytes against the month segment and the header recomputed from the item
  with the block's own tokenizer, the vector code against the embedding files, the `kids` list,
  the prefix of LM-less records with a vector, and that the item's time is inside its month; a
  sample (default 2 %, or `all`, or `none`) of the records decoded with the block's LM and compared
  with the Parquet text;
- the word index **rebuilt byte for byte**, and the manifest counters (`n_docs`, `n_late`,
  `n_deleted`, id and time ranges, record kinds, tokens, vectors, prefixes, `lex.info`).

It streams month by month: per document it keeps the id, time and parent (≈4 GB at 49 M
documents, most of it the word index being rebuilt). A problem in a source file (a segment that
does not match its Parquet file, malformed embedding files) is a plain error, not a block error.
Exit codes: 0 ok, 2 corrupt, 3 incompatible, 4 the block differs from its sources, 1 any other
error.

## Tools

| Command | What |
|---|---|
| `hnz globals --out work/global data/*/*.parquet` | global df of every term (`df.fst`), the global author table (`authors.zst`), per-file counts and hashes (`globals.json`); 117 s for all HN |
| `hnz lexicon --globals work/global --out work/lex-all data/*/*.parquet` | the all-HN word index alone, for exact size projections; 579 s |
| `hnz prompts --out <dir> data/<y>/<m>.parquet…` | the document prompts (v2) as Gemma token ids, for `scripts/embed_prompts.py` |
| `hnz pack --spec pack.json --out <dir>` | builds the block (spec: months, path patterns of the Parquet files, LM segments and embeddings, globals, models, bits, c1, prompt version) |
| `hnz verify-block <dir> --level header|quick|full|deep` | the checks above |
| `hnz inspect <dir> [--project N --lex lex.json] [--id ID]` | parts, offsets, sizes, bytes per document; projection to N documents; where an item's bytes are |
| `hnz search|eval|get --root <dir>` | the usual commands work on a block |

Python: `hnz.open(path, ram=…, verify="quick")`, `hnz.verify_block(path, level="full")`,
`hnz.inspect(path, project=None, lex=None, id=None)`. `ram` covers what an open `Archive` holds;
`verify_block` (quick or full) and `inspect(id=…)` open the block on their own, outside any
budget: the always-loaded parts and the LM model (≈0.15 GB for the sample, ≈0.4 GB for all of
HN), plus a few 4 MiB hash buffers.

## Sample block (2010-06, 2015-06, 2024-01: 637,046 documents)

| Part | Bytes | B/doc |
|---|---|---|
| `model/lm` | 71,713,726 | (fixed) |
| `text/groups` | 36,145,749 | 56.7 |
| `vec/codes` | 19,572,032 | 30.7 |
| `meta/authors` | 4,969,307 | (global, fixed) |
| `model/embed.tok` | 3,923,175 | (fixed) |
| `lex/post` | 3,334,285 | 5.2 |
| `meta/blocks` | 3,242,030 | 5.1 |
| `lex/fst` | 2,615,606 | 4.1 |
| `model/lm.tok` | 516,373 | (fixed) |
| `lex/norms` | 462,229 | 0.7 |
| `meta/kids` | 139,564 | 0.2 |
| others (`text/gsum`, `meta/flags`, `text/idx`, `vec/has`, `text/prefix_fallback`, `meta/index`, `meta/kids.idx`, `meta/late`) | 236,897 | 0.4 |
| **container** | **146,930,254** | |
| `embed.bin` + `embed.xml` | 275,962,545 | (fixed) |

Built in 60 s; the same inputs give the same bytes (1 or 16 threads). **Projection to all of HN**
(`hnz inspect --project 49324111 --lex work/lex-all/lex.json`, with the exact all-HN word index:
365 MB): **5.32 GB, 107.8 B/doc** (with the current LM; the LM retrained on 5 % of HN may lower
the text part).

Quality on the block (300 queries per month, `hnz eval`, P@10): vector 0.208 / 0.147 / 0.185
(2024 / 2010 / 2015; mean 0.180, as E30a's 0.181 for 256-bit prefix codes); words alone 0.119 /
0.056 / 0.080 (only rare terms have postings; P4 scores the frequent ones over the semantic
candidates); hybrid w = 0.15, c = 20: 0.241 / 0.169 / 0.217. The P4 reranking (E30c) is not in
P3.

## Known limits of v1

- **Keyword mode with only frequent words** ("rust async runtime", "postgres vs mysql": all of
  them above 0.02 %) returns nothing: frequent terms have no postings. P4 scores every query word
  over the candidates of the vector stage, which covers it.
- `get()` does not return `kids` (stored, see above).
- `hnz pack` holds the months it packs in memory: right for the sample; the all-HN build (P5)
  streams them.
- Vector codes and postings are not hashed per read (see Integrity).
