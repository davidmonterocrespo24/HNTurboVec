# hnz: Compressed Hacker News with hybrid search

Minimum-size HN archive with semantic + keyword search, date/author/type filters and verbatim
retrieval of the results. Design and measurements in [`../doc/`](../doc/).

## Requirements
- **x86-64-v3 CPU** (AVX2/FMA/BMI2: Intel Haswell or later, AMD Zen or later): the binary is
  compiled with `target-cpu=x86-64-v3` (`hnz/.cargo/config.toml`). AVX-VNNI is used if present; the
  LM result is identical with or without it.
- Windows: Rust `stable-x86_64-pc-windows-gnu` and GCC (WinLibs) on the PATH to compile.
- Python environment only for preparing models and embeddings (CPU torch first, from its index),
  from the project root:
  ```bash
  uv venv .venv --python 3.12
  uv pip install --python .venv -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu
  uv pip install --python .venv -r requirements.txt
  .venv\Scripts\activate       # PowerShell/cmd (in Git Bash: source .venv/Scripts/activate)
  python scripts/prepare_models.py     # creates models/: EmbeddingGemma 2 (IR), tokenizers, .hnzlm LM
  ```
  The `python …` commands in this document assume the environment is activated (or use
  `.venv\Scripts\python` instead).
- At runtime, query embedding uses `openvino_c.dll` (from the `openvino` Python package, or from an
  OpenVINO installation): `--ov-libs` option or `HNZ_OV_LIBS` variable.

```bash
cd hnz && cargo build --release          # binary in hnz/target/release/hnz.exe
```

## Building an archive (from the project root)
Just download the whole dataset (resumable, with SHA-256): `python scripts/download_hn.py`.
All of HN, by year and resumable: `python scripts/pipeline.py --years 2006-2026` (see its help).
Running it again downloads the months that have changed in the dataset (the current month is
published partially), recomputes their embeddings and rebuilds only the affected years; a segment is
never rebuilt just because it changes tier: `--retier` cools the years that are no longer hot down
to LM (with `--delete-old`, verifying everything before deleting the original). A year that is
already built, indexed and up to date with the dataset does not download or embed anything again,
so its Parquet files and embeddings can be deleted (except those of the current year); if
already-archived Parquet files are missing it does not rebuild the year (it would lose data), and
it does not touch segments built by hand (without `sources.json`). If the dataset does not respond
it stops; `--offline` works only with the local Parquet files.
By hand, segment by segment:
```bash
hnz build --out out/seg-2024-01 --tier zstd data/2024/2024-01.parquet                 # hot
hnz build --out out/seg-2010-06 --tier lm --lm models/lm-pythia-70m.hnzlm data/2010/2010-06.parquet   # cold
hnz index --seg out/seg-2024-01 --emb emb/g2_2024-01.f16.npy --ids emb/g2_2024-01.ids.npy --bits 768
hnz index --seg out/seg-2010-06 --emb emb/g2_2010-06.f16.npy --ids emb/g2_2010-06.ids.npy --bits 512
hnz verify --seg out/seg-2010-06 data/2010/2010-06.parquet   # compares every record with the original
hnz retier --seg out/seg-2024-01 --tier lm                   # cool a segment down (leaves <seg>.old)
hnz stats                                                    # size per component of all of out/
```
Do not mix overlapping segments in the same folder (e.g. `seg-2010` and `seg-2010-06`): the engine
refuses to open them.

## Compact block (P3)
One read-only file `hn.hnzb` plus the embedder IR next to it, for all the months of a pack spec
(layout, integrity levels and measurements: [doc/08-block.md](../doc/08-block.md)):
```bash
hnz globals --out work/global data/*/*.parquet          # global df of every term + global author table
hnz pack --spec work/p3/pack.json --out work/p3/hn-sample.hnz   # needs the months' LM segments and embeddings
hnz verify-block work/p3/hn-sample.hnz --level full    # header | quick | full | deep (deep: against the sources)
hnz inspect work/p3/hn-sample.hnz --project 49324111 --lex work/lex-all/lex.json   # sizes; projection to all of HN
hnz search --root work/p3/hn-sample.hnz "how do I deal with burnout"            # every command works on a block
```
Exit codes of `verify-block`: 0 ok, 2 corrupt, 3 incompatible, 4 differs from its sources, 1 other.

## Searching
```bash
hnz search "how do I deal with burnout" -k 10
hnz search rust async runtime --mode keyword --from 2015 --to 2018-06 --type comment
hnz search "startups" --by pg --json --full         # JSON with full text (RAG)
hnz get 9700000                                      # one item by id (full text)
hnz serve --port 8088                                # web page at http://127.0.0.1:8088/
```
- `-k` number of results (max. 500; 100 over HTTP) · `--from/--to` year, year-month or date, inclusive ·
  `--mode hybrid|vector|keyword` · `--type story,comment,job,poll` · `--by author` · `--dead`.
- Filters are applied during the search (author and rare types scan the metadata of the range): up
  to k results that meet all the filters are returned, and only documents with some signal from the
  query (at least one term in keyword mode, a vector in vector mode, either of the two in hybrid).
- Without text (`hnz search --type job --from 2024`), the most recent items that meet the filters
  are listed.
- Cold tier: by default a **snippet** is returned (full title and url + 64 tokens of the body,
  `partial: true`); `--full` or `GET /item/<id>` decompress the full text.

HTTP API: `GET /search?q=…&k=10&from=2015&to=2018&type=comment&by=pg&mode=hybrid&full=0`,
`GET /item/<id>`.

## From Python (`import hnz`)
Package in `hnz/py/` (Rust core via PyO3; `abi3` wheel for Python >= 3.10, Windows x86-64):
```bash
cd hnz/py && maturin develop --release        # installs into the active environment (or: maturin build --release)
```
```python
import hnz
with hnz.open(r"C:\path\out", ram="2GB", preload=True) as hn:   # models in out/models or out/../models
    r = hn.search("how do I deal with burnout", k=10, since="2015", until="2018-06", types="comment")
    for h in r:                                                    # Hit dataclasses (or raw=True: dicts)
        print(h.rank, h.date, h.by, h.story_title, h.text[:120])
    item = hn.get(9700000)                                         # verbatim: item.html = original HTML
    jobs = hn.search(types="job", since="2024")                    # no text: most recent first
    print(r.timings_ms, hn.memory()["budget"])
```
- `Archive` is thread-safe and releases the GIL: a backend (e.g. FastAPI with one worker) serves
  several queries at once on the same open archive. Bad arguments -> `ValueError`.
- `ram=`: ceiling for everything hnz holds (models, embedder, loaded indexes, LM batch, caches,
  per-query memory); what does not fit is read from disk on demand, with identical results.
  `hnz.BudgetError` if it is below the minimum (~750 MB for the sample archive). `max_concurrent`
  (8) and `timeout` (30 s): admission control, `hnz.BusyError` when there is no turn in time.
  `cache_dir=`: optional persistent cache of decompressed records. Details and measurements:
  [doc/07-ram.md](../doc/07-ram.md).
- A compact block opens the same way (`hnz.open(r"C:\path\hn-sample.hnz")`); `verify="quick"`
  (default) checks its structure, the always-loaded parts and 4 decoded canaries (~0.3 s);
  `hnz.verify_block(path, "full")` hashes everything; damaged or unreadable blocks raise
  `hnz.CorruptBlockError` / `hnz.IncompatibleBlockError` (both `hnz.BlockError`).
  `hnz.inspect(path)` describes it.
- OpenVINO's DLLs come from the installed `openvino` package (or `ov_libs=` / `HNZ_OV_LIBS`).
- Tests: `python -m pytest hnz/py/tests` (same results as the CLI, lossless against the Parquet
  files, 8 threads at once, disk mode == RAM mode, process memory <= 1.03 x `ram`; on the old
  archive and on the sample block, plus a tamper matrix for the block).

## Environment variables
- `HNZ_MODELS` (models folder), `HNZ_OV_LIBS` (`openvino_c.dll` folder).
- `HNZ_LM_THREADS` (LM threads; by default half of the logical threads, minimum 1),
  `HNZ_KERNEL=avx2|scalar` (force a kernel; the result is identical), `HNZ_PROF=1` (LM profiling).
