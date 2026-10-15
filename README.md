# HNTurboVec

All of **Hacker News** (~49.3 M items) compressed **losslessly**, with **hybrid search**
(semantic + keyword, number of results, date ranges, type and author) and **verbatim retrieval** of
the results. A Rust engine (`hnz`) used from the command line, over HTTP or from **Python**
(`import hnz`).

- Today: ~10.4 GB for all of HN (search in 0.03-0.3 s); measured plan for a ~5 GB compact profile.
- Status, decisions and numbers: [doc/README.md](doc/README.md). All experiments:
  [doc/BITACORA.md](doc/BITACORA.md). Library design: [doc/06-libreria.md](doc/06-libreria.md).

## Structure

| Folder | Contents |
|---|---|
| `hnz/` | Rust engine (library + CLI `hnz.exe`) and Python package in `hnz/py` |
| `scripts/` | Tools to build the archive: `download_hn.py`, `pipeline.py`, `prepare_models.py`, `embed_month.py`, `export_*.py`, `compress_embedder.py`, `train_lm.py` |
| `experimentos/` | Scripts for the experiments in the log (`bench_*.py`, `make_eval.py`) |
| `resultados/` | Experiment outputs (`results_*.txt`) |
| `eval/` | Evaluation query sets |
| `doc/` | Documentation (`doc/investigacion/`: the prior research) |
| `data/`, `emb/`, `models/`, `out/` | Not in git: dataset, embeddings, models and built archive |

## Quick start (Windows, from this folder)
```bash
uv venv .venv --python 3.12
uv pip install --python .venv -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv -r requirements.txt
.venv\Scripts\activate
python scripts/prepare_models.py                 # models into models/
python scripts/download_hn.py                    # the whole dataset into data/ (resumable, SHA-256)
python scripts/pipeline.py --years 2006-2026     # build the archive in out/ (days; resumable)
cd hnz && cargo build --release && cd py && maturin develop --release   # CLI and Python package
```
```python
import hnz
with hnz.open("out", ram="4GB") as hn:
    for h in hn.search("how do I deal with burnout", k=5, since="2015", until="2018"):
        print(h.date, h.by, h.text[:100])
```
More details (requirements, CLI, HTTP, environment variables): [hnz/README.md](hnz/README.md).
