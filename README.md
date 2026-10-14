# HNTurboVec

Todo **Hacker News** (~49,3 M ítems) comprimido **sin pérdida**, con **búsqueda híbrida** (semántica
+ palabras, nº de resultados, rangos de fechas, tipo y autor) y **recuperación literal** de los
resultados. Motor en Rust (`hnz`) que se usa desde la línea de órdenes, por HTTP o desde **Python**
(`import hnz`).

- Hoy: ~10,4 GB para todo HN (búsqueda de 0,03-0,3 s); plan medido de un perfil compacto de ~5 GB.
- Estado, decisiones y números: [doc/README.md](doc/README.md). Todos los experimentos:
  [doc/BITACORA.md](doc/BITACORA.md). Diseño de la librería: [doc/06-libreria.md](doc/06-libreria.md).

## Estructura

| Carpeta | Qué hay |
|---|---|
| `hnz/` | Motor en Rust (biblioteca + CLI `hnz.exe`) y paquete de Python en `hnz/py` |
| `scripts/` | Herramientas para construir el archivo: `download_hn.py`, `pipeline.py`, `prepare_models.py`, `embed_month.py`, `export_*.py`, `compress_embedder.py`, `train_lm.py` |
| `experimentos/` | Scripts de los experimentos de la bitácora (`bench_*.py`, `make_eval.py`) |
| `resultados/` | Salidas de los experimentos (`results_*.txt`) |
| `eval/` | Conjuntos de consultas de evaluación |
| `doc/` | Documentación (`doc/investigacion/`: la investigación previa) |
| `data/`, `emb/`, `models/`, `out/` | Fuera de git: dataset, embeddings, modelos y archivo construido |

## Inicio rápido (Windows, desde esta carpeta)
```bash
uv venv .venv --python 3.12
uv pip install --python .venv -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv -r requirements.txt
.venv\Scripts\activate
python scripts/prepare_models.py                 # modelos en models/
python scripts/download_hn.py                    # todo el dataset en data/ (reanudable, SHA-256)
python scripts/pipeline.py --years 2006-2026     # construir el archivo en out/ (días; reanudable)
cd hnz && cargo build --release && cd py && maturin develop --release   # CLI y paquete de Python
```
```python
import hnz
with hnz.open("out", ram="4GB") as hn:
    for h in hn.search("how do I deal with burnout", k=5, since="2015", until="2018"):
        print(h.date, h.by, h.text[:100])
```
Más detalles (requisitos, CLI, HTTP, variables de entorno): [hnz/README.md](hnz/README.md).
