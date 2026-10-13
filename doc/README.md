# HNTurboVec — documentación del proyecto

**Objetivo:** tener *todo* Hacker News (~49,3 M ítems) en el **mínimo espacio en disco** posible,
con **búsqueda tipo Google/RAG** (consulta libre, palabras, nº de resultados, rango de años/fechas)
y **recuperación del texto literal** de los resultados. La preparación puede tardar; lo que
importa es el uso diario: **rápido, preciso y compacto**. Implementación en **Rust**.

## Cómo está organizada esta carpeta

| Documento | Contenido |
|---|---|
| [BITACORA.md](BITACORA.md) | **Registro cronológico de TODOS los experimentos** (qué se probó, script, resultado). Consultar antes de probar algo para no repetir. |
| [01-almacenamiento-texto.md](01-almacenamiento-texto.md) | Compresión del texto/metadatos: zstd, bloques, orden, LM como compresor. |
| [02-busqueda.md](02-busqueda.md) | Índices de búsqueda: vectores (binario, PQ, RaBitQ, turbovec), BM25, calidad real, modelos de embeddings. |
| [03-etiquetas-jerarquia.md](03-etiquetas-jerarquia.md) | Idea de ~1000 etiquetas + grafo de niveles: compresión, enrutado, jerarquía. |
| [04-investigacion-arxiv.md](04-investigacion-arxiv.md) | Papers revisados (prioridad Google/Microsoft), qué sirve y qué no. |
| [05-arquitectura.md](05-arquitectura.md) | Diseño del sistema que se está construyendo en Rust (decisiones + motivos). |

## Estado actual (actualizar en cada paso)

- [x] Medición del tamaño de HN y de la línea base (investigación previa en Hermes).
- [x] Benchmarks de almacenamiento, vectores, BM25, calidad real, LM, etiquetas (mes 2024-01).
- [x] EmbeddingGemma 2 en la iGPU (OpenVINO): ~204 docs/s → todo HN en ~2,8 días (E10, E13).
- [x] Sistema `hnz` en Rust: segmentos con nivel zstd (caliente) y nivel LM (frío), BM25,
      vectores binarios, búsqueda híbrida con filtros, CLI + HTTP + web, verificación sin pérdida.
- [x] LM compresor determinista propio (Pythia int8, AVX-VNNI, codificador aritmético) (E17).
- [~] Optimización: núcleo VNNI 2x4, caché L2, BM25 más compacto, `retier` (implementado, midiendo).
- [ ] Construcción completa de los 628 meses (`pipeline.py`; días de CPU/iGPU).
- [ ] Reranker cross-encoder (equivalente open source a Jev/System One) — no probado aún.

## Entorno

- Python de experimentos: `.venv` (uv, Python 3.12). Paquetes: numpy, pyarrow, duckdb, zstandard,
  brotli, faiss-cpu, fastembed, tantivy, torch (CPU), transformers, sentence-transformers 6.1,
  torchvision, turbovec, openvino.
- Datos: dataset `open-index/hacker-news` de Hugging Face (Parquet mensual, 628 ficheros).
  Descargado `data/2024/2024-01.parquet` (382.681 ítems).
- Embeddings cacheados en `emb/` (`*.npy`) para no recalcular.
- Máquina: Intel Core Ultra 7 265H (16 núcleos, iGPU Arc, NPU), 31 GB RAM, sin GPU NVIDIA.
- Rust: toolchain `stable-x86_64-pc-windows-gnu` (instalación de usuario, sin Visual Studio).
