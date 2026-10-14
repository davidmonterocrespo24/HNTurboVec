# HNTurboVec — documentación del proyecto

**Objetivo:** tener *todo* Hacker News (~49,3 M ítems) en el **mínimo espacio en disco** posible,
con **búsqueda tipo Google/RAG** (consulta libre, palabras, nº de resultados, rango de años/fechas)
y **recuperación del texto literal** de los resultados. La preparación puede tardar; lo que
importa es el uso diario: **rápido, preciso y compacto**. Implementación en **Rust** (`hnz/`).

## Cómo está organizada esta carpeta

| Documento | Contenido |
|---|---|
| [BITACORA.md](BITACORA.md) | **Registro cronológico de TODOS los experimentos** (qué se probó, script, resultado). Consultar antes de probar algo para no repetir. |
| [01-almacenamiento-texto.md](01-almacenamiento-texto.md) | Compresión del texto/metadatos: zstd, bloques, orden, LM como compresor. |
| [02-busqueda.md](02-busqueda.md) | Índices de búsqueda: vectores (binario, PQ, RaBitQ, turbovec), BM25, calidad real, modelos de embeddings. |
| [03-etiquetas-jerarquia.md](03-etiquetas-jerarquia.md) | Idea de ~1000 etiquetas + grafo de niveles: compresión, enrutado, jerarquía. |
| [04-investigacion-arxiv.md](04-investigacion-arxiv.md) | Papers revisados (prioridad Google/Microsoft), qué sirve y qué no. |
| [05-arquitectura.md](05-arquitectura.md) | El sistema construido en Rust: formato, consulta, cachés, robustez, tamaños. |

## Estado actual

- [x] Medición del tamaño de HN y de la línea base (investigación previa en Hermes).
- [x] Benchmarks de almacenamiento, vectores, BM25, calidad real, LM, etiquetas (mes 2024-01).
- [x] EmbeddingGemma 2 en la iGPU (OpenVINO): ~204 docs/s → todo HN en ~2,8 días (E10, E13).
- [x] Sistema `hnz` en Rust: segmentos con nivel zstd (caliente) y nivel LM (frío), BM25,
      vectores binarios, búsqueda híbrida con filtros, CLI + HTTP + web, verificación sin pérdida.
- [x] LM compresor determinista propio (Pythia int8, AVX-VNNI, codificador aritmético) (E17, E20).
- [x] Cachés L1/L2, `retier`, embedder de consultas int8, latencias medidas (E23, E24).
- [x] Revisión adversarial (141 agentes): 60 hallazgos confirmados y corregidos (E25).
- [x] 2ª y 3ª revisión (18 + 4 hallazgos corregidos); segmentos de muestra reconstruidos con el
      binario final y verificados; mismo resultado byte a byte con dos binarios distintos (E26).
- [x] Reranker cross-encoder (equivalente open source a Jev/System One) probado (E26): poca
      ganancia para su coste en CPU → no se usa por defecto.
- [ ] Construcción completa de los 239 meses (`pipeline.py`; días de CPU/iGPU).

## Resultado con la muestra de 3 meses (637.046 ítems, E26)

| | Caliente (zstd) | Frío (LM) |
|---|---|---|
| Tamaño por ítem (texto + metadatos + BM25 + vectores) | 267,6 B | 187-188 B |
| Todo HN (~49,3 M; 2 años en caliente) | **≈ 10,0 GB** + modelos 0,38 GB (≈ 10,4 GB) | |
| Búsqueda híbrida (embedding + búsqueda + texto, k=10) | 24-29 ms | 280-350 ms (fragmentos) |
| Texto completo de 10 resultados fríos | — | 464 ms; 25 ms en caché RAM, 79 ms en caché disco |
| Abrir el archivo | 180 ms | |

Calidad (P@10 híbrida, título → comentarios del hilo, todo el archivo): 0,278 / 0,208 / 0,249 con
consultas de 2024 / 2010 / 2015. Con más meses esta cifra baja aunque la búsqueda no empeore: los
hilos parecidos de otros años cuentan como fallos (E26).

## Entorno

- Python: `.venv` creado con `uv` (Python 3.12) e instalado desde [`../requirements-torch.txt`](../requirements-torch.txt) (torch CPU, índice de PyTorch) y [`../requirements.txt`](../requirements.txt)
  (versiones fijadas). Modelos: `.venv\Scripts\python prepare_models.py` (o con el entorno activado; ver `hnz/README.md`).
- Datos: dataset `open-index/hacker-news` de Hugging Face: **239 Parquet mensuales** (2006-10 →
  2026-08) más ficheros horarios de los últimos días (`today/…`) y estadísticas (628 ficheros en total).
  Muestras descargadas: 2010-06, 2015-06 y 2024-01.
- Embeddings cacheados en `emb/` (`*.npy`) para no recalcular.
- Descargas de Hugging Face con `HF_HUB_DISABLE_XET=1`: el descargador `hf_xet` muere a mitad de
  descarga en esta máquina (E26); `pipeline.py`, `prepare_models.py` y `bench_rerank.py` lo fijan.
- Máquina: Intel Core Ultra 7 265H (16 núcleos, iGPU Arc 140T, NPU), 31 GB RAM, sin GPU NVIDIA.
- Rust: toolchain `stable-x86_64-pc-windows-gnu` (instalación de usuario, sin Visual Studio); el
  binario requiere CPU x86-64-v3 (AVX2).
