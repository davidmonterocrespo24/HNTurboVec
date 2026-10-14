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
- [x] 2ª, 3ª y 4ª revisión (18 + 4 + 18 hallazgos corregidos); segmentos de muestra reconstruidos con el
      binario final y verificados al completo (637.046 registros idénticos al original); mismo
      resultado byte a byte con dos binarios distintos (E26).
- [x] Reranker cross-encoder (equivalente open source a Jev/System One) probado (E26): +6-10 % de
      P@10 pero multiplica la latencia por 5-7 (necesita el texto completo) → no se usa por defecto.
- [x] Todo el dataset descargado (`download_hn.py`: 239 meses, 12,42 GB, SHA-256 comprobados).
- [x] LM entrenado con HN en la iGPU: −9 % de texto con el mismo coste (E29).
- [x] Plan de **perfil compacto** medido (E27-E29): **~5 GB** para todo HN (5,74 sin pérdida, 5,24
      con −1 %) con consultas de ~4 s; documentado en [05-arquitectura.md](05-arquitectura.md), sin
      implementar por ahora.
- [ ] Construcción completa de los 239 meses (`pipeline.py`; días de CPU/iGPU).

## Resultado con la muestra de 3 meses (637.046 ítems, E26)

| | Caliente (zstd) | Frío (LM) |
|---|---|---|
| Tamaño por ítem (texto + metadatos + BM25 + vectores) | 267,6 B | 187-188 B |
| Todo HN (~49,3 M; 2 años en caliente) | **≈ 10,0 GB** + modelos 0,38 GB (≈ 10,4 GB) | |
| Búsqueda híbrida k=10 (embedding + búsqueda + texto; mediana, caché vacía) | 26 ms | 280 ms (fragmentos de 64 tokens) |
| Texto completo de 10 resultados fríos | — | 527 ms; 24 ms en caché RAM, 53 ms en caché disco |
| Abrir el archivo | 180 ms | |

En todo el archivo, la latencia depende de cuántos resultados salgan del nivel frío (~25 ms cada
uno): 103 ms de mediana con consultas sobre 2024 (2 de 10 fríos) y 264 ms con consultas sobre 2010
(8 de 10). Sin texto (listados con filtros): 4 ms.

Calidad (P@10 híbrida, título → comentarios del hilo, todo el archivo): 0,278 / 0,208 / 0,249 con
consultas de 2024 / 2010 / 2015. Con más meses esta cifra baja aunque la búsqueda no empeore: los
hilos parecidos de otros años cuentan como fallos (E26).

## Entorno

- Python: `.venv` creado con `uv` (Python 3.12) e instalado desde [`../requirements-torch.txt`](../requirements-torch.txt) (torch CPU, índice de PyTorch) y [`../requirements.txt`](../requirements.txt)
  (versiones fijadas). Modelos: `.venv\Scripts\python prepare_models.py` (o con el entorno activado; ver `hnz/README.md`).
- Datos: dataset `open-index/hacker-news` de Hugging Face: **239 Parquet mensuales** (2006-10 →
  2026-08) más ficheros horarios de los últimos días (`today/…`) y estadísticas (628 ficheros en total).
  Todo el dataset se baja con `download_hn.py` (~12,4 GB en `data/`): reanudable byte a byte si se
  corta (basta con volver a lanzarlo), versión fija del dataset y SHA-256 de cada fichero.
- Embeddings cacheados en `emb/` (`*.npy`) para no recalcular.
- Descargas de Hugging Face con `HF_HUB_DISABLE_XET=1`: el descargador `hf_xet` muere a mitad de
  descarga en esta máquina (E26); `pipeline.py`, `prepare_models.py` y `bench_rerank.py` lo fijan.
- Máquina: Intel Core Ultra 7 265H (16 núcleos, iGPU Arc 140T, NPU), 31 GB RAM, sin GPU NVIDIA.
- Rust: toolchain `stable-x86_64-pc-windows-gnu` (instalación de usuario, sin Visual Studio); el
  binario requiere CPU x86-64-v3 (AVX2).
