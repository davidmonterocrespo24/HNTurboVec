# Bitácora de experimentos

Regla: **cada experimento se anota aquí** (fecha, script, datos, resultado, conclusión) para no
repetir pruebas. Los números "49,3 M" son extrapolaciones lineales desde el mes de muestra.
Muestra por defecto: `data/2024/2024-01.parquet` = **382.681 ítems** (346.509 comentarios,
36.093 stories), texto 125 MB, ~335 B/ítem de texto crudo.

---

### E00 — Investigación previa (sesión Hermes, 2026-10-13)
- HN: `maxitem` = 50.064.927; ~96,8 % de ids con datos → **~49,3 M ítems**; ~86 % comentarios.
- JSON crudo ≈ 24 GB; dump Parquet de `open-index/hacker-news` = 12,4 GB (252 B/ítem).
- Línea base: zstd-19 + diccionario 64 KB **por registro** = 185 B/reg → **9,1 GB**, 3 µs/lectura.
- turbovec (TurboQuant) 384d: 2-bit 101 B/vec (4,99 GB), 4-bit 198 B/vec (9,78 GB). 3-bit = mismo
  coste que 4-bit (padding) → no usar. Necesita `calibrate()`. Índice plano → hay que trocear.
- Un embedding NO contiene el texto (192 B de vector vs 330 B de texto; vec2text es lossy).
- Jev / "System One" (TypeSafe): modelo de API que devuelve decisiones tipadas (Choice/Score/Noul).
  No comprime. Equivalente open source útil aquí: **reranker cross-encoder** (puntúa candidatos).

### E01 — Almacenamiento de texto (`bench_store.py` → `results_store.txt`)
| Variante | B/ítem | 49,3 M | leer 1 bloque |
|---|---|---|---|
| JSON por registro, zstd19+dict (línea base reproducida) | 181,2 | 8,94 GB | 2-3 µs |
| Columnar por registro (metadatos en varints delta) | 168,4 | 8,30 GB | 3 µs |
| + transformación HTML reversible (94,5 % de registros) | 166,7 | 8,22 GB | 3 µs |
| Bloques de 16, orden id | 125,5 | 6,19 GB | 11-25 µs |
| Bloques de 64, orden id | 121,4 | 5,98 GB | 34 µs |
| Bloques de 256, orden id | 118,5 | 5,84 GB | 0,13-0,25 ms |
| Bloques de 1024, orden id | 115,8 | 5,71 GB | 0,7 ms |
| Bloques de 16 / 64 / 256 / 1024, **orden por hilo** | 123,9 / 118,0 / 114,4 / 113,1 | 6,11 / 5,82 / 5,64 / 5,58 GB | — |
| zstd-22 + dict, bloque 256, hilo | 114,4 | 5,64 GB | sin mejora |
| brotli-11 (ventana 16 MB), bloque 256, hilo | 113,8 | 5,61 GB | 0,54 ms |
| xz-9e, bloque 4096, hilo | 106,6 | 5,26 GB | ms |
| xz-9e TODO junto (suelo, sin acceso aleatorio) | 91,0 | 4,49 GB | — |

**Conclusión:** lo que más rinde es pasar de registro a **bloques pequeños** (181 → 124 B).
Compresores más fuertes (zstd-22, brotli, xz) apenas aportan. La transformación HTML aporta ~1 %.

### E02 — Metadatos y texto por separado (comando ad hoc, 2026-10-13)
- Metadatos (id, time, parent, type, score, descendants, id de autor) en columnas delta por
  bloques de 1024: **5,9 B/ítem**. Diccionario de autores: 0,25 MB (46.084 autores/mes).
- Solo texto (título+cuerpo), zstd19+dict por registro: 136,6 B/reg (crudo 335 B).

### E03 — LM como compresor (`bench_lm.py`, idea de DeepMind arXiv 2309.10668)
Cota de codificación aritmética: bits = Σ −log2 p(token) + 2 bytes de cierre; cada registro solo.
| Modelo | Registros | B/reg texto | bits/byte | velocidad |
|---|---|---|---|---|
| SmolLM2-135M | 300 | 52,9 | 1,26 | (CPU compartida) |
| SmolLM2-135M | 100 | 46,3 | 1,28 | 66 ms/reg **codificando** (pasada paralela) |
→ con metadatos (~6 B) + URL (~3 B): **~55-62 B/ítem ≈ 2,7-3,1 GB** para todo HN (vs 124 B zstd).
**Pendiente:** medir la *descompresión*, que es secuencial token a token (más lenta) y la
determinación bit a bit (codificador y decodificador deben calcular probabilidades idénticas).

### E04 — Índice de palabras (BM25) (`bench_lex.py` → `results_lex.txt`)
20,1 M tokens, 184.433 términos, 14,91 M postings (39/doc); las 200 palabras más frecuentes = 47 %.
| Variante | B/doc | 49,3 M |
|---|---|---|
| Todos los términos + tf, varint+zstd | 51,9 | 2,56 GB |
| Todos los términos + tf, Elias-Fano | 53,6 | 2,64 GB |
| Sin 200 stopwords + tf, varint+zstd | **37,0** | **1,82 GB** |
| Sin 200 stopwords, booleano, varint+zstd | 31,1 | 1,53 GB |
| tantivy real (freq, sin posiciones) | 83,3 | 4,11 GB |

### E05 — Fidelidad de vectores comprimidos (`bench_vec.py` → `results_vec.txt`; Gemma 2 en `results_g2.txt`)
recall@10 frente a la búsqueda exacta float32 *del mismo modelo* (300 consultas = títulos).
Velocidad de embeddings en CPU: potion ~800-4450 docs/s; bge-small ~31-35; EmbeddingGemma v1 ~11;
**EmbeddingGemma 2 ~14-17 docs/s** (65 docs/s solo con frases cortas).
| Modelo / dim | int8 | 4-bit | binario | bin+rescore100 | bin+rescore300 | PQ 32 B | RaBitQ |
|---|---|---|---|---|---|---|---|
| potion 512d | .994 | .909 | .508 | .876 | .941 | .497 | .676 |
| bge-small 384d | .991 | .857 | .429 | .862 | .954 | .346 | .602 |
| EmbeddingGemma v1 768d | .993 | .919 | .538 | .957 | .996 | .400 | .715 |
| **EmbeddingGemma 2 768d** | .993 | .900 | .479 | .878 | .931 | .408 | .708 |
**Ojo:** esta métrica mide "parecido al float", no calidad. Truncar dimensiones (MRL) se ve
muy penalizado aquí pero apenas pierde calidad real (ver E07). PQ de 8-32 B: malo siempre.

### E06 — Puntuación asimétrica y rescore con int8 truncado (comando ad hoc)
Consulta en float, documentos en 1 bit (sin coste extra): Gemma v1 768d .651, bge 384d .560
(fidelidad). Rescore con int8 de 128-256d guardado: limitado por la truncación → no compensa.

### E07 — CALIDAD real (`bench_quality.py` → `results_quality_bge.txt`, `results_g2.txt`)
Consulta = título de una story; relevantes = comentarios de ESE hilo. 200 consultas, 16.667 docs
(6.667 relevantes + 10.000 distractores). P@10 / R@100.
| Variante | potion | bge-small | **EmbeddingGemma 2** |
|---|---|---|---|
| float32 dim completa | .474 / .380 | .578 / .433 | **.634 / .496** (3072 B) |
| binario asimétrico dim completa | .430 | .538 | **.624 / .497 (96 B)** |
| binario + re-embed top-100 dim completa | .471 | .575 | .636 |
| 512d binario asimétrico / + re-embed | — | — | .607 / .639 (64 B) |
| 256d float / binario asim. / + re-embed | .468 / .408 / .470 | .541 / .489 / .573 | .614 / .543 / **.639 (32 B)** |
| 128d float / binario asim. / + re-embed | .432 / .339 / .448 | .475 / .370 / .541 | .502 / .366 / .571 |
| BM25 | .490 / .339 | .490 | .490 / .339 |
| Híbrido RRF (BM25 + binario) | .501 / .388 | **.585** / .449 | .607 / .484 (peor que vector solo) |
**Conclusiones:** EmbeddingGemma 2 es el mejor. Binario asimétrico 768d (96 B) ≈ float (−1,6 %).
Re-embed del top-100 recupera todo incluso a 256d, pero en CPU cuesta ~7 s/consulta → no viable
sin GPU. Con Gemma 2, RRF con BM25 empeora: hay que fusionar con pesos, no RRF a ciegas.

### E08 — Etiquetas (1000 temas k-means) + enrutado + jerarquía (`bench_tags.py` → `results_tags.txt`)
Embeddings potion de los 382.681 ítems (4.450 docs/s). k-means esférico 1000: mediana 329 docs
por etiqueta, máx. 17.804. Coste: 1 etiqueta/doc = 0,06 GB; 3 = 0,18 GB (49,3 M).
Enrutado (corpus completo del mes, P@10 de búsqueda completa = .167):
| 3 etiquetas/doc, buscar en N | P@10 | % escaneado |
|---|---|---|
| N=1 / 3 / 5 / 10 / 20 / 50 | .100 / .124 / .127 / .136 / .144 / .153 | 0,3 / 0,9 / 1,5 / 2,9 / 5,6 / 13,3 % |
| Solo etiquetas, sin vectores | **.004** | — |
Jerarquía 1000→40 áreas: hay temas claros (Python/pandas, vivienda, copyright/DMCA, moderación),
pero potion agrupa también "por tono" ("nice, lovely, glad") → usar mejor modelo + nombres con LLM.

### E09 — Compresión con bloques ordenados por tema (`bench_store_topic.py` → `results_topic_*.txt`)
| Orden | bloque 16 | 64 | 256 |
|---|---|---|---|
| id | 125,5 | 121,4 | 118,5 |
| hilo | 123,9 | 118,0 | 114,4 |
| tema | 124,5 | 119,6 | 116,0 |
| hilo agrupado por tema | 123,5 | 117,5 | **113,5** |
**Conclusión:** agrupar por tema aporta < 1 %. El diccionario de zstd ya captura el vocabulario.

### E10 — Velocidad de EmbeddingGemma 2 por dispositivo (`export_gemma2_ov.py`, `bench_embed_ir.py`, `bench_embed_speed.py`)
- El ONNX de `onnx-community/embeddinggemma-2-ONNX` (multimodal) **falla en GPU/NPU** (entradas de
  imagen/vídeo/audio de tamaño 0). Solución: exportar **solo la parte de texto** desde PyTorch a
  OpenVINO IR (formas estáticas batch 32 × 128 tokens, pesos fp16). Coseno vs PyTorch: 0,99998.
- **Hallazgo:** sentence-transformers carga el modelo en **bfloat16**, que esta CPU emula → lento.
  Hay que forzar `.float()`.
- Textos reales de HN (media 60 tokens, 15,3 % truncados a 128):
| Dispositivo | docs/s | 49,3 M |
|---|---|---|
| **OpenVINO GPU (Arc 140T iGPU, fp16)** | **168,8** | **81 h (3,4 días)** |
| OpenVINO CPU | 19,9 | 28,6 días |
| PyTorch CPU float32 | 10,7 | 53 días |
| onnxruntime CPU (ONNX fp16) | 4,5 | — |
| NPU (AI Boost) | falla al compilar | — |
**Conclusión:** embeber todo HN es viable en la iGPU (~3 días, mejorable con buckets de longitud y
pesos int8). La CPU queda libre en paralelo para la compresión LM.

### E11 — Qué LM usar como compresor (`bench_lm2.py` → `results_lm2.txt`)
300 registros por mes; bits/byte del texto (título+cuerpo), B/reg incluye 2 B de cierre del
codificador; velocidades con PyTorch CPU float32 (dominadas por overhead del framework en modelos
pequeños; un motor propio int8 será bastante más rápido). Round-trip del tokenizador: 100 % en todos.
| Modelo | 2024-01 bpb / B | 2015-06 bpb / B | Decodif. lote 10 (ms/paso) | Codificar 49,3 M (PyTorch) |
|---|---|---|---|---|
| pythia-14m | 1,637 / 70,4 | 1,619 / 65,5 | 6,0 | 4,9 días |
| pythia-31m | 1,488 / 64,2 | 1,462 / 59,3 | 6,6 | 6,7 días |
| pythia-70m | 1,401 / 60,6 | 1,374 / 55,9 | 30,0 | 13,7 días |
| pythia-160m | 1,273 / 55,2 | 1,242 / 50,7 | 27,4 | 34,5 días |
| gpt2 (124M) | 1,444 / 62,4 | 1,421 / 57,7 | 51,2 | 42,3 días |
| SmolLM2-135M | 1,217 / 52,9 | 1,187 / 48,6 | 36,7 | 46,4 días |
| SmolLM2-360M | 1,129 / 49,2 | 1,094 / 44,9 | 105,9 | 107,8 días |
**Conclusiones:**
- A más parámetros, mejor compresión pero coste casi lineal al codificar y al decodificar.
- Pythia (entrenado en el Pile, que incluye HN hasta ~2020) no muestra memorización notable en 2015.
- Cálculo de MACs/token (la cabeza de vocabulario domina en modelos pequeños): pythia-31m ~18 M,
  pythia-70m ~45 M, pythia-160m ~124 M, SmolLM2-135M ~134 M.
- **Elección inicial: arquitectura GPT-NeoX (familia Pythia), motor propio en Rust con matmul int8
  exacta (determinista), modelo configurable.** Por defecto pythia-70m (equilibrio: ~61 B/reg de texto,
  decodificación estimada ~50-90 ms para un top-10 con motor propio). Mejora futura: entrenar un LM
  pequeño específico de HN (en GPU) que supere a SmolLM2-135M con menos parámetros.

### Notas de datos (2026-10-13)
- Tipos: 1 story, 2 comment, 3 poll, 4 pollopt, 5 job.
- 2010-06 y 2015-06 tienen filas `deleted` (sin contenido), `kids` no vacíos y saltos de línea crudos
  en `text`; 2024-01 no tiene ninguno de los tres → el formato debe soportar ambos casos.
- **Decisión:** se guardan sin pérdida texto, título, url, autor, fecha, score, descendants, parent,
  poll/parts y `dead`. No se guardan las filas `deleted` (vacías) ni el orden de `kids` (la estructura
  del hilo se reconstruye con `parent`; se pierde solo el orden en que HN mostraba las respuestas).

### E12 — Núcleo en Rust: meta + texto zstd (`hnz build/verify/stats/bench`)
Segmentos de 1 mes, bloques de 16, zstd-19 + diccionario 112 KB, metadatos en bloques de 1024.
| Mes | Docs | B/ítem | 49,3 M | Transformación HTML | Verificación |
|---|---|---|---|---|---|
| 2010-06 | 80.363 (+2.366 borrados omitidos) | 128,0 | 6,31 GB | 35,7 % moderno + 64,3 % antiguo | **idénticos 100 %** |
| 2015-06 | 166.390 (+5.246 borrados) | 133,1 | 6,56 GB | 100 % moderno | **idénticos 100 %** |
| 2024-01 | 382.681 | 122,2 | 6,03 GB | 98,2 % moderno, 1,8 % crudo | **idénticos 100 %** |
- Desglose 2024-01: texto 116,0 B, meta 4,84 B, autores 0,60 B, índice de bloques 0,52 B, diccionario 0,30 B.
- Construir un mes: ~4,5 s (Python: minutos). Lectura aleatoria sin caché: ~67 µs/registro (1 hilo).
- **Dos dialectos de HTML en HN:** antiguo (~2007-2012: `&#38; &#60; &#62;`, comillas/apóstrofos/barras
  sin escapar) y moderno (`&amp; &#x27; &#x2F; …`). **Enlaces truncados** ("…", 60-95 caracteres) son
  el 30-40 % de los enlaces → se guardan como (url, longitud).
