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
  poll/parts y `dead`. [Corregido en E25: las filas `deleted` SÍ se guardan como metadatos, porque
  descartarlas rompía la cadena de los hilos.] No se guardan las filas `deleted` (vacías) ni el orden de `kids` (la estructura
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

### E13 — Embeddings de un mes completo en la iGPU (`embed_month.py` → `results_embed_2024-01.txt`)
364.891 ítems con contenido (de 382.681), buckets de longitud 32/64/128/256 tokens (3,8 % truncados a 256):
| bucket | docs | docs/s |
|---|---|---|
| 32 / 64 / 128 / 256 | 110.360 / 105.631 / 91.087 / 57.813 | 565 / 357 / 163 / 81 |
**Total 204 docs/s (29,8 min) → 49,3 M en ~2,8 días** (vs 168,8 docs/s sin buckets).

### E14 — Búsqueda completa en Rust (`hnz search` / `hnz eval` → `results_eval_rust.txt`)
629.434 docs (3 meses), embedding de consulta con OpenVINO desde Rust (IR fp16, CPU).
Latencias (consulta "burnout…"): embedding 45 ms, búsqueda 6,5 ms, recuperar+descomprimir 1,0 ms;
carga del modelo 3,8 s (una vez). BM25 solo: búsqueda 1,3-1,7 ms.
Calidad (300 consultas = títulos de 2024-01, relevantes = comentarios del hilo, k=10):
| Configuración | P@10 | MRR | ms/consulta |
|---|---|---|---|
| vector 2 etapas (Hamming 256 bits → asimétrico 768) | 0,291 | 0,574 | 6,3 |
| vector exacto (asimétrico sobre todo) | 0,293 | 0,577 | 14,5 |
| BM25 | 0,201 | 0,466 | 3,4 |
| **híbrida w=0,15 c=20 (elegida)** | **0,318** | 0,591 | 5,7 |
(rejilla completa en el fichero; w ∈ 0,05-0,4, c ∈ 5-20: todas entre 0,304 y 0,318)
Tamaños por doc (2024-01): vectores 96 B + bitmap 0,13 B; BM25 49 B (fst 5,2 + postings 42,9 + normas 1,0).

### E15 — Representación del texto para el LM (`bench_lm_repr.py` → `results_lm_repr.txt`, pythia-70m)
| Representación | 2024-01 B/reg (tokens) | 2010-06 B/reg (tokens) |
|---|---|---|
| a) HTML original | 61,1 (87,1) | 56,8 (77,3) |
| b) compacta con caracteres de control | 61,4 (71,4) | 59,6 (72,0) |
| c) "natural" (cota, NO reversible) | 57,0 (72,2) | 55,3 (72,7) |
| d) reversible: `\n\n` + marcadores de control | 57,7 (72,5) | 56,3 (73,1) |
| **e) reversible: `\n\n` + enlaces como `<url>`** | **57,2 (72,4)** | **55,6 (73,0)** |
**Elegida e):** −6 % bytes y −17 % tokens (= decodificación más rápida) frente al HTML.

### E16 — Cuantización del LM (`bench_lm_quant.py` → `results_lm_quant.txt`, representación e)
| Modelo | fp32 | W8A32 | **W8A8** |
|---|---|---|---|
| pythia-70m | 57,21 | 57,42 | **57,94 (+1,3 %)** |
| pythia-160m | 52,64 | 52,77 | **53,21 (+1,1 %)** |
**Decisión:** pesos int8 por fila + activaciones int8 por token → matmul ENTERA exacta: resultado
idéntico en cualquier CPU (escalar, AVX2 o AVX-VNNI) = determinismo para el codificador aritmético.

### E17 — Motor LM propio en Rust (nivel frío) (`hnz lmtest`, modelo `export_lm.py` → `.hnzlm`)
GPT-NeoX/Pythia W8A8: pesos int8 por fila, activaciones int8 por token, matmul ENTERA exacta;
LayerNorm/softmax/GELU con exp/erf propias y sumas en orden fijo; RoPE precalculada en el fichero.
Codificador aritmético binario (estilo LZMA, 16 bits de probabilidad, token = 16 decisiones sobre
el árbol del vocabulario), cierre mínimo (recorte de ceros). Representación "e" (E15).
- pythia-70m exportado: 71,7 MB; pythia-160m: 164 MB.
- **Corrección:** 200/200 registros reales de 2024-01 comprimidos y descomprimidos idénticos;
  los bytes comprimidos son iguales con núcleo escalar y AVX-VNNI (prueba de determinismo).
- **Tamaño:** 60,7 B/registro (título+url+texto, con cabecera), 1,467 bits/byte, 77,4 tokens/reg
  (zstd en bloques de 16 en el mismo mes: ~116 B/reg de texto).
- **Velocidad de compresión** (16 hilos, un registro por hilo): escalar 9 reg/s → AVX-VNNI con
  micronúcleo 4×32: **97 reg/s (7.520 tokens/s)**.
- **Velocidad de descompresión** (lote de 10 registros en paralelo, KV-cache, lockstep):
  | Implementación | ms/lote (mediana) |
  |---|---|
  | rayon, 16 hilos | 1.205-1.865 |
  | pool "spin" propio, 16 hilos (sobresuscripción con rayon) | 17.304 (patológico) |
  | **pool spin, 8 hilos (por defecto = mitad de los lógicos)** | **1.033** |
  | pool spin, 6 / 4 hilos | 1.083 / 1.164 |
  Por paso (10 activos): matmul capas 0,9 ms, atención 0,7 ms, cabeza de vocabulario 0,9 ms,
  softmax+decodificador 0,3 ms. El lote dura lo que su registro más largo (pasos = sus tokens).
**Pendiente:** núcleos más eficientes (~27 GMAC/s/hilo, lejos del pico), modo "fragmento" (decodificar
solo los primeros ~64 tokens para la lista de resultados) y caché por capas.

### E18 — Primer segmento FRÍO real y BM25 más compacto (`hnz build --tier lm`, `hnz rebm25`)
**2010-06 en nivel LM** (pythia-70m): 80.355 registros con LM + 8 con zstd de reserva, 79,6 tokens/reg,
61,2 B/reg de texto; construcción 1.118 s (72 reg/s, con la CPU compartida con la iGPU).
Desglose por doc: texto LM 62,5 B · BM25 56,9 B · metadatos 5,4 B → **124,9 B/doc** (sin vectores).
Ahora **el BM25 pesa casi tanto como el texto** → variantes sobre 2024-01 (300 consultas):
| Variante BM25 | B/doc | P@10 palabras | P@10 híbrida |
|---|---|---|---|
| base (tf, todos los términos) | 49,1 | 0,223 | 0,316 |
| sin tf (booleano) | 39,9 | 0,192 | 0,306 |
| df ≤ 5 % | 47,8 | 0,224 | 0,316 |
| **df ≤ 2 % (elegida)** | **43,3** | 0,220 | **0,315** |
| sin tf + df ≤ 5 % | 39,0 | 0,192 | 0,305 |
**Decisión (REVERTIDA en E25):** no indexar términos presentes en > 2 % de los documentos (−12 %, calidad igual);
la frecuencia del término sí importa (−3 % en híbrida sin ella).
Embeddings de 2010-06 / 2015-06 en la iGPU: 126-131 docs/s (con la CPU ocupada por el LM).

### E19 — Bits de los vectores (MRL) en el mes completo (`hnz index --bits`, `hnz eval`)
Con BM25 df ≤ 2 %, 300 consultas, 382.681 docs:
| Bits | B/doc | P@10 vector | P@10 híbrida | MRR híbrida |
|---|---|---|---|---|
| 768 | 96 | 0,291 | 0,315 | 0,588 |
| **512** | **64** | 0,278 | **0,307 (−2,5 %)** | 0,581 |
| 384 | 48 | 0,258 | 0,298 (−5,4 %) | 0,572 |
| 256 | 32 | 0,235 | 0,289 (−8,3 %) | 0,556 |
**Decisión:** 768 bits en el nivel caliente, **512 en el frío**.

### E20 — Optimización del motor LM (`hnz lmtest`)
- Núcleo AVX-VNNI 2 filas × 4 tokens con reducción `hadd`: compresión +12 % (77→86 reg/s),
  descompresión −4 %; bytes idénticos (determinismo entre núcleos confirmado).
- Perfil de compresión (1 hilo, 30 registros): la **atención** era el 43 % del tiempo (cuadrática en
  registros largos, ~0,8 GFLOPS). Cambio a **claves transpuestas** [cabeza·dim][posición] y bloques de
  32 claves / 64 dims con acumuladores en registros → atención 1,0 s → 0,4 s.
  Es un cambio de la definición numérica → `MATH_VERSION = 2` guardado en `lm.json`; un segmento con
  otra versión se rechaza con un mensaje claro (hay que recomprimirlo).
- Resultado (16 hilos): **compresión 122 reg/s (9.462 tokens/s)**, +42 %; descompresión de 10
  registros en lote: **mediana 531 ms** (antes 656 ms), p90 1.055 ms.
- **Problema detectado:** el pool "spin" acapara la CPU cuando hay OTROS procesos pesados (una
  verificación tardaba horas a 5 reg/s compitiendo con la compresión). Arreglo: girar solo ~20 µs y
  después `yield_now()` hasta 3 ms, luego dormir.

### E21 — Segmento caliente definitivo (2024-01, zstd + BM25 df ≤ 2 % + 768 bits)
Verificación 100 % idéntica. **261,7 B/doc**: texto 116,8 · vectores 96,1 · BM25 43,3 · meta 5,5.

### E22 — turbovec (TurboQuant) vs índice binario asimétrico, a igual tamaño (`bench_turbovec_rerank.py`)
Mes completo 2024-01 (364.891 docs con vector), 300 consultas, solo vector (sin BM25):
| Índice | B/vec | P@10 | MRR |
|---|---|---|---|
| float32 768d (techo) | 3072 | 0,294 | 0,563 |
| **binario asimétrico 768 bits (hnz)** | **96** | **0,293** | **0,577** |
| binario asimétrico 512 bits (hnz) | 64 | 0,280 | 0,558 |
| binario asimétrico 256 bits | 32 | 0,235 | 0,514 |
| turbovec 2-bit 256d (calibrado) | ~64 | 0,285 | 0,557 |
| turbovec 2-bit 384d | ~96 | 0,301 | 0,563 |
| turbovec 4-bit 256d | ~128 | 0,277 | 0,543 |
| turbovec 2-bit 768d | ~192 | 0,310 | 0,575 |
**Conclusión:** a igual tamaño turbovec da +1,8-2,7 % de P@10 pero MRR igual o peor; con 300 consultas el
error estándar de P@10 es ~0,015 → diferencias dentro del ruido. El binario asimétrico (sin
entrenamiento, búsqueda de 2 etapas en Rust) se mantiene; turbovec queda como alternativa futura.
La parte (2) del script (reranker cross-encoder, equivalente a Jev/System One) **no llegó a ejecutarse**:
el sistema detuvo los procesos en segundo plano por falta de memoria RAM.

### E23 — EmbeddingGemma 2 de consultas en int8 (`compress_embedder.py`, NNCF de Intel)
Pesos int8 asimétricos por canal: **545 MB → 276 MB**, **~2,5x más rápido en CPU**, coseno con fp16:
media 0,99980, mínimo 0,99942 (100 consultas). `hnz` usa el IR int8 si existe.

### Incidencia (2026-10-13 ~06:30): procesos detenidos por falta de memoria
El sistema detuvo en segundo plano la recompresión v2 de 2010-06/2015-06 (iba por 50k/80k de 2010) y
el benchmark de reranker. `out/seg-2010-06` quedó incompleto (sin manifest: el motor lo ignora).
Pendiente de relanzar, de uno en uno para no agotar la RAM.

### E24 — Segmentos fríos v2 verificados al completo + latencias reales del servidor
- Recompresión v2 (512 bits): 2010-06 en 625 s (129 reg/s, **181,6 B/doc**), 2015-06 en 1.368 s
  (122 reg/s, **183,4 B/doc**: texto LM 65,3 · vectores 64,1 · BM25 48,2 · meta 5,8).
- **Verificación completa con el LM: 80.363 + 166.390 = 246.753 registros idénticos al original.**
  (verify a 24 reg/s: el lote de 64 espera a su registro más largo → mejora futura: lotes continuos).
- Latencias con `hnz serve` (modelos cargados una vez; embedder int8), k=10 (`results_latency.txt`):
| Consulta | Total | Embedding | Búsqueda | Recuperar/descomprimir |
|---|---|---|---|---|
| caliente 2024, híbrida | 49-51 ms | 25-37 | 6-15 | 6-11 |
| caliente 2024, palabras | 11-20 ms | — | 2-5 | 9-15 |
| frío 2010 / 2015, fragmentos | 430 / 326 ms | ~30 | 4-7 | 393 / 284 |
| todo el archivo, híbrida | 44-49 ms | ~30 | 5-6 | 3-13 |
| frío, texto completo, 1ª vez | 601-641 ms | | | 568-603 |
| frío, texto completo, repetida (L1) | 116 → **46 ms** | | | 74 → **3** |
| frío, texto completo, tras reinicio (L2) | 655 → **83 ms** | | | 627 → **54** |
| `/item/<id>` frío sin caché | ~200 ms | | | |
- **Bug encontrado y corregido:** la caché L2 nunca se escribía en Windows (un fichero abierto en modo
  append no admite `set_len` → `L2::open` fallaba y el error se ignoraba). Ahora se abre en lectura/
  escritura y se avisa si falla. Además: caché de títulos de stories y un único lote LM para
  resultados + títulos del hilo.


### E25 — Revisión adversarial del sistema y correcciones (workflow de 141 agentes)
6 revisores en paralelo (formato sin pérdida, LM/codificador, concurrencia/unsafe/cachés, búsqueda,
CLI/servidor/pipeline, documentación), solo lectura; cada hallazgo lo intentaron refutar 2
verificadores con enfoques distintos (+1 de desempate). Resultado: **65 hallazgos, 60 confirmados,
5 refutados**. Correcciones principales (commit "Correcciones de la revisión adversarial"):
| Problema confirmado | Corrección |
|---|---|
| Filtros --by/--type/dead aplicados tras recortar candidatos (podía devolver 0 resultados) | se aplican al generar candidatos: recorrido exacto de metadatos para autor y tipos raros; sobre-muestreo + reintento para el resto |
| Filas `deleted` descartadas → cadena de hilos rota (141 comentarios en 2010-06) | se guardan como metadatos (bit en meta; 1 byte en el nivel LM) |
| Poda BM25 por df en cada segmento (E18) → términos frecuentes de un periodo desaparecen | sin poda (decisión E18 revertida) |
| Tokenizador: "php", "html", "net" como stopwords; "C"/"R" descartadas | buscables |
| La doc prometía un "CRC por registro" que no existía | 1 byte de comprobación por registro LM (`rec_check`) |
| Caché L2: clave solo por nombre (texto de otro ítem tras reconstruir), sin validar cabecera ni bloquear | clave con `build_id`, cabecera validada, acceso exclusivo; además no funcionaba en Windows (E24) |
| `retier`: borraba el original tras verificar 200 docs; `--seg .` destruía el segmento; restos `.retier` servidos como segmentos | conserva `<seg>.old`, verifica 2.000 o todo, limpia si falla, ruta canónica; el motor ignora carpetas transitorias |
| Segmentos solapados (mensual + anual) → resultados duplicados y `get` fallido | el motor los detecta y se niega a abrir |
| `build` sobre carpeta existente: un corte deja un segmento mezclado | construcción atómica en `<seg>.building` |
| `k` sin límite en HTTP (agota memoria) | k ≤ 500 (100 por HTTP) |
| Pool no seguro ante pánicos; `HNZ_LM_THREADS=0` divide por cero | `catch_unwind` + relanzar; mínimo 1 hilo |
| id del modelo con muestreo 1/97 bytes | hash completo (se acepta el id antiguo) |
| Un solo modelo LM por archivo | un modelo por segmento |
| `pipeline.py`: ~27 GB de RAM al unir un año; 768 bits en frío; año "terminado" aunque falten meses; límite caliente relativo al tramo | unión en streaming (memmap) + `hnz index` por mmap; 512 bits en frío; reconstruye si cambian los meses; límite absoluto respecto al dataset |
| `make_eval.py` con `>` en PowerShell genera UTF-16 | `--out` en UTF-8; `hnz eval` tolera BOM |
| Docs desactualizadas ("628 meses": son 239; tokenizador de modelos no generado; requisito x86-64-v3 no documentado) | docs reescritas; `prepare_models.py`; `requirements.txt` |
Además: lote continuo (`decode_stream`) en verify/retier/rebm25 (antes cada lote esperaba a su
registro más largo: verify a 24 reg/s).
Notas de medición: las velocidades de E17 y E20 se midieron con muestras y cargas de CPU distintas
(E17 con trabajos en paralelo); comparar solo dentro de cada experimento.


### E26 — 2ª, 3ª y 4ª revisión, segmentos finales, puntuación entre niveles y reranker
**Revisiones.** 2ª revisión de las correcciones de E25 (mismo método): 19 hallazgos, **18 confirmados**,
1 refutado. 3ª revisión (un revisor sobre el commit resultante): **4 más**. Todo corregido:
| Problema confirmado | Corrección |
|---|---|
| `build` borraba cualquier carpeta `--out` existente (p.ej. `out/` o `data/`) | se niega si no es un segmento (sin `manifest.json`) |
| Sustitución no atómica: un borrado a medias dejaba un segmento roto | el viejo se aparta con un rename (`<seg>.old.<id>`) y se borra al final |
| Modo palabras + autor/tipo raro rellenaba k con docs sin ningún término | solo docs con señal: bm25 > 0 (palabras), vector (vectorial), una de las dos (híbrida) |
| El filtro descomprimía un bloque de metadatos por candidato (casi todo `meta.bin` por consulta) | columna de banderas de 1 B/doc en RAM, guardada como `meta.flags.zst` (0,1-0,16 B/doc) |
| Saturación global: un segmento con < k válidos no se ampliaba si otros sumaban k | reintento por segmento (×4 hasta ×16), solo si su lista se recortó |
| Sin texto, en híbrido/vectorial no salía "lo más reciente primero" | listado: sin texto no se calcula embedding y se recorre del más nuevo al más viejo |
| Texto con solo palabras vacías en modo palabras ("how to") devolvía lo más reciente (3ª) | con texto pero sin señal utilizable: ningún resultado |
| Montículo de `cand+1` por trozo de 64K docs (memoria con k grande) | acotado por el tamaño del trozo |
| `retier/index --bits` sin validar: con 1-7 bits `Engine::open` entraba en pánico (3ª) | múltiplo de 64; `vec.bin` se comprueba contra el manifiesto al abrir |
| Sin vector, la puntuación vectorial valía -1 en la fusión (3ª) | 0 (neutra), como antes de E25 |
| `retier` desde dentro del segmento, `.retier.building` sin limpiar, `sources.json` perdido, 768 bits en frío | se niega, limpia, lo conserva con el nivel nuevo, recorta a 512 bits por MRL (`--bits`) |
| `pipeline.py`: el mes en curso se bajaba a medias y nunca se actualizaba | compara tamaños con el dataset; rebaja, recalcula embeddings (`.src.json`) y reconstruye el año |
| `pipeline.py` reconstruía (o devolvía a zstd) un segmento ya enfriado | nunca por el nivel: `--retier` enfría; bits según el nivel real; no reconstruye si faltan Parquet ya archivados |
| `pipeline.py` sin red / `IndexError` sin Parquet | Parquet locales; el índice usa los meses de `sources.json` |
| `uv` no resolvía `torch==…+cpu` mezclando índices | `requirements-torch.txt` aparte |
| La doc decía que el byte de comprobación protege fragmentos y títulos | aclarado: solo registros completos; los fragmentos, el id del modelo + `MATH_VERSION` |

**4ª revisión** (workflow de 51 agentes: 4 revisores —pipeline, correcciones Rust, validez de los
experimentos, documentación— y 2 verificadores por hallazgo + desempate): 20 hallazgos, **18
confirmados**, 2 refutados. Correcciones:
| Problema confirmado | Corrección |
|---|---|
| `pipeline.py` adoptaba embeddings sin registro con el tamaño del Parquet recién rebajado: vectores viejos para siempre | solo se adoptan si son posteriores al Parquet |
| Un segmento sin `sources.json` se reconstruía con los Parquet locales que quedaran (perdía los meses archivados) | no se toca; registro provisional `seg-YYYY.sources.json` antes del build, que se adopta si el proceso muere justo después |
| Una ejecución normal rebajaba los Parquet archivados y volvía a embeber (días de iGPU) años ya completos | año completo y al día con el dataset: no se baja ni se embebe nada |
| Un fallo de red pasaba a modo sin conexión en silencio (límite caliente/frío según los Parquet locales) | se detiene; `--offline` explícito con el último año visto en línea (`out/pipeline_state.json`) |
| `VecIndex::open` aceptaba un `vec.bin` de otro ancho si era múltiplo del código: pánico del servidor | tamaño exacto `n_docs × bytes` y `vec.has` suficiente |
| `vec_cand = Some(0)` entraba en pánico; `--c-bm25 0` daba NaN; fechas imposibles (2015-13) se desbordaban | candidatos ≥ k; `c > 0`, `w ≥ 0`; fechas validadas |
| Conclusión de mixbits exagerada (diferencias del orden del ruido) | bootstrap pareado y conclusión más prudente (abajo) |
| Reranker: latencia sin la descarga de candidatos y ganancia sin prueba de ruido | latencia de extremo a extremo e IC 95 % (abajo) |
| Latencias del servidor de una sola consulta y con caché caliente | medianas de 30 consultas distintas con el servidor recién arrancado y caché vacía (abajo) |
| Docs: `python` sin activar el entorno; texto 116,8 B/doc (no 116,6); modelos 0,38 GB (no 0,35); 05 desactualizado; byte de comprobación presentado como garantía | corregidos |

**Segmentos finales** (`results_build_v4.txt`, binario final):
- 2024-01 (zstd): **382.681 registros idénticos** al original.
- 2010-06 (LM) reconstruido con el binario final: **todos los ficheros idénticos byte a byte** al v3
  del binario anterior, ya verificado al completo (**82.729 registros idénticos**, 2.768 s) →
  determinismo también entre binarios distintos.
- 2015-06 (LM): 171.636 ítems (5.246 deleted), 63,0 B/reg de texto LM; verificación completa: **171.636 registros idénticos** (5.971 s, 29 reg/s).
- Los dos meses LM se comprimieron a la vez (la mitad de los hilos cada uno): 2010-06 en 1.325 s y
  2015-06 en 2.079 s; uno tras otro (E24) habrían sido ~625 + ~1.370 s, así que en paralelo no se gana.

**Medidas finales** (`results_final.txt`, CPU en reposo):
- Tamaño (`hnz stats`): caliente **267,6 B/doc** (texto 116,8 · vectores 96,1 · BM25 49,2 · meta 5,5 · banderas 0,1);
  frío **187,0 / 188,2 B/doc** (texto LM 64,4 · vectores 64,1 · BM25 53,9 · meta 5,9). El BM25 frío
  crece ~5 B/doc al quitar la poda (E25). Proyección para todo HN: **≈ 10,0 GB** + 0,38 GB de modelos (≈ 10,4 GB en total).
- Abrir el archivo: **180 ms** (antes ~950 ms construyendo las banderas al abrir).
- Latencias del servidor (`bench_serve.py`): mediana y p90 de 30 consultas distintas por caso, con el
  servidor recién arrancado y la caché L2 vacía (la 1ª versión de E26 medía una sola consulta con la
  caché caliente: los "29 ms" de todo el archivo eran un acierto de caché):
| Consulta (k=10 salvo indicación) | Mediana | p90 | Embedding | Búsqueda | Texto | Del nivel LM |
|---|---|---|---|---|---|---|
| caliente 2024, híbrida | 25,9 ms | 27,4 | 19,2 | 4,3 | 2,5 | 0 % |
| caliente 2024, palabras | 3,1 ms | 5,1 | — | 0,6 | 2,4 | 0 % |
| todo el archivo, híbrida (consultas de 2024) | 103 ms | 158 | 20,1 | 4,1 | 73,5 | 19 % |
| todo el archivo, híbrida (consultas de 2010) | 264 ms | 311 | 21,7 | 3,9 | 239 | 80 % |
| frío 2010 / 2015, híbrida, fragmentos | 280 / 284 ms | 321 / 310 | 20-24 | 2,8 | 256 / 257 | 100 % |
| frío 2010, texto completo | 525 ms | 1.092 | 20,9 | 2,8 | 502 | 100 % |
| todo el archivo, k=100, fragmentos | 1.306 ms | 1.846 | 24,3 | 10,0 | 1.271 | 60 % |
| tipo job + palabras (recorrido exacto) | 189 ms | 332 | — | 13,8 | 176 | 77 % |
| autor, híbrida | 217 ms | 306 | 20,0 | 13,9 | 184 | 73 % |
| sin texto: lo más reciente con filtros | 4,0 ms | 6,0 | — | 0,4 | 1,7 | 0 % |
| texto completo frío: 1ª vez / repetida (L1) / tras reiniciar (L2) | 527 / 24 / 53 ms | | | | 503 / 2 / 29 | |
| `/item/<id>` frío completo, sin caché | 326 ms | 589 | | | | |
  Lo que cuesta es el nivel LM: ~25 ms por resultado frío en un lote de 10 fragmentos.

**Calidad: por qué baja P@10 y qué NO hay que corregir.**
- `hnz eval` (consultas de 2024-01, todo el archivo): vector 0,255, híbrida 0,278 frente a 0,291 / 0,315
  de E14/E19. Control con solo el segmento 2024-01: 0,291 / 0,314 → el motor está bien.
- En E14 los meses fríos aún no tenían vectores y no aportaban distractores. Ahora sí, y con la
  etiqueta "comentarios del mismo hilo" un hilo parecido de 2010 cuenta como fallo.
- ¿Influye mezclar 768 bits (caliente) y 512 (frío)? Simulación con los embeddings reales
  (`bench_mixbits.py` → `results_mixbits.txt`), P@10 vectorial con 3 conjuntos de consultas
  (`make_eval.py`; relevantes en 2024, 2010 o 2015):
| Puntuación de los segmentos de 512 bits | consultas 2024 | 2010 | 2015 | media |
|---|---|---|---|---|
| todo a 768 bits (referencia) | 0,265 | 0,197 | 0,234 | 0,232 |
| **coseno renormalizado en el prefijo MRL (hnz)** | 0,259 | 0,194 | 0,235 | **0,229** |
| calibración por regresión (s768 ≈ 0,936·s512 + 0,019) | 0,268 | 0,177 | 0,210 | 0,218 |
| calibración por momentos (s768 ≈ s512 − 0,006) | 0,265 | 0,183 | 0,221 | 0,223 |
| sin renormalizar la consulta | 0,292 | 0,065 | 0,070 | 0,142 |
  Con bootstrap pareado de consultas (IC 95 % de la diferencia de P@10 frente a lo que hace hnz):
  todo a 768 bits +0,003 [−0,001, +0,007] en las 900 consultas (no significativo); regresión
  −0,011 [−0,014, −0,008] y momentos −0,006 [−0,008, −0,004] (significativamente peores: ganan en las
  consultas de 2024, +0,009 y +0,006, y pierden más en las de 2010 y 2015). Conclusión: ninguna
  calibración lineal mejora el coseno renormalizado por prefijo MRL (cada prefijo se entrena como un
  coseno propio), y tenerlo todo a 768 bits apenas cambiaría nada. **Lección: no validar con un
  único conjunto de consultas cuyas respuestas están todas en el mismo nivel.**
- Consultas sobre el nivel frío con el sistema real (`hnz eval`, todo el archivo): 2010-06 vector
  0,193, híbrida 0,208; 2015-06 vector 0,236, híbrida 0,249.
- Esta métrica baja al crecer el archivo aunque la búsqueda no empeore: comparar versiones siempre
  con la misma composición.

**Reranker cross-encoder** (equivalente open source a Jev/System One; `bench_rerank.py`: reordena el
top-30 real de hnz pedido al servidor):
- 1ª ejecución: P@10 0,444 (+59 %) → **falso**: el texto que veía el reranker incluía
  "re: <título del hilo>" y la consulta ES ese título (fuga de la etiqueta). Corregido: el reranker
  ve solo el propio documento, como los embeddings y BM25.
- Resultado real con IC 95 % (bootstrap pareado; diferencia de P@10 frente a hnz) y latencia de extremo
  a extremo (mediana; servidor recién arrancado y caché vacía):
| Consultas | El reranker lee | hnz | MiniLM-L6 (80 MB) | MiniLM-L12 (120 MB) | RRF hnz + L12 | Latencia: hnz → con L6 / L12 |
|---|---|---|---|---|---|---|
| 2024-01 | texto completo | 0,279 | +0,015 [+0,004, +0,027] | +0,020 [+0,008, +0,033] | +0,023 [+0,015, +0,032] | 90 ms → 696 / 1.119 ms |
| 2024-01 | fragmentos | 0,279 | +0,017 [+0,005, +0,029] | +0,023 [+0,010, +0,035] | +0,024 [+0,016, +0,033] | 91 ms → 595 / 956 ms |
| 2015-06 | texto completo | 0,249 | +0,017 [+0,005, +0,030] | +0,019 [+0,006, +0,032] | +0,026 [+0,017, +0,035] | 249 ms → 1.212 / 1.616 ms |
| 2015-06 | fragmentos | 0,249 | −0,011 [−0,025, +0,003] | −0,013 [−0,026, +0,001] | +0,001 [−0,007, +0,010] | 264 ms → 668 / 848 ms |
- Lectura: con el texto completo el reranker mejora de forma significativa (+6-10 % relativo; mejor en
  fusión RRF con el orden de hnz) pero multiplica la latencia por 5-7. Con los fragmentos de 64 tokens
  (lo barato) no mejora en el nivel frío: el fragmento es demasiado corto para juzgar (en 2024 da igual
  porque el nivel caliente devuelve el texto entero). **Decisión: no por defecto**; un modo "preciso"
  opcional tendría que descomprimir el texto completo de los 30 candidatos.
- Rerankers mayores (jina v1 tiny/turbo, bge-reranker-v2-m3 int8): la descarga falló por timeouts de red.

**Incidencias:** el descargador `hf_xet` de huggingface_hub muere a mitad de descarga en esta
máquina (exit 127) → `HF_HUB_DISABLE_XET=1` en los scripts que descargan.
