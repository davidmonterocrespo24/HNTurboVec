# Arquitectura del sistema `hnz` (Rust)

Estado: **implementado y verificado sobre 3 meses de muestra** (2010-06 y 2015-06 en nivel frío,
2024-01 en nivel caliente). Evidencias numéricas en [BITACORA.md](BITACORA.md); uso en
[../hnz/README.md](../hnz/README.md). Revisión adversarial y correcciones: E25.

## Principios
1. **Lo que más pesa es el texto** → niveles por antigüedad (como un LSM): zstd para lo reciente
   (lectura en µs) y **LM + codificación aritmética** para lo antiguo (la mitad de tamaño). Un
   segmento se puede "enfriar" después con `hnz retier`.
2. **Búsqueda híbrida**: vectores binarios de EmbeddingGemma 2 + BM25 compacto (cada uno falla
   donde el otro acierta; DeepMind 2508.21038). Fusión calibrada (E14): coseno + 0,15·bm25/(bm25+20).
3. **Orden temporal**: los docIDs siguen el id de HN, así un rango de fechas es un rango contiguo.
4. **Sin pérdida y verificable**: `hnz verify` compara cada registro con el Parquet original (también
   las filas `deleted`, que se guardan como metadatos para no romper los hilos).
5. **Determinismo del nivel LM**: matmul int8 exacta + matemática propia en orden fijo → el mismo
   binario da los mismos bits en cualquier CPU compatible (requiere x86-64-v3 / AVX2; AVX-VNNI es
   opcional: con o sin él el resultado es idéntico). Cada registro lleva 1 byte de comprobación
   (se valida al descomprimirlo entero; los fragmentos no lo cubren) y al abrir se exige el mismo
   id de modelo y `MATH_VERSION`.

## Segmentos (carpeta autocontenida e inmutable; `build` es atómico vía `<seg>.building`)
| Fichero | Contenido | B/doc caliente (2024-01) | B/doc frío (2015-06) |
|---|---|---|---|
| `manifest.json` | rango de ids/fechas, nivel, modelo LM, `build_id` | — | — |
| `meta.bin/.idx`, `authors.zst` | id/time/parent/tipo/dead/deleted/score/descendants/autor/poll/parts, bloques de 1024, varint delta + zstd | 5,5 | 5,7 |
| `meta.flags.zst` | 1 B/doc (tipo, dead, deleted) para filtrar sin descomprimir; se carga en RAM al abrir | 0,10 | 0,16 |
| `text.bin/.idx/.dict` (**nivel zstd**) | bloques de 16, columnar, HTML→forma compacta reversible (2 dialectos, enlaces truncados), zstd-19 + diccionario | 116,8 | — |
| `lm.bin/.idx/.json` (**nivel LM**) | por registro: cabecera varint + byte de comprobación + flujo aritmético; grupos de 32 | — | 64,4 |
| `bm25.fst/.post/.norms/.json` | FST de términos (df=1 inline), postings en bloques de 128 con BitPacker4x, normas 1 B, MaxScore; sin poda (E25) | 49,2 | 53,9 |
| `vec.bin/.has` | signo de las dims de EmbeddingGemma 2: 768 bits (caliente) / 512 bits (frío, E19) | 96,1 | 64,1 |
| **Total** (`hnz stats`, E26) | | **267,6** | **188,2** |

Proyección para ~49,3 M ítems con los 2 últimos años en caliente (~9 M) y el resto en frío:
≈ 9 M × 267,6 B + 40,3 M × 188 B ≈ **10,0 GB**, más los modelos en ejecución (0,38 GB: embedder int8, LM y tokenizadores) → **≈ 10,4 GB** en total.

## Consulta
```
consulta (texto, k ≤ 500, desde/hasta, tipo, autor)
  ├─ sin texto: listado de lo más reciente que cumple los filtros (sin embedding)
  ├─ embedding: EmbeddingGemma 2 (IR OpenVINO int8, CPU) desde Rust ................ ~20-25 ms
  ├─ por segmento que solapa las fechas (rayon):
  │    · con autor o tipos raros (job/poll): recorrido exacto de metadatos + puntuación
  │    · si no: vectores (Hamming 256 bits → asimétrico) + BM25 (MaxScore con IDF global);
  │      filtros con la columna de banderas en RAM, en orden de puntuación; si el segmento no
  │      llega a k y su lista de candidatos se recortó, se repite solo él con ×4 y luego ×16
  │    · solo docs con señal: algún término (palabras), vector (vectorial), una de las dos (híbrida)
  ├─ fusión → top-k ........................................................ ~3-10 ms
  └─ recuperar: zstd (µs) | nivel LM: resultados + títulos del hilo en UN lote (KV-cache,
     pool de hilos propio); fragmento de 64 tokens del cuerpo por defecto, completo bajo demanda
```
Latencias medidas: ver la tabla de [README.md](README.md) y E26 en la bitácora.

## Caché por capas
- **L1 RAM**: registros del nivel LM ya descomprimidos (20.000) y títulos de stories.
- **L2 disco** (`<root>/_cache_l2.bin`, ≤ 512 MB, no cuenta en `hnz stats`): registros fríos ya
  descomprimidos, re-comprimidos con zstd rápido; clave con `build_id` del segmento, cabecera
  validada, acceso exclusivo (un proceso).
- **L3**: el almacén (zstd o LM).

## Robustez
- `build` atómico; el motor ignora `*.building`, `*.retier`, `*.old` y rechaza segmentos solapados.
- `retier` reconstruye en `<seg>.retier`, verifica (muestra de 2.000 o `--verify-all`) y deja el
  original como `<seg>.old` (solo lo borra con `--verify-all --delete-old`).
- Cambiar la matemática del LM incrementa `MATH_VERSION`: los segmentos antiguos se rechazan con un
  mensaje claro y hay que reconstruirlos desde el Parquet (`hnz build --tier lm`).
- Varios modelos LM pueden convivir en un archivo (cada segmento indica el suyo).

## Construcción (offline)
1. `prepare_models.py`: IR de EmbeddingGemma 2 (documentos en buckets, consultas int8), tokenizador y
   LM `.hnzlm`.
2. `pipeline.py --years 2006-2026`: descarga → embeddings en la iGPU (~204 docs/s, ~2,8 días para todo)
   → segmento por año (zstd los 2 últimos años del dataset, LM el resto; ~120 reg/s en CPU, ~4 días)
   → vectores (768/512 bits). Reanudable; reconstruye el año en curso si aparecen meses nuevos.
3. `hnz verify` por segmento (lote continuo en el nivel LM).

## Modelos (cuentan en el tamaño total del sistema)
| Modelo | Uso | Tamaño |
|---|---|---|
| EmbeddingGemma 2 (texto) IR int8 + tokenizer | embedding de consultas | 276 MB + 32 MB |
| EmbeddingGemma 2 IR fp16 por buckets | solo construcción (iGPU) | 4 × 542 MB (no hace falta para consultar) |
| pythia-70m `.hnzlm` int8 + tokenizer | (des)compresión del nivel frío | 72 MB + 3,5 MB |

## Pendiente / mejoras
- [ ] Construcción completa de los 239 meses (días de cómputo).
- [ ] LM específico de HN (entrenado en GPU): mejor compresión con menos parámetros.
- [x] Reranker cross-encoder (equivalente open source a Jev/System One), probado en E26: +6-10 % de
      P@10 (significativo) leyendo el texto completo de 30 candidatos, a 0,6-1,6 s por consulta en
      CPU; con fragmentos no mejora en el nivel frío → no por defecto; quedan por probar los grandes.
- [ ] Capa de etiquetas/grafo por temas (exploración, enrutado) — opcional (E08).
