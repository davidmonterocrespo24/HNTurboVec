# Arquitectura del sistema `hnz` (Rust)

Estado: **diseño en curso**; se actualiza con cada decisión. Evidencias en [BITACORA.md](BITACORA.md).

## Principios
1. **Lo que más pesa es el texto** → niveles por antigüedad: zstd (rápido) para lo reciente,
   LM + codificación aritmética (máxima compresión) para lo antiguo. Tipo LSM: los segmentos
   envejecen y se "compactan" a niveles más fríos.
2. **Búsqueda híbrida** (vectores binarios EmbeddingGemma 2 + BM25 compacto), porque cada una
   falla donde la otra acierta (DeepMind 2508.21038).
3. **Todo lo que no es texto se guarda en el mínimo de bits** y en orden temporal: un rango de
   fechas = un rango contiguo de documentos (sin índices extra).
4. Preparación lenta aceptable; **consulta diaria rápida**: mmap, SIMD, caché por capas.

## Segmentos
Un segmento = un año (el año en curso, por meses). Inmutable. Cada segmento es autocontenido:

| Fichero | Contenido | Coste aprox. |
|---|---|---|
| `manifest.json` | rango de ids/fechas, nivel, parámetros | — |
| `meta.*` | columnas id/time/parent/tipo/flags/score/descendants/autor en bloques de 1024, varint+zstd | ~6 B/ítem |
| `text.*` | **nivel caliente/templado:** bloques de 16/256 registros, columnar (título, url, texto con transformación HTML reversible), zstd-19 + diccionario del segmento | ~118-124 B/ítem |
| `lm.*` | **nivel frío:** cada registro codificado con LM + codificador aritmético; longitudes en varint por grupo | ~55-62 B/ítem (estim.) |
| `bm25.*` | diccionario de términos (FST) + postings en bloques de 128 bit-packed (docID delta + tf) + máximos por bloque + normas de longitud (1 B) | ~38-45 B/ítem |
| `vec.bin` | códigos binarios de EmbeddingGemma 2 (signo por dimensión), 768 bits = 96 B (o 512 bits = 64 B en niveles fríos) | 64-96 B/ítem |

## Consulta
```
consulta ("rust async runtime", k=10, desde=2015, hasta=2018, filtros opcionales by:/site:/type:)
  │
  ├─ 1. Segmentos que solapan el rango de fechas → rango de docIDs por segmento (búsqueda binaria)
  ├─ 2. Embedding de la consulta (EmbeddingGemma 2, OpenVINO CPU, ~10 ms) y términos BM25
  ├─ 3. En paralelo por segmento (rayon):
  │      vectores: Hamming sobre los primeros 256 bits → top-C → puntuación asimétrica 768 bits (LUT)
  │      BM25: MaxScore/WAND sobre postings, saltando bloques fuera del rango
  ├─ 4. Fusión ponderada (calibrada) → top-k global
  ├─ 5. (opcional) reranker cross-encoder sobre el top-N (equivalente open source a Jev "Score")
  └─ 6. Recuperar resultados: metadatos + texto (zstd: µs | LM: decodificación en lote)
         + contexto del hilo (título de la story, recorriendo parent) + fragmento resaltado
```

## Caché por capas
- **L1 RAM:** LRU de registros/bloques ya descomprimidos.
- **L2 disco:** registros del nivel LM ya decodificados, guardados en zstd rápido (tamaño acotado).
- **L3:** el almacén real (zstd o LM).

## Construcción (offline, puede tardar días)
1. Descargar Parquet mensuales (`open-index/hacker-news`).
2. Embeddings EmbeddingGemma 2 en la **iGPU Arc** (OpenVINO, buckets de longitud): ~3 días para todo.
   En paralelo, la **CPU** comprime el nivel frío con el LM.
3. `hnz build`: por segmento → meta, texto (nivel según antigüedad), BM25, códigos binarios.
4. Reconstrucción incremental: el mes en curso se añade como segmento nuevo; al envejecer se
   recompacta a un nivel más frío.

## Modelos (cuentan en el tamaño total del sistema)
| Modelo | Uso | Tamaño |
|---|---|---|
| EmbeddingGemma 2 (texto, 270M) | embeddings de consulta | ~0,5 GB fp16 (int8: ~0,27 GB) |
| LM compresor (por elegir, E11) | descomprimir nivel frío | ~15-140 MB int8 |
| Reranker cross-encoder (opcional) | precisión del top-N | ~20-100 MB |

## Decisiones abiertas
- [ ] LM compresor: modelo y velocidad de decodificación (E11).
- [ ] 768 vs 512 bits en niveles fríos; Hamming vs asimétrico en la 1ª etapa (E12, mes completo).
- [ ] Pesos de fusión BM25+vector; ¿reranker? (E12).
