# Arquitectura del sistema `hnz` (Rust)

Estado: **implementado y funcionando sobre 3 meses de muestra** (2010-06, 2015-06, 2024-01).
Evidencias numéricas en [BITACORA.md](BITACORA.md). Uso en [../hnz/README.md](../hnz/README.md).

## Principios
1. **Lo que más pesa es el texto** → niveles por antigüedad (como un LSM): zstd para lo reciente
   (lectura en µs), **LM + codificación aritmética** para lo antiguo (la mitad de tamaño).
2. **Búsqueda híbrida**: vectores binarios de EmbeddingGemma 2 + BM25 compacto (cada uno falla
   donde el otro acierta; DeepMind 2508.21038). Fusión calibrada (E14).
3. **Todo en orden temporal**: un rango de fechas = un rango contiguo de documentos.
4. **Sin pérdida y verificable**: `hnz verify` compara cada registro con el Parquet original.
5. **Determinismo** del nivel LM: matmul entera exacta + matemática propia → mismos bits en
   cualquier CPU x86-64 (con o sin AVX-VNNI).

## Segmentos (directorio autocontenido, inmutable)
| Fichero | Contenido | B/ítem (2024-01) |
|---|---|---|
| `manifest.json` | rango de ids/fechas, nivel, modelo LM | — |
| `meta.bin/.idx`, `authors.zst` | id/time/parent/tipo/dead/score/descendants/autor/poll/parts, bloques de 1024, varint delta + zstd | ~5,5 |
| `text.bin/.idx/.dict` (**nivel zstd**) | bloques de 16 registros, columnar, HTML→forma compacta reversible (2 dialectos, enlaces truncados), zstd-19 + diccionario | ~116 |
| `lm.bin/.idx/.json` (**nivel LM**) | cada registro: cabecera varint (nº tokens, modo) + flujo aritmético; grupos de 32 | ~61 (+meta) |
| `bm25.fst/.post/.norms/.json` | FST de términos (df=1 inline), postings en bloques de 128 con BitPacker4x, normas 1 B, MaxScore | ~49 |
| `vec.bin/.has` | signo de los 768 dims de EmbeddingGemma 2 (MRL permite 512/256) | 96 |

## Consulta
```
consulta (texto, k, desde/hasta, tipo, autor)
  ├─ embedding de la consulta: EmbeddingGemma 2 (IR OpenVINO, CPU) desde Rust ........ ~45 ms
  ├─ por segmento que solapa las fechas (rayon):
  │    vectores: Hamming 256 bits → 2000+ candidatos → puntuación asimétrica 768 bits
  │    BM25: MaxScore con IDF global, saltando bloques fuera del rango de docIDs
  ├─ fusión: score = coseno + 0,15·bm25/(bm25+20) → top-k con filtros exactos ........ ~6 ms
  └─ recuperar: zstd (µs) | nivel LM: TODOS los resultados en un lote (KV-cache, pool spin);
     modo fragmento (64 tokens) por defecto, texto completo bajo demanda; contexto del hilo
     (título de la story raíz, decodificando solo hasta el primer salto de línea)
```

## Caché por capas
- **L1 RAM**: registros del nivel LM ya descomprimidos completos (20.000 entradas).
- **L2 disco**: pendiente (guardar en zstd los registros fríos consultados a menudo).
- **L3**: el almacén (zstd o LM).

## Construcción
1. `embed_month.py` (iGPU Arc, OpenVINO, buckets de longitud): ~204 docs/s → 49,3 M en ~2,8 días.
2. `hnz build --tier zstd|lm` → meta + texto + BM25. Nivel LM: ~97 reg/s (16 hilos, AVX-VNNI)
   → la CPU comprime mientras la iGPU embebe.
3. `hnz index --emb … --ids …` → vectores binarios.
4. `hnz verify` → comparación registro a registro con el original.

## Modelos (cuentan en el tamaño total del sistema)
| Modelo | Uso | Tamaño |
|---|---|---|
| EmbeddingGemma 2 (texto) IR fp16 + tokenizer | embedding de consultas | 542 MB + 32 MB (int8: ~0,27 GB, pendiente) |
| pythia-70m `.hnzlm` int8 + tokenizer | (des)compresión del nivel frío | 72 MB + 3,5 MB |

## Decisiones abiertas / mejoras
- [ ] Núcleos int8 más eficientes (hoy ~27 GMAC/s/hilo) → compresión y descompresión LM más rápidas.
- [ ] Caché L2 en disco para el nivel LM.
- [ ] Bits por nivel (768 en caliente, 512 en frío) y BM25 más compacto (sin tf / Elias-Fano).
- [ ] Capa de etiquetas/grafo por temas (exploración, enrutado) — opcional (E08).
- [ ] Pipeline completo para los 628 meses (descarga → embeddings → segmentos por año).
