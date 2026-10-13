# Investigación en arXiv (prioridad Google / Microsoft)

IDs de arXiv verificados contra la API de arXiv (export.arxiv.org) el 2026-10-13. Las cifras
marcadas con ≈ son de memoria de los papers y no se releyeron.

## Lo que SÍ sirve para este proyecto

| Técnica | Paper | Quién | Idea | Uso aquí |
|---|---|---|---|---|
| TurboQuant | 2504.19874 | Google Research 2025 | Rotación aleatoria + cuantizador escalar óptimo + residual QJL 1 bit; sin entrenamiento | Base de **turbovec** (índice 2/4 bit). Evaluado en E00 |
| Matryoshka (MRL) | 2205.13147 | Google + UW 2022 | Embeddings truncables | EmbeddingGemma 2 lo trae: 768→512→256→128 |
| EmbeddingGemma | 2509.20354 | Google 2025 | Modelo 300M, MRL, robusto a cuantización | v1 evaluado (E05/E07) |
| **EmbeddingGemma 2** | (model card, 2026-10-06) | Google DeepMind | 270M texto (130M transformer + 140M tabla), MRL, 8K ctx | **Elegido** (mejor calidad, E07) |
| RaBitQ / ext-RaBitQ | 2405.12497 / 2409.09913 | NTU | 1 bit/dim + 2 floats, estimador insesgado | Probado en faiss: peor que binario asimétrico en calidad/byte |
| Language Modeling Is Compression | 2309.10668 | Google DeepMind 2023 | LM + codificación aritmética comprime mejor que gzip/zstd | **Nivel frío del texto** (E03) |
| Training LLMs over Neurally Compressed Text | 2404.03626 | Google DeepMind 2024 | "Equal-info windows": bloques codificados decodificables por separado | Diseño de acceso aleatorio con LM |
| FineZip / LLMZip | 2409.17141 / 2306.04050 | — | LM compression práctica (batching, fine-tuning) | Ideas de batching |
| Inverted index compression | 1908.10598 (Pibiri & Venturini) | — | Elias-Fano particionado, BP128 ≈ 2-5 bits/docID | BM25 compacto (E04) |
| SPANN / SPFresh | 2111.08566 / 2410.14452 | Microsoft | Centroides en RAM + listas en disco | Enrutado por particiones (E08) |
| Theoretical Limitations of Embedding-Based Retrieval | 2508.21038 | Google DeepMind 2025 | Vectores pequeños no pueden representar todos los top-k | Mantener BM25 junto a vectores |
| GraphRAG | 2404.16130 | Microsoft 2024 | Grafo de entidades/comunidades jerárquico con resúmenes por nivel | Capa de exploración por temas (E08) |

## Lo que NO sirve (y por qué)

| Técnica | Paper | Motivo |
|---|---|---|
| DSI / DSI++ (modelo = índice) | 2202.06991 / 2212.09744 (Google) | No escala: en 8,8 M pasajes queda por debajo de dual encoders (2305.11841); no devuelve texto literal |
| NCI / MEVI | 2206.02743 / 2309.13335 (Microsoft) | Igual: ~320k docs |
| FM-index / r-index / Succinct | 0712.3360 / 1705.10382 | 25-40 % del texto en inglés no repetitivo: peor que zstd en bloques |
| ColBERTv2 / PLAID / XTR / MUVERA | 2112.01488 / 2205.09707 / 2304.01982 / 2405.19504 | Multi-vector: ≥ 1 KB/doc |
| SPLADE / uniCOIL | 2109.10086 / 2106.14807 | 2-4× el tamaño de BM25 |
| PQ / Distill-VQ / RepCONC | 2204.00185 / 2110.05789 | PQ de 8-32 B medido: calidad muy baja (E05) sin entrenar el encoder |
| Gist tokens / ICAE / xRAG / vec2text | 2304.08467 / 2307.06945 / 2405.13792 / 2310.06816 | Compresión con pérdida: no recupera texto literal |
| BitFunnel (Bing) | SIGIR'17 | Firmas bit-sliced: rápidas, no más pequeñas |

## Jev / "System One" (TypeSafe AI)
Modelo de API (early access sept-2026) no autorregresivo que devuelve decisiones tipadas
(Choice / Score / Noul). **No comprime ni es un modelo de embeddings.** Su papel equivalente en un
sistema open source es un **reranker cross-encoder** (puntuar relevancia de candidatos) o un
clasificador zero-shot (etiquetar). Ver experimento de reranker en la bitácora.
