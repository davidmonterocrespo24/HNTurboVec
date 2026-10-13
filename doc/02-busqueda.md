# Búsqueda (índices)

Detalle numérico en [BITACORA.md](BITACORA.md) (E04-E07).

## Qué hemos aprendido

1. **Medir calidad real, no solo fidelidad.** "recall vs float" castiga mucho la cuantización y la
   truncación; con etiquetas reales (título → comentarios del mismo hilo) la pérdida es mínima.
2. **EmbeddingGemma 2 (Google, oct-2026) es el mejor modelo probado**: P@10 .634 en float, frente a
   .578 de bge-small y .474 de potion.
3. **Binario asimétrico** (documento en 1 bit/dim, consulta en float): 768d = 96 B/doc con P@10 .624
   (−1,6 % vs float de 3.072 B). Truncar (MRL) a 512d: 64 B, .607. A 256d: 32 B, .543.
4. **Re-embeber el top-100** recupera toda la calidad incluso a 256d (.639), pero en CPU cuesta ~7 s
   por consulta con Gemma 2 → solo viable con GPU.
5. **PQ (8-32 B) y RaBitQ** quedan por debajo del binario asimétrico a igual tamaño.
6. **BM25 compacto** (sin stopwords, Elias-Fano/varint): ~37 B/doc (1,8 GB) vs 83 B de tantivy.
   Calidad .490 en esta tarea (temática); imprescindible para nombres/términos exactos.
7. **Fusión:** RRF a ciegas empeora con Gemma 2 (.607 < .624) → usar fusión ponderada / BM25 como
   refuerzo, a calibrar.
8. **El cuello de botella es generar embeddings**: Gemma 2 ~15 docs/s en CPU con texto real de HN →
   49,3 M ≈ 38 días. Hay que acelerar (ONNX int8, OpenVINO en la iGPU Arc/NPU) o usar GPU.

## Pendiente de medir (antes de cerrar el diseño)
- turbovec 2/4 bit con EmbeddingGemma 2 en la prueba de calidad.
- Búsqueda en 2 etapas: Hamming sobre los primeros 256 bits → rescore asimétrico 768.
- Reranker cross-encoder (alternativa open source a Jev/System One) sobre el top-N fusionado.
- Velocidad de EmbeddingGemma 2 en ONNX/OpenVINO (iGPU/NPU).
