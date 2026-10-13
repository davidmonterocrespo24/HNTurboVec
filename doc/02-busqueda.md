# Búsqueda (índices)

Detalle numérico en [BITACORA.md](BITACORA.md) (E04-E07, E10, E13, E14, E18, E19, E22, E25).

## Qué hemos aprendido

1. **Medir calidad real, no solo fidelidad.** "recall vs float" castiga mucho la cuantización y la
   truncación; con etiquetas reales (título → comentarios del mismo hilo) la pérdida es mínima.
2. **EmbeddingGemma 2 (Google, oct-2026) es el mejor modelo probado**: P@10 .634 en float sobre el
   conjunto pequeño, frente a .578 de bge-small y .474 de potion.
3. **Binario asimétrico** (documento en 1 bit/dim, consulta en float): 768 bits = 96 B/doc casi igual
   que float (E07, E22). En el mes completo (E19): 768 bits P@10 híbrida .315; 512 bits (64 B) .307;
   384 bits .298; 256 bits .289 → **768 en caliente, 512 en frío**.
4. **Búsqueda en 2 etapas** (Hamming sobre 256 bits → asimétrico): pierde 0,7 % y es 2,3x más rápida
   que el asimétrico completo (E14).
5. **PQ (8-32 B) y RaBitQ** (faiss) no mejoran al binario asimétrico por byte. **turbovec** (TurboQuant)
   a igual tamaño queda empatado dentro del ruido (E22).
6. **BM25 compacto**: FST + postings BitPacker4x + normas de 1 B, sin posiciones: ~49 B/doc frente a
   83 de tantivy. **No se podan términos por df**: la poda por segmento (E18) hacía desaparecer términos
   frecuentes de un periodo (p.ej. "covid" en 2020) para la búsqueda por palabras (E25).
7. **Fusión** coseno + 0,15·bm25/(bm25+20): +9 % de P@10 sobre solo vectores (E14).
8. **Embeddings a escala**: EmbeddingGemma 2 solo-texto exportado a OpenVINO en la **iGPU Arc**:
   ~204 docs/s con buckets de longitud → todo HN en ~2,8 días (E13). Consultas en CPU con pesos int8:
   ~25-40 ms (E23).
9. **Filtros** (autor, tipo, dead, fechas) se aplican al generar candidatos: el autor y los tipos raros
   recorren los metadatos del rango (exacto); el resto se sobre-muestrea y reintenta (E25).

## Pendiente
- Reranker cross-encoder sobre el top-N (equivalente open source a Jev/System One): la prueba se
  interrumpió por falta de memoria (E22); repetir con menos RAM.
- Evaluar con consultas de palabras clave y preguntas (hoy solo títulos → hilo).
