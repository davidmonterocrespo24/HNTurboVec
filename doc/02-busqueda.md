# Búsqueda (índices)

Detalle numérico en [BITACORA.md](BITACORA.md) (E04-E07, E10, E13, E14, E18, E19, E22, E25, E26).

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
   recorren los metadatos del rango (exacto); el resto se sobre-muestrea y se filtra en orden de
   puntuación con una columna de banderas de 1 B/doc en RAM (`meta.flags.zst`), reintentando solo
   el segmento que se queda corto (E25, E26). Sin texto: listado de lo más reciente.
10. **Segmentos con distintos bits (768 caliente / 512 frío) ya son comparables**: el coseno
    renormalizado en cada prefijo MRL queda a 0,003 de P@10 de tenerlo todo a 768; calibrar
    (regresión o momentos) hunde a los fríos y empeora las consultas sobre años viejos (E26).
11. **Validar con consultas cuyas respuestas estén en distintos niveles** (2024, 2010, 2015): con un
    solo conjunto, cualquier sesgo contra lo antiguo parece una mejora (E26).
12. **La métrica título → hilo baja al crecer el archivo** (los hilos parecidos de otros años cuentan
    como fallos): comparar versiones con la misma composición del archivo (E26).
13. **Reranker cross-encoder** (MiniLM 80-120 MB, equivalente a Jev/System One): +5-10 % de P@10 por
    0,4-0,8 s/consulta en CPU → no por defecto. Cuidado con la fuga de la etiqueta: el reranker debe ver
    solo el propio documento, no el título del hilo (E26).

## Pendiente
- Rerankers mayores (jina, bge-reranker-v2-m3 int8): descarga fallida por la red (E26).
- Evaluar con consultas de palabras clave y preguntas (hoy solo títulos → hilo).
