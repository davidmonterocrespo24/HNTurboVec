# Etiquetas (~1000 temas) y grafo por niveles

Detalle numérico en [BITACORA.md](BITACORA.md) (E08, E09).

## Idea evaluada
Etiquetar el contenido con ~1000 etiquetas, agruparlas en niveles (1000 → 40 áreas) y usar eso
para compactar y/o buscar.

## Resultados
- **Espacio:** guardar 3 etiquetas por documento cuesta ~0,18 GB para todo HN. Barato.
- **Compresión:** ordenar los bloques por tema mejora < 1 % → no compacta.
- **Como único índice:** P@10 = .004 → inútil (una etiqueta agrupa ~50.000 documentos y no puede
  ordenar cuál responde mejor).
- **Como enrutador** (buscar solo en las N etiquetas más cercanas a la consulta): escanea 3-13 % del
  corpus con −19 % / −8 % de calidad. Útil para acelerar, no para ahorrar espacio.
- **Jerarquía:** con potion salen temas claros pero también grupos "por tono". Para un mapa
  navegable: agrupar con EmbeddingGemma 2 y nombrar las 1000 etiquetas con un LLM (1000 llamadas).

## Decisión
Capa opcional de **exploración, filtros por tema y enrutado** (tipo GraphRAG de Microsoft), no de
compresión. Además el grafo natural de HN (respuesta → padre → story) se usa para dar contexto a
cada resultado (título de la story, comentario padre).
