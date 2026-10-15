# Tags (~1000 topics) and multi-level graph

Numerical details in [BITACORA.md](BITACORA.md) (E08, E09).

## Evaluated idea
Tag the content with ~1000 tags, group them into levels (1000 → 40 areas) and use that
to compact and/or search.

## Results
- **Space:** storing 3 tags per document costs ~0.18 GB for all of HN. Cheap.
- **Compression:** sorting the blocks by topic improves by < 1% → it does not compact.
- **As the only index:** P@10 = .004 → useless (a tag groups ~50,000 documents and cannot
  rank which one answers best).
- **As a router** (search only in the N tags closest to the query): scans 3-13% of the
  corpus with −19% / −8% quality. Useful for speeding up, not for saving space.
- **Hierarchy:** with potion, clear topics emerge, but also groups "by tone". For a navigable
  map: cluster with EmbeddingGemma 2 and name the 1000 tags with an LLM (1000 calls).

## Decision
Optional layer for **exploration, topic filters and routing** (like Microsoft's GraphRAG), not for
compression. In addition, HN's natural graph (reply → parent → story) is used to give context to
each result (story title, parent comment).
