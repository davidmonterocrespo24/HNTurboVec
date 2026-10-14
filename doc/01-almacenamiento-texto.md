# Almacenamiento del texto (la información recuperable)

Detalle numérico en [BITACORA.md](BITACORA.md) (E00-E03, E09, E12, E15-E17, E20, E24, E25).

## Qué hemos aprendido

1. **El texto es lo que más pesa** (~335 B/ítem crudo). Metadatos en columnas delta: ~5,5 B/ítem.
2. **Bloques pequeños > compresor más fuerte.** Pasar de 1 registro a 16 por bloque baja de 181 a
   ~124 B/ítem; zstd-22, brotli-11 y xz apenas mejoran a zstd-19 + diccionario.
3. **Orden por hilo** ayuda un poco (118,5 → 114,4 B en bloques de 256); por tema < 1 %. Pero un
   orden distinto del id obliga a guardar un mapa doc→bloque (~4 B/doc) → **orden por id** (= orden
   temporal), que además convierte un rango de fechas en un rango contiguo de documentos.
4. **El salto grande es el LM como compresor**: con pythia-70m int8 y la representación "e" (E15),
   ~61-64 B/registro de texto frente a ~117 B con zstd en bloques de 16. Cada registro se codifica
   solo → acceso aleatorio perfecto; el "diccionario" es el modelo (72 MB).
5. **HTML de HN**: dos dialectos de escapado (antiguo ~2007-2012 y moderno) y enlaces truncados con
   "…" (30-40 % de los enlaces) → transformación reversible verificada registro a registro.
6. **Filas `deleted`**: se guardan como metadatos (tipo, fecha, padre, dead) para no romper la cadena
   de los hilos (en 2010-06, 141 comentarios cuelgan de un borrado). El orden de `kids` no se guarda.

## Cómo se resolvieron los riesgos del LM

- **Determinismo bit a bit:** motor propio en Rust con matmul int8 exacta (el resultado no depende
  del núcleo SIMD ni de los hilos), exp/erf propias y sumas en orden fijo; `MATH_VERSION` en `lm.json`.
- **Integridad:** 1 byte de comprobación por registro, validado al descomprimir el registro
  **completo** (`--full`, `get`, `verify`): una divergencia (otro binario con el mismo modelo y la
  misma `MATH_VERSION`) da error en ~255 de cada 256 registros; el resto solo se rechaza si además
  sale UTF-8 inválido o un registro mal formado, así que no es una garantía absoluta (`hnz verify`
  sí lo es: compara con el original). Los fragmentos y títulos se cortan antes del final y no lo
  comprueban; ahí protege que al abrir el segmento se exige el mismo id de modelo y `MATH_VERSION`.
- **Descompresión secuencial:** lote de todos los resultados de una consulta, pool de hilos propio,
  modo fragmento (64 tokens del cuerpo) y cachés L1 (RAM) + L2 (disco).
- **Coste de compresión:** ~120 reg/s en 16 hilos (≈ 4 días para ~40 M ítems fríos), en paralelo con
  los embeddings en la iGPU.

## Niveles implementados

| Nivel | Uso típico | Formato | Texto B/ítem (medido) | Lectura de un top-10 |
|---|---|---|---|---|
| Caliente | años recientes | zstd-19 + dict, bloques de 16 | ~117 | ~6-11 ms (con búsqueda) |
| Frío | años antiguos | LM pythia-70m + codificación aritmética | ~61-65 | ~0,3-0,45 s fragmentos; ~0,6 s completo (46 ms en caché) |

El tamaño de bloque zstd es configurable (`--block`): bloques de 256 dan ~114 B con lecturas de
~0,2 ms (un posible nivel "templado"). Un segmento caliente se enfría con `hnz retier --tier lm`.
