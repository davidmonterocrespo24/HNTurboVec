# Almacenamiento del texto (la información recuperable)

Detalle numérico en [BITACORA.md](BITACORA.md) (E00-E03, E09).

## Qué hemos aprendido

1. **El texto es lo que más pesa** (~335 B/ítem crudo). Metadatos en columnas delta: solo ~6 B/ítem.
2. **Bloques pequeños > compresor más fuerte.** Pasar de 1 registro a 16 por bloque baja de 181 a
   124 B/ítem; zstd-22, brotli-11 y xz apenas mejoran a zstd-19 + diccionario.
3. **Orden por hilo** ayuda un poco (los comentarios citan al padre): 118,5 → 114,4 B en bloques de 256.
   Por tema: < 1 %. Pero un orden distinto del id obliga a guardar un mapa doc→bloque (~4 B/doc),
   que se come la ganancia → **orden por id (= orden temporal)**, que además hace que un rango de
   fechas sea un rango contiguo de documentos.
4. **El salto grande es el LM como compresor**: ~46-53 B/ítem de texto (1,26-1,28 bits/byte con
   SmolLM2-135M) frente a ~118-137 B con zstd. Cada registro se codifica solo → acceso aleatorio
   perfecto, el "diccionario" es el modelo.

## Riesgos del LM como compresor (a resolver en el diseño)

- **Determinismo bit a bit:** el decodificador debe calcular exactamente las mismas probabilidades
  que el codificador. Diferencias de coma flotante entre máquinas/librerías (o GPU vs CPU) hacen el
  archivo indescifrable → motor de inferencia propio en Rust, determinista (orden fijo de operaciones,
  sin fast-math, funciones exp/log propias), mismo código para codificar y decodificar + CRC por registro.
- **Descompresión secuencial:** cada token necesita el anterior → no se paraleliza dentro de un
  registro. Se compensa decodificando **en lote** todos los resultados de una consulta a la vez
  (los pesos se leen una vez por paso para todo el lote), pesos cuantizados y caché por capas.
- **Coste de compresión inicial:** ~4.400 M tokens para todo HN. En CPU es largo (días) → se acepta
  porque la preparación puede tardar; lo que importa es el uso diario.

## Decisión (niveles por antigüedad)

| Nivel | Antigüedad | Formato | B/ítem (medido) | Lectura |
|---|---|---|---|---|
| Caliente | último año | zstd-19 + dict, bloques de 16 | ~124 | ~10-25 µs |
| Templado | 1-3 años (configurable) | zstd-19 + dict, bloques de 256 | ~118 | ~0,1-0,25 ms |
| Frío | más antiguo | **LM + codificación aritmética**, por registro | ~55-62 (estimado) | por medir (lote) |

Caché por capas: RAM (LRU de registros ya descomprimidos) → disco (registros fríos ya decodificados,
guardados en zstd rápido, tamaño acotado) → nivel real.
