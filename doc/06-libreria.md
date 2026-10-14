# Librería `hnz` para backends: diseño (2026-10-14)

**Estado:** aprobadas P0 y P1. **P1 hecha** (`import hnz` sobre el formato actual; E30). P0 en curso
(E30b medido: embeber con CPU). El resto, pendiente.

Diseño hecho con un panel de 5 agentes (3 especialistas —API y empaquetado, presupuesto de RAM,
formato compacto y consulta—, un arquitecto que lo integra y una revisión adversarial contra el
código y la bitácora). Salida completa del panel (en inglés): [anexos/libreria-panel.json](anexos/libreria-panel.json).

## Requisitos (decididos con el usuario)
- Un **bloque sólido** usable desde un backend, importable desde **Python** (otros lenguajes, opcional).
- **Lo más compacto posible en disco** cuando no se usa: perfil compacto (~5 GB, ~4 s en frío).
- Al cargarlo, un **presupuesto de RAM** (p.ej. 2 GB o 10 GB): puede descomprimir/cachear en RAM
  para ir más rápido, pero **todo** (modelos, embedder, índices, memoria del LM, cachés) cabe en él.
- **Sin pérdida.** Windows x86-64 (como esta máquina). Instantánea de solo lectura que se reconstruye
  de vez en cuando. Un proceso con varias consultas a la vez (p.ej. FastAPI con un worker).

## Qué se entregaría
- **Paquete Python** `pip install hnz`: rueda `abi3` (Python ≥ 3.10, win_amd64) con el núcleo en Rust
  (PyO3). Solo depende del paquete `openvino` (usa sus DLL; no lo importa).
- **Bloque de datos**: carpeta `hn-AAAA-MM.hnz\` con `hn.hnzb` (contenedor: datos, LM, tokenizadores,
  manifiesto y hash por parte) + `embed.xml`/`embed.bin` (EmbeddingGemma 2 int8). **≈ 5,45 GB** con
  vectores de 256 bits, **≈ 5,1 GB** con 192. Nada más en disco por defecto.
- `hnz.exe` (CLI + servidor HTTP) y, opcional, `hnz.dll` + `hnz.h` (C, JSON de entrada y salida) para
  otros lenguajes.

## API Python
```python
import hnz
with hnz.open(r"D:\hn\hn-2026-10.hnz", ram="4GB") as hn:
    r = hn.search("how do I deal with burnout", k=10, since="2015", until="2018-06", types="comment")
    for h in r:
        print(h.rank, h.date, h.by, h.story_title, (h.text or "")[:120])
    it = hn.get(9700000)                          # texto literal (HTML original de HN)
    jobs = hn.search(types="job", since="2024")   # sin texto: lo más reciente
    print(r.timings_ms, hn.stats()["memory"])
```
- Seguro entre hilos, libera el GIL; `asearch`/`aget` asíncronas; `raw=True` devuelve dicts con el
  mismo JSON que el servidor HTTP (FastAPI los devuelve tal cual).
- `deadline`: si se agota, devuelve lo ya reordenado y completa con el orden de la 1ª etapa
  (`degraded=True`) en vez de fallar. Calidad elegible: `fast` / `balanced` / `precise`.

## Presupuesto de RAM (`ram=`)
Techo para todo lo que tiene hnz. Se controla con una **contabilidad propia** (reservas por
componente: modelos, índices, memoria del LM, cachés, memoria de cada consulta; lo de OpenVINO y los
tokenizadores, medido al cargarlos), no con el "working set" del proceso, que en un backend incluye
la memoria de la propia aplicación (corrección de la revisión).

| | 1 GB | 2 GB | 4 GB | 10 GB |
|---|---|---|---|---|
| Embedder | CPU | iGPU si cabe (si no, CPU) | iGPU | iGPU |
| Vectores (1,1-1,5 GB) | leídos del disco | en parte (lo más reciente) | todos en RAM | todos en RAM |
| Índice de palabras, metadatos | disco | disco | en RAM | en RAM + columna de autores |
| Cachés (texto descomprimido, embeddings) | — | ~0,3 GB | ~0,3-0,9 GB | ~5 GB |
| Consulta en frío (mediana, estimación) | solo palabras/listados | ~4-5 s | ~3,3-3,8 s | ~3,3-3,8 s |
| Consulta ya en caché | 0,1-0,3 s | 0,1-0,3 s | 0,08-0,15 s | 0,08-0,15 s |
| Consulta repetida | < 10 ms | < 10 ms | < 10 ms | < 10 ms |

**Límite honesto:** el texto comprimido con el LM se descomprime a ~30 registros/s (8 hilos): todo
HN tardaría ~19 días, así que **no se puede descomprimir el bloque entero al cargarlo**. La RAM
acelera porque los índices quedan residentes, el embedder va en la iGPU y se guarda lo que ya se
descomprimió; las consultas en frío siguen marcadas por el LM a cualquier nivel. **Por encima de
~4 GB apenas mejora el frío**, salvo que se active el precalentamiento en segundo plano de lo más
reciente (~137.000 docs/hora, opcional) o un fichero de caché persistente (opcional).

## Consulta en el perfil compacto
1. Embedding de la consulta (~8 ms en la iGPU, ~20 ms en CPU) y rango de fechas → rango de documentos.
2. 1ª etapa: vectores binarios (Hamming → tabla asimétrica) + términos raros → ~115 candidatos
   (en RAM: decenas de ms; **a escala completa está sin validar**: podría ser 0,4-0,6 s).
3. Descomprimir los primeros 128 tokens de cada candidato que no esté en caché (~7.000 tokens ≈ 3 s)
   y re-embeberlos en paralelo (iGPU); BM25 exacto con todos los términos sobre ese texto; fusión.
4. Top-k: metadatos, fragmentos (gratis: ya están descomprimidos), títulos de los hilos; texto
   completo solo si se pide (+0,5 s mediana).

## Lo que corrigió la revisión adversarial
- **RAM:** contabilidad propia, no "working set menos línea base" (en un backend, el crecimiento de la
  aplicación provocaría rechazos permanentes).
- **Embedder en CPU:** re-embeber candidatos en CPU va a ~20 docs/s (medido) → 6-10 s por consulta en
  frío, no 4. Usar la iGPU siempre que exista (mínimo ~1,7 GB de RAM).
- **Planificador del LM:** sin "aparcar" trabajos, un `get()` o un fragmento esperaría a las consultas
  pesadas en curso (cada paso con 64 secuencias activas tarda ~28 ms). Hay que limitar las secuencias
  activas cuando hay trabajo interactivo esperando.
- **Sin pérdida de verdad:** guardar también el orden de `kids` (≤ ~6 MB para todo HN) y comprobar el
  100 % de los bytes al empaquetar.
- Congelar la función que construye el texto a embeber (`prompt.rs`) antes de días de iGPU;
  conservar ≥ 512 bits de signo en el material de construcción por si hay que cambiar el ancho.
- Validar la 1ª etapa a 49 M documentos (objetivo ~5 ns/doc/hilo) y la reordenación con el desempate
  real; asegurar el redondeo del procesador (MXCSR) en los hilos del LM dentro de un proceso ajeno.
- Tiempo de máquina realista para construir todo HN: **~8 días** (LM en CPU y embeddings en la iGPU
  compiten), mejor por meses para reanudar con poca pérdida.

## Fases (estimación del panel)
| Fase | Trabajo | Aceptación |
|---|---|---|
| P0 Mediciones previas | embeddings de prefijos (E30a), BM25 al vuelo y fusión (E30c), índice global (E30e), 1ª etapa a escala | intervalos de confianza; congelar P, bits, K, umbrales |
| P1 Librería Python sobre el formato actual | workspace Cargo, `api.rs`, PyO3, cargador de OpenVINO, servicio de embedding | se instala en un venv limpio; mismos resultados que `hnz search`; LM bit a bit idéntico |
| P2 Gobierno de la RAM | contabilidad, planificador, lecturas sin mmap, admisión | memoria ≤ 1,03 × `ram` con 8 clientes durante 30 min |
| P3 Bloque compacto v1 | contenedor, `hnz pack`, índice global, `verify-block` | sin pérdida en la muestra; tamaño proyectado ≤ 5,6 GB |
| P4 Reordenación | descompresión de prefijos + re-embedding + BM25 exacto + fusión | calidad ≥ hoy − 0,01; latencias medidas |
| P5 Construcción completa | `pipeline.py --profile compact` de todo HN | tamaño, sin pérdida (muestra 0,5 % + comprobaciones), memoria |
| P6 Empaquetado | rueda, DLL de OpenVINO, servidor, C ABI opcional, tests | instalación limpia en Windows |
| P7 Mejoras opcionales | cabeceras de registro, reordenación en cascada, embedder int4, caché persistente, nivel zstd para lo más reciente | cada una medida |

## Decisiones pendientes del usuario
1. Empezar P0 (mediciones) y P1 (librería sobre el formato actual), que no comprometen el formato.
2. LM final: el actual (1 % de HN) o entrenarlo con el 5 % (~7 h, −2-4 % de texto).
3. Bits de los vectores: 256 (≈ 5,45 GB) o 192 (≈ 5,1 GB); decidir tras E30d.
4. Respuesta parcial (`degraded`) al agotar el plazo: ¿aceptable?
5. Búsqueda solo por palabras frecuentes: ¿basta con encontrarlas entre los candidatos semánticos, o
   índice de frecuentes (+1,4-1,8 GB)?
6. ¿Habrá iGPU Intel en el servidor? (sin ella, ~0,5-1 s más por consulta en frío, o más).
7. Disco fuera del bloque: ¿se permite, opcional, una caché persistente o precalentamiento?
8. Herramientas de compilación: ¿se pueden instalar las MSVC Build Tools para las ruedas?
