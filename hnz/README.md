# hnz — Hacker News comprimido con búsqueda híbrida

Archivo de HN de mínimo tamaño con búsqueda semántica + palabras clave, filtros de fecha y
recuperación literal de los resultados. Ver la documentación de diseño en [`../doc/`](../doc/).

## Requisitos (Windows)
- Rust (toolchain `stable-x86_64-pc-windows-gnu`) y GCC (WinLibs) en el PATH para compilar.
- Para el embedding de consultas: `openvino_c.dll` (p.ej. del entorno Python `.venv`, paquete
  `openvino`) y en `models/`: `gemma2_text_b1_s64.{xml,bin}` + `gemma2_tokenizer.json`
  (generados con `export_gemma2_ov.py`).
- Para el nivel frío: `models/lm-pythia-70m.hnzlm` + `.tokenizer.json` (`export_lm.py`).

```bash
cargo build --release          # binario en hnz/target/release/hnz.exe
```

## Construir un archivo (desde la raíz del proyecto)
```bash
# nivel caliente (zstd) y nivel frío (LM)
hnz build --out out/seg-2024-01 --tier zstd data/2024/2024-01.parquet
hnz build --out out/seg-2010-06 --tier lm --lm models/lm-pythia-70m.hnzlm data/2010/2010-06.parquet
# vectores (embeddings calculados con embed_month.py en la iGPU)
hnz index --seg out/seg-2024-01 --emb emb/g2_2024-01.f16.npy --ids emb/g2_2024-01.ids.npy
# comprobar que no se pierde nada
hnz verify --seg out/seg-2010-06 data/2010/2010-06.parquet
hnz stats                      # tamaño por componente de todo out/
```

## Buscar
```bash
hnz search "how do I deal with burnout" -k 10
hnz search rust async runtime --mode keyword --from 2015 --to 2018-06 --type comment
hnz search "best text editor" --by pg --json --full   # JSON con texto completo (RAG)
hnz get 9700000                                         # un ítem por id
hnz serve --port 8088                                   # web en http://127.0.0.1:8088/
```
Parámetros: `-k` nº de resultados · `--from/--to` (año, año-mes o fecha, inclusive) ·
`--mode hybrid|vector|keyword` · `--type story,comment,job,poll` · `--by autor` ·
`--full` (texto completo del nivel LM; por defecto un fragmento de 64 tokens).

API HTTP: `GET /search?q=…&k=10&from=2015&to=2018&type=comment&mode=hybrid&full=0`,
`GET /item/<id>` (texto completo, descomprimido bajo demanda).

## Variables de entorno
- `HNZ_MODELS` (carpeta de modelos), `HNZ_OV_LIBS` (carpeta de `openvino_c.dll`).
- `HNZ_LM_THREADS` (hilos del LM; por defecto la mitad de los lógicos), `HNZ_KERNEL=avx2|scalar`
  (forzar núcleo; el resultado es idéntico), `HNZ_PROF=1` (perfil de la decodificación LM).
