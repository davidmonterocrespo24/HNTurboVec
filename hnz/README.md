# hnz — Hacker News comprimido con búsqueda híbrida

Archivo de HN de mínimo tamaño con búsqueda semántica + palabras clave, filtros de fecha/autor/tipo
y recuperación literal de los resultados. Diseño y mediciones en [`../doc/`](../doc/).

## Requisitos
- **CPU x86-64-v3** (AVX2/FMA/BMI2: Intel Haswell o posterior, AMD Zen o posterior): el binario se
  compila con `target-cpu=x86-64-v3` (`hnz/.cargo/config.toml`). AVX-VNNI se usa si existe; el
  resultado del LM es idéntico con o sin él.
- Windows: Rust `stable-x86_64-pc-windows-gnu` y GCC (WinLibs) en el PATH para compilar.
- Entorno Python (`requirements.txt`) solo para preparar modelos y embeddings:
  ```bash
  uv venv .venv --python 3.12
  uv pip install --python .venv -r requirements.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
  python prepare_models.py     # crea models/: EmbeddingGemma 2 (IR), tokenizador, LM .hnzlm
  ```
- En ejecución, el embedding de consultas usa `openvino_c.dll` (del paquete Python `openvino`, o de
  una instalación de OpenVINO): opción `--ov-libs` o variable `HNZ_OV_LIBS`.

```bash
cd hnz && cargo build --release          # binario en hnz/target/release/hnz.exe
```

## Construir un archivo (desde la raíz del proyecto)
Todo HN, por años y reanudable: `python pipeline.py --years 2006-2026` (ver su ayuda).
A mano, por segmentos:
```bash
hnz build --out out/seg-2024-01 --tier zstd data/2024/2024-01.parquet                 # caliente
hnz build --out out/seg-2010-06 --tier lm --lm models/lm-pythia-70m.hnzlm data/2010/2010-06.parquet   # frío
hnz index --seg out/seg-2024-01 --emb emb/g2_2024-01.f16.npy --ids emb/g2_2024-01.ids.npy --bits 768
hnz index --seg out/seg-2010-06 --emb emb/g2_2010-06.f16.npy --ids emb/g2_2010-06.ids.npy --bits 512
hnz verify --seg out/seg-2010-06 data/2010/2010-06.parquet   # compara cada registro con el original
hnz retier --seg out/seg-2024-01 --tier lm                   # enfriar un segmento (deja <seg>.old)
hnz stats                                                    # tamaño por componente de todo out/
```
No mezcle en la misma carpeta segmentos que se solapen (p.ej. `seg-2010` y `seg-2010-06`): el motor
se niega a abrirlos.

## Buscar
```bash
hnz search "how do I deal with burnout" -k 10
hnz search rust async runtime --mode keyword --from 2015 --to 2018-06 --type comment
hnz search "startups" --by pg --json --full         # JSON con texto completo (RAG)
hnz get 9700000                                      # un ítem por id (texto completo)
hnz serve --port 8088                                # web en http://127.0.0.1:8088/
```
- `-k` nº de resultados (máx. 500; 100 por HTTP) · `--from/--to` año, año-mes o fecha, inclusive ·
  `--mode hybrid|vector|keyword` · `--type story,comment,job,poll` · `--by autor` · `--dead`.
- Los filtros se aplican al buscar (autor y tipos raros recorren los metadatos del rango): se
  devuelven hasta k resultados que cumplen todos los filtros.
- Nivel frío: por defecto se devuelve un **fragmento** (título y url completos + 64 tokens del
  cuerpo, `partial: true`); `--full` o `GET /item/<id>` descomprimen el texto completo.

API HTTP: `GET /search?q=…&k=10&from=2015&to=2018&type=comment&by=pg&mode=hybrid&full=0`,
`GET /item/<id>`.

## Variables de entorno
- `HNZ_MODELS` (carpeta de modelos), `HNZ_OV_LIBS` (carpeta de `openvino_c.dll`).
- `HNZ_LM_THREADS` (hilos del LM; por defecto la mitad de los lógicos, mínimo 1),
  `HNZ_KERNEL=avx2|scalar` (forzar núcleo; el resultado es idéntico), `HNZ_PROF=1` (perfil del LM).
