# hnz — Hacker News comprimido con búsqueda híbrida

Archivo de HN de mínimo tamaño con búsqueda semántica + palabras clave, filtros de fecha/autor/tipo
y recuperación literal de los resultados. Diseño y mediciones en [`../doc/`](../doc/).

## Requisitos
- **CPU x86-64-v3** (AVX2/FMA/BMI2: Intel Haswell o posterior, AMD Zen o posterior): el binario se
  compila con `target-cpu=x86-64-v3` (`hnz/.cargo/config.toml`). AVX-VNNI se usa si existe; el
  resultado del LM es idéntico con o sin él.
- Windows: Rust `stable-x86_64-pc-windows-gnu` y GCC (WinLibs) en el PATH para compilar.
- Entorno Python solo para preparar modelos y embeddings (torch CPU primero, desde su índice),
  desde la raíz del proyecto:
  ```bash
  uv venv .venv --python 3.12
  uv pip install --python .venv -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu
  uv pip install --python .venv -r requirements.txt
  .venv\Scripts\activate       # PowerShell/cmd (en Git Bash: source .venv/Scripts/activate)
  python prepare_models.py     # crea models/: EmbeddingGemma 2 (IR), tokenizadores, LM .hnzlm
  ```
  Los comandos `python …` de este documento suponen el entorno activado (o usar
  `.venv\Scripts\python` en su lugar).
- En ejecución, el embedding de consultas usa `openvino_c.dll` (del paquete Python `openvino`, o de
  una instalación de OpenVINO): opción `--ov-libs` o variable `HNZ_OV_LIBS`.

```bash
cd hnz && cargo build --release          # binario en hnz/target/release/hnz.exe
```

## Construir un archivo (desde la raíz del proyecto)
Solo descargar todo el dataset (reanudable, con SHA-256): `python download_hn.py`.
Todo HN, por años y reanudable: `python pipeline.py --years 2006-2026` (ver su ayuda). Al volver a
ejecutarlo baja los meses que han cambiado en el dataset (el mes en curso se publica a medias),
recalcula sus embeddings y reconstruye solo los años afectados; un segmento nunca se reconstruye
solo por cambiar de nivel: `--retier` enfría a LM los años que ya no son calientes (con
`--delete-old`, verificando todo antes de borrar el original). Un año ya construido e indexado y al
día con el dataset no vuelve a bajar ni a embeber nada, así que sus Parquet y embeddings se pueden
borrar (salvo los del año en curso); si faltan Parquet ya archivados no reconstruye el año
(perdería datos), y no toca segmentos construidos a mano (sin `sources.json`). Si el dataset no
responde se detiene; `--offline` trabaja solo con los Parquet locales.
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
  devuelven hasta k resultados que cumplen todos los filtros, y solo documentos con alguna señal de
  la consulta (algún término en modo palabras, vector en modo vectorial, una de las dos en híbrido).
- Sin texto (`hnz search --type job --from 2024`) se listan los ítems más recientes que cumplen los
  filtros.
- Nivel frío: por defecto se devuelve un **fragmento** (título y url completos + 64 tokens del
  cuerpo, `partial: true`); `--full` o `GET /item/<id>` descomprimen el texto completo.

API HTTP: `GET /search?q=…&k=10&from=2015&to=2018&type=comment&by=pg&mode=hybrid&full=0`,
`GET /item/<id>`.

## Desde Python (`import hnz`)
Paquete en `hnz/py/` (núcleo en Rust con PyO3; rueda `abi3` para Python ≥ 3.10, Windows x86-64):
```bash
cd hnz/py && maturin develop --release        # instala en el entorno activo (o: maturin build --release)
```
```python
import hnz
with hnz.open(r"C:\ruta\out", ram="4GB", preload=True) as hn:     # modelos en out/models o out/../models
    r = hn.search("how do I deal with burnout", k=10, since="2015", until="2018-06", types="comment")
    for h in r:                                                    # dataclasses Hit (o raw=True: dicts)
        print(h.rank, h.date, h.by, h.story_title, h.text[:120])
    item = hn.get(9700000)                                         # literal: item.html = HTML original
    nuevos = hn.search(types="job", since="2024")                  # sin texto: lo más reciente
    print(r.timings_ms, hn.stats())
```
- `Archive` es seguro entre hilos y suelta el GIL: un backend (p.ej. FastAPI con un worker) atiende
  varias consultas a la vez sobre el mismo archivo abierto. Errores de argumentos → `ValueError`.
- Las DLL de OpenVINO se toman del paquete `openvino` instalado (o `ov_libs=` / `HNZ_OV_LIBS`).
- `ram=`: en esta versión se guarda e informa; el control estricto de la memoria es la fase P2
  ([doc/06-libreria.md](../doc/06-libreria.md)).
- Pruebas: `python -m pytest hnz/py/tests` (mismos resultados que la CLI, sin pérdida frente al
  Parquet, 8 hilos a la vez).

## Variables de entorno
- `HNZ_MODELS` (carpeta de modelos), `HNZ_OV_LIBS` (carpeta de `openvino_c.dll`).
- `HNZ_LM_THREADS` (hilos del LM; por defecto la mitad de los lógicos, mínimo 1),
  `HNZ_KERNEL=avx2|scalar` (forzar núcleo; el resultado es idéntico), `HNZ_PROF=1` (perfil del LM).
