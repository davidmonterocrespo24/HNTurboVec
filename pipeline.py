"""Pipeline completo y REANUDABLE para construir el archivo de HN por años.

Pasos por año (se salta lo que ya está hecho):
  1. descargar los Parquet mensuales de open-index/hacker-news (data/YYYY/YYYY-MM.parquet); un mes
     que ha cambiado en el dataset (p.ej. el mes en curso, que se publica a medias) se vuelve a bajar
  2. embeddings EmbeddingGemma 2 en la iGPU (embed_month.py)              -> emb/g2_YYYY-MM.*
     (se recalculan si su Parquet ha cambiado)
  3. segmento del año: nivel zstd si es reciente, nivel LM si es antiguo   -> out/seg-YYYY
     (se reconstruye si cambian los meses o su contenido; nunca solo por el nivel: un segmento ya
     construido se enfría con `hnz retier`, o con --retier aquí)
  4. índice vectorial del año: 768 bits en zstd, 512 en LM (E19)
Un año ya construido e indexado con los mismos meses y tamaños que el dataset no vuelve a bajar
ni a embeber nada: sus Parquet y embeddings se pueden borrar para ahorrar espacio (salvo los del
año en curso, que se reconstruye entero cuando el dataset añade un mes). Los segmentos sin
sources.json (construidos a mano con `hnz build`) no se tocan.

Ejemplos (con el entorno .venv activado, o con .venv\\Scripts\\python en vez de python):
  python pipeline.py --years 2006-2026                       # todo HN (caliente = últimos 2 años)
  python pipeline.py --years 2024 --steps download,embed     # solo preparar 2024
  python pipeline.py --years 2006-2015 --all-zstd            # rápido (horas); enfriar luego con --retier
  python pipeline.py --years 2006-2024 --steps build --retier   # enfriar a LM los años fríos ya construidos
  python pipeline.py --years 2026 --offline                  # sin conexión: solo Parquet locales

La compresión LM de todo HN tarda días en CPU: una estrategia práctica es construir todo con
--all-zstd y enfriar después los años viejos con --retier (o `hnz retier --tier lm`), que
conserva el original como <seg>.old salvo con --delete-old (verifica todo antes de borrarlo).
Si el dataset no responde, el pipeline se detiene (no pasa a modo sin conexión en silencio); con
--offline el límite caliente/frío sale del último año visto en línea (out/pipeline_state.json).
Los segmentos de un mismo out/ no pueden solaparse: no mezcle segmentos mensuales (seg-YYYY-MM) y
anuales (seg-YYYY) del mismo periodo.
"""
import argparse, glob, json, os, re, subprocess, sys, time
import numpy as np
# el descargador hf_xet de huggingface_hub muere a mitad de descarga en este Windows (exit 127,
# E26): se usa la descarga HTTP de siempre (también reanudable). Se hereda en los subprocesos.
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

ROOT = os.path.dirname(os.path.abspath(__file__))
HNZ = os.path.join(ROOT, 'hnz', 'target', 'release', 'hnz.exe' if os.name == 'nt' else 'hnz')
PY = sys.executable
REPO = 'open-index/hacker-news'
MONTH_RE = re.compile(r'^data/(\d{4})/(\d{4})-(\d{2})\.parquet$')


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def years_arg(s):
    if '-' in s:
        a, b = s.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(',')]


_listing = None
_online = True
STATE = os.path.join(ROOT, 'out', 'pipeline_state.json')   # último año visto en el dataset


def listing():
    """Ficheros mensuales del dataset: {año: {ruta relativa: tamaño}} (solo YYYY/YYYY-MM.parquet).
    Con --offline (o HF_HUB_OFFLINE=1), los Parquet locales. Un fallo de red NO pasa a modo sin
    conexión en silencio: el límite caliente/frío y las descargas dependen del listado real."""
    global _listing
    if _listing is None:
        if _online:
            try:
                from huggingface_hub import HfApi
                tree = HfApi().list_repo_tree(REPO, path_in_repo='data', repo_type='dataset', recursive=True)
                files = [(f.path, f.size) for f in tree if getattr(f, 'size', None) is not None]
            except Exception as e:
                raise SystemExit(f'no se pudo listar el dataset {REPO} ({type(e).__name__}: {str(e)[:200]}): '
                                 'reintente, o use --offline para trabajar solo con los Parquet locales')
        else:
            files = [(os.path.relpath(p, ROOT).replace(os.sep, '/'), os.path.getsize(p))
                     for p in glob.glob(os.path.join(ROOT, 'data', '*', '*.parquet'))]
        _listing = {}
        for f, size in files:
            m = MONTH_RE.match(f)
            if m:
                _listing.setdefault(int(m.group(1)), {})[f] = size
        if _online and _listing:
            os.makedirs(os.path.dirname(STATE), exist_ok=True)
            json.dump({'dataset_max_year': max(_listing)}, open(STATE, 'w', encoding='utf-8'))
    return _listing


def year_done(year):
    """¿seg-YYYY ya está construido e indexado con los mismos meses y tamaños que el dataset?
    Entonces no hay que bajar ni embeber nada (sus Parquet y embeddings pueden estar borrados)."""
    seg = os.path.join(ROOT, 'out', f'seg-{year}')
    man, side = os.path.join(seg, 'manifest.json'), os.path.join(seg, 'sources.json')
    if not (os.path.exists(man) and os.path.exists(side)):
        return False
    if json.load(open(man, encoding='utf-8')).get('vec_bits', 0) == 0:
        return False
    built = json.load(open(side, encoding='utf-8')).get('sizes') or {}
    avail = {os.path.basename(f): s for f, s in listing().get(year, {}).items()}
    if _online:
        return built == avail
    # sin conexión solo se ven los Parquet locales: hecho si ninguno es nuevo o distinto
    return all(built.get(n) == s for n, s in avail.items())


def download(year):
    files = listing().get(year, {})
    if not files:
        log(f'  (sin ficheros para {year} en el dataset; se omite)')
    if not _online:
        return
    # descarga reanudable byte a byte y con comprobación de SHA-256 (download_hn.py)
    import download_hn
    if download_hn.run({year})['fallo']:
        raise SystemExit(f'no se pudieron bajar todos los meses de {year}: vuelva a ejecutar para reanudar')


def months(year):
    return sorted(glob.glob(os.path.join(ROOT, 'data', str(year), f'{year}-[0-9][0-9].parquet')))


def emb_paths(m):
    e = os.path.join(ROOT, 'emb', f'g2_{m}')
    return e + '.f16.npy', e + '.ids.npy', e + '.src.json'


def emb_fresh(p):
    """¿Hay embeddings calculados a partir de ESTE Parquet? (registro con su tamaño)"""
    e, i, side = emb_paths(os.path.basename(p).replace('.parquet', ''))
    if not (os.path.exists(e) and os.path.exists(i)):
        return False
    if not os.path.exists(side):
        # embeddings sin registro (embed_month.py a mano o pipeline anterior): se adoptan solo si
        # son posteriores al Parquet; uno recién rebajado es más nuevo que ellos -> se recalculan
        if os.path.getmtime(i) < os.path.getmtime(p):
            return False
        json.dump({'parquet_size': os.path.getsize(p)}, open(side, 'w', encoding='utf-8'))
        return True
    return json.load(open(side, encoding='utf-8')).get('parquet_size') == os.path.getsize(p)


def embed(year, device):
    for p in months(year):
        if emb_fresh(p):
            continue
        m = os.path.basename(p).replace('.parquet', '')
        log(f'  embeddings {m} ({device})')
        # primero se borra lo anterior: si se interrumpe, no quedan ficheros de dos versiones
        for f in emb_paths(m):
            if os.path.exists(f):
                os.remove(f)
        subprocess.run([PY, os.path.join(ROOT, 'embed_month.py'), p, device], check=True, cwd=ROOT)
        json.dump({'parquet_size': os.path.getsize(p)}, open(emb_paths(m)[2], 'w', encoding='utf-8'))


def check_no_monthly_overlap(year):
    mon = [d for d in glob.glob(os.path.join(ROOT, 'out', f'seg-{year}-[0-9][0-9]')) if os.path.exists(os.path.join(d, 'manifest.json'))]
    if mon:
        raise SystemExit(f'out/ contiene segmentos mensuales de {year} ({", ".join(os.path.basename(x) for x in mon)}) que '
                         f'se solaparían con out/seg-{year}: muévalos a otra carpeta antes de seguir.')


def seg_tier(seg):
    return json.load(open(os.path.join(seg, 'manifest.json'), encoding='utf-8')).get('text_tier')


def build(year, tier, a):
    seg = os.path.join(ROOT, 'out', f'seg-{year}')
    man = os.path.join(seg, 'manifest.json')
    exists = os.path.exists(man)
    side = os.path.join(seg, 'sources.json')
    # registro provisional, escrito ANTES de `hnz build` (que sustituye la carpeta entera)
    pending = seg + '.sources.json'
    if exists and not os.path.exists(side) and os.path.exists(pending) and os.path.getmtime(man) >= os.path.getmtime(pending):
        # el build terminó pero el proceso murió antes de mover el registro: se adopta
        os.replace(pending, side)
    prev = json.load(open(side, encoding='utf-8')) if exists and os.path.exists(side) else None
    if exists and prev is None:
        # construido fuera del pipeline (hnz build a mano): no se sabe qué meses contiene y
        # reconstruirlo con los Parquet locales podría perder datos
        log(f'  {os.path.basename(seg)} no tiene sources.json (no lo construyó pipeline.py): no se toca')
        return None
    ps = months(year)
    src = [os.path.basename(p) for p in ps]
    sizes = {os.path.basename(p): os.path.getsize(p) for p in ps}
    if prev is not None:
        missing = sorted(set(prev.get('months', [])) - set(src))
        if missing:
            # p.ej. se borraron Parquet ya archivados para ahorrar espacio: reconstruir perdería datos
            log(f'  faltan Parquet locales que seg-{year} ya contiene ({", ".join(missing)}): no se reconstruye')
            return retier(seg, tier, a)
        if prev.get('months') == src and prev.get('sizes') == sizes:
            return retier(seg, tier, a)
        log(f'  los meses de {year} o su contenido han cambiado: se reconstruye')
    if not ps:
        log(f'  (sin Parquet para {year}; no se construye)')
        return None
    check_no_monthly_overlap(year)
    log(f'  segmento {year} nivel {tier} ({len(ps)} meses)')
    json.dump({'months': src, 'sizes': sizes, 'tier': tier}, open(pending, 'w', encoding='utf-8'))
    cmd = [HNZ, 'build', '--out', seg, '--tier', tier] + (['--lm', a.lm] if tier == 'lm' else []) + ps
    subprocess.run(cmd, check=True, cwd=ROOT)
    os.replace(pending, side)
    return seg


def retier(seg, tier, a):
    """Un segmento ya construido no se reconstruye por cambiar de nivel: se enfría con retier."""
    cur = seg_tier(seg)
    if cur == 'zstd' and tier == 'lm':
        if not a.retier:
            log(f'  {os.path.basename(seg)} está en zstd y le corresponde lm: use --retier (o hnz retier --tier lm)')
            return seg
        log(f'  enfriando {os.path.basename(seg)} a nivel lm')
        cmd = [HNZ, 'retier', '--seg', seg, '--tier', 'lm', '--lm', a.lm] + (['--verify-all', '--delete-old'] if a.delete_old else [])
        subprocess.run(cmd, check=True, cwd=ROOT)
    return seg


def index(seg, a):
    man = json.load(open(os.path.join(seg, 'manifest.json'), encoding='utf-8'))
    # bits según el nivel REAL del segmento (un año frío aún sin enfriar sigue con 768)
    bits = 768 if man.get('text_tier') == 'zstd' else 512
    if man.get('vec_bits', 0) == bits or (man.get('vec_bits', 0) > 0 and not a.reindex):
        return
    side = os.path.join(seg, 'sources.json')
    if not os.path.exists(side):
        log(f'  ({os.path.basename(seg)} sin sources.json: no se sabe qué embeddings le corresponden; no se indexa)')
        return
    names = [x.replace('.parquet', '') for x in json.load(open(side, encoding='utf-8'))['months']]
    if not names:
        log(f'  (sin meses conocidos para {os.path.basename(seg)}: no se indexa)')
        return
    parts = []
    for m in names:
        e, i, _ = emb_paths(m)
        if not (os.path.exists(e) and os.path.exists(i)):
            log(f'  (sin embeddings para {m}: el segmento queda sin vectores)')
            return
        parts.append((e, i))
    # unir los meses en streaming (memmap): un año reciente ocupa ~7 GB en float16
    shapes = [np.load(e, mmap_mode='r').shape for e, _ in parts]
    total, dim = sum(s[0] for s in shapes), shapes[0][1]
    name = os.path.basename(seg)
    ye = os.path.join(ROOT, 'emb', f'g2_{name}.f16.npy')
    yi = os.path.join(ROOT, 'emb', f'g2_{name}.ids.npy')
    E = np.lib.format.open_memmap(ye, mode='w+', dtype=np.float16, shape=(total, dim))
    I = np.lib.format.open_memmap(yi, mode='w+', dtype=np.uint32, shape=(total,))
    o, prev = 0, -1
    for (e, i), s in zip(parts, shapes):
        ids = np.load(i)
        if len(ids) and (ids[0] <= prev or np.any(np.diff(ids.astype(np.int64)) <= 0)):
            raise SystemExit(f'ids no crecientes en {i}: no se puede unir en streaming')
        E[o:o + s[0]] = np.load(e, mmap_mode='r')
        I[o:o + s[0]] = ids
        prev = int(ids[-1]) if len(ids) else prev
        o += s[0]
    E.flush(); I.flush()
    del E, I
    log(f'  vectores {name}: {total:,} ({bits} bits)')
    try:
        subprocess.run([HNZ, 'index', '--seg', seg, '--emb', ye, '--ids', yi, '--bits', str(bits)], check=True, cwd=ROOT)
    finally:
        os.remove(ye); os.remove(yi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', type=years_arg, required=True)
    ap.add_argument('--hot-years', type=int, default=2, help='nº de años más recientes del DATASET en nivel zstd')
    ap.add_argument('--hot-from', type=int, default=None, help='año a partir del cual el nivel es zstd (anula --hot-years)')
    ap.add_argument('--all-zstd', action='store_true')
    ap.add_argument('--retier', action='store_true', help='enfriar a lm los segmentos zstd de años fríos (hnz retier)')
    ap.add_argument('--delete-old', action='store_true', help='con --retier: verificar todo y borrar el original')
    ap.add_argument('--reindex', action='store_true', help='rehacer los vectores si tienen otros bits que los del nivel')
    ap.add_argument('--steps', default='download,embed,build,index')
    ap.add_argument('--offline', action='store_true', help='sin conexión: solo los Parquet locales (también con HF_HUB_OFFLINE=1)')
    ap.add_argument('--device', default='GPU')
    ap.add_argument('--lm', default=os.path.join('models', 'lm-pythia-70m.hnzlm'))
    a = ap.parse_args()
    steps = set(a.steps.split(','))
    global _online
    _online = not (a.offline or os.environ.get('HF_HUB_OFFLINE', '').lower() in ('1', 'true', 'yes', 'on'))
    if not _online:
        log('sin conexión: solo se usan los Parquet locales')
    # el límite caliente/frío se fija respecto al año más reciente del DATASET (no del tramo pedido),
    # para que ejecutar por tramos dé el mismo resultado; sin conexión, el último año visto en línea
    if a.all_zstd:
        hot_from = min(a.years)
        log('todo en nivel zstd')
    else:
        if a.hot_from is not None:
            hot_from = a.hot_from
        elif _online and listing():
            hot_from = max(listing()) - a.hot_years + 1
        elif not _online and os.path.exists(STATE):
            hot_from = json.load(open(STATE, encoding='utf-8'))['dataset_max_year'] - a.hot_years + 1
        else:
            raise SystemExit('no se sabe cuál es el año más reciente del dataset: indique --hot-from')
        log(f'nivel zstd para años >= {hot_from}; nivel LM para el resto')
    for y in a.years:
        tier = 'zstd' if y >= hot_from else 'lm'
        log(f'== {y} (nivel {tier})')
        if ('download' in steps or 'embed' in steps) and year_done(y):
            # ya construido e indexado y al día: no rebajar Parquet ni recalcular embeddings
            # que se hayan borrado para ahorrar espacio
            log(f'  seg-{y} completo y al día con el dataset: no se baja ni se embebe nada')
            steps_y = steps - {'download', 'embed'}
        else:
            steps_y = steps
        if 'download' in steps_y:
            download(y)
        if 'embed' in steps_y:
            embed(y, a.device)
        seg = os.path.join(ROOT, 'out', f'seg-{y}')
        if 'build' in steps_y:
            seg = build(y, tier, a)
        if seg and 'index' in steps_y and os.path.exists(os.path.join(seg, 'manifest.json')) \
                and os.path.exists(os.path.join(seg, 'sources.json')):
            index(seg, a)
    log('listo')


if __name__ == '__main__':
    main()
