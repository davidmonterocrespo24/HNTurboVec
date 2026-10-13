"""Pipeline completo y REANUDABLE para construir el archivo de HN por años.

Pasos por año (se salta lo que ya está hecho):
  1. descargar los Parquet mensuales de open-index/hacker-news (data/YYYY/YYYY-MM.parquet)
  2. embeddings EmbeddingGemma 2 en la iGPU (embed_month.py)              -> emb/g2_YYYY-MM.*
  3. segmento del año: nivel zstd si es reciente, nivel LM si es antiguo   -> out/seg-YYYY
     (se reconstruye si cambian los meses disponibles, p.ej. el año en curso)
  4. índice vectorial del año: 768 bits en caliente, 512 en frío (E19)

Ejemplos:
  python pipeline.py --years 2006-2026                       # todo HN (caliente = últimos 2 años)
  python pipeline.py --years 2024 --steps download,embed     # solo preparar 2024
  python pipeline.py --years 2006-2015 --all-zstd            # rápido (horas); enfriar luego con hnz retier

La compresión LM de todo HN tarda días en CPU: una estrategia práctica es construir todo con
--all-zstd y enfriar después los años viejos con `hnz retier --tier lm` (conserva el original como .old).
Los segmentos de un mismo out/ no pueden solaparse: no mezcle segmentos mensuales (seg-YYYY-MM) y
anuales (seg-YYYY) del mismo periodo.
"""
import argparse, glob, json, os, re, subprocess, sys, time
import numpy as np

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


def listing():
    """Ficheros mensuales del dataset: {año: [ruta relativa, ...]} (solo YYYY/YYYY-MM.parquet)."""
    global _listing
    if _listing is None:
        from huggingface_hub import list_repo_files
        _listing = {}
        for f in list_repo_files(REPO, repo_type='dataset'):
            m = MONTH_RE.match(f)
            if m:
                _listing.setdefault(int(m.group(1)), []).append(f)
        for y in _listing:
            _listing[y].sort()
    return _listing


def download(year):
    from huggingface_hub import hf_hub_download
    files = listing().get(year, [])
    if not files:
        log(f'  (sin ficheros para {year} en el dataset; se omite)')
    for f in files:
        dst = os.path.join(ROOT, f)
        if not os.path.exists(dst):
            log(f'  descargando {f}')
            p = hf_hub_download(REPO, f, repo_type='dataset', local_dir=os.path.join(ROOT, '_dl'))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.replace(p, dst)


def months(year):
    return sorted(glob.glob(os.path.join(ROOT, 'data', str(year), f'{year}-[0-9][0-9].parquet')))


def embed(year, device):
    for p in months(year):
        m = os.path.basename(p).replace('.parquet', '')
        if os.path.exists(os.path.join(ROOT, 'emb', f'g2_{m}.ids.npy')):
            continue
        log(f'  embeddings {m} ({device})')
        subprocess.run([PY, os.path.join(ROOT, 'embed_month.py'), p, device], check=True, cwd=ROOT)


def check_no_monthly_overlap(year):
    mon = [d for d in glob.glob(os.path.join(ROOT, 'out', f'seg-{year}-[0-9][0-9]')) if os.path.exists(os.path.join(d, 'manifest.json'))]
    if mon:
        raise SystemExit(f'out/ contiene segmentos mensuales de {year} ({", ".join(os.path.basename(x) for x in mon)}) que '
                         f'se solaparían con out/seg-{year}: muévalos a otra carpeta antes de seguir.')


def build(year, tier, lm):
    seg = os.path.join(ROOT, 'out', f'seg-{year}')
    ps = months(year)
    if not ps:
        log(f'  (sin Parquet para {year}; no se construye)')
        return None
    src = [os.path.basename(p) for p in ps]
    side = os.path.join(seg, 'sources.json')
    if os.path.exists(os.path.join(seg, 'manifest.json')) and os.path.exists(side):
        prev = json.load(open(side, encoding='utf-8'))
        if prev.get('months') == src and prev.get('tier') == tier:
            return seg
        log(f'  los meses o el nivel de {year} han cambiado: se reconstruye')
    check_no_monthly_overlap(year)
    log(f'  segmento {year} nivel {tier} ({len(ps)} meses)')
    cmd = [HNZ, 'build', '--out', seg, '--tier', tier] + (['--lm', lm] if tier == 'lm' else []) + ps
    subprocess.run(cmd, check=True, cwd=ROOT)
    json.dump({'months': src, 'tier': tier}, open(side, 'w', encoding='utf-8'))
    return seg


def index(year, seg, bits):
    man = json.load(open(os.path.join(seg, 'manifest.json'), encoding='utf-8'))
    if man.get('vec_bits', 0) == bits:
        return
    parts = []
    for p in months(year):
        m = os.path.basename(p).replace('.parquet', '')
        e = os.path.join(ROOT, 'emb', f'g2_{m}.f16.npy')
        i = os.path.join(ROOT, 'emb', f'g2_{m}.ids.npy')
        if not (os.path.exists(e) and os.path.exists(i)):
            log(f'  (sin embeddings para {m}: el segmento queda sin vectores)')
            return
        parts.append((e, i))
    # unir los meses en streaming (memmap): un año reciente ocupa ~7 GB en float16
    shapes = [np.load(e, mmap_mode='r').shape for e, _ in parts]
    total, dim = sum(s[0] for s in shapes), shapes[0][1]
    ye = os.path.join(ROOT, 'emb', f'g2_{year}.f16.npy')
    yi = os.path.join(ROOT, 'emb', f'g2_{year}.ids.npy')
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
    log(f'  vectores {year}: {total:,} ({bits} bits)')
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
    ap.add_argument('--steps', default='download,embed,build,index')
    ap.add_argument('--device', default='GPU')
    ap.add_argument('--lm', default=os.path.join('models', 'lm-pythia-70m.hnzlm'))
    a = ap.parse_args()
    steps = set(a.steps.split(','))
    # el límite caliente/frío se fija respecto al año más reciente del DATASET (no del tramo pedido),
    # para que ejecutar por tramos dé el mismo resultado
    hot_from = a.hot_from if a.hot_from is not None else max(listing()) - a.hot_years + 1
    log(f'nivel zstd para años >= {hot_from}; nivel LM para el resto' if not a.all_zstd else 'todo en nivel zstd')
    for y in a.years:
        tier = 'zstd' if a.all_zstd or y >= hot_from else 'lm'
        log(f'== {y} (nivel {tier})')
        if 'download' in steps:
            download(y)
        if 'embed' in steps:
            embed(y, a.device)
        seg = os.path.join(ROOT, 'out', f'seg-{y}')
        if 'build' in steps:
            seg = build(y, tier, a.lm)
        if seg and 'index' in steps and os.path.exists(os.path.join(seg, 'manifest.json')):
            index(y, seg, 768 if tier == 'zstd' else 512)
    log('listo')


if __name__ == '__main__':
    main()
