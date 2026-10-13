"""Pipeline completo y REANUDABLE para construir el archivo de HN por años.

Pasos por año (se salta lo que ya existe):
  1. descargar los Parquet mensuales de open-index/hacker-news
  2. embeddings EmbeddingGemma 2 en la iGPU (embed_month.py)            -> emb/g2_YYYY-MM.*
  3. segmento del año: nivel zstd si es reciente, nivel LM si es antiguo -> out/seg-YYYY
  4. índice vectorial del año (une los embeddings mensuales)

Ejemplos:
  python pipeline.py --years 2007-2026 --hot-years 2          # todo HN
  python pipeline.py --years 2024 --steps download,embed       # solo preparar 2024
  python pipeline.py --years 2010-2015 --all-zstd              # rápido (horas), retier después

Como la compresión LM tarda días para todo HN, una estrategia práctica es construir todo con
--all-zstd (rápido) y luego enfriar los años viejos con `hnz retier --tier lm` poco a poco.
"""
import argparse, os, subprocess, sys, time, glob
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
HNZ = os.path.join(ROOT, 'hnz', 'target', 'release', 'hnz.exe' if os.name == 'nt' else 'hnz')
PY = sys.executable


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def years_arg(s):
    if '-' in s:
        a, b = s.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(',')]


def download(year):
    from huggingface_hub import list_repo_files, hf_hub_download
    files = [f for f in list_repo_files('open-index/hacker-news', repo_type='dataset') if f.startswith(f'data/{year}/') and f.endswith('.parquet')]
    out = []
    for f in sorted(files):
        dst = os.path.join(ROOT, f)
        if not os.path.exists(dst):
            log(f'  descargando {f}')
            p = hf_hub_download('open-index/hacker-news', f, repo_type='dataset', local_dir=os.path.join(ROOT, '_dl'))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.replace(p, dst)
        out.append(dst)
    return out


def months(year):
    return sorted(glob.glob(os.path.join(ROOT, 'data', str(year), f'{year}-*.parquet')))


def embed(year, device):
    for p in months(year):
        m = os.path.basename(p).replace('.parquet', '')
        if os.path.exists(os.path.join(ROOT, 'emb', f'g2_{m}.ids.npy')):
            continue
        log(f'  embeddings {m} ({device})')
        subprocess.run([PY, os.path.join(ROOT, 'embed_month.py'), p, device], check=True, cwd=ROOT)


def build(year, tier, lm):
    seg = os.path.join(ROOT, 'out', f'seg-{year}')
    if os.path.exists(os.path.join(seg, 'manifest.json')):
        return seg
    ps = months(year)
    if not ps:
        raise SystemExit(f'no hay Parquet para {year}')
    log(f'  segmento {year} nivel {tier} ({len(ps)} meses)')
    cmd = [HNZ, 'build', '--out', seg, '--tier', tier] + (['--lm', lm] if tier == 'lm' else []) + ps
    subprocess.run(cmd, check=True, cwd=ROOT)
    return seg


def index(year, seg):
    import json
    man = json.load(open(os.path.join(seg, 'manifest.json')))
    if man.get('vec_bits', 0) > 0:
        return
    embs, ids = [], []
    for p in months(year):
        m = os.path.basename(p).replace('.parquet', '')
        e = os.path.join(ROOT, 'emb', f'g2_{m}.f16.npy')
        if not os.path.exists(e):
            log(f'  (sin embeddings para {m}: el segmento quedará sin vectores)')
            return
        embs.append(np.load(e)); ids.append(np.load(os.path.join(ROOT, 'emb', f'g2_{m}.ids.npy')))
    E = np.concatenate(embs); I = np.concatenate(ids)
    order = np.argsort(I, kind='stable')
    ye = os.path.join(ROOT, 'emb', f'g2_{year}.f16.npy'); yi = os.path.join(ROOT, 'emb', f'g2_{year}.ids.npy')
    np.save(ye, E[order]); np.save(yi, I[order])
    log(f'  vectores {year}: {len(I):,}')
    subprocess.run([HNZ, 'index', '--seg', seg, '--emb', ye, '--ids', yi], check=True, cwd=ROOT)
    os.remove(ye); os.remove(yi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', type=years_arg, required=True)
    ap.add_argument('--hot-years', type=int, default=2, help='años más recientes en nivel zstd')
    ap.add_argument('--all-zstd', action='store_true')
    ap.add_argument('--steps', default='download,embed,build,index')
    ap.add_argument('--device', default='GPU')
    ap.add_argument('--lm', default=os.path.join('models', 'lm-pythia-70m.hnzlm'))
    a = ap.parse_args()
    steps = set(a.steps.split(','))
    last = max(a.years)
    for y in a.years:
        tier = 'zstd' if a.all_zstd or y > last - a.hot_years else 'lm'
        log(f'== {y} (nivel {tier})')
        if 'download' in steps:
            download(y)
        if 'embed' in steps:
            embed(y, a.device)
        seg = os.path.join(ROOT, 'out', f'seg-{y}')
        if 'build' in steps:
            seg = build(y, tier, a.lm)
        if 'index' in steps and os.path.exists(os.path.join(seg, 'manifest.json')):
            index(y, seg)
    log('listo')


if __name__ == '__main__':
    main()
