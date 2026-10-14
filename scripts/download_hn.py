"""Descarga TODO Hacker News (dataset open-index/hacker-news de Hugging Face) de forma reanudable.

- Fija la versión (commit) del dataset al empezar: tamaños y SHA-256 coherentes aunque se actualice
  mientras tanto.
- Cada fichero se baja a <destino>.part con peticiones HTTP Range: si se corta (red, apagado,
  Ctrl+C), al volver a ejecutar continúa desde el último byte guardado. (hf_hub_download no sirve:
  cada intento usa un temporal con nombre aleatorio y empieza de cero.)
- Comprueba tamaño y SHA-256 (los publica el propio dataset) antes de darlo por bueno; un fichero
  corrupto se borra y se vuelve a bajar.
- Reintentos con espera creciente y varios ficheros en paralelo.
- Registro data/_descargas.json (fichero -> tamaño, sha256, versión): un fichero ya comprobado no
  se vuelve a leer entero. Un mes que cambie en el dataset (el mes en curso se publica a medias)
  se vuelve a bajar.

Uso (con el entorno .venv activado):
  python scripts/download_hn.py                    # todos los meses: data/YYYY/YYYY-MM.parquet (~12,4 GB)
  python scripts/download_hn.py --years 2010-2012  # solo esos años
  python scripts/download_hn.py --verify           # recalcula el SHA-256 de todo lo ya bajado
Si se corta, basta con volver a lanzar el mismo comando.
"""
import argparse, concurrent.futures as cf, hashlib, json, os, re, sys, threading, time

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')   # hf_xet muere a mitad de descarga en esta máquina (E26)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # raíz del proyecto
REPO = 'open-index/hacker-news'
MONTH_RE = re.compile(r'^data/(\d{4})/(\d{4})-(\d{2})\.parquet$')
REGISTRO = os.path.join(ROOT, 'data', '_descargas.json')
_lock = threading.Lock()
_remote = None


def log(*a):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), *a, flush=True)


def remote_files():
    """{ruta: (tamaño, sha256)} de los Parquet mensuales en una versión fija del dataset."""
    global _remote
    if _remote is None:
        from huggingface_hub import HfApi
        api = HfApi()
        rev = api.dataset_info(REPO).sha
        files = {}
        for f in api.list_repo_tree(REPO, path_in_repo='data', repo_type='dataset', recursive=True, revision=rev):
            if getattr(f, 'size', None) is not None and MONTH_RE.match(f.path):
                files[f.path] = (f.size, f.lfs.sha256 if getattr(f, 'lfs', None) else None)
        _remote = (rev, files)
    return _remote


def load_registro():
    try:
        return json.load(open(REGISTRO, encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def save_registro(reg):
    tmp = REGISTRO + '.tmp'
    json.dump(reg, open(tmp, 'w', encoding='utf-8'), indent=0, sort_keys=True)
    os.replace(tmp, REGISTRO)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def fetch(path, size, sha, rev, reg, attempts=10):
    """Baja (o completa) un fichero; devuelve 'ok' (ya estaba), 'bajado' o 'fallo'."""
    import httpx
    from huggingface_hub import hf_hub_url
    dst = os.path.join(ROOT, *path.split('/'))
    part = dst + '.part'
    with _lock:
        done = reg.get(path)
    if os.path.exists(dst) and os.path.getsize(dst) == size and (sha is None or (done and done.get('sha256') == sha)):
        return 'ok'
    if os.path.exists(dst) and os.path.getsize(dst) == size and sha is not None and not os.path.exists(part):
        # bajado antes del registro (p.ej. con pipeline.py): se comprueba una vez
        if sha256_of(dst) == sha:
            with _lock:
                reg[path] = {'size': size, 'sha256': sha, 'rev': rev}
                save_registro(reg)
            return 'ok'
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    url = hf_hub_url(REPO, path, repo_type='dataset', revision=rev)
    for k in range(1, attempts + 1):
        try:
            have = os.path.getsize(part) if os.path.exists(part) else 0
            if have > size:
                os.remove(part)
                have = 0
            t0, start = time.time(), have
            if have < size:
                with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60, connect=30)) as c:
                    with c.stream('GET', url, headers={'Range': f'bytes={have}-'}) as r:
                        if r.status_code == 200 and have:
                            have = 0                      # el servidor ignoró el Range: desde cero
                        elif r.status_code not in (200, 206):
                            raise IOError(f'HTTP {r.status_code}')
                        with open(part, 'ab' if have else 'wb') as f:
                            for chunk in r.iter_bytes(1 << 20):
                                f.write(chunk)
            got = os.path.getsize(part)
            if got != size:
                raise IOError(f'incompleto: {got:,} de {size:,} B')
            if sha is not None and sha256_of(part) != sha:
                os.remove(part)
                raise IOError('SHA-256 distinto: se descarta y se vuelve a bajar')
            os.replace(part, dst)
            with _lock:
                reg[path] = {'size': size, 'sha256': sha, 'rev': rev}
                save_registro(reg)
            mb = (size - start) / 1e6
            log(f'  {path}: {size / 1e6:.1f} MB ({mb / max(time.time() - t0, 1e-3):.1f} MB/s)' + (' [reanudado]' if start else ''))
            return 'bajado'
        except Exception as e:
            wait = min(300, 5 * 2 ** (k - 1))
            log(f'  {path}: intento {k}/{attempts} falló ({type(e).__name__}: {str(e)[:120]}); reintento en {wait} s')
            time.sleep(wait)
    return 'fallo'


def run(years=None, workers=4, verify=False):
    rev, files = remote_files()
    reg = load_registro()
    sel = {p: v for p, v in files.items() if years is None or int(MONTH_RE.match(p).group(1)) in years}
    if verify:
        for p, (size, sha) in sorted(sel.items()):
            dst = os.path.join(ROOT, *p.split('/'))
            if os.path.exists(dst) and (os.path.getsize(dst) != size or (sha and sha256_of(dst) != sha)):
                log(f'  {p}: no coincide con el dataset; se volverá a bajar')
                reg.pop(p, None)
                os.remove(dst)
        save_registro(reg)
    total = sum(v[0] for v in sel.values())
    log(f'dataset {REPO} versión {rev[:12]}: {len(sel)} meses, {total / 1e9:.2f} GB')
    res = {'ok': 0, 'bajado': 0, 'fallo': 0}
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fetch, p, v[0], v[1], rev, reg): p for p, v in sorted(sel.items())}
        for fu in cf.as_completed(futs):
            res[fu.result()] += 1
    have = sum(os.path.getsize(os.path.join(ROOT, *p.split('/'))) for p in sel if os.path.exists(os.path.join(ROOT, *p.split('/'))))
    log(f'resumen: {res["ok"]} ya estaban, {res["bajado"]} bajados, {res["fallo"]} con fallo; '
        f'{have / 1e9:.2f} de {total / 1e9:.2f} GB en data/')
    if res['fallo']:
        log('hay ficheros sin bajar: vuelva a ejecutar el mismo comando para reanudar')
    return res


def years_arg(s):
    if '-' in s:
        a, b = s.split('-')
        return set(range(int(a), int(b) + 1))
    return {int(x) for x in s.split(',')}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--years', type=years_arg, default=None)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--verify', action='store_true')
    a = ap.parse_args()
    r = run(a.years, a.workers, a.verify)
    sys.exit(1 if r['fallo'] else 0)
