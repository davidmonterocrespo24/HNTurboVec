"""Downloads ALL of Hacker News (the open-index/hacker-news dataset on Hugging Face), resumably.

- Pins the dataset version (commit) at start: sizes and SHA-256 stay consistent even if the
  dataset is updated meanwhile.
- Each file is downloaded to <destination>.part with HTTP Range requests: if it stops (network,
  shutdown, Ctrl+C), running again continues from the last saved byte. (hf_hub_download does not
  work for this: each attempt uses a temp file with a random name and starts from zero.)
- Checks size and SHA-256 (published by the dataset itself) before accepting a file; a corrupt
  file is deleted and downloaded again.
- Retries with growing waits, and several files in parallel.
- Registry data/_downloads.json (file -> size, sha256, version): a file already checked is not
  read in full again. A month that changes in the dataset (the current month is published
  partially) is downloaded again.

Usage (with the .venv environment active):
  python scripts/download_hn.py                    # all months: data/YYYY/YYYY-MM.parquet (~12.4 GB)
  python scripts/download_hn.py --years 2010-2012  # only those years
  python scripts/download_hn.py --verify           # recompute the SHA-256 of everything already downloaded
If it stops, just run the same command again.
"""
import argparse, concurrent.futures as cf, hashlib, json, os, re, sys, threading, time

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')   # hf_xet dies mid-download on this machine (E26)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # project root
REPO = 'open-index/hacker-news'
MONTH_RE = re.compile(r'^data/(\d{4})/(\d{4})-(\d{2})\.parquet$')
REGISTRO = os.path.join(ROOT, 'data', '_downloads.json')
_lock = threading.Lock()
_remote = None


def log(*a):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), *a, flush=True)


def remote_files():
    """{path: (size, sha256)} of the monthly Parquet files in a pinned version of the dataset."""
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
    """Downloads (or completes) one file. Returns 'ok' (already there), 'bajado' (downloaded)
    or 'fallo' (failed). These Spanish keys are kept because pipeline.py reads them."""
    import httpx
    from huggingface_hub import hf_hub_url
    dst = os.path.join(ROOT, *path.split('/'))
    part = dst + '.part'
    with _lock:
        done = reg.get(path)
    if os.path.exists(dst) and os.path.getsize(dst) == size and (sha is None or (done and done.get('sha256') == sha)):
        return 'ok'
    if os.path.exists(dst) and os.path.getsize(dst) == size and sha is not None and not os.path.exists(part):
        # downloaded before the registry existed (e.g. by pipeline.py): check it once
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
                            have = 0                      # the server ignored the Range: start from zero
                        elif r.status_code not in (200, 206):
                            raise IOError(f'HTTP {r.status_code}')
                        with open(part, 'ab' if have else 'wb') as f:
                            for chunk in r.iter_bytes(1 << 20):
                                f.write(chunk)
            got = os.path.getsize(part)
            if got != size:
                raise IOError(f'incomplete: {got:,} of {size:,} B')
            if sha is not None and sha256_of(part) != sha:
                os.remove(part)
                raise IOError('SHA-256 mismatch: discarded, will download again')
            os.replace(part, dst)
            with _lock:
                reg[path] = {'size': size, 'sha256': sha, 'rev': rev}
                save_registro(reg)
            mb = (size - start) / 1e6
            log(f'  {path}: {size / 1e6:.1f} MB ({mb / max(time.time() - t0, 1e-3):.1f} MB/s)' + (' [resumed]' if start else ''))
            return 'bajado'
        except Exception as e:
            wait = min(300, 5 * 2 ** (k - 1))
            log(f'  {path}: attempt {k}/{attempts} failed ({type(e).__name__}: {str(e)[:120]}); retrying in {wait} s')
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
                log(f'  {p}: does not match the dataset; will download again')
                reg.pop(p, None)
                os.remove(dst)
        save_registro(reg)
    total = sum(v[0] for v in sel.values())
    log(f'dataset {REPO} version {rev[:12]}: {len(sel)} months,{total / 1e9:.2f} GB')
    res = {'ok': 0, 'bajado': 0, 'fallo': 0}
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fetch, p, v[0], v[1], rev, reg): p for p, v in sorted(sel.items())}
        for fu in cf.as_completed(futs):
            res[fu.result()] += 1
    have = sum(os.path.getsize(os.path.join(ROOT, *p.split('/'))) for p in sel if os.path.exists(os.path.join(ROOT, *p.split('/'))))
    log(f'summary: {res["ok"]} already there, {res["bajado"]} downloaded, {res["fallo"]} failed; '
        f'{have / 1e9:.2f} of {total / 1e9:.2f} GB in data/')
    if res['fallo']:
        log('some files are not downloaded: run the same command again to resume')
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
