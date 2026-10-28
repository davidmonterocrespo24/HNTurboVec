"""Full, RESUMABLE pipeline that builds the HN archive year by year.

Steps per year (whatever is already done is skipped):
  1. download the monthly Parquet files of open-index/hacker-news (data/YYYY/YYYY-MM.parquet); a month
     that changed in the dataset (e.g. the current month, which is published partially) is downloaded again
  2. EmbeddingGemma 2 embeddings on the iGPU (embed_month.py)             -> emb/g2_YYYY-MM.*
     (recomputed if their Parquet changed)
  3. the year's segment: zstd tier if recent, LM tier if old               -> out/seg-YYYY
     (rebuilt if the months or their content change; never just because of the tier: a segment
     that is already built is cooled with `hnz retier`, or with --retier here)
  4. the year's vector index: 768 bits in zstd, 512 in LM (E19)
A year that is already built and indexed with the same months and sizes as the dataset is not
downloaded or embedded again: its Parquet files and embeddings can be deleted to save space (except
those of the current year, which is rebuilt in full when the dataset adds a month). Segments without
sources.json (built by hand with `hnz build`) are left alone.

Examples (with the .venv environment active, or with .venv\\Scripts\\python instead of python):
  python scripts/pipeline.py --years 2006-2026                       # all of HN (hot = last 2 years)
  python scripts/pipeline.py --years 2024 --steps download,embed     # only prepare 2024
  python scripts/pipeline.py --years 2006-2015 --all-zstd            # fast (hours); cool later with --retier
  python scripts/pipeline.py --years 2006-2024 --steps build --retier   # cool the cold years already built to LM
  python scripts/pipeline.py --years 2026 --offline                  # offline: local Parquet files only

LM compression of all of HN takes days on CPU. A practical strategy is to build everything with
--all-zstd and later cool the old years with --retier (or `hnz retier --tier lm`), which keeps the
original as <seg>.old unless --delete-old is given (it verifies everything before deleting it).
If the dataset does not respond, the pipeline stops (it does not silently switch to offline mode).
With --offline, the hot/cold boundary comes from the last year seen online (out/pipeline_state.json).
Segments in the same out/ must not overlap: do not mix monthly segments (seg-YYYY-MM) and
yearly segments (seg-YYYY) for the same period.
"""
import argparse, glob, json, os, re, subprocess, sys, time
import numpy as np
# huggingface_hub's hf_xet downloader dies mid-download on this Windows machine (exit 127,
# E26): use the classic HTTP download instead (also resumable). Subprocesses inherit this.
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # project root
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
STATE = os.path.join(ROOT, 'out', 'pipeline_state.json')   # last year seen in the dataset


def listing():
    """Monthly files of the dataset: {year: {relative path: size}} (only YYYY/YYYY-MM.parquet).
    With --offline (or HF_HUB_OFFLINE=1), the local Parquet files. A network failure does NOT
    silently switch to offline mode: the hot/cold boundary and the downloads depend on the real listing."""
    global _listing
    if _listing is None:
        if _online:
            try:
                from huggingface_hub import HfApi
                tree = HfApi().list_repo_tree(REPO, path_in_repo='data', repo_type='dataset', recursive=True)
                files = [(f.path, f.size) for f in tree if getattr(f, 'size', None) is not None]
            except Exception as e:
                raise SystemExit(f'could not list the dataset {REPO} ({type(e).__name__}: {str(e)[:200]}): '
                                 'retry, or use --offline to work with the local Parquet files only')
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
    """Is seg-YYYY already built and indexed with the same months and sizes as the dataset?
    Then there is nothing to download or embed (its Parquet files and embeddings may be deleted)."""
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
    # offline, only the local Parquet files are visible: done if none is new or different
    return all(built.get(n) == s for n, s in avail.items())


def download(year):
    files = listing().get(year, {})
    if not files:
        log(f'  (no files for {year} in the dataset; skipped)')
    if not _online:
        return
    # byte-level resumable download with SHA-256 check (download_hn.py)
    import download_hn
    if download_hn.run({year})['fallo']:
        raise SystemExit(f'could not download all months of {year}: run again to resume')


def months(year):
    return sorted(glob.glob(os.path.join(ROOT, 'data', str(year), f'{year}-[0-9][0-9].parquet')))


def emb_paths(m):
    e = os.path.join(ROOT, 'emb', f'g2_{m}')
    return e + '.f16.npy', e + '.ids.npy', e + '.src.json'


def emb_fresh(p):
    """Are there embeddings computed from THIS Parquet? (a record with its size)"""
    e, i, side = emb_paths(os.path.basename(p).replace('.parquet', ''))
    if not (os.path.exists(e) and os.path.exists(i)):
        return False
    if not os.path.exists(side):
        # embeddings without a record (embed_month.py by hand, or an older pipeline): adopt them only
        # if they are newer than the Parquet; a freshly re-downloaded one is newer -> recompute
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
        # delete the old files first: if interrupted, no files from two versions are left
        for f in emb_paths(m):
            if os.path.exists(f):
                os.remove(f)
        subprocess.run([PY, os.path.join(ROOT, 'scripts', 'embed_month.py'), p, device], check=True, cwd=ROOT)
        json.dump({'parquet_size': os.path.getsize(p)}, open(emb_paths(m)[2], 'w', encoding='utf-8'))


def check_no_monthly_overlap(year):
    mon = [d for d in glob.glob(os.path.join(ROOT, 'out', f'seg-{year}-[0-9][0-9]')) if os.path.exists(os.path.join(d, 'manifest.json'))]
    if mon:
        raise SystemExit(f'out/ holds monthly segments of {year} ({", ".join(os.path.basename(x) for x in mon)}) that '
                         f'would overlap with out/seg-{year}: move them to another folder before going on.')


def seg_tier(seg):
    return json.load(open(os.path.join(seg, 'manifest.json'), encoding='utf-8')).get('text_tier')


def build(year, tier, a):
    seg = os.path.join(ROOT, 'out', f'seg-{year}')
    man = os.path.join(seg, 'manifest.json')
    exists = os.path.exists(man)
    side = os.path.join(seg, 'sources.json')
    # provisional record, written BEFORE `hnz build` (which replaces the whole folder)
    pending = seg + '.sources.json'
    if exists and not os.path.exists(side) and os.path.exists(pending) and os.path.getmtime(man) >= os.path.getmtime(pending):
        # the build finished but the process died before moving the record: adopt it
        os.replace(pending, side)
    prev = json.load(open(side, encoding='utf-8')) if exists and os.path.exists(side) else None
    if exists and prev is None:
        # built outside the pipeline (hnz build by hand): we do not know which months it holds, and
        # rebuilding it from the local Parquet files could lose data
        log(f'  {os.path.basename(seg)} has no sources.json (not built by pipeline.py): left alone')
        return None
    ps = months(year)
    src = [os.path.basename(p) for p in ps]
    sizes = {os.path.basename(p): os.path.getsize(p) for p in ps}
    if prev is not None:
        missing = sorted(set(prev.get('months', [])) - set(src))
        if missing:
            # e.g. already archived Parquet files were deleted to save space: rebuilding would lose data
            log(f'  local Parquet files that seg-{year} already holds are missing ({", ".join(missing)}): not rebuilt')
            return retier(seg, tier, a)
        if prev.get('months') == src and prev.get('sizes') == sizes:
            return retier(seg, tier, a)
        log(f'  the months of {year} or their content changed: rebuilding')
    if not ps:
        log(f'  (no Parquet files for {year}; not built)')
        return None
    check_no_monthly_overlap(year)
    log(f'  segment {year} tier {tier} ({len(ps)} months)')
    json.dump({'months': src, 'sizes': sizes, 'tier': tier}, open(pending, 'w', encoding='utf-8'))
    cmd = [HNZ, 'build', '--out', seg, '--tier', tier] + (['--lm', a.lm] if tier == 'lm' else []) + ps
    subprocess.run(cmd, check=True, cwd=ROOT)
    os.replace(pending, side)
    return seg


def retier(seg, tier, a):
    """A segment that is already built is not rebuilt because its tier changes: it is cooled with retier."""
    cur = seg_tier(seg)
    if cur == 'zstd' and tier == 'lm':
        if not a.retier:
            log(f'  {os.path.basename(seg)} is in zstd but should be lm: use --retier (or hnz retier --tier lm)')
            return seg
        log(f'  cooling {os.path.basename(seg)} to the lm tier')
        cmd = [HNZ, 'retier', '--seg', seg, '--tier', 'lm', '--lm', a.lm] + (['--verify-all', '--delete-old'] if a.delete_old else [])
        subprocess.run(cmd, check=True, cwd=ROOT)
    return seg


def index(seg, a):
    man = json.load(open(os.path.join(seg, 'manifest.json'), encoding='utf-8'))
    # bits follow the segment's REAL tier (a cold year not cooled yet keeps 768)
    bits = 768 if man.get('text_tier') == 'zstd' else 512
    if man.get('vec_bits', 0) == bits or (man.get('vec_bits', 0) > 0 and not a.reindex):
        return
    side = os.path.join(seg, 'sources.json')
    if not os.path.exists(side):
        log(f'  ({os.path.basename(seg)} has no sources.json: unknown which embeddings belong to it; not indexed)')
        return
    names = [x.replace('.parquet', '') for x in json.load(open(side, encoding='utf-8'))['months']]
    if not names:
        log(f'  (no known months for {os.path.basename(seg)}: not indexed)')
        return
    parts = []
    for m in names:
        e, i, _ = emb_paths(m)
        if not (os.path.exists(e) and os.path.exists(i)):
            log(f'  (no embeddings for {m}: the segment stays without vectors)')
            return
        parts.append((e, i))
    # join the months by streaming (memmap): a recent year takes ~7 GB in float16
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
            raise SystemExit(f'ids not increasing in {i}: cannot join by streaming')
        E[o:o + s[0]] = np.load(e, mmap_mode='r')
        I[o:o + s[0]] = ids
        prev = int(ids[-1]) if len(ids) else prev
        o += s[0]
    E.flush(); I.flush()
    del E, I
    log(f'  vectors {name}: {total:,} ({bits} bits)')
    try:
        subprocess.run([HNZ, 'index', '--seg', seg, '--emb', ye, '--ids', yi, '--bits', str(bits)], check=True, cwd=ROOT)
    finally:
        os.remove(ye); os.remove(yi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', type=years_arg, required=True)
    ap.add_argument('--hot-years', type=int, default=2, help='number of most recent DATASET years in the zstd tier')
    ap.add_argument('--hot-from', type=int, default=None, help='year from which the tier is zstd (overrides --hot-years)')
    ap.add_argument('--all-zstd', action='store_true')
    ap.add_argument('--retier', action='store_true', help='cool the zstd segments of cold years to lm (hnz retier)')
    ap.add_argument('--delete-old', action='store_true', help='with --retier: verify everything and delete the original')
    ap.add_argument('--reindex', action='store_true', help='redo the vectors if their bits differ from the tier\'s')
    ap.add_argument('--steps', default='download,embed,build,index')
    ap.add_argument('--offline', action='store_true', help='offline: local Parquet files only (also with HF_HUB_OFFLINE=1)')
    ap.add_argument('--device', default='GPU')
    ap.add_argument('--lm', default=os.path.join('models', 'lm-pythia-70m.hnzlm'))
    a = ap.parse_args()
    steps = set(a.steps.split(','))
    global _online
    _online = not (a.offline or os.environ.get('HF_HUB_OFFLINE', '').lower() in ('1', 'true', 'yes', 'on'))
    if not _online:
        log('offline: only the local Parquet files are used')
    # the hot/cold boundary is set from the most recent year of the DATASET (not of the requested range),
    # so that running range by range gives the same result; offline, the last year seen online
    if a.all_zstd:
        hot_from = min(a.years)
        log('everything in the zstd tier')
    else:
        if a.hot_from is not None:
            hot_from = a.hot_from
        elif _online and listing():
            hot_from = max(listing()) - a.hot_years + 1
        elif not _online and os.path.exists(STATE):
            hot_from = json.load(open(STATE, encoding='utf-8'))['dataset_max_year'] - a.hot_years + 1
        else:
            raise SystemExit('the most recent year of the dataset is unknown: pass --hot-from')
        log(f'zstd tier for years >= {hot_from}; LM tier for the rest')
    for y in a.years:
        tier = 'zstd' if y >= hot_from else 'lm'
        log(f'== {y} (tier {tier})')
        if ('download' in steps or 'embed' in steps) and year_done(y):
            # already built, indexed and up to date: do not re-download Parquet files or recompute
            # embeddings that were deleted to save space
            log(f'  seg-{y} complete and up to date with the dataset: nothing to download or embed')
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
    log('done')


if __name__ == '__main__':
    main()
