"""P5: the material of the compact block of all of HN, month by month and resumable.

Two workers, each in its own process (start both; they do not depend on each other):
  lm     each month's LM segment: `hnz build --tier lm` with the HN LM (the CPU; ~4.7 days in all
         at ~122 records/s)                                          -> work/p5/seg-YYYY-MM
  embed  each month's document prompts (`hnz prompts`, v2) and their vectors on the iGPU
         (`scripts/embed_prompts.py`; ~3 days)                       -> emb/g2v2_YYYY-MM.*
Each worker skips the months it has finished, so stopping and starting it again loses at most the
month in progress (built under a temporary name, renamed when complete). Months with no items are
skipped. A failed month is logged and retried on the next start. Before each month a worker waits
while free RAM is below --min-free-gb, and while it runs it asks Windows not to sleep (the screen
may still lock: that does not stop anything).

Usage (from the project root):
  python scripts/p5_build.py lm      [--months 2006-10..2026-08] [--hnz work/bin/hnz-p5.exe]
  python scripts/p5_build.py embed   [--months ...] [--device GPU]
  python scripts/p5_build.py status
Logs: logs/p5_<worker>.log (one line per month) and logs/p5/<worker>-<month>.log (the tools' output).
"""
import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLOBALS = os.path.join(ROOT, 'work', 'global', 'globals.json')
SEG = os.path.join(ROOT, 'work', 'p5', 'seg-{m}')
PROMPTS = os.path.join(ROOT, 'work', 'p5', 'prompts')
EMB = os.path.join(ROOT, 'emb', 'g2v2_{m}')
LM = os.path.join(ROOT, 'models', 'lm-hn-70m.hnzlm')
LOGS = os.path.join(ROOT, 'logs')


class _Mem(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong),
                ('ullAvailPhys', ctypes.c_ulonglong), ('ullTotalPageFile', ctypes.c_ulonglong),
                ('ullAvailPageFile', ctypes.c_ulonglong), ('ullTotalVirtual', ctypes.c_ulonglong),
                ('ullAvailVirtual', ctypes.c_ulonglong), ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]


def free_gb():
    m = _Mem()
    m.dwLength = ctypes.sizeof(_Mem)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2 ** 30


def _k32():
    from ctypes import wintypes
    k = ctypes.windll.kernel32
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k.SetPriorityClass.restype = wintypes.BOOL
    k.SetThreadExecutionState.argtypes = [wintypes.DWORD]
    k.SetThreadExecutionState.restype = wintypes.DWORD
    return k


def keep_awake():
    """Asks Windows not to sleep while this process runs (ES_CONTINUOUS | ES_SYSTEM_REQUIRED); the
    display may still turn off and the session lock."""
    if os.name == 'nt':
        _k32().SetThreadExecutionState(0x80000000 | 0x00000001)


def set_priority(cls):
    """This process's priority class (the processes it starts inherit it)."""
    if os.name == 'nt':
        k = _k32()
        if not k.SetPriorityClass(k.GetCurrentProcess(), cls):
            print(f'warning: SetPriorityClass failed ({ctypes.GetLastError()})', flush=True)


def months_with_items(sel):
    files = json.load(open(GLOBALS, encoding='utf-8'))['files']
    out = []
    for f in files:
        m = os.path.basename(f['path'])[:-len('.parquet')]
        if f['rows'] > 0 and (sel is None or sel[0] <= m <= sel[1]):
            out.append((m, os.path.join(ROOT, f['path']), f['rows']))
    return out


def log(worker, msg):
    os.makedirs(LOGS, exist_ok=True)
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
    print(line, flush=True)
    with open(os.path.join(LOGS, f'p5_{worker}.log'), 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def run(cmd, worker, m):
    os.makedirs(os.path.join(LOGS, 'p5'), exist_ok=True)
    with open(os.path.join(LOGS, 'p5', f'{worker}-{m}.log'), 'a', encoding='utf-8') as f:
        f.write(f'$ {" ".join(cmd)}\n')
        f.flush()
        return subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode


def wait_for_ram(worker, need):
    warned = False
    while free_gb() < need:
        if not warned:
            log(worker, f'waiting: {free_gb():.1f} GB of RAM free (< {need} GB)')
            warned = True
        time.sleep(60)


def lm_done(m):
    return os.path.exists(os.path.join(SEG.format(m=m), 'manifest.json'))


def emb_done(m):
    return os.path.exists(EMB.format(m=m) + '.f16.npy') and os.path.exists(EMB.format(m=m) + '.ids.npy')


def worker_lm(months, hnz, min_free):
    for m, parquet, rows in months:
        out = SEG.format(m=m)
        if lm_done(m):
            continue
        wait_for_ram('lm', min_free)
        tmp = out + '.tmp'
        for d in (tmp, out):
            if os.path.exists(d):
                shutil.rmtree(d)
        t0 = time.time()
        rc = run([hnz, 'build', '--out', tmp, '--tier', 'lm', '--lm', LM, parquet], 'lm', m)
        if rc == 0 and os.path.exists(os.path.join(tmp, 'manifest.json')):
            os.replace(tmp, out)
            dt = time.time() - t0
            log('lm', f'{m}: {rows:,} items in {dt / 60:.1f} min ({rows / max(dt, 1e-9):.0f} items/s)')
        else:
            log('lm', f'{m}: FAILED (exit {rc}); see logs/p5/lm-{m}.log')


def worker_embed(months, hnz, device, min_free):
    os.makedirs(PROMPTS, exist_ok=True)
    py = sys.executable
    for m, parquet, rows in months:
        if emb_done(m):
            continue
        wait_for_ram('embed', min_free)
        t0 = time.time()
        tok = os.path.join(PROMPTS, f'{m}.tok.npy')
        if not os.path.exists(tok):
            # into a temporary folder, moved when complete (a cut run leaves no half prompts)
            tmp = os.path.join(PROMPTS, f'.tmp-{m}')
            if os.path.exists(tmp):
                shutil.rmtree(tmp)
            rc = run([hnz, 'prompts', '--out', tmp, '--lm', LM, parquet], 'embed', m)
            if rc != 0:
                log('embed', f'{m}: prompts FAILED (exit {rc}); see logs/p5/embed-{m}.log')
                continue
            for ext in ('ids.npy', 'len.npy', 'tok.npy'):   # tok last: it marks the prompts complete
                os.replace(os.path.join(tmp, f'{m}.{ext}'), os.path.join(PROMPTS, f'{m}.{ext}'))
            shutil.rmtree(tmp)
        t1 = time.time()
        rc = run([py, os.path.join(ROOT, 'scripts', 'embed_prompts.py'), PROMPTS, m, '--device', device, '--above-normal'], 'embed', m)
        if rc == 0 and emb_done(m):
            os.remove(tok)   # the token ids are only needed to embed (hnz prompts makes them again)
            log('embed', f'{m}: {rows:,} items, prompts {(t1 - t0) / 60:.1f} min, vectors {(time.time() - t1) / 60:.1f} min')
        else:
            log('embed', f'{m}: vectors FAILED (exit {rc}); see logs/p5/embed-{m}.log')


def status(months):
    n = len(months)
    rows = sum(r for _, _, r in months)
    for name, done in (('lm', lm_done), ('embed', emb_done)):
        d = [x for x in months if done(x[0])]
        dr = sum(r for _, _, r in d)
        print(f'{name:>5}: {len(d)}/{n} months, {dr:,}/{rows:,} items ({dr / rows:.1%})')
    print(f'free RAM {free_gb():.1f} GB')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('worker', choices=['lm', 'embed', 'status'])
    ap.add_argument('--months', help='a range YYYY-MM..YYYY-MM (default: every month with items)')
    ap.add_argument('--hnz', default=os.path.join(ROOT, 'work', 'bin', 'hnz-p5.exe'))
    ap.add_argument('--device', default='GPU')
    ap.add_argument('--min-free-gb', type=float, default=4.0)
    a = ap.parse_args()
    sel = tuple(a.months.split('..')) if a.months else None
    months = months_with_items(sel)
    if a.worker == 'status':
        status(months)
        return
    keep_awake()
    if a.worker == 'lm':
        # compression runs one large batch: every logical thread pays off (E20: 122 records/s with
        # 16 threads); hnz's default, half of them, suits the small batches of searches
        os.environ.setdefault('HNZ_LM_THREADS', str(os.cpu_count() or 8))
    if a.worker == 'embed' and os.name == 'nt':
        # above normal (its tools inherit it): it needs little CPU, but promptly, to keep the iGPU
        # busy next to the LM worker's threads (below normal it starved: ~12 docs/s, not ~200)
        set_priority(0x8000)
    log(a.worker, f'start: {len(months)} months, {sum(r for _, _, r in months):,} items')
    if a.worker == 'lm':
        worker_lm(months, a.hnz, a.min_free_gb)
    else:
        worker_embed(months, a.hnz, a.device, a.min_free_gb)
    pending = [m for m, _, _ in months if not (lm_done(m) if a.worker == 'lm' else emb_done(m))]
    log(a.worker, f'end: {len(months) - len(pending)}/{len(months)} months done' + (f'; pending {pending}' if pending else ''))


if __name__ == '__main__':
    main()
