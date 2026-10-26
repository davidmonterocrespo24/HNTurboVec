"""After P5: the build, the compact block of all of HN, its checks and the full-scale measurements,
as resumable stages (one process; it runs at logon from the Startup folder, see p6_start.ps1).

Stages, in order (each needs the ones in its `needs`):
  cli          the CLI from the current code                -> work/bin/hnz-p6.exe
  pack         hnz pack --stream of all of HN (pack-all.json) -> work/p6/hn-2026-08.hnz
  verify-full  every part's hash, the padding, every canary
  wheel        the Python wheel (release, abi3)              -> work/wheels; also the dev package's _hnz.pyd
  rust-tests   cargo test --release --lib
  py-tests     pytest hnz/py/tests (the sample block)
  verify-deep  the block against its Parquet sources (0.5 % of the records decoded with the LM)
  eval         P@10 / MRR on the eval queries, reranked, on the full block
  eval-month   the same, each query restricted to its month (comparable with E33e)
A finished stage leaves work/p6/state/<stage>.json and is skipped from then on; a stage cut by a
shutdown runs again from its start on the next run; a failed one is logged and retried on the next
start (the stages that need it are skipped meanwhile).

Usage (project root):  python scripts/p6_run.py [--only STAGE] | status
Logs: logs/p6.log (one line per stage) and logs/p6/<stage>.log (the tools' output).
"""
import argparse
import ctypes
import glob
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HNZ = os.path.join(ROOT, 'hnz')
CLI = os.path.join(ROOT, 'work', 'bin', 'hnz-p6.exe')
BLOCK = os.path.join(ROOT, 'work', 'p6', 'hn-2026-08.hnz')
STATE = os.path.join(ROOT, 'work', 'p6', 'state')
LOGS = os.path.join(ROOT, 'logs')
PY = os.path.join(ROOT, '.venv', 'Scripts', 'python.exe')
TOOLCHAIN = [os.path.expandvars(r'%USERPROFILE%\.cargo\bin'),
             os.path.expandvars(r'%LOCALAPPDATA%\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin')]


def log(msg):
    os.makedirs(LOGS, exist_ok=True)
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
    print(line, flush=True)
    with open(os.path.join(LOGS, 'p6.log'), 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def run(stage, cmd, cwd=ROOT):
    """Runs a tool, its output to logs/p6/<stage>.log; returns its exit code."""
    os.makedirs(os.path.join(LOGS, 'p6'), exist_ok=True)
    env = dict(os.environ, PATH=os.pathsep.join(TOOLCHAIN + [os.environ.get('PATH', '')]))
    with open(os.path.join(LOGS, 'p6', f'{stage}.log'), 'a', encoding='utf-8') as f:
        f.write(f'\n{time.strftime("%Y-%m-%d %H:%M:%S")} $ {" ".join(cmd)}\n')
        f.flush()
        return subprocess.call(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)


def st_cli():
    if run('cli', ['cargo', 'build', '--release', '--bin', 'hnz'], cwd=HNZ):
        return False
    shutil.copy2(os.path.join(HNZ, 'target', 'release', 'hnz.exe'), CLI)
    return True


def st_pack():
    for d in (BLOCK, BLOCK + '.tmp'):
        if os.path.exists(d):
            shutil.rmtree(d)
    os.makedirs(os.path.dirname(BLOCK), exist_ok=True)
    return run('pack', [CLI, 'pack', '--stream', '--spec', 'work/p5/pack-all.json', '--out', BLOCK, '--tmp', BLOCK + '.tmp']) == 0


def st_verify_full():
    return run('verify-full', [CLI, 'verify-block', BLOCK, '--level', 'full']) == 0


def st_wheel():
    if run('wheel', [os.path.join(ROOT, '.venv', 'Scripts', 'maturin.exe'), 'build', '--release', '--out',
                     os.path.join(ROOT, 'work', 'wheels')], cwd=os.path.join(HNZ, 'py')):
        return False
    # the dev venv imports the package from hnz/py/python (editable): give it the same extension
    shutil.copy2(os.path.join(HNZ, 'target', 'release', '_hnz.dll'), os.path.join(HNZ, 'py', 'python', 'hnz', '_hnz.pyd'))
    return True


def st_rust_tests():
    return run('rust-tests', ['cargo', 'test', '--release', '--lib'], cwd=HNZ) == 0


def st_py_tests():
    return run('py-tests', [PY, '-m', 'pytest', 'hnz/py/tests', '-q', '-p', 'no:cacheprovider']) == 0


def st_verify_deep():
    return run('verify-deep', [CLI, 'verify-block', BLOCK, '--level', 'deep', '--decode', '0.005']) == 0


def st_eval():
    ok = True
    for f in sorted(glob.glob(os.path.join(ROOT, 'eval', '*.jsonl'))):
        ok &= run('eval', [CLI, 'eval', '--root', BLOCK, '--file', os.path.relpath(f, ROOT), '--rerank', '--emb', 'ov', '--quick']) == 0
    return ok


def st_eval_month():
    """The same eval, each query restricted to its file's month: comparable with the month blocks (E33e)."""
    if run('eval-month', ['cargo', 'build', '--release', '--bin', 'hnz'], cwd=HNZ):
        return False
    shutil.copy2(os.path.join(HNZ, 'target', 'release', 'hnz.exe'), CLI)
    ok = True
    for f in sorted(glob.glob(os.path.join(ROOT, 'eval', '*.jsonl'))):
        ok &= run('eval-month', [CLI, 'eval', '--root', BLOCK, '--file', os.path.relpath(f, ROOT), '--rerank', '--emb', 'ov',
                                 '--quick', '--within-month']) == 0
    return ok


STAGES = [  # (name, function, needs)
    ('cli', st_cli, []),
    ('pack', st_pack, ['cli']),
    ('verify-full', st_verify_full, ['pack']),
    ('wheel', st_wheel, []),
    ('rust-tests', st_rust_tests, []),
    ('py-tests', st_py_tests, ['wheel']),
    ('verify-deep', st_verify_deep, ['verify-full']),
    ('eval', st_eval, ['verify-full']),
    ('eval-month', st_eval_month, ['verify-full']),
]


def done(name):
    return os.path.exists(os.path.join(STATE, f'{name}.json'))


def keep_awake():
    if os.name == 'nt':
        ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint32(0x80000001))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cmd', nargs='?', default='run', choices=['run', 'status'])
    ap.add_argument('--only', help='run just this stage (if its needs are done)')
    a = ap.parse_args()
    if a.cmd == 'status':
        for name, _, _ in STAGES:
            p = os.path.join(STATE, f'{name}.json')
            print(f'{name:>12}: ' + (json.load(open(p))['finished'] if os.path.exists(p) else 'pending'))
        return
    keep_awake()
    os.makedirs(STATE, exist_ok=True)
    log('start')
    failed = set()
    for name, fn, needs in STAGES:
        if done(name) or (a.only and name != a.only):
            continue
        if any(not done(n) for n in needs):
            log(f'{name}: skipped (needs {", ".join(n for n in needs if not done(n))})')
            failed.add(name)
            continue
        t0 = time.time()
        log(f'{name}: started')
        try:
            ok = fn()
        except Exception as e:   # a stage's own error must not stop the others
            log(f'{name}: error {type(e).__name__}: {e}')
            ok = False
        dt = time.time() - t0
        if ok:
            with open(os.path.join(STATE, f'{name}.json'), 'w') as f:
                json.dump({'finished': time.strftime('%Y-%m-%d %H:%M:%S'), 'seconds': round(dt)}, f)
            log(f'{name}: done in {dt / 60:.1f} min')
        else:
            failed.add(name)
            log(f'{name}: FAILED after {dt / 60:.1f} min; see logs/p6/{name}.log')
    log('end: ' + ('all stages done' if all(done(n) for n, _, _ in STAGES) else f'pending {sorted(n for n, _, _ in STAGES if not done(n))}'))


if __name__ == '__main__':
    main()
