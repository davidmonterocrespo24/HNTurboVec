"""Latencias reales del servidor hnz (modelos cargados una vez, como en uso diario).

Cada caso se mide con el servidor recién arrancado (caché L1 vacía), sin caché L2 en disco y con N
consultas DISTINTAS (títulos de eval/*.jsonl): se da la mediana y el p90 (una sola consulta con la
caché caliente no es representativa, E26). Después, los escenarios de caché a propósito: el mismo
texto completo la 1ª vez, repetido (L1) y tras reiniciar el servidor (L2 en disco).
Uso: python experimentos/bench_serve.py [--n 30]   (con out/ construido y hnz compilado en hnz/target/release)
"""
import argparse, json, os, subprocess, time, urllib.parse, urllib.request
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # raíz del proyecto
HNZ = os.path.join(ROOT, 'hnz', 'target', 'release', 'hnz.exe' if os.name == 'nt' else 'hnz')
PORT = 8099
L2 = os.path.join(ROOT, 'out', '_cache_l2.bin')


def start(clear_l2=True):
    if clear_l2 and os.path.exists(L2):
        os.remove(L2)
    p = subprocess.Popen([HNZ, 'serve', '--root', 'out', '--port', str(PORT)], cwd=ROOT,
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    t0 = time.time()
    while True:
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{PORT}/', timeout=1).read()
            return p, time.time() - t0
        except Exception:
            if p.poll() is not None:
                raise SystemExit('el servidor terminó: ' + p.stderr.read().decode(errors='replace'))
            time.sleep(0.2)


def stop(p):
    p.terminate()
    p.wait()


def q(params):
    url = f'http://127.0.0.1:{PORT}/search?' + urllib.parse.urlencode(params)
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(url, timeout=600).read())
    wall = (time.time() - t0) * 1000
    if 'error' in r:
        raise SystemExit(r['error'])
    return r['ms'], wall, r['results']


def item(i):
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(f'http://127.0.0.1:{PORT}/item/{i}', timeout=600).read())
    return r.get('ms'), (time.time() - t0) * 1000


def titles(m):
    return [json.loads(l)['q'] for l in open(os.path.join(ROOT, 'eval', f'{m}.jsonl'), encoding='utf-8-sig') if l.strip()]


def row(label, ms, wall, lm_frac):
    tot = [x['total'] for x in ms]
    med = lambda k: np.median([x[k] for x in ms])
    print(f'{label:<50} total mediana {np.median(tot):7.1f} ms, p90 {np.percentile(tot, 90):7.1f} '
          f'(embedding {med("embedding"):5.1f}, búsqueda {med("search"):5.1f}, texto {med("fetch"):7.1f}) '
          f'| pared {np.median(wall):7.1f} ms | {lm_frac:.0%} del nivel LM', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=30)
    a = ap.parse_args()
    T = {m: titles(m) for m in ('2024-01', '2010-06', '2015-06')}
    words = ['rust', 'python', 'remote', 'engineer', 'startup', 'data', 'security', 'golang', 'design', 'growth']
    cases = [
        ('caliente (2024), híbrida, k=10', [dict(q=t, k=10, **{'from': '2024', 'to': '2024'}) for t in T['2024-01']]),
        ('caliente (2024), palabras, k=10', [dict(q=t, k=10, mode='keyword', **{'from': '2024', 'to': '2024'}) for t in T['2024-01']]),
        ('todo el archivo, híbrida, k=10 (consultas 2024)', [dict(q=t, k=10) for t in T['2024-01']]),
        ('todo el archivo, híbrida, k=10 (consultas 2010)', [dict(q=t, k=10) for t in T['2010-06']]),
        ('frío (2010), híbrida, k=10, fragmentos', [dict(q=t, k=10, **{'from': '2010', 'to': '2010'}) for t in T['2010-06']]),
        ('frío (2015), híbrida, k=10, fragmentos', [dict(q=t, k=10, **{'from': '2015', 'to': '2015'}) for t in T['2015-06']]),
        ('frío (2010), híbrida, k=10, texto completo', [dict(q=t, k=10, full=1, **{'from': '2010', 'to': '2010'}) for t in T['2010-06']]),
        ('todo el archivo, híbrida, k=100, fragmentos', [dict(q=t, k=100) for t in T['2015-06']]),
        ('filtro tipo job + palabras (recorrido exacto)', [dict(q=w, k=10, type='job', mode='keyword') for w in words]),
        ('filtro autor, híbrida', [dict(q=t, k=10, by=b) for t, b in zip(T['2015-06'], ['pg', 'tptacek', 'patio11', 'jacquesm', 'dang'] * 10)]),
        ('sin texto: lo más reciente con filtros', [dict(q='', k=10, type=ty, **{'from': y}) for ty in ('story', 'comment', 'job') for y in ('2010', '2015', '2024')]),
    ]
    p, t_start = start()
    print(f'servidor listo en {t_start:.1f} s (carga de segmentos + modelos); {a.n} consultas distintas por caso')
    stop(p)
    for label, params in cases:
        p, _ = start()
        try:
            q({'q': 'warmup', 'k': 3, 'mode': 'keyword'})
            ms, wall, lm, n = [], [], 0, 0
            for prm in params[:a.n]:
                m, w, res = q(prm)
                ms.append(m); wall.append(w)
                lm += sum(r['tier'] == 'lm' for r in res); n += len(res)
            row(label, ms, wall, lm / max(n, 1))
        finally:
            stop(p)
    # escenarios de caché, a propósito: las mismas consultas con texto completo
    prm = [dict(q=t, k=10, full=1, **{'from': '2010', 'to': '2010'}) for t in T['2010-06'][100:100 + min(a.n, 20)]]
    p, _ = start()
    try:
        first = [q(x) for x in prm]
        again = [q(x) for x in prm]
    finally:
        stop(p)
    p, t_restart = start(clear_l2=False)
    try:
        after = [q(x) for x in prm]
    finally:
        stop(p)
    row('frío (2010), texto completo, 1ª vez', [x[0] for x in first], [x[1] for x in first], 1)
    row('  la misma consulta repetida (caché L1)', [x[0] for x in again], [x[1] for x in again], 1)
    row(f'  tras reiniciar en {t_restart:.1f} s (caché L2 en disco)', [x[0] for x in after], [x[1] for x in after], 1)
    # /item/<id> de registros fríos no cacheados (completos, bajo demanda)
    p, _ = start()
    try:
        ids = []
        for t in T['2015-06'][200:200 + 10]:
            _, _, res = q(dict(q=t, k=5, **{'from': '2015', 'to': '2015'}))
            ids += [r['id'] for r in res if r['tier'] == 'lm' and r['partial']][:1]
    finally:
        stop(p)
    p, _ = start()   # caché vacía: esos registros solo se habían leído como fragmento
    try:
        w = [item(i)[1] for i in ids]
        print(f'{"/item/<id> frío (completo, sin caché)":<50} mediana {np.median(w):7.1f} ms, p90 {np.percentile(w, 90):7.1f} ms ({len(w)} ítems)')
    finally:
        stop(p)


if __name__ == '__main__':
    main()
