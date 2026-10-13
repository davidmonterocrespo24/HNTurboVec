"""Latencias reales del servidor hnz (modelos cargados una vez, como en uso diario).
Uso: python bench_serve.py   (con out/ construido y hnz compilado en hnz/target/release)"""
import json, subprocess, sys, time, urllib.parse, urllib.request, os

ROOT = os.path.dirname(os.path.abspath(__file__))
HNZ = os.path.join(ROOT, 'hnz', 'target', 'release', 'hnz.exe')
PORT = 8099


def start():
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


def q(params):
    url = f'http://127.0.0.1:{PORT}/search?' + urllib.parse.urlencode(params)
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(url, timeout=600).read())
    wall = (time.time() - t0) * 1000
    if 'error' in r:
        raise SystemExit(r['error'])
    tiers = {}
    for x in r['results']:
        tiers[x['tier']] = tiers.get(x['tier'], 0) + 1
    return r['ms'], wall, tiers, r['results']


def item(i):
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(f'http://127.0.0.1:{PORT}/item/{i}', timeout=600).read())
    return r.get('ms'), (time.time() - t0) * 1000


def show(label, ms, wall, tiers):
    print(f'{label:<58} total {ms["total"]:7.1f} ms (embedding {ms["embedding"]:5.1f}, búsqueda {ms["search"]:5.1f}, '
          f'descompresión {ms["fetch"]:7.1f}) | pared {wall:7.1f} ms | {tiers}', flush=True)


p, t_start = start()
print(f'servidor listo en {t_start:.1f} s (carga de segmentos + modelos)')
Q1 = 'how do I deal with burnout as a software engineer'
Q2 = 'is the iPad going to kill netbooks'
Q3 = 'rust async runtime'
q({'q': 'warmup', 'k': 3})
for label, params in [
    ('caliente (2024), híbrida, k=10', {'q': Q1, 'k': 10, 'from': '2024', 'to': '2024'}),
    ('caliente (2024), palabras, k=10', {'q': Q3, 'k': 10, 'from': '2024', 'to': '2024', 'mode': 'keyword'}),
    ('frío (2010), híbrida, k=10, fragmentos', {'q': Q2, 'k': 10, 'from': '2010', 'to': '2010'}),
    ('frío (2015), híbrida, k=10, fragmentos', {'q': Q1, 'k': 10, 'from': '2015', 'to': '2015'}),
    ('todo el archivo, híbrida, k=10, fragmentos', {'q': Q1, 'k': 10}),
    ('todo el archivo, híbrida, k=100, fragmentos', {'q': Q1, 'k': 100}),
    ('filtro tipo job + palabras (recorrido exacto)', {'q': Q3, 'k': 10, 'type': 'job', 'mode': 'keyword'}),
    ('filtro autor (by=pg), híbrida', {'q': 'startups', 'k': 10, 'by': 'pg'}),
    ('filtro tipo story 2015, híbrida', {'q': Q1, 'k': 10, 'type': 'story', 'from': '2015', 'to': '2015'}),
    ('sin texto: stories más recientes de 2024', {'q': '', 'k': 10, 'type': 'story', 'from': '2024'}),
    ('frío (2010), k=10, TEXTO COMPLETO (1ª vez)', {'q': Q2, 'k': 10, 'from': '2010', 'to': '2010', 'full': 1}),
    ('frío (2010), k=10, texto completo, repetida (caché L1)', {'q': Q2, 'k': 10, 'from': '2010', 'to': '2010', 'full': 1}),
]:
    ms, wall, tiers, res = q(params)
    show(label, ms, wall, tiers)
cold_ids = [x['id'] for x in res if x['tier'] == 'lm']
p.terminate(); p.wait()
p, t_start = start()
print(f'(servidor reiniciado en {t_start:.1f} s: la caché L1 se vacía, la L2 en disco persiste)')
ms, wall, tiers, _ = q({'q': Q2, 'k': 10, 'from': '2010', 'to': '2010', 'full': 1})
show('frío (2010), k=10, texto completo, tras reinicio (caché L2)', ms, wall, tiers)
# un ítem frío no cacheado, completo, bajo demanda (/item)
ms2, wall2, tiers2, res2 = q({'q': Q1, 'k': 20, 'from': '2015', 'to': '2015'})
fresh = [x for x in res2 if x['tier'] == 'lm' and x['partial']]
if fresh:
    m, w = item(fresh[-1]['id'])
    print(f'{"/item/<id> de un registro frío (completo, sin caché)":<58} {m:7.1f} ms en servidor | pared {w:7.1f} ms')
p.terminate(); p.wait()
