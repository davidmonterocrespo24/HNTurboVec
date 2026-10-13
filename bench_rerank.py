"""Reranker cross-encoder (equivalente open source a Jev/System One "Score") sobre el top-30 de hnz.

Usa el servidor de hnz (búsqueda híbrida real sobre todo out/, como `hnz eval`) y reordena sus 30
primeros resultados con cross-encoders pequeños de fastembed (ONNX, CPU). Consultas: eval/2024-01.jsonl
(título de una story → relevantes = comentarios de su hilo; la propia story no cuenta). Mide P@10, MRR
y ms/consulta del reranker. Memoria baja: solo los textos de 30 candidatos por consulta.
Uso: python bench_rerank.py [--n 300] [--cand 30]
"""
import argparse, json, os, subprocess, sys, time, urllib.parse, urllib.request
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
HNZ = os.path.join(ROOT, 'hnz', 'target', 'release', 'hnz.exe' if os.name == 'nt' else 'hnz')
PORT = 8097
# pequeños (80-150 MB) y uno grande de referencia (BGE-M3 int8, 570 MB)
MODELS = ['Xenova/ms-marco-MiniLM-L-6-v2', 'Xenova/ms-marco-MiniLM-L-12-v2', 'jinaai/jina-reranker-v1-tiny-en',
          'jinaai/jina-reranker-v1-turbo-en', 'BAAI/bge-reranker-v2-m3-int8']


def start():
    p = subprocess.Popen([HNZ, 'serve', '--root', 'out', '--port', str(PORT)], cwd=ROOT,
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    while True:
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{PORT}/', timeout=1).read()
            return p
        except Exception:
            if p.poll() is not None:
                raise SystemExit('el servidor terminó: ' + p.stderr.read().decode(errors='replace'))
            time.sleep(0.2)


def search(q, k):
    url = f'http://127.0.0.1:{PORT}/search?' + urllib.parse.urlencode({'q': q, 'k': k, 'full': 1})
    r = json.loads(urllib.request.urlopen(url, timeout=600).read())
    if 'error' in r:
        raise SystemExit(r['error'])
    return r['results']


def doc_text(r):
    # lo que vería el reranker: título (o el de su hilo) + texto plano
    head = r['title'] or ('re: ' + (r.get('story_title') or ''))
    return (head + '\n' + r['text'])[:1500]


def metrics(qs, lists):
    p10 = mrr = 0.0
    for q, lst in zip(qs, lists):
        rel = [r.get('story_id') == q['story'] for r in lst[:10]]
        p10 += sum(rel) / 10
        mrr += next((1 / (i + 1) for i, x in enumerate(rel) if x), 0)
    return p10 / len(qs), mrr / len(qs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--cand', type=int, default=30)
    ap.add_argument('--models', default=','.join(MODELS))
    a = ap.parse_args()
    qs = [json.loads(l) for l in open(os.path.join(ROOT, 'eval', '2024-01.jsonl'), encoding='utf-8-sig') if l.strip()][:a.n]
    srv = start()
    try:
        cands = []
        for q in qs:
            res = [r for r in search(q['q'], a.cand + 1) if r['id'] != q['story']][:a.cand]
            cands.append(res)
    finally:
        srv.terminate()
    p, m = metrics(qs, cands)
    print(f'{len(qs)} consultas, top-{a.cand} de hnz (híbrida, todo out/)')
    print(f'  {"hnz híbrida (sin reranker)":<44} P@10 {p:.3f}  MRR {m:.3f}', flush=True)
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    threads = max(1, (os.cpu_count() or 2) // 2)
    for name in a.models.split(','):
        try:
            ce = TextCrossEncoder(name, threads=threads)
            list(ce.rerank('warm up', ['a', 'b']))
            lists, fused, t = [], [], 0.0
            for q, c in zip(qs, cands):
                t0 = time.perf_counter()
                sc = np.array(list(ce.rerank(q['q'], [doc_text(r) for r in c])))
                t += time.perf_counter() - t0
                order = np.argsort(-sc)
                lists.append([c[i] for i in order])
                # fusión por rangos recíprocos (reranker + orden de hnz)
                rr = np.empty(len(c)); rr[order] = np.arange(len(c))
                f = 1 / (60 + rr) + 1 / (60 + np.arange(len(c)))
                fused.append([c[i] for i in np.argsort(-f)])
            p, m = metrics(qs, lists)
            pf, mf = metrics(qs, fused)
            ms = t / len(qs) * 1000
            print(f'  {name.split("/")[1]:<44} P@10 {p:.3f}  MRR {m:.3f}  | fusión RRF P@10 {pf:.3f}  MRR {mf:.3f} | {ms:.0f} ms/cons. ({threads} hilos)', flush=True)
            del ce
        except Exception as ex:
            print(f'  {name}: FALLO {type(ex).__name__}: {str(ex)[:150]}', flush=True)


if __name__ == '__main__':
    main()
