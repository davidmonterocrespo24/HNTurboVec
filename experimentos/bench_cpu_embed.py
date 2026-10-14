"""E30b: ¿a qué velocidad embebe documentos la CPU sola (sin iGPU)? Decide cuántos candidatos se pueden
re-embeber por consulta en un servidor sin GPU.

EmbeddingGemma 2 (IR de OpenVINO) en CPU, modo THROUGHPUT con varias peticiones en paralelo, sobre
documentos reales de HN (prompt "title: … | text: …", hasta 128 tokens), con 16 hilos y con 8 (lo que
quedaría libre mientras el LM descomprime). Variantes: fp16 estático 32x128 y su versión int8 (NNCF).
Uso: python experimentos/bench_cpu_embed.py
"""
import html, json, os, re, time
import numpy as np, openvino as ov, pyarrow.parquet as pq
from tokenizers import Tokenizer

core = ov.Core()
tok = Tokenizer.from_file('models/gemma2_tokenizer.json')
SRC = 'models/gemma2_text_b32_s128.xml'
INT8 = 'models/gemma2_text_b32_s128_int8.xml'
if not os.path.exists(INT8):
    import nncf
    m = nncf.compress_weights(core.read_model(SRC), mode=nncf.CompressWeightsMode.INT8_ASYM)
    ov.save_model(m, INT8, compress_to_fp16=False)
    print('IR int8 de documentos creado:', INT8, flush=True)

t = pq.read_table('data/2024/2024-01.parquet', columns=['title', 'text', 'deleted'])
title, text, deleted = (t[c].to_pylist() for c in ['title', 'text', 'deleted'])
clean = lambda s: re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', s or ''))).strip()
docs = [f'title: {clean(title[k]) if title[k] else "none"} | text: {clean(text[k])}'
        for k in range(0, len(title), 97) if not deleted[k] and (title[k] or text[k])][:1024]
enc = tok.encode_batch(docs)
B, S = 32, 128
ids = np.zeros((len(docs), S), np.int64)
att = np.zeros((len(docs), S), np.int64)
for i, e in enumerate(enc):
    x = e.ids[:S]
    ids[i, :len(x)] = x
    att[i, :len(x)] = 1
print(f'{len(docs)} documentos, {att.sum(1).mean():.0f} tokens de media (máx. {S})', flush=True)


def bench(path, threads, nreq):
    cm = core.compile_model(path, 'CPU', {'PERFORMANCE_HINT': 'THROUGHPUT', 'INFERENCE_NUM_THREADS': threads,
                                          'NUM_STREAMS': nreq})
    q = ov.AsyncInferQueue(cm, nreq)
    batches = [(ids[i:i + B], att[i:i + B]) for i in range(0, len(docs) - B + 1, B)]
    for a, m in batches[:nreq]:                      # calentamiento
        q.start_async([a, m])
    q.wait_all()
    t0 = time.time()
    for a, m in batches:
        q.start_async([a, m])
    q.wait_all()
    return len(batches) * B / (time.time() - t0)


for name, path in (('fp16 32x128', SRC), ('int8 32x128', INT8)):
    for threads, nreq in ((16, 4), (16, 2), (8, 2), (8, 1)):
        print(f'  {name:<12} {threads:>2} hilos, {nreq} flujos: {bench(path, threads, nreq):6.1f} docs/s', flush=True)
