"""Compresses the weights of the EmbeddingGemma 2 query IR to int8 (NNCF, Intel), so the system
takes less space and embeds faster on CPU. Compares cosine and latency against the fp16 IR.
Usage: python scripts/compress_embedder.py models/gemma2_text_b1_s64.xml"""
import sys, time, json
import numpy as np, openvino as ov, nncf
from tokenizers import Tokenizer

src = sys.argv[1] if len(sys.argv) > 1 else 'models/gemma2_text_b1_s64.xml'
dst = src.replace('.xml', '_int8.xml')
core = ov.Core()
m = core.read_model(src)
t0 = time.time()
mc = nncf.compress_weights(m, mode=nncf.CompressWeightsMode.INT8_ASYM)
ov.save_model(mc, dst, compress_to_fp16=False)
print(f'compressed in {time.time()-t0:.0f}s -> {dst}')

tok = Tokenizer.from_file('models/gemma2_tokenizer.json')
import os
if os.path.exists('eval/2024-01.jsonl'):
    qs = [json.loads(l)['q'] for l in open('eval/2024-01.jsonl', encoding='utf-8')][:100]
else:
    qs = ['how do I deal with burnout', 'rust async runtime', 'best text editor for programmers',
          'is the iPad going to kill netbooks', 'startup funding advice', 'postgres vs mysql'] * 5
def run(path):
    cm = core.compile_model(path, 'CPU', {'PERFORMANCE_HINT': 'LATENCY'})
    req = cm.create_infer_request()
    out, times = [], []
    for q in qs:
        ids = tok.encode('task: search result | query: ' + q).ids[:64]
        a = np.zeros((1, 64), np.int64); b = np.zeros((1, 64), np.int64)
        a[0, :len(ids)] = ids; b[0, :len(ids)] = 1
        t = time.time(); r = req.infer([a, b]); times.append((time.time() - t) * 1000)
        v = r[cm.output(0)][0]; out.append(v / np.linalg.norm(v))
    return np.array(out), np.median(times[5:])
A, ta = run(src)
B, tb = run(dst)
import os
sz = lambda p: (os.path.getsize(p) + os.path.getsize(p.replace('.xml', '.bin'))) / 1e6
print(f'fp16: {sz(src):.0f} MB, {ta:.1f} ms/query | int8: {sz(dst):.0f} MB, {tb:.1f} ms/query | '
      f'mean cosine {float(np.mean(np.sum(A*B, 1))):.5f}, min {float(np.min(np.sum(A*B, 1))):.5f}')
