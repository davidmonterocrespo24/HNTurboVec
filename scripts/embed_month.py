"""Generates EmbeddingGemma 2 embeddings (768d) for one month of HN on the iGPU (OpenVINO), grouped by
length (buckets of 32/64/128/256 tokens). Output: emb/g2_<month>.f16.npy (N x 768, float16, same order
as the filtered parquet) + emb/g2_<month>.ids.npy (HN ids).
Usage: python scripts/embed_month.py data/2024/2024-01.parquet [GPU|CPU]"""
import sys, os, time, re, html
import numpy as np, pyarrow.parquet as pq, openvino as ov
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

PATH = sys.argv[1]; DEV = sys.argv[2] if len(sys.argv) > 2 else 'GPU'
month = os.path.basename(PATH).replace('.parquet', '')
BUCKETS = [(32, 128), (64, 64), (128, 32), (256, 16)]  # (seq, batch)
tok = Tokenizer.from_file(hf_hub_download('google/embeddinggemma-2', 'tokenizer.json'))
t = pq.read_table(PATH, columns=['id', 'title', 'text', 'deleted'])
ids_all, title, text, deleted = (t[c].to_pylist() for c in ['id', 'title', 'text', 'deleted'])
clean = lambda s: re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', s))).strip()
keep = [k for k in range(len(ids_all)) if not deleted[k] and (title[k] or text[k])]
prompts = []
for k in keep:  # EmbeddingGemma document prompt: "title: X | text: Y"
    prompts.append(f'title: {clean(title[k]) if title[k] else "none"} | text: {clean(text[k])}')
enc = tok.encode_batch(prompts)
lens = np.array([len(e.ids) for e in enc])
print(f'{month}: {len(keep):,} items with content; mean tokens {lens.mean():.0f}, >256: {np.mean(lens > 256):.1%}', flush=True)

core = ov.Core()
out = np.zeros((len(keep), 768), np.float16)
t0 = time.time()
lo = 0
for seq, bs in BUCKETS:
    sel = np.where((lens > lo) & (lens <= seq))[0] if seq != 256 else np.where(lens > lo)[0]
    lo = seq
    if len(sel) == 0: continue
    cfg = {'PERFORMANCE_HINT': 'THROUGHPUT'}
    if DEV == 'GPU': cfg['INFERENCE_PRECISION_HINT'] = 'f16'
    cm = core.compile_model(f'models/gemma2_text_b{bs}_s{seq}.xml', DEV, cfg)
    q = ov.AsyncInferQueue(cm, cm.get_property('OPTIMAL_NUMBER_OF_INFER_REQUESTS'))
    def done(req, rows):
        e = req.get_output_tensor(0).data[:len(rows)]
        out[rows] = (e / np.linalg.norm(e, axis=1, keepdims=True)).astype(np.float16)
    q.set_callback(done)
    tb = time.time()
    for i in range(0, len(sel), bs):
        rows = sel[i:i + bs]
        ids = np.zeros((bs, seq), np.int64); mask = np.zeros((bs, seq), np.int64)
        for j, r in enumerate(rows):
            x = enc[r].ids[:seq]; ids[j, :len(x)] = x; mask[j, :len(x)] = 1
        if len(rows) < bs: mask[len(rows):, 0] = 1  # padding rows must be valid
        q.start_async([ids, mask], rows)
    q.wait_all()
    print(f'  bucket {seq:>3}: {len(sel):>7,} docs  {len(sel)/(time.time()-tb):7.1f} docs/s', flush=True)
dt = time.time() - t0
os.makedirs('emb', exist_ok=True)
np.save(f'emb/g2_{month}.f16.npy', out); np.save(f'emb/g2_{month}.ids.npy', np.array([ids_all[k] for k in keep], np.uint32))
print(f'TOTAL {len(keep)/dt:.1f} docs/s ({dt/60:.1f} min) -> 49.3M in {49.3e6/(len(keep)/dt)/86400:.1f} days', flush=True)
