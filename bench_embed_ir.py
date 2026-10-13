"""Velocidad de EmbeddingGemma 2 (solo texto, IR OpenVINO exportado por export_gemma2_ov.py)
en GPU (Arc iGPU) / NPU / CPU, + PyTorch float32 de referencia. Textos reales de HN."""
import sys, time, re, html, os
import numpy as np, pyarrow.parquet as pq, openvino as ov
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

IR = sys.argv[1] if len(sys.argv) > 1 else 'models/gemma2_text_b32_s128.xml'
DEVICES = sys.argv[2].split(',') if len(sys.argv) > 2 else ['GPU', 'CPU', 'NPU', 'TORCH']
BATCH, SEQ = 32, 128
NDOC = int(os.environ.get('NDOC', 1024))
tok = Tokenizer.from_file(hf_hub_download('google/embeddinggemma-2', 'tokenizer.json'))
t = pq.read_table('data/2024/2024-01.parquet', columns=['text', 'title'])
clean = lambda s: html.unescape(re.sub(r'<[^>]+>', ' ', s)).strip()
docs = [clean((a + '\n' + b).strip()) for a, b in zip(t['title'].to_pylist()[5000:5000 + NDOC], t['text'].to_pylist()[5000:5000 + NDOC])]
enc = tok.encode_batch(['title: none | text: ' + d for d in docs])
ids = np.zeros((NDOC, SEQ), np.int64); mask = np.zeros((NDOC, SEQ), np.int64)
for i, e in enumerate(enc):
    x = e.ids[:SEQ]; ids[i, :len(x)] = x; mask[i, :len(x)] = 1
print(f'{NDOC} docs, seq={SEQ}, tokens reales medios={mask.sum(1).mean():.0f}, truncados={np.mean([len(e.ids) > SEQ for e in enc]):.1%}')
core = ov.Core()
ref = None
for dev in DEVICES:
    try:
        if dev == 'TORCH':
            import torch
            from sentence_transformers import SentenceTransformer
            torch.set_num_threads(os.cpu_count())
            st = SentenceTransformer('google/embeddinggemma-2', device='cpu').float(); st.max_seq_length = SEQ
            run = lambda a, b: st.encode([{'text': d} for d in docs[i0:i0 + BATCH]], prompt_name='document', batch_size=BATCH)
            def bench():
                global i0
                out = []
                for i0 in range(0, NDOC, BATCH): out.append(run(None, None))
                return np.concatenate(out)
            bench_fn = bench
        else:
            cfg = {'PERFORMANCE_HINT': 'THROUGHPUT'} if dev in ('GPU', 'CPU') else {}
            if dev == 'GPU': cfg['INFERENCE_PRECISION_HINT'] = 'f16'
            cm = core.compile_model(IR, dev, cfg)
            nreq = cm.get_property('OPTIMAL_NUMBER_OF_INFER_REQUESTS')
            q = ov.AsyncInferQueue(cm, nreq)
            res = {}
            q.set_callback(lambda r, i: res.__setitem__(i, r.get_output_tensor(0).data.copy()))
            def bench_fn():
                for i in range(0, NDOC, BATCH):
                    q.start_async([ids[i:i + BATCH], mask[i:i + BATCH]], i)
                q.wait_all()
                return np.concatenate([res[i] for i in range(0, NDOC, BATCH)])
            bench_fn()  # calentar / compilar kernels
        t0 = time.time(); E = bench_fn(); dt = time.time() - t0
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        if ref is None: ref = E
        print(f'  {dev:<6} {NDOC/dt:8.1f} docs/s  coseno vs {DEVICES[0]}: {float(np.mean(np.sum(E*ref,1))):.4f}  '
              f'-> 49,3M en {49.3e6/(NDOC/dt)/3600:7.1f} h ({49.3e6/(NDOC/dt)/86400:.1f} dias)', flush=True)
    except Exception as ex:
        print(f'  {dev:<6} FALLO: {type(ex).__name__}: {str(ex)[:300]}', flush=True)
