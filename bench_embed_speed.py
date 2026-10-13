"""Velocidad de EmbeddingGemma 2 (ONNX) en CPU (onnxruntime) y OpenVINO CPU/GPU(Arc iGPU)/NPU.
Valida que el embedding coincide con sentence-transformers (coseno ~1)."""
import sys, time, re, html, os
import numpy as np, pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

REPO = 'onnx-community/embeddinggemma-2-ONNX'
VARIANT = sys.argv[1] if len(sys.argv) > 1 else 'model_fp16'   # model | model_fp16 | model_quantized
DEVICES = sys.argv[2].split(',') if len(sys.argv) > 2 else ['ORT-CPU', 'OV-CPU', 'OV-GPU', 'OV-NPU']
NDOC, SEQ, BATCH = 512, 128, 32

onnx_path = hf_hub_download(REPO, f'onnx/{VARIANT}.onnx')
hf_hub_download(REPO, f'onnx/{VARIANT}.onnx_data')
tok = Tokenizer.from_file(hf_hub_download(REPO, 'tokenizer.json'))

t = pq.read_table('data/2024/2024-01.parquet', columns=['text', 'title'])
clean = lambda s: html.unescape(re.sub(r'<[^>]+>', ' ', s)).strip()
docs = [clean((a + '\n' + b).strip()) for a, b in zip(t['title'].to_pylist()[5000:5000 + NDOC], t['text'].to_pylist()[5000:5000 + NDOC])]
enc = tok.encode_batch(['title: none | text: ' + d for d in docs])
ids = np.zeros((NDOC, SEQ), np.int64); mask = np.zeros((NDOC, SEQ), np.int64)
for i, e in enumerate(enc):
    x = e.ids[:SEQ]; ids[i, :len(x)] = x; mask[i, :len(x)] = 1
print(f'{VARIANT}: {NDOC} docs, seq={SEQ}, tokens medios reales={mask.sum(1).mean():.0f}')

def feeds(sl):
    return {'input_ids': ids[sl], 'attention_mask': mask[sl],
            'image_features': np.zeros((0, 512), np.float32), 'video_features': np.zeros((0, 512), np.float32),
            'audio_features': np.zeros((0, 512), np.float32)}

ref = None
results = {}
for dev in DEVICES:
    try:
        if dev == 'ORT-CPU':
            import onnxruntime as ort
            so = ort.SessionOptions(); so.intra_op_num_threads = os.cpu_count()
            s = ort.InferenceSession(onnx_path, so, providers=['CPUExecutionProvider'])
            run = lambda sl: s.run(['sentence_embedding'], feeds(sl))[0]
        else:
            import openvino as ov
            core = ov.Core()
            m = core.read_model(onnx_path)
            # formas estaticas (GPU/NPU las prefieren)
            m.reshape({'input_ids': [BATCH, SEQ], 'attention_mask': [BATCH, SEQ], 'image_features': [0, 512],
                       'video_features': [0, 512], 'audio_features': [0, 512]})
            cfg = {'PERFORMANCE_HINT': 'THROUGHPUT'} if dev != 'OV-NPU' else {}
            cm = core.compile_model(m, dev.split('-')[1], cfg)
            req = cm.create_infer_request()
            out = cm.output('sentence_embedding')
            def run(sl, req=req, out=out):
                r = req.infer(feeds(sl)); return r[out]
        run(slice(0, BATCH))  # calentar
        t0 = time.time(); E = np.concatenate([run(slice(i, i + BATCH)) for i in range(0, NDOC, BATCH)])
        dt = time.time() - t0
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        if ref is None: ref = E
        cos = float(np.mean(np.sum(E * ref, 1)))
        results[dev] = NDOC / dt
        print(f'  {dev:<8} {NDOC/dt:7.1f} docs/s   coseno vs {DEVICES[0]}: {cos:.4f}   -> 49,3M en {49.3e6/(NDOC/dt)/3600:6.1f} h', flush=True)
    except Exception as ex:
        print(f'  {dev:<8} FALLO: {type(ex).__name__}: {str(ex)[:200]}', flush=True)

# validacion contra sentence-transformers (primeros 32 docs)
if '--check' in sys.argv:
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer('google/embeddinggemma-2', device='cpu'); st.max_seq_length = SEQ
    S = st.encode([{'text': d} for d in docs[:32]], prompt_name='document', normalize_embeddings=True)
    print(f'  coseno ONNX({DEVICES[0]}) vs sentence-transformers: {float(np.mean(np.sum(ref[:32]*S,1))):.4f}')
