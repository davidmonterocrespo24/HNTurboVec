"""Exporta SOLO la parte de texto de google/embeddinggemma-2 (transformer + pooling + normalize)
a OpenVINO IR con formas estaticas, para ejecutarla en la iGPU Arc / NPU / CPU."""
import os, sys, time
import numpy as np, torch, openvino as ov
from sentence_transformers import SentenceTransformer

BATCH, SEQ = int(os.environ.get('B', 32)), int(os.environ.get('S', 128))
OUT = f'models/gemma2_text_b{BATCH}_s{SEQ}.xml'
os.makedirs('models', exist_ok=True)
st = SentenceTransformer('google/embeddinggemma-2', device='cpu').float()
st.eval()
print(st)

class TextEmb(torch.nn.Module):
    def __init__(self, st):
        super().__init__(); self.st = st
    def forward(self, input_ids, attention_mask):
        feats = {'input_ids': input_ids, 'attention_mask': attention_mask}
        for mod in self.st:          # Transformer -> Pooling -> Normalize
            feats = mod(feats)
        return feats['sentence_embedding']

w = TextEmb(st).eval()
ids = torch.randint(10, 1000, (BATCH, SEQ)); mask = torch.ones(BATCH, SEQ, dtype=torch.long)
with torch.no_grad():
    ref = w(ids, mask)
print('salida torch', tuple(ref.shape))
t0 = time.time()
m = ov.convert_model(w, example_input=(ids, mask), input=[([BATCH, SEQ], ov.Type.i64), ([BATCH, SEQ], ov.Type.i64)])
ov.save_model(m, OUT, compress_to_fp16=True)
print(f'guardado {OUT} en {time.time()-t0:.0f}s')
core = ov.Core()
cm = core.compile_model(OUT, 'CPU')
o = cm([ids.numpy(), mask.numpy()])[0]
print('coseno OV-CPU vs torch:', float(np.mean(np.sum(o * ref.numpy(), 1) / np.linalg.norm(o, axis=1) / np.linalg.norm(ref.numpy(), axis=1))))
