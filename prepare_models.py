"""Prepara en models/ TODO lo que necesita el sistema (se salta lo que ya existe):
  - EmbeddingGemma 2 (solo texto) en OpenVINO IR para los embeddings de documentos en la iGPU
    (buckets 32/64/128/256 tokens) y para las consultas (1 x 64 tokens, pesos int8 con NNCF)
  - gemma2_tokenizer.json (tokenizador de EmbeddingGemma 2)
  - lm-pythia-70m.hnzlm + .tokenizer.json (LM compresor del nivel frío)
Uso: python prepare_models.py            (requiere el entorno Python de requirements.txt)"""
import os, shutil, subprocess, sys
# el descargador hf_xet de huggingface_hub muere a mitad de descarga en este Windows (exit 127,
# E26): se usa la descarga HTTP de siempre (también reanudable). Se hereda en los subprocesos.
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

ROOT = os.path.dirname(os.path.abspath(__file__))
M = os.path.join(ROOT, 'models')
os.makedirs(M, exist_ok=True)
PY = sys.executable


def run(args, env=None):
    print('>', ' '.join(args), flush=True)
    subprocess.run(args, check=True, cwd=ROOT, env={**os.environ, **(env or {})})


tok = os.path.join(M, 'gemma2_tokenizer.json')
if not os.path.exists(tok):
    from huggingface_hub import hf_hub_download
    shutil.copy(hf_hub_download('google/embeddinggemma-2', 'tokenizer.json'), tok)
    print('copiado', tok)
for b, s in [(128, 32), (64, 64), (32, 128), (16, 256), (1, 64)]:
    if not os.path.exists(os.path.join(M, f'gemma2_text_b{b}_s{s}.xml')):
        run([PY, 'export_gemma2_ov.py'], env={'B': str(b), 'S': str(s)})
if not os.path.exists(os.path.join(M, 'gemma2_text_b1_s64_int8.xml')):
    run([PY, 'compress_embedder.py', os.path.join('models', 'gemma2_text_b1_s64.xml')])
if not os.path.exists(os.path.join(M, 'lm-pythia-70m.hnzlm')):
    run([PY, 'export_lm.py', 'EleutherAI/pythia-70m', os.path.join('models', 'lm-pythia-70m.hnzlm')])
print('modelos listos en', M)
