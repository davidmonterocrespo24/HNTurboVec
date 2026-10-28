"""Prepares in models/ EVERYTHING the system needs (whatever already exists is skipped):
  - EmbeddingGemma 2 (text only) as OpenVINO IR, for document embeddings on the iGPU
    (buckets of 32/64/128/256 tokens) and for queries (1 x 64 tokens, int8 weights with NNCF)
  - gemma2_tokenizer.json (the EmbeddingGemma 2 tokenizer)
  - lm-pythia-70m.hnzlm + .tokenizer.json (the compressor LM of the cold tier)
Usage: python scripts/prepare_models.py            (needs the Python environment from requirements.txt)"""
import os, shutil, subprocess, sys
# huggingface_hub's hf_xet downloader dies mid-download on this Windows machine (exit 127,
# E26): use the classic HTTP download instead (also resumable). Subprocesses inherit this.
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # project root
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
    print('copied', tok)
for b, s in [(128, 32), (64, 64), (32, 128), (16, 256), (1, 64)]:
    if not os.path.exists(os.path.join(M, f'gemma2_text_b{b}_s{s}.xml')):
        run([PY, os.path.join('scripts', 'export_gemma2_ov.py')], env={'B': str(b), 'S': str(s)})
if not os.path.exists(os.path.join(M, 'gemma2_text_b1_s64_int8.xml')):
    run([PY, os.path.join('scripts', 'compress_embedder.py'), os.path.join('models', 'gemma2_text_b1_s64.xml')])
if not os.path.exists(os.path.join(M, 'lm-pythia-70m.hnzlm')):
    run([PY, os.path.join('scripts', 'export_lm.py'), 'EleutherAI/pythia-70m', os.path.join('models', 'lm-pythia-70m.hnzlm')])
print('models ready in', M)
