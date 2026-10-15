"""Exports EmbeddingGemma 2's text model (transformer + pooling + normalize) to OpenVINO IR with
DYNAMIC batch and sequence length, and compresses its weights to int8 (NNCF): one IR for the
queries (1 x up to 64 tokens) and for the document prompts the reranker re-embeds (P4: batches of
up to 256 tokens), instead of one static IR per shape. Compares it with the static IRs.

Usage: python scripts/export_gemma2_dyn.py   ->  models/gemma2_text_dyn.xml (fp16) and
                                                 models/gemma2_text_dyn_int8.xml
"""
import time

import numpy as np
import openvino as ov
import torch
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer

OUT, OUT8 = 'models/gemma2_text_dyn.xml', 'models/gemma2_text_dyn_int8.xml'


class TextEmb(torch.nn.Module):
    def __init__(self, st):
        super().__init__()
        self.st = st

    def forward(self, input_ids, attention_mask):
        feats = {'input_ids': input_ids, 'attention_mask': attention_mask}
        for mod in self.st:          # Transformer -> Pooling -> Normalize
            feats = mod(feats)
        return feats['sentence_embedding']


def main():
    st = SentenceTransformer('google/embeddinggemma-2', device='cpu').float().eval()
    w = TextEmb(st).eval()
    ids = torch.randint(10, 1000, (4, 96))
    mask = torch.ones(4, 96, dtype=torch.long)
    t0 = time.time()
    m = ov.convert_model(w, example_input=(ids, mask), input=[([-1, -1], ov.Type.i64), ([-1, -1], ov.Type.i64)])
    ov.save_model(m, OUT, compress_to_fp16=True)
    print(f'{OUT} in {time.time() - t0:.0f} s', flush=True)
    import nncf
    m8 = nncf.compress_weights(ov.Core().read_model(OUT), mode=nncf.CompressWeightsMode.INT8_ASYM)
    ov.save_model(m8, OUT8, compress_to_fp16=False)
    print(f'{OUT8} in {time.time() - t0:.0f} s', flush=True)

    # the same vectors as the static IRs: queries (b1 s64 int8) and documents (b32 s128 fp16)
    tok = Tokenizer.from_file('models/gemma2_tokenizer.json')
    core = ov.Core()
    dyn = core.compile_model(OUT8, 'CPU')
    q64 = core.compile_model('models/gemma2_text_b1_s64_int8.xml', 'CPU')
    texts = ['task: search result | query: how do I deal with burnout', 'task: search result | query: rust async runtime',
             'title: Show HN: a thing | text: Hello world, see the docs (https://a.b/c) and more text here.']
    for t in texts:
        x = tok.encode(t).ids[:64]
        a = np.array([x], np.int64)
        e1 = dyn([a, np.ones_like(a)])[0][0]
        p = np.zeros((1, 64), np.int64)
        mk = np.zeros((1, 64), np.int64)
        p[0, :len(x)] = x
        mk[0, :len(x)] = 1
        e2 = q64([p, mk])[0][0]
        print(f'cosine dynamic vs static b1_s64 int8: {float(e1 @ e2 / np.linalg.norm(e1) / np.linalg.norm(e2)):.5f} ({len(x)} tokens)')


if __name__ == '__main__':
    main()
