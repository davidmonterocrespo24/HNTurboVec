"""E30a: PREFIX embeddings (what the reranking can re-embed without decompressing the whole record).

In the compact profile all the text goes in the LM tier: reranking a candidate means decompressing its
first tokens and embedding it again. This computes, for each document of a month, the EmbeddingGemma 2
embedding of its P-token LM prefix (full title and url + the first P tokens of the body, at most
T = P + 96 tokens in total), with the exact text of the cold tier (`hnz lmdump`).
Output: emb/g2p{P}_{month}.f16.npy + .ids.npy (same format as embed_month.py).
Usage: python experiments/embed_prefix.py data_lm/full_2024-01.jsonl 2024-01 128 [GPU|CPU]
"""
import html, json, os, re, sys, time
import numpy as np, openvino as ov
from tokenizers import Tokenizer
from transformers import AutoTokenizer

src, month, P = sys.argv[1], sys.argv[2], int(sys.argv[3])
DEV = sys.argv[4] if len(sys.argv) > 4 else 'GPU'
T = P + 96
BUCKETS = [(32, 128), (64, 64), (128, 32), (256, 16)]
lmtok = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')
gtok = Tokenizer.from_file('models/gemma2_tokenizer.json')
clean = lambda s: re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', s))).strip()


def plain(body):
    """Readable body text in the LM representation (an approximation of plain_prefix in Rust):
    "e" = paragraphs with \\n\\n and links <url>, <url>k>, <url|text>; otherwise compact HTML."""
    if '<a href' in body or '<p>' in body:
        return clean(body)
    body = re.sub(r'</?i>|<pre><code>|</code></pre>', '', body)
    body = re.sub(r'<([^>|]+)\|([^>]*)>', r'\2 (\1)', body)          # <url|text>
    body = re.sub(r'<([^>]+)>\d+>', r'\1', body)                        # truncated link <url>k>
    body = re.sub(r'<([^>\s]+)>', r'\1', body)                          # <url>
    return clean(html.escape(body, quote=False))


ids, prompts, cut = [], [], 0
with open(src, encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        s = d['s']
        title, url, body = (s.split('\n', 2) + ['', ''])[:3]
        head = lmtok(f'{title}\n{url}\n', add_special_tokens=False)['input_ids']
        b = lmtok(body, add_special_tokens=False)['input_ids']
        if len(b) > P or len(head) + len(b) > T:
            b = b[:max(0, min(P, T - len(head)))]
            body = lmtok.decode(b)
            cut += 1
        ids.append(d['id'])
        prompts.append(f'title: {clean(title) if title else "none"} | text: {plain(body)}')
enc = gtok.encode_batch(prompts)
lens = np.array([len(e.ids) for e in enc])
print(f'{month} P={P}: {len(ids):,} documents, {cut:,} truncated ({cut / len(ids):.1%}), {lens.mean():.0f} Gemma tokens on average', flush=True)

core = ov.Core()
out = np.zeros((len(ids), 768), np.float16)
t0, lo = time.time(), 0
for seq, bs in BUCKETS:
    sel = np.where((lens > lo) & (lens <= seq))[0] if seq != 256 else np.where(lens > lo)[0]
    lo = seq
    if len(sel) == 0:
        continue
    cfg = {'PERFORMANCE_HINT': 'THROUGHPUT'}
    if DEV == 'GPU':
        cfg['INFERENCE_PRECISION_HINT'] = 'f16'
    cm = core.compile_model(f'models/gemma2_text_b{bs}_s{seq}.xml', DEV, cfg)
    q = ov.AsyncInferQueue(cm, cm.get_property('OPTIMAL_NUMBER_OF_INFER_REQUESTS'))

    def done(req, rows):
        e = req.get_output_tensor(0).data[:len(rows)]
        out[rows] = (e / np.linalg.norm(e, axis=1, keepdims=True)).astype(np.float16)
    q.set_callback(done)
    for i in range(0, len(sel), bs):
        rows = sel[i:i + bs]
        a = np.zeros((bs, seq), np.int64)
        m = np.zeros((bs, seq), np.int64)
        for j, r in enumerate(rows):
            x = enc[r].ids[:seq]
            a[j, :len(x)] = x
            m[j, :len(x)] = 1
        if len(rows) < bs:
            m[len(rows):, 0] = 1
        q.start_async([a, m], rows)
    q.wait_all()
np.save(f'emb/g2p{P}_{month}.f16.npy', out)
np.save(f'emb/g2p{P}_{month}.ids.npy', np.array(ids, np.uint32))
print(f'{month} P={P}: {len(ids) / (time.time() - t0):.0f} docs/s ({(time.time() - t0) / 60:.1f} min)', flush=True)
