"""Embeds the document prompts written by `hnz prompts` (prompt version 2, defined in hnz/src/prompt.rs)
with EmbeddingGemma 2 on OpenVINO, with the length buckets and IRs of E30a (experimentos/embed_prefix.py):
    <prompts>/<month>.ids.npy, .len.npy, .tok.npy  ->  <out>.f16.npy (N x 768, L2-normalized) + <out>.ids.npy
The token ids come from Rust, so the vectors match the prompts the engine rebuilds at query time.

Usage: python scripts/embed_prompts.py work/p3/prompts 2015-06 [--device GPU|CPU] [--limit N] [--out emb/g2v2_2015-06]
       python scripts/embed_prompts.py work/p3/prompts 2015-06 --compare emb/g2p128_2015-06   # cosine vs other vectors
"""
import argparse
import os
import time

import numpy as np

BUCKETS = [(32, 128), (64, 64), (128, 32), (256, 16)]   # (sequence length, batch) of the IRs


def load(prompts, month, limit):
    ids = np.load(f'{prompts}/{month}.ids.npy')
    lens = np.load(f'{prompts}/{month}.len.npy').astype(np.int64)
    tok = np.load(f'{prompts}/{month}.tok.npy')
    off = np.concatenate([[0], np.cumsum(lens)])
    if limit and limit < len(ids):
        rows = np.sort(np.random.default_rng(1).choice(len(ids), limit, replace=False))
        ids, lens, seqs = ids[rows], lens[rows], [tok[off[r]:off[r + 1]] for r in rows]
    else:
        seqs = [tok[off[r]:off[r + 1]] for r in range(len(ids))]
    return ids, lens, seqs


def embed(lens, seqs, device):
    import openvino as ov
    core = ov.Core()
    # compiled models cached on disk: a month after the first one skips the GPU compile (~20 s)
    core.set_property({'CACHE_DIR': 'work/ov_cache'})
    out = np.zeros((len(seqs), 768), np.float16)
    lo = 0
    for seq, bs in BUCKETS:
        sel = np.where((lens > lo) & (lens <= seq))[0] if seq != BUCKETS[-1][0] else np.where(lens > lo)[0]
        lo = seq
        if len(sel) == 0:
            continue
        cfg = {'PERFORMANCE_HINT': 'THROUGHPUT'}
        if device == 'GPU':
            cfg['INFERENCE_PRECISION_HINT'] = 'f16'
        cm = core.compile_model(f'models/gemma2_text_b{bs}_s{seq}.xml', device, cfg)
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
                x = seqs[r][:seq]
                a[j, :len(x)] = x
                m[j, :len(x)] = 1
            if len(rows) < bs:
                m[len(rows):, 0] = 1                  # padding rows: one visible token, never read
            q.start_async([a, m], rows)
        q.wait_all()
    return out


def above_normal_priority():
    """Windows: this process above normal priority (it needs little CPU, but promptly, to keep the
    GPU busy while other processes load every core; children do not inherit this class)."""
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = wintypes.HANDLE
        k.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k.SetPriorityClass(k.GetCurrentProcess(), 0x8000)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('prompts')
    ap.add_argument('month')
    ap.add_argument('--device', default='GPU')
    ap.add_argument('--limit', type=int, default=0, help='a random sample of this many documents')
    ap.add_argument('--out', help='output path without extension (default emb/g2v2_<month>)')
    ap.add_argument('--compare', help='other vectors (path without .f16.npy) to compare with, by id')
    ap.add_argument('--above-normal', action='store_true', help='run above normal priority (Windows)')
    a = ap.parse_args()
    if a.above_normal:
        above_normal_priority()
    ids, lens, seqs = load(a.prompts, a.month, a.limit)
    t0 = time.time()
    vec = embed(lens, seqs, a.device)
    dt = time.time() - t0
    print(f'{a.month}: {len(ids):,} prompts on {a.device}, {lens.mean():.0f} tokens on average: '
          f'{len(ids) / dt:.0f} docs/s ({dt / 60:.1f} min)', flush=True)
    if a.compare:
        other = np.load(f'{a.compare}.f16.npy').astype(np.float32)
        pos = {int(i): k for k, i in enumerate(np.load(f'{a.compare}.ids.npy'))}
        both = [k for k, i in enumerate(ids) if int(i) in pos]
        cos = np.einsum('ij,ij->i', vec[both].astype(np.float32), other[[pos[int(ids[k])] for k in both]])
        print(f'cosine vs {a.compare} on {len(both):,} documents: mean {cos.mean():.4f}, median {np.median(cos):.4f}, '
              f'p5 {np.percentile(cos, 5):.4f}, < 0.95: {(cos < 0.95).mean():.1%}', flush=True)
    if not a.limit or a.out:
        base = a.out or f'emb/g2v2_{a.month}'
        # written under temporary names and renamed, the vectors last: a cut run never leaves a
        # pair of files that looks complete
        for ext, arr in (('ids.npy', ids.astype(np.uint32)), ('f16.npy', vec)):
            with open(f'{base}.{ext}.tmp', 'wb') as f:
                np.save(f, arr)
            os.replace(f'{base}.{ext}.tmp', f'{base}.{ext}')
        print(f'-> {base}.f16.npy', flush=True)


if __name__ == '__main__':
    main()
