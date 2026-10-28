"""Exports a Pythia (GPT-NeoX) model to the .hnzlm format used by the deterministic Rust engine:
- matrices in int8 per row (one f32 scale per row), vectors in f32
- query_key_value rows reordered to [Q of all heads | K | V]
- RoPE table (cos/sin) precomputed in f32 (so Rust does not use sin/cos from libm)
- vocabulary: the bytes of each token (byte-level BPE)
Usage: python scripts/export_lm.py EleutherAI/pythia-70m models/lm-pythia-70m.hnzlm"""
import sys, json, struct, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name, out = sys.argv[1], sys.argv[2]
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
tok = AutoTokenizer.from_pretrained(name)
cfg = model.config
sd = {k: v.detach().float().numpy() for k, v in model.state_dict().items()}
H, NH, L, I, V = cfg.hidden_size, cfg.num_attention_heads, cfg.num_hidden_layers, cfg.intermediate_size, cfg.vocab_size
HS = H // NH
RP = cfg.rope_parameters
ROT = int(HS * RP.get('partial_rotary_factor', 1.0))
MAXPOS = cfg.max_position_embeddings
assert cfg.use_parallel_residual and cfg.hidden_act in ('gelu',), cfg.hidden_act

blobs, tensors, off = [], [], 0
def add(nm, arr, kind):
    global off
    arr = np.ascontiguousarray(arr)
    if kind == 'q8':
        s = np.abs(arr).max(1).clip(min=1e-12).astype(np.float32) / 127.0
        q = np.clip(np.round(arr / s[:, None]), -127, 127).astype(np.int8)
        data = s.astype('<f4').tobytes() + q.tobytes()
        tensors.append({'name': nm, 'kind': 'q8', 'rows': arr.shape[0], 'cols': arr.shape[1], 'offset': off})
    else:
        data = arr.astype('<f4').tobytes()
        tensors.append({'name': nm, 'kind': 'f32', 'len': int(arr.size), 'offset': off})
    pad = (-len(data)) % 64
    blobs.append(data + b'\0' * pad); off += len(data) + pad

# qkv row permutation: HF groups by head [q_h, k_h, v_h]; we use [Q | K | V]
perm = []
for part in range(3):
    for h in range(NH):
        base = h * 3 * HS + part * HS
        perm.extend(range(base, base + HS))
perm = np.array(perm)

add('embed_in', sd['gpt_neox.embed_in.weight'][:V], 'q8')
for l in range(L):
    p = f'gpt_neox.layers.{l}.'
    add(f'l{l}.ln1.w', sd[p + 'input_layernorm.weight'], 'f32'); add(f'l{l}.ln1.b', sd[p + 'input_layernorm.bias'], 'f32')
    add(f'l{l}.ln2.w', sd[p + 'post_attention_layernorm.weight'], 'f32'); add(f'l{l}.ln2.b', sd[p + 'post_attention_layernorm.bias'], 'f32')
    add(f'l{l}.qkv.w', sd[p + 'attention.query_key_value.weight'][perm], 'q8'); add(f'l{l}.qkv.b', sd[p + 'attention.query_key_value.bias'][perm], 'f32')
    add(f'l{l}.o.w', sd[p + 'attention.dense.weight'], 'q8'); add(f'l{l}.o.b', sd[p + 'attention.dense.bias'], 'f32')
    add(f'l{l}.fc1.w', sd[p + 'mlp.dense_h_to_4h.weight'], 'q8'); add(f'l{l}.fc1.b', sd[p + 'mlp.dense_h_to_4h.bias'], 'f32')
    add(f'l{l}.fc2.w', sd[p + 'mlp.dense_4h_to_h.weight'], 'q8'); add(f'l{l}.fc2.b', sd[p + 'mlp.dense_4h_to_h.bias'], 'f32')
add('lnf.w', sd['gpt_neox.final_layer_norm.weight'], 'f32'); add('lnf.b', sd['gpt_neox.final_layer_norm.bias'], 'f32')
add('embed_out', sd['lm_head.weight' if 'lm_head.weight' in sd else 'embed_out.weight'][:V], 'q8')
# RoPE: angle[pos][i] = pos * inv_freq[i], i < ROT/2
inv = 1.0 / (RP['rope_theta'] ** (np.arange(0, ROT, 2, dtype=np.float64) / ROT))
ang = np.outer(np.arange(MAXPOS, dtype=np.float64), inv)
add('rope_cos', np.cos(ang).astype(np.float32), 'f32'); add('rope_sin', np.sin(ang).astype(np.float32), 'f32')

# vocabulary: the bytes of each id (byte-level BPE of GPT-2/NeoX)
def bytes_to_unicode():
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}
inv_map = bytes_to_unicode()
id2tok = {i: t for t, i in tok.get_vocab().items()}
special = set(tok.all_special_ids)
added = {i: t.content for i, t in tok.added_tokens_decoder.items()}
vocab_bytes = []
for i in range(V):
    t = id2tok.get(i)
    if t is None or i in special:
        vocab_bytes.append(b'')
    elif i in added:
        vocab_bytes.append(added[i].encode('utf-8'))  # added tokens: literal text
    else:
        vocab_bytes.append(bytes(inv_map[c] for c in t))
# check: decoding ids with the table == the tokenizer's decode
for txt in ['Hello world', '  indented' + chr(10) + chr(10) + '  code  ', 'naïve café — “quotes”', 'url <https://x.com/a?b=1>80> end']:
    ids = tok(txt, add_special_tokens=False)['input_ids']
    assert b''.join(vocab_bytes[i] for i in ids).decode('utf-8') == txt, txt
vb = b''.join(vocab_bytes); vo = np.cumsum([0] + [len(x) for x in vocab_bytes]).astype('<u4')
add_raw_off = off
blobs.append(vo.tobytes()); off += len(vo.tobytes())
blobs.append(vb); off += len(vb)

bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
header = {'arch': 'gpt_neox', 'source': name, 'hidden': H, 'heads': NH, 'layers': L, 'inter': I, 'vocab': V,
          'rotary_ndims': ROT, 'max_pos': MAXPOS, 'eps': cfg.layer_norm_eps, 'bos': bos,
          'tensors': tensors, 'vocab_offsets': add_raw_off, 'vocab_bytes': add_raw_off + len(vo.tobytes()), 'vocab_bytes_len': len(vb)}
hj = json.dumps(header).encode()
with open(out, 'wb') as f:
    f.write(b'HNZLM1\0\0'); f.write(struct.pack('<I', len(hj))); f.write(hj)
    pad = (-(12 + len(hj))) % 64; f.write(b'\0' * pad)
    for b in blobs: f.write(b)
# copy of the tokenizer (for encoding from Rust)
tok.backend_tokenizer.save(out.replace('.hnzlm', '.tokenizer.json'))
print(f'{out}: H={H} heads={NH} layers={L} inter={I} vocab={V} rot={ROT} | {sum(len(b) for b in blobs)/1e6:.1f} MB')
