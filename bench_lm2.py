"""Comparativa de LMs como compresor (codificacion aritmetica, cada registro solo):
- bits/byte en texto real de HN (mes reciente y mes antiguo)
- velocidad de CODIFICAR (pasada paralela, tokens/s)
- velocidad de DECODIFICAR (secuencial con KV-cache, lote de 10 registros, ms/paso)
- round-trip del tokenizador (decode(encode(x)) == x)
Uso: python bench_lm2.py <parquet> <n_registros> <modelo1,modelo2,...>"""
import sys, math, time, os
import numpy as np, torch, pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForCausalLM

PATH, NS, MODELS = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(',')
torch.set_num_threads(os.cpu_count())
t = pq.read_table(PATH, columns=['text', 'title'])
text, title = t['text'].to_pylist(), t['title'].to_pylist()
rng = np.random.default_rng(7)
sel = [i for i in rng.permutation(len(text)) if text[i] or title[i]][:NS]
recs = [(title[i] + '\n' + text[i]) if title[i] else text[i] for i in sel]
raw_bytes = sum(len(r.encode()) for r in recs)
print(f'{PATH}: {NS} registros, {raw_bytes/NS:.0f} B/reg crudo')

for name in MODELS:
    tok = AutoTokenizer.from_pretrained(name)
    lm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    nparam = sum(p.numel() for p in lm.parameters())
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    toks = [tok(r, add_special_tokens=False)['input_ids'][:2048] for r in recs]
    rt = np.mean([tok.decode(x, clean_up_tokenization_spaces=False) == r for x, r in zip(toks, recs)])
    # --- bits (pasada paralela por lotes ordenados por longitud) ---
    order = np.argsort([len(x) for x in toks]); bits = 0.0; ntok = 0
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, NS, 16):
            batch = [toks[i] for i in order[b:b + 16]]
            L = max(len(x) for x in batch) + 1
            ids = torch.full((len(batch), L), bos); att = torch.zeros((len(batch), L), dtype=torch.long)
            for j, x in enumerate(batch):
                ids[j, 1:len(x) + 1] = torch.tensor(x); att[j, :len(x) + 1] = 1
            logp = torch.log_softmax(lm(input_ids=ids, attention_mask=att).logits[:, :-1].float(), -1)
            tgt = ids[:, 1:]
            lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * att[:, 1:]
            bits += float(-lp.sum()) / math.log(2); ntok += int(att[:, 1:].sum())
    t_enc = time.time() - t0
    bytes_per_rec = (bits / 8 + 2 * NS) / NS  # +2 bytes de cierre del codificador por registro
    # --- decodificacion secuencial con KV-cache, lote de 10 ---
    B, STEPS = 10, 48
    with torch.no_grad():
        x = torch.full((B, 1), bos)
        out = lm(input_ids=x, use_cache=True); pkv = out.past_key_values
        nxt = torch.randint(100, 1000, (B, 1))
        for _ in range(4):
            out = lm(input_ids=nxt, past_key_values=pkv, use_cache=True); pkv = out.past_key_values
        t0 = time.time()
        for _ in range(STEPS):
            out = lm(input_ids=nxt, past_key_values=pkv, use_cache=True); pkv = out.past_key_values
            p = torch.softmax(out.logits[:, -1].float(), -1)  # distribucion completa (como el decodificador)
        ms_step = (time.time() - t0) / STEPS * 1000
    tok_per_rec = ntok / NS
    print(f'{name:<32} {nparam/1e6:6.0f}M  {bits/raw_bytes:5.3f} bits/byte  {bytes_per_rec:6.1f} B/reg  '
          f'tokens/reg {tok_per_rec:5.1f}  roundtrip {rt:.3f}  | codificar {ntok/t_enc:7.0f} tok/s '
          f'(49,3M: {49.3e6*tok_per_rec/(ntok/t_enc)/86400:5.1f} dias) | decodificar lote 10: {ms_step:5.1f} ms/paso '
          f'-> top-10 en ~{ms_step*tok_per_rec/1000:4.2f} s', flush=True)
