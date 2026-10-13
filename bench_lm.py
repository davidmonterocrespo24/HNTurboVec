"""Limite de compresion con un LM (codificacion aritmetica): bits = sum(-log2 p(token)).
Cada registro se comprime SOLO (acceso aleatorio perfecto, sin diccionario: el modelo es el diccionario).
El codificador aritmetico real anade ~2 bytes/registro, incluidos abajo."""
import sys, math, time, numpy as np, torch, pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL, NS = sys.argv[1], int(sys.argv[2])
torch.set_num_threads(12)
t = pq.read_table('data/2024/2024-01.parquet', columns=['text', 'title', 'dead'])
text, title = t['text'].to_pylist(), t['title'].to_pylist()
rng = np.random.default_rng(7)
sel = [i for i in rng.permutation(len(text)) if text[i] or title[i]][:NS]
recs = [(title[i] + '\n' + text[i]) if title[i] else text[i] for i in sel]

tok = AutoTokenizer.from_pretrained(MODEL)
lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
tot_bits, tot_bytes = 0.0, 0
t0 = time.time()
with torch.no_grad():
    for r in recs:
        ids = tok(r, add_special_tokens=False)['input_ids'][:1024]
        x = torch.tensor([[bos] + ids])
        logp = torch.log_softmax(lm(x).logits[0, :-1].float(), -1)
        nll = -logp[torch.arange(len(ids)), torch.tensor(ids)].sum().item() / math.log(2)
        tot_bits += nll + 16 + 8 * 0  # +2 bytes de cierre del codificador
        tot_bytes += len(r.encode())
dt = time.time() - t0
print(f'{MODEL}: {NS} registros, {tot_bytes/NS:.0f} B/reg crudo -> {tot_bits/8/NS:.1f} B/reg '
      f'({tot_bits/tot_bytes:.2f} bits/byte)  ~{dt/NS*1000:.0f} ms/registro en CPU '
      f'| 49.3M -> {tot_bits/8/NS*49.3e6/1e9:.2f} GB (+ el modelo)')
