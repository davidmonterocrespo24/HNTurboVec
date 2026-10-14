"""E28b: ¿cuánto ahorra el LM si cada comentario se codifica con contexto del hilo?

pythia-70m (PyTorch, float32): bits de los tokens del comentario (−log2 p) en tres condiciones:
  - solo: <|endoftext|> + comentario (como hoy: cada registro solo)
  - título: <|endoftext|> + título de la story del hilo + "\\n\\n" + comentario
  - título + padre: además, los primeros 64 tokens del comentario padre (si el padre no es la story)
Solo se cuentan los bits del comentario (mismos tokens en las tres). Muestra: 300 comentarios de cada
mes (2024-01, 2015-06), texto plano.
"""
import html, os, re, numpy as np, pyarrow.parquet as pq, torch
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')
model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-70m', torch_dtype=torch.float32).eval()
torch.set_num_threads(8)
EOS = tok.eos_token_id
plain = lambda s: re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', s or ''))).strip()


def bits(ctx, ids):
    x = torch.tensor([[EOS] + ctx + ids])
    with torch.no_grad():
        lp = torch.log_softmax(model(x).logits[0, :-1].float(), -1)
    tgt = x[0, 1:]
    nll = -lp[torch.arange(len(tgt)), tgt][len(ctx):]      # solo los tokens del comentario
    return float(nll.sum() / np.log(2))


for m, y in (('2024-01', 2024), ('2015-06', 2015)):
    t = pq.read_table(f'data/{y}/{m}.parquet', columns=['id', 'type', 'parent', 'title', 'text', 'dead', 'deleted'])
    ids, typ, par, title, text, dead, deleted = (t[c].to_pylist() for c in ['id', 'type', 'parent', 'title', 'text', 'dead', 'deleted'])
    idx = {i: k for k, i in enumerate(ids)}
    rng = np.random.default_rng(0)
    cand = [k for k in range(len(ids)) if typ[k] == 2 and not dead[k] and not deleted[k] and text[k] and par[k] in idx]
    tot = {'solo': 0.0, 'título': 0.0, 'título + padre': 0.0}
    nbytes = n = 0
    for k in rng.permutation(cand):
        p, cur = par[k], k
        while p and p in idx:
            cur = idx[p]; p = par[cur]
        if p or typ[cur] != 1 or not title[cur]:
            continue                                         # hilo incompleto en este mes
        c = plain(text[k])
        ci = tok(c)['input_ids'][:512]
        ti = tok(plain(title[cur]) + '\n\n')['input_ids']
        pk = idx[par[k]]
        pi = tok(plain(text[pk]))['input_ids'][:64] + tok('\n\n')['input_ids'] if typ[pk] == 2 else []
        tot['solo'] += bits([], ci)
        tot['título'] += bits(ti, ci)
        tot['título + padre'] += bits(ti + pi, ci)
        nbytes += len(c.encode('utf-8'))
        n += 1
        if n == 300:
            break
    base = tot['solo']
    print(f'{m}: {n} comentarios, {nbytes / n:.0f} B de texto de media')
    for kname, v in tot.items():
        print(f'  {kname:<16} {v / n / 8:6.1f} B/comentario  {v / nbytes:.3f} bits/byte  ({(v - base) / base * 100:+.1f} %)', flush=True)
