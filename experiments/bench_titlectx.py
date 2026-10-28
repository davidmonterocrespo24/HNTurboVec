"""E28b: how much does the LM save if each comment is coded with the thread as context?

pythia-70m (PyTorch, float32): bits of the comment's tokens (−log2 p) under three conditions:
  - alone: <|endoftext|> + comment (as today: each record alone)
  - title: <|endoftext|> + title of the thread's story + "\\n\\n" + comment
  - title + parent: also the first 64 tokens of the parent comment (if the parent is not the story)
Only the comment's bits are counted (the same tokens in all three). Sample: 300 comments from each
month (2024-01, 2015-06), plain text.
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
    nll = -lp[torch.arange(len(tgt)), tgt][len(ctx):]      # only the comment's tokens
    return float(nll.sum() / np.log(2))


for m, y in (('2024-01', 2024), ('2015-06', 2015)):
    t = pq.read_table(f'data/{y}/{m}.parquet', columns=['id', 'type', 'parent', 'title', 'text', 'dead', 'deleted'])
    ids, typ, par, title, text, dead, deleted = (t[c].to_pylist() for c in ['id', 'type', 'parent', 'title', 'text', 'dead', 'deleted'])
    idx = {i: k for k, i in enumerate(ids)}
    rng = np.random.default_rng(0)
    cand = [k for k in range(len(ids)) if typ[k] == 2 and not dead[k] and not deleted[k] and text[k] and par[k] in idx]
    tot = {'alone': 0.0, 'title': 0.0, 'title + parent': 0.0}
    nbytes = n = 0
    for k in rng.permutation(cand):
        p, cur = par[k], k
        while p and p in idx:
            cur = idx[p]; p = par[cur]
        if p or typ[cur] != 1 or not title[cur]:
            continue                                         # thread incomplete in this month
        c = plain(text[k])
        ci = tok(c)['input_ids'][:512]
        ti = tok(plain(title[cur]) + '\n\n')['input_ids']
        pk = idx[par[k]]
        pi = tok(plain(text[pk]))['input_ids'][:64] + tok('\n\n')['input_ids'] if typ[pk] == 2 else []
        tot['alone'] += bits([], ci)
        tot['title'] += bits(ti, ci)
        tot['title + parent'] += bits(ti + pi, ci)
        nbytes += len(c.encode('utf-8'))
        n += 1
        if n == 300:
            break
    base = tot['alone']
    print(f'{m}: {n} comments, {nbytes / n:.0f} B of text on average')
    for kname, v in tot.items():
        print(f'  {kname:<16} {v / n / 8:6.1f} B/comment  {v / nbytes:.3f} bits/byte  ({(v - base) / base * 100:+.1f} %)', flush=True)
