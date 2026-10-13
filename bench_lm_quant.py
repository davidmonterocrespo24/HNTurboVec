"""¿Cuánto empeora la compresión si el LM corre cuantizado?
- fp32 (referencia)
- W8A32: pesos int8 por fila, activaciones float
- W8A8: pesos int8 por fila + activaciones int8 por token (matmul ENTERA exacta -> determinista)
Se simula en PyTorch envolviendo todas las capas Linear (incluida la cabeza de vocabulario)."""
import sys, math, re, html, numpy as np, torch, pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'EleutherAI/pythia-70m'
torch.set_num_threads(16)
A_RE = re.compile(r'<a href="([^"]*)" rel="nofollow">(.*?)</a>', re.S)

def rep_e(s):
    def a(m):
        href, txt = m.group(1), m.group(2)
        u = html.unescape(href)
        if txt == href: return '<' + u + '>'
        if txt.endswith('...') and href.startswith(txt[:-3]): return '<' + u + '>' + str(len(txt) - 3) + '>'
        return '<' + u + '|' + txt + '>'
    return html.unescape(A_RE.sub(a, s).replace('<p>', '\n\n'))

class QLinear(torch.nn.Module):
    def __init__(self, lin, act_bits):
        super().__init__()
        w = lin.weight.data.float()
        s = w.abs().amax(1, keepdim=True).clamp(min=1e-12) / 127.0
        self.wq = torch.round(w / s).clamp(-127, 127)
        self.ws = s
        self.bias = None if lin.bias is None else lin.bias.data.float()
        self.act_bits = act_bits
    def forward(self, x):
        if self.act_bits:
            xs = x.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 127.0
            xq = torch.round(x / xs).clamp(-127, 127)
            y = (xq @ self.wq.t()) * xs * self.ws.t()
        else:
            y = x @ (self.wq * self.ws).t()
        return y if self.bias is None else y + self.bias

def quantize(m, act_bits):
    for name, child in list(m.named_children()):
        if isinstance(child, torch.nn.Linear):
            setattr(m, name, QLinear(child, act_bits))
        else:
            quantize(child, act_bits)

tok = AutoTokenizer.from_pretrained(MODEL)
bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
t = pq.read_table('data/2024/2024-01.parquet', columns=['text', 'title'])
text, title = t['text'].to_pylist(), t['title'].to_pylist()
rng = np.random.default_rng(7)
sel = [i for i in rng.permutation(len(text)) if text[i] or title[i]][:300]
recs = [title[i] + '\n' + rep_e(text[i]) for i in sel]
for label, act in [('fp32', None), ('W8A32', 0), ('W8A8', 8)]:
    lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    if act is not None:
        quantize(lm, act)
    bits = 0.0
    with torch.no_grad():
        for r in recs:
            ids = tok(r, add_special_tokens=False)['input_ids'][:2048]
            x = torch.tensor([[bos] + ids])
            lp = torch.log_softmax(lm(x).logits[0, :-1].float(), -1)
            bits += float(-lp[torch.arange(len(ids)), torch.tensor(ids)].sum()) / math.log(2)
    print(f'{MODEL:<24} {label:<6} {bits/8/300 + 2:6.2f} B/reg', flush=True)
