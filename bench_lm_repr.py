r"""¿Qué representación del texto comprime mejor con el LM? (bytes y tokens por registro)
(a) HTML original de HN
(b) forma compacta con caracteres de control (como el nivel zstd)
(c) compacta "natural" NO reversible (cota): <p> -> \n\n, enlaces como url desnuda
(d) reversible: <p> -> \n\n, enlaces con marcadores de control, resto igual
(e) reversible: <p> -> \n\n, enlaces como <url> (el LM ve una url normal)"""
import sys, math, re, html, numpy as np, torch, pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'EleutherAI/pythia-70m'
torch.set_num_threads(16)
A_RE = re.compile(r'<a href="([^"]*)" rel="nofollow">(.*?)</a>', re.S)
TAGS = [('<p>', '\x05'), ('<i>', '\x06'), ('</i>', '\x07'), ('<pre><code>', '\x0e'), ('</code></pre>', '\x0f')]


def anchors(s, style):
    def a(m):
        href, txt = m.group(1), m.group(2)
        u = html.unescape(href)
        same = txt == href
        trunc = (not same) and txt.endswith('...') and href.startswith(txt[:-3])
        if style == 'bare':
            return u
        if style == 'ctrl':
            if same:
                return '\x01' + u + '\x02'
            if trunc:
                return '\x11' + u + '\x02' + str(len(txt) - 3) + '\x04'
            return '\x03' + u + '\x02' + txt + '\x04'
        if style == 'angle':
            if same:
                return '<' + u + '>'
            if trunc:
                return '<' + u + '>' + str(len(txt) - 3) + '>'
            return '<' + u + '|' + txt + '>'
    return A_RE.sub(a, s)


def rep(s, kind):
    if kind == 'a':
        return s
    if kind == 'b':
        s2 = anchors(s, 'ctrl')
        for k, v in TAGS:
            s2 = s2.replace(k, v)
        return html.unescape(s2)
    if kind == 'c':
        s2 = anchors(s, 'bare').replace('<p>', '\n\n')
        return html.unescape(s2)
    if kind == 'd':
        return html.unescape(anchors(s, 'ctrl').replace('<p>', '\n\n'))
    if kind == 'e':
        return html.unescape(anchors(s, 'angle').replace('<p>', '\n\n'))


tok = AutoTokenizer.from_pretrained(MODEL)
lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
NAMES = {'a': 'a) HTML original', 'b': 'b) compacta (control)', 'c': 'c) natural (cota, no rev.)',
         'd': 'd) rev.: \\n\\n + marcadores', 'e': 'e) rev.: \\n\\n + <url>'}
for path in ['data/2024/2024-01.parquet', 'data/2010/2010-06.parquet']:
    t = pq.read_table(path, columns=['text', 'title', 'deleted'])
    text, title, dele = t['text'].to_pylist(), t['title'].to_pylist(), t['deleted'].to_pylist()
    rng = np.random.default_rng(7)
    sel = [i for i in rng.permutation(len(text)) if (text[i] or title[i]) and not dele[i]][:300]
    raw = sum(len((title[i] + '\n' + text[i]).encode()) for i in sel)
    print(f'{path} ({MODEL}), {raw/300:.0f} B/reg HTML original')
    for kind in 'abcde':
        recs = [title[i] + '\n' + rep(text[i], kind) for i in sel]
        bits = 0.0; ntok = 0
        with torch.no_grad():
            for r in recs:
                ids = tok(r, add_special_tokens=False)['input_ids'][:2048]
                x = torch.tensor([[bos] + ids])
                lp = torch.log_softmax(lm(x).logits[0, :-1].float(), -1)
                bits += float(-lp[torch.arange(len(ids)), torch.tensor(ids)].sum()) / math.log(2); ntok += len(ids)
        print(f'   {NAMES[kind]:<30} {bits/8/300 + 2:6.1f} B/reg   {ntok/300:5.1f} tokens/reg', flush=True)
