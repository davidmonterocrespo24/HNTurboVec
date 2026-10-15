"""E29: continuar el preentrenamiento de pythia-70m con texto de HN (para comprimir mejor el nivel frío).

Misma arquitectura, tokenizador y coste por token que pythia-70m: el motor de Rust y export_lm.py
sirven sin cambios y la descompresión va igual de rápida. Solo cambia lo bien que predice HN.
- Datos: JSONL de `hnz lmdump` (el texto EXACTO que codifica el nivel frío: título\\nurl\\ncuerpo en
  la representación "e"); cada registro es una secuencia propia que empieza en BOS, como al comprimir.
- Evaluación: bits/byte en meses que NO están en el entrenamiento (2010-06, 2015-06, 2024-01).
- Reanudable: punto de control (modelo + optimizador + paso) cada --save pasos; al relanzar sigue.
Uso:
  python scripts/train_lm.py --data data_lm/train.jsonl --eval data_lm/eval_*.jsonl --steps 6000 --out models/lm-hn-70m
  python scripts/train_lm.py ... --probe 20        # solo mide tokens/s
"""
import argparse, glob, json, math, os, random, time
import numpy as np, torch
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
from transformers import AutoModelForCausalLM, AutoTokenizer


def log(*a):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), *a, flush=True)


def load(path, tok, max_len, limit=None):
    """[(token ids con BOS al principio, bytes del texto)]"""
    recs = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            s = json.loads(line)['s']
            recs.append(s)
            if limit and len(recs) >= limit:
                break
    out = []
    for i in range(0, len(recs), 4096):
        enc = tok(recs[i:i + 4096], add_special_tokens=False)['input_ids']
        for s, ids in zip(recs[i:i + 4096], enc):
            # int32 arrays, not Python lists: 5 % of HN is ~184 M tokens (lists would take ~6-8 GB)
            out.append((np.array(([tok.bos_token_id] + ids)[:max_len], dtype=np.int32), len(s.encode('utf-8'))))
    return out


def batches(data, batch_tokens, rng):
    """Lotes de longitud parecida (poco relleno), en orden aleatorio."""
    order = sorted(range(len(data)), key=lambda i: (len(data[i][0]), rng.random()))
    out, cur, longest = [], [], 0
    for i in order:
        L = len(data[i][0])
        if cur and max(longest, L) * (len(cur) + 1) > batch_tokens:
            out.append(cur)
            cur, longest = [], 0
        cur.append(i)
        longest = max(longest, L)
    if cur:
        out.append(cur)
    rng.shuffle(out)
    return out


DEV = torch.device('cpu')
AMP = False   # fp16 mixto en la GPU (pesos y optimizador en float32). OJO: bf16 rompe pythia
              # (pérdida 5,7 frente a 4,0 en fp32); fp16, su formato nativo, da lo mismo que fp32


def tensors(data, idx, pad):
    L = max(len(data[i][0]) for i in idx)
    x = torch.full((len(idx), L), pad, dtype=torch.long)
    m = torch.zeros((len(idx), L), dtype=torch.long)
    for r, i in enumerate(idx):
        ids = data[i][0]
        x[r, :len(ids)] = torch.from_numpy(ids)
        m[r, :len(ids)] = 1
    return x.to(DEV), m.to(DEV)


def nll_bits(model, x, m, amp=None):
    with torch.autocast(device_type=DEV.type, dtype=torch.float16, enabled=AMP if amp is None else amp):
        logits = model(input_ids=x, attention_mask=m).logits[:, :-1]
    tgt, msk = x[:, 1:], m[:, 1:].bool()
    # entropía cruzada fusionada en float32 solo sobre las posiciones reales (sin relleno): no se
    # guarda una copia completa de log-probabilidades (vocabulario de 50k) para el gradiente
    nll = torch.nn.functional.cross_entropy(logits[msk].float(), tgt[msk], reduction='sum')
    return nll / math.log(2), msk.sum()


@torch.no_grad()
def evaluate(model, sets, pad):
    model.eval()
    res = {}
    for name, data in sets.items():
        bits = nbytes = 0.0
        for idx in batches(data, 8192, random.Random(0)):
            x, m = tensors(data, idx, pad)
            b, _ = nll_bits(model, x, m, amp=False)   # evaluación exacta en float32
            bits += float(b)
            nbytes += sum(data[i][1] for i in idx)
        res[name] = bits / nbytes
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--eval', nargs='*', default=[])
    ap.add_argument('--base', default='EleutherAI/pythia-70m')
    ap.add_argument('--out', default='models/lm-hn-70m')
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--batch-tokens', type=int, default=8192)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--max-len', type=int, default=512)
    ap.add_argument('--save', type=int, default=250)
    ap.add_argument('--eval-every', type=int, default=500)
    ap.add_argument('--eval-n', type=int, default=1000)
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--device', default='auto', help='cpu, xpu (iGPU Intel) o auto')
    ap.add_argument('--fp32', action='store_true', help='sin fp16 mixto en la GPU')
    ap.add_argument('--probe', type=int, default=0)
    a = ap.parse_args()
    if os.name == 'nt' and not a.probe:
        # keep Windows from sleeping while this process trains (released when it exits)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    torch.set_num_threads(a.threads)
    global DEV, AMP
    DEV = torch.device('xpu' if a.device == 'auto' and hasattr(torch, 'xpu') and torch.xpu.is_available() else
                       ('cpu' if a.device == 'auto' else a.device))
    AMP = DEV.type != 'cpu' and not a.fp32
    log(f'dispositivo {DEV}' + (' (fp16 mixto)' if AMP else ''))
    scaler = torch.amp.GradScaler(DEV.type, enabled=AMP)
    tok = AutoTokenizer.from_pretrained(a.base)
    ck = os.path.join(a.out, 'checkpoint')
    start = os.path.exists(os.path.join(ck, 'state.pt'))
    model = AutoModelForCausalLM.from_pretrained(ck if start else a.base, dtype=torch.float32).to(DEV)
    model.train()
    log(f'cargando datos de {a.data}')
    data = load(a.data, tok, a.max_len)
    ntok = sum(len(d[0]) for d in data)
    log(f'{len(data):,} registros, {ntok / 1e6:.1f} M tokens ({ntok / len(data):.0f}/registro)')
    sets = {os.path.basename(p).replace('.jsonl', ''): load(p, tok, a.max_len, a.eval_n) for p in a.eval}
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
    step, rng = 0, random.Random(1234)
    if start:
        st = torch.load(os.path.join(ck, 'state.pt'), weights_only=False, map_location=DEV)
        opt.load_state_dict(st['opt'])
        step, rng = st['step'], st['rng']
        log(f'reanudando en el paso {step}')
    elif sets and not a.probe:
        log('base (pythia-70m sin entrenar): ' + ', '.join(f'{k} {v:.4f} bits/byte' for k, v in evaluate(model, sets, tok.bos_token_id).items()))
    sched = lambda s: a.lr * min(1.0, (s + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    queue, t0, seen = [], time.time(), 0
    while step < (a.probe or a.steps):
        if not queue:
            queue = batches(data, a.batch_tokens, rng)
        idx = queue.pop()
        x, m = tensors(data, idx, tok.bos_token_id)
        bits, n = nll_bits(model, x, m)
        loss = bits * math.log(2) / n
        for g in opt.param_groups:
            g['lr'] = sched(step)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        seen += int(m.sum())
        if step % 20 == 0 or a.probe:
            dt = time.time() - t0
            log(f'paso {step}: pérdida {float(loss):.4f}, {seen / dt:,.0f} tokens/s, lr {sched(step):.2e}')
        if a.probe:
            continue
        if step % a.save == 0 or step == a.steps:
            os.makedirs(ck, exist_ok=True)
            model.save_pretrained(ck)
            tok.save_pretrained(ck)
            torch.save({'opt': opt.state_dict(), 'step': step, 'rng': rng}, os.path.join(ck, 'state.pt.tmp'))
            os.replace(os.path.join(ck, 'state.pt.tmp'), os.path.join(ck, 'state.pt'))
        if sets and (step % a.eval_every == 0 or step == a.steps):
            log(f'evaluación paso {step}: ' + ', '.join(f'{k} {v:.4f} bits/byte' for k, v in evaluate(model, sets, tok.bos_token_id).items()))
    if not a.probe:
        model.save_pretrained(a.out)
        tok.save_pretrained(a.out)
        log(f'modelo final en {a.out} (exportar con: python scripts/export_lm.py {a.out} {a.out}.hnzlm)')


if __name__ == '__main__':
    main()
