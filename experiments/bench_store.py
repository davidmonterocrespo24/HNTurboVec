"""Storage benchmark: how many bytes/record it costs to store HN losslessly with random access,
varying the order, block size, transform and compressor."""
import sys, time, re, html, struct, lzma, json
import numpy as np, pyarrow.parquet as pq, zstandard as zstd, brotli

PATH = sys.argv[1] if len(sys.argv) > 1 else 'data/2024/2024-01.parquet'
t = pq.read_table(PATH)
N = t.num_rows
cols = {c: t.column(c).to_pylist() for c in
        ['id', 'deleted', 'type', 'by', 'text', 'dead', 'parent', 'poll', 'url', 'score', 'title', 'parts', 'descendants']}
cols['time'] = (t.column('time').cast('int64').to_numpy() // 1000).tolist()
print(f'{N:,} records')

# ---------- reversible transform of HN's HTML ----------
A_RE = re.compile(r'<a href="([^"]*)" rel="nofollow">(.*?)</a>', re.S)
TAGS = [('<p>', '\x05'), ('<i>', '\x06'), ('</i>', '\x07'), ('<pre><code>', '\x0e'), ('</code></pre>', '\x0f')]

def esc(s):
    return (s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
             .replace('"', '&quot;').replace("'", '&#x27;').replace('/', '&#x2F;'))

def enc(s):
    if any(c in s for c in '\x01\x02\x03\x04\x05\x06\x07\x0e\x0f'):
        return None
    def a(m):
        href, txt = m.group(1), m.group(2)
        if txt == href:
            return '\x01' + html.unescape(href) + '\x02'
        return '\x03' + html.unescape(href) + '\x02' + txt + '\x04'
    s2 = A_RE.sub(a, s)
    for k, v in TAGS:
        s2 = s2.replace(k, v)
    if '<' in s2 or '>' in s2:
        return None
    return html.unescape(s2)

def dec(s):
    out = []
    i = 0
    # rebuild the anchors
    def unanchor(seg):
        r = []
        j = 0
        while j < len(seg):
            c = seg[j]
            if c == '\x01':
                k = seg.index('\x02', j)
                u = esc(seg[j + 1:k]); r.append(f'<a href="{u}" rel="nofollow">{u}</a>'); j = k + 1
            elif c == '\x03':
                k = seg.index('\x02', j); m = seg.index('\x04', k)
                u = esc(seg[j + 1:k]); r.append(f'<a href="{u}" rel="nofollow">{seg[k+1:m]}</a>'); j = m + 1
            else:
                k = j
                while k < len(seg) and seg[k] not in '\x01\x03':
                    k += 1
                r.append(esc(seg[j:k]))
                j = k
        return ''.join(r)
    s = unanchor(s)
    for k, v in TAGS:
        s = s.replace(v, k)
    return s

def transform(s):
    if not s:
        return '\x00' + s
    e = enc(s)
    if e is not None and dec(e) == s:
        return '\x00' + e
    return '\x10' + s  # raw fallback

tt = time.time()
ttext = [transform(s) for s in cols['text']]
ok = sum(1 for x in ttext if x[0] == '\x00')
raw_text = sum(len(s.encode()) for s in cols['text'])
new_text = sum(len(s.encode()) - 1 for s in ttext)
print(f'HTML transform: {ok/N:.4f} reversible, text {raw_text/1e6:.1f} MB -> {new_text/1e6:.1f} MB ({time.time()-tt:.1f}s)')

# ---------- thread order ----------
idx_of = {i: k for k, i in enumerate(cols['id'])}
root = [0] * N
for k in range(N):
    p, cur = cols['parent'][k], k
    while p and p in idx_of:
        cur = idx_of[p]; p = cols['parent'][cur]
    root[k] = cols['id'][cur] if not p else p  # root outside the month -> group by the external parent
order_id = list(range(N))
order_thread = sorted(range(N), key=lambda k: (root[k], cols['id'][k]))

# ---------- serialization ----------
def varint(n, out):
    while n >= 0x80:
        out.append((n & 0x7f) | 0x80); n >>= 7
    out.append(n)

def zz(n):
    return (n << 1) ^ (n >> 63)

by_dict = {}
def ser_json(k, text):
    d = {c: cols[c][k] for c in cols if c != 'text'}
    d['text'] = text
    return json.dumps({x: y for x, y in d.items() if y not in (0, '', [], None)}, ensure_ascii=False,
                      separators=(',', ':')).encode()

def ser_block_columnar(ks, texts):
    """columnar block: metadata as varints (deltas), then strings separated by \\x00, one column at a time"""
    out = bytearray()
    prev_id = cols['id'][ks[0]]; prev_t = cols['time'][ks[0]]
    varint(prev_id, out); varint(prev_t, out)
    for k in ks:
        i = cols['id'][k]
        varint(zz(i - prev_id), out); prev_id = i
        varint(zz(cols['time'][k] - prev_t), out); prev_t = cols['time'][k]
        out.append(cols['type'][k] | (cols['dead'][k] << 4) | (cols['deleted'][k] << 5))
        p = cols['parent'][k]
        varint(zz(i - p) if p else 0, out)
        varint(zz(cols['score'][k]), out); varint(zz(cols['descendants'][k]), out)
        varint(cols['poll'][k], out); varint(len(cols['parts'][k]), out)
        for x in cols['parts'][k]:
            varint(x, out)
        b = cols['by'][k]
        if b not in by_dict:
            by_dict[b] = len(by_dict)
        varint(by_dict[b], out)
    for col in ('title', 'url'):
        out += '\x00'.join(cols[col][k] for k in ks).encode() + b'\x01'
    out += '\x00'.join(texts[k] for k in ks).encode()
    return bytes(out)

def run(name, order, blk, fmt, texts, comp, sample_dict=True):
    blocks = []
    for s in range(0, N, blk):
        ks = order[s:s + blk]
        if fmt == 'json':
            blocks.append(b'\n'.join(ser_json(k, texts[k]) for k in ks))
        else:
            blocks.append(ser_block_columnar(ks, texts))
    raw = sum(len(b) for b in blocks)
    t0 = time.time()
    if comp.startswith('zstd'):
        lvl = int(comp.split('-')[1])
        dsz = int(comp.split('-')[2]) * 1024 if comp.count('-') > 1 else 0
        if dsz:
            # train a dictionary on ~4000 sample blocks/records
            rng = np.random.default_rng(0)
            samp = [blocks[i] for i in rng.choice(len(blocks), min(len(blocks), 20000 if blk == 1 else 4000), replace=False)]
            dd = zstd.train_dictionary(dsz, samp, level=lvl)
            c = zstd.ZstdCompressor(level=lvl, dict_data=dd, write_checksum=False, write_content_size=False)
            d = zstd.ZstdDecompressor(dict_data=dd)
            extra = len(dd.as_bytes())
        else:
            c = zstd.ZstdCompressor(level=lvl, write_checksum=False, write_content_size=False); d = zstd.ZstdDecompressor(); extra = 0
        comp_blocks = [c.compress(b) for b in blocks]
        t1 = time.time()
        samp_idx = list(range(0, len(comp_blocks), max(1, len(comp_blocks) // 2000)))
        for i in samp_idx:
            d.decompress(comp_blocks[i], max_output_size=len(blocks[i]) + 64)
        dt = (time.time() - t1) / len(samp_idx)
    elif comp == 'brotli-11':
        comp_blocks = [brotli.compress(b, quality=11, lgwin=24) for b in blocks]; extra = 0
        t1 = time.time(); samp_idx = list(range(0, len(comp_blocks), max(1, len(comp_blocks) // 500)))
        for i in samp_idx: brotli.decompress(comp_blocks[i])
        dt = (time.time() - t1) / len(samp_idx)
    elif comp == 'xz-9e':
        comp_blocks = [lzma.compress(b, format=lzma.FORMAT_RAW,
                                     filters=[{'id': lzma.FILTER_LZMA2, 'preset': 9 | lzma.PRESET_EXTREME}]) for b in blocks]; extra = 0
        dt = float('nan')
    tot = sum(len(b) for b in comp_blocks) + extra + 8 * len(blocks)  # +8B offset per block
    print(f'{name:<44} blk={blk:<6} raw={raw/N:6.1f} B/rec -> {tot/N:6.1f} B/rec  x{raw/tot:4.2f}  '
          f'comp {time.time()-t0:6.1f}s  read 1 block {dt*1e6:8.0f} us  | 49.3M -> {tot/N*49.3e6/1e9:5.2f} GB', flush=True)
    return tot / N

texts_raw = cols['text']
print('\n== Baseline (from your research) ==')
run('json/record zstd19+dict64K (baseline)', order_id, 1, 'json', texts_raw, 'zstd-19-64')
print('\n== Effect of each idea, one at a time ==')
run('columnar/record zstd19+dict64K', order_id, 1, 'col', texts_raw, 'zstd-19-64')
run('columnar+transf/record zstd19+dict64K', order_id, 1, 'col', ttext, 'zstd-19-64')
for blk in (16, 64, 256, 1024):
    run('columnar+transf id-order zstd19+dict', order_id, blk, 'col', ttext, 'zstd-19-112')
for blk in (16, 64, 256, 1024):
    run('columnar+transf THREAD-order zstd19+dict', order_thread, blk, 'col', ttext, 'zstd-19-112')
print('\n== Stronger compressors (thread order, blocks of 256 and 4096) ==')
for blk in (256, 4096):
    run('columnar+transf THREAD-order zstd22+dict', order_thread, blk, 'col', ttext, 'zstd-22-112')
    run('columnar+transf THREAD-order brotli-11', order_thread, blk, 'col', ttext, 'brotli-11')
    run('columnar+transf THREAD-order xz-9e', order_thread, blk, 'col', ttext, 'xz-9e')
print('\n== Floor: everything in one block (no random access) ==')
run('columnar+transf THREAD-order xz-9e ALL', order_thread, N, 'col', ttext, 'xz-9e')
