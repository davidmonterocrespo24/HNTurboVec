"""Size of an inverted index (keywords / BM25) for HN, with different posting list encodings.
docID = position in thread order (the same blocks as the text store)."""
import re, html, time, collections
import numpy as np, pyarrow.parquet as pq, zstandard as zstd

t = pq.read_table('data/2024/2024-01.parquet', columns=['id', 'parent', 'title', 'text', 'url'])
ids, par, title, text, url = (t[c].to_pylist() for c in ['id', 'parent', 'title', 'text', 'url'])
N = len(ids)
idx_of = {i: k for k, i in enumerate(ids)}
root = []
for k in range(N):
    p, cur = par[k], k
    while p and p in idx_of:
        cur = idx_of[p]; p = par[cur]
    root.append(ids[cur] if not p else p)
order = sorted(range(N), key=lambda k: (root[k], ids[k]))

TOK = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
post = collections.defaultdict(list)  # term -> [(docid, tf)]
ntok = 0
t0 = time.time()
for doc, k in enumerate(order):
    s = html.unescape(re.sub(r'<[^>]+>', ' ', title[k] + ' ' + text[k])).lower()
    c = collections.Counter(TOK.findall(s))
    ntok += sum(c.values())
    for w, f in c.items():
        post[w].append((doc, f))
npost = sum(len(v) for v in post.values())
print(f'{N:,} docs, {ntok/1e6:.1f}M tokens, {len(post):,} terms, {npost/1e6:.2f}M postings ({npost/N:.0f}/doc)  [{time.time()-t0:.0f}s]')

def varint(n, out):
    while n >= 0x80:
        out.append((n & 0x7f) | 0x80); n >>= 7
    out.append(n)

def ef_bits(n, u):
    # Elias-Fano: n*(2 + ceil(log2(u/n))) bits
    return n * (2 + max(0, int(np.ceil(np.log2(max(u / n, 1))))))

def size(terms, with_tf, codec):
    tot = 0
    vocab = '\n'.join(terms).encode()
    tot += len(zstd.ZstdCompressor(level=19).compress(vocab))  # term dictionary
    if codec == 'varint+zstd':
        buf = bytearray()
        for w in terms:
            prev = -1
            for d, f in post[w]:
                varint(d - prev - 1, buf); prev = d
                if with_tf: varint(f - 1, buf)
        tot += len(zstd.ZstdCompressor(level=19).compress(bytes(buf)))
    elif codec == 'elias-fano':
        bits = 0
        for w in terms:
            n = len(post[w]); bits += ef_bits(n, N) + 32  # +offset per term
            if with_tf:
                bits += sum(2 * int(np.log2(f)) + 1 for _, f in post[w])  # gamma
        tot += bits // 8
    tot += 4 * len(terms)  # pointer to each list
    return tot

terms_all = sorted(post)
df = {w: len(post[w]) for w in terms_all}
stop = set(sorted(terms_all, key=lambda w: -df[w])[:200])
terms_nostop = [w for w in terms_all if w not in stop]
print(f'top-200 terms (stopwords) = {sum(df[w] for w in stop)/npost:.0%} of the postings')
print(f'{"variant":<46}{"B/doc":>8}{"49.3M":>10}')
for lab, terms in [('all terms', terms_all), ('without 200 stopwords', terms_nostop)]:
    for with_tf in (True, False):
        for codec in ('varint+zstd', 'elias-fano'):
            s = size(terms, with_tf, codec)
            print(f'{lab+(" +tf" if with_tf else " boolean")+" "+codec:<46}{s/N:>8.1f}{s/N*49.3e6/1e9:>8.2f}GB', flush=True)

# reference: a real tantivy index (Lucene-like engine), without storing text
try:
    import tantivy, os, shutil
    shutil.rmtree('tantivy_idx', ignore_errors=True); os.makedirs('tantivy_idx')
    sb = tantivy.SchemaBuilder()
    sb.add_text_field('body', stored=False, index_option='freq')
    sb.add_unsigned_field('doc', stored=False, indexed=False, fast=False)
    ix = tantivy.Index(sb.build(), path='tantivy_idx')
    w = ix.writer(heap_size=500_000_000)
    for doc, k in enumerate(order):
        w.add_document(tantivy.Document(body=html.unescape(re.sub(r'<[^>]+>', ' ', title[k] + ' ' + text[k]))))
    w.commit(); w.wait_merging_threads()
    sz = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk('tantivy_idx') for f in fs)
    print(f'{"tantivy (real BM25, freq, no positions)":<46}{sz/N:>8.1f}{sz/N*49.3e6/1e9:>8.2f}GB')
except Exception as e:
    print('tantivy failed:', e)
