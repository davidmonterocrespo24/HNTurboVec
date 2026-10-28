"""Builds the architecture diagrams and the treemap of the block (HTML, light and dark), in the
Hacker News skin of the diagram-design skill (.claude/skills/diagram-design, style-guide.md).

    python doc/diagrams/make_diagrams.py
Then export the SVGs for the README with the skill's helper (see doc/diagrams/README.md).
"""
import html
import os

HERE = os.path.dirname(os.path.abspath(__file__))

SKINS = {
    'light': dict(paper='#f6f6ef', ink='#222220', ink_rgb='34,34,32', muted='#5a5a55', soft='#6b6b66',
                  accent='#b84a00', tint='rgba(255,102,0,0.12)', tint_cell='rgba(255,102,0,0.16)',
                  backend='#ffffff', link='#2e5aa8'),
    'dark': dict(paper='#1c1c1a', ink='#f6f6ef', ink_rgb='246,246,239', muted='#c9c7bc', soft='#a8a69c',
                 accent='#ff7a1f', tint='rgba(255,102,0,0.14)', tint_cell='rgba(255,102,0,0.18)',
                 backend='#262624', link='#739fdf'),
}
SANS = "'Geist', system-ui, sans-serif"
MONO = "'Geist Mono', ui-monospace, monospace"
SERIF = "'Instrument Serif', 'Noto Serif', serif"
FONTS = ('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist:wght@400;500;600'
         '&family=Geist+Mono:wght@400;500;600&display=swap')


def e(s):
    return html.escape(str(s), quote=True)


def ink(k, a):
    return f'rgba({k["ink_rgb"]},{a})'


class Svg:
    def __init__(self, k):
        self.k, self.parts = k, []

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, fill, size, family=SANS, weight=None, anchor=None, spacing=None):
        a = f' font-weight="{weight}"' if weight else ''
        a += f' text-anchor="{anchor}"' if anchor else ''
        a += f' letter-spacing="{spacing}"' if spacing else ''
        self.add(f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" font-family="{family}"{a}>{e(s)}</text>')

    def line(self, x1, y1, x2, y2, accent=False):
        k = self.k
        col, m, w = (k['accent'], 'arrow-accent', 1.4) if accent else (k['muted'], 'arrow', 1.2)
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="{w}" marker-end="url(#{m})"/>')

    def path(self, d):
        self.add(f'<path d="{d}" fill="none" stroke="{self.k["muted"]}" stroke-width="1.2" marker-end="url(#arrow)"/>')

    def label_right(self, x, ymid, s, color=None):
        """An arrow label beside a vertical segment at x: an 8px gap, an opaque mask."""
        w = round(len(s) * 8 * 0.62 + 8)
        w += (-w) % 4
        self.add(f'<rect x="{x + 8}" y="{ymid - 6}" width="{w}" height="12" rx="2" fill="{self.k["paper"]}"/>')
        self.text(x + 8 + w / 2, ymid + 3, s, color or self.k['soft'], 8, MONO, anchor='middle', spacing='0.08em')

    def node(self, x, y, w, h, kind, tag, name, subs):
        k = self.k
        fill, stroke, sw = {
            'focal': (k['tint'], k['accent'], 1.2),
            'backend': (k['backend'], k['ink'], 1),
            'store': (ink(k, 0.05), k['muted'], 1),
            'input': (ink(k, 0.03), ink(k, 0.30), 1),
        }[kind]
        tag_col = k['accent'] if kind == 'focal' else k['soft']
        tw = round(len(tag) * 7 * 0.62 + 10)
        tw += (-tw) % 2
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{k["paper"]}"/>')
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        self.add(f'<rect x="{x + 8}" y="{y + 8}" width="{tw}" height="12" rx="2" fill="none" stroke="{tag_col}" stroke-opacity="0.6" stroke-width="0.8"/>')
        self.text(x + 8 + tw / 2, y + 17, tag, tag_col, 7, MONO, anchor='middle', spacing='0.08em')
        cx = x + w / 2
        top = y + h / 2 + 2 - 6 * (len(subs) - 1)
        self.text(cx, top, name, k['ink'], 12, SANS, 600, 'middle')
        for i, s in enumerate(subs):
            self.text(cx, top + 17 + i * 13, s, k['muted'], 9, MONO, anchor='middle')

    def legend(self, y, width, items, note=None):
        k = self.k
        self.add(f'<line x1="16" y1="{y}" x2="{width - 16}" y2="{y}" stroke="{ink(k, 0.10)}" stroke-width="0.8"/>')
        self.text(16, y + 16, 'LEGEND', k['muted'], 8, MONO, spacing='0.18em')
        if note:
            self.text(width - 16, y + 16, note, k['muted'], 8, MONO, anchor='end', spacing='0.06em')
        x = 16
        for kind, label in items:
            fill, stroke = {'focal': (k['tint'], k['accent']), 'backend': (k['backend'], k['ink']),
                            'store': (ink(k, 0.05), k['muted']), 'input': (ink(k, 0.03), ink(k, 0.30)),
                            'cell': (ink(k, 0.14), ink(k, 0.30)), 'cell-focal': (k['tint_cell'], k['accent'])}[kind]
            self.add(f'<rect x="{x}" y="{y + 26}" width="14" height="10" rx="2" fill="{fill}" stroke="{stroke}" stroke-width="1"/>')
            self.text(x + 20, y + 34, label, k['muted'], 8.5)
            x += 20 + round(len(label) * 8.5 * 0.55) + 32

    def render(self, slug, w, h, title, desc):
        k = self.k
        defs = ''.join(
            f'<marker id="{i}" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{c}"/></marker>'
            for i, c in (('arrow', k['muted']), ('arrow-accent', k['accent']), ('arrow-link', k['link'])))
        body = '\n        '.join(self.parts)
        return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="{slug}-title {slug}-desc">\n'
                f'        <title id="{slug}-title">{e(title)}</title>\n        <desc id="{slug}-desc">{e(desc)}</desc>\n'
                f'        <defs>{defs}</defs>\n        <rect width="100%" height="100%" fill="{k["paper"]}"/>\n        {body}\n      </svg>')


def page(k, slug, w, eyebrow, h1, svg):
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{e(h1)}</title>
  <link href="{e(FONTS)}" rel="stylesheet">
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    :root {{ --color-paper: {k["paper"]}; --color-ink: {k["ink"]}; --color-muted: {k["muted"]}; --color-accent: {k["accent"]};
            --font-sans: {SANS}; --font-serif: {SERIF}; --font-mono: {MONO}; }}
    body {{ font-family: var(--font-sans); background: var(--color-paper); color: var(--color-ink); min-height: 100vh;
           display: flex; align-items: center; justify-content: center; padding: 3rem 2rem; }}
    .frame {{ max-width: 1200px; width: 100%; min-width: 0; }}
    .diagram-container {{ width: 100%; overflow-x: auto; }}
    .eyebrow {{ font-family: var(--font-mono); font-size: 0.66rem; font-weight: 500; letter-spacing: 0.18em;
               text-transform: uppercase; color: var(--color-muted); margin-bottom: 0.5rem; }}
    h1 {{ font-family: var(--font-serif); font-size: clamp(1.5rem, 2.4vw + 0.75rem, 2rem); font-weight: 400;
         letter-spacing: -0.02em; line-height: 1.15; color: var(--color-ink); margin-bottom: 1.5rem; }}
    svg {{ width: 100%; min-width: {w}px; display: block; }}
    @media print {{ .diagram-container {{ overflow-x: visible; }} svg {{ min-width: 0; }} }}
  </style>
</head>
<body>
  <div class="frame">
    <p class="eyebrow">{e(eyebrow)}</p>
    <h1>{e(h1)}</h1>
    <div class="diagram-container">
      {svg}
    </div>
  </div>
</body>
</html>
'''


def query(k, slug):
    s, W, H = Svg(k), 960, 416
    xs = [16 + i * 160 for i in range(6)]
    y, h, w = 40, 96, 128
    for i in range(5):                                   # the flow, left to right (arrows before boxes)
        s.line(xs[i] + w, y + h // 2, xs[i + 1], y + h // 2)
    for cx, lab in ((400, 'VECTORS, WORDS'), (560, 'LM-CODED TEXT'), (880, 'METADATA')):
        s.line(cx, 248, cx, y + h)                       # the block feeds three steps
        s.label_right(cx, 192, lab)
    s.node(xs[0], y, w, h, 'input', 'IN', 'Query', ['text and filters', 'dates · type · by'])
    s.node(xs[1], y, w, h, 'backend', 'CPU', 'Embed query', ['EmbeddingGemma 2', 'int8 · 768 dims'])
    s.node(xs[2], y, w, h, 'backend', 'CPU', 'Stage 1', ['top 80 by vector', 'top 20 by BM25'])
    s.node(xs[3], y, w, h, 'focal', 'LM', 'Decode prefixes', ['first 128 tokens', 'batched, shared'])
    s.node(xs[4], y, w, h, 'backend', 'CPU', 'Rerank', ['BM25 of every word', 're-embed top 50'])
    s.node(xs[5], y, w, h, 'backend', 'OUT', 'Results', ['k items, verbatim', '~2.3 s cold'])
    s.node(336, 248, 608, 80, 'store', 'FILE', 'The block · hn.hnzb', ['5.07 GB, read-only · every part hashed (XXH3-128)'])
    s.legend(364, W, [('focal', 'Most of the time goes here'), ('backend', 'A step on the CPU'),
                      ('store', 'Read from the block'), ('input', 'Input')])
    return s.render(slug, W, H, 'How a query runs',
                    'A hybrid search embeds the query, takes 100 candidates from the 256-bit vectors and the rare-term '
                    'index, decodes the first 128 tokens of each with the language model, reranks them and returns k items.')


def build(k, slug):
    s, W, H = Svg(k), 960, 456
    # fan out from the input's right edge, fan in to pack's left edge (ports 22px apart)
    s.path('M176,198 H200 Q208,198 208,190 V116 Q208,108 216,108 H240')
    s.line(176, 220, 240, 220)
    s.path('M176,242 H200 Q208,242 208,250 V324 Q208,332 216,332 H240')
    s.path('M480,108 H504 Q512,108 512,116 V190 Q512,198 520,198 H544')
    s.line(480, 220, 544, 220)
    s.path('M480,332 H504 Q512,332 512,324 V250 Q512,242 520,242 H544')
    s.line(704, 220, 768, 220, accent=True)
    s.node(16, 176, 160, 88, 'input', 'IN', 'Parquet files', ['239 months · 12.4 GB', 'open-index/hacker-news'])
    s.node(240, 64, 240, 88, 'focal', 'CPU', 'LM compression', ['pythia-70m trained on HN', 'arithmetic coding · ~7 days'])
    s.node(240, 176, 240, 88, 'backend', 'iGPU', 'Document vectors', ['EmbeddingGemma 2 → 256 bits', '~4.5 days, alongside'])
    s.node(240, 288, 240, 88, 'backend', 'CPU', 'Word index and metadata', ['BM25 of rare terms', 'time-ordered columns'])
    s.node(544, 176, 160, 88, 'backend', 'CPU', 'hnz pack', ['month by month', '~1 h · 3.2 GB RAM'])
    s.node(768, 176, 176, 88, 'store', 'FILE', 'The block', ['hn.hnzb 5.07 GB', '+ embedder 0.28 GB'])
    s.legend(404, W, [('focal', 'The slow part: ~7 days of CPU'), ('backend', 'A build step'),
                      ('store', 'The result'), ('input', 'Input')])
    return s.render(slug, W, H, 'How the block is built',
                    'The monthly Parquet files are compressed by the language model, embedded on the integrated GPU and '
                    'indexed in parallel, then packed month by month into one 5.07 GB block plus the query embedder.')


PARTS = [  # (name, detail, bytes, focal)
    ('Text', 'coded by the LM · 59 B per item', 2_813_292_945, True),
    ('Semantic vectors', '256 bits per item', 1_516_298_016, False),
    ('Word index', '4.2 M rare terms', 365_013_098, False),
    ('Models', 'embedder + LM', 352_112_618, False),
    ('Metadata', '+ 12 MB directory', 284_744_791 + 11_671_963, False),
]


def squarify():
    """The cells on the 4px grid: text as a column, vectors on top of the rest, three cells below.
    Searches the free sizes for the smallest worst relative area error."""
    total = sum(p[2] for p in PARTS)
    share = [p[2] / total for p in PARTS]
    best = None
    for tw in range(460, 504, 4):
        for vh in range(212, 244, 4):
            rx, rw = 40 + tw + 4, 956 - (40 + tw + 4)
            bh = 380 - vh - 4
            rest = sum(share[2:])
            avail = rw - 8
            ws = [round(avail * sh / rest / 4) * 4 for sh in share[2:]]
            ws[-1] = avail - sum(ws[:-1])
            if ws[-1] % 4:
                continue
            cells = [(40, 40, tw, 380), (rx, 40, rw, vh)]
            x = rx
            for wd in ws:
                cells.append((x, 40 + vh + 4, wd, bh))
                x += wd + 4
            areas = [c[2] * c[3] for c in cells]
            err = max(abs(a / sum(areas) - sh) / sh for a, sh in zip(areas, share))
            if best is None or err < best[0]:
                best = (err, cells)
    return best, share


def block(k, slug):
    s, W, H = Svg(k), 1000, 500
    (err, cells), share = squarify()
    ramp = [None, 0.14, 0.10, 0.07, 0.04]
    for i, ((name, detail, b, focal), (x, y, w, h)) in enumerate(zip(PARTS, cells)):
        fill = k['tint_cell'] if focal else ink(k, ramp[i])
        stroke, sw = (k['accent'], 1.5) if focal else (ink(k, 0.30), 1)
        s.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="2" fill="{k["paper"]}"/>')
        s.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="2" data-share="{100 * share[i]:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        s.text(x + 16, y + 28, name, k['ink'], 14 if i < 2 else 12, SANS, 600)
        s.text(x + 16, y + 46, f'{b / 1e9:.2f} GB · {100 * share[i]:.0f}%', k['muted'], 9, MONO)
        s.text(x + 16, y + 60, detail, k['muted'], 9, MONO)
    s.add(f'<line x1="40" y1="462" x2="960" y2="462" stroke="{ink(k, 0.10)}" stroke-width="0.8"/>')
    s.text(40, 478, 'LEGEND', k['muted'], 8, MONO, spacing='0.18em')
    s.text(960, 478, 'AREA = BYTES · HN-2026-08.HNZ · 5,343,133,431 BYTES · OCT 2026', k['muted'], 8, MONO, anchor='end', spacing='0.06em')
    s.add(f'<rect x="40" y="488" width="16" height="10" rx="2" fill="{k["tint_cell"]}" stroke="{k["accent"]}" stroke-width="1"/>')
    s.text(64, 497, 'Text · the focal part, more than half the block', k['muted'], 8.5)
    s.add(f'<rect x="300" y="488" width="16" height="10" rx="2" fill="{ink(k, 0.14)}" stroke="{ink(k, 0.30)}" stroke-width="1"/>')
    s.text(324, 497, 'Other parts · stronger contrast is larger', k['muted'], 8.5)
    print(f'  treemap worst relative area error: {100 * err:.1f}%')
    return s.render(slug, W, H, 'Inside the 5.34 GB block',
                    'Treemap of the block by bytes: the LM-coded text is 53 percent, the semantic vectors 28 percent, '
                    'the word index, the models and the metadata 5 to 7 percent each.')


def main():
    for name, fn, w, eyebrow, h1 in (
            ('query', query, 960, 'Architecture · hnz', 'How a query runs'),
            ('build', build, 960, 'Architecture · hnz', 'How the block is built'),
            ('block', block, 1000, 'Treemap · hnz', 'Inside the 5.34 GB block')):
        for mode, k in SKINS.items():
            slug = f'{name}-dark' if mode == 'dark' else name
            out = os.path.join(HERE, f'{slug}.html')
            open(out, 'w', encoding='utf-8', newline='\n').write(page(k, slug, w, eyebrow, h1, fn(k, slug)))
            print(out)


if __name__ == '__main__':
    main()
