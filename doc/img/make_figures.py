"""Builds the figures of the README and the write-up (SVG, a light and a dark version of each chart).

Standard library only. The numbers are the measured ones of doc/BITACORA.md (E00, E01, E34, E36);
the terminal captures are rendered from the real outputs (logs/p6/*.log, accept_full.txt).

    python doc/img/make_figures.py
"""
import html
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SANS = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, 'Cascadia Mono', Consolas, Menlo, monospace"

THEMES = {  # Hacker News: its beige page, its orange header, its grey subtext
    'light': dict(surface='#f6f6ef', ink='#222220', ink2='#5a5a55', muted='#6b6b66', grid='#e3e2d6',
                  axis='#c9c7bc', s1='#ff6600', s2='#828282', neutral='#d3d1c4', border='rgba(34,34,32,0.10)'),
    'dark': dict(surface='#1c1c1a', ink='#f6f6ef', ink2='#c9c7bc', muted='#a8a69c', grid='#2e2e2b',
                 axis='#45443f', s1='#ff6600', s2='#8a8880', neutral='#4a4943', border='rgba(246,246,239,0.10)'),
}

def esc(s):
    return html.escape(str(s), quote=True)


def svg(w, h, body, t, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'role="img" aria-label="{esc(title)}">\n<title>{esc(title)}</title>\n'
            f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="10" fill="{t["surface"]}" stroke="{t["border"]}"/>\n'
            f'{body}</svg>\n')


def text(x, y, s, fill, size=13, weight=400, anchor='start', family=SANS):
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{esc(s)}</text>\n')


def hbar(x, y, w, h, fill, r=4):
    """A horizontal bar from x (square end) to x+w (4px rounded data end)."""
    if w <= 0:
        return ''
    r = min(r, w, h / 2)
    return (f'<path d="M{x:.1f},{y:.1f} H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} '
            f'V{y + h - r:.1f} Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f} H{x:.1f} Z" fill="{fill}"/>\n')


def vbar(x, y0, w, h, fill, r=4):
    """A column from the baseline y0 up by h (rounded top)."""
    if h <= 0:
        return ''
    r = min(r, h, w / 2)
    y = y0 - h
    return (f'<path d="M{x:.1f},{y0:.1f} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} H{x + w - r:.1f} '
            f'Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} V{y0:.1f} Z" fill="{fill}"/>\n')


def header(t, title, subtitle, w):
    return text(24, 34, title, t['ink'], 17, 600) + text(24, 56, subtitle, t['ink2'], 13)


def bars_chart(t, title, subtitle, rows, unit_fmt, xmax, ticks, note=None, w=860, label_w=300):
    """Horizontal bars: rows = (label, sublabel, value, emphasized)."""
    top, row_h, bar_h = 84, 46, 22
    plot_x, plot_w = label_w, w - label_w - 90
    h = top + row_h * len(rows) + 40 + (22 if note else 0)
    b = header(t, title, subtitle, w)
    base_y = top + row_h * len(rows)
    for v in ticks:  # recessive hairline grid
        x = plot_x + plot_w * v / xmax
        b += f'<line x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{base_y}" stroke="{t["grid"]}" stroke-width="1"/>\n'
        b += text(x, base_y + 18, unit_fmt(v, tick=True), t['muted'], 12, anchor='middle')
    b += f'<line x1="{plot_x}" y1="{top - 6}" x2="{plot_x}" y2="{base_y}" stroke="{t["axis"]}" stroke-width="1"/>\n'
    for i, (label, sub, v, emph) in enumerate(rows):
        y = top + i * row_h
        b += text(plot_x - 14, y + 15, label, t['ink'], 13.5, 600 if emph else 400, 'end')
        if sub:
            b += text(plot_x - 14, y + 32, sub, t['muted'], 12, anchor='end')
        bw = plot_w * v / xmax
        b += hbar(plot_x, y + 4, bw, bar_h, t['s1'] if emph else t['neutral'])
        b += text(plot_x + bw + 8, y + 20, unit_fmt(v), t['ink'], 13.5, 600 if emph else 400)
    if note:
        b += text(24, h - 16, note, t['muted'], 12)
    return svg(w, h, b, t, title)


def size_chart(t):
    rows = [
        ('Raw JSON (HN API)', 'text and metadata, no search', 24.0, False),
        ('Parquet dump', 'open-index/hacker-news, no search', 12.42, False),
        ('zstd-19 + dictionary, per item', 'random access, no search', 9.1, False),
        ('hnz v1 segments', 'zstd + LM tiers, hybrid search', 10.4, False),
        ('hnz block', 'LM-compressed, hybrid search, lossless', 5.34, True),
    ]
    return bars_chart(t, 'All of Hacker News on disk', '49.3 M items, 2006-10 to 2026-08 · GB (10⁹ bytes)', rows,
                      lambda v, tick=False: f'{v:g}' if tick else f'{v:.2f} GB' if v < 6 else f'{v:.1f} GB',
                      26, [0, 5, 10, 15, 20, 25],
                      note='The hnz block also holds the semantic index, the word index, the LM and the query embedder.')


def block_chart(t):
    rows = [
        ('Text', 'arithmetic-coded with an LM trained on HN', 2.813, True),
        ('Semantic vectors', '256-bit codes, 47.4 M items', 1.516, True),
        ('Word index', 'BM25 postings of 4.2 M rare terms', 0.365, True),
        ('Models', 'query embedder 0.28 + LM 0.08', 0.352, True),
        ('Metadata', 'times, authors, scores, threads', 0.285, True),
        ('Other', 'directories, fallbacks, padding', 0.012, True),
    ]
    total = sum(r[2] for r in rows)
    return bars_chart(t, 'What is inside the 5.34 GB', 'hn.hnzb (5.07 GB) + the embedder files (0.28 GB) · GB', rows,
                      lambda v, tick=False: f'{v:g}' if tick else f'{v:.2f} GB · {100 * v / total:.0f} %' if v >= 0.05 else f'{v * 1000:.0f} MB',
                      3.2, [0, 1, 2, 3], w=860, label_w=330)


def quality_chart(t, w=860):
    sets = ['2010-06', '2015-06', '2024-01', 'dev 2010-06', 'dev 2015-06', 'dev 2024-01']
    hyb = [0.312, 0.341, 0.309, 0.356, 0.336, 0.277]
    kw = [0.251, 0.266, 0.260, 0.287, 0.263, 0.220]
    title = 'Search quality on the full block'
    top, plot_h, left, right = 110, 240, 60, 24
    h = top + plot_h + 70
    b = header(t, title, 'P@10 · 300 queries per set, each searched within its month among 49.3 M items', w)
    # legend (two series): swatch + text-token label
    lx = 24
    for name, col in (('Hybrid search, reranked (the default)', t['s1']), ('Keyword search only', t['s2'])):
        b += f'<rect x="{lx}" y="74" width="12" height="12" rx="3" fill="{col}"/>\n' + text(lx + 18, 85, name, t['ink2'], 13)
        lx += 18 + len(name) * 7.2 + 26
    ymax, base = 0.4, top + plot_h
    pw = w - left - right
    for v in (0, 0.1, 0.2, 0.3, 0.4):
        y = base - plot_h * v / ymax
        b += f'<line x1="{left}" y1="{y:.1f}" x2="{w - right}" y2="{y:.1f}" stroke="{t["grid"] if v else t["axis"]}" stroke-width="1"/>\n'
        b += text(left - 8, y + 4, f'{v:.1f}', t['muted'], 12, anchor='end')
    gw = pw / len(sets)
    bw = min(24, gw * 0.3)
    for i, s in enumerate(sets):
        cx = left + gw * i + gw / 2
        x1, x2 = cx - bw - 1, cx + 1   # a 2px surface gap between the pair
        h1, h2 = plot_h * hyb[i] / ymax, plot_h * kw[i] / ymax
        b += vbar(x1, base, bw, h1, t['s1']) + vbar(x2, base, bw, h2, t['s2'])
        b += text(x1 + bw / 2, base - h1 - 6, f'{hyb[i]:.2f}', t['ink'], 11.5, 600, 'middle')
        b += text(x2 + bw / 2, base - h2 - 6, f'{kw[i]:.2f}', t['ink2'], 11.5, 400, 'middle')
        b += text(cx, base + 20, s, t['ink2'], 12.5, anchor='middle')
    b += text(24, h - 16, 'Relevant = the comments of the query’s own thread. Without the month filter, threads on the same subject '
              'from other years count as misses (P@10 ≈ 0.05).', t['muted'], 11.5)
    return svg(w, h, b, t, title)


def latency_chart(t):
    rows = [
        ('get() an item', 'verbatim text, LM-decoded', 0.08, True),
        ('Keyword search', 'mean of 1,800 eval queries', 0.22, True),
        ('Hybrid, first stage only', 'rerank=False', 0.40, True),
        ('Hybrid, the same query again', 'decoded prefixes reused', 1.8, True),
        ('Hybrid, reranked, cold', 'mean of 1,800 eval queries', 2.31, True),
        ('Open the block', 'with the query embedder loaded', 4.0, True),
    ]
    return bars_chart(t, 'Latency on the full block', 'seconds · laptop CPU (Intel Core Ultra 7 265H, 16 threads), ram="2GB"', rows,
                      lambda v, tick=False: f'{v:g} s' if tick else (f'{v * 1000:.0f} ms' if v < 1 else f'{v:.1f} s'),
                      4.5, [0, 1, 2, 3, 4], label_w=300)


# --- terminal captures -------------------------------------------------------------------------------
TERM = dict(bg='#14161a', bar='#22252b', fg='#d6d8dc', dim='#7d828c', ok='#4cc38a', prompt='#ff6600', warn='#f0b84d')


def terminal(title, lines, w=920):
    """lines: (kind, text) with kind in cmd / out / ok / dim / blank."""
    lh, pad_top = 19, 52
    h = pad_top + lh * len(lines) + 20
    b = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="{esc(title)}">\n'
         f'<title>{esc(title)}</title>\n<rect width="{w}" height="{h}" rx="10" fill="{TERM["bg"]}"/>\n'
         f'<path d="M0,10 Q0,0 10,0 H{w - 10} Q{w},0 {w},10 V30 H0 Z" fill="{TERM["bar"]}"/>\n')
    for i, c in enumerate(('#ff5f57', '#febc2e', '#28c840')):
        b += f'<circle cx="{18 + i * 18}" cy="15" r="5.5" fill="{c}"/>\n'
    b += text(w / 2, 20, title, TERM['dim'], 12, anchor='middle')
    for i, (kind, s) in enumerate(lines):
        y = pad_top + i * lh
        if kind == 'cmd':
            b += (f'<text x="18" y="{y}" font-family="{MONO}" font-size="13" xml:space="preserve">'
                  f'<tspan fill="{TERM["prompt"]}">$ </tspan><tspan fill="{TERM["fg"]}">{esc(s)}</tspan></text>\n')
        elif kind == 'pass':
            name, _, detail = s.partition('  (')
            b += (f'<text x="18" y="{y}" font-family="{MONO}" font-size="13" xml:space="preserve">'
                  f'<tspan fill="{TERM["ok"]}">PASS</tspan><tspan fill="{TERM["fg"]}">  {esc(name[4:].strip())}</tspan>'
                  + (f'<tspan fill="{TERM["dim"]}">  ({esc(detail)}</tspan>' if detail else '') + '</text>\n')
        elif kind != 'blank':
            col = {'out': TERM['fg'], 'ok': TERM['ok'], 'dim': TERM['dim']}[kind]
            b += f'<text x="18" y="{y}" fill="{col}" font-family="{MONO}" font-size="13" xml:space="preserve">{esc(s)}</text>\n'
    return b + '</svg>\n'


def shorten(s, n=124):
    return s if len(s) <= n else s[:n - 1] + '…'


def accept_capture():
    raw = open(os.path.join(HERE, 'accept_full.txt'), encoding='utf-8-sig').read().splitlines()
    lines = [('cmd', 'python scripts/accept_full.py --root work/p6/hn-2026-08.hnz --ram 2GB')]
    for s in raw:
        s = re.sub(r'from \S+', 'from site-packages', s) if s.startswith('hnz ') else s
        s = re.sub(r'\(\S*python\.exe -m hnz\)', '(python -m hnz)', s)
        s = re.sub(r'http://127\.0\.0\.1:\d+/ \(Ctrl\+C to stop\)', 'http://127.0.0.1:…/', s)
        if not s.strip():
            lines.append(('blank', ''))
        elif s.startswith('PASS'):
            lines.append(('pass', shorten(s)))
        elif s.startswith('ALL PASSED'):
            lines.append(('ok', s))
        else:
            lines.append(('dim', shorten(s)))
    return terminal('acceptance on the block of all of HN (the installed wheel, public API only)', lines, w=1010)


def tests_capture():
    rust = [l for l in open(os.path.join(ROOT, 'logs', 'p6', 'rust-tests.log'), encoding='utf-8', errors='replace').read().splitlines()
            if l.startswith('test ')]
    last = max(i for i, l in enumerate(rust) if l.startswith('test result'))
    run = rust[last - 35:last + 1]
    pick = [l for l in run if any(k in l for k in ('corruption_matrix', 'kernels_agree_exactly', 'roundtrip_and_reproducible',
                                                   'admission_is_fifo', 'request_heads', 'roundtrip_random_distributions'))]
    lines = [('cmd', 'cargo test --release --lib')] + [('dim', l) for l in pick] + [('dim', '…'), ('ok', run[-1])]
    lines += [('blank', ''), ('cmd', 'python -m pytest hnz/py/tests -q'),
              ('dim', '.' * 72 + ' [ 98%]'), ('dim', '.' + ' ' * 71 + ' [100%]'), ('ok', '73 passed in 1130.23s (0:18:50)')]
    lines += [('blank', ''), ('cmd', 'hnz verify-block hn-2026-08.hnz --level deep --decode 0.005'),
              ('dim', '  "deep: metadata"      rows 49,324,111   late 309,761'),
              ('dim', '  "deep: 2006-10" … "deep: 2026-08"   237 months against their Parquet sources'),
              ('dim', '  "deep: items"         246,546 records decoded with the LM, 15,748,814 kids lists'),
              ('dim', '  "deep: word index"    4,223,195 terms'),
              ('ok', '  "differences": 0,  "ok": true,  "seconds": 34871.89'),
              ('dim', '  "snapshot": "ef11713b247a66bde569c8a6d7a727ee"')]
    return terminal('tests and the deep verification (condensed from logs/p6/*.log)', lines)


def main():
    charts = {'size': size_chart, 'quality': quality_chart, 'latency': latency_chart}
    for name, fn in charts.items():
        for mode, t in THEMES.items():
            p = os.path.join(HERE, f'{name}{"-dark" if mode == "dark" else ""}.svg')
            open(p, 'w', encoding='utf-8', newline='\n').write(fn(t))
    open(os.path.join(HERE, 'term-accept.svg'), 'w', encoding='utf-8', newline='\n').write(accept_capture())
    open(os.path.join(HERE, 'term-tests.svg'), 'w', encoding='utf-8', newline='\n').write(tests_capture())
    print('\n'.join(sorted(f for f in os.listdir(HERE) if f.endswith('.svg'))))


if __name__ == '__main__':
    main()
