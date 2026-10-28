# Diagram sources

The architecture diagrams and the treemap of the block, drawn with the `diagram-design` skill in a
Hacker News skin (HN's beige paper, its orange as the one accent). `make_diagrams.py` writes each
one as a self-contained HTML page, light and dark; the README and the write-up use SVG exports of
them in `doc/img/diagram-*.svg`.

```powershell
python doc/diagrams/make_diagrams.py
foreach ($f in 'query','query-dark','build','build-dark','block','block-dark') {
  python .agents/skills/diagram-design/scripts/self_check.py doc/diagrams/$f.html
  python .agents/skills/diagram-design/scripts/export_svg.py doc/diagrams/$f.html doc/img/diagram-$f.svg
}
```

The charts (size, quality, latency) and the terminal captures come from `doc/img/make_figures.py`.
