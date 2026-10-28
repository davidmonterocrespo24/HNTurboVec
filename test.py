import sys, hnz
q = "esp32"
with hnz.open(r'C:\Desarrollo\Odoo\AI\HNTurboVec\work\p6\hn-2026-08.hnz', ram='2GB', preload=True) as hn:
      for mode in ('hybrid', 'keyword'):
          r = hn.search(q, k=100, mode=mode)
          print(f'\n=== {mode}: {r.timings_ms.get("total", r.timings_ms)} ms  warning={r.warning}')
          for h in r:
              t = (h.title or h.text or '').replace('\n', ' ')
              print(f'{h.rank:2} {h.date[:10]} {h.kind:7} {h.by:15} [{h.story_title or ""}]'[:110])
              print(f'     {t[:200]}  -> {h.link}')
print("END")