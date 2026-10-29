"""Measure how big all of HN is as raw JSON, the way the official API serves it.

Every row of the Parquet dump is rebuilt as one HN API item (https://github.com/HackerNews/API):
type as a string, time as Unix seconds, and empty or zero fields left out, like the API does.
The derived `words` column is not part of the API and is skipped. Nothing is written to disk; the
script adds up the UTF-8 length of each compact JSON line (one item per line, newline included).

    .venv\\Scripts\\python experiments\\measure_raw_json.py
"""
import glob
import json
import sys
import time

import pyarrow.parquet as pq

KIND = {1: 'story', 2: 'comment', 3: 'poll', 4: 'pollopt', 5: 'job'}
COLS = ['id', 'deleted', 'type', 'by', 'time', 'text', 'dead', 'parent', 'poll', 'kids', 'url',
        'score', 'title', 'parts', 'descendants']


def api_item(r):
    """One Parquet row as the API's JSON object, with the API's field order."""
    o = {}
    if r['by']:
        o['by'] = r['by']
    if r['dead']:
        o['dead'] = True
    if r['deleted']:
        o['deleted'] = True
    if r['type'] in (1, 3) or r['descendants']:
        o['descendants'] = r['descendants']
    o['id'] = r['id']
    if r['kids']:
        o['kids'] = r['kids']
    if r['parent']:
        o['parent'] = r['parent']
    if r['parts']:
        o['parts'] = r['parts']
    if r['poll']:
        o['poll'] = r['poll']
    if r['type'] in (1, 3, 4, 5) or r['score']:
        o['score'] = r['score']
    if r['text']:
        o['text'] = r['text']
    o['time'] = r['time']
    if r['title']:
        o['title'] = r['title']
    o['type'] = KIND.get(r['type'], 'story')
    if r['url']:
        o['url'] = r['url']
    return o


def main():
    files = sorted(glob.glob('data/*/*.parquet'))
    total = items = pretty = 0
    t0 = time.time()
    for i, f in enumerate(files):
        t = pq.read_table(f, columns=COLS)
        t = t.set_column(t.schema.get_field_index('time'), 'time',
                         t['time'].cast('int64'))  # ms since the epoch
        for r in t.to_pylist():
            r['time'] //= 1000
            o = api_item(r)
            total += len(json.dumps(o, ensure_ascii=False, separators=(',', ':')).encode()) + 1
            pretty += len(json.dumps(o).encode()) + 1  # Python default: ", " and ": ", \u escapes
            items += 1
        print(f'{i + 1}/{len(files)} {f}  {items:,} items  {total / 1e9:.2f} GB  '
              f'{time.time() - t0:.0f} s', file=sys.stderr, flush=True)
    print(json.dumps({'items': items, 'files': len(files),
                      'compact_utf8_bytes': total, 'compact_bytes_per_item': total / items,
                      'default_dumps_bytes': pretty, 'default_bytes_per_item': pretty / items}))


if __name__ == '__main__':
    main()
