"""Compression with blocks sorted by topic (tags from bench_tags.py) vs by thread / by id.
Usage: python experiments/bench_store_topic.py <block_size>"""
import os, sys
import numpy as np
BLK = int(sys.argv[1]); sys.argv = sys.argv[:1]  # bench_store reads sys.argv[1] as the parquet path
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bench_store.py'), encoding='utf-8').read()
exec(src.split("texts_raw = cols['text']")[0])
order_topic = np.load('emb/order_topic.npy').tolist()
order_thr_topic = np.load('emb/order_thread_topic.npy').tolist()
run('id-order', order_id, BLK, 'col', ttext, 'zstd-19-112')
run('THREAD-order', order_thread, BLK, 'col', ttext, 'zstd-19-112')
run('TOPIC-order', order_topic, BLK, 'col', ttext, 'zstd-19-112')
run('THREAD-order grouped by TOPIC', order_thr_topic, BLK, 'col', ttext, 'zstd-19-112')
