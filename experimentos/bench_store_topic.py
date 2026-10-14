"""Compresion con bloques ordenados por tema (etiquetas de bench_tags.py) vs por hilo / por id.
Uso: python experimentos/bench_store_topic.py <tamano_bloque>"""
import os, sys
import numpy as np
BLK = int(sys.argv[1]); sys.argv = sys.argv[:1]  # bench_store lee sys.argv[1] como ruta del parquet
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bench_store.py'), encoding='utf-8').read()
exec(src.split("texts_raw = cols['text']")[0])
order_topic = np.load('emb/order_topic.npy').tolist()
order_thr_topic = np.load('emb/order_thread_topic.npy').tolist()
run('orden-id', order_id, BLK, 'col', ttext, 'zstd-19-112')
run('orden-HILO', order_thread, BLK, 'col', ttext, 'zstd-19-112')
run('orden-TEMA', order_topic, BLK, 'col', ttext, 'zstd-19-112')
run('orden-HILO agrupado por TEMA', order_thr_topic, BLK, 'col', ttext, 'zstd-19-112')
