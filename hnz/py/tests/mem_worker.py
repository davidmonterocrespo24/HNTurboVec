"""Load generator for the RAM tests (run in a fresh process so the baseline is clean).

    python mem_worker.py <archive> <ram|none> <threads> <seconds>

Opens the archive with `ram`, runs mixed queries from `threads` threads for `seconds`, and prints
one JSON line: the process working set before opening, its peak (as the OS reports it), hnz's own
ledger and the number of operations and errors.
"""
import ctypes
import ctypes.wintypes as w
import json
import random
import sys
import threading
import time


class _PMC(ctypes.Structure):
    _fields_ = [('cb', w.DWORD), ('PageFaultCount', w.DWORD), ('PeakWorkingSetSize', ctypes.c_size_t),
                ('WorkingSetSize', ctypes.c_size_t), ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPagedPoolUsage', ctypes.c_size_t), ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                ('QuotaNonPagedPoolUsage', ctypes.c_size_t), ('PagefileUsage', ctypes.c_size_t),
                ('PeakPagefileUsage', ctypes.c_size_t), ('PrivateUsage', ctypes.c_size_t)]


_k32 = ctypes.WinDLL('kernel32')
_k32.GetCurrentProcess.restype = w.HANDLE
_k32.K32GetProcessMemoryInfo.argtypes = [w.HANDLE, ctypes.POINTER(_PMC), w.DWORD]


def process_memory():
    c = _PMC()
    c.cb = ctypes.sizeof(_PMC)
    _k32.K32GetProcessMemoryInfo(_k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return {'working_set': c.WorkingSetSize, 'peak_working_set': c.PeakWorkingSetSize, 'private': c.PrivateUsage}


QUERIES = ['how do I deal with burnout as a software engineer', 'rust async runtime', 'is the iPad going to kill netbooks',
           'startup funding advice', 'postgres vs mysql', 'best text editor for programmers', 'why is my python code slow',
           'show hn', 'remote work productivity', 'the future of javascript frameworks', 'learning machine learning',
           'how to price a saas product', 'linux on the desktop', 'privacy and google', 'hiring engineers is hard']


def main():
    root, ram, threads, seconds = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
    import hnz
    base = process_memory()
    a = hnz.open(root, ram=None if ram == 'none' else int(ram), preload=True)
    stop = time.time() + seconds
    ops, errors, busy = [0], [], [0]
    lock = threading.Lock()

    def client(seed):
        rnd = random.Random(seed)
        while time.time() < stop:
            q = rnd.choice(QUERIES) + ('' if rnd.random() < 0.5 else f' {rnd.randint(1, 999)}')
            kind = rnd.random()
            try:
                if kind < 0.35:
                    a.search(q, k=10)
                elif kind < 0.55:
                    a.search(q, k=rnd.choice([10, 40]), mode='keyword', text='full')
                elif kind < 0.7:
                    a.search(q, k=20, since=str(rnd.choice([2010, 2015, 2024])), until=str(rnd.choice([2010, 2015, 2024])))
                elif kind < 0.8:
                    a.search(types=rnd.choice(['story', 'comment', 'job']), k=30)
                else:
                    a.get(rnd.choice([9747338, 1424000, 1430000, 9700000, 38900000, 39100000]) + rnd.randint(0, 5000))
                with lock:
                    ops[0] += 1
            except hnz.BusyError:
                with lock:
                    busy[0] += 1
            except Exception as e:  # noqa: BLE001
                with lock:
                    errors.append(repr(e))

    ts = [threading.Thread(target=client, args=(i,)) for i in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    m = a.memory()
    print(json.dumps({'base': base, 'end': process_memory(), 'budget': m['budget'], 'plan': m['plan'],
                      'parts': m['engine']['parts'], 'embedder': m['embedder'], 'admission': m['admission'],
                      'ops': ops[0], 'busy': busy[0], 'errors': errors[:5]}))


if __name__ == '__main__':
    main()
