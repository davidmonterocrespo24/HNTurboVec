//! "Spin" thread pool for LM decoding: between steps the threads spin for a few milliseconds
//! instead of going to sleep, so each parallel matmul costs ~µs of synchronization
//! (with rayon, waking the threads for every small matmul cost ~100 µs).
//! Results do not depend on how the work is split: each task writes a disjoint chunk.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

struct Job {
    f: *const (dyn Fn(usize) + Sync),
    n: usize,
    next: AtomicUsize,
    done: AtomicUsize,
    /// a panic from some task (rethrown on the thread that called `run` once all tasks finish)
    panic: Mutex<Option<Box<dyn std::any::Any + Send>>>,
}

impl Job {
    /// Runs task i and catches a possible panic. It ALWAYS counts the task as done, so `run`
    /// never waits forever and never returns while another task still uses `f`.
    fn exec(&self, i: usize) {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe { (*self.f)(i) }));
        if let Err(p) = r {
            let mut g = self.panic.lock().unwrap_or_else(|e| e.into_inner());
            if g.is_none() {
                *g = Some(p);
            }
        }
        self.done.fetch_add(1, Ordering::Release);
    }
}
unsafe impl Send for Job {}
unsafe impl Sync for Job {}

pub struct SpinPool {
    slot: Mutex<Option<Arc<Job>>>,
    gen: AtomicU64,
    cv: Condvar,
    pub threads: usize,
}

const SPIN: Duration = Duration::from_millis(3);
const SPIN_HOT: Duration = Duration::from_micros(20);

impl SpinPool {
    fn new(threads: usize) -> &'static SpinPool {
        let pool: &'static SpinPool = Box::leak(Box::new(SpinPool { slot: Mutex::new(None), gen: AtomicU64::new(0), cv: Condvar::new(), threads }));
        for _ in 1..threads {
            std::thread::spawn(move || pool.worker());
        }
        pool
    }

    fn worker(&self) {
        // the LM must compute as it did when compressing: default MXCSR on this thread
        super::fpenv::set_default();
        let mut seen = 0u64;
        loop {
            // wait for new work: spin for a while, then sleep on the condvar
            let t0 = Instant::now();
            loop {
                if self.gen.load(Ordering::Acquire) != seen {
                    break;
                }
                let el = t0.elapsed();
                if el < SPIN_HOT {
                    std::hint::spin_loop();
                } else if el < SPIN {
                    // yield the CPU if another thread or process needs it (avoids oversubscription)
                    std::thread::yield_now();
                } else {
                    let g = self.slot.lock().unwrap();
                    if self.gen.load(Ordering::Acquire) == seen {
                        let _g = self.cv.wait_timeout(g, Duration::from_millis(50)).unwrap();
                    }
                }
            }
            seen = self.gen.load(Ordering::Acquire);
            let job = self.slot.lock().unwrap().clone();
            if let Some(job) = job {
                loop {
                    let i = job.next.fetch_add(1, Ordering::AcqRel);
                    if i >= job.n {
                        break;
                    }
                    job.exec(i);
                }
            }
        }
    }

    /// Runs f(0..n) split across the pool's threads (and the caller). Blocks until done.
    pub fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        if n == 0 {
            return;
        }
        if n == 1 || self.threads <= 1 {
            for i in 0..n {
                f(i);
            }
            return;
        }
        // SAFETY: `f` lives until `done == n`; no thread calls it after that (indices >= n
        // are not run) and we wait here until all tasks finish.
        let fp: *const (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(f as *const (dyn Fn(usize) + Sync + '_)) };
        let job = Arc::new(Job { f: fp, n, next: AtomicUsize::new(0), done: AtomicUsize::new(0), panic: Mutex::new(None) });
        {
            let mut s = self.slot.lock().unwrap();
            *s = Some(job.clone());
            self.gen.fetch_add(1, Ordering::AcqRel);
        }
        self.cv.notify_all();
        loop {
            let i = job.next.fetch_add(1, Ordering::AcqRel);
            if i >= n {
                break;
            }
            job.exec(i);
        }
        let t0 = Instant::now();
        while job.done.load(Ordering::Acquire) < n {
            if t0.elapsed() < SPIN_HOT {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        let p = job.panic.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(p) = p {
            std::panic::resume_unwind(p);
        }
    }
}

pub fn pool() -> &'static SpinPool {
    static P: OnceLock<&'static SpinPool> = OnceLock::new();
    P.get_or_init(|| {
        let n = std::env::var("HNZ_LM_THREADS")
            .ok()
            .and_then(|x| x.parse::<usize>().ok())
            .filter(|&x| x > 0)
            // by default half the logical threads: on hybrid CPUs this avoids depending on the
            // slow cores (E/LP-E) and on oversubscription (measured: 6-8 >> 16 on a Core Ultra 7)
            .unwrap_or_else(|| (std::thread::available_parallelism().map(|x| x.get()).unwrap_or(4) / 2).max(1));
        SpinPool::new(n)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn runs_all_tasks_and_propagates_panics() {
        let p = pool();
        let n = AtomicUsize::new(0);
        p.run(100, &|_| {
            n.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(n.load(Ordering::Relaxed), 100);
        // a panic in a task is rethrown in the caller, without hanging
        let r = std::panic::catch_unwind(|| {
            p.run(50, &|i| {
                if i == 7 {
                    panic!("task 7");
                }
            })
        });
        assert!(r.is_err());
        // the pool still works afterwards
        let m = AtomicUsize::new(0);
        p.run(10, &|_| {
            m.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(m.load(Ordering::Relaxed), 10);
    }
}
