//! Pool de hilos "spin" para la decodificación del LM: entre pasos los hilos giran unos
//! milisegundos en vez de dormirse, así cada matmul paralela cuesta ~µs de sincronización
//! (con rayon, despertar hilos por cada matmul pequeña costaba ~100 µs).
//! Los resultados no dependen del reparto: cada tarea escribe un trozo disjunto.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

struct Job {
    f: *const (dyn Fn(usize) + Sync),
    n: usize,
    next: AtomicUsize,
    done: AtomicUsize,
    /// pánico de alguna tarea (se relanza en el hilo que llamó a `run` cuando todas terminan)
    panic: Mutex<Option<Box<dyn std::any::Any + Send>>>,
}

impl Job {
    /// Ejecuta la tarea i capturando un posible pánico; SIEMPRE cuenta la tarea como terminada,
    /// así `run` nunca espera para siempre ni retorna mientras otra tarea aún usa `f`.
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
            // esperar trabajo nuevo: girar un rato y luego dormir en la condvar
            let t0 = Instant::now();
            loop {
                if self.gen.load(Ordering::Acquire) != seen {
                    break;
                }
                let el = t0.elapsed();
                if el < SPIN_HOT {
                    std::hint::spin_loop();
                } else if el < SPIN {
                    // ceder la CPU si otro hilo/proceso la necesita (evita sobresuscripción)
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

    /// Ejecuta f(0..n) repartido entre los hilos del pool (y el llamante). Bloquea hasta acabar.
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
        // SAFETY: `f` vive hasta que `done == n`; ningún hilo lo invoca después (los índices >= n
        // no se ejecutan) y esperamos aquí a que terminen todas las tareas.
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
            // por defecto la mitad de los hilos lógicos: en CPUs híbridas evita depender de los
            // núcleos lentos (E/LP-E) y de la sobresuscripción (medido: 6-8 >> 16 en Core Ultra 7)
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
        // un pánico en una tarea se relanza en el llamante, sin colgarse
        let r = std::panic::catch_unwind(|| {
            p.run(50, &|i| {
                if i == 7 {
                    panic!("tarea 7");
                }
            })
        });
        assert!(r.is_err());
        // el pool sigue funcionando después
        let m = AtomicUsize::new(0);
        p.run(10, &|_| {
            m.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(m.load(Ordering::Relaxed), 10);
    }
}
