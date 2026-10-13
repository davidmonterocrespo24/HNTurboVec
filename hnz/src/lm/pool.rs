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
                    unsafe { (*job.f)(i) };
                    job.done.fetch_add(1, Ordering::Release);
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
        let job = Arc::new(Job { f: fp, n, next: AtomicUsize::new(0), done: AtomicUsize::new(0) });
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
            f(i);
            job.done.fetch_add(1, Ordering::Release);
        }
        let t0 = Instant::now();
        while job.done.load(Ordering::Acquire) < n {
            if t0.elapsed() < SPIN_HOT {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }
}

pub fn pool() -> &'static SpinPool {
    static P: OnceLock<&'static SpinPool> = OnceLock::new();
    P.get_or_init(|| {
        let n = std::env::var("HNZ_LM_THREADS")
            .ok()
            .and_then(|x| x.parse().ok())
            // por defecto la mitad de los hilos lógicos: en CPUs híbridas evita depender de los
            // núcleos lentos (E/LP-E) y de la sobresuscripción (medido: 6-8 >> 16 en Core Ultra 7)
            .unwrap_or_else(|| (std::thread::available_parallelism().map(|x| x.get()).unwrap_or(4) / 2).max(1));
        SpinPool::new(n)
    })
}
