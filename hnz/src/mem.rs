//! RAM budget (P2): hnz's own ledger, not the process "working set" (which in a backend includes
//! the host application's memory). Every component that takes memory records it with a `Charge`,
//! released automatically when dropped (RAII). Opaque parts (OpenVINO, tokenizers) are measured
//! with the process memory counters while loading them.
use serde::Serialize;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

pub const MB: u64 = 1 << 20;

/// Size for messages, in decimal units like `ram="2GB"` (2 x 10^9 bytes).
pub fn human(b: u64) -> String {
    if b >= 1_000_000_000 {
        format!("{:.2} GB", b as f64 / 1e9)
    } else {
        format!("{:.0} MB", b as f64 / 1e6)
    }
}

/// The RAM budget does not cover the bare minimum (in Python: `hnz.BudgetError`).
#[derive(Debug)]
pub struct BudgetError(pub String);
impl std::fmt::Display for BudgetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for BudgetError {}

/// No turn for the operation within the timeout (in Python: `hnz.BusyError`).
#[derive(Debug)]
pub struct BusyError(pub String);
impl std::fmt::Display for BusyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for BusyError {}

/// Memory classes reported separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// models and tokenizers (including OpenVINO's opaque memory)
    Model = 0,
    /// resident index structures (vectors, BM25, metadata, flags…)
    Index = 1,
    /// caches (decompressed records, titles…): can be emptied
    Cache = 2,
    /// working memory of running queries (LM batch, candidates…)
    Working = 3,
}

const NAMES: [&str; 4] = ["models", "index", "cache", "working"];

pub struct Budget {
    /// limit in bytes (u64::MAX = no limit)
    pub limit: u64,
    used: [AtomicU64; 4],
    peak: AtomicU64,
}

/// Memory recorded in the budget; released when dropped.
pub struct Charge {
    budget: Arc<Budget>,
    class: Class,
    bytes: u64,
}

impl Budget {
    pub fn new(limit: Option<u64>) -> Arc<Self> {
        Arc::new(Self { limit: limit.unwrap_or(u64::MAX), used: Default::default(), peak: AtomicU64::new(0) })
    }

    pub fn used(&self) -> u64 {
        self.used.iter().map(|u| u.load(Ordering::Relaxed)).sum()
    }

    pub fn free(&self) -> u64 {
        self.limit.saturating_sub(self.used())
    }

    /// Records memory without checking the limit (fixed costs already planned, e.g. at open).
    pub fn charge(self: &Arc<Self>, class: Class, bytes: u64) -> Charge {
        self.used[class as usize].fetch_add(bytes, Ordering::Relaxed);
        self.peak.fetch_max(self.used(), Ordering::Relaxed);
        Charge { budget: self.clone(), class, bytes }
    }

    /// Reserves memory only if it fits in the limit.
    pub fn try_reserve(self: &Arc<Self>, class: Class, bytes: u64) -> Option<Charge> {
        let slot = &self.used[class as usize];
        loop {
            let total = self.used();
            if total.saturating_add(bytes) > self.limit {
                return None;
            }
            let cur = slot.load(Ordering::Relaxed);
            if slot.compare_exchange(cur, cur + bytes, Ordering::AcqRel, Ordering::Relaxed).is_ok() {
                self.peak.fetch_max(self.used(), Ordering::Relaxed);
                return Some(Charge { budget: self.clone(), class, bytes });
            }
        }
    }

    pub fn report(&self) -> Report {
        let by = |c: Class| self.used[c as usize].load(Ordering::Relaxed);
        Report {
            limit: (self.limit != u64::MAX).then_some(self.limit),
            used: self.used(),
            peak: self.peak.load(Ordering::Relaxed),
            models: by(Class::Model),
            index: by(Class::Index),
            cache: by(Class::Cache),
            working: by(Class::Working),
        }
    }
}

impl Charge {
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Changes the recorded size (e.g. a cache that grows or shrinks).
    pub fn resize(&mut self, bytes: u64) {
        let slot = &self.budget.used[self.class as usize];
        if bytes >= self.bytes {
            slot.fetch_add(bytes - self.bytes, Ordering::Relaxed);
            self.budget.peak.fetch_max(self.budget.used(), Ordering::Relaxed);
        } else {
            slot.fetch_sub(self.bytes - bytes, Ordering::Relaxed);
        }
        self.bytes = bytes;
    }
}

impl Drop for Charge {
    fn drop(&mut self) {
        self.budget.used[self.class as usize].fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

#[derive(Serialize, Clone, Copy, Debug)]
pub struct Report {
    pub limit: Option<u64>,
    pub used: u64,
    pub peak: u64,
    pub models: u64,
    pub index: u64,
    pub cache: u64,
    pub working: u64,
}

impl Report {
    pub fn names() -> [&'static str; 4] {
        NAMES
    }
}

/// Admission control: at most `max` operations at once and, among all of them, `pool` bytes of
/// working memory (what each one declares when entering). Without room, waits in arrival order
/// (FIFO: a big request is not overtaken forever by small ones) up to `timeout`, then fails with
/// `BusyError` (never blocks forever).
pub struct Admission {
    st: Mutex<AdmState>,
    cv: Condvar,
    pub max: usize,
    pub pool: u64,
    pub timeout: Duration,
    waits: AtomicU64,
    rejected: AtomicU64,
}

struct AdmState {
    running: usize,
    in_use: u64,
    /// number given to the next arrival
    next: u64,
    /// waiting arrivals, oldest first
    queue: VecDeque<u64>,
}

/// A granted turn; given back when dropped.
pub struct Ticket<'a> {
    adm: &'a Admission,
    bytes: AtomicU64,
}

#[derive(Serialize, Clone, Copy, Debug)]
pub struct AdmissionReport {
    pub max_concurrent: usize,
    pub running: usize,
    pub pool: u64,
    pub pool_in_use: u64,
    pub timeout_s: f64,
    /// operations that had to wait / that were rejected by the timeout
    pub waits: u64,
    pub rejected: u64,
}

impl Admission {
    pub fn new(max: usize, pool: u64, timeout: Duration) -> Self {
        Self {
            st: Mutex::new(AdmState { running: 0, in_use: 0, next: 0, queue: VecDeque::new() }),
            cv: Condvar::new(),
            max: max.max(1),
            pool,
            timeout,
            waits: AtomicU64::new(0),
            rejected: AtomicU64::new(0),
        }
    }

    /// Enters with `bytes` of working memory (at most the whole pool: a bigger operation runs
    /// alone).
    pub fn enter(&self, bytes: u64) -> Result<Ticket<'_>, BusyError> {
        let bytes = bytes.min(self.pool);
        let deadline = Instant::now() + self.timeout;
        let fits = |g: &AdmState| g.running < self.max && g.in_use + bytes <= self.pool;
        let mut g = self.st.lock().unwrap_or_else(|e| e.into_inner());
        let me = g.next;
        g.next += 1;
        if !(g.queue.is_empty() && fits(&g)) {
            self.waits.fetch_add(1, Ordering::Relaxed);
            g.queue.push_back(me);
            loop {
                if g.queue.front() == Some(&me) && fits(&g) {
                    g.queue.pop_front();
                    break;
                }
                let now = Instant::now();
                if now >= deadline {
                    g.queue.retain(|&x| x != me);
                    self.rejected.fetch_add(1, Ordering::Relaxed);
                    let msg = format!(
                        "hnz ocupado: {} operaciones en curso (máximo {}) y {} de {} de memoria de trabajo en uso; se esperó {:.1} s",
                        g.running,
                        self.max,
                        human(g.in_use),
                        human(self.pool),
                        self.timeout.as_secs_f64()
                    );
                    drop(g);
                    self.cv.notify_all(); // the next in line may go now
                    return Err(BusyError(msg));
                }
                g = self.cv.wait_timeout(g, deadline - now).unwrap_or_else(|e| e.into_inner()).0;
            }
        }
        g.running += 1;
        g.in_use += bytes;
        let more = !g.queue.is_empty();
        drop(g);
        if more {
            self.cv.notify_all(); // the next in line may also fit
        }
        Ok(Ticket { adm: self, bytes: AtomicU64::new(bytes) })
    }

    pub fn report(&self) -> AdmissionReport {
        let g = self.st.lock().unwrap_or_else(|e| e.into_inner());
        AdmissionReport {
            max_concurrent: self.max,
            running: g.running,
            pool: self.pool,
            pool_in_use: g.in_use,
            timeout_s: self.timeout.as_secs_f64(),
            waits: self.waits.load(Ordering::Relaxed),
            rejected: self.rejected.load(Ordering::Relaxed),
        }
    }
}

impl Ticket<'_> {
    /// Asks for `extra` working memory without waiting (e.g. to widen a search): granted only if
    /// it fits and nobody is waiting for a turn.
    pub fn try_grow(&self, extra: u64) -> bool {
        let mut g = self.adm.st.lock().unwrap_or_else(|e| e.into_inner());
        if g.queue.is_empty() && g.in_use + extra <= self.adm.pool {
            g.in_use += extra;
            self.bytes.fetch_add(extra, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    pub fn bytes(&self) -> u64 {
        self.bytes.load(Ordering::Relaxed)
    }
}

impl Drop for Ticket<'_> {
    fn drop(&mut self) {
        let mut g = self.adm.st.lock().unwrap_or_else(|e| e.into_inner());
        g.running -= 1;
        g.in_use -= self.bytes.load(Ordering::Relaxed);
        drop(g);
        self.adm.cv.notify_all();
    }
}

/// Memory that `f` adds to the process according to the OS: (result, steady, peak). Steady is the
/// larger growth of working set and private memory; peak, the working-set peak during `f` above
/// the start (if the process peak went up). For opaque parts (OpenVINO, tokenizers); approximate if
/// other threads work at the same time.
pub fn measure<T>(f: impl FnOnce() -> T) -> (T, u64, u64) {
    let a = process_memory();
    let r = f();
    let b = process_memory();
    let steady = b.working_set.saturating_sub(a.working_set).max(b.private.saturating_sub(a.private));
    let peak = if b.peak_working_set > a.peak_working_set { b.peak_working_set.saturating_sub(a.working_set) } else { 0 };
    (r, steady, peak.max(steady))
}

/// Process memory according to the OS (to measure opaque parts and to check the budget).
#[derive(Serialize, Clone, Copy, Debug, Default)]
pub struct ProcMem {
    /// working set (physical RAM in use now)
    pub working_set: u64,
    pub peak_working_set: u64,
    /// committed private memory
    pub private: u64,
}

#[cfg(windows)]
pub fn process_memory() -> ProcMem {
    use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c: PROCESS_MEMORY_COUNTERS_EX = unsafe { std::mem::zeroed() };
    c.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
    let ok = unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut c as *mut _ as *mut PROCESS_MEMORY_COUNTERS, c.cb) };
    if ok == 0 {
        return ProcMem::default();
    }
    ProcMem { working_set: c.WorkingSetSize as u64, peak_working_set: c.PeakWorkingSetSize as u64, private: c.PrivateUsage as u64 }
}

#[cfg(not(windows))]
pub fn process_memory() -> ProcMem {
    ProcMem::default()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn charges_and_limit() {
        let b = Budget::new(Some(1000));
        let a = b.charge(Class::Model, 600);
        assert!(b.try_reserve(Class::Working, 500).is_none());
        let mut c = b.try_reserve(Class::Cache, 300).unwrap();
        assert_eq!(b.used(), 900);
        c.resize(100);
        assert_eq!(b.used(), 700);
        drop(a);
        assert_eq!(b.used(), 100);
        drop(c);
        assert_eq!(b.used(), 0);
        assert_eq!(b.report().peak, 900);
        assert!(process_memory().working_set > 0 || !cfg!(windows));
    }

    #[test]
    fn admission_waits_and_times_out() {
        let a = Arc::new(Admission::new(2, 100, Duration::from_millis(150)));
        let t1 = a.enter(60).unwrap();
        // does not fit in memory: waits and fails at the timeout
        let t0 = Instant::now();
        assert!(a.enter(50).is_err());
        assert!(t0.elapsed() >= Duration::from_millis(140));
        // fits
        let t2 = a.enter(40).unwrap();
        // by number of operations
        assert!(a.enter(0).is_err());
        assert_eq!(a.report().rejected, 2);
        // when a turn is given back, the waiting one enters
        let a2 = a.clone();
        let h = std::thread::spawn(move || a2.enter(30).map(|t| t.bytes()).is_ok());
        std::thread::sleep(Duration::from_millis(30));
        drop(t1);
        assert!(h.join().unwrap());
        drop(t2);
        // bigger than the whole pool: runs alone
        let big = a.enter(1000).unwrap();
        assert_eq!(a.report().pool_in_use, 100);
        drop(big);
        assert_eq!(a.report().running, 0);
        // growing a running operation: only if it fits
        let t = a.enter(50).unwrap();
        assert!(t.try_grow(40) && t.bytes() == 90);
        assert!(!t.try_grow(20));
        drop(t);
        assert_eq!(a.report().pool_in_use, 0);
    }

    #[test]
    fn admission_is_fifo() {
        // pool 100: a request of 80 waiting is not overtaken by later small ones
        let a = Arc::new(Admission::new(8, 100, Duration::from_secs(5)));
        let t1 = a.enter(30).unwrap();
        let order = Arc::new(Mutex::new(Vec::new()));
        let spawn = |bytes: u64, tag: &'static str, delay: u64| {
            let (a, order) = (a.clone(), order.clone());
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(delay));
                let t = a.enter(bytes).unwrap();
                order.lock().unwrap().push(tag);
                std::thread::sleep(Duration::from_millis(20));
                drop(t);
            })
        };
        let big = spawn(80, "big", 0);
        std::thread::sleep(Duration::from_millis(50));
        // fits right now (30 + 30 <= 100), but the big one is waiting first (and once the big one is
        // in, 80 + 30 > 100: so the order they are recorded in is deterministic)
        let small = spawn(30, "small", 0);
        std::thread::sleep(Duration::from_millis(50));
        assert!(order.lock().unwrap().is_empty());
        assert!(!t1.try_grow(1), "no growth while someone waits");
        drop(t1);
        big.join().unwrap();
        small.join().unwrap();
        assert_eq!(*order.lock().unwrap(), vec!["big", "small"]);
        assert_eq!(a.report().running, 0);
    }
}
