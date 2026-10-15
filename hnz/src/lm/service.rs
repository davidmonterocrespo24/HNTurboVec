//! The shared LM decoder of an engine (P4). One thread runs one continuous batch for every query
//! (`super::jobs`), so concurrent queries share decoding steps instead of waiting for each other,
//! and the batch never uses more than the LM pool. Interactive jobs (results, `get()`, titles) are
//! admitted first and, while any is running, bulk jobs (the reranker's prefixes) are paused: left
//! out of the steps with their KV cache kept. If an interactive job needs room, the most recently
//! admitted bulk sequence is restarted and queued again; decoding is batch-independent, so no
//! result changes. A panic while stepping fails the jobs in the batch, not the service.
use super::jobs::{Decoded, Prep, Runner, Seq, Want};
use super::store::LmCodec;
use anyhow::{anyhow, Result};
use std::collections::{HashMap, VecDeque};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};

/// Priority of a submission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prio {
    /// someone is waiting for this text right now (results, `get()`)
    Interactive,
    /// part of a larger computation (the reranker's prefixes)
    Bulk,
}

/// One record to decode.
pub struct Job {
    /// the record's bytes (header included)
    pub rec: Vec<u8>,
    pub has_check: bool,
    pub want: Want,
}

/// The results of a submission, as each job finishes. Dropping it cancels the jobs not done yet.
pub struct Ticket {
    rx: mpsc::Receiver<(usize, Result<Decoded>)>,
    cancel: Arc<AtomicBool>,
    pub n: usize,
}

impl Ticket {
    /// The next finished job: (its index in the submission, its result); None once all of them
    /// have been delivered (or if the service is gone).
    pub fn recv(&self) -> Option<(usize, Result<Decoded>)> {
        self.rx.recv().ok()
    }

    /// As `recv`, waiting at most until `deadline`: Ok(None) when it passes first.
    pub fn recv_until(&self, deadline: std::time::Instant) -> Result<Option<(usize, Result<Decoded>)>> {
        match self.rx.recv_timeout(deadline.saturating_duration_since(std::time::Instant::now())) {
            Ok(x) => Ok(Some(x)),
            Err(mpsc::RecvTimeoutError::Timeout) => Ok(None),
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(anyhow!("el decodificador LM se detuvo")),
        }
    }

    /// Every result, in submission order (the first error ends the wait).
    pub fn wait_all(self) -> Result<Vec<Decoded>> {
        let mut out: Vec<Option<Decoded>> = (0..self.n).map(|_| None).collect();
        for _ in 0..self.n {
            let (i, r) = self.recv().ok_or_else(|| anyhow!("el decodificador LM se detuvo"))?;
            out[i] = Some(r?);
        }
        Ok(out.into_iter().map(|x| x.expect("one result per job")).collect())
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

impl Drop for Ticket {
    fn drop(&mut self) {
        self.cancel();
    }
}

enum Msg {
    Submit { jobs: Vec<Job>, prio: Prio, reply: mpsc::Sender<(usize, Result<Decoded>)>, cancel: Arc<AtomicBool> },
    Pool(u64),
}

/// The LM decoding service of one model.
pub struct LmService {
    tx: Mutex<mpsc::Sender<Msg>>,
}

impl LmService {
    /// Starts the service's thread; `pool`: memory cap of its batch (bytes).
    pub fn new(codec: Arc<LmCodec>, pool: u64) -> Result<Self> {
        let (tx, rx) = mpsc::channel();
        std::thread::Builder::new().name("hnz-lm".into()).spawn(move || serve(&codec, pool, rx))?;
        Ok(Self { tx: Mutex::new(tx) })
    }

    pub fn submit(&self, jobs: Vec<Job>, prio: Prio) -> Ticket {
        let (reply, rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let n = jobs.len();
        // a dead service drops the message: the ticket then ends with an error
        let _ = self.tx.lock().unwrap_or_else(|e| e.into_inner()).send(Msg::Submit { jobs, prio, reply, cancel: cancel.clone() });
        Ticket { rx, cancel, n }
    }

    /// Decodes `jobs` and waits for all of them.
    pub fn run(&self, jobs: Vec<Job>, prio: Prio) -> Result<Vec<Decoded>> {
        self.submit(jobs, prio).wait_all()
    }

    /// New memory cap for the batch (sequences already running keep their room).
    pub fn set_pool(&self, bytes: u64) {
        let _ = self.tx.lock().unwrap_or_else(|e| e.into_inner()).send(Msg::Pool(bytes));
    }
}

/// A submission being served.
struct Sub {
    reply: mpsc::Sender<(usize, Result<Decoded>)>,
    cancel: Arc<AtomicBool>,
    left: usize,
}

/// A sequence waiting for room.
struct Waiting {
    seq: Box<Seq>,
    /// steps it has waited (the oldest bulk job eventually stops younger ones from passing it)
    waited: u64,
}

/// Bulk jobs a waiting bulk job lets pass before nothing younger is admitted ahead of it.
const MAX_WAIT_STEPS: u64 = 64;

/// While interactive sequences run, the bulk ones sit out the steps, but never for longer than
/// this: then they take part in one step. A long interactive decode, or a stream of them, slows
/// the reranker's prefixes without stopping them (the interactive ones keep >= ~2/3 of their speed).
const BULK_PAUSE: std::time::Duration = std::time::Duration::from_millis(100);

fn serve(codec: &LmCodec, pool: u64, rx: mpsc::Receiver<Msg>) {
    let _fp = super::fpenv::FpGuard::new();
    let mut run = Runner::new(codec, pool);
    let mut subs: HashMap<u64, Sub> = HashMap::new();
    let (mut wait_i, mut wait_b): (VecDeque<Waiting>, VecDeque<Waiting>) = (VecDeque::new(), VecDeque::new());
    let mut next_id = 0u64;
    let mut admitted = 0u64;
    let mut scratch: Option<super::math::ScratchRelease> = None;
    let mut paused_since: Option<std::time::Instant> = None;
    let deliver = |subs: &mut HashMap<u64, Sub>, tag: u64, r: Result<Decoded>| {
        let id = tag >> 32;
        if let Some(s) = subs.get_mut(&id) {
            let _ = s.reply.send(((tag & 0xffff_ffff) as usize, r));
            s.left -= 1;
            if s.left == 0 {
                subs.remove(&id);
            }
        }
    };
    loop {
        // messages: wait for one when idle, else take what has arrived
        let idle = run.running.is_empty() && wait_i.is_empty() && wait_b.is_empty();
        if idle {
            scratch = None; // the sampling scratch is given back while nothing runs
        }
        let msgs: Vec<Msg> = if idle {
            match rx.recv() {
                Ok(m) => vec![m],
                Err(_) => return, // the engine is gone
            }
        } else {
            rx.try_iter().collect()
        };
        for m in msgs {
            match m {
                Msg::Pool(b) => run.set_pool(b),
                Msg::Submit { jobs, prio, reply, cancel } => {
                    if jobs.is_empty() {
                        continue;
                    }
                    let id = next_id;
                    next_id += 1;
                    subs.insert(id, Sub { reply, cancel, left: jobs.len() });
                    for (i, j) in jobs.into_iter().enumerate() {
                        let tag = id << 32 | i as u64;
                        match codec.prepare(tag, &j.rec, j.has_check, j.want) {
                            Ok(Prep::Done(d)) => deliver(&mut subs, tag, Ok(d)),
                            Err(e) => deliver(&mut subs, tag, Err(e)),
                            Ok(Prep::Seq(mut s)) => {
                                s.interactive = prio == Prio::Interactive;
                                let w = Waiting { seq: s, waited: 0 };
                                if prio == Prio::Interactive {
                                    wait_i.push_back(w);
                                } else {
                                    wait_b.push_back(w);
                                }
                            }
                        }
                    }
                }
            }
        }
        // cancelled submissions (their ticket was dropped) leave the queues and the batch
        let gone = |subs: &HashMap<u64, Sub>, tag: u64| subs.get(&(tag >> 32)).is_none_or(|s| s.cancel.load(Ordering::Relaxed));
        wait_i.retain(|w| !gone(&subs, w.seq.tag));
        wait_b.retain(|w| !gone(&subs, w.seq.tag));
        let mut i = 0;
        while i < run.running.len() {
            if gone(&subs, run.running[i].tag) {
                drop(run.evict(i));
            } else {
                i += 1;
            }
        }
        subs.retain(|_, s| !s.cancel.load(Ordering::Relaxed));
        // admission: interactive jobs first (making room by restarting the youngest bulk ones)
        while let Some(w) = wait_i.front() {
            if run.fits(&w.seq) {
                let mut w = wait_i.pop_front().unwrap();
                w.seq.admitted = admitted;
                admitted += 1;
                run.start(w.seq);
                continue;
            }
            let youngest = (0..run.running.len()).filter(|&k| !run.running[k].interactive).max_by_key(|&k| run.running[k].admitted);
            match youngest {
                Some(k) => {
                    let s = run.evict(k);
                    wait_b.push_front(Waiting { seq: s, waited: 0 });
                }
                None => break,
            }
        }
        // then bulk jobs, first fit, unless the oldest has waited long enough to block the rest
        let mut k = 0;
        while k < wait_b.len() {
            if run.fits(&wait_b[k].seq) {
                let mut w = wait_b.remove(k).unwrap();
                w.seq.admitted = admitted;
                admitted += 1;
                run.start(w.seq);
            } else if k == 0 && wait_b[0].waited >= MAX_WAIT_STEPS {
                break;
            } else {
                k += 1;
            }
        }
        if run.running.is_empty() {
            continue;
        }
        if scratch.is_none() {
            scratch = Some(super::math::ScratchRelease);
        }
        // one step: only the interactive sequences while there are any, but the bulk ones join
        // one step after BULK_PAUSE
        let any_i = run.running.iter().any(|s| s.interactive);
        let any_b = run.running.iter().any(|s| !s.interactive);
        let only_i = any_i && any_b && paused_since.get_or_insert_with(std::time::Instant::now).elapsed() < BULK_PAUSE;
        if !only_i {
            paused_since = None;
        }
        let pick = move |s: &Seq| !only_i || s.interactive;
        match catch_unwind(AssertUnwindSafe(|| run.step(true, &pick))) {
            Ok(done) => {
                for (tag, r) in done {
                    deliver(&mut subs, tag, r);
                }
            }
            Err(_) => {
                for s in run.evict_all() {
                    deliver(&mut subs, s.tag, Err(anyhow!("fallo interno al decodificar con el LM")));
                }
                // the batch's state is unknown after a panic: start a fresh one
                let p = run.pool_bytes();
                run = Runner::new(codec, p);
            }
        }
        for w in wait_b.iter_mut() {
            w.waited += 1;
        }
    }
}
