//! Stable facade to use hnz as a library (Python via PyO3, C, HTTP): open an archive, search and
//! fetch items. Thread-safe: many queries at once on the same `Archive`.
//!
//! - RAM (P2): everything hnz holds (models, embedder, loaded indexes, LM batch memory, caches,
//!   query working memory) is charged to a ledger with the `ram` limit. At open, a planner
//!   (`plan.rs`) decides what to load into RAM; the rest is read from disk on demand (positional
//!   reads that go through the OS file cache, outside the process). Searches and gets go through
//!   admission control (`max_concurrent` + working-memory pool, `BusyError` after `timeout`).
//! - The query embedder (OpenVINO) lives in its own thread: its objects cannot be shared between
//!   threads, so no `unsafe` is needed; queries ask it for vectors through a channel. Until it is
//!   loaded, its load peak is reserved in the budget; once loaded, its measured size is charged.
//! - LM decompression goes through one shared service per LM model (`lm::service`): one
//!   continuous batch serves every concurrent query within that model's share of the LM pool;
//!   zstd-tier or cached queries do not wait for LM-tier ones.
use crate::blob::Residency;
use crate::dates;
use crate::embed::Embedder;
use crate::engine::{Engine, EngineOptions, Mode, Query, ResultItem, MAX_K};
use crate::htmlx;
use crate::mem::{human, Admission, Budget, Charge, Class, MB};
use crate::plan::{self, Candidate, Plan};
use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

// the engine is shared between threads: if it ever stops being Send + Sync, fail at compile time
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Engine>();
};

/// Open options.
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct OpenOptions {
    /// models folder (default `<archive>/models` or `<archive>/../models`)
    pub models: Option<PathBuf>,
    /// folder with `openvino_c.dll` (default: the `HNZ_OV_LIBS` variable)
    pub ov_libs: Option<PathBuf>,
    /// load the embedder at open (otherwise, at the first semantic search)
    pub preload: bool,
    /// RAM budget in bytes for everything hnz holds (None = no limit)
    pub ram: Option<u64>,
    /// at most this many searches/gets at the same time (the rest wait their turn)
    pub max_concurrent: usize,
    /// seconds an operation may wait for its turn before failing with `BusyError`
    pub timeout: f64,
    /// folder for the optional persistent cache of decompressed records (None = off)
    pub cache_dir: Option<PathBuf>,
    /// maximum size of that cache on disk, in bytes
    pub cache_dir_bytes: u64,
    /// compact blocks: what is checked at open: "header", "quick" (+ always-loaded parts and 4
    /// short canaries decoded) or "full" (+ every part and external file, all canaries)
    pub verify: String,
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self {
            models: None,
            ov_libs: None,
            preload: false,
            ram: None,
            max_concurrent: 8,
            timeout: 30.0,
            cache_dir: None,
            cache_dir_bytes: 512 * MB,
            verify: "quick".into(),
        }
    }
}

/// A search. Dates as in the CLI: "2015", "2015-06", "2015-06-01" (both ends inclusive).
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct SearchRequest {
    pub query: String,
    pub k: usize,
    pub since: Option<String>,
    pub until: Option<String>,
    /// exact limits in UTC epoch seconds (both inclusive); not allowed together with since/until
    pub since_ts: Option<i64>,
    pub until_ts: Option<i64>,
    /// story, comment, job, poll, pollopt (empty = all)
    pub types: Vec<String>,
    pub author: Option<String>,
    /// hybrid | vector | keyword
    pub mode: String,
    /// full text for the LM tier too (otherwise a 64-token snippet)
    pub full: bool,
    pub include_dead: bool,
    /// weight and saturation of BM25 in the fusion `vector + w·b/(b + c)`; None = the defaults
    /// (0.15 and 20 in stage 1, 0.15 and 40 when reranked)
    pub w_bm25: Option<f32>,
    pub c_bm25: Option<f32>,
    /// P4: rerank the candidates by their prefix (compact blocks); None = whenever possible
    pub rerank: Option<bool>,
    /// P4: answer within this many milliseconds from the call (opt-in): a reranking not done by
    /// then gives way to stage 1's results, or to the cheap score's order, marked `degraded`
    pub deadline_ms: Option<f64>,
    /// P4: vector candidates to rerank (None = 80; fewer is faster, more reaches deeper)
    pub rerank_depth: Option<usize>,
}

impl Default for SearchRequest {
    fn default() -> Self {
        Self {
            query: String::new(),
            k: 10,
            since: None,
            until: None,
            since_ts: None,
            until_ts: None,
            types: Vec::new(),
            author: None,
            mode: "hybrid".into(),
            full: false,
            include_dead: false,
            w_bm25: None,
            c_bm25: None,
            rerank: None,
            deadline_ms: None,
            rerank_depth: None,
        }
    }
}

#[derive(Serialize, Default, Clone, Copy)]
pub struct Timings {
    /// waiting for a turn (admission control)
    pub queue: f64,
    pub embedding: f64,
    pub search: f64,
    pub fetch: f64,
    pub total: f64,
}

#[derive(Serialize)]
pub struct SearchResponse {
    pub query: String,
    pub k: usize,
    pub results: Vec<ResultItem>,
    pub ms: Timings,
    /// what the reranking did (P4), if the search was reranked
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rerank: Option<crate::rerank::Report>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warning: Option<String>,
}

/// An error in a query's arguments (in Python: ValueError).
#[derive(Debug)]
pub struct QueryError(pub String);
impl std::fmt::Display for QueryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for QueryError {}

pub use crate::mem::{BudgetError, BusyError};

/// Longest query text accepted (the embedder reads its first 64 tokens; BM25 uses at most the
/// 64 rarest terms).
pub const MAX_QUERY_BYTES: usize = 8192;

fn check_text(text: &str) -> Result<()> {
    if text.len() > MAX_QUERY_BYTES {
        return Err(anyhow::Error::new(QueryError(format!("query too long: {} bytes (maximum {MAX_QUERY_BYTES})", text.len()))));
    }
    Ok(())
}

/// Validates a query and turns it into the engine's query.
pub fn make_query(r: &SearchRequest) -> Result<Query> {
    let bad = |m: String| anyhow::Error::new(QueryError(m));
    check_text(&r.query)?;
    if r.k == 0 || r.k > MAX_K {
        return Err(bad(format!("k must be between 1 and {MAX_K} (got {})", r.k)));
    }
    if r.c_bm25.is_some_and(|c| !(c.is_finite() && c > 0.0)) || r.w_bm25.is_some_and(|w| !(w.is_finite() && w >= 0.0)) {
        return Err(bad(format!("c_bm25 must be > 0 and w_bm25 >= 0 (got {:?} and {:?})", r.c_bm25, r.w_bm25)));
    }
    if r.rerank_depth.is_some_and(|d| d == 0 || d > crate::rerank::MAX_DEPTH) {
        return Err(bad(format!("rerank_depth must be between 1 and {} (got {:?})", crate::rerank::MAX_DEPTH, r.rerank_depth)));
    }
    if r.deadline_ms.is_some_and(|d| !(d.is_finite() && d > 0.0 && d <= 3.6e6)) {
        return Err(bad(format!("deadline_ms must be between 0 and 3,600,000 (got {:?})", r.deadline_ms)));
    }
    let mut types = Vec::new();
    for t in &r.types {
        types.push(match t.as_str() {
            "story" => 1u8,
            "comment" => 2,
            "poll" => 3,
            "pollopt" => 4,
            "job" => 5,
            o => return Err(bad(format!("unknown type: {o} (use story, comment, job, poll or pollopt)"))),
        });
    }
    let mode = match r.mode.as_str() {
        "hybrid" => Mode::Hybrid,
        "vector" => Mode::Vector,
        "keyword" => Mode::Keyword,
        o => return Err(bad(format!("unknown mode: {o} (use hybrid, vector or keyword)"))),
    };
    let date = |s: &Option<String>, end: bool, name: &str| -> Result<Option<i64>> {
        match s {
            Some(x) => dates::parse(x, end).map(Some).ok_or_else(|| bad(format!("invalid {name} date: {x:?} (use 2015, 2015-06 or 2015-06-01)"))),
            None => Ok(None),
        }
    };
    if (r.since.is_some() && r.since_ts.is_some()) || (r.until.is_some() && r.until_ts.is_some()) {
        return Err(bad("give since or since_ts (and until or until_ts), not both".into()));
    }
    let from = match r.since_ts {
        Some(t) => Some(t),
        None => date(&r.since, false, "since")?,
    };
    let to = match r.until_ts {
        Some(t) => Some(t.checked_add(1).ok_or_else(|| bad("until_ts out of range".into()))?),
        None => date(&r.until, true, "until")?,
    };
    Ok(Query {
        text: r.query.clone(),
        k: r.k,
        from,
        to,
        mode,
        types,
        by: r.author.clone(),
        include_dead: r.include_dead,
        w_bm25: r.w_bm25.unwrap_or(0.15),
        c_bm25: r.c_bm25.unwrap_or(20.0),
        vec_cand: None,
    })
}

/// Work for the embedder's thread: a query (answered before the next round of documents), or
/// document prompts (P4) with a flag their search sets when it stops waiting for them.
enum EmbedJob {
    Query(String, mpsc::Sender<Result<Vec<f32>>>),
    Docs(Vec<String>, mpsc::Sender<Result<Vec<Vec<f32>>>>, Arc<std::sync::atomic::AtomicBool>),
}

/// Minimum working memory for queries (on top of the vector read windows).
const QUERY_BASE: u64 = 16 * MB;

/// Working memory declared by a search besides its candidate lists (`Engine::search_cost`):
/// metadata blocks, thread context and the results.
fn search_cost(k: usize, full: bool) -> u64 {
    2 * MB + k as u64 * if full { 64 << 10 } else { 8 << 10 }
}

/// Working memory declared by a get (one full record).
const GET_COST: u64 = MB;

/// Embedder measurements: (steady bytes, load peak bytes).
#[derive(Clone, Copy, Debug, Serialize)]
struct EmbMem {
    steady: u64,
    peak: u64,
    /// the embedder can embed document prompts (an IR with dynamic shapes)
    docs: bool,
    /// document prompts embedded at the same time (CPU streams)
    streams: usize,
    /// mean sign flips (of 256) of the re-embedded canaries against their stored codes (P4)
    check_flips: Option<f64>,
}

/// Un archivo hnz abierto.
pub struct Archive {
    eng: Engine,
    root: PathBuf,
    models: PathBuf,
    ov_libs: Option<PathBuf>,
    budget: Arc<Budget>,
    plan: Plan,
    adm: Admission,
    _query_charge: Charge,
    /// embedder: its load peak is reserved until it is loaded; then, its measured size
    emb_charge: Mutex<Charge>,
    emb_mem: Mutex<Option<EmbMem>>,
    /// channel to the embedder thread (created on first use; reading it never blocks)
    emb: OnceLock<mpsc::Sender<EmbedJob>>,
    /// serializes the attempts to load the embedder and remembers the last failure (so the load is
    /// not retried on every query)
    /// the last failed load: when, its message, and whether it was a damaged (1) or an
    /// incompatible (2) block, so a retry inside EMB_RETRY raises the same error type
    emb_init: Mutex<Option<(Instant, String, u8)>>,
}

/// After a failure to load the embedder, it is not retried until this time has passed.
const EMB_RETRY: Duration = Duration::from_secs(30);

fn default_models(root: &Path) -> Result<PathBuf> {
    for c in [root.join("models"), root.parent().map(|p| p.join("models")).unwrap_or_default()] {
        if c.is_dir() {
            return Ok(c);
        }
    }
    bail!("no models found next to {} (pass models=...)", root.display())
}

/// Embedder files: the int8 version if present (half the size, ~2.5x faster, cosine 0.9998).
pub fn embedder_files(models: &Path) -> (PathBuf, PathBuf, PathBuf) {
    let name = if models.join("gemma2_text_b1_s64_int8.xml").exists() { "gemma2_text_b1_s64_int8" } else { "gemma2_text_b1_s64" };
    (models.join(format!("{name}.xml")), models.join(format!("{name}.bin")), models.join("gemma2_tokenizer.json"))
}

/// Estimated load peak of the embedder. E31: the int8 weights (272 MB) + a 32 MB tokenizer peaked
/// at +746 MB when the tokenizer was parsed after compiling the model; parsing it first, the peak
/// is the steady size, +447 MiB. Estimate: 1.4 x weights + 4 x tokenizer + 32 MiB (~16 % margin).
fn embedder_estimate(models: &Path) -> u64 {
    let (_, bin, tok) = embedder_files(models);
    let size = |p: &Path| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
    size(&bin) * 7 / 5 + 4 * size(&tok) + 32 * MB
}

impl Archive {
    pub fn open(root: &Path, opts: OpenOptions) -> Result<Self> {
        let level = match opts.verify.as_str() {
            "header" => crate::block::VerifyLevel::Header,
            "quick" => crate::block::VerifyLevel::Quick,
            "full" => crate::block::VerifyLevel::Full,
            o => return Err(anyhow::Error::new(QueryError(format!("verify must be header, quick or full (got {o:?})")))),
        };
        // a compact block (P3), or a folder of segments
        let block = match crate::block::Block::locate(root) {
            Some(p) => Some(crate::block::Block::open(&p)?),
            None => None,
        };
        // first, the folder must have segments (the most likely error: wrong folder)
        let has_segments = root.is_dir()
            && std::fs::read_dir(root)
                .with_context(|| format!("cannot read {}", root.display()))?
                .filter_map(|e| e.ok())
                .any(|e| e.path().join("manifest.json").exists());
        if block.is_some() && has_segments {
            bail!("{} contains a block (hn.hnzb) and also segment folders: keep one or the other", root.display());
        }
        if block.is_none() && !has_segments {
            let hint = if root.join("manifest.json").exists() {
                " (that folder IS a segment: pass the folder that contains the segments)"
            } else {
                ""
            };
            bail!("no segments (folders with manifest.json) in {}{hint}", root.display());
        }
        let timeout = Duration::try_from_secs_f64(opts.timeout).ok().filter(|_| opts.max_concurrent > 0).ok_or_else(|| {
            anyhow::Error::new(QueryError(format!(
                "timeout must be a number of seconds >= 0 (and reasonable) and max_concurrent >= 1 (got {} and {})",
                opts.timeout, opts.max_concurrent
            )))
        })?;
        let models = match (&block, opts.models) {
            (Some(b), _) => b.dir(),
            (None, Some(m)) => m,
            (None, None) => default_models(root)?,
        };
        let budget = Budget::new(opts.ram);
        // open with nothing loaded (only the fixed structures), then plan what to load; the disk
        // cache is attached after planning, with its RAM index already capped
        let eopts = EngineOptions { budget: budget.clone(), residency: Residency::NONE, l2: None, cache_bytes: 0, lm_pool: 0 };
        if let Some(d) = &opts.cache_dir {
            std::fs::create_dir_all(d).with_context(|| format!("cache_dir {}", d.display()))?;
        }
        let mut eng = match &block {
            Some(b) => Engine::open_block(b.clone(), eopts)?,
            None => Engine::open_with(root, &models, eopts).with_context(|| format!("opening {}", root.display()))?,
        };
        if eng.segs.is_empty() {
            let hint = if root.join("manifest.json").exists() {
                " (that folder IS a segment: pass the folder that contains the segments)"
            } else {
                ""
            };
            bail!("no segments (folders with manifest.json) in {}{hint}", root.display());
        }
        let emb_est = match &block {
            Some(b) => {
                let bin = b.external().iter().find(|e| e.name == "embed.bin").map_or(0, |e| e.len);
                let tok = b.part("model/embed.tok").map_or(0, |p| p.raw_len);
                // + what embedding document prompts can add (P4: one request per stream)
                bin * 7 / 5 + 4 * tok + 32 * MB + crate::embed::doc_streams() as u64 * crate::embed::DOC_REQUEST_BYTES
            }
            None => embedder_estimate(&models),
        };
        let emb_charge = budget.charge(Class::Model, emb_est);
        let candidates =
            eng.parts().into_iter().map(|(seg, part, _, bytes)| Candidate { seg, part, bytes, recency: eng.segs[seg].s.man.time_max }).collect();
        let input = plan::Input {
            limit: opts.ram,
            models: eng.model_bytes() + emb_est,
            fixed: eng.fixed_bytes(),
            candidates,
            lm_min: eng.lm_min(),
            query_min: QUERY_BASE + eng.vec_read_bytes(),
        };
        let plan = plan::plan(&input)?;
        for &(si, p) in &plan.resident {
            eng.promote(si, p).with_context(|| format!("loading {} of {}", p.name(), eng.segs[si].s.man.name))?;
        }
        eng.set_lm_pool(plan.lm_pool);
        // with a persistent cache, its RAM index takes 1/8 of the cache share (newest entries kept)
        if let Some(d) = &opts.cache_dir {
            let max_entries = (plan.cache / 8 / crate::cache::L2_ENTRY_BYTES) as usize;
            match crate::cache::L2::open_limited(&d.join("hnz-cache.bin"), opts.cache_dir_bytes, max_entries) {
                Ok(l2) => eng.set_l2(l2),
                Err(e) => eprintln!("warning: persistent cache disabled ({e})"),
            }
        }
        eng.set_cache_bytes(plan.cache);
        if let Some(b) = &block {
            // canaries: records decoded now, compared with the text they had when packed
            if level >= crate::block::VerifyLevel::Quick {
                eng.check_canaries(&["short"])?;
            }
            if level >= crate::block::VerifyLevel::Full {
                b.verify_full()?;
                eng.check_canaries(&["long", "fallback", "empty"])?;
            }
        }
        let query_charge = budget.charge(Class::Working, plan.query_pool);
        let adm = Admission::new(opts.max_concurrent, plan.query_pool, timeout);
        let ov_libs = opts.ov_libs.or_else(|| std::env::var_os("HNZ_OV_LIBS").map(PathBuf::from));
        let a = Archive {
            eng,
            root: root.to_path_buf(),
            models,
            ov_libs,
            budget,
            plan,
            adm,
            _query_charge: query_charge,
            emb_charge: Mutex::new(emb_charge),
            emb_mem: Mutex::new(None),
            emb: OnceLock::new(),
            emb_init: Mutex::new(None),
        };
        if opts.preload {
            a.embed_inner("warm up")?;
        }
        Ok(a)
    }

    fn embedder(&self) -> Result<mpsc::Sender<EmbedJob>> {
        if let Some(tx) = self.emb.get() {
            return Ok(tx.clone());
        }
        let mut last = self.emb_init.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(tx) = self.emb.get() {
            return Ok(tx.clone()); // another thread loaded it while we waited
        }
        if let Some((t, msg, kind)) = last.as_ref() {
            if t.elapsed() < EMB_RETRY {
                let m = format!("the embedder failed to load {:.0} s ago: {msg}", t.elapsed().as_secs_f64());
                return Err(match kind {
                    1 => crate::block::corrupt(m),
                    2 => crate::block::incompatible(m),
                    _ => anyhow!(m),
                });
            }
        }
        match self.spawn_embedder() {
            Ok((tx, mem)) => {
                self.embedder_loaded(mem);
                let _ = self.emb.set(tx.clone());
                *last = None;
                Ok(tx)
            }
            Err(e) => {
                let kind = if e.downcast_ref::<crate::block::CorruptBlockError>().is_some() {
                    1
                } else if e.downcast_ref::<crate::block::IncompatibleBlockError>().is_some() {
                    2
                } else {
                    0
                };
                *last = Some((Instant::now(), format!("{e:#}"), kind));
                Err(e)
            }
        }
    }

    /// Replaces the reservation of the embedder's load peak by its measured size; the difference
    /// goes to (or comes from) the caches, so the plan still adds up to the limit.
    fn embedder_loaded(&self, mem: EmbMem) {
        let mut c = self.emb_charge.lock().unwrap_or_else(|e| e.into_inner());
        let reserved = c.bytes();
        // measurements with other threads working can fall short: never below 60 % of the estimate
        let keep = mem.steady.max(reserved * 3 / 5);
        if mem.peak > reserved {
            eprintln!("warning: the embedder reached {} while loading ({} reserved)", human(mem.peak), human(reserved));
        }
        c.resize(keep);
        let cache = self.eng.cache_bytes();
        let new_cache = if keep <= reserved { cache + (reserved - keep) } else { cache.saturating_sub(keep - reserved).max(plan::CACHE_MIN) };
        self.eng.set_cache_bytes(new_cache);
        *self.emb_mem.lock().unwrap_or_else(|e| e.into_inner()) = Some(mem);
    }

    fn spawn_embedder(&self) -> Result<(mpsc::Sender<EmbedJob>, EmbMem)> {
        let libs = self.ov_libs.clone().context("the OpenVINO folder is missing (set ov_libs or the HNZ_OV_LIBS variable)")?;
        let models = self.models.clone();
        // a block: its IR files (checked against their hashes) and the tokenizer inside it, read
        // in the embedder's thread so the measured load includes them and frees the tokenizer
        let block = self.eng.block.clone();
        // the reranking chain's canaries: decoded now, re-embedded once the embedder is loaded
        let checks = self.eng.embed_checks()?;
        let (tx, rx) = mpsc::channel::<EmbedJob>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<EmbMem>>();
        std::thread::Builder::new().name("hnz-embedder".into()).spawn(move || {
            // measure the load and a first inference (it allocates OpenVINO's buffers and threads)
            let (r, steady, peak) = crate::mem::measure(|| {
                let e = match &block {
                    Some(b) => b.embedder_files().and_then(|(xml, bin, tok)| Embedder::from_files(&xml, &bin, &tok, &libs)),
                    None => Embedder::new(&models, &libs),
                };
                e.and_then(|mut e| {
                    e.embed_query("warm up")?;
                    if e.can_embed_docs() {
                        // the longest document shape on every stream, so the measured size
                        // includes all their buffers
                        e.embed_docs(&vec!["warm up ".repeat(200); e.doc_parallelism()])?;
                    }
                    // the chain LM prefix -> prompt -> tokenizer -> IR must give back the
                    // vectors packed in the block (their stage-1 signs)
                    let flips = crate::embed::check_chain(&mut e, &checks)?;
                    Ok((e, flips))
                })
            });
            let mut emb = match r {
                Ok((e, check_flips)) => {
                    let _ = ready_tx.send(Ok(EmbMem { steady, peak, docs: e.can_embed_docs(), streams: e.doc_parallelism(), check_flips }));
                    e
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                    return;
                }
            };
            // the queries queued go first (not those arriving meanwhile, so a stream of queries
            // cannot hold documents back); then documents advance one round (one document per CPU
            // stream, the oldest jobs first, jobs nobody waits for any more dropped); the thread
            // ends when the archive is closed (all senders dropped) and nothing is pending
            type DocJob = (Vec<String>, Vec<Vec<f32>>, mpsc::Sender<Result<Vec<Vec<f32>>>>, Arc<std::sync::atomic::AtomicBool>);
            let par = emb.doc_parallelism();
            let mut docs: std::collections::VecDeque<DocJob> = std::collections::VecDeque::new();
            loop {
                let first = if docs.is_empty() {
                    match rx.recv() {
                        Ok(j) => Some(j),
                        Err(_) => return,
                    }
                } else {
                    None
                };
                let queued: Vec<EmbedJob> = first.into_iter().chain(rx.try_iter()).collect();
                for job in queued {
                    match job {
                        EmbedJob::Query(t, reply) => {
                            let _ = reply.send(emb.embed_query(&t));
                        }
                        EmbedJob::Docs(p, reply, cancel) => docs.push_back((p, Vec::new(), reply, cancel)),
                    }
                }
                docs.retain(|j| !j.3.load(std::sync::atomic::Ordering::Relaxed));
                if docs.is_empty() {
                    continue;
                }
                let mut round: Vec<String> = Vec::with_capacity(par);
                let mut owners: Vec<usize> = Vec::with_capacity(par);
                for (ji, job) in docs.iter().enumerate() {
                    let room = par - round.len();
                    if room == 0 {
                        break;
                    }
                    let (from, to) = (job.1.len(), (job.1.len() + room).min(job.0.len()));
                    round.extend(job.0[from..to].iter().cloned());
                    owners.extend(std::iter::repeat_n(ji, to - from));
                }
                match emb.embed_docs(&round) {
                    Ok(v) => {
                        for (ji, x) in owners.into_iter().zip(v) {
                            docs[ji].1.push(x);
                        }
                    }
                    Err(e) => {
                        // every job with a document in the failed round fails
                        for _ in 0..owners.last().map_or(0, |&j| j + 1) {
                            let (_, _, reply, _) = docs.pop_front().unwrap();
                            let _ = reply.send(Err(anyhow!("{e:#}")));
                        }
                        continue;
                    }
                }
                while docs.front().is_some_and(|j| j.1.len() == j.0.len()) {
                    let (_, out, reply, _) = docs.pop_front().unwrap();
                    let _ = reply.send(Ok(out));
                }
            }
        })?;
        let mem = ready_rx.recv().map_err(|_| anyhow!("the embedder thread stopped while starting"))??;
        Ok((tx, mem))
    }

    fn embed_inner(&self, text: &str) -> Result<Vec<f32>> {
        let (rtx, rrx) = mpsc::channel();
        self.embedder()?.send(EmbedJob::Query(text.to_string(), rtx)).map_err(|_| anyhow!("the embedder is not available"))?;
        rrx.recv().map_err(|_| anyhow!("the embedder did not answer"))?
    }

    /// The vectors of document prompts (P4 reranking), waiting at most until `deadline`: Ok(None)
    /// when it passes first (the embedder then drops what is left of the job).
    fn embed_docs(&self, prompts: Vec<String>, deadline: Option<Instant>) -> Result<Option<Vec<Vec<f32>>>> {
        if prompts.is_empty() {
            return Ok(Some(Vec::new()));
        }
        let (rtx, rrx) = mpsc::channel();
        let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
        self.embedder()?.send(EmbedJob::Docs(prompts, rtx, cancel.clone())).map_err(|_| anyhow!("the embedder is not available"))?;
        let got = match deadline {
            Some(d) => match rrx.recv_timeout(d.saturating_duration_since(Instant::now())) {
                Ok(r) => r,
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                    return Ok(None);
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => return Err(anyhow!("the embedder did not answer")),
            },
            None => rrx.recv().map_err(|_| anyhow!("the embedder did not answer"))?,
        };
        got.map(Some)
    }

    /// Searches of this archive can be reranked (P4): a compact block with prompt v2 and an
    /// embedder that embeds documents. Loads the embedder if needed.
    fn can_rerank(&self) -> Result<bool> {
        let Some(b) = &self.eng.block else { return Ok(false) };
        if b.manifest()["vec"]["prompt_version"].as_u64() != Some(2) {
            return Ok(false);
        }
        self.embedder()?;
        Ok(self.emb_mem.lock().unwrap_or_else(|e| e.into_inner()).is_some_and(|m| m.docs))
    }

    /// Vector of a query with EmbeddingGemma 2 (the same one semantic search uses).
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        check_text(text)?;
        let _t = self.adm.enter(0)?;
        self.embed_inner(text)
    }

    pub fn search(&self, req: &SearchRequest) -> Result<SearchResponse> {
        let q = make_query(req)?;
        let t0 = Instant::now();
        // the deadline counts from the call: waiting for a turn gives up at it too (BusyError)
        let deadline = req.deadline_ms.map(|d| t0 + Duration::from_secs_f64(d / 1e3));
        let cfg = {
            let d = req.rerank_depth.map_or_else(crate::rerank::RerankCfg::default, crate::rerank::RerankCfg::with_depth);
            crate::rerank::RerankCfg { w: req.w_bm25.unwrap_or(d.w), c: req.c_bm25.unwrap_or(d.c), ..d }
        };
        // a search that may be reranked declares the reranker's first round of stage 1 and the
        // prefixes, prompts and vectors of its candidates (about 8 KiB each)
        let may_rerank = req.rerank != Some(false) && self.eng.block.as_ref().is_some_and(|b| b.manifest()["vec"]["prompt_version"].as_u64() == Some(2));
        let cost = if may_rerank {
            let wide = Query { k: q.k.max(cfg.kv), ..q.clone() };
            search_cost(q.k, req.full) + self.eng.search_cost(&wide) + (cfg.kv + cfg.kb_keyword) as u64 * (8 << 10)
        } else {
            search_cost(q.k, req.full) + self.eng.search_cost(&q)
        };
        let turn = self.adm.enter_until(cost, deadline)?;
        let t_q = t0.elapsed();
        let has_text = !q.text.trim().is_empty();
        // P4: rerank when asked (or by default when the archive can); a keyword query needs the
        // query vector only for its frequent words
        let rerank = has_text
            && match req.rerank {
                Some(false) => false,
                Some(true) => {
                    if !self.can_rerank()? {
                        return Err(anyhow::Error::new(QueryError("rerank not available: it needs a compact block with v2 prompts and an embedder with dynamic shapes".into())));
                    }
                    true
                }
                None => self.can_rerank().unwrap_or(false),
            };
        let frequent = rerank && q.mode == Mode::Keyword && self.eng.query_terms(&q.text)?.iter().any(|t| t.post == crate::bm25::TermPost::Frequent);
        let qv = if has_text && (q.mode != Mode::Keyword || frequent) { Some(self.embed_inner(&q.text)?) } else { None };
        let t_emb = t0.elapsed();
        let (hits, prefixes, report, warning) = match (&qv, rerank) {
            (Some(v), true) => {
                let o = crate::rerank::search(&self.eng, &q, v, &cfg, self, deadline, &|extra| turn.try_grow(extra))?;
                (o.hits, o.prefixes, Some(o.report), o.warning)
            }
            _ => (self.eng.search_with(&q, qv.as_deref(), &|extra| turn.try_grow(extra))?, Default::default(), None, None),
        };
        let t_s = t0.elapsed();
        let results = self.eng.fetch_with(&hits, req.full, &prefixes)?;
        let t_all = t0.elapsed();
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        Ok(SearchResponse {
            query: req.query.clone(),
            k: q.k,
            results,
            ms: Timings { queue: ms(t_q), embedding: ms(t_emb - t_q), search: ms(t_s - t_emb), fetch: ms(t_all - t_s), total: ms(t_all) },
            rerank: report,
            warning,
        })
    }

    /// A full item by HN id (verbatim text: `html` is HN's original HTML).
    pub fn get(&self, id: u32) -> Result<Option<serde_json::Value>> {
        let _turn = self.adm.enter(GET_COST)?;
        let it = self.eng.get(id)?;
        Ok(match it {
            Some(it) => {
                let mut v = serde_json::to_value(&it)?;
                v["text_plain"] = serde_json::Value::String(htmlx::to_plain(&it.text));
                Some(v)
            }
            None => None,
        })
    }

    /// Memory breakdown: the ledger (limit, used, peak and per class), the plan, what is loaded,
    /// the caches, the embedder, admission control and the process as the OS sees it.
    pub fn memory(&self) -> serde_json::Value {
        let emb = *self.emb_mem.lock().unwrap_or_else(|e| e.into_inner());
        serde_json::json!({
            "budget": self.budget.report(),
            "plan": self.plan,
            "engine": self.eng.memory_report(),
            "embedder": {
                "loaded": self.emb.get().is_some(),
                "charged": self.emb_charge.lock().unwrap_or_else(|e| e.into_inner()).bytes(),
                "measured": emb,
            },
            "admission": self.adm.report(),
            "process": crate::mem::process_memory(),
        })
    }

    pub fn stats(&self) -> serde_json::Value {
        let segs: Vec<serde_json::Value> = self
            .eng
            .segs
            .iter()
            .map(|sg| {
                let m = &sg.s.man;
                serde_json::json!({
                    "name": m.name, "docs": m.n_docs, "tier": m.text_tier, "vec_bits": m.vec_bits,
                    "date_min": dates::fmt(m.time_min), "date_max": dates::fmt(m.time_max),
                })
            })
            .collect();
        serde_json::json!({
            "path": self.root.display().to_string(),
            "models": self.models.display().to_string(),
            "docs": self.eng.total_docs,
            "segments": segs,
            "ram_budget": self.plan.limit,
            "block": self.eng.block.as_ref().map(|b| serde_json::json!({
                "path": b.path.display().to_string(), "snapshot": b.snapshot_hex(), "writer": b.writer,
                "parts": b.parts().len(), "bytes": std::fs::metadata(&b.path).map(|m| m.len()).unwrap_or(0),
                "manifest_snapshot": b.manifest()["snapshot"],
            })),
            "embedder_loaded": self.emb.get().is_some(),
            "memory": self.memory(),
        })
    }
}

impl crate::rerank::DocEmbed for Archive {
    fn embed(&self, docs: &[(u32, String)]) -> Result<Vec<Vec<f32>>> {
        Ok(self.embed_docs(docs.iter().map(|d| d.1.clone()).collect(), None)?.unwrap_or_default())
    }

    fn embed_until(&self, docs: &[(u32, String)], deadline: Option<Instant>) -> Result<Option<Vec<Vec<f32>>>> {
        self.embed_docs(docs.iter().map(|d| d.1.clone()).collect(), deadline)
    }

    fn batch(&self) -> usize {
        self.emb_mem.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map_or(1, |m| m.streams)
    }
}
