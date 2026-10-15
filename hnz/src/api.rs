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
//! - The LM decompresses one batch at a time in the whole process (a lock inside `Engine`, only
//!   while decompressing): zstd-tier or cached queries do not wait for LM-tier ones.
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

/// Una búsqueda. Fechas como en la CLI: "2015", "2015-06", "2015-06-01" (ambos extremos inclusive).
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct SearchRequest {
    pub query: String,
    pub k: usize,
    pub since: Option<String>,
    pub until: Option<String>,
    /// límites exactos en segundos epoch UTC (ambos inclusive); excluyen a since/until
    pub since_ts: Option<i64>,
    pub until_ts: Option<i64>,
    /// story, comment, job, poll, pollopt (vacío = todos)
    pub types: Vec<String>,
    pub author: Option<String>,
    /// hybrid | vector | keyword
    pub mode: String,
    /// texto completo también en el nivel LM (si no, fragmento de 64 tokens)
    pub full: bool,
    pub include_dead: bool,
    pub w_bm25: f32,
    pub c_bm25: f32,
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
            w_bm25: 0.15,
            c_bm25: 20.0,
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
}

/// Error en los argumentos de una consulta (en Python: ValueError).
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
        return Err(anyhow::Error::new(QueryError(format!("consulta demasiado larga: {} bytes (máximo {MAX_QUERY_BYTES})", text.len()))));
    }
    Ok(())
}

/// Valida una consulta y la traduce a la del motor.
pub fn make_query(r: &SearchRequest) -> Result<Query> {
    let bad = |m: String| anyhow::Error::new(QueryError(m));
    check_text(&r.query)?;
    if r.k == 0 || r.k > MAX_K {
        return Err(bad(format!("k debe estar entre 1 y {MAX_K} (es {})", r.k)));
    }
    if !(r.c_bm25.is_finite() && r.c_bm25 > 0.0 && r.w_bm25.is_finite() && r.w_bm25 >= 0.0) {
        return Err(bad(format!("c_bm25 debe ser > 0 y w_bm25 >= 0 (son {} y {})", r.c_bm25, r.w_bm25)));
    }
    let mut types = Vec::new();
    for t in &r.types {
        types.push(match t.as_str() {
            "story" => 1u8,
            "comment" => 2,
            "poll" => 3,
            "pollopt" => 4,
            "job" => 5,
            o => return Err(bad(format!("tipo desconocido: {o} (story, comment, job, poll, pollopt)"))),
        });
    }
    let mode = match r.mode.as_str() {
        "hybrid" => Mode::Hybrid,
        "vector" => Mode::Vector,
        "keyword" => Mode::Keyword,
        o => return Err(bad(format!("modo desconocido: {o} (hybrid, vector, keyword)"))),
    };
    let date = |s: &Option<String>, end: bool, name: &str| -> Result<Option<i64>> {
        match s {
            Some(x) => dates::parse(x, end).map(Some).ok_or_else(|| bad(format!("fecha {name} inválida: {x:?} (2015, 2015-06 o 2015-06-01)"))),
            None => Ok(None),
        }
    };
    if (r.since.is_some() && r.since_ts.is_some()) || (r.until.is_some() && r.until_ts.is_some()) {
        return Err(bad("indique since o since_ts (y until o until_ts), no ambos".into()));
    }
    let from = match r.since_ts {
        Some(t) => Some(t),
        None => date(&r.since, false, "since")?,
    };
    let to = match r.until_ts {
        Some(t) => Some(t.checked_add(1).ok_or_else(|| bad("until_ts fuera de rango".into()))?),
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
        w_bm25: r.w_bm25,
        c_bm25: r.c_bm25,
        vec_cand: None,
    })
}

type EmbedReply = mpsc::Sender<Result<Vec<f32>>>;

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
    emb: OnceLock<mpsc::Sender<(String, EmbedReply)>>,
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
    bail!("no se encuentran los modelos junto a {} (indique models=...)", root.display())
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
            o => return Err(anyhow::Error::new(QueryError(format!("verify debe ser header, quick o full (es {o:?})")))),
        };
        // a compact block (P3), or a folder of segments
        let block = match crate::block::Block::locate(root) {
            Some(p) => Some(crate::block::Block::open(&p)?),
            None => None,
        };
        // first, the folder must have segments (the most likely error: wrong folder)
        let has_segments = root.is_dir()
            && std::fs::read_dir(root)
                .with_context(|| format!("no se puede leer {}", root.display()))?
                .filter_map(|e| e.ok())
                .any(|e| e.path().join("manifest.json").exists());
        if block.is_some() && has_segments {
            bail!("{} contiene un bloque (hn.hnzb) y también carpetas de segmentos: use una cosa u otra", root.display());
        }
        if block.is_none() && !has_segments {
            let hint = if root.join("manifest.json").exists() {
                " (esa carpeta ES un segmento: indique la carpeta que contiene los segmentos)"
            } else {
                ""
            };
            bail!("no hay segmentos (carpetas con manifest.json) en {}{hint}", root.display());
        }
        let timeout = Duration::try_from_secs_f64(opts.timeout).ok().filter(|_| opts.max_concurrent > 0).ok_or_else(|| {
            anyhow::Error::new(QueryError(format!(
                "timeout debe ser un número de segundos >= 0 (y razonable) y max_concurrent >= 1 (son {} y {})",
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
            None => Engine::open_with(root, &models, eopts).with_context(|| format!("abriendo {}", root.display()))?,
        };
        if eng.segs.is_empty() {
            let hint = if root.join("manifest.json").exists() {
                " (esa carpeta ES un segmento: indique la carpeta que contiene los segmentos)"
            } else {
                ""
            };
            bail!("no hay segmentos (carpetas con manifest.json) en {}{hint}", root.display());
        }
        let emb_est = match &block {
            Some(b) => {
                let bin = b.external().iter().find(|e| e.name == "embed.bin").map_or(0, |e| e.len);
                let tok = b.part("model/embed.tok").map_or(0, |p| p.raw_len);
                bin * 7 / 5 + 4 * tok + 32 * MB
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
            eng.promote(si, p).with_context(|| format!("cargando {} de {}", p.name(), eng.segs[si].s.man.name))?;
        }
        eng.set_lm_pool(plan.lm_pool);
        // with a persistent cache, its RAM index takes 1/8 of the cache share (newest entries kept)
        if let Some(d) = &opts.cache_dir {
            let max_entries = (plan.cache / 8 / crate::cache::L2_ENTRY_BYTES) as usize;
            match crate::cache::L2::open_limited(&d.join("hnz-cache.bin"), opts.cache_dir_bytes, max_entries) {
                Ok(l2) => eng.set_l2(l2),
                Err(e) => eprintln!("aviso: caché persistente desactivada ({e})"),
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

    fn embedder(&self) -> Result<mpsc::Sender<(String, EmbedReply)>> {
        if let Some(tx) = self.emb.get() {
            return Ok(tx.clone());
        }
        let mut last = self.emb_init.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(tx) = self.emb.get() {
            return Ok(tx.clone()); // another thread loaded it while we waited
        }
        if let Some((t, msg, kind)) = last.as_ref() {
            if t.elapsed() < EMB_RETRY {
                let m = format!("el embedder no se pudo cargar hace {:.0} s: {msg}", t.elapsed().as_secs_f64());
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
            eprintln!("aviso: el embedder llegó a {} al cargarse (reservados {})", human(mem.peak), human(reserved));
        }
        c.resize(keep);
        let cache = self.eng.cache_bytes();
        let new_cache = if keep <= reserved { cache + (reserved - keep) } else { cache.saturating_sub(keep - reserved).max(plan::CACHE_MIN) };
        self.eng.set_cache_bytes(new_cache);
        *self.emb_mem.lock().unwrap_or_else(|e| e.into_inner()) = Some(mem);
    }

    fn spawn_embedder(&self) -> Result<(mpsc::Sender<(String, EmbedReply)>, EmbMem)> {
        let libs = self.ov_libs.clone().context("falta la carpeta de OpenVINO (ov_libs o la variable HNZ_OV_LIBS)")?;
        let models = self.models.clone();
        // a block: its IR files (checked against their hashes) and the tokenizer inside it, read
        // in the embedder's thread so the measured load includes them and frees the tokenizer
        let block = self.eng.block.clone();
        let (tx, rx) = mpsc::channel::<(String, EmbedReply)>();
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
                    Ok(e)
                })
            });
            let mut emb = match r {
                Ok(e) => {
                    let _ = ready_tx.send(Ok(EmbMem { steady, peak }));
                    e
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                    return;
                }
            };
            // ends only when the archive is closed (all senders dropped)
            for (text, reply) in rx {
                let _ = reply.send(emb.embed_query(&text));
            }
        })?;
        let mem = ready_rx.recv().map_err(|_| anyhow!("el hilo del embedder terminó al arrancar"))??;
        Ok((tx, mem))
    }

    fn embed_inner(&self, text: &str) -> Result<Vec<f32>> {
        let (rtx, rrx) = mpsc::channel();
        self.embedder()?.send((text.to_string(), rtx)).map_err(|_| anyhow!("el embedder no está disponible"))?;
        rrx.recv().map_err(|_| anyhow!("el embedder no respondió"))?
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
        let turn = self.adm.enter(search_cost(q.k, req.full) + self.eng.search_cost(&q))?;
        let t_q = t0.elapsed();
        let qv = if q.mode != Mode::Keyword && !q.text.trim().is_empty() { Some(self.embed_inner(&q.text)?) } else { None };
        let t_emb = t0.elapsed();
        let hits = self.eng.search_with(&q, qv.as_deref(), &|extra| turn.try_grow(extra))?;
        let t_s = t0.elapsed();
        let results = self.eng.fetch(&hits, req.full)?;
        let t_all = t0.elapsed();
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        Ok(SearchResponse {
            query: req.query.clone(),
            k: q.k,
            results,
            ms: Timings { queue: ms(t_q), embedding: ms(t_emb - t_q), search: ms(t_s - t_emb), fetch: ms(t_all - t_s), total: ms(t_all) },
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
