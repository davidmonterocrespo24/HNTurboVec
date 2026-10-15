//! Motor de consulta: reparte la búsqueda entre segmentos (en paralelo), aplica filtros de fecha
//! como rangos de docIDs, fusiona vectores + BM25 y arma los resultados con el contexto del hilo.
//! Los resultados de segmentos del nivel LM se descomprimen TODOS en un mismo lote.

use crate::blob::{Blob, Residency};
use crate::bm25::{Bm25, TermPost};
use crate::cache::GenCache;
use crate::item::Item;
use crate::lm::store::{LmCodec, LmStore};
use crate::mem::{Budget, Charge, Class, MB};
use crate::meta::{MetaRow, META_BLOCK};
use crate::plan::Part;
use crate::segment::{Segment, TextSource};
use crate::textstore::TextRec;
use crate::vecs::{QueryLut, VecIndex};
use crate::{dates, htmlx, tokenize};
use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Tokens que se decodifican por resultado del nivel LM en modo fragmento.
pub const SNIPPET_TOKENS: usize = 64;
/// máximo de resultados por consulta (protege la memoria del servidor)
pub const MAX_K: usize = 500;
/// At most this many query terms count for BM25, the rarest ones (each term opens a cursor per
/// segment; common terms add little to the score).
pub const MAX_TERMS: usize = 64;

/// Candidates of one search round in a segment with `n` documents in range: (kc = scored
/// candidates, c1 = stage-1 vector candidates). `mult` widens the round (x4 each time).
fn round_sizes(q: &Query, k: usize, n: usize, mult: usize, c1_min: usize) -> (usize, usize) {
    let mut kc = (k * 20).max(200) * mult;
    if !q.types.is_empty() {
        kc *= 5; // stories ~14 % de los ítems
    }
    let kc = kc.min(n);
    let c1 = q.vec_cand.unwrap_or((kc * 10).max(c1_min)).max(kc).min(n);
    (kc, c1)
}

/// Working memory of one round in one segment: stage-1 Hamming pairs (with growth slack) and the
/// scored list, plus BM25 heap, merged ids and scored hits of up to 2 x kc candidates.
fn round_cost(kc: usize, c1: usize) -> u64 {
    c1 as u64 * 24 + kc as u64 * 64
}

/// Un solo lote del LM a la vez en todo el proceso: su pool de hilos es global y dos lotes a la vez
/// solo competirían por los mismos núcleos. Lo demás (metadatos, nivel zstd, cachés) va en paralelo.
use crate::lm::jobs::{Decoded, Want};
use crate::lm::service::{Job, LmService, Prio};

/// Bytes counted per cached record: strings + hash-table entry with growth slack.
fn rec_weight(r: &TextRec) -> u64 {
    256 + (r.title.len() + r.url.len() + r.text.len()) as u64
}

fn title_weight(t: &String) -> u64 {
    160 + t.len() as u64
}

fn prefix_weight(p: &Arc<crate::rerank::Prefix>) -> u64 {
    96 + p.bytes.len() as u64
}

pub struct Seg {
    pub s: Segment,
    pub bm25: Option<Bm25>,
    pub vec: Option<VecIndex>,
}

impl Seg {
    /// File behind a loadable part (None if the segment does not have it).
    pub fn blob(&self, p: Part) -> Option<&Blob> {
        match p {
            Part::Vectors => self.vec.as_ref().map(|v| v.blob()),
            Part::Meta => Some(self.s.meta.blob()),
            Part::Postings => self.bm25.as_ref().map(|b| b.blob()),
            Part::Text => Some(match &self.s.text {
                TextSource::Lm(l) => l.blob(),
                TextSource::Zstd(t) => t.blob(),
            }),
        }
    }

    fn blob_mut(&mut self, p: Part) -> Option<&mut Blob> {
        match p {
            Part::Vectors => self.vec.as_mut().map(|v| v.blob_mut()),
            Part::Meta => Some(self.s.meta.blob_mut()),
            Part::Postings => self.bm25.as_mut().map(|b| b.blob_mut()),
            Part::Text => Some(match &mut self.s.text {
                TextSource::Lm(l) => l.blob_mut(),
                TextSource::Zstd(t) => t.blob_mut(),
            }),
        }
    }

    /// RAM used by this segment (fixed structures + loaded parts).
    pub fn ram_bytes(&self) -> usize {
        self.s.ram_bytes() + self.bm25.as_ref().map_or(0, |b| b.ram_bytes()) + self.vec.as_ref().map_or(0, |v| v.ram_bytes())
    }

    /// RAM of what is always loaded (block indexes, BM25 dictionary and norms, vector bitmap…).
    pub fn fixed_bytes(&self) -> usize {
        self.ram_bytes() - Part::ALL.iter().filter_map(|&p| self.blob(p)).map(|b| b.resident_bytes()).sum::<usize>()
    }
}

/// How to open the engine.
pub struct EngineOptions {
    /// RAM budget that loaded indexes, models and caches are charged to
    pub budget: Arc<Budget>,
    /// which segment parts are loaded at open (the planner may load more afterwards)
    pub residency: Residency,
    /// persistent disk cache of decompressed LM records (file, max size); None = off
    pub l2: Option<(PathBuf, u64)>,
    /// RAM for the caches of decompressed text
    pub cache_bytes: u64,
    /// memory cap of the LM batch (u64::MAX = no cap)
    pub lm_pool: u64,
}

impl EngineOptions {
    /// CLI defaults: no budget, vectors in RAM and everything else read from disk, L2 cache at
    /// `<archive>/_cache_l2.bin` (512 MB) and 256 MB of RAM caches.
    pub fn cli(root: &Path) -> Self {
        Self { budget: Budget::new(None), residency: Residency::DEFAULT, l2: Some((root.join("_cache_l2.bin"), 512 * MB)), cache_bytes: 256 * MB, lm_pool: u64::MAX }
    }
}

pub struct Engine {
    pub segs: Vec<Seg>,
    pub total_docs: usize,
    /// modelos LM de los segmentos fríos (por nombre de fichero)
    pub lms: HashMap<String, Arc<LmCodec>>,
    /// the LM decoding service of each model (started when first needed): one continuous batch
    /// shared by every query, within the LM pool
    lm_svc: Mutex<HashMap<String, Arc<LmService>>>,
    /// RAM cache of fully decompressed LM-tier records
    cache: Mutex<GenCache<(usize, u32), TextRec>>,
    /// caché L2 (disco) de registros del nivel LM ya descomprimidos
    pub l2: Option<crate::cache::L2>,
    /// already decoded titles of LM-tier stories (thread context)
    titles: Mutex<GenCache<(usize, u32), String>>,
    /// blocks: decoded prompt prefixes of reranked candidates (P4), by document
    pub prefixes: Mutex<GenCache<u32, Arc<crate::rerank::Prefix>>>,
    seg_keys: Vec<u64>,
    /// por segmento, 1 byte por doc: tipo | dead<<3 | deleted<<4 (filtros sin descomprimir nada)
    flags: Vec<Vec<u8>>,
    pub budget: Arc<Budget>,
    /// memory cap of one LM batch (charged to the budget as working memory)
    lm_pool: u64,
    lm_charge: Charge,
    index_charge: Charge,
    model_charge: Charge,
    /// minimum stage-1 vector candidates (2,000 for segments; the block's manifest says its own)
    c1_min: usize,
    /// the compact block this engine reads, if it is one (P3)
    pub block: Option<Arc<crate::block::Block>>,
    /// blocks: the prompt prefix of each document with a vector whose record has no LM tokens
    prefix_fb: HashMap<u32, (u8, String)>,
    _prefix_charge: Charge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Hybrid,
    Vector,
    Keyword,
}

#[derive(Clone, Debug)]
pub struct Query {
    pub text: String,
    pub k: usize,
    /// epoch inclusive
    pub from: Option<i64>,
    /// epoch exclusivo
    pub to: Option<i64>,
    pub mode: Mode,
    /// tipos permitidos (vacío = todos): 1 story, 2 comment, 3 poll, 4 pollopt, 5 job
    pub types: Vec<u8>,
    pub by: Option<String>,
    pub include_dead: bool,
    /// fusión: score = v + w * b / (b + c)
    pub w_bm25: f32,
    pub c_bm25: f32,
    /// candidatos de la etapa 1 (Hamming) del índice vectorial; None = automático
    pub vec_cand: Option<usize>,
}

impl Default for Query {
    fn default() -> Self {
        Self {
            text: String::new(),
            k: 10,
            from: None,
            to: None,
            mode: Mode::Hybrid,
            types: Vec::new(),
            by: None,
            include_dead: false,
            w_bm25: 0.15,
            c_bm25: 20.0,
            vec_cand: None,
        }
    }
}

/// A query word for reranking (P4): its global idf and its postings (`TermPost::Frequent` for
/// the words too common to have postings in a block).
#[derive(Clone, Debug)]
pub struct QTerm {
    pub text: String,
    pub idf: f32,
    pub post: TermPost,
}

/// A stage-1 candidate (P4): its vector score s1 (0 without a vector).
#[derive(Clone, Copy, Debug)]
pub struct Cand {
    pub doc: u32,
    pub s1: f32,
    pub has_vec: bool,
}

#[derive(Clone, Debug)]
pub struct Hit {
    pub seg: usize,
    pub doc: u32,
    pub score: f32,
    pub v: f32,
    pub b: f32,
}

#[derive(Serialize, Debug)]
pub struct ResultItem {
    pub rank: usize,
    pub id: u32,
    pub kind: &'static str,
    pub by: String,
    pub date: String,
    pub time: i64,
    pub points: i32,
    pub title: String,
    pub url: String,
    /// texto plano (para leer / RAG)
    pub text: String,
    /// HTML original de HN (literal); vacío si `partial`
    pub html: String,
    /// true = solo el principio del texto (nivel LM en modo fragmento); el completo en /item/<id>
    pub partial: bool,
    /// "zstd" o "lm"
    pub tier: &'static str,
    pub parent_id: Option<u32>,
    pub story_id: Option<u32>,
    pub story_title: Option<String>,
    pub score: f32,
    pub vscore: f32,
    pub bm25: f32,
    pub link: String,
}

pub fn kind_name(t: u8) -> &'static str {
    match t {
        1 => "story",
        2 => "comment",
        3 => "poll",
        4 => "pollopt",
        5 => "job",
        _ => "?",
    }
}

/// The versioned definitions a block's bytes depend on (`manifest[section][key]`), and the
/// versions this binary reads.
pub const BLOCK_DEFS: &[crate::block::Definition<'static>] = &[
    crate::block::Definition { section: "text", key: "math_version", supported: &[crate::lm::model::MATH_VERSION as u64] },
    crate::block::Definition { section: "text", key: "transform_version", supported: &[1] },
    crate::block::Definition { section: "text", key: "rec_check", supported: &[1] },
    crate::block::Definition { section: "text", key: "framing", supported: &[1] },
    crate::block::Definition { section: "text", key: "group", supported: &[32] },
    crate::block::Definition { section: "meta", key: "row_encoding", supported: &[2] },
    crate::block::Definition { section: "lex", key: "value_layout", supported: &[1] },
    crate::block::Definition { section: "lex", key: "tokenizer_version", supported: &[1] },
    crate::block::Definition { section: "vec", key: "sign_rule", supported: &[1] },
    crate::block::Definition { section: "kids", key: "encoding", supported: &[1] },
];

/// The parts of a block this binary knows, with the versions it reads.
pub const BLOCK_PARTS: &[(&str, &[u32])] = &[
    ("meta/index", &[1]),
    ("meta/flags", &[1]),
    ("meta/late", &[1]),
    ("meta/authors", &[1]),
    ("meta/blocks", &[1]),
    ("meta/kids", &[1]),
    ("meta/kids.idx", &[1]),
    ("text/idx", &[1]),
    ("text/gsum", &[1]),
    ("text/prefix_fallback", &[1]),
    ("text/groups", &[1]),
    ("vec/has", &[1]),
    ("vec/codes", &[1]),
    ("lex/fst", &[1]),
    ("lex/norms", &[1]),
    ("lex/post", &[1]),
    ("model/lm", &[1]),
    ("model/lm.tok", &[1]),
    ("model/embed.tok", &[1]),
];

impl Engine {
    /// Opens all segments in `root` (ignoring half-built ones and `retier` leftovers) and loads every
    /// distinct LM model used by cold-tier segments. CLI defaults (`EngineOptions::cli`).
    pub fn open(root: &Path, models: &Path) -> Result<Self> {
        if let Some(p) = crate::block::Block::locate(root) {
            return Self::open_block(crate::block::Block::open(&p)?, EngineOptions::cli(root));
        }
        Self::open_with(root, models, EngineOptions::cli(root))
    }

    /// Opens with explicit residency, caches and budget (the library plans them from `ram`).
    pub fn open_with(root: &Path, models: &Path, opts: EngineOptions) -> Result<Self> {
        let res = opts.residency;
        let dirs: Vec<PathBuf> = std::fs::read_dir(root)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.join("manifest.json").exists() && !crate::segment::is_transient_dir(p))
            .collect();
        let mut segs = Vec::new();
        for d in dirs {
            let s = Segment::open_with(&d, res)?;
            let bm25 = if d.join("bm25.fst").exists() { Some(Bm25::open_with(&d, res.postings)?) } else { None };
            let vec = if s.man.vec_bits > 0 && d.join("vec.bin").exists() { Some(VecIndex::open_with(&d, s.man.vec_bits, s.man.n_docs, res.vectors)?) } else { None };
            segs.push(Seg { s, bm25, vec });
        }
        segs.sort_by_key(|x| x.s.man.id_min);
        // los segmentos no pueden solaparse (p.ej. seg-2015 junto a seg-2015-06): duplicarían
        // resultados y romperían la búsqueda por id
        for w in segs.windows(2) {
            if w[1].s.man.id_min <= w[0].s.man.id_max {
                bail!(
                    "los segmentos {} (ids {}..{}) y {} (ids {}..{}) se solapan: deje solo uno de los dos en {}",
                    w[0].s.man.name,
                    w[0].s.man.id_min,
                    w[0].s.man.id_max,
                    w[1].s.man.name,
                    w[1].s.man.id_min,
                    w[1].s.man.id_max,
                    root.display()
                );
            }
        }
        // LM models: their RAM is measured while loading (at least the size of the file)
        let mut lms: HashMap<String, Arc<LmCodec>> = HashMap::new();
        let mut model_bytes = 0u64;
        for sg in &segs {
            if let (Some(f), TextSource::Lm(l)) = (&sg.s.man.lm_model, &sg.s.text) {
                if !lms.contains_key(f) {
                    let path = models.join(f);
                    let (c, steady, _) = crate::mem::measure(|| LmCodec::load(&path));
                    let c = c.with_context(|| format!("modelo LM {f}"))?;
                    model_bytes += steady.max(std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0));
                    lms.insert(f.clone(), Arc::new(c));
                }
                l.check_model(&lms[f]).with_context(|| format!("segmento {}", sg.s.man.name))?;
            }
        }
        let l2 = match (&opts.l2, lms.is_empty()) {
            (Some((path, cap)), false) => match crate::cache::L2::open(path, *cap) {
                Ok(c) => {
                    c.set_charge(opts.budget.charge(Class::Cache, 0));
                    Some(c)
                }
                Err(e) => {
                    eprintln!("aviso: caché L2 desactivada ({e})");
                    None
                }
            },
            _ => None,
        };
        // columna de banderas en RAM (1 B/doc), construida en paralelo una vez al abrir
        let flags: Vec<Vec<u8>> = segs.iter().map(|sg| crate::meta::load_flags(&sg.s.dir, &sg.s.meta)).collect::<Result<_>>()?;
        Ok(Self::assemble(segs, lms, model_bytes, flags, l2, opts, 2000, None))
    }

    /// Opens a compact block (P3) as one segment: checks the versions of every definition its bytes
    /// depend on, builds the readers over its parts (each part in RAM or read from disk, per
    /// `opts.residency`), and loads the LM model stored in the block.
    pub fn open_block(block: Arc<crate::block::Block>, opts: EngineOptions) -> Result<Self> {
        use crate::block::incompatible;
        block.check_parts(BLOCK_PARTS)?;
        block.check_compat(BLOCK_DEFS)?;
        let res = opts.residency;
        let m = block.manifest().clone();
        let n = block.n_docs as usize;
        let num = |v: &serde_json::Value| v.as_u64().unwrap_or(0);
        let authors = crate::meta::Authors::from_text(String::from_utf8(block.read_part("meta/authors")?).map_err(|_| crate::block::corrupt("meta/authors"))?);
        let meta = crate::meta::MetaStore::from_parts(block.blob("meta/blocks", res.meta)?, &block.read_part("meta/index")?, authors, &block.read_part("meta/late")?, n)?;
        let t = &m["text"];
        let info = crate::lm::store::LmInfo {
            math: num(&t["math_version"]) as u32,
            model_id: t["model_id"].as_str().unwrap_or_default().to_string(),
            n_docs: n,
            group: num(&t["group"]) as usize,
            lm_records: num(&t["lm_records"]) as usize,
            zstd_records: num(&t["zstd_records"]) as usize,
            tokens: num(&t["tokens"]),
            rec_check: true,
        };
        let lm = LmStore::from_parts(block.blob("text/groups", res.text)?, &block.read_part("text/idx")?, &block.read_part("text/gsum")?, info)?;
        // intact bytes (hashed) this binary cannot parse are an incompatible block, not a damaged one
        let (codec, steady, _) = crate::mem::measure(|| -> Result<LmCodec> {
            let bytes = block.read_part("model/lm")?;
            LmCodec::from_parts(bytes, None).map_err(|e| incompatible(format!("model/lm: {e:#}")))
        });
        let codec = codec?;
        if codec.model.id != lm.info.model_id {
            return Err(incompatible(format!("el modelo LM del bloque es {} y el manifiesto dice {}", codec.model.id, lm.info.model_id)));
        }
        let model_bytes = steady.max(block.require("model/lm")?.len);
        let li: crate::bm25::Bm25Info = serde_json::from_value(m["lex"]["info"].clone()).map_err(|e| incompatible(format!("lex.info: {e}")))?;
        let (fst, norms) = (block.read_part("lex/fst")?, block.read_part("lex/norms")?);
        let bm25 = Bm25::from_parts(fst, block.blob("lex/post", res.postings)?, norms, li).map_err(|e| incompatible(format!("lex: {e:#}")))?;
        let bits = num(&m["vec"]["bits"]) as usize;
        let vec = VecIndex::from_parts(&block.read_part("vec/has")?, block.blob("vec/codes", res.vectors)?, bits, n)?;
        let flags = block.read_part("meta/flags")?;
        if flags.len() != n {
            return Err(crate::block::corrupt("meta/flags: tamaño"));
        }
        let s = &m["snapshot"];
        let man = crate::segment::Manifest {
            version: 3,
            name: "block".into(),
            n_docs: n,
            deleted_skipped: num(&s["n_deleted"]) as usize,
            build_id: block.snapshot as u64,
            id_min: num(&s["id_min"]) as u32,
            id_max: num(&s["id_max"]) as u32,
            time_min: s["time_min"].as_i64().unwrap_or(0),
            time_max: s["time_max"].as_i64().unwrap_or(0),
            text_tier: "lm".into(),
            text_block: 0,
            text_level: 0,
            lm_model: Some("block".into()),
            transformed_pct: 0.0,
            has_bm25: true,
            vec_bits: bits,
        };
        let seg = Segment { dir: block.dir(), man, meta, text: TextSource::Lm(lm) };
        let mut lms = HashMap::new();
        lms.insert("block".to_string(), Arc::new(codec));
        let c1 = num(&m["search"]["c1"]).max(2000) as usize;
        let mut opts = opts;
        opts.l2 = None; // the block folder is read-only data: no disk cache written next to it
        let prefix_fb = match block.part("text/prefix_fallback") {
            Some(_) => crate::lm::store::parse_prefix_fallback(&block.read_part("text/prefix_fallback")?).ok_or_else(|| crate::block::corrupt("text/prefix_fallback"))?,
            None => HashMap::new(),
        };
        let fb_bytes: u64 = prefix_fb.values().map(|v| v.1.capacity() as u64 + 48).sum();
        let mut eng = Self::assemble(vec![Seg { s: seg, bm25: Some(bm25), vec: Some(vec) }], lms, model_bytes, vec![flags], None, opts, c1, Some(block));
        eng._prefix_charge = eng.budget.charge(Class::Index, fb_bytes);
        eng.prefix_fb = prefix_fb;
        Ok(eng)
    }

    #[allow(clippy::too_many_arguments)]
    fn assemble(
        segs: Vec<Seg>,
        lms: HashMap<String, Arc<LmCodec>>,
        model_bytes: u64,
        flags: Vec<Vec<u8>>,
        l2: Option<crate::cache::L2>,
        opts: EngineOptions,
        c1_min: usize,
        block: Option<Arc<crate::block::Block>>,
    ) -> Self {
        let total_docs = segs.iter().map(|x| x.s.man.n_docs).sum();
        // clave de la L2: nombre + identificador de construcción + nº de docs (si el segmento se
        // reconstruye, sus entradas antiguas dejan de coincidir)
        let seg_keys = segs
            .iter()
            .map(|s| crate::cache::seg_key(&s.s.man.name) ^ s.s.man.build_id.rotate_left(13) ^ (s.s.man.n_docs as u64).rotate_left(41))
            .collect();
        let b = &opts.budget;
        let index = segs.iter().map(|sg| sg.ram_bytes() as u64).sum::<u64>() + flags.iter().map(|f| f.capacity() as u64).sum::<u64>();
        // blocks keep a quarter for the reranker's prefixes
        let prefixes_cap = if block.is_some() { opts.cache_bytes / 4 } else { 0 };
        let (titles_cap, records_cap) = (opts.cache_bytes / 8, opts.cache_bytes - opts.cache_bytes / 8 - prefixes_cap);
        Self {
            segs,
            total_docs,
            lms,
            lm_svc: Mutex::new(HashMap::new()),
            cache: Mutex::new(GenCache::new(records_cap, rec_weight, Some(b.charge(Class::Cache, 0)))),
            l2,
            titles: Mutex::new(GenCache::new(titles_cap, title_weight, Some(b.charge(Class::Cache, 0)))),
            prefixes: Mutex::new(GenCache::new(prefixes_cap, prefix_weight, Some(b.charge(Class::Cache, 0)))),
            seg_keys,
            flags,
            lm_pool: opts.lm_pool,
            lm_charge: b.charge(Class::Working, if opts.lm_pool == u64::MAX { 0 } else { opts.lm_pool }),
            index_charge: b.charge(Class::Index, index),
            model_charge: b.charge(Class::Model, model_bytes),
            c1_min,
            block,
            prefix_fb: HashMap::new(),
            _prefix_charge: b.charge(Class::Index, 0),
            budget: opts.budget,
        }
    }

    /// Checks the block's canaries of the given kinds ("short", "long", "fallback", "empty"):
    /// their stored bytes (else `CorruptBlockError`) and, decoded in one batch, their text (else
    /// `IncompatibleBlockError`: intact bytes that this binary or CPU decodes differently).
    pub fn check_canaries(&self, kinds: &[&str]) -> Result<usize> {
        let Some(block) = &self.block else { return Ok(0) };
        let xxh = |b: &[u8]| format!("xxh3-64:{:016x}", twox_hash::XxHash3_64::oneshot(b));
        let cans: Vec<&serde_json::Value> = block.manifest()["canaries"]
            .as_array()
            .map(|a| a.iter().filter(|c| c["kind"].as_str().is_some_and(|k| kinds.contains(&k))).collect())
            .unwrap_or_default();
        let TextSource::Lm(lm) = &self.segs[0].s.text else { return Ok(0) };
        let mut wants = Vec::new();
        for c in &cans {
            let d = c["doc"].as_u64().unwrap_or(u64::MAX) as usize;
            if d >= self.total_docs || lm.raw(d).map(|r| xxh(&r)).ok().as_deref() != c["rec"].as_str() {
                return Err(crate::block::corrupt(format!("canario {}: bytes del registro", c["id"])));
            }
            wants.push((0usize, d, None::<usize>, false));
        }
        let jobs: Vec<Job> = wants.iter().map(|w| Ok(Job { rec: lm.raw(w.1)?.into_owned(), has_check: true, want: Want::Full })).collect::<Result<_>>()?;
        let n = jobs.len();
        let dec = self.lm_service("block")?.run(jobs, Prio::Interactive).map_err(|e| crate::block::incompatible(format!("canarios: {e:#}")))?;
        for (c, d) in cans.iter().zip(dec) {
            let Decoded::Text(r, _) = d else { return Err(crate::block::incompatible("canarios: decodificación inesperada")) };
            if xxh(format!("{}\0{}\0{}", r.title, r.url, r.text).as_bytes()).as_str() != c["text"].as_str().unwrap_or_default() {
                return Err(crate::block::incompatible(format!("canario {}: el texto descomprimido no coincide (¿binario o CPU distintos?)", c["id"])));
            }
        }
        Ok(n)
    }

    /// The query embedder: the block's own (IR next to the container, tokenizer inside it) or,
    /// for a folder of segments, the one in `models`.
    pub fn embedder(&self, models: &Path, ov_libs: &Path) -> Result<crate::embed::Embedder> {
        match &self.block {
            Some(b) => {
                let (xml, bin, tok) = b.embedder_files()?;
                crate::embed::Embedder::from_files(&xml, &bin, &tok, ov_libs)
            }
            None => crate::embed::Embedder::new(models, ov_libs),
        }
    }

    /// RAM of what is always loaded, in all segments (incl. the flag columns).
    pub fn fixed_bytes(&self) -> u64 {
        self.segs.iter().map(|sg| sg.fixed_bytes() as u64).sum::<u64>() + self.flags.iter().map(|f| f.capacity() as u64).sum::<u64>()
    }

    /// RAM of the LM models (measured when loading them).
    pub fn model_bytes(&self) -> u64 {
        self.model_charge.bytes()
    }

    /// Loadable parts: (segment, part, bytes in RAM, total bytes).
    pub fn parts(&self) -> Vec<(usize, Part, u64, u64)> {
        let mut v = Vec::new();
        for (si, sg) in self.segs.iter().enumerate() {
            for p in Part::ALL {
                if let Some(b) = sg.blob(p) {
                    v.push((si, p, b.resident_bytes() as u64, b.len() as u64));
                }
            }
        }
        v
    }

    /// Loads a segment part into RAM (charged to the budget). Returns the bytes loaded.
    pub fn promote(&mut self, si: usize, p: Part) -> Result<u64> {
        let n = match self.segs[si].blob_mut(p) {
            Some(b) => b.make_resident()? as u64,
            None => 0,
        };
        let total = self.index_charge.bytes() + n;
        self.index_charge.resize(total);
        Ok(n)
    }

    /// The LM batch never needs less than this (the longest record of any loaded model, plus
    /// the batch's fixed sampling scratch).
    pub fn lm_min(&self) -> u64 {
        self.lms.values().map(|c| c.min_pool()).max().unwrap_or(0)
    }

    /// Attaches the persistent disk cache (its RAM index is charged to the budget).
    pub fn set_l2(&mut self, l2: crate::cache::L2) {
        if self.lms.is_empty() {
            return; // nothing to cache: no LM-tier segments
        }
        l2.set_charge(self.budget.charge(Class::Cache, 0));
        self.l2 = Some(l2);
    }

    pub fn lm_pool(&self) -> u64 {
        self.lm_pool
    }

    /// Sets the memory cap of the LM batch (and its charge in the budget).
    pub fn set_lm_pool(&mut self, bytes: u64) {
        self.lm_pool = bytes;
        self.lm_charge.resize(if bytes == u64::MAX { 0 } else { bytes });
        for s in self.lm_svc.lock().unwrap_or_else(|e| e.into_inner()).values() {
            s.set_pool(bytes);
        }
    }

    /// The decoding service of LM model `name`, started on first use with the current pool.
    pub fn lm_service(&self, name: &str) -> Result<Arc<LmService>> {
        let mut m = self.lm_svc.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(s) = m.get(name) {
            return Ok(s.clone());
        }
        let codec = self.lms.get(name).with_context(|| format!("falta el modelo LM {name}"))?.clone();
        let s = Arc::new(LmService::new(codec, self.lm_pool)?);
        m.insert(name.to_string(), s.clone());
        Ok(s)
    }

    /// Sets the RAM of the caches: 1/8 for the index of the disk cache (if there is one), and of
    /// the rest, 1/8 for titles and 7/8 for records. Shrinking drops old entries.
    pub fn set_cache_bytes(&self, bytes: u64) {
        let l2 = match &self.l2 {
            Some(c) => {
                c.set_index_bytes(bytes / 8);
                bytes / 8
            }
            None => 0,
        };
        let ram = bytes - l2;
        let prefixes = if self.block.is_some() { ram / 4 } else { 0 };
        self.titles.lock().unwrap_or_else(|e| e.into_inner()).set_cap(ram / 8);
        self.prefixes.lock().unwrap_or_else(|e| e.into_inner()).set_cap(prefixes);
        self.cache.lock().unwrap_or_else(|e| e.into_inner()).set_cap(ram - ram / 8 - prefixes);
    }

    /// RAM given to the caches (as last set by `set_cache_bytes`).
    pub fn cache_bytes(&self) -> u64 {
        let l2 = self.l2.as_ref().map_or(0, |c| c.index_bytes_cap().min(u64::MAX / 4));
        // one lock at a time (`texts_each` takes records then titles: never hold both here)
        let records = self.cache.lock().unwrap_or_else(|e| e.into_inner()).cap();
        let titles = self.titles.lock().unwrap_or_else(|e| e.into_inner()).cap();
        let prefixes = self.prefixes.lock().unwrap_or_else(|e| e.into_inner()).cap();
        records + titles + prefixes + l2
    }

    /// Memory needed by the disk reads of the vector scan when some codes are not in RAM: one
    /// read window per worker thread (0 if all vectors are loaded).
    pub fn vec_read_bytes(&self) -> u64 {
        let nb = self.segs.iter().filter_map(|sg| sg.vec.as_ref().filter(|v| !v.blob().is_resident())).map(|v| v.nbytes).max().unwrap_or(0);
        (rayon::current_num_threads() * crate::vecs::DISK_READ_DOCS * nb) as u64
    }

    /// Memory breakdown of the engine (for `stats()["memory"]`).
    pub fn memory_report(&self) -> serde_json::Value {
        let mut parts = serde_json::Map::new();
        for p in Part::ALL {
            let (mut r, mut t, mut nr, mut n) = (0u64, 0u64, 0usize, 0usize);
            for sg in &self.segs {
                if let Some(b) = sg.blob(p) {
                    n += 1;
                    t += b.len() as u64;
                    if b.is_resident() {
                        nr += 1;
                        r += b.len() as u64;
                    }
                }
            }
            parts.insert(p.name().into(), serde_json::json!({"resident_bytes": r, "total_bytes": t, "resident_segments": nr, "segments": n}));
        }
        let c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        let t = self.titles.lock().unwrap_or_else(|e| e.into_inner());
        serde_json::json!({
            "parts": parts,
            "fixed_bytes": self.fixed_bytes(),
            "lm_models_bytes": self.model_bytes(),
            "lm_pool": if self.lm_pool == u64::MAX { None } else { Some(self.lm_pool) },
            "cache": {
                "records": {"bytes": c.bytes(), "cap": c.cap(), "entries": c.len(), "hits": c.hits, "misses": c.misses},
                "titles": {"bytes": t.bytes(), "cap": t.cap(), "entries": t.len(), "hits": t.hits, "misses": t.misses},
                "l2_entries": self.l2.as_ref().map(|l| l.entries()),
            },
        })
    }

    /// Modelo LM de un segmento del nivel frío.
    pub fn codec_for(&self, si: usize) -> Option<&LmCodec> {
        self.lms.get(self.segs[si].s.man.lm_model.as_ref()?).map(|c| c.as_ref())
    }

    /// ¿Cumple el documento los filtros de la consulta? (las filas deleted nunca se devuelven)
    fn passes(&self, q: &Query, si: usize, row: &MetaRow, author: Option<u32>) -> bool {
        !row.deleted
            && (q.include_dead || !row.dead)
            && (q.types.is_empty() || q.types.contains(&row.typ))
            && !q.from.is_some_and(|f| row.time < f)
            && !q.to.is_some_and(|t| row.time >= t)
            && match (&q.by, author) {
                (None, _) => true,
                (Some(_), Some(a)) => row.author == a,
                (Some(by), None) => self.segs[si].s.meta.authors.get(row.author as usize) == by.as_str(),
            }
    }

    /// Filtros de tipo/dead/deleted con la columna de banderas en RAM (sin descomprimir nada).
    #[inline]
    fn passes_flags(&self, q: &Query, si: usize, d: u32) -> bool {
        let f = self.flags[si][d as usize];
        let (typ, dead, deleted) = (f & 7, (f >> 3) & 1 == 1, (f >> 4) & 1 == 1);
        !deleted && (q.include_dead || !dead) && (q.types.is_empty() || q.types.contains(&typ))
    }

    /// Búsqueda top-k. Los filtros se aplican antes del recorte final:
    /// - sin texto: listado de lo más reciente a lo más antiguo que cumple los filtros (parada en k);
    /// - autor o tipos raros (job/poll/pollopt): recorrido exacto de los metadatos del rango;
    /// - resto: candidatos de vectores + BM25 por segmento, puntuados y filtrados en orden de
    ///   puntuación con la columna de banderas; si un segmento no llega a k válidos y su lista
    ///   estaba recortada, se amplía SOLO ese segmento (4x, hasta 16x).
    /// Las fechas se comprueban exactamente en los bloques extremos del rango; en el interior se
    /// usa el orden temporal de los ids.
    pub fn search(&self, q: &Query, qvec: Option<&[f32]>) -> Result<Vec<Hit>> {
        self.search_with(q, qvec, &|_| true)
    }

    /// Working memory a search declares up front: the first round of every segment it touches,
    /// as many at once as there are worker threads (widening a round asks for more on the way).
    pub fn search_cost(&self, q: &Query) -> u64 {
        if q.text.trim().is_empty() {
            return 0; // listing: k rows per segment
        }
        let k = q.k.clamp(1, MAX_K);
        let mut costs: Vec<u64> = self
            .segs
            .iter()
            .filter(|sg| {
                let m = &sg.s.man;
                m.n_docs > 0 && !q.from.is_some_and(|f| m.time_max < f) && !q.to.is_some_and(|t| m.time_min >= t)
            })
            .map(|sg| {
                let (kc, c1) = round_sizes(q, k, sg.s.man.n_docs, 1, self.c1_min);
                round_cost(kc, c1)
            })
            .collect();
        costs.sort_unstable_by(|a, b| b.cmp(a));
        costs.truncate(rayon::current_num_threads());
        costs.iter().sum()
    }

    /// Like `search`; before widening a segment's round (x4) it asks `grow` for the extra working
    /// memory, and without it returns what it has (possibly fewer than k for very selective filters).
    /// Damaged data it reads is a `CorruptBlockError`.
    pub fn search_with(&self, q: &Query, qvec: Option<&[f32]>, grow: &(dyn Fn(u64) -> bool + Sync)) -> Result<Vec<Hit>> {
        let k = q.k.clamp(1, MAX_K);
        let mut terms = tokenize::terms(&q.text);
        terms.sort();
        terms.dedup();
        let has_text = !q.text.trim().is_empty();
        let use_bm25 = q.mode != Mode::Vector && !terms.is_empty();
        let use_vec = q.mode != Mode::Keyword && qvec.is_some() && has_text;
        // IDF global (df sumado en todos los segmentos)
        let n = self.total_docs as f32;
        let mut dfs: Vec<u64> = vec![0; terms.len()];
        for (i, t) in terms.iter().enumerate() {
            for s in self.segs.iter().filter_map(|s| s.bm25.as_ref()) {
                dfs[i] += s.lookup(t)?.map_or(0, |x| x.1 as u64);
            }
        }
        let mut idf: Vec<f32> = dfs.iter().map(|&df| ((n - df as f32 + 0.5) / (df as f32 + 0.5) + 1.0).ln()).collect();
        // very long queries: only the MAX_TERMS rarest terms that appear somewhere count
        if terms.len() > MAX_TERMS {
            let mut ix: Vec<usize> = (0..terms.len()).filter(|&i| dfs[i] > 0).collect();
            ix.sort_by(|&a, &b| idf[b].total_cmp(&idf[a]).then(a.cmp(&b)));
            ix.truncate(MAX_TERMS);
            ix.sort_unstable();
            terms = ix.iter().map(|&i| terms[i].clone()).collect();
            idf = ix.iter().map(|&i| idf[i]).collect();
        }
        // sin texto: listado de lo más reciente; con texto pero sin señal utilizable (solo palabras
        // vacías en modo palabras, o sin vector de consulta): ningún resultado
        let listing = !has_text;
        if has_text && !use_bm25 && !use_vec {
            return Ok(Vec::new());
        }
        let only_rare_types = !q.types.is_empty() && q.types.iter().all(|t| matches!(t, 3..=5));
        let scan = q.by.is_some() || only_rare_types;
        let fuse = |v: f32, b: f32| match q.mode {
            Mode::Vector => v,
            Mode::Keyword => b,
            Mode::Hybrid => v + q.w_bm25 * b / (b + q.c_bm25),
        };
        let per_seg: Vec<Option<Vec<Hit>>> = self
            .segs
            .par_iter()
            .enumerate()
            .map(|(si, sg)| -> Result<Option<Vec<Hit>>> {
                let m = &sg.s.man;
                if m.n_docs == 0 || q.from.is_some_and(|f| m.time_max < f) || q.to.is_some_and(|t| m.time_min >= t) {
                    return Ok(None);
                }
                let lo = match q.from {
                    Some(t) => sg.s.meta.try_doc_at_time(t)?,
                    None => 0,
                } as u32;
                let hi = match q.to {
                    Some(t) => sg.s.meta.try_doc_at_time(t)?,
                    None => m.n_docs,
                } as u32;
                if lo >= hi {
                    return Ok(None);
                }
                let author = match &q.by {
                    Some(by) => match sg.s.meta.authors.position(by) {
                        Some(a) => Some(a as u32),
                        None => return Ok(None),
                    },
                    None => None,
                };
                let (b0, b1) = (lo as usize / META_BLOCK, (hi as usize - 1) / META_BLOCK);
                if listing {
                    // sin señal de texto: lo más reciente primero, con parada temprana
                    let mut out = Vec::with_capacity(k);
                    'blocks: for b in (b0..=b1).rev() {
                        let rows = sg.s.meta.try_block(b)?;
                        for kk in (0..rows.len()).rev() {
                            let d = b * META_BLOCK + kk;
                            if d >= lo as usize && d < hi as usize && self.passes(q, si, &rows[kk], author) {
                                out.push(Hit { seg: si, doc: d as u32, v: 0.0, b: 0.0, score: 0.0 });
                                if out.len() >= k {
                                    break 'blocks;
                                }
                            }
                        }
                    }
                    return Ok(Some(out));
                }
                let tps: Vec<(TermPost, f32)> = match (&sg.bm25, use_bm25) {
                    (Some(b), true) => terms.iter().zip(&idf).filter_map(|(t, &w)| b.lookup(t).transpose().map(|x| x.map(|x| (x.0, w)))).collect::<Result<_>>()?,
                    _ => Vec::new(),
                };
                let lut = match (&sg.vec, use_vec) {
                    (Some(v), true) => Some(QueryLut::new(qvec.unwrap(), v.nbytes)),
                    _ => None,
                };
                let score = |cand: &[u32]| -> Result<Vec<Hit>> {
                    let vs = match (&sg.vec, &lut) {
                        (Some(v), Some(l)) => v.score_docs(l, cand)?,
                        _ => vec![f32::NEG_INFINITY; cand.len()],
                    };
                    let bs = match (&sg.bm25, tps.is_empty()) {
                        (Some(b), false) => b.score_docs(&tps, cand)?,
                        _ => vec![0.0; cand.len()],
                    };
                    Ok(cand
                        .iter()
                        .enumerate()
                        // solo documentos con alguna señal de la consulta: término (palabras) o vector
                        .filter(|&(i, _)| match q.mode {
                            Mode::Keyword => bs[i] > 0.0,
                            Mode::Vector => vs[i].is_finite(),
                            Mode::Hybrid => vs[i].is_finite() || bs[i] > 0.0,
                        })
                        .map(|(i, &d)| {
                            let v = if vs[i].is_finite() { vs[i] } else { 0.0 };
                            Hit { seg: si, doc: d, v, b: bs[i], score: fuse(v, bs[i]) }
                        })
                        .collect())
                };
                if scan {
                    // recorrido exacto de los metadatos del rango (autor / tipos raros)
                    let mut cand: Vec<u32> = Vec::new();
                    for b in b0..=b1 {
                        for (kk, row) in sg.s.meta.try_block(b)?.iter().enumerate() {
                            let d = b * META_BLOCK + kk;
                            if d >= lo as usize && d < hi as usize && self.passes(q, si, row, author) {
                                cand.push(d as u32);
                            }
                        }
                    }
                    let mut hs = score(&cand)?;
                    hs.sort_by(|a, b| b.score.total_cmp(&a.score).then(b.doc.cmp(&a.doc)));
                    hs.truncate(k);
                    return Ok(Some(hs));
                }
                // filas de los bloques extremos (para la comprobación exacta de fechas)
                let edge: HashMap<usize, Vec<MetaRow>> = if q.from.is_some() || q.to.is_some() {
                    [b0, b1].iter().map(|&b| Ok((b, sg.s.meta.try_block(b)?))).collect::<Result<_>>()?
                } else {
                    HashMap::new()
                };
                let time_ok = |d: u32| -> bool {
                    match edge.get(&(d as usize / META_BLOCK)) {
                        Some(rows) => {
                            let t = rows[d as usize % META_BLOCK].time;
                            !q.from.is_some_and(|f| t < f) && !q.to.is_some_and(|x| t >= x)
                        }
                        None => true,
                    }
                };
                let mut mult = 1;
                loop {
                    let (kc, c1) = round_sizes(q, k, (hi - lo) as usize, mult, self.c1_min);
                    let mut raw: Vec<u32> = Vec::new();
                    let mut saturated = false;
                    if let (Some(v), Some(l)) = (&sg.vec, &lut) {
                        let r = v.search(l, lo as usize, hi as usize, c1, kc)?;
                        saturated |= r.len() >= kc && kc < (hi - lo) as usize;
                        raw.extend(r.into_iter().map(|x| x.0));
                    }
                    if let (Some(b), false) = (&sg.bm25, tps.is_empty()) {
                        let r = b.top_k(&tps, lo, hi, kc)?;
                        saturated |= r.len() >= kc && kc < (hi - lo) as usize;
                        raw.extend(r.into_iter().map(|x| x.0));
                    }
                    raw.sort_unstable();
                    raw.dedup();
                    let mut hs = score(&raw)?;
                    hs.sort_by(|a, b| b.score.total_cmp(&a.score).then(b.doc.cmp(&a.doc)));
                    // filtrar en orden de puntuación hasta tener k válidos
                    let mut out = Vec::with_capacity(k);
                    for h in hs {
                        if self.passes_flags(q, si, h.doc) && time_ok(h.doc) {
                            out.push(h);
                            if out.len() >= k {
                                break;
                            }
                        }
                    }
                    if out.len() >= k || !saturated || mult >= 16 {
                        return Ok(Some(out));
                    }
                    // widening needs more working memory: ask first (never waits)
                    let (kc4, c14) = round_sizes(q, k, (hi - lo) as usize, mult * 4, self.c1_min);
                    if !grow(round_cost(kc4, c14).saturating_sub(round_cost(kc, c1))) {
                        return Ok(Some(out));
                    }
                    mult *= 4;
                }
            })
            .collect::<Result<_>>()?;
        let mut hits: Vec<Hit> = per_seg.into_iter().flatten().flatten().collect();
        // a igual puntuación (p.ej. listado sin texto), lo más reciente primero
        hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(b.seg.cmp(&a.seg)).then(b.doc.cmp(&a.doc)));
        hits.truncate(k);
        Ok(hits)
    }

    /// The stored prompt prefix (flags, prefix) of a block document whose record has no LM tokens.
    pub fn fallback_prefix(&self, doc: u32) -> Option<(u8, &str)> {
        self.prefix_fb.get(&doc).map(|(f, s)| (*f, s.as_str()))
    }

    /// The query's words for BM25 over a block (P4): the global idf from the df of every term
    /// (frequent ones included), words the index does not know dropped, at most `MAX_TERMS` (the
    /// rarest), sorted.
    pub fn query_terms(&self, text: &str) -> Result<Vec<QTerm>> {
        let Some(bm) = self.segs.first().and_then(|s| s.bm25.as_ref()) else { return Ok(Vec::new()) };
        let mut words = tokenize::terms(text);
        words.sort();
        words.dedup();
        let n = self.total_docs as f32;
        let mut out = Vec::new();
        for w in words {
            if let Some((post, df)) = bm.lookup(&w)? {
                let idf = ((n - df as f32 + 0.5) / (df as f32 + 0.5) + 1.0).ln();
                out.push(QTerm { text: w, idf, post });
            }
        }
        if out.len() > MAX_TERMS {
            out.sort_by(|a, b| b.idf.total_cmp(&a.idf).then(a.text.cmp(&b.text)));
            out.truncate(MAX_TERMS);
            out.sort_by(|a, b| a.text.cmp(&b.text));
        }
        Ok(out)
    }

    /// P4 stage 1 on a block: the top `kv` documents by the vector score and then the top `kb`
    /// by the rare-term score (`rare`: postings and idf), among those that pass the query's
    /// filters, without repeats. A round is widened (x4, up to x16) while filters leave too few.
    pub fn candidates(&self, q: &Query, qvec: Option<&[f32]>, rare: &[(TermPost, f32)], kv: usize, kb: usize, grow: &(dyn Fn(u64) -> bool + Sync)) -> Result<Vec<Cand>> {
        let sg = &self.segs[0];
        let n_docs = sg.s.man.n_docs;
        let lo = match q.from {
            Some(t) => sg.s.meta.try_doc_at_time(t)?,
            None => 0,
        } as u32;
        let hi = match q.to {
            Some(t) => sg.s.meta.try_doc_at_time(t)?,
            None => n_docs,
        } as u32;
        if lo >= hi {
            return Ok(Vec::new());
        }
        let author = match &q.by {
            Some(by) => match sg.s.meta.authors.position(by) {
                Some(a) => Some(a as u32),
                None => return Ok(Vec::new()),
            },
            None => None,
        };
        let lut = match (&sg.vec, qvec) {
            (Some(v), Some(x)) => Some(QueryLut::new(x, v.nbytes)),
            _ => None,
        };
        let s1_of = |docs: &[u32]| -> Result<Vec<f32>> {
            match (&sg.vec, &lut) {
                (Some(v), Some(l)) => v.score_docs(l, docs),
                _ => Ok(vec![f32::NEG_INFINITY; docs.len()]),
            }
        };
        let only_rare_types = !q.types.is_empty() && q.types.iter().all(|t| matches!(t, 3..=5));
        let (mut vecs, mut rares): (Vec<u32>, Vec<u32>) = (Vec::new(), Vec::new());
        if q.by.is_some() || only_rare_types {
            // an exact scan of the range's metadata (author / rare types)
            let (b0, b1) = (lo as usize / META_BLOCK, (hi as usize - 1) / META_BLOCK);
            let mut docs = Vec::new();
            for b in b0..=b1 {
                for (kk, row) in sg.s.meta.try_block(b)?.iter().enumerate() {
                    let d = b * META_BLOCK + kk;
                    if d >= lo as usize && d < hi as usize && self.passes(q, 0, row, author) {
                        docs.push(d as u32);
                    }
                }
            }
            let s1 = s1_of(&docs)?;
            let br = match (&sg.bm25, rare.is_empty()) {
                (Some(b), false) => b.score_docs(rare, &docs)?,
                _ => vec![0.0; docs.len()],
            };
            let mut by_v: Vec<usize> = (0..docs.len()).filter(|&i| s1[i].is_finite()).collect();
            by_v.sort_by(|&a, &b| s1[b].total_cmp(&s1[a]).then(docs[b].cmp(&docs[a])));
            vecs = by_v.iter().take(kv).map(|&i| docs[i]).collect();
            let mut by_b: Vec<usize> = (0..docs.len()).filter(|&i| br[i] > 0.0).collect();
            by_b.sort_by(|&a, &b| br[b].total_cmp(&br[a]).then(docs[b].cmp(&docs[a])));
            rares = by_b.iter().take(kb).map(|&i| docs[i]).collect();
        } else {
            let n_rng = (hi - lo) as usize;
            if let (Some(v), Some(l), true) = (&sg.vec, &lut, kv > 0) {
                let mut mult = 1;
                loop {
                    let (kc, c1) = round_sizes(q, kv, n_rng, mult, self.c1_min);
                    let r = v.search(l, lo as usize, hi as usize, c1, kc)?;
                    let saturated = r.len() >= kc && kc < n_rng;
                    vecs = r.into_iter().map(|x| x.0).filter(|&d| self.passes_flags(q, 0, d)).take(kv).collect();
                    if vecs.len() >= kv || !saturated || mult >= 16 {
                        break;
                    }
                    let (kc4, c14) = round_sizes(q, kv, n_rng, mult * 4, self.c1_min);
                    if !grow(round_cost(kc4, c14).saturating_sub(round_cost(kc, c1))) {
                        break;
                    }
                    mult *= 4;
                }
            }
            if let (Some(b), false) = (&sg.bm25, rare.is_empty() || kb == 0) {
                let mut mult = 1;
                loop {
                    let kc = ((kb * 20).max(200) * mult).min(n_rng);
                    let r = b.top_k(rare, lo, hi, kc)?;
                    let saturated = r.len() >= kc && kc < n_rng;
                    rares = r.into_iter().map(|x| x.0).filter(|&d| self.passes_flags(q, 0, d)).take(kb).collect();
                    if rares.len() >= kb || !saturated || mult >= 16 {
                        break;
                    }
                    mult *= 4;
                }
            }
        }
        // the vector list first, then the rare-term list
        let mut seen = std::collections::HashSet::new();
        let docs: Vec<u32> = vecs.into_iter().chain(rares).filter(|d| seen.insert(*d)).collect();
        let s1 = s1_of(&docs)?;
        Ok(docs.iter().zip(s1).map(|(&doc, s)| Cand { doc, s1: if s.is_finite() { s } else { 0.0 }, has_vec: s.is_finite() }).collect())
    }

    /// Busca un id de HN en todo el archivo: (segmento, doc, metadatos).
    pub fn find_meta(&self, id: u32) -> Result<Option<(usize, usize, MetaRow)>> {
        let Some(si) = self.segs.partition_point(|s| s.s.man.id_min <= id).checked_sub(1) else { return Ok(None) };
        let sg = &self.segs[si];
        if id > sg.s.man.id_max {
            return Ok(None);
        }
        let Some(doc) = sg.s.meta.try_doc_of_id(id)? else { return Ok(None) };
        let row = sg.s.meta.try_row(doc)?;
        // a lookup that lands on another id (a wrong late-table entry) finds nothing
        Ok((row.id == id).then_some((si, doc, row)))
    }

    /// Textos de varios documentos: zstd directo, LM en un único lote (con cachés L1/L2).
    /// `limit`: tokens máximos para el nivel LM (fragmento); `title_only`: solo títulos.
    pub fn texts(&self, docs: &[(usize, usize)], limit: Option<usize>, title_only: bool) -> Result<Vec<(TextRec, bool)>> {
        let w: Vec<(usize, usize, Option<usize>, bool)> = docs.iter().map(|&(s, d)| (s, d, limit, title_only)).collect();
        self.texts_each(&w)
    }

    /// Igual que `texts` pero con (segmento, doc, límite, solo_título) por documento: todo lo que
    /// haya que decodificar con el LM va en UN solo lote.
    pub fn texts_each(&self, wants: &[(usize, usize, Option<usize>, bool)]) -> Result<Vec<(TextRec, bool)>> {
        let mut out: Vec<Option<(TextRec, bool)>> = vec![None; wants.len()];
        // 1) RAM caches (under the locks, no I/O)
        {
            let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            let mut titles = self.titles.lock().unwrap_or_else(|e| e.into_inner());
            for (i, &(si, doc, _, title_only)) in wants.iter().enumerate() {
                if let TextSource::Lm(_) = &self.segs[si].s.text {
                    let key = (si, doc as u32);
                    if let Some(r) = cache.get(&key) {
                        out[i] = Some((r, false));
                    } else if title_only {
                        if let Some(t) = titles.get(&key) {
                            out[i] = Some((TextRec { title: t, ..Default::default() }, true));
                        }
                    }
                }
            }
        }
        // 2) zstd tier and the L2 disk cache (disk reads, no locks held)
        let mut pending: Vec<usize> = Vec::new();
        for (i, &(si, doc, _, _)) in wants.iter().enumerate() {
            if out[i].is_some() {
                continue;
            }
            match &self.segs[si].s.text {
                TextSource::Zstd(t) => out[i] = Some((t.get(doc)?, false)),
                TextSource::Lm(_) => match self.l2.as_ref().and_then(|l2| l2.get(self.seg_keys[si], doc as u32)) {
                    Some(r) => out[i] = Some((r, false)),
                    None => pending.push(i),
                },
            }
        }
        // agrupar lo pendiente por modelo LM (normalmente uno solo)
        let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
        for &i in &pending {
            let f = self.segs[wants[i].0].s.man.lm_model.clone().unwrap_or_default();
            match groups.iter_mut().find(|g| g.0 == f) {
                Some(g) => g.1.push(i),
                None => groups.push((f, vec![i])),
            }
        }
        for (f, pending) in groups {
            // someone waits for these texts: interactive jobs of the shared batch
            let jobs: Vec<Job> = pending
                .iter()
                .map(|&i| {
                    let TextSource::Lm(l) = &self.segs[wants[i].0].s.text else { unreachable!() };
                    let want = if wants[i].3 { Want::Title } else { wants[i].2.map_or(Want::Full, Want::Snippet) };
                    Ok(Job { rec: l.raw(wants[i].1)?.into_owned(), has_check: l.rec_check(), want })
                })
                .collect::<Result<_>>()?;
            let dec: Vec<(TextRec, bool)> = self
                .lm_service(&f)?
                .run(jobs, Prio::Interactive)?
                .into_iter()
                .map(|d| match d {
                    Decoded::Text(t, p) => Ok((t, p)),
                    Decoded::Prefix(_) => anyhow::bail!("decodificación inesperada"),
                })
                .collect::<Result<_>>()?;
            {
                let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
                let mut titles = self.titles.lock().unwrap_or_else(|e| e.into_inner());
                for (&i, (rec, partial)) in pending.iter().zip(dec) {
                    let key = (wants[i].0, wants[i].1 as u32);
                    titles.put(key, rec.title.clone());
                    if !partial {
                        cache.put(key, rec.clone());
                    }
                    out[i] = Some((rec, partial));
                }
            }
            // the disk cache is written without holding the RAM-cache locks
            if let Some(l2) = &self.l2 {
                for &i in &pending {
                    if let Some((rec, false)) = &out[i] {
                        l2.put(self.seg_keys[wants[i].0], wants[i].1 as u32, rec);
                    }
                }
            }
        }
        Ok(out.into_iter().map(|x| x.unwrap()).collect())
    }

    pub fn get(&self, id: u32) -> Result<Option<Item>> {
        match self.find_meta(id)? {
            Some((si, doc, m)) => {
                let (t, _) = self.texts(&[(si, doc)], None, false)?.remove(0);
                Ok(Some(crate::segment::join(&m, t, &self.segs[si].s.meta.authors)))
            }
            None => Ok(None),
        }
    }

    /// id de la story raíz del hilo de un documento (él mismo si no tiene padre).
    /// `None` si la cadena de padres sale del archivo (padre en un mes no cargado).
    pub fn root_id(&self, seg: usize, doc: u32) -> Result<Option<u32>> {
        let row = self.segs[seg].s.meta.try_row(doc as usize)?;
        let (mut id, mut parent) = (row.id, row.parent);
        for _ in 0..1024 {
            if parent == 0 {
                return Ok(Some(id));
            }
            let Some((_, _, r)) = self.find_meta(parent)? else { return Ok(None) };
            id = r.id;
            parent = r.parent;
        }
        Ok(None)
    }

    /// Resultados completos. `full=false`: en el nivel LM solo se decodifica un fragmento.
    pub fn fetch(&self, hits: &[Hit], full: bool) -> Result<Vec<ResultItem>> {
        let rows: Vec<MetaRow> = hits.iter().map(|h| self.segs[h.seg].s.meta.try_row(h.doc as usize)).collect::<Result<_>>()?;
        // contexto del hilo: story raíz de cada comentario (títulos en un solo lote)
        let roots: Vec<Option<u32>> = hits
            .iter()
            .zip(&rows)
            .map(|(h, r)| if r.parent == 0 { Ok(None) } else { self.root_id(h.seg, h.doc) })
            .collect::<Result<_>>()?;
        let mut root_docs: Vec<(usize, usize)> = Vec::new();
        let mut root_pos: HashMap<u32, usize> = HashMap::new();
        for r in roots.iter().flatten() {
            if root_pos.contains_key(r) {
                continue;
            }
            if let Some((si, doc, _)) = self.find_meta(*r)? {
                root_pos.insert(*r, root_docs.len());
                root_docs.push((si, doc));
            }
        }
        // resultados (fragmento o completo) + títulos de las stories: un solo lote para el LM
        let limit = if full { None } else { Some(SNIPPET_TOKENS) };
        let mut wants: Vec<(usize, usize, Option<usize>, bool)> = hits.iter().map(|h| (h.seg, h.doc as usize, limit, false)).collect();
        wants.extend(root_docs.iter().map(|&(si, d)| (si, d, None, true)));
        let mut got = self.texts_each(&wants)?;
        let root_titles: Vec<(TextRec, bool)> = got.split_off(hits.len());
        let texts = got;
        let mut out = Vec::with_capacity(hits.len());
        for (i, ((h, m), (t, partial))) in hits.iter().zip(&rows).zip(texts).enumerate() {
            let story_id = roots[i];
            let story_title = story_id.and_then(|r| root_pos.get(&r)).map(|&p| root_titles[p].0.title.clone());
            out.push(ResultItem {
                rank: i + 1,
                id: m.id,
                kind: kind_name(m.typ),
                by: self.segs[h.seg].s.meta.authors.get(m.author as usize).to_string(),
                date: dates::fmt(m.time),
                time: m.time,
                points: m.score,
                text: if partial { t.text.clone() } else { htmlx::to_plain(&t.text) },
                html: if partial { String::new() } else { t.text.clone() },
                partial,
                tier: if self.segs[h.seg].s.is_lm() { "lm" } else { "zstd" },
                parent_id: if m.parent == 0 { None } else { Some(m.parent) },
                story_id,
                story_title,
                score: h.score,
                vscore: h.v,
                bm25: h.b,
                link: format!("https://news.ycombinator.com/item?id={}", m.id),
                title: t.title,
                url: t.url,
            });
        }
        Ok(out)
    }
}
