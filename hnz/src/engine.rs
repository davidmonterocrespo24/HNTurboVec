//! Motor de consulta: reparte la búsqueda entre segmentos (en paralelo), aplica filtros de fecha
//! como rangos de docIDs, fusiona vectores + BM25 y arma los resultados con el contexto del hilo.
//! Los resultados de segmentos del nivel LM se descomprimen TODOS en un mismo lote.

use crate::blob::{Blob, Residency};
use crate::bm25::{Bm25, TermPost};
use crate::cache::GenCache;
use crate::item::Item;
use crate::lm::store::LmCodec;
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
fn round_sizes(q: &Query, k: usize, n: usize, mult: usize) -> (usize, usize) {
    let mut kc = (k * 20).max(200) * mult;
    if !q.types.is_empty() {
        kc *= 5; // stories ~14 % de los ítems
    }
    let kc = kc.min(n);
    let c1 = q.vec_cand.unwrap_or((kc * 10).max(2000)).max(kc).min(n);
    (kc, c1)
}

/// Working memory of one round in one segment: stage-1 Hamming pairs (with growth slack) and the
/// scored list, plus BM25 heap, merged ids and scored hits of up to 2 x kc candidates.
fn round_cost(kc: usize, c1: usize) -> u64 {
    c1 as u64 * 24 + kc as u64 * 64
}

/// Un solo lote del LM a la vez en todo el proceso: su pool de hilos es global y dos lotes a la vez
/// solo competirían por los mismos núcleos. Lo demás (metadatos, nivel zstd, cachés) va en paralelo.
static LM_GATE: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Bytes counted per cached record: strings + hash-table entry with growth slack.
fn rec_weight(r: &TextRec) -> u64 {
    256 + (r.title.len() + r.url.len() + r.text.len()) as u64
}

fn title_weight(t: &String) -> u64 {
    160 + t.len() as u64
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
    pub lms: HashMap<String, LmCodec>,
    /// RAM cache of fully decompressed LM-tier records
    cache: Mutex<GenCache<(usize, u32), TextRec>>,
    /// caché L2 (disco) de registros del nivel LM ya descomprimidos
    pub l2: Option<crate::cache::L2>,
    /// already decoded titles of LM-tier stories (thread context)
    titles: Mutex<GenCache<(usize, u32), String>>,
    seg_keys: Vec<u64>,
    /// por segmento, 1 byte por doc: tipo | dead<<3 | deleted<<4 (filtros sin descomprimir nada)
    flags: Vec<Vec<u8>>,
    pub budget: Arc<Budget>,
    /// memory cap of one LM batch (charged to the budget as working memory)
    lm_pool: u64,
    lm_charge: Charge,
    index_charge: Charge,
    model_charge: Charge,
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

impl Engine {
    /// Opens all segments in `root` (ignoring half-built ones and `retier` leftovers) and loads every
    /// distinct LM model used by cold-tier segments. CLI defaults (`EngineOptions::cli`).
    pub fn open(root: &Path, models: &Path) -> Result<Self> {
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
        let mut lms: HashMap<String, LmCodec> = HashMap::new();
        let mut model_bytes = 0u64;
        for sg in &segs {
            if let (Some(f), TextSource::Lm(l)) = (&sg.s.man.lm_model, &sg.s.text) {
                if !lms.contains_key(f) {
                    let path = models.join(f);
                    let (c, steady, _) = crate::mem::measure(|| LmCodec::load(&path));
                    let c = c.with_context(|| format!("modelo LM {f}"))?;
                    model_bytes += steady.max(std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0));
                    lms.insert(f.clone(), c);
                }
                l.check_model(&lms[f]).with_context(|| format!("segmento {}", sg.s.man.name))?;
            }
        }
        let total_docs = segs.iter().map(|x| x.s.man.n_docs).sum();
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
        // clave de la L2: nombre + identificador de construcción + nº de docs (si el segmento se
        // reconstruye, sus entradas antiguas dejan de coincidir)
        let seg_keys = segs
            .iter()
            .map(|s| crate::cache::seg_key(&s.s.man.name) ^ s.s.man.build_id.rotate_left(13) ^ (s.s.man.n_docs as u64).rotate_left(41))
            .collect();
        // columna de banderas en RAM (1 B/doc), construida en paralelo una vez al abrir
        let flags: Vec<Vec<u8>> = segs.iter().map(|sg| crate::meta::load_flags(&sg.s.dir, &sg.s.meta)).collect::<Result<_>>()?;
        let b = &opts.budget;
        let index = segs.iter().map(|sg| sg.ram_bytes() as u64).sum::<u64>() + flags.iter().map(|f| f.capacity() as u64).sum::<u64>();
        let (titles_cap, records_cap) = (opts.cache_bytes / 8, opts.cache_bytes - opts.cache_bytes / 8);
        Ok(Self {
            segs,
            total_docs,
            lms,
            cache: Mutex::new(GenCache::new(records_cap, rec_weight, Some(b.charge(Class::Cache, 0)))),
            l2,
            titles: Mutex::new(GenCache::new(titles_cap, title_weight, Some(b.charge(Class::Cache, 0)))),
            seg_keys,
            flags,
            lm_pool: opts.lm_pool,
            lm_charge: b.charge(Class::Working, if opts.lm_pool == u64::MAX { 0 } else { opts.lm_pool }),
            index_charge: b.charge(Class::Index, index),
            model_charge: b.charge(Class::Model, model_bytes),
            budget: opts.budget,
        })
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
        self.titles.lock().unwrap_or_else(|e| e.into_inner()).set_cap(ram / 8);
        self.cache.lock().unwrap_or_else(|e| e.into_inner()).set_cap(ram - ram / 8);
    }

    /// RAM given to the caches (as last set by `set_cache_bytes`).
    pub fn cache_bytes(&self) -> u64 {
        let l2 = self.l2.as_ref().map_or(0, |c| c.index_bytes_cap().min(u64::MAX / 4));
        // one lock at a time (`texts_each` takes records then titles: never hold both here)
        let records = self.cache.lock().unwrap_or_else(|e| e.into_inner()).cap();
        let titles = self.titles.lock().unwrap_or_else(|e| e.into_inner()).cap();
        records + titles + l2
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
        self.lms.get(self.segs[si].s.man.lm_model.as_ref()?)
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
    pub fn search(&self, q: &Query, qvec: Option<&[f32]>) -> Vec<Hit> {
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
                let (kc, c1) = round_sizes(q, k, sg.s.man.n_docs, 1);
                round_cost(kc, c1)
            })
            .collect();
        costs.sort_unstable_by(|a, b| b.cmp(a));
        costs.truncate(rayon::current_num_threads());
        costs.iter().sum()
    }

    /// Like `search`; before widening a segment's round (x4) it asks `grow` for the extra working
    /// memory, and without it returns what it has (possibly fewer than k for very selective filters).
    pub fn search_with(&self, q: &Query, qvec: Option<&[f32]>, grow: &(dyn Fn(u64) -> bool + Sync)) -> Vec<Hit> {
        let k = q.k.clamp(1, MAX_K);
        let mut terms = tokenize::terms(&q.text);
        terms.sort();
        terms.dedup();
        let has_text = !q.text.trim().is_empty();
        let use_bm25 = q.mode != Mode::Vector && !terms.is_empty();
        let use_vec = q.mode != Mode::Keyword && qvec.is_some() && has_text;
        // IDF global (df sumado en todos los segmentos)
        let n = self.total_docs as f32;
        let dfs: Vec<u64> = terms.iter().map(|t| self.segs.iter().filter_map(|s| s.bm25.as_ref()?.lookup(t).map(|x| x.1 as u64)).sum()).collect();
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
            return Vec::new();
        }
        let only_rare_types = !q.types.is_empty() && q.types.iter().all(|t| matches!(t, 3..=5));
        let scan = q.by.is_some() || only_rare_types;
        let fuse = |v: f32, b: f32| match q.mode {
            Mode::Vector => v,
            Mode::Keyword => b,
            Mode::Hybrid => v + q.w_bm25 * b / (b + q.c_bm25),
        };
        let per_seg: Vec<Vec<Hit>> = self
            .segs
            .par_iter()
            .enumerate()
            .filter_map(|(si, sg)| {
                let m = &sg.s.man;
                if m.n_docs == 0 || q.from.is_some_and(|f| m.time_max < f) || q.to.is_some_and(|t| m.time_min >= t) {
                    return None;
                }
                let lo = q.from.map(|t| sg.s.meta.doc_at_time(t)).unwrap_or(0) as u32;
                let hi = q.to.map(|t| sg.s.meta.doc_at_time(t)).unwrap_or(m.n_docs) as u32;
                if lo >= hi {
                    return None;
                }
                let author = match &q.by {
                    Some(by) => Some(sg.s.meta.authors.position(by)? as u32),
                    None => None,
                };
                let (b0, b1) = (lo as usize / META_BLOCK, (hi as usize - 1) / META_BLOCK);
                if listing {
                    // sin señal de texto: lo más reciente primero, con parada temprana
                    let mut out = Vec::with_capacity(k);
                    'blocks: for b in (b0..=b1).rev() {
                        let rows = sg.s.meta.block(b);
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
                    return Some(out);
                }
                let tps: Vec<(TermPost, f32)> = match (&sg.bm25, use_bm25) {
                    (Some(b), true) => terms.iter().zip(&idf).filter_map(|(t, &w)| b.lookup(t).map(|x| (x.0, w))).collect(),
                    _ => Vec::new(),
                };
                let lut = match (&sg.vec, use_vec) {
                    (Some(v), true) => Some(QueryLut::new(qvec.unwrap(), v.nbytes)),
                    _ => None,
                };
                let score = |cand: &[u32]| -> Vec<Hit> {
                    let vs = match (&sg.vec, &lut) {
                        (Some(v), Some(l)) => v.score_docs(l, cand),
                        _ => vec![f32::NEG_INFINITY; cand.len()],
                    };
                    let bs = match (&sg.bm25, tps.is_empty()) {
                        (Some(b), false) => b.score_docs(&tps, cand),
                        _ => vec![0.0; cand.len()],
                    };
                    cand.iter()
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
                        .collect()
                };
                if scan {
                    // recorrido exacto de los metadatos del rango (autor / tipos raros)
                    let mut cand: Vec<u32> = Vec::new();
                    for b in b0..=b1 {
                        for (kk, row) in sg.s.meta.block(b).iter().enumerate() {
                            let d = b * META_BLOCK + kk;
                            if d >= lo as usize && d < hi as usize && self.passes(q, si, row, author) {
                                cand.push(d as u32);
                            }
                        }
                    }
                    let mut hs = score(&cand);
                    hs.sort_by(|a, b| b.score.total_cmp(&a.score).then(b.doc.cmp(&a.doc)));
                    hs.truncate(k);
                    return Some(hs);
                }
                // filas de los bloques extremos (para la comprobación exacta de fechas)
                let edge: HashMap<usize, Vec<MetaRow>> = if q.from.is_some() || q.to.is_some() {
                    [b0, b1].iter().map(|&b| (b, sg.s.meta.block(b))).collect()
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
                    let (kc, c1) = round_sizes(q, k, (hi - lo) as usize, mult);
                    let mut raw: Vec<u32> = Vec::new();
                    let mut saturated = false;
                    if let (Some(v), Some(l)) = (&sg.vec, &lut) {
                        let r = v.search(l, lo as usize, hi as usize, c1, kc);
                        saturated |= r.len() >= kc && kc < (hi - lo) as usize;
                        raw.extend(r.into_iter().map(|x| x.0));
                    }
                    if let (Some(b), false) = (&sg.bm25, tps.is_empty()) {
                        let r = b.top_k(&tps, lo, hi, kc);
                        saturated |= r.len() >= kc && kc < (hi - lo) as usize;
                        raw.extend(r.into_iter().map(|x| x.0));
                    }
                    raw.sort_unstable();
                    raw.dedup();
                    let mut hs = score(&raw);
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
                        return Some(out);
                    }
                    // widening needs more working memory: ask first (never waits)
                    let (kc4, c14) = round_sizes(q, k, (hi - lo) as usize, mult * 4);
                    if !grow(round_cost(kc4, c14).saturating_sub(round_cost(kc, c1))) {
                        return Some(out);
                    }
                    mult *= 4;
                }
            })
            .collect();
        let mut hits: Vec<Hit> = per_seg.into_iter().flatten().collect();
        // a igual puntuación (p.ej. listado sin texto), lo más reciente primero
        hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(b.seg.cmp(&a.seg)).then(b.doc.cmp(&a.doc)));
        hits.truncate(k);
        hits
    }

    /// Busca un id de HN en todo el archivo: (segmento, doc, metadatos).
    pub fn find_meta(&self, id: u32) -> Option<(usize, usize, MetaRow)> {
        let si = self.segs.partition_point(|s| s.s.man.id_min <= id).checked_sub(1)?;
        let sg = &self.segs[si];
        if id > sg.s.man.id_max {
            return None;
        }
        let doc = sg.s.meta.doc_of_id(id)?;
        Some((si, doc, sg.s.meta.row(doc)))
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
            let codec = self.lms.get(&f).with_context(|| format!("falta el modelo LM {f}"))?;
            let owned: Vec<std::borrow::Cow<[u8]>> = pending
                .iter()
                .map(|&i| match &self.segs[wants[i].0].s.text {
                    TextSource::Lm(l) => l.raw(wants[i].1),
                    _ => unreachable!(),
                })
                .collect::<Result<_>>()?;
            let raws: Vec<&[u8]> = owned.iter().map(|c| c.as_ref()).collect();
            let limits: Vec<Option<usize>> = pending.iter().map(|&i| wants[i].2).collect();
            let tonly: Vec<bool> = pending.iter().map(|&i| wants[i].3).collect();
            let checks: Vec<bool> = pending
                .iter()
                .map(|&i| match &self.segs[wants[i].0].s.text {
                    TextSource::Lm(l) => l.rec_check(),
                    _ => false,
                })
                .collect();
            // one batch at a time in the process, within the LM memory cap
            let dec = {
                let _g = LM_GATE.lock().unwrap_or_else(|e| e.into_inner());
                codec.decode_records_pool(&raws, true, &limits, &tonly, &checks, self.lm_pool)?
            };
            drop(raws);
            drop(owned);
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
        match self.find_meta(id) {
            Some((si, doc, m)) => {
                let (t, _) = self.texts(&[(si, doc)], None, false)?.remove(0);
                Ok(Some(crate::segment::join(&m, t, &self.segs[si].s.meta.authors)))
            }
            None => Ok(None),
        }
    }

    /// id de la story raíz del hilo de un documento (él mismo si no tiene padre).
    /// `None` si la cadena de padres sale del archivo (padre en un mes no cargado).
    pub fn root_id(&self, seg: usize, doc: u32) -> Option<u32> {
        let row = self.segs[seg].s.meta.row(doc as usize);
        let (mut id, mut parent) = (row.id, row.parent);
        for _ in 0..1024 {
            if parent == 0 {
                return Some(id);
            }
            let (_, _, r) = self.find_meta(parent)?;
            id = r.id;
            parent = r.parent;
        }
        None
    }

    /// Resultados completos. `full=false`: en el nivel LM solo se decodifica un fragmento.
    pub fn fetch(&self, hits: &[Hit], full: bool) -> Result<Vec<ResultItem>> {
        let rows: Vec<MetaRow> = hits.iter().map(|h| self.segs[h.seg].s.meta.row(h.doc as usize)).collect();
        // contexto del hilo: story raíz de cada comentario (títulos en un solo lote)
        let roots: Vec<Option<u32>> = hits
            .iter()
            .zip(&rows)
            .map(|(h, r)| if r.parent == 0 { None } else { self.root_id(h.seg, h.doc) })
            .collect();
        let mut root_docs: Vec<(usize, usize)> = Vec::new();
        let mut root_pos: HashMap<u32, usize> = HashMap::new();
        for r in roots.iter().flatten() {
            if root_pos.contains_key(r) {
                continue;
            }
            if let Some((si, doc, _)) = self.find_meta(*r) {
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
