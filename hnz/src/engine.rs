//! Motor de consulta: reparte la búsqueda entre segmentos (en paralelo), aplica filtros de fecha
//! como rangos de docIDs, fusiona vectores + BM25 y arma los resultados con el contexto del hilo.
//! Los resultados de segmentos del nivel LM se descomprimen TODOS en un mismo lote.

use crate::bm25::{Bm25, TermPost};
use crate::item::Item;
use crate::lm::store::LmCodec;
use crate::meta::{MetaRow, META_BLOCK};
use crate::segment::{Segment, TextSource};
use crate::textstore::TextRec;
use crate::vecs::{QueryLut, VecIndex};
use crate::{dates, htmlx, tokenize};
use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Tokens que se decodifican por resultado del nivel LM en modo fragmento.
pub const SNIPPET_TOKENS: usize = 64;
/// máximo de resultados por consulta (protege la memoria del servidor)
pub const MAX_K: usize = 500;
const CACHE_MAX: usize = 20_000;

pub struct Seg {
    pub s: Segment,
    pub bm25: Option<Bm25>,
    pub vec: Option<VecIndex>,
}

pub struct Engine {
    pub segs: Vec<Seg>,
    pub total_docs: usize,
    /// modelos LM de los segmentos fríos (por nombre de fichero)
    pub lms: HashMap<String, LmCodec>,
    /// caché L1 (RAM) de registros del nivel LM ya descomprimidos completos
    cache: Mutex<HashMap<(usize, u32), TextRec>>,
    /// caché L2 (disco) de registros del nivel LM ya descomprimidos
    pub l2: Option<crate::cache::L2>,
    /// títulos de stories del nivel LM ya decodificados (contexto del hilo)
    titles: Mutex<HashMap<(usize, u32), String>>,
    seg_keys: Vec<u64>,
    /// por segmento, 1 byte por doc: tipo | dead<<3 | deleted<<4 (filtros sin descomprimir nada)
    flags: Vec<Vec<u8>>,
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
    /// Abre todos los segmentos de `root` (ignora construcciones a medias y restos de `retier`).
    /// Carga cada modelo LM distinto que usen los segmentos del nivel frío.
    pub fn open(root: &Path, models: &Path) -> Result<Self> {
        let dirs: Vec<PathBuf> = std::fs::read_dir(root)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.join("manifest.json").exists() && !crate::segment::is_transient_dir(p))
            .collect();
        let mut segs = Vec::new();
        for d in dirs {
            let s = Segment::open(&d)?;
            let bm25 = if d.join("bm25.fst").exists() { Some(Bm25::open(&d)?) } else { None };
            let vec = if s.man.vec_bits > 0 && d.join("vec.bin").exists() { Some(VecIndex::open(&d, s.man.vec_bits)?) } else { None };
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
        let mut lms: HashMap<String, LmCodec> = HashMap::new();
        for sg in &segs {
            if let (Some(f), TextSource::Lm(l)) = (&sg.s.man.lm_model, &sg.s.text) {
                if !lms.contains_key(f) {
                    let c = LmCodec::load(&models.join(f)).with_context(|| format!("modelo LM {f}"))?;
                    lms.insert(f.clone(), c);
                }
                l.check_model(&lms[f]).with_context(|| format!("segmento {}", sg.s.man.name))?;
            }
        }
        let total_docs = segs.iter().map(|x| x.s.man.n_docs).sum();
        let l2 = if !lms.is_empty() {
            match crate::cache::L2::open(&root.join("_cache_l2.bin"), 512 << 20) {
                Ok(c) => Some(c),
                Err(e) => {
                    eprintln!("aviso: caché L2 desactivada ({e})");
                    None
                }
            }
        } else {
            None
        };
        // clave de la L2: nombre + identificador de construcción + nº de docs (si el segmento se
        // reconstruye, sus entradas antiguas dejan de coincidir)
        let seg_keys = segs
            .iter()
            .map(|s| crate::cache::seg_key(&s.s.man.name) ^ s.s.man.build_id.rotate_left(13) ^ (s.s.man.n_docs as u64).rotate_left(41))
            .collect();
        // columna de banderas en RAM (1 B/doc), construida en paralelo una vez al abrir
        let flags: Vec<Vec<u8>> = segs.iter().map(|sg| crate::meta::load_flags(&sg.s.dir, &sg.s.meta)).collect::<Result<_>>()?;
        Ok(Self { segs, total_docs, lms, cache: Mutex::new(HashMap::new()), l2, titles: Mutex::new(HashMap::new()), seg_keys, flags })
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
                (Some(by), None) => &self.segs[si].s.meta.authors[row.author as usize] == by,
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
        let k = q.k.clamp(1, MAX_K);
        let mut terms = tokenize::terms(&q.text);
        terms.sort();
        terms.dedup();
        let has_text = !q.text.trim().is_empty();
        let use_bm25 = q.mode != Mode::Vector && !terms.is_empty();
        let use_vec = q.mode != Mode::Keyword && qvec.is_some() && has_text;
        // IDF global (df sumado en todos los segmentos)
        let n = self.total_docs as f32;
        let idf: Vec<f32> = terms
            .iter()
            .map(|t| {
                let df: u64 = self.segs.iter().filter_map(|s| s.bm25.as_ref()?.lookup(t).map(|x| x.1 as u64)).sum();
                ((n - df as f32 + 0.5) / (df as f32 + 0.5) + 1.0).ln()
            })
            .collect();
        let listing = !use_bm25 && !use_vec;
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
                    Some(by) => Some(sg.s.meta.authors.iter().position(|a| a == by)? as u32),
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
                            let v = if vs[i].is_finite() { vs[i] } else { -1.0 };
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
                    let mut kc = (k * 20).max(200) * mult;
                    if !q.types.is_empty() {
                        kc *= 5; // stories ~14 % de los ítems
                    }
                    let kc = kc.min((hi - lo) as usize);
                    let mut raw: Vec<u32> = Vec::new();
                    let mut saturated = false;
                    if let (Some(v), Some(l)) = (&sg.vec, &lut) {
                        let c1 = q.vec_cand.unwrap_or((kc * 10).max(2000)).min((hi - lo) as usize);
                        let r = v.search(l, lo as usize, hi as usize, c1, kc);
                        saturated |= r.len() >= kc;
                        raw.extend(r.into_iter().map(|x| x.0));
                    }
                    if let (Some(b), false) = (&sg.bm25, tps.is_empty()) {
                        let r = b.top_k(&tps, lo, hi, kc);
                        saturated |= r.len() >= kc;
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
        let mut pending: Vec<usize> = Vec::new();
        {
            let cache = self.cache.lock().unwrap();
            let titles = self.titles.lock().unwrap();
            for (i, &(si, doc, _, title_only)) in wants.iter().enumerate() {
                match &self.segs[si].s.text {
                    TextSource::Zstd(t) => out[i] = Some((t.get(doc)?, false)),
                    TextSource::Lm(_) => {
                        let key = (si, doc as u32);
                        if let Some(r) = cache.get(&key) {
                            out[i] = Some((r.clone(), false));
                        } else if let Some(r) = self.l2.as_ref().and_then(|l2| l2.get(self.seg_keys[si], doc as u32)) {
                            out[i] = Some((r, false));
                        } else if let (true, Some(t)) = (title_only, titles.get(&key)) {
                            out[i] = Some((TextRec { title: t.clone(), ..Default::default() }, true));
                        } else {
                            pending.push(i);
                        }
                    }
                }
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
            let raws: Vec<&[u8]> = pending
                .iter()
                .map(|&i| match &self.segs[wants[i].0].s.text {
                    TextSource::Lm(l) => l.raw(wants[i].1),
                    _ => unreachable!(),
                })
                .collect();
            let limits: Vec<Option<usize>> = pending.iter().map(|&i| wants[i].2).collect();
            let tonly: Vec<bool> = pending.iter().map(|&i| wants[i].3).collect();
            let checks: Vec<bool> = pending
                .iter()
                .map(|&i| match &self.segs[wants[i].0].s.text {
                    TextSource::Lm(l) => l.rec_check(),
                    _ => false,
                })
                .collect();
            let dec = codec.decode_records_each(&raws, true, &limits, &tonly, &checks)?;
            let mut cache = self.cache.lock().unwrap();
            let mut titles = self.titles.lock().unwrap();
            for (&i, (rec, partial)) in pending.iter().zip(dec) {
                let key = (wants[i].0, wants[i].1 as u32);
                if titles.len() >= CACHE_MAX {
                    titles.clear();
                }
                titles.insert(key, rec.title.clone());
                if !partial {
                    if cache.len() >= CACHE_MAX {
                        cache.clear();
                    }
                    cache.insert(key, rec.clone());
                    if let Some(l2) = &self.l2 {
                        l2.put(self.seg_keys[wants[i].0], wants[i].1 as u32, &rec);
                    }
                }
                out[i] = Some((rec, partial));
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
                by: self.segs[h.seg].s.meta.authors[m.author as usize].clone(),
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
