//! Motor de consulta: reparte la búsqueda entre segmentos (en paralelo), aplica filtros de fecha
//! como rangos de docIDs, fusiona vectores + BM25 y arma los resultados con el contexto del hilo.

use crate::bm25::{Bm25, TermPost};
use crate::item::Item;
use crate::meta::MetaRow;
use crate::segment::Segment;
use crate::vecs::{QueryLut, VecIndex};
use crate::{dates, htmlx, tokenize};
use anyhow::Result;
use rayon::prelude::*;
use serde::Serialize;
use std::path::{Path, PathBuf};

pub struct Seg {
    pub s: Segment,
    pub bm25: Option<Bm25>,
    pub vec: Option<VecIndex>,
}

pub struct Engine {
    pub segs: Vec<Seg>,
    pub total_docs: usize,
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
            c_bm25: 10.0,
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
    /// HTML original de HN (literal)
    pub html: String,
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
    pub fn open(root: &Path) -> Result<Self> {
        let dirs: Vec<PathBuf> = std::fs::read_dir(root)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.join("manifest.json").exists())
            .collect();
        let mut segs = Vec::new();
        for d in dirs {
            let s = Segment::open(&d)?;
            let bm25 = if d.join("bm25.fst").exists() { Some(Bm25::open(&d)?) } else { None };
            let vec = if s.man.vec_bits > 0 && d.join("vec.bin").exists() {
                Some(VecIndex::open(&d, s.man.vec_bits)?)
            } else {
                None
            };
            segs.push(Seg { s, bm25, vec });
        }
        segs.sort_by_key(|x| x.s.man.id_min);
        let total_docs = segs.iter().map(|x| x.s.man.n_docs).sum();
        Ok(Self { segs, total_docs })
    }

    pub fn search(&self, q: &Query, qvec: Option<&[f32]>) -> Vec<Hit> {
        let mut terms = tokenize::terms(&q.text);
        terms.sort();
        terms.dedup();
        let use_bm25 = q.mode != Mode::Vector && !terms.is_empty();
        let use_vec = q.mode != Mode::Keyword && qvec.is_some();
        // IDF global (df sumado en todos los segmentos)
        let n = self.total_docs as f32;
        let idf: Vec<f32> = terms
            .iter()
            .map(|t| {
                let df: u64 = self.segs.iter().filter_map(|s| s.bm25.as_ref()?.lookup(t).map(|x| x.1 as u64)).sum();
                ((n - df as f32 + 0.5) / (df as f32 + 0.5) + 1.0).ln()
            })
            .collect();
        let kc = (q.k * 20).max(200);
        let per_seg: Vec<Vec<Hit>> = self
            .segs
            .par_iter()
            .enumerate()
            .filter_map(|(si, sg)| {
                let m = &sg.s.man;
                if q.from.is_some_and(|f| m.time_max < f) || q.to.is_some_and(|t| m.time_min >= t) {
                    return None;
                }
                let lo = q.from.map(|t| sg.s.meta.doc_at_time(t)).unwrap_or(0) as u32;
                let hi = q.to.map(|t| sg.s.meta.doc_at_time(t)).unwrap_or(m.n_docs) as u32;
                if lo >= hi {
                    return None;
                }
                let tps: Vec<(TermPost, f32)> = match (&sg.bm25, use_bm25) {
                    (Some(b), true) => terms.iter().zip(&idf).filter_map(|(t, &w)| b.lookup(t).map(|x| (x.0, w))).collect(),
                    _ => Vec::new(),
                };
                let lut = match (&sg.vec, use_vec) {
                    (Some(v), true) => Some(QueryLut::new(qvec.unwrap(), v.nbytes)),
                    _ => None,
                };
                let mut cand: Vec<u32> = Vec::new();
                if let (Some(v), Some(l)) = (&sg.vec, &lut) {
                    cand.extend(v.search(l, lo as usize, hi as usize, (kc * 10).max(2000), kc).into_iter().map(|x| x.0));
                }
                if let (Some(b), false) = (&sg.bm25, tps.is_empty()) {
                    cand.extend(b.top_k(&tps, lo, hi, kc).into_iter().map(|x| x.0));
                }
                cand.sort_unstable();
                cand.dedup();
                let vs = match (&sg.vec, &lut) {
                    (Some(v), Some(l)) => v.score_docs(l, &cand),
                    _ => vec![0.0; cand.len()],
                };
                let bs = match (&sg.bm25, tps.is_empty()) {
                    (Some(b), false) => b.score_docs(&tps, &cand),
                    _ => vec![0.0; cand.len()],
                };
                Some(
                    cand.iter()
                        .enumerate()
                        .map(|(i, &d)| Hit { seg: si, doc: d, v: if vs[i].is_finite() { vs[i] } else { -1.0 }, b: bs[i], score: 0.0 })
                        .collect(),
                )
            })
            .collect();
        let mut hits: Vec<Hit> = per_seg.into_iter().flatten().collect();
        for h in hits.iter_mut() {
            h.score = match q.mode {
                Mode::Vector => h.v,
                Mode::Keyword => h.b,
                Mode::Hybrid => h.v + q.w_bm25 * h.b / (h.b + q.c_bm25),
            };
        }
        hits.sort_by(|a, b| b.score.total_cmp(&a.score));
        let mut out = Vec::with_capacity(q.k);
        for h in hits {
            if out.len() >= q.k {
                break;
            }
            let sg = &self.segs[h.seg];
            let row = sg.s.meta.row(h.doc as usize);
            if (!q.include_dead && row.dead)
                || (!q.types.is_empty() && !q.types.contains(&row.typ))
                || q.from.is_some_and(|f| row.time < f)
                || q.to.is_some_and(|t| row.time >= t)
                || q.by.as_ref().is_some_and(|by| &sg.s.meta.authors[row.author as usize] != by)
            {
                continue;
            }
            out.push(h);
        }
        out
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

    pub fn get(&self, id: u32) -> Result<Option<Item>> {
        match self.find_meta(id) {
            Some((si, doc, _)) => Ok(Some(self.segs[si].s.item(doc)?)),
            None => Ok(None),
        }
    }

    /// Story raíz de un comentario (recorriendo `parent`).
    pub fn thread_root(&self, it: &Item) -> (Option<u32>, Option<String>) {
        if it.parent == 0 {
            return (None, None);
        }
        let mut cur = it.parent;
        for _ in 0..256 {
            match self.find_meta(cur) {
                Some((si, doc, row)) => {
                    if row.parent == 0 {
                        let title = self.segs[si].s.text.get(doc).map(|t| t.title).unwrap_or_default();
                        return (Some(cur), Some(title));
                    }
                    cur = row.parent;
                }
                None => return (Some(cur), None),
            }
        }
        (Some(cur), None)
    }

    pub fn fetch(&self, hits: &[Hit]) -> Result<Vec<ResultItem>> {
        hits.par_iter()
            .enumerate()
            .map(|(i, h)| {
                let it = self.segs[h.seg].s.item(h.doc as usize)?;
                let (sid, stitle) = self.thread_root(&it);
                Ok(ResultItem {
                    rank: i + 1,
                    id: it.id,
                    kind: kind_name(it.typ),
                    date: dates::fmt(it.time),
                    time: it.time,
                    points: it.score,
                    text: htmlx::to_plain(&it.text),
                    link: format!("https://news.ycombinator.com/item?id={}", it.id),
                    parent_id: if it.parent == 0 { None } else { Some(it.parent) },
                    story_id: sid,
                    story_title: stitle,
                    score: h.score,
                    vscore: h.v,
                    bm25: h.b,
                    by: it.by,
                    title: it.title,
                    url: it.url,
                    html: it.text,
                })
            })
            .collect()
    }
}
