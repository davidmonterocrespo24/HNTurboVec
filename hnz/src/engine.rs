//! Motor de consulta: reparte la búsqueda entre segmentos (en paralelo), aplica filtros de fecha
//! como rangos de docIDs, fusiona vectores + BM25 y arma los resultados con el contexto del hilo.
//! Los resultados de segmentos del nivel LM se descomprimen TODOS en un mismo lote.

use crate::bm25::{Bm25, TermPost};
use crate::item::Item;
use crate::lm::store::LmCodec;
use crate::meta::MetaRow;
use crate::segment::{Segment, TextSource};
use crate::textstore::TextRec;
use crate::vecs::{QueryLut, VecIndex};
use crate::{dates, htmlx, tokenize};
use anyhow::{Context, Result};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Tokens que se decodifican por resultado del nivel LM en modo fragmento.
pub const SNIPPET_TOKENS: usize = 64;
const CACHE_MAX: usize = 20_000;

pub struct Seg {
    pub s: Segment,
    pub bm25: Option<Bm25>,
    pub vec: Option<VecIndex>,
}

pub struct Engine {
    pub segs: Vec<Seg>,
    pub total_docs: usize,
    pub lm: Option<LmCodec>,
    /// caché L1 (RAM) de registros del nivel LM ya descomprimidos completos
    cache: Mutex<HashMap<(usize, u32), TextRec>>,
    /// caché L2 (disco) de registros del nivel LM ya descomprimidos
    pub l2: Option<crate::cache::L2>,
    /// títulos de stories del nivel LM ya decodificados (contexto del hilo)
    titles: Mutex<HashMap<(usize, u32), String>>,
    seg_keys: Vec<u64>,
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
    /// Abre todos los segmentos de `root`. Si alguno es del nivel LM carga el modelo de `models`.
    pub fn open(root: &Path, models: &Path) -> Result<Self> {
        let dirs: Vec<PathBuf> = std::fs::read_dir(root)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.join("manifest.json").exists())
            .collect();
        let mut segs = Vec::new();
        let mut lm_file: Option<String> = None;
        for d in dirs {
            let s = Segment::open(&d)?;
            if let Some(f) = &s.man.lm_model {
                lm_file.get_or_insert_with(|| f.clone());
            }
            let bm25 = if d.join("bm25.fst").exists() { Some(Bm25::open(&d)?) } else { None };
            let vec = if s.man.vec_bits > 0 && d.join("vec.bin").exists() { Some(VecIndex::open(&d, s.man.vec_bits)?) } else { None };
            segs.push(Seg { s, bm25, vec });
        }
        segs.sort_by_key(|x| x.s.man.id_min);
        let lm = match lm_file {
            Some(f) => {
                let c = LmCodec::load(&models.join(&f)).with_context(|| format!("modelo LM {f}"))?;
                for sg in &segs {
                    if let TextSource::Lm(l) = &sg.s.text {
                        l.check_model(&c)?;
                    }
                }
                Some(c)
            }
            None => None,
        };
        let total_docs = segs.iter().map(|x| x.s.man.n_docs).sum();
        let l2 = if lm.is_some() {
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
        let seg_keys = segs.iter().map(|s| crate::cache::seg_key(&s.s.man.name)).collect();
        Ok(Self { segs, total_docs, lm, cache: Mutex::new(HashMap::new()), l2, titles: Mutex::new(HashMap::new()), seg_keys })
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
                    let c1 = q.vec_cand.unwrap_or((kc * 10).max(2000));
                    cand.extend(v.search(l, lo as usize, hi as usize, c1, kc).into_iter().map(|x| x.0));
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
        if !pending.is_empty() {
            let codec = self.lm.as_ref().context("falta el modelo LM")?;
            let raws: Vec<&[u8]> = pending
                .iter()
                .map(|&i| match &self.segs[wants[i].0].s.text {
                    TextSource::Lm(l) => l.raw(wants[i].1),
                    _ => unreachable!(),
                })
                .collect();
            let limits: Vec<Option<usize>> = pending.iter().map(|&i| wants[i].2).collect();
            let tonly: Vec<bool> = pending.iter().map(|&i| wants[i].3).collect();
            let dec = codec.decode_records_each(&raws, true, &limits, &tonly)?;
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

    /// id de la raíz del hilo de un documento (él mismo si no tiene padre).
    pub fn root_id(&self, seg: usize, doc: u32) -> u32 {
        let row = self.segs[seg].s.meta.row(doc as usize);
        let (mut id, mut parent) = (row.id, row.parent);
        for _ in 0..256 {
            if parent == 0 {
                return id;
            }
            match self.find_meta(parent) {
                Some((_, _, r)) => {
                    id = r.id;
                    parent = r.parent;
                }
                None => return parent,
            }
        }
        id
    }

    /// Resultados completos. `full=false`: en el nivel LM solo se decodifica un fragmento.
    pub fn fetch(&self, hits: &[Hit], full: bool) -> Result<Vec<ResultItem>> {
        let rows: Vec<MetaRow> = hits.iter().map(|h| self.segs[h.seg].s.meta.row(h.doc as usize)).collect();
        // contexto del hilo: story raíz de cada comentario (títulos en un solo lote)
        let roots: Vec<Option<u32>> = hits
            .iter()
            .zip(&rows)
            .map(|(h, r)| if r.parent == 0 { None } else { Some(self.root_id(h.seg, h.doc)) })
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
