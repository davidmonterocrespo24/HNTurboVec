//! P4: reranking a compact block's candidates by their prefix (E30c, E30f).
//!
//! A cold query: stage 1 gives the vector candidates and the rare-term candidates (already
//! filtered, `Engine::candidates`); the shared LM service decodes each candidate's prefix (title,
//! url and the first `prompt::PREFIX_TOKENS` body tokens, `crate::prompt`); b = BM25 of every query
//! word (frequent ones included) over the prefix text, with the global idf and the document's
//! full-length norm; then by mode:
//! - hybrid: the cheap score R0 = s1 + r0_w·b/(b + r0_c) orders the candidates, the top
//!   `embed_top` are re-embedded and scored cos + w·b/(b + c), the rest follow in R0 order;
//! - vector: every candidate re-embedded, scored by cos;
//! - keyword with a frequent word: only the documents with b > 0, by b (frequent words can only
//!   be found among these candidates, in their prefix); all-rare keyword queries keep stage 1.
use crate::bm25::TermPost;
use crate::engine::{Engine, Hit, Mode, Query, MAX_K, SNIPPET_TOKENS};
use crate::lm::jobs::{Decoded, Want};
use crate::lm::service::{Job, Prio};
use crate::prompt::{body_text, doc_prompt, prefix_string, PREFIX_TOKENS};
use crate::segment::TextSource;
use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

/// Reranking settings (defaults: E30c, E30f and the user's decisions of 2026-10-15).
#[derive(Clone, Debug)]
pub struct RerankCfg {
    /// vector candidates (stage 1)
    pub kv: usize,
    /// rare-term candidates (stage 1)
    pub kb: usize,
    /// rare-term candidates of a keyword query with a frequent word
    pub kb_keyword: usize,
    /// hybrid: re-embed only the top N by R0 (None = all)
    pub embed_top: Option<usize>,
    pub r0_w: f32,
    pub r0_c: f32,
    pub w: f32,
    pub c: f32,
}

impl Default for RerankCfg {
    fn default() -> Self {
        Self { kv: 100, kb: 20, kb_keyword: 50, embed_top: Some(50), r0_w: 0.3, r0_c: 20.0, w: 0.15, c: 40.0 }
    }
}

/// Embeds document prompts for the reranker: 768-dim, L2-normalized vectors.
pub trait DocEmbed: Sync {
    /// `docs`: (HN id, prompt) of each document.
    fn embed(&self, docs: &[(u32, String)]) -> Result<Vec<Vec<f32>>>;
}

/// Stored document vectors used in place of re-embedding (evaluation without OpenVINO): the
/// `<base>.f16.npy` + `<base>.ids.npy` pairs of `scripts/embed_prompts.py`.
pub struct OracleEmbed {
    files: Vec<(memmap2::Mmap, usize, usize)>,
    by_id: HashMap<u32, (usize, usize)>,
}

impl OracleEmbed {
    pub fn open(bases: &[std::path::PathBuf]) -> Result<Self> {
        let mut files = Vec::new();
        let mut by_id = HashMap::new();
        for (f, base) in bases.iter().enumerate() {
            let (shape, descr, map, off) = crate::vecs::read_npy(Path::new(&format!("{}.f16.npy", base.display())))?;
            let (_, _, imap, ioff) = crate::vecs::read_npy(Path::new(&format!("{}.ids.npy", base.display())))?;
            if descr != "<f2" || shape.len() != 2 {
                bail!("{}: float16 N x dim expected", base.display());
            }
            for (r, c) in imap[ioff..].chunks_exact(4).enumerate().take(shape[0]) {
                by_id.insert(u32::from_le_bytes(c.try_into().unwrap()), (f, r));
            }
            files.push((map, off, shape[1]));
        }
        Ok(Self { files, by_id })
    }
}

impl DocEmbed for OracleEmbed {
    fn embed(&self, docs: &[(u32, String)]) -> Result<Vec<Vec<f32>>> {
        docs.iter()
            .map(|(id, _)| {
                let &(f, r) = self.by_id.get(id).with_context(|| format!("no stored vector for id {id}"))?;
                let (map, off, dim) = &self.files[f];
                let b = &map[off + r * dim * 2..off + (r + 1) * dim * 2];
                let v: Vec<f32> = b.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect();
                let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                Ok(v.into_iter().map(|x| x / n).collect())
            })
            .collect()
    }
}

/// The decoded prefix of a candidate (results reuse it: snippet, or the whole record).
#[derive(Clone, Debug)]
pub struct Prefix {
    pub bytes: Vec<u8>,
    pub flags: u8,
    pub complete: bool,
    pub snippet_end: usize,
}

/// What a reranked search did.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Report {
    pub candidates: usize,
    pub decoded: usize,
    pub tokens: u64,
    pub embedded: usize,
    pub ms_stage1: f64,
    pub ms_decode: f64,
    pub ms_embed: f64,
}

pub struct Outcome {
    pub hits: Vec<Hit>,
    pub prefixes: HashMap<u32, std::sync::Arc<Prefix>>,
    pub report: Report,
    pub warning: Option<String>,
}

/// One candidate's features.
struct Feat {
    doc: u32,
    s1: f32,
    b: f32,
    r0: f32,
    prompt: Option<String>,
}

fn ms(t: Instant) -> f64 {
    (t.elapsed().as_secs_f64() * 1e4).round() / 10.0
}

/// A reranked search of a block (see the module doc). `qvec`: the query's embedding.
pub fn search(eng: &Engine, q: &Query, qvec: &[f32], cfg: &RerankCfg, emb: &dyn DocEmbed) -> Result<Outcome> {
    let t0 = Instant::now();
    let k = q.k.clamp(1, MAX_K);
    let terms = eng.query_terms(&q.text)?;
    let frequent = terms.iter().any(|t| t.post == TermPost::Frequent);
    if q.text.trim().is_empty() || (q.mode == Mode::Keyword && !frequent) {
        // a listing, or a keyword query whose words all have postings: stage 1 is exact
        let hits = eng.search(q, if q.mode == Mode::Keyword { None } else { Some(qvec) })?;
        return Ok(Outcome { hits, prefixes: HashMap::new(), report: Report::default(), warning: None });
    }
    let rare: Vec<(TermPost, f32)> = terms.iter().filter(|t| t.post != TermPost::Frequent).map(|t| (t.post, t.idf)).collect();
    let kb = match q.mode {
        Mode::Vector => 0,
        Mode::Keyword => cfg.kb_keyword,
        Mode::Hybrid => cfg.kb,
    };
    let cands = eng.candidates(q, Some(qvec), &rare, cfg.kv, kb, &|_| true)?;
    let mut report = Report { candidates: cands.len(), ms_stage1: ms(t0), ..Default::default() };

    // prefixes: the stored ones of LM-less records, the rest decoded by the shared LM service
    let t1 = Instant::now();
    let sg = &eng.segs[0];
    let TextSource::Lm(lm) = &sg.s.text else { bail!("el reordenado necesita un bloque compacto") };
    let mut prefixes: HashMap<u32, std::sync::Arc<Prefix>> = HashMap::new();
    let mut texts: HashMap<u32, String> = HashMap::new();
    let (mut jobs, mut job_doc) = (Vec::new(), Vec::new());
    {
        let mut cache = eng.prefixes.lock().unwrap_or_else(|e| e.into_inner());
        for c in &cands {
            if let Some(p) = cache.get(&c.doc) {
                prefixes.insert(c.doc, p);
            }
        }
    }
    for c in &cands {
        if prefixes.contains_key(&c.doc) {
            continue;
        }
        if let Some((flags, p)) = eng.fallback_prefix(c.doc) {
            prefixes.insert(c.doc, std::sync::Arc::new(Prefix { bytes: p.as_bytes().to_vec(), flags, complete: false, snippet_end: p.len() }));
            continue;
        }
        jobs.push(Job { rec: lm.raw(c.doc as usize)?.into_owned(), has_check: lm.rec_check(), want: Want::Prefix { body: PREFIX_TOKENS, mark: SNIPPET_TOKENS } });
        job_doc.push(c.doc);
    }
    report.decoded = jobs.len();
    let dec = if jobs.is_empty() { Vec::new() } else { eng.lm_service("block")?.run(jobs, Prio::Bulk)? };
    for (&d, r) in job_doc.iter().zip(dec) {
        match r {
            Decoded::Prefix(p) => {
                report.tokens += p.tokens as u64;
                let p = std::sync::Arc::new(Prefix { bytes: p.bytes, flags: p.flags, complete: p.complete, snippet_end: p.snippet_end });
                eng.prefixes.lock().unwrap_or_else(|e| e.into_inner()).put(d, p.clone());
                prefixes.insert(d, p);
            }
            // an empty record, or a zstd record without a vector: BM25 over its whole text
            Decoded::Text(t, _) => {
                texts.insert(d, crate::segment::index_text(&t.title, &t.url, &t.text));
            }
        }
    }
    report.ms_decode = ms(t1);

    // b over each prefix (every query word, global idf, the document's full-length norm), R0
    let bm = sg.bm25.as_ref().context("el bloque no tiene índice de palabras")?;
    let in_rare: Vec<Vec<bool>> = if q.mode == Mode::Keyword {
        // keyword mode: a rare word counts once in a document of its postings, even past the prefix
        let mut sorted: Vec<u32> = cands.iter().map(|c| c.doc).collect();
        sorted.sort_unstable();
        let pos: HashMap<u32, usize> = sorted.iter().enumerate().map(|(i, &d)| (d, i)).collect();
        terms
            .iter()
            .map(|t| {
                if t.post == TermPost::Frequent {
                    return Ok(vec![false; cands.len()]);
                }
                let s = bm.score_docs(&[(t.post, t.idf)], &sorted)?;
                Ok(cands.iter().map(|c| s[pos[&c.doc]] > 0.0).collect())
            })
            .collect::<Result<_>>()?
    } else {
        Vec::new()
    };
    let feats: Vec<Feat> = cands
        .iter()
        .enumerate()
        .map(|(ci, c)| {
            let (text, prompt) = match prefixes.get(&c.doc) {
                Some(p) => {
                    let s = prefix_string(&p.bytes);
                    let mut parts = s.splitn(3, '\n');
                    let (title, url, body) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""), parts.next().unwrap_or(""));
                    let host = url.split("://").nth(1).unwrap_or("").split('/').next().unwrap_or("");
                    (format!("{title} {host} {}", body_text(body, p.flags as u64)), c.has_vec.then(|| doc_prompt(&s, p.flags as u64)))
                }
                None => (texts.get(&c.doc).cloned().unwrap_or_default(), None),
            };
            let mut tf: HashMap<&str, u32> = HashMap::new();
            let words = crate::tokenize::terms(&text);
            for w in &words {
                if let Some(t) = terms.iter().find(|t| t.text == *w) {
                    *tf.entry(t.text.as_str()).or_default() += 1;
                }
            }
            let mut b = 0f32;
            for (ti, t) in terms.iter().enumerate() {
                let mut n = tf.get(t.text.as_str()).copied().unwrap_or(0);
                if q.mode == Mode::Keyword && n == 0 && in_rare[ti][ci] {
                    n = 1;
                }
                if n > 0 {
                    b += bm.score_tf(t.idf, n, c.doc);
                }
            }
            Feat { doc: c.doc, s1: c.s1, b, r0: c.s1 + cfg.r0_w * b / (b + cfg.r0_c), prompt }
        })
        .collect();

    // the final order
    let t2 = Instant::now();
    let desc = |a: f32, b: f32, da: u32, db: u32| b.total_cmp(&a).then(db.cmp(&da));
    let mut scored: Vec<(u32, f32, f32, f32)> = Vec::new(); // (doc, score, v, b)
    let mut warning = None;
    match q.mode {
        Mode::Keyword => {
            let mut f: Vec<&Feat> = feats.iter().filter(|f| f.b > 0.0).collect();
            f.sort_by(|a, b| b.b.total_cmp(&a.b).then(b.s1.total_cmp(&a.s1)).then(b.doc.cmp(&a.doc)));
            scored = f.iter().map(|f| (f.doc, f.b, f.s1, f.b)).collect();
            warning = Some("las palabras frecuentes solo se buscan entre los documentos más cercanos a la consulta, en su comienzo".to_string());
        }
        Mode::Vector | Mode::Hybrid => {
            let mut order: Vec<usize> = (0..feats.len()).collect();
            order.sort_by(|&a, &b| desc(feats[a].r0, feats[b].r0, feats[a].doc, feats[b].doc));
            let n_e = if q.mode == Mode::Vector { order.len() } else { cfg.embed_top.unwrap_or(order.len()).min(order.len()) };
            let (top, rest) = order.split_at(n_e);
            let ids: Vec<(usize, (u32, String))> = top
                .iter()
                .filter_map(|&i| Some((i, (sg.s.meta.try_row(feats[i].doc as usize).ok()?.id, feats[i].prompt.clone()?))))
                .collect();
            let vecs = emb.embed(&ids.iter().map(|x| x.1.clone()).collect::<Vec<_>>())?;
            report.embedded = ids.len();
            let mut cos: HashMap<usize, f32> = HashMap::new();
            for ((i, _), v) in ids.iter().zip(&vecs) {
                cos.insert(*i, v.iter().zip(qvec).map(|(a, b)| a * b).sum());
            }
            let w = if q.mode == Mode::Vector { 0.0 } else { cfg.w };
            let mut head: Vec<(u32, f32, f32, f32)> = top
                .iter()
                .map(|&i| {
                    let f = &feats[i];
                    let c = cos.get(&i).copied().unwrap_or(0.0);
                    (f.doc, c + w * f.b / (f.b + cfg.c), c, f.b)
                })
                .collect();
            head.sort_by(|a, b| desc(a.1, b.1, a.0, b.0));
            scored.extend(head);
            scored.extend(rest.iter().map(|&i| (feats[i].doc, feats[i].r0, feats[i].s1, feats[i].b)));
        }
    }
    report.ms_embed = ms(t2);
    let mut hits: Vec<Hit> = scored.into_iter().map(|(doc, score, v, b)| Hit { seg: 0, doc, score, v, b }).collect();
    // beyond the candidates: stage 1's order
    if hits.len() < k && q.mode != Mode::Keyword {
        let have: std::collections::HashSet<u32> = hits.iter().map(|h| h.doc).collect();
        let more = eng.search(q, Some(qvec))?;
        hits.extend(more.into_iter().filter(|h| !have.contains(&h.doc)));
    }
    hits.truncate(k);
    Ok(Outcome { hits, prefixes, report, warning })
}
