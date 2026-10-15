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
    /// How many documents to hand over at a time: the embedder's parallelism (smaller hand-overs
    /// keep the reranker's priorities fresh).
    fn batch(&self) -> usize {
        8
    }
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

/// An embedder owned by one caller (the CLI): document prompts embedded one by one.
impl DocEmbed for std::sync::Mutex<crate::embed::Embedder> {
    fn embed(&self, docs: &[(u32, String)]) -> Result<Vec<Vec<f32>>> {
        let mut e = self.lock().unwrap_or_else(|p| p.into_inner());
        e.embed_docs(&docs.iter().map(|d| d.1.clone()).collect::<Vec<_>>())
    }

    fn batch(&self) -> usize {
        self.lock().unwrap_or_else(|p| p.into_inner()).doc_parallelism()
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
    /// until the last prefix is decoded (re-embedding overlaps it)
    pub ms_decode: f64,
    /// waiting for the re-embedding after the last prefix
    pub ms_embed: f64,
    /// the deadline passed: "decode" (the answer is stage 1's) or "embed" (the cheap score's)
    pub degraded: Option<&'static str>,
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

/// Candidates waiting to be re-embedded: the decoding loop adds them, the embedding worker takes
/// the best R0 first and drops those that can no longer be among the `n_e` best.
struct EmbedQueue {
    /// (candidate, R0, HN id, prompt)
    pending: Vec<(usize, f32, u32, String)>,
    /// the `n_e`-th best lower bound of R0: a pending candidate below it is out for good
    theta: f32,
    /// nothing more will be added
    done: bool,
    /// (candidate, vector) embedded so far
    got: Vec<(usize, Vec<f32>)>,
    /// the worker has ended (its error, if any, comes back when its thread is joined)
    ended: bool,
}

type SharedQueue = (std::sync::Mutex<EmbedQueue>, std::sync::Condvar);

fn new_queue() -> SharedQueue {
    let q = EmbedQueue { pending: Vec::new(), theta: f32::NEG_INFINITY, done: false, got: Vec::new(), ended: false };
    (std::sync::Mutex::new(q), std::sync::Condvar::new())
}

fn lock_queue(q: &SharedQueue) -> std::sync::MutexGuard<'_, EmbedQueue> {
    q.0.lock().unwrap_or_else(|e| e.into_inner())
}

/// Ends the embedding worker when the search returns early (an error, or the deadline while
/// decoding): what is pending is dropped.
struct StopWorker<'a>(&'a SharedQueue);

impl Drop for StopWorker<'_> {
    fn drop(&mut self) {
        let mut g = lock_queue(self.0);
        if !g.done {
            g.pending.clear();
            g.done = true;
        }
        drop(g);
        self.0 .1.notify_all();
    }
}

/// Tells the waiting search that the worker ended, however it ended.
struct WorkerEnded<'a>(&'a SharedQueue);

impl Drop for WorkerEnded<'_> {
    fn drop(&mut self) {
        lock_queue(self.0).ended = true;
        self.0 .1.notify_all();
    }
}

/// The embedding worker of one reranked search: up to `emb.batch()` pending candidates at a time,
/// the best R0 first, until nothing more will come and nothing is pending. The vectors go to the
/// queue's `got` as they are made.
fn embed_worker(q: &SharedQueue, emb: &dyn DocEmbed) -> Result<()> {
    let _ended = WorkerEnded(q);
    let batch = emb.batch().max(1);
    loop {
        let take: Vec<(usize, f32, u32, String)> = {
            let mut g = lock_queue(q);
            loop {
                let theta = g.theta;
                g.pending.retain(|p| p.1 >= theta);
                if !g.pending.is_empty() || g.done {
                    break;
                }
                g = q.1.wait(g).unwrap_or_else(|e| e.into_inner());
            }
            if g.pending.is_empty() {
                return Ok(());
            }
            g.pending.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            let n = g.pending.len().min(batch);
            g.pending.drain(..n).collect()
        };
        let (cis, docs): (Vec<usize>, Vec<(u32, String)>) = take.into_iter().map(|(ci, _, id, p)| (ci, (id, p))).unzip();
        let vecs = emb.embed(&docs)?;
        lock_queue(q).got.extend(cis.into_iter().zip(vecs));
        q.1.notify_all();
    }
}

fn ms(t: Instant) -> f64 {
    (t.elapsed().as_secs_f64() * 1e4).round() / 10.0
}

/// A reranked search of a block (see the module doc). `qvec`: the query's embedding.
///
/// Decoding and re-embedding overlap: each candidate's features are computed as soon as its
/// prefix is there, and a candidate whose R0 is already among the `embed_top` best lower bounds
/// (R0 when known, s1 while it is being decoded: b >= 0) is embedded at once, while the LM
/// decodes the rest. The final choice uses every R0, so the result never depends on timing; at
/// worst a few speculative embeddings are wasted.
///
/// `deadline` (opt-in): if the prefixes are not all decoded by then, the rest are cancelled and
/// the answer is stage 1's; if the top candidates are not all re-embedded by then, they keep the
/// cheap score's order. Either way `report.degraded` says so; without a deadline the result never
/// depends on timing.
pub fn search(eng: &Engine, q: &Query, qvec: &[f32], cfg: &RerankCfg, emb: &dyn DocEmbed, deadline: Option<Instant>) -> Result<Outcome> {
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
    let n = cands.len();
    let mut report = Report { candidates: n, ms_stage1: ms(t0), ..Default::default() };
    let sg = &eng.segs[0];
    let TextSource::Lm(lm) = &sg.s.text else { bail!("el reordenado necesita un bloque compacto") };
    let bm = sg.bm25.as_ref().context("el bloque no tiene índice de palabras")?;

    // keyword mode: a rare word counts once in a document of its postings, even past the prefix
    let in_rare: Vec<Vec<bool>> = if q.mode == Mode::Keyword {
        let mut sorted: Vec<u32> = cands.iter().map(|c| c.doc).collect();
        sorted.sort_unstable();
        let pos: HashMap<u32, usize> = sorted.iter().enumerate().map(|(i, &d)| (d, i)).collect();
        terms
            .iter()
            .map(|t| {
                if t.post == TermPost::Frequent {
                    return Ok(vec![false; n]);
                }
                let s = bm.score_docs(&[(t.post, t.idf)], &sorted)?;
                Ok(cands.iter().map(|c| s[pos[&c.doc]] > 0.0).collect())
            })
            .collect::<Result<_>>()?
    } else {
        Vec::new()
    };
    // a candidate's features from its prefix (or, without one, the whole text of its record):
    // b = BM25 of every query word over that text, R0, and its document prompt
    let feat_of = |ci: usize, p: Option<&Prefix>, whole: Option<&str>| -> Feat {
        let c = &cands[ci];
        let (text, prompt) = match p {
            Some(p) => {
                let s = prefix_string(&p.bytes);
                let mut parts = s.splitn(3, '\n');
                let (title, url, body) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""), parts.next().unwrap_or(""));
                let host = url.split("://").nth(1).unwrap_or("").split('/').next().unwrap_or("");
                (format!("{title} {host} {}", body_text(body, p.flags as u64)), c.has_vec.then(|| doc_prompt(&s, p.flags as u64)))
            }
            None => (whole.unwrap_or("").to_string(), None),
        };
        let mut tf: HashMap<&str, u32> = HashMap::new();
        for w in crate::tokenize::terms(&text) {
            if let Some(t) = terms.iter().find(|t| t.text == w) {
                *tf.entry(t.text.as_str()).or_default() += 1;
            }
        }
        let mut b = 0f32;
        for (ti, t) in terms.iter().enumerate() {
            let mut m = tf.get(t.text.as_str()).copied().unwrap_or(0);
            if q.mode == Mode::Keyword && m == 0 && in_rare[ti][ci] {
                m = 1;
            }
            if m > 0 {
                b += bm.score_tf(t.idf, m, c.doc);
            }
        }
        Feat { doc: c.doc, s1: c.s1, b, r0: c.s1 + cfg.r0_w * b / (b + cfg.r0_c), prompt }
    };

    // prefixes: cached, stored (LM-less records), or decoded by the shared LM service
    let t1 = Instant::now();
    let mut prefixes: HashMap<u32, std::sync::Arc<Prefix>> = HashMap::new();
    let mut feats: Vec<Option<Feat>> = (0..n).map(|_| None).collect();
    let (mut jobs, mut job_ci) = (Vec::new(), Vec::new());
    {
        let mut cache = eng.prefixes.lock().unwrap_or_else(|e| e.into_inner());
        for (ci, c) in cands.iter().enumerate() {
            if let Some(p) = cache.get(&c.doc) {
                prefixes.insert(c.doc, p);
            } else if let Some((flags, p)) = eng.fallback_prefix(c.doc) {
                prefixes.insert(c.doc, std::sync::Arc::new(Prefix { bytes: p.as_bytes().to_vec(), flags, complete: false, snippet_end: p.len() }));
            } else {
                continue;
            }
            feats[ci] = Some(feat_of(ci, prefixes.get(&c.doc).map(|p| p.as_ref()), None));
        }
    }
    for (ci, c) in cands.iter().enumerate() {
        if feats[ci].is_none() {
            jobs.push(Job { rec: lm.raw(c.doc as usize)?.into_owned(), has_check: lm.rec_check(), want: Want::Prefix { body: PREFIX_TOKENS, mark: SNIPPET_TOKENS } });
            job_ci.push(ci);
        }
    }
    report.decoded = jobs.len();
    let n_e = match q.mode {
        Mode::Vector => n,
        Mode::Hybrid => cfg.embed_top.unwrap_or(n).min(n),
        Mode::Keyword => 0,
    };
    let ids: Vec<u32> = cands.iter().map(|c| sg.s.meta.try_row(c.doc as usize).map(|r| r.id)).collect::<Result<_>>()?;
    let queue = new_queue();
    // None: the deadline passed while decoding; Some((vectors, late)): late = it passed while
    // re-embedding
    type Scoped = Option<(HashMap<usize, Vec<f32>>, bool)>;
    let scoped: Scoped = std::thread::scope(|sc| -> Result<Scoped> {
        let worker = (n_e > 0).then(|| sc.spawn(|| embed_worker(&queue, emb)));
        let _stop = StopWorker(&queue);
        let mut sent = vec![false; n];
        // queue every decoded candidate whose R0 is among the n_e best lower bounds; the bound
        // only rises, so the worker drops the queued ones that fall below it
        let speculate = |feats: &[Option<Feat>], sent: &mut [bool]| {
            if n_e == 0 {
                return;
            }
            let mut lb: Vec<f32> = (0..n).map(|i| feats[i].as_ref().map_or(cands[i].s1, |f| f.r0)).collect();
            let theta = *lb.select_nth_unstable_by(n_e - 1, |a, b| b.total_cmp(a)).1;
            let mut g = lock_queue(&queue);
            g.theta = theta;
            for i in 0..n {
                if let (false, Some(f)) = (sent[i], &feats[i]) {
                    if f.r0 >= theta {
                        if let Some(p) = &f.prompt {
                            g.pending.push((i, f.r0, ids[i], p.clone()));
                        }
                        sent[i] = true;
                    }
                }
            }
            drop(g);
            queue.1.notify_all();
        };
        speculate(&feats, &mut sent);
        if !jobs.is_empty() {
            // dropping the ticket (the deadline, an error) cancels what is not decoded yet
            let ticket = eng.lm_service("block")?.submit(jobs, Prio::Bulk);
            for _ in 0..ticket.n {
                let (j, r) = match deadline {
                    Some(d) => match ticket.recv_until(d)? {
                        Some(x) => x,
                        None => {
                            report.ms_decode = ms(t1);
                            return Ok(None);
                        }
                    },
                    None => ticket.recv().context("el decodificador LM se detuvo")?,
                };
                let ci = job_ci[j];
                let d = cands[ci].doc;
                feats[ci] = Some(match r? {
                    Decoded::Prefix(p) => {
                        report.tokens += p.tokens as u64;
                        let p = std::sync::Arc::new(Prefix { bytes: p.bytes, flags: p.flags, complete: p.complete, snippet_end: p.snippet_end });
                        eng.prefixes.lock().unwrap_or_else(|e| e.into_inner()).put(d, p.clone());
                        let f = feat_of(ci, Some(&p), None);
                        prefixes.insert(d, p);
                        f
                    }
                    // an empty record, or a zstd record without a vector: BM25 over its whole text
                    Decoded::Text(t, _) => feat_of(ci, None, Some(&crate::segment::index_text(&t.title, &t.url, &t.text))),
                });
                speculate(&feats, &mut sent);
            }
        }
        report.ms_decode = ms(t1);
        let t_tail = Instant::now();
        // the final choice: the top n_e by R0; the queued ones outside it are dropped, those
        // inside it not queued yet are added
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            let (fa, fb) = (feats[a].as_ref().unwrap(), feats[b].as_ref().unwrap());
            fb.r0.total_cmp(&fa.r0).then(fb.doc.cmp(&fa.doc))
        });
        {
            let mut top = vec![false; n];
            for &i in &order[..n_e] {
                top[i] = true;
            }
            let mut g = lock_queue(&queue);
            g.pending.retain(|p| top[p.0]);
            for &i in &order[..n_e] {
                if !sent[i] {
                    let f = feats[i].as_ref().unwrap();
                    if let Some(p) = &f.prompt {
                        g.pending.push((i, f.r0, ids[i], p.clone()));
                    }
                    sent[i] = true;
                }
            }
            g.done = true;
        }
        queue.1.notify_all();
        // wait for the vectors of the top n_e (until the deadline, if any)
        let need: Vec<usize> = order[..n_e].iter().copied().filter(|&i| feats[i].as_ref().is_some_and(|f| f.prompt.is_some())).collect();
        let mut late = false;
        {
            let mut g = lock_queue(&queue);
            loop {
                let have: std::collections::HashSet<usize> = g.got.iter().map(|x| x.0).collect();
                if g.ended || need.iter().all(|i| have.contains(i)) {
                    break;
                }
                g = match deadline {
                    Some(d) => {
                        let now = Instant::now();
                        if now >= d {
                            // the worker ends after the document it is embedding
                            g.pending.clear();
                            late = true;
                            break;
                        }
                        queue.1.wait_timeout(g, d - now).unwrap_or_else(|e| e.into_inner()).0
                    }
                    None => queue.1.wait(g).unwrap_or_else(|e| e.into_inner()),
                };
            }
        }
        queue.1.notify_all();
        if let Some(w) = worker {
            w.join().map_err(|_| anyhow::anyhow!("fallo interno al embeber documentos"))??;
        }
        report.ms_embed = ms(t_tail);
        let got = std::mem::take(&mut lock_queue(&queue).got);
        Ok(Some((got.into_iter().collect(), late)))
    })?;
    let Some((embedded, late)) = scoped else {
        // the deadline passed while decoding: stage 1's answer (decoded prefixes still serve the
        // results' text)
        report.degraded = Some("decode");
        let mut hits = eng.search(q, if q.mode == Mode::Keyword { None } else { Some(qvec) })?;
        hits.truncate(k);
        let warning = Some("se agotó el plazo durante la decodificación: resultados de la etapa 1, sin reordenar".to_string());
        return Ok(Outcome { hits, prefixes, report, warning });
    };
    if late {
        report.degraded = Some("embed");
    }
    report.embedded = embedded.len();
    let feats: Vec<Feat> = feats.into_iter().map(|f| f.expect("every candidate has its features")).collect();

    // the final order
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
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| desc(feats[a].r0, feats[b].r0, feats[a].doc, feats[b].doc));
            // the deadline passed before the top n_e were re-embedded: everything in R0 order
            let (top, rest) = order.split_at(if late { 0 } else { n_e });
            let w = if q.mode == Mode::Vector { 0.0 } else { cfg.w };
            let mut head: Vec<(u32, f32, f32, f32)> = top
                .iter()
                .map(|&i| {
                    let f = &feats[i];
                    let c = embedded.get(&i).map_or(0.0, |v| v.iter().zip(qvec).map(|(a, b)| a * b).sum());
                    (f.doc, c + w * f.b / (f.b + cfg.c), c, f.b)
                })
                .collect();
            head.sort_by(|a, b| desc(a.1, b.1, a.0, b.0));
            scored.extend(head);
            scored.extend(rest.iter().map(|&i| (feats[i].doc, feats[i].r0, feats[i].s1, feats[i].b)));
            if late {
                warning = Some("se agotó el plazo durante el re-embebido: resultados en el orden de la puntuación barata".to_string());
            }
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Records the ids it is asked to embed; each vector is [id].
    struct Rec(Mutex<Vec<u32>>);

    impl DocEmbed for Rec {
        fn embed(&self, docs: &[(u32, String)]) -> Result<Vec<Vec<f32>>> {
            self.0.lock().unwrap().extend(docs.iter().map(|d| d.0));
            Ok(docs.iter().map(|d| vec![d.0 as f32]).collect())
        }

        fn batch(&self) -> usize {
            1
        }
    }

    #[test]
    fn worker_takes_the_best_first_and_drops_the_hopeless() {
        let q = new_queue();
        {
            let mut g = lock_queue(&q);
            for (ci, r0) in [(0usize, 0.5f32), (1, 0.9), (2, 0.1), (3, 0.7), (4, 0.4)] {
                g.pending.push((ci, r0, 100 + ci as u32, String::new()));
            }
            // candidate 2 can no longer be among the best; 4 is exactly at the bound and stays
            g.theta = 0.4;
            g.done = true;
        }
        let emb = Rec(Mutex::new(Vec::new()));
        embed_worker(&q, &emb).unwrap();
        assert_eq!(*emb.0.lock().unwrap(), vec![101, 103, 100, 104]);
        let g = lock_queue(&q);
        assert!(g.ended);
        assert_eq!(g.got.iter().map(|g| (g.0, g.1[0])).collect::<Vec<_>>(), vec![(1, 101.0), (3, 103.0), (0, 100.0), (4, 104.0)]);
    }

    #[test]
    fn worker_waits_for_work_and_ends_when_stopped() {
        let q = new_queue();
        let emb = Rec(Mutex::new(Vec::new()));
        std::thread::scope(|sc| {
            let w = sc.spawn(|| embed_worker(&q, &emb));
            lock_queue(&q).pending.push((0, 1.0, 7, String::new()));
            q.1.notify_all();
            // an early return: the guard ends the worker (what is still pending is dropped)
            drop(StopWorker(&q));
            w.join().unwrap().unwrap();
        });
        let g = lock_queue(&q);
        assert!(g.done && g.ended && g.got.len() <= 1);
    }
}
