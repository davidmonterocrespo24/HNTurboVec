//! Índice BM25 compacto por segmento.
//! - Diccionario de términos: FST (`fst`), valor = offset en postings, o el posting inline si df=1.
//! - Postings: bloques de 128 (docID delta + tf) empaquetados con BitPacker4x (SIMD-BP128),
//!   cabeceras de salto (último doc de cada bloque) delante; cola en varints.
//! - Normas de longitud: 1 byte/doc (escala logarítmica).
//! - Búsqueda top-k con MaxScore y saltos por rango de docIDs (= rango de fechas).

use crate::tokenize;
use crate::varint::{put, try_get};
use anyhow::{Context, Result};
use bitpacking::{BitPacker, BitPacker4x};
use fst::{Map, MapBuilder};
use crate::blob::Blob;
use std::borrow::Cow;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

pub const K1: f32 = 1.2;
pub const B: f32 = 0.75;
const BLK: usize = BitPacker4x::BLOCK_LEN; // 128
pub const END: u32 = u32::MAX;

fn norm_encode(len: u32) -> u8 {
    ((16.0 * (1.0 + len as f32).log2()).round() as u32).min(255) as u8
}
fn norm_decode(q: u8) -> f32 {
    (2f32.powf(q as f32 / 16.0) - 1.0).max(0.0)
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Bm25Info {
    pub n_docs: usize,
    pub avgdl: f32,
    pub n_terms: usize,
    pub n_postings: u64,
    /// false = postings booleanos (tf=1), más pequeños
    #[serde(default = "yes")]
    pub with_tf: bool,
    /// términos con df/n_docs > este valor no se indexan (stopwords automáticas)
    #[serde(default = "one")]
    pub max_df_ratio: f32,
    /// dictionary value layout: 0 = segment (Single/List), 1 = block v1 (tagged, see `LexWriter`)
    #[serde(default)]
    pub layout: u32,
    /// sum of the documents' lengths (indexable terms); block layout
    #[serde(default)]
    pub total_len: u64,
    /// block layout: terms with a global df above this have no postings (only their df)
    #[serde(default)]
    pub max_df_abs: u64,
}

/// Dictionary value layout of a block (`Bm25Info::layout = 1`): tag = v & 3.
pub const LAYOUT_BLOCK: u32 = 1;
const TAG_FREQUENT: u64 = 0;
const TAG_SINGLE: u64 = 1;
const TAG_LIST: u64 = 2;

fn yes() -> bool {
    true
}
fn one() -> f32 {
    1.0
}

pub struct Builder {
    dict: HashMap<String, u32>,
    post: Vec<Vec<(u32, u32)>>,
    norms: Vec<u8>,
    total_len: u64,
}

impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}

impl Builder {
    pub fn new() -> Self {
        Self { dict: HashMap::new(), post: Vec::new(), norms: Vec::new(), total_len: 0 }
    }

    /// Añade documentos consecutivos (docID = nº de documentos ya añadidos).
    pub fn add_batch(&mut self, texts: &[String]) {
        let start = self.norms.len() as u32;
        let toks: Vec<(Vec<(String, u32)>, u32)> = texts
            .par_iter()
            .map(|t| {
                let mut m: HashMap<String, u32> = HashMap::new();
                let mut n = 0u32;
                tokenize::for_each_term(t, |w| {
                    n += 1;
                    if let Some(c) = m.get_mut(w) {
                        *c += 1;
                    } else {
                        m.insert(w.to_string(), 1);
                    }
                });
                (m.into_iter().collect(), n)
            })
            .collect();
        for (k, (terms, n)) in toks.into_iter().enumerate() {
            let doc = start + k as u32;
            self.norms.push(norm_encode(n));
            self.total_len += n as u64;
            for (w, tf) in terms {
                let id = match self.dict.get(&w) {
                    Some(&i) => i,
                    None => {
                        let i = self.post.len() as u32;
                        self.dict.insert(w, i);
                        self.post.push(Vec::new());
                        i
                    }
                };
                self.post[id as usize].push((doc, tf));
            }
        }
    }

    pub fn write(self, dir: &Path) -> Result<Bm25Info> {
        self.write_opts(dir, true, 1.0)
    }

    pub fn write_opts(self, dir: &Path, with_tf: bool, max_df_ratio: f32) -> Result<Bm25Info> {
        let max_df = (max_df_ratio as f64 * self.norms.len() as f64).ceil() as usize;
        let mut terms: Vec<(String, u32)> = self.dict.into_iter().filter(|(_, id)| self.post[*id as usize].len() <= max_df.max(1)).collect();
        terms.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        let mut post_bytes: Vec<u8> = Vec::new();
        let mut fb = MapBuilder::new(BufWriter::new(File::create(dir.join("bm25.fst"))?))?;
        let mut n_postings = 0u64;
        for (w, id) in &terms {
            let p = &self.post[*id as usize];
            n_postings += p.len() as u64;
            let val = if p.len() == 1 {
                let tf = if with_tf { p[0].1.min(255) } else { 1 };
                1u64 | ((tf as u64) << 1) | ((p[0].0 as u64) << 9)
            } else {
                let off = post_bytes.len() as u64;
                encode_postings(p, &mut post_bytes, with_tf);
                off << 1
            };
            fb.insert(w.as_bytes(), val)?;
        }
        fb.finish()?;
        std::fs::write(dir.join("bm25.post"), &post_bytes)?;
        std::fs::write(dir.join("bm25.norms"), &self.norms)?;
        let info = Bm25Info {
            n_docs: self.norms.len(),
            avgdl: self.total_len as f32 / self.norms.len().max(1) as f32,
            n_terms: terms.len(),
            n_postings,
            with_tf,
            max_df_ratio,
            layout: 0,
            total_len: self.total_len,
            max_df_abs: 0,
        };
        std::fs::write(dir.join("bm25.json"), serde_json::to_string_pretty(&info)?)?;
        Ok(info)
    }
}

fn encode_postings(p: &[(u32, u32)], out: &mut Vec<u8>, with_tf: bool) {
    put(out, p.len() as u64);
    let nfull = p.len() / BLK;
    let bp = BitPacker4x::new();
    let mut headers = Vec::new();
    let mut payloads = Vec::new();
    let mut prev_last = 0u32;
    let mut buf = vec![0u8; 4 * BLK + 16];
    let mut docs = [0u32; BLK];
    let mut tfs = [0u32; BLK];
    for b in 0..nfull {
        for (i, x) in p[b * BLK..(b + 1) * BLK].iter().enumerate() {
            docs[i] = x.0;
            tfs[i] = x.1 - 1;
        }
        let db = bp.num_bits_sorted(prev_last, &docs);
        let tb = if with_tf { bp.num_bits(&tfs) } else { 0 };
        let start = payloads.len();
        payloads.push(db);
        payloads.push(tb);
        let n = bp.compress_sorted(prev_last, &docs, &mut buf, db);
        payloads.extend_from_slice(&buf[..n]);
        if with_tf {
            let n = bp.compress(&tfs, &mut buf, tb);
            payloads.extend_from_slice(&buf[..n]);
        }
        put(&mut headers, (docs[BLK - 1] - prev_last) as u64);
        put(&mut headers, (payloads.len() - start) as u64);
        prev_last = docs[BLK - 1];
    }
    out.extend_from_slice(&headers);
    out.extend_from_slice(&payloads);
    let mut prev = prev_last;
    for x in &p[nfull * BLK..] {
        put(out, (x.0 - prev) as u64);
        if with_tf {
            put(out, x.1 as u64);
        }
        prev = x.0;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TermPost {
    Single { doc: u32, tf: u32 },
    List { off: usize },
    /// a term too frequent to have postings (block layout): only its df is known
    Frequent,
}

/// Cursor over a postings list. Each block of 128 is read when the cursor reaches it: borrowed if
/// bm25.post is in RAM, or with one disk read (memory does not depend on the list's size). Every
/// block is validated (sizes, bit widths, increasing documents below `n_docs`): a damaged list
/// ends the cursor and is reported by `bad` (the postings of a block are read without a hash).
pub struct Cursor<'a> {
    post: &'a Blob,
    n_docs: usize,
    /// what was wrong with the list (read error or damaged bytes), if anything
    bad: Option<String>,
    /// start of the list in bm25.post
    base: usize,
    with_tf: bool,
    single: Option<(u32, u32)>,
    df: usize,
    nfull: usize,
    lasts: Vec<u32>,
    /// start of the payload of each full block (relative to `base`)
    pay: Vec<usize>,
    tail_off: usize,
    blk: usize,
    docs: [u32; BLK],
    tfs: [u32; BLK],
    len: usize,
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(post: &'a Blob, t: TermPost, with_tf: bool, n_docs: usize) -> Self {
        let mut c = Cursor {
            post,
            n_docs,
            bad: None,
            base: 0,
            with_tf,
            single: None,
            df: 1,
            nfull: 0,
            lasts: Vec::new(),
            pay: Vec::new(),
            tail_off: 0,
            blk: 0,
            docs: [0; BLK],
            tfs: [0; BLK],
            len: 0,
            pos: 0,
        };
        match t {
            TermPost::Single { doc, tf } => {
                c.single = Some((doc, tf));
                c.docs[0] = doc;
                c.tfs[0] = tf;
                c.len = 1;
            }
            TermPost::List { off } => {
                c.base = off;
                if !c.header() {
                    c.exhaust();
                }
            }
            TermPost::Frequent => c.exhaust(),
        }
        c
    }

    /// Bytes [o, o+len) of the list (clipped at the end of the file); None if the read fails.
    fn read(&mut self, o: usize, len: usize) -> Option<Cow<'a, [u8]>> {
        let post = self.post;
        match post.get(self.base + o, len) {
            Ok(b) => Some(b),
            Err(e) => {
                self.fail(format!("{e:#}"));
                None
            }
        }
    }

    /// Ends the cursor and records why (the first reason is kept).
    fn fail(&mut self, why: String) {
        if self.bad.is_none() {
            self.bad = Some(why);
        }
        self.exhaust();
    }

    fn damaged(&mut self, what: &str) -> bool {
        self.fail(format!("postings dañados ({what}) en la posición {}", self.base));
        false
    }

    /// Why the list could not be read whole, if it could not.
    pub fn bad(&self) -> Option<&str> {
        self.bad.as_deref()
    }

    /// Exhausted list (e.g. after a read error): `doc()` returns END.
    fn exhaust(&mut self) {
        self.single = None;
        self.nfull = 0;
        self.blk = 0;
        self.len = 0;
        self.pos = 0;
    }

    /// Header: df and, per full block, its last doc and payload length. Loads block 0.
    fn header(&mut self) -> bool {
        let Some(head) = self.read(0, 10) else { return false };
        let mut p = 0;
        let Some(df) = try_get(&head, &mut p) else { return self.damaged("df") };
        if df == 0 || df > self.n_docs as u64 {
            return self.damaged("df");
        }
        let df = df as usize;
        let nfull = df / BLK;
        let Some(h) = self.read(0, p + nfull * 20) else { return false };
        let mut last = 0u64;
        let mut plen = Vec::with_capacity(nfull);
        for b in 0..nfull {
            let Some(dl) = try_get(&h, &mut p) else { return self.damaged("cabecera") };
            let Some(pl) = try_get(&h, &mut p) else { return self.damaged("cabecera") };
            // each full block holds 128 increasing documents; its payload is 2 bytes + at most 32
            // bits per document and per frequency
            last += dl;
            if last >= self.n_docs as u64 || (b > 0 && dl < BLK as u64) || (b == 0 && last < BLK as u64 - 1) || !(2..=1026).contains(&pl) {
                return self.damaged("cabecera");
            }
            self.lasts.push(last as u32);
            plen.push(pl as usize);
        }
        for l in plen {
            self.pay.push(p);
            p += l;
        }
        self.df = df;
        self.nfull = nfull;
        self.tail_off = p;
        self.load(0)
    }

    fn load(&mut self, b: usize) -> bool {
        self.blk = b;
        self.pos = 0;
        let bp = BitPacker4x::new();
        if b < self.nfull {
            let o = self.pay[b];
            let end = if b + 1 < self.nfull { self.pay[b + 1] } else { self.tail_off };
            let Some(data) = self.read(o, end - o) else { return false };
            if data.len() != end - o || data.len() < 2 {
                return self.damaged("bloque");
            }
            let (db, tb) = (data[0], data[1]);
            let need = 2 + BitPacker4x::compressed_block_size(db) + if self.with_tf { BitPacker4x::compressed_block_size(tb) } else { 0 };
            if db > 32 || tb > 32 || data.len() != need {
                return self.damaged("bloque");
            }
            let init = if b == 0 { 0 } else { self.lasts[b - 1] };
            let n = bp.decompress_sorted(init, &data[2..], &mut self.docs, db);
            if (b > 0 && self.docs[0] <= init) || self.docs.windows(2).any(|w| w[1] <= w[0]) || self.docs[BLK - 1] != self.lasts[b] {
                return self.damaged("bloque");
            }
            if self.with_tf {
                bp.decompress(&data[2 + n..], &mut self.tfs, tb);
                for t in self.tfs.iter_mut() {
                    *t = t.saturating_add(1);
                }
            } else {
                self.tfs.fill(1);
            }
            self.len = BLK;
        } else {
            let cnt = self.df - self.nfull * BLK;
            let Some(data) = self.read(self.tail_off, cnt * 20) else { return false };
            let mut p = 0;
            let mut prev = if self.nfull == 0 { 0u64 } else { self.lasts[self.nfull - 1] as u64 };
            for i in 0..cnt {
                let Some(d) = try_get(&data, &mut p) else { return self.damaged("cola") };
                // increasing documents (only the very first of a list may be doc 0)
                if d == 0 && (i > 0 || self.nfull > 0) || prev + d >= self.n_docs as u64 {
                    return self.damaged("cola");
                }
                prev += d;
                self.docs[i] = prev as u32;
                self.tfs[i] = if self.with_tf {
                    match try_get(&data, &mut p) {
                        Some(t) => t.min(u32::MAX as u64) as u32,
                        None => return self.damaged("cola"),
                    }
                } else {
                    1
                };
            }
            self.len = cnt;
        }
        true
    }

    #[inline]
    pub fn doc(&self) -> u32 {
        if self.pos < self.len {
            self.docs[self.pos]
        } else {
            END
        }
    }

    #[inline]
    pub fn tf(&self) -> u32 {
        self.tfs[self.pos]
    }

    pub fn next(&mut self) {
        self.pos += 1;
        if self.pos >= self.len && self.single.is_none() && self.blk < self.nfull {
            let _ = self.load(self.blk + 1);
        }
    }

    /// Avanza hasta el primer doc >= target.
    pub fn advance(&mut self, target: u32) {
        if self.doc() >= target {
            return;
        }
        if self.single.is_some() {
            self.pos = 1;
            return;
        }
        if self.blk < self.nfull && self.lasts[self.blk] < target {
            let mut b = self.blk + 1;
            while b < self.nfull && self.lasts[b] < target {
                b += 1;
            }
            if !self.load(b) {
                return;
            }
        }
        loop {
            while self.pos < self.len && self.docs[self.pos] < target {
                self.pos += 1;
            }
            if self.pos < self.len || self.blk >= self.nfull {
                return;
            }
            if !self.load(self.blk + 1) {
                return;
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, PartialOrd)]
struct Sc(f32);
impl Eq for Sc {}
impl Ord for Sc {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&o.0)
    }
}

pub struct Bm25 {
    fst: Map<Vec<u8>>,
    post: Blob,
    norms: Vec<u8>,
    pub info: Bm25Info,
}

impl Bm25 {
    pub fn open(dir: &Path) -> Result<Self> {
        Self::open_with(dir, true)
    }

    /// The dictionary (FST) and the norms are always in RAM; the postings lists, in RAM if
    /// `resident_postings`, otherwise read from disk block by block.
    pub fn open_with(dir: &Path, resident_postings: bool) -> Result<Self> {
        let fst = Map::new(std::fs::read(dir.join("bm25.fst")).context("bm25.fst")?)?;
        let post = Blob::open(&dir.join("bm25.post"), resident_postings)?;
        let norms = std::fs::read(dir.join("bm25.norms"))?;
        let info: Bm25Info = serde_json::from_str(&std::fs::read_to_string(dir.join("bm25.json"))?)?;
        Ok(Self { fst, post, norms, info })
    }

    /// From the parts of a block: the dictionary, the postings (in RAM or on disk) and the norms.
    pub fn from_parts(fst: Vec<u8>, post: Blob, norms: Vec<u8>, info: Bm25Info) -> Result<Self> {
        if norms.len() != info.n_docs {
            anyhow::bail!("norms: {} bytes for {} documents", norms.len(), info.n_docs);
        }
        Ok(Self { fst: Map::new(fst)?, post, norms, info })
    }

    /// Bytes it takes in RAM.
    pub fn ram_bytes(&self) -> usize {
        self.fst.as_fst().as_bytes().len() + self.post.resident_bytes() + self.norms.capacity()
    }

    pub fn blob(&self) -> &Blob {
        &self.post
    }

    pub fn blob_mut(&mut self) -> &mut Blob {
        &mut self.post
    }

    /// (postings, df) del término en este segmento.
    pub fn lookup(&self, term: &str) -> Option<(TermPost, u32)> {
        let v = self.fst.get(term)?;
        if self.info.layout == LAYOUT_BLOCK {
            return match v & 3 {
                TAG_FREQUENT => Some((TermPost::Frequent, (v >> 2) as u32)),
                TAG_SINGLE => Some((TermPost::Single { doc: (v >> 2) as u32, tf: 1 }, 1)),
                TAG_LIST => {
                    let off = (v >> 2) as usize;
                    let head = self.post.get(off, 10).ok()?;
                    let mut p = 0;
                    let df = crate::varint::try_get(&head, &mut p)? as u32;
                    Some((TermPost::List { off }, df))
                }
                _ => None,
            };
        }
        if v & 1 == 1 {
            Some((TermPost::Single { doc: (v >> 9) as u32, tf: ((v >> 1) & 0xff) as u32 }, 1))
        } else {
            let off = (v >> 1) as usize;
            let head = self.post.get(off, 10).ok()?;
            let mut p = 0;
            let df = try_get(&head, &mut p)? as u32;
            Some((TermPost::List { off }, df))
        }
    }

    pub fn cursor(&self, t: TermPost) -> Cursor<'_> {
        Cursor::new(&self.post, t, self.info.with_tf, self.info.n_docs)
    }

    /// A damaged or unreadable postings list among the cursors -> `CorruptBlockError`.
    fn check<'b, 'c: 'b>(&self, mut cur: impl Iterator<Item = &'b Cursor<'c>>) -> Result<()> {
        match cur.find_map(|c| c.bad()) {
            Some(why) => Err(crate::block::corrupt(format!("índice de palabras: {why}"))),
            None => Ok(()),
        }
    }

    #[inline]
    fn score(&self, idf: f32, tf: u32, doc: u32) -> f32 {
        let dl = norm_decode(self.norms[doc as usize]);
        let tf = tf as f32;
        idf * tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * dl / self.info.avgdl))
    }

    /// Top-k en [lo, hi) con MaxScore. `terms`: (postings, idf global).
    pub fn top_k(&self, terms: &[(TermPost, f32)], lo: u32, hi: u32, k: usize) -> Result<Vec<(u32, f32)>> {
        if terms.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        let mut cur: Vec<(Cursor, f32, f32)> = terms
            .iter()
            .map(|&(t, idf)| {
                let mut c = self.cursor(t);
                c.advance(lo);
                (c, idf, idf * (K1 + 1.0))
            })
            .collect();
        cur.sort_by(|a, b| a.2.total_cmp(&b.2));
        let n = cur.len();
        let mut prefix = vec![0f32; n];
        let mut acc = 0.0;
        for i in 0..n {
            acc += cur[i].2;
            prefix[i] = acc;
        }
        let mut heap: BinaryHeap<Reverse<(Sc, u32)>> = BinaryHeap::with_capacity(k + 1);
        let mut theta = 0f32;
        let mut first_ess = 0usize;
        loop {
            while first_ess < n && prefix[first_ess] <= theta {
                first_ess += 1;
            }
            if first_ess >= n {
                break;
            }
            let doc = cur[first_ess..].iter().map(|c| c.0.doc()).min().unwrap();
            if doc >= hi || doc == END {
                break;
            }
            let mut s = 0f32;
            for c in cur[first_ess..].iter_mut() {
                if c.0.doc() == doc {
                    s += self.score(c.1, c.0.tf(), doc);
                    c.0.next();
                }
            }
            for i in (0..first_ess).rev() {
                if s + prefix[i] <= theta {
                    break;
                }
                let c = &mut cur[i];
                c.0.advance(doc);
                if c.0.doc() == doc {
                    s += self.score(c.1, c.0.tf(), doc);
                }
            }
            if heap.len() < k {
                heap.push(Reverse((Sc(s), doc)));
                if heap.len() == k {
                    theta = heap.peek().unwrap().0 .0 .0;
                }
            } else if s > theta {
                heap.pop();
                heap.push(Reverse((Sc(s), doc)));
                theta = heap.peek().unwrap().0 .0 .0;
            }
        }
        self.check(cur.iter().map(|c| &c.0))?;
        let mut v: Vec<(u32, f32)> = heap.into_iter().map(|Reverse((s, d))| (d, s.0)).collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(v)
    }

    /// Puntuación BM25 de una lista de docs (ordenados ascendentemente).
    pub fn score_docs(&self, terms: &[(TermPost, f32)], docs: &[u32]) -> Result<Vec<f32>> {
        let mut out = vec![0f32; docs.len()];
        for &(t, idf) in terms {
            let mut c = self.cursor(t);
            for (i, &d) in docs.iter().enumerate() {
                c.advance(d);
                if c.doc() == END {
                    break;
                }
                if c.doc() == d {
                    out[i] += self.score(idf, c.tf(), d);
                }
            }
            self.check(std::iter::once(&c))?;
        }
        Ok(out)
    }
}

/// Builds the word index of a block (P3, layout 1): for every term its df in the block; postings
/// (document ids, without tf) only for the terms that are rare in all of HN (global df <=
/// `max_df_abs`, from `hnz globals`). Dictionary values: tag = v & 3, 0 FREQUENT (df = v >> 2),
/// 1 SINGLE (doc = v >> 2), 2 LIST (offset = v >> 2 into the postings, which start with the df).
pub struct LexWriter<'a> {
    classes: &'a Map<Vec<u8>>,
    max_df_abs: u64,
    max_df_ratio: f32,
    /// term -> (df in the block, documents if the term is rare)
    terms: HashMap<String, (u32, Option<Vec<u32>>)>,
    norms: Vec<u8>,
    total_len: u64,
    missing: u64,
}

/// The parts of a block's word index.
pub struct LexParts {
    pub fst: Vec<u8>,
    pub post: Vec<u8>,
    pub norms: Vec<u8>,
    pub info: Bm25Info,
    /// (frequent, single, list) term counts
    pub classes: (u64, u64, u64),
}

impl<'a> LexWriter<'a> {
    pub fn new(classes: &'a Map<Vec<u8>>, max_df_abs: u64, max_df_ratio: f32) -> Self {
        Self { classes, max_df_abs, max_df_ratio, terms: HashMap::new(), norms: Vec::new(), total_len: 0, missing: 0 }
    }

    /// Documents with consecutive ids (the next id = documents added so far).
    pub fn add_batch(&mut self, texts: &[String]) {
        let start = self.norms.len() as u32;
        let toks: Vec<(Vec<String>, u32)> = texts
            .par_iter()
            .map(|t| {
                let mut v: Vec<String> = Vec::new();
                let mut n = 0u32;
                tokenize::for_each_term(t, |w| {
                    n += 1;
                    v.push(w.to_string());
                });
                v.sort_unstable();
                v.dedup();
                (v, n)
            })
            .collect();
        for (k, (terms, n)) in toks.into_iter().enumerate() {
            let doc = start + k as u32;
            self.norms.push(norm_encode(n));
            self.total_len += n as u64;
            for w in terms {
                match self.terms.get_mut(&w) {
                    Some(e) => {
                        e.0 += 1;
                        if let Some(d) = e.1.as_mut() {
                            d.push(doc);
                        }
                    }
                    None => {
                        let rare = match self.classes.get(&w) {
                            Some(g) => g <= self.max_df_abs,
                            None => {
                                self.missing += 1;
                                true
                            }
                        };
                        self.terms.insert(w, (1, rare.then(|| vec![doc])));
                    }
                }
            }
        }
    }

    pub fn n_docs(&self) -> usize {
        self.norms.len()
    }

    pub fn finish(self) -> Result<LexParts> {
        if self.missing > 0 {
            anyhow::bail!("{} terms are not in the global table (rebuild it with `hnz globals` over the same files)", self.missing);
        }
        let mut terms: Vec<(String, (u32, Option<Vec<u32>>))> = self.terms.into_iter().collect();
        terms.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        let mut fb = MapBuilder::memory();
        let mut post = Vec::new();
        let (mut nf, mut ns, mut nl, mut n_postings) = (0u64, 0u64, 0u64, 0u64);
        for (w, (df, docs)) in &terms {
            let v = match docs {
                None => {
                    nf += 1;
                    (*df as u64) << 2 | TAG_FREQUENT
                }
                Some(d) if d.len() == 1 => {
                    ns += 1;
                    n_postings += 1;
                    (d[0] as u64) << 2 | TAG_SINGLE
                }
                Some(d) => {
                    nl += 1;
                    n_postings += d.len() as u64;
                    let off = post.len() as u64;
                    let pairs: Vec<(u32, u32)> = d.iter().map(|&x| (x, 1)).collect();
                    encode_postings(&pairs, &mut post, false);
                    off << 2 | TAG_LIST
                }
            };
            fb.insert(w.as_bytes(), v)?;
        }
        let n_docs = self.norms.len();
        let info = Bm25Info {
            n_docs,
            avgdl: self.total_len as f32 / n_docs.max(1) as f32,
            n_terms: terms.len(),
            n_postings,
            with_tf: false,
            max_df_ratio: self.max_df_ratio,
            layout: LAYOUT_BLOCK,
            total_len: self.total_len,
            max_df_abs: self.max_df_abs,
        };
        Ok(LexParts { fst: fb.into_inner()?, post, norms: self.norms, info, classes: (nf, ns, nl) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn postings_roundtrip_and_search() {
        let dir = std::env::temp_dir().join(format!("hnz_bm25_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut texts = Vec::new();
        for i in 0..1000u32 {
            let mut t = String::from("filler words here ");
            if i % 3 == 0 {
                t.push_str("rust ");
            }
            if i % 7 == 0 {
                t.push_str("rust async tokio ");
            }
            if i == 999 {
                t.push_str("uniqueterm ");
            }
            texts.push(t);
        }
        let mut b = Builder::new();
        b.add_batch(&texts[..500]);
        b.add_batch(&texts[500..]);
        b.write(&dir).unwrap();
        let ix = Bm25::open(&dir).unwrap();
        let (tp, df) = ix.lookup("rust").unwrap();
        let expect: Vec<u32> = (0..1000).filter(|i| i % 3 == 0 || i % 7 == 0).collect();
        assert_eq!(df as usize, expect.len());
        let mut c = ix.cursor(tp);
        let mut got = Vec::new();
        while c.doc() != END {
            got.push(c.doc());
            c.next();
        }
        assert_eq!(got, expect);
        let mut c = ix.cursor(tp);
        c.advance(500);
        assert_eq!(c.doc(), 501);
        let (tu, dfu) = ix.lookup("uniqueterm").unwrap();
        assert_eq!(dfu, 1);
        let r = ix.top_k(&[(tu, 5.0)], 0, 1000, 3).unwrap();
        assert_eq!(r[0].0, 999);
        let (ta, _) = ix.lookup("async").unwrap();
        let r = ix.top_k(&[(tp, 1.0), (ta, 2.0)], 100, 400, 5).unwrap();
        assert!(r.iter().all(|x| x.0 >= 100 && x.0 < 400 && x.0 % 7 == 0));
        let s = ix.score_docs(&[(ta, 2.0)], &[7, 8, 14]).unwrap();
        assert!(s[0] > 0.0 && s[1] == 0.0 && s[2] > 0.0);
        // with the lists on disk (read block by block), exactly the same
        let dk = Bm25::open_with(&dir, false).unwrap();
        assert!(!dk.blob().is_resident());
        let mut c = dk.cursor(tp);
        let mut got2 = Vec::new();
        while c.doc() != END {
            got2.push(c.doc());
            c.next();
        }
        assert_eq!(got2, expect);
        let mut c = dk.cursor(tp);
        c.advance(500);
        assert_eq!(c.doc(), 501);
        assert_eq!(dk.top_k(&[(tp, 1.0), (ta, 2.0)], 100, 400, 5).unwrap(), ix.top_k(&[(tp, 1.0), (ta, 2.0)], 100, 400, 5).unwrap());
        assert_eq!(dk.score_docs(&[(ta, 2.0)], &[7, 8, 14]).unwrap(), s);
        drop(dk);
        drop(ix);
        std::fs::remove_dir_all(&dir).ok();
    }

    fn texts() -> Vec<String> {
        (0..1000u32)
            .map(|i| {
                let mut t = String::from("filler words here ");
                if i % 3 == 0 {
                    t.push_str("rust ");
                }
                if i % 7 == 0 {
                    t.push_str("rust async tokio ");
                }
                if i == 999 {
                    t.push_str("uniqueterm ");
                }
                t
            })
            .collect()
    }

    fn classes(texts: &[String]) -> Map<Vec<u8>> {
        let mut df: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
        for t in texts {
            let mut s: Vec<String> = tokenize::terms(t);
            s.sort();
            s.dedup();
            for w in s {
                *df.entry(w).or_insert(0) += 1;
            }
        }
        Map::from_iter(df).unwrap()
    }

    #[test]
    fn block_layout_matches_segment_without_tf() {
        let texts = texts();
        let cls = classes(&texts);
        // every term rare: the same results as a segment index without tf
        let mut lw = LexWriter::new(&cls, u64::MAX, 1.0);
        lw.add_batch(&texts[..400]);
        lw.add_batch(&texts[400..]);
        let lp = lw.finish().unwrap();
        assert_eq!(lp.classes.0, 0);
        let blk = Bm25::from_parts(lp.fst.clone(), Blob::Resident(lp.post.clone()), lp.norms.clone(), lp.info.clone()).unwrap();
        let dir = std::env::temp_dir().join(format!("hnz_lex_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut b = Builder::new();
        b.add_batch(&texts);
        b.write_opts(&dir, false, 1.0).unwrap();
        let seg = Bm25::open(&dir).unwrap();
        for w in ["rust", "async", "uniqueterm", "filler"] {
            let (a, da) = blk.lookup(w).unwrap();
            let (b, db) = seg.lookup(w).unwrap();
            assert_eq!(da, db, "{w}");
            let idf = 1.5;
            assert_eq!(blk.top_k(&[(a, idf)], 0, 1000, 20).unwrap(), seg.top_k(&[(b, idf)], 0, 1000, 20).unwrap(), "{w}");
            assert_eq!(blk.score_docs(&[(a, idf)], &[0, 3, 7, 999]).unwrap(), seg.score_docs(&[(b, idf)], &[0, 3, 7, 999]).unwrap());
        }
        let (ta, _) = blk.lookup("async").unwrap();
        let (tr, _) = blk.lookup("rust").unwrap();
        let (sa, _) = seg.lookup("async").unwrap();
        let (sr, _) = seg.lookup("rust").unwrap();
        assert_eq!(blk.top_k(&[(tr, 1.0), (ta, 2.0)], 100, 400, 5).unwrap(), seg.top_k(&[(sr, 1.0), (sa, 2.0)], 100, 400, 5).unwrap());
        std::fs::remove_dir_all(&dir).ok();
        // with a threshold: frequent terms keep their df but have no postings
        let mut lw = LexWriter::new(&cls, 200, 0.2);
        lw.add_batch(&texts);
        let lp = lw.finish().unwrap();
        let blk = Bm25::from_parts(lp.fst, Blob::Resident(lp.post), lp.norms, lp.info).unwrap();
        let (t, df) = blk.lookup("filler").unwrap();
        assert_eq!((t, df), (TermPost::Frequent, 1000));
        let (t, df) = blk.lookup("rust").unwrap();
        assert_eq!((t, df), (TermPost::Frequent, 429));
        let (t, df) = blk.lookup("async").unwrap();
        assert!(matches!(t, TermPost::List { .. }) && df == 143);
        assert_eq!(blk.lookup("uniqueterm").unwrap(), (TermPost::Single { doc: 999, tf: 1 }, 1));
        assert!(blk.top_k(&[(TermPost::Frequent, 1.0)], 0, 1000, 5).unwrap().is_empty());
        assert_eq!(lp.classes, (3, 1, 2)); // filler, words, rust ("here" is a stopword) | uniqueterm | async, tokio
        // a term missing from the global table is an error
        let few = classes(&texts[..10]);
        let mut lw = LexWriter::new(&few, 200, 0.2);
        lw.add_batch(&texts);
        assert!(lw.finish().is_err());
    }
}
