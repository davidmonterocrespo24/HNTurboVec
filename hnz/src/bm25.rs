//! Índice BM25 compacto por segmento.
//! - Diccionario de términos: FST (`fst`), valor = offset en postings, o el posting inline si df=1.
//! - Postings: bloques de 128 (docID delta + tf) empaquetados con BitPacker4x (SIMD-BP128),
//!   cabeceras de salto (último doc de cada bloque) delante; cola en varints.
//! - Normas de longitud: 1 byte/doc (escala logarítmica).
//! - Búsqueda top-k con MaxScore y saltos por rango de docIDs (= rango de fechas).

use crate::tokenize;
use crate::varint::{get, put};
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
}

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

#[derive(Clone, Copy, Debug)]
pub enum TermPost {
    Single { doc: u32, tf: u32 },
    List { off: usize },
}

/// Cursor over a postings list. Each block of 128 is read when the cursor reaches it: borrowed if
/// bm25.post is in RAM, or with one disk read (memory does not depend on the list's size).
pub struct Cursor<'a> {
    post: &'a Blob,
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
    fn new(post: &'a Blob, t: TermPost, with_tf: bool) -> Self {
        let mut c = Cursor {
            post,
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
        }
        c
    }

    /// Bytes [o, o+len) of the list (clipped at the end of the file); None if the read fails.
    fn read(&self, o: usize, len: usize) -> Option<Cow<'a, [u8]>> {
        match self.post.get(self.base + o, len) {
            Ok(b) => Some(b),
            Err(e) => {
                eprintln!("hnz: no se pudieron leer postings de bm25.post: {e:#}");
                None
            }
        }
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
        let df = get(&head, &mut p) as usize;
        let nfull = df / BLK;
        let Some(h) = self.read(0, p + nfull * 20) else { return false };
        let mut last = 0u32;
        let mut plen = Vec::with_capacity(nfull);
        for _ in 0..nfull {
            last += get(&h, &mut p) as u32;
            self.lasts.push(last);
            plen.push(get(&h, &mut p) as usize);
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
            let Some(data) = self.read(o, end - o) else {
                self.exhaust();
                return false;
            };
            let (db, tb) = (data[0], data[1]);
            let init = if b == 0 { 0 } else { self.lasts[b - 1] };
            let n = bp.decompress_sorted(init, &data[2..], &mut self.docs, db);
            if self.with_tf {
                bp.decompress(&data[2 + n..], &mut self.tfs, tb);
                for t in self.tfs.iter_mut() {
                    *t += 1;
                }
            } else {
                self.tfs.fill(1);
            }
            self.len = BLK;
        } else {
            let cnt = self.df - self.nfull * BLK;
            let Some(data) = self.read(self.tail_off, cnt * 20) else {
                self.exhaust();
                return false;
            };
            let mut p = 0;
            let mut prev = if self.nfull == 0 { 0 } else { self.lasts[self.nfull - 1] };
            for i in 0..cnt {
                prev += get(&data, &mut p) as u32;
                self.docs[i] = prev;
                self.tfs[i] = if self.with_tf { get(&data, &mut p) as u32 } else { 1 };
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
        if v & 1 == 1 {
            Some((TermPost::Single { doc: (v >> 9) as u32, tf: ((v >> 1) & 0xff) as u32 }, 1))
        } else {
            let off = (v >> 1) as usize;
            let head = self.post.get(off, 10).ok()?;
            let mut p = 0;
            let df = get(&head, &mut p) as u32;
            Some((TermPost::List { off }, df))
        }
    }

    pub fn cursor(&self, t: TermPost) -> Cursor<'_> {
        Cursor::new(&self.post, t, self.info.with_tf)
    }

    #[inline]
    fn score(&self, idf: f32, tf: u32, doc: u32) -> f32 {
        let dl = norm_decode(self.norms[doc as usize]);
        let tf = tf as f32;
        idf * tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * dl / self.info.avgdl))
    }

    /// Top-k en [lo, hi) con MaxScore. `terms`: (postings, idf global).
    pub fn top_k(&self, terms: &[(TermPost, f32)], lo: u32, hi: u32, k: usize) -> Vec<(u32, f32)> {
        if terms.is_empty() || k == 0 {
            return Vec::new();
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
        let mut v: Vec<(u32, f32)> = heap.into_iter().map(|Reverse((s, d))| (d, s.0)).collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v
    }

    /// Puntuación BM25 de una lista de docs (ordenados ascendentemente).
    pub fn score_docs(&self, terms: &[(TermPost, f32)], docs: &[u32]) -> Vec<f32> {
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
        }
        out
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
        let r = ix.top_k(&[(tu, 5.0)], 0, 1000, 3);
        assert_eq!(r[0].0, 999);
        let (ta, _) = ix.lookup("async").unwrap();
        let r = ix.top_k(&[(tp, 1.0), (ta, 2.0)], 100, 400, 5);
        assert!(r.iter().all(|x| x.0 >= 100 && x.0 < 400 && x.0 % 7 == 0));
        let s = ix.score_docs(&[(ta, 2.0)], &[7, 8, 14]);
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
        assert_eq!(dk.top_k(&[(tp, 1.0), (ta, 2.0)], 100, 400, 5), ix.top_k(&[(tp, 1.0), (ta, 2.0)], 100, 400, 5));
        assert_eq!(dk.score_docs(&[(ta, 2.0)], &[7, 8, 14]), s);
        drop(dk);
        drop(ix);
        std::fs::remove_dir_all(&dir).ok();
    }
}
