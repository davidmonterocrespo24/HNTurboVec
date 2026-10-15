//! Índice vectorial binario: 1 bit por dimensión (signo) de EmbeddingGemma 2 (MRL: los primeros
//! `bits` dims). Búsqueda en 2 etapas: Hamming sobre los primeros 256 bits (popcount) -> top-C ->
//! puntuación ASIMÉTRICA (consulta en float, documento en ±1) con tabla de consulta por byte.

use anyhow::{bail, Context, Result};
use half::f16;
use memmap2::Mmap;
use rayon::prelude::*;
use crate::blob::Blob;
use std::borrow::Cow;
use std::collections::BinaryHeap;
use std::fs::File;
use std::path::Path;

/// Abre un .npy (v1/v2) por mmap (los de un año entero ocupan GB). Devuelve
/// (shape, descr, mmap, inicio de los datos).
pub fn read_npy(path: &Path) -> Result<(Vec<usize>, String, Mmap, usize)> {
    let f = File::open(path).with_context(|| format!("{}", path.display()))?;
    let b = unsafe { Mmap::map(&f)? };
    if &b[..6] != b"\x93NUMPY" {
        bail!("no es .npy: {}", path.display());
    }
    let (hlen, hstart) = if b[6] == 1 {
        (u16::from_le_bytes([b[8], b[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12)
    };
    let header = std::str::from_utf8(&b[hstart..hstart + hlen])?;
    let descr = header.split("'descr':").nth(1).context("descr")?.split('\'').nth(1).context("descr")?.to_string();
    let shape_s = header.split("'shape':").nth(1).context("shape")?;
    let shape_s = &shape_s[shape_s.find('(').unwrap() + 1..shape_s.find(')').unwrap()];
    let shape: Vec<usize> = shape_s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
    let start = hstart + hlen;
    Ok((shape, descr, b, start))
}

/// Construye vec.bin (n_docs x bits/8) y vec.has (bitmap) alineando por id de HN.
pub fn build(dir: &Path, doc_ids: &[u32], emb_npy: &Path, ids_npy: &Path, bits: usize) -> Result<usize> {
    let (shape, descr, emap, eoff) = read_npy(emb_npy)?;
    let data = &emap[eoff..];
    if descr != "<f2" || shape.len() != 2 || shape[1] < bits {
        bail!("embeddings deben ser float16 Nx>={bits} (es {descr} {shape:?})");
    }
    let (ishape, idescr, imap, ioff) = read_npy(ids_npy)?;
    let idata = &imap[ioff..];
    if idescr != "<u4" || ishape[0] != shape[0] {
        bail!("ids deben ser uint32 de longitud {}", shape[0]);
    }
    let dim = shape[1];
    let ids: Vec<u32> = idata.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let nbytes = bits / 8;
    let mut codes = vec![0u8; doc_ids.len() * nbytes];
    let mut has = vec![0u64; doc_ids.len().div_ceil(64)];
    let mut matched = 0;
    let mut j = 0;
    for (doc, &id) in doc_ids.iter().enumerate() {
        while j < ids.len() && ids[j] < id {
            j += 1;
        }
        if j < ids.len() && ids[j] == id {
            let row = &data[j * dim * 2..(j * dim + bits) * 2];
            let code = &mut codes[doc * nbytes..(doc + 1) * nbytes];
            for d in 0..bits {
                let v = f16::from_le_bytes([row[2 * d], row[2 * d + 1]]);
                if v.to_f32() > 0.0 {
                    code[d / 8] |= 1 << (d % 8);
                }
            }
            has[doc / 64] |= 1 << (doc % 64);
            matched += 1;
        }
    }
    std::fs::write(dir.join("vec.bin"), &codes)?;
    let hb: Vec<u8> = has.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write(dir.join("vec.has"), hb)?;
    Ok(matched)
}

/// Documents per read when the codes are on disk (memory per thread: this x code bytes).
pub const DISK_READ_DOCS: usize = 8192;

pub struct VecIndex {
    codes: Blob,
    has: Vec<u64>,
    pub nbytes: usize,
    pub n: usize,
    /// sparse layout (blocks): codes only for the documents with a vector, in document order;
    /// `rank[w]` = vectors before word `w` of `has` (derived at open)
    rank: Option<Vec<u32>>,
    /// blocks: ties at the Hamming cut are broken by a hash of (doc, query) instead of the doc
    /// number (in one global, time-ordered segment the doc number would always favour old docs)
    pub tie_seeded: bool,
    /// read errors of the codes so far (a search fails if any happens while it runs) and the last
    read_errors: std::sync::atomic::AtomicU64,
    last_error: std::sync::Mutex<String>,
}

/// 32-bit integer hash (murmur3 finalizer).
#[inline]
fn mix32(x: u32) -> u32 {
    let mut z = x.wrapping_mul(0x9E37_79B1);
    z ^= z >> 16;
    z = z.wrapping_mul(0x85EB_CA6B);
    z ^= z >> 13;
    z = z.wrapping_mul(0xC2B2_AE35);
    z ^ (z >> 16)
}

/// Tabla de consulta: para cada byte del código, suma de q[d] de los bits a 1.
pub struct QueryLut {
    lut: Vec<[f32; 256]>,
    sum_q: f32,
    dims: usize,
    sign1: Vec<u64>,
}

impl QueryLut {
    pub fn new(q: &[f32], nbytes: usize) -> Self {
        let dims = nbytes * 8;
        let norm = q[..dims].iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        let qn: Vec<f32> = q[..dims].iter().map(|x| x / norm).collect();
        let mut lut = vec![[0f32; 256]; nbytes];
        for (p, t) in lut.iter_mut().enumerate() {
            for (b, v) in t.iter_mut().enumerate() {
                let mut s = 0.0;
                for j in 0..8 {
                    if b >> j & 1 == 1 {
                        s += qn[p * 8 + j];
                    }
                }
                *v = s;
            }
        }
        // signo de los primeros 256 bits (etapa 1)
        let s1 = nbytes.min(32);
        let mut bytes = vec![0u8; 32];
        for d in 0..s1 * 8 {
            if qn[d] > 0.0 {
                bytes[d / 8] |= 1 << (d % 8);
            }
        }
        let sign1 = bytes.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        Self { lut, sum_q: qn.iter().sum(), dims, sign1 }
    }

    /// Coseno estimado entre la consulta (float) y el documento (±1).
    #[inline]
    pub fn score(&self, code: &[u8]) -> f32 {
        let mut s = 0.0f32;
        for (p, &b) in code.iter().enumerate() {
            s += self.lut[p][b as usize];
        }
        (2.0 * s - self.sum_q) / (self.dims as f32).sqrt()
    }
}

impl VecIndex {
    pub fn open(dir: &Path, bits: usize, n_docs: usize) -> Result<Self> {
        Self::open_with(dir, bits, n_docs, true)
    }

    /// `resident`: codes in RAM (the usual: every query scans them whole); otherwise they are read
    /// from disk in windows.
    pub fn open_with(dir: &Path, bits: usize, n_docs: usize, resident: bool) -> Result<Self> {
        let codes = Blob::open(&dir.join("vec.bin"), resident).context("vec.bin")?;
        let hb = std::fs::read(dir.join("vec.has"))?;
        let has: Vec<u64> = hb.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        let nbytes = bits / 8;
        // un vec.bin de otro ancho (p.ej. un `index` interrumpido antes de escribir el manifiesto)
        // puede ser múltiplo del tamaño del código por casualidad: se exige el tamaño exacto
        if bits == 0 || bits % 64 != 0 || codes.len() != n_docs * nbytes || hb.len() % 8 != 0 || has.len() * 64 < n_docs {
            bail!(
                "vec.bin ({} B) / vec.has ({} B) no cuadran con {n_docs} docs de {bits} bits del manifiesto: rehaga `hnz index`",
                codes.len(),
                hb.len()
            );
        }
        Ok(Self { n: n_docs, codes, has, nbytes, rank: None, tie_seeded: false, read_errors: Default::default(), last_error: Default::default() })
    }

    /// From the parts of a block: the bitmap of documents with a vector and their codes (only
    /// those, in document order; in RAM or on disk).
    pub fn from_parts(has_bytes: &[u8], codes: Blob, bits: usize, n_docs: usize) -> Result<Self> {
        let nbytes = bits / 8;
        if bits == 0 || bits % 64 != 0 || has_bytes.len() % 8 != 0 || has_bytes.len() / 8 < n_docs.div_ceil(64) {
            return Err(crate::block::corrupt("vec/has: tamaño"));
        }
        let has: Vec<u64> = has_bytes.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        let mut rank = Vec::with_capacity(has.len());
        let mut acc = 0u32;
        for w in &has {
            rank.push(acc);
            acc += w.count_ones();
        }
        if codes.len() != acc as usize * nbytes {
            return Err(crate::block::corrupt(format!("vec/codes: {} bytes para {acc} vectores de {nbytes} B", codes.len())));
        }
        Ok(Self { n: n_docs, codes, has, nbytes, rank: Some(rank), tie_seeded: true, read_errors: Default::default(), last_error: Default::default() })
    }

    /// Code slot of a document (dense: the document; sparse: vectors before it).
    #[inline]
    fn slot(&self, doc: usize) -> usize {
        match &self.rank {
            None => doc,
            Some(r) => r[doc / 64] as usize + (self.has[doc / 64] & ((1u64 << (doc % 64)) - 1)).count_ones() as usize,
        }
    }

    /// Calls `f(doc, code)` for the documents of [st, en) that have a vector, reading their codes in
    /// windows (one borrowed slice when resident; reads of at most `DISK_READ_DOCS` codes on disk).
    fn scan(&self, st: usize, en: usize, mut f: impl FnMut(usize, &[u8])) {
        let nb = self.nbytes;
        if self.rank.is_none() {
            self.windows(st, en, |a, bytes| {
                for d in a..a + bytes.len() / nb {
                    if self.has(d) {
                        f(d, &bytes[(d - a) * nb..(d - a + 1) * nb]);
                    }
                }
            });
            return;
        }
        let (s0, s1) = (self.slot(st), if en >= self.n { self.slot_end() } else { self.slot(en) });
        let step = if self.codes.is_resident() { (s1 - s0).max(1) } else { DISK_READ_DOCS };
        let mut d = st;
        let mut s = s0;
        while s < s1 {
            let e = (s + step).min(s1);
            let bytes = self.codes.get(s * nb, (e - s) * nb).unwrap_or_else(|err| {
                self.read_failed(&err);
                Cow::Owned(vec![0; (e - s) * nb])
            });
            for i in 0..e - s {
                while !self.has(d) {
                    d += 1;
                }
                f(d, &bytes[i * nb..(i + 1) * nb]);
                d += 1;
            }
            s = e;
        }
    }

    /// Number of code slots (sparse: documents with a vector).
    fn slot_end(&self) -> usize {
        match &self.rank {
            None => self.n,
            Some(r) => r.last().map_or(0, |&x| x as usize + self.has.last().map_or(0, |w| w.count_ones() as usize)),
        }
    }

    #[inline]
    pub fn has(&self, doc: usize) -> bool {
        self.has[doc / 64] >> (doc % 64) & 1 == 1
    }

    pub fn ram_bytes(&self) -> usize {
        self.codes.resident_bytes() + self.has.capacity() * 8 + self.rank.as_ref().map_or(0, |r| r.capacity() * 4)
    }

    pub fn blob(&self) -> &Blob {
        &self.codes
    }

    pub fn blob_mut(&mut self) -> &mut Blob {
        &mut self.codes
    }

    /// Walks [st, en) in windows: `f(first doc, consecutive codes)`. Resident: one borrowed window;
    /// on disk, reads of at most `DISK_READ_DOCS` documents.
    fn windows(&self, st: usize, en: usize, mut f: impl FnMut(usize, &[u8])) {
        let step = if self.codes.is_resident() { (en - st).max(1) } else { DISK_READ_DOCS };
        let mut a = st;
        while a < en {
            let b = (a + step).min(en);
            f(a, &self.codes_of(a, b));
            a = b;
        }
    }

    /// Code of the document (borrowed if the codes are resident; read from disk if not).
    #[inline]
    pub fn code(&self, doc: usize) -> Cow<'_, [u8]> {
        self.try_code(doc).unwrap_or_else(|e| {
            self.read_failed(&e);
            Cow::Owned(vec![0; self.nbytes])
        })
    }

    /// A failed read of codes: counted, so the search that was running reports it.
    fn read_failed(&self, e: &anyhow::Error) {
        *self.last_error.lock().unwrap_or_else(|p| p.into_inner()) = format!("{e:#}");
        self.read_errors.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }

    /// Runs `f`; an error if a read of codes failed meanwhile (their place was read as zeros).
    fn checked<T>(&self, f: impl FnOnce() -> T) -> Result<T> {
        let before = self.read_errors.load(std::sync::atomic::Ordering::SeqCst);
        let r = f();
        if self.read_errors.load(std::sync::atomic::Ordering::SeqCst) != before {
            bail!("no se pudieron leer los códigos de los vectores: {}", self.last_error.lock().unwrap_or_else(|p| p.into_inner()));
        }
        Ok(r)
    }

    /// Code of the document, or the read error.
    pub fn try_code(&self, doc: usize) -> Result<Cow<'_, [u8]>> {
        self.codes.get(self.slot(doc) * self.nbytes, self.nbytes)
    }

    /// Codes of documents [st, en), consecutive (a single read if they are on disk).
    fn codes_of(&self, st: usize, en: usize) -> Cow<'_, [u8]> {
        self.codes.get(st * self.nbytes, (en - st) * self.nbytes).unwrap_or_else(|e| {
            self.read_failed(&e);
            Cow::Owned(vec![0; (en - st) * self.nbytes])
        })
    }

    /// Distancia de Hamming de la etapa 1 (los primeros `s1` × 64 bits) entre un documento y la consulta.
    #[inline]
    fn ham(q: &QueryLut, c: &[u8], s1: usize) -> u32 {
        let mut h = 0u32;
        for w in 0..s1 {
            let x = u64::from_le_bytes(c[w * 8..w * 8 + 8].try_into().unwrap());
            h += (x ^ q.sign1[w]).count_ones();
        }
        h
    }

    /// Los `cand` documentos con menor (Hamming, doc) de [lo, hi), en ese orden. Sin montículos (E30d:
    /// con 49 M documentos y 20.000 candidatos, ordenar lo que guardaban los montículos de cada trozo
    /// costaba ~0,6 s): una muestra estima la distancia de corte; una sola pasada recoge lo que queda
    /// por debajo y cuenta el histograma exacto; si la estimación se quedó corta, una 2ª pasada con el
    /// corte exacto. Mismo conjunto y orden que la versión con montículos (`hamming_select_heap`).
    fn hamming_select(&self, q: &QueryLut, lo: usize, hi: usize, cand: usize) -> Vec<u32> {
        const CH: usize = 1 << 16;
        let s1 = self.nbytes.min(32) / 8;
        let maxd = s1 * 64;
        // 1) estimated cut from a sample of up to 256 windows of 256 documents spread over the range
        //    (x2 margin); windows rather than single documents, so that on disk it takes few reads
        const W: usize = 256;
        let nwin = ((hi - lo) / W).clamp(1, 256);
        let gap = (hi - lo) / nwin;
        let mut sample = vec![0u64; maxd + 1];
        let (mut ns, mut seen) = (0u64, 0u64);
        for w in 0..nwin {
            let st = lo + w * gap;
            let en = (st + W).min(hi);
            seen += (en - st) as u64;
            self.scan(st, en, |_, c| {
                sample[Self::ham(q, c, s1) as usize] += 1;
                ns += 1;
            });
        }
        let frac = cand as f64 / (ns as f64 * (hi - lo) as f64 / seen.max(1) as f64).max(1.0);
        let want = (frac * ns as f64 * 2.0).ceil() as u64 + 16;
        let cut = |hist: &[u64], need: u64| -> u32 {
            let mut acc = 0u64;
            for (d, &c) in hist.iter().enumerate() {
                acc += c;
                if acc >= need {
                    return d as u32;
                }
            }
            maxd as u32
        };
        let t_est = cut(&sample, want);
        // ties at the cut: by doc number (segments) or by a hash of (doc, query) (blocks)
        let seed = q.sign1.iter().fold(0x2545_F491u32, |a, &w| mix32(a ^ w as u32 ^ (w >> 32) as u32));
        let tie = |d: usize| if self.tie_seeded { mix32(d as u32 ^ seed) } else { d as u32 };
        // 2) una pasada: candidatos con Hamming <= t e histograma exacto de todo el rango
        let pass = |t: u32| -> (Vec<(u32, u32, u32)>, Vec<u64>) {
            let starts: Vec<usize> = (lo..hi).step_by(CH).collect();
            let parts: Vec<(Vec<(u32, u32, u32)>, Vec<u64>)> = starts
                .par_iter()
                .map(|&st| {
                    let en = (st + CH).min(hi);
                    let mut v = Vec::new();
                    let mut hist = vec![0u64; maxd + 1];
                    self.scan(st, en, |d, c| {
                        let h = Self::ham(q, c, s1);
                        hist[h as usize] += 1;
                        if h <= t {
                            v.push((h, tie(d), d as u32));
                        }
                    });
                    (v, hist)
                })
                .collect();
            let mut all = Vec::new();
            let mut hist = vec![0u64; maxd + 1];
            for (v, h) in parts {
                all.extend(v);
                for (a, b) in hist.iter_mut().zip(h) {
                    *a += b;
                }
            }
            (all, hist)
        };
        let (mut all, hist) = pass(t_est);
        let t_exact = cut(&hist, cand as u64);
        if t_exact > t_est {
            all = pass(t_exact).0; // la muestra subestimó el corte (raro)
        }
        if all.len() > cand {
            all.select_nth_unstable(cand - 1);
            all.truncate(cand);
        }
        all.sort_unstable();
        all.into_iter().map(|x| x.2).collect()
    }

    /// Versión anterior de la etapa 1 (montículos por trozo), solo para comparar en pruebas y en el
    /// banco de pruebas `examples/stage1.rs`.
    #[doc(hidden)]
    pub fn hamming_select_heap(&self, q: &QueryLut, lo: usize, hi: usize, cand: usize) -> Vec<u32> {
        const CH: usize = 1 << 16;
        let s1 = self.nbytes.min(32) / 8;
        let starts: Vec<usize> = (lo..hi).step_by(CH).collect();
        let heaps: Vec<BinaryHeap<(u32, u32)>> = starts
            .par_iter()
            .map(|&st| {
                let en = (st + CH).min(hi);
                let mut h: BinaryHeap<(u32, u32)> = BinaryHeap::with_capacity(cand.min(en - st) + 1);
                self.scan(st, en, |d, c| {
                    let ham = Self::ham(q, c, s1);
                    if h.len() < cand {
                        h.push((ham, d as u32));
                    } else if ham < h.peek().unwrap().0 {
                        h.pop();
                        h.push((ham, d as u32));
                    }
                });
                h
            })
            .collect();
        let mut all: Vec<(u32, u32)> = heaps.into_iter().flatten().collect();
        all.sort_unstable();
        all.truncate(cand);
        all.into_iter().map(|x| x.1).collect()
    }

    /// Para el banco de pruebas: la selección nueva, expuesta.
    #[doc(hidden)]
    pub fn hamming_select_pub(&self, q: &QueryLut, lo: usize, hi: usize, cand: usize) -> Vec<u32> {
        self.hamming_select(q, lo, hi, cand)
    }

    /// Top-k en [lo, hi): Hamming (256 bits) -> `cand` candidatos -> asimétrico. A failed read of
    /// codes is an error.
    pub fn search(&self, q: &QueryLut, lo: usize, hi: usize, cand: usize, k: usize) -> Result<Vec<(u32, f32)>> {
        self.checked(|| self.search_unchecked(q, lo, hi, cand, k))
    }

    fn search_unchecked(&self, q: &QueryLut, lo: usize, hi: usize, cand: usize, k: usize) -> Vec<(u32, f32)> {
        let hi = hi.min(self.n);
        if lo >= hi || cand == 0 || k == 0 {
            return Vec::new();
        }
        let cands: Vec<u32> = if hi - lo <= cand {
            (lo as u32..hi as u32).filter(|&d| self.has(d as usize)).collect()
        } else {
            self.hamming_select(q, lo, hi, cand)
        };
        let mut scored: Vec<(u32, f32)> = cands.par_iter().map(|&d| (d, q.score(&self.code(d as usize)))).collect();
        scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(k);
        scored
    }

    pub fn score_docs(&self, q: &QueryLut, docs: &[u32]) -> Result<Vec<f32>> {
        docs.iter()
            .map(|&d| if self.has(d as usize) { Ok(q.score(&self.try_code(d as usize)?)) } else { Ok(f32::NEG_INFINITY) })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(x: &mut u64) -> u64 {
        *x ^= *x << 13;
        *x ^= *x >> 7;
        *x ^= *x << 17;
        *x
    }

    #[test]
    fn sparse_equals_dense() {
        let (n, nb) = (20_000usize, 32usize);
        let mut x = 99u64;
        let mut dense = vec![0u8; n * nb];
        let mut has = vec![0u64; n.div_ceil(64)];
        let mut sparse = Vec::new();
        for d in 0..n {
            if rnd(&mut x) % 10 != 0 {
                has[d / 64] |= 1 << (d % 64);
                for b in 0..nb {
                    dense[d * nb + b] = rnd(&mut x) as u8;
                }
                sparse.extend_from_slice(&dense[d * nb..(d + 1) * nb]);
            }
        }
        let hb: Vec<u8> = has.iter().flat_map(|w| w.to_le_bytes()).collect();
        let dv = VecIndex { codes: Blob::Resident(dense), has: has.clone(), nbytes: nb, n, rank: None, tie_seeded: false, read_errors: Default::default(), last_error: Default::default() };
        for resident in [true, false] {
            let codes = if resident {
                Blob::Resident(sparse.clone())
            } else {
                let p = std::env::temp_dir().join(format!("hnz_vec_sparse_{}.bin", std::process::id()));
                std::fs::write(&p, &sparse).unwrap();
                Blob::open(&p, false).unwrap()
            };
            let mut sv = VecIndex::from_parts(&hb, codes, 256, n).unwrap();
            sv.tie_seeded = false;
            let q: Vec<f32> = (0..768).map(|_| (rnd(&mut x) % 1000) as f32 / 500.0 - 1.0).collect();
            let lut = QueryLut::new(&q, nb);
            for d in [0, 1, 63, 64, 999, n - 1] {
                assert_eq!(dv.has(d), sv.has(d));
                if dv.has(d) {
                    assert_eq!(dv.code(d), sv.code(d), "doc {d}");
                }
            }
            for (lo, hi) in [(0, n), (100, 7_000), (12_345, 19_999)] {
                for cand in [50, 2_000] {
                    assert_eq!(dv.search(&lut, lo, hi, cand, 30).unwrap(), sv.search(&lut, lo, hi, cand, 30).unwrap(), "{lo}..{hi} {cand}");
                    assert_eq!(dv.hamming_select_heap(&lut, lo, hi, cand), sv.hamming_select_heap(&lut, lo, hi, cand));
                }
            }
            let docs: Vec<u32> = (0..n as u32).step_by(7).collect();
            assert_eq!(dv.score_docs(&lut, &docs).unwrap(), sv.score_docs(&lut, &docs).unwrap());
            // seeded ties: same candidates up to the tie group, not biased towards low doc numbers
            sv.tie_seeded = true;
            let a = sv.hamming_select_pub(&lut, 0, n, 500);
            assert_eq!(a.len(), 500);
            let mean = a.iter().map(|&d| d as f64).sum::<f64>() / a.len() as f64;
            assert!(mean > n as f64 * 0.35, "mean doc {mean}");
        }
        assert!(VecIndex::from_parts(&hb, Blob::Resident(vec![0; 64]), 256, n).is_err());
    }
}
