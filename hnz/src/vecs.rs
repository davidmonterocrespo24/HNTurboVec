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
        Ok(Self { n: n_docs, codes, has, nbytes })
    }

    #[inline]
    pub fn has(&self, doc: usize) -> bool {
        self.has[doc / 64] >> (doc % 64) & 1 == 1
    }

    pub fn ram_bytes(&self) -> usize {
        self.codes.resident_bytes() + self.has.capacity() * 8
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
        self.codes.get(doc * self.nbytes, self.nbytes).unwrap_or_else(|e| {
            eprintln!("hnz: no se pudo leer vec.bin: {e:#}");
            Cow::Owned(vec![0; self.nbytes])
        })
    }

    /// Codes of documents [st, en), consecutive (a single read if they are on disk).
    fn codes_of(&self, st: usize, en: usize) -> Cow<'_, [u8]> {
        self.codes.get(st * self.nbytes, (en - st) * self.nbytes).unwrap_or_else(|e| {
            eprintln!("hnz: no se pudo leer vec.bin: {e:#}");
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
            let bytes = self.codes_of(st, en);
            for d in st..en {
                seen += 1;
                if self.has(d) {
                    sample[Self::ham(q, &bytes[(d - st) * self.nbytes..], s1) as usize] += 1;
                    ns += 1;
                }
            }
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
        // 2) una pasada: candidatos con Hamming <= t e histograma exacto de todo el rango
        let pass = |t: u32| -> (Vec<(u32, u32)>, Vec<u64>) {
            let starts: Vec<usize> = (lo..hi).step_by(CH).collect();
            let parts: Vec<(Vec<(u32, u32)>, Vec<u64>)> = starts
                .par_iter()
                .map(|&st| {
                    let en = (st + CH).min(hi);
                    let mut v = Vec::new();
                    let mut hist = vec![0u64; maxd + 1];
                    self.windows(st, en, |a, bytes| {
                        for d in a..a + bytes.len() / self.nbytes {
                            if !self.has(d) {
                                continue;
                            }
                            let h = Self::ham(q, &bytes[(d - a) * self.nbytes..], s1);
                            hist[h as usize] += 1;
                            if h <= t {
                                v.push((h, d as u32));
                            }
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
        all.into_iter().map(|x| x.1).collect()
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
                self.windows(st, en, |a, bytes| {
                    for d in a..a + bytes.len() / self.nbytes {
                        if !self.has(d) {
                            continue;
                        }
                        let ham = Self::ham(q, &bytes[(d - a) * self.nbytes..], s1);
                        if h.len() < cand {
                            h.push((ham, d as u32));
                        } else if ham < h.peek().unwrap().0 {
                            h.pop();
                            h.push((ham, d as u32));
                        }
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

    /// Top-k en [lo, hi): Hamming (256 bits) -> `cand` candidatos -> asimétrico.
    pub fn search(&self, q: &QueryLut, lo: usize, hi: usize, cand: usize, k: usize) -> Vec<(u32, f32)> {
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

    pub fn score_docs(&self, q: &QueryLut, docs: &[u32]) -> Vec<f32> {
        docs.iter()
            .map(|&d| if self.has(d as usize) { q.score(&self.code(d as usize)) } else { f32::NEG_INFINITY })
            .collect()
    }
}
