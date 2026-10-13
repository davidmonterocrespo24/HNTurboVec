//! Índice vectorial binario: 1 bit por dimensión (signo) de EmbeddingGemma 2 (MRL: los primeros
//! `bits` dims). Búsqueda en 2 etapas: Hamming sobre los primeros 256 bits (popcount) -> top-C ->
//! puntuación ASIMÉTRICA (consulta en float, documento en ±1) con tabla de consulta por byte.

use anyhow::{bail, Context, Result};
use half::f16;
use memmap2::Mmap;
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::fs::File;
use std::path::Path;

/// Lee un .npy (v1/v2) de float16 o uint32 2D/1D. Devuelve (shape, bytes de datos).
pub fn read_npy(path: &Path) -> Result<(Vec<usize>, String, Vec<u8>)> {
    let b = std::fs::read(path).with_context(|| format!("{}", path.display()))?;
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
    Ok((shape, descr, b[hstart + hlen..].to_vec()))
}

/// Construye vec.bin (n_docs x bits/8) y vec.has (bitmap) alineando por id de HN.
pub fn build(dir: &Path, doc_ids: &[u32], emb_npy: &Path, ids_npy: &Path, bits: usize) -> Result<usize> {
    let (shape, descr, data) = read_npy(emb_npy)?;
    if descr != "<f2" || shape.len() != 2 || shape[1] < bits {
        bail!("embeddings deben ser float16 Nx>={bits} (es {descr} {shape:?})");
    }
    let (ishape, idescr, idata) = read_npy(ids_npy)?;
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

pub struct VecIndex {
    codes: Mmap,
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
    pub fn open(dir: &Path, bits: usize) -> Result<Self> {
        let codes = unsafe { Mmap::map(&File::open(dir.join("vec.bin")).context("vec.bin")?)? };
        let hb = std::fs::read(dir.join("vec.has"))?;
        let has = hb.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        let nbytes = bits / 8;
        Ok(Self { n: codes.len() / nbytes, codes, has, nbytes })
    }

    #[inline]
    pub fn has(&self, doc: usize) -> bool {
        self.has[doc / 64] >> (doc % 64) & 1 == 1
    }

    #[inline]
    pub fn code(&self, doc: usize) -> &[u8] {
        &self.codes[doc * self.nbytes..(doc + 1) * self.nbytes]
    }

    /// Top-k en [lo, hi): Hamming (256 bits) -> `cand` candidatos -> asimétrico.
    pub fn search(&self, q: &QueryLut, lo: usize, hi: usize, cand: usize, k: usize) -> Vec<(u32, f32)> {
        let hi = hi.min(self.n);
        if lo >= hi {
            return Vec::new();
        }
        let cands: Vec<u32> = if hi - lo <= cand {
            (lo as u32..hi as u32).filter(|&d| self.has(d as usize)).collect()
        } else {
            const CH: usize = 1 << 16;
            let s1 = self.nbytes.min(32) / 8; // palabras de 64 bits en la etapa 1
            let starts: Vec<usize> = (lo..hi).step_by(CH).collect();
            let heaps: Vec<BinaryHeap<(u32, u32)>> = starts
                .par_iter()
                .map(|&st| {
                    let en = (st + CH).min(hi);
                    let mut h: BinaryHeap<(u32, u32)> = BinaryHeap::with_capacity(cand + 1);
                    for d in st..en {
                        if !self.has(d) {
                            continue;
                        }
                        let c = self.code(d);
                        let mut ham = 0u32;
                        for w in 0..s1 {
                            let x = u64::from_le_bytes(c[w * 8..w * 8 + 8].try_into().unwrap());
                            ham += (x ^ q.sign1[w]).count_ones();
                        }
                        if h.len() < cand {
                            h.push((ham, d as u32));
                        } else if ham < h.peek().unwrap().0 {
                            h.pop();
                            h.push((ham, d as u32));
                        }
                    }
                    h
                })
                .collect();
            let mut all: Vec<(u32, u32)> = heaps.into_iter().flatten().collect();
            all.sort_unstable();
            all.truncate(cand);
            all.into_iter().map(|x| x.1).collect()
        };
        let mut scored: Vec<(u32, f32)> = cands.par_iter().map(|&d| (d, q.score(self.code(d as usize)))).collect();
        scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(k);
        scored
    }

    pub fn score_docs(&self, q: &QueryLut, docs: &[u32]) -> Vec<f32> {
        docs.iter()
            .map(|&d| if self.has(d as usize) { q.score(self.code(d as usize)) } else { f32::NEG_INFINITY })
            .collect()
    }
}
