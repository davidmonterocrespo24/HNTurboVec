//! Almacén de texto de los niveles caliente/templado: bloques de B registros, columnar
//! (títulos | urls | textos transformados), zstd-19 con diccionario entrenado por segmento.

use crate::htmlx::{self, Mode};
use crate::varint::{get, put};
use anyhow::{Context, Result};
use crate::blob::Blob;
use rayon::prelude::*;
use std::fs::File;
use std::io::Write;
use std::path::Path;

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct TextRec {
    pub title: String,
    pub url: String,
    /// HTML original de HN
    pub text: String,
}

/// Bloque crudo: longitudes (varint) de títulos, urls y textos (len<<1 | crudo) + bytes.
fn encode_block(recs: &[(&str, &str, (Mode, String))]) -> Vec<u8> {
    let mut out = Vec::new();
    for r in recs {
        put(&mut out, r.0.len() as u64);
    }
    for r in recs {
        put(&mut out, r.1.len() as u64);
    }
    for r in recs {
        put(&mut out, ((r.2 .1.len() as u64) << 2) | r.2 .0 as u64);
    }
    for r in recs {
        out.extend_from_slice(r.0.as_bytes());
    }
    for r in recs {
        out.extend_from_slice(r.1.as_bytes());
    }
    for r in recs {
        out.extend_from_slice(r.2 .1.as_bytes());
    }
    out
}

fn decode_block(buf: &[u8], n: usize) -> Result<Vec<TextRec>> {
    let mut p = 0;
    let lt: Vec<usize> = (0..n).map(|_| get(buf, &mut p) as usize).collect();
    let lu: Vec<usize> = (0..n).map(|_| get(buf, &mut p) as usize).collect();
    let lx: Vec<u64> = (0..n).map(|_| get(buf, &mut p)).collect();
    let mut recs = vec![TextRec::default(); n];
    for (r, &l) in recs.iter_mut().zip(&lt) {
        r.title = std::str::from_utf8(&buf[p..p + l])?.to_string();
        p += l;
    }
    for (r, &l) in recs.iter_mut().zip(&lu) {
        r.url = std::str::from_utf8(&buf[p..p + l])?.to_string();
        p += l;
    }
    for (r, &l) in recs.iter_mut().zip(&lx) {
        let len = (l >> 2) as usize;
        let s = std::str::from_utf8(&buf[p..p + len])?;
        r.text = htmlx::restore(Mode::from_bits(l & 3), s);
        p += len;
    }
    Ok(recs)
}

pub struct BuildStats {
    /// registros por modo: [moderno, antiguo, crudo]
    pub modes: [usize; 3],
    pub raw_bytes: u64,
}

/// Escribe text.bin, text.idx (offsets u64) y text.dict.
pub fn write(dir: &Path, recs: &[(&str, &str, &str)], block: usize, level: i32, dict_size: usize) -> Result<BuildStats> {
    let trans: Vec<(Mode, String)> = recs.par_iter().map(|r| htmlx::transform(r.2)).collect();
    let mut modes = [0usize; 3];
    for t in &trans {
        modes[t.0 as usize] += 1;
    }
    let raw_blocks: Vec<Vec<u8>> = recs
        .par_chunks(block)
        .zip(trans.par_chunks(block))
        .map(|(rc, tc)| {
            let v: Vec<(&str, &str, (Mode, String))> =
                rc.iter().zip(tc).map(|(r, t)| (r.0, r.1, t.clone())).collect();
            encode_block(&v)
        })
        .collect();
    let raw_bytes: u64 = raw_blocks.iter().map(|b| b.len() as u64).sum();
    // diccionario entrenado con una muestra de bloques
    let step = (raw_blocks.len() / 4000).max(1);
    let samples: Vec<&[u8]> = raw_blocks.iter().step_by(step).map(|b| &b[..]).collect();
    let dict = zstd::dict::from_samples(&samples, dict_size).context("entrenando diccionario zstd")?;
    let comp: Vec<Vec<u8>> = raw_blocks
        .par_chunks(64)
        .flat_map_iter(|chunk| {
            let mut c = zstd::bulk::Compressor::with_dictionary(level, &dict).expect("compressor");
            c.include_contentsize(true).ok();
            c.include_checksum(false).ok();
            chunk.iter().map(move |b| c.compress(b).expect("zstd")).collect::<Vec<_>>()
        })
        .collect();
    let mut bin = File::create(dir.join("text.bin"))?;
    let mut idx = Vec::with_capacity((comp.len() + 1) * 8);
    let mut off = 0u64;
    for c in &comp {
        idx.extend_from_slice(&off.to_le_bytes());
        bin.write_all(c)?;
        off += c.len() as u64;
    }
    idx.extend_from_slice(&off.to_le_bytes());
    std::fs::write(dir.join("text.idx"), idx)?;
    std::fs::write(dir.join("text.dict"), &dict)?;
    Ok(BuildStats { modes, raw_bytes })
}

pub struct TextStore {
    bin: Blob,
    off: Vec<u64>,
    dict: zstd::dict::DecoderDictionary<'static>,
    dict_len: usize,
    pub block: usize,
    pub n_docs: usize,
}

impl TextStore {
    pub fn open(dir: &Path, block: usize, n_docs: usize) -> Result<Self> {
        Self::open_with(dir, block, n_docs, false)
    }

    pub fn open_with(dir: &Path, block: usize, n_docs: usize, resident: bool) -> Result<Self> {
        let bin = Blob::open(&dir.join("text.bin"), resident).context("text.bin")?;
        let idx = std::fs::read(dir.join("text.idx"))?;
        let off = idx.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        let dict_bytes = std::fs::read(dir.join("text.dict"))?;
        let dict = zstd::dict::DecoderDictionary::copy(&dict_bytes);
        Ok(Self { bin, off, dict, dict_len: dict_bytes.len(), block, n_docs })
    }

    pub fn n_blocks(&self) -> usize {
        self.off.len() - 1
    }

    /// Bytes it takes in RAM (with the prepared zstd dictionary, ~3x its size).
    pub fn ram_bytes(&self) -> usize {
        self.bin.resident_bytes() + self.off.capacity() * 8 + self.dict_len * 3
    }

    pub fn blob(&self) -> &Blob {
        &self.bin
    }

    pub fn blob_mut(&mut self) -> &mut Blob {
        &mut self.bin
    }

    pub fn get_block(&self, b: usize) -> Result<Vec<TextRec>> {
        let data = self.bin.get(self.off[b] as usize, (self.off[b + 1] - self.off[b]) as usize)?;
        let data = &data[..];
        let cap = zstd::zstd_safe::get_frame_content_size(data).ok().flatten().unwrap_or(1 << 20) as usize;
        let mut d = zstd::bulk::Decompressor::with_prepared_dictionary(&self.dict)?;
        let raw = d.decompress(data, cap)?;
        let n = (self.n_docs - b * self.block).min(self.block);
        decode_block(&raw, n)
    }

    pub fn get(&self, doc: usize) -> Result<TextRec> {
        Ok(self.get_block(doc / self.block)?.swap_remove(doc % self.block))
    }
}
