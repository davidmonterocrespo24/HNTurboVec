//! Metadatos por columnas (id, time, tipo/dead, parent, autor, score, descendants, poll, parts)
//! en bloques de 1024 documentos: varints delta/zigzag + zstd. ~6 B/ítem.

use crate::varint::{get, put, unzz, zz};
use anyhow::{Context, Result};
use crate::blob::Blob;
use std::fs::File;
use std::io::Write;
use std::path::Path;

pub const META_BLOCK: usize = 1024;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MetaRow {
    pub id: u32,
    pub time: i64,
    pub typ: u8,
    pub dead: bool,
    pub deleted: bool,
    pub parent: u32,
    pub author: u32,
    pub score: i32,
    pub descendants: i32,
    pub poll: u32,
    pub parts: Vec<u32>,
}

fn encode_block(rows: &[MetaRow]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows.len() * 12);
    let mut prev = 0u32;
    for (k, r) in rows.iter().enumerate() {
        put(&mut out, if k == 0 { r.id as u64 } else { (r.id - prev) as u64 });
        prev = r.id;
    }
    let mut pt = 0i64;
    for (k, r) in rows.iter().enumerate() {
        put(&mut out, if k == 0 { r.time as u64 } else { zz(r.time - pt) });
        pt = r.time;
    }
    for r in rows {
        out.push(r.typ | ((r.dead as u8) << 3) | ((r.deleted as u8) << 4));
    }
    for r in rows {
        put(&mut out, if r.parent == 0 { 0 } else { zz(r.id as i64 - r.parent as i64) });
    }
    for r in rows {
        put(&mut out, r.author as u64);
    }
    for r in rows {
        put(&mut out, zz(r.score as i64));
    }
    for r in rows {
        put(&mut out, zz(r.descendants as i64));
    }
    for r in rows {
        put(&mut out, if r.poll == 0 { 0 } else { zz(r.id as i64 - r.poll as i64) });
    }
    for r in rows {
        put(&mut out, r.parts.len() as u64);
        for &p in &r.parts {
            put(&mut out, zz(p as i64 - r.id as i64));
        }
    }
    out
}

fn decode_block(buf: &[u8], n: usize) -> Vec<MetaRow> {
    let mut rows = vec![MetaRow::default(); n];
    let mut p = 0;
    let mut prev = 0u32;
    for (k, r) in rows.iter_mut().enumerate() {
        let v = get(buf, &mut p) as u32;
        r.id = if k == 0 { v } else { prev + v };
        prev = r.id;
    }
    let mut pt = 0i64;
    for (k, r) in rows.iter_mut().enumerate() {
        let v = get(buf, &mut p);
        r.time = if k == 0 { v as i64 } else { pt + unzz(v) };
        pt = r.time;
    }
    for r in rows.iter_mut() {
        let b = buf[p];
        p += 1;
        r.typ = b & 7;
        r.dead = (b >> 3) & 1 == 1;
        r.deleted = (b >> 4) & 1 == 1;
    }
    for r in rows.iter_mut() {
        let v = get(buf, &mut p);
        r.parent = if v == 0 { 0 } else { (r.id as i64 - unzz(v)) as u32 };
    }
    for r in rows.iter_mut() {
        r.author = get(buf, &mut p) as u32;
    }
    for r in rows.iter_mut() {
        r.score = unzz(get(buf, &mut p)) as i32;
    }
    for r in rows.iter_mut() {
        r.descendants = unzz(get(buf, &mut p)) as i32;
    }
    for r in rows.iter_mut() {
        let v = get(buf, &mut p);
        r.poll = if v == 0 { 0 } else { (r.id as i64 - unzz(v)) as u32 };
    }
    for r in rows.iter_mut() {
        let c = get(buf, &mut p) as usize;
        r.parts = (0..c).map(|_| (r.id as i64 + unzz(get(buf, &mut p))) as u32).collect();
    }
    rows
}

/// Escribe meta.bin + meta.idx + authors.zst. `authors` ya ordenados por frecuencia.
pub fn write(dir: &Path, rows: &[MetaRow], authors: &[String]) -> Result<()> {
    use rayon::prelude::*;
    let blocks: Vec<Vec<u8>> = rows
        .par_chunks(META_BLOCK)
        .map(|c| zstd::bulk::compress(&encode_block(c), 19).expect("zstd"))
        .collect();
    let mut bin = File::create(dir.join("meta.bin"))?;
    let mut idx = Vec::with_capacity(blocks.len() * 16 + 8);
    let mut off = 0u64;
    for (b, chunk) in blocks.iter().zip(rows.chunks(META_BLOCK)) {
        idx.extend_from_slice(&chunk[0].id.to_le_bytes());
        idx.extend_from_slice(&(chunk[0].time as u32).to_le_bytes());
        idx.extend_from_slice(&off.to_le_bytes());
        bin.write_all(b)?;
        off += b.len() as u64;
    }
    idx.extend_from_slice(&off.to_le_bytes());
    std::fs::write(dir.join("meta.idx"), idx)?;
    let joined = authors.join("\n");
    std::fs::write(dir.join("authors.zst"), zstd::bulk::compress(joined.as_bytes(), 19)?)?;
    // columna de banderas (1 B/doc: tipo | dead<<3 | deleted<<4) para filtrar sin descomprimir
    // bloques; muy repetitiva -> ~0,1 B/doc con zstd
    let flags: Vec<u8> = rows.iter().map(flag_byte).collect();
    std::fs::write(dir.join("meta.flags.zst"), zstd::bulk::compress(&flags, 19)?)?;
    Ok(())
}

pub fn flag_byte(r: &MetaRow) -> u8 {
    r.typ | ((r.dead as u8) << 3) | ((r.deleted as u8) << 4)
}

/// Columna de banderas de un segmento: del fichero si existe; si no (segmentos antiguos), se
/// calcula descomprimiendo todos los bloques de metadatos.
pub fn load_flags(dir: &Path, meta: &MetaStore) -> Result<Vec<u8>> {
    let p = dir.join("meta.flags.zst");
    if p.exists() {
        let f = zstd::stream::decode_all(&std::fs::read(&p)?[..])?;
        if f.len() == meta.n_docs {
            return Ok(f);
        }
    }
    use rayon::prelude::*;
    let parts: Vec<Vec<u8>> = (0..meta.n_blocks()).into_par_iter().map(|b| meta.block(b).iter().map(flag_byte).collect()).collect();
    Ok(parts.concat())
}

/// Author names of a segment in one buffer plus offsets: two allocations, so its RAM is exact
/// (a `Vec<String>` costs ~2x the counted bytes in slack and allocator overhead).
pub struct Authors {
    buf: String,
    /// start of each name; the last entry is the end of the buffer + 1
    off: Vec<u32>,
}

impl Authors {
    /// From the names separated by '\n' (as in authors.zst).
    pub fn from_text(mut buf: String) -> Self {
        buf.shrink_to_fit();
        let mut off = vec![0u32];
        off.extend(buf.bytes().enumerate().filter(|&(_, b)| b == b'\n').map(|(i, _)| i as u32 + 1));
        off.push(buf.len() as u32 + 1);
        off.shrink_to_fit();
        Self { buf, off }
    }

    pub fn len(&self) -> usize {
        self.off.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, i: usize) -> &str {
        &self.buf[self.off[i] as usize..self.off[i + 1] as usize - 1]
    }

    pub fn position(&self, name: &str) -> Option<usize> {
        (0..self.len()).find(|&i| self.get(i) == name)
    }

    pub fn ram_bytes(&self) -> usize {
        self.buf.capacity() + self.off.capacity() * 4
    }
}

pub struct MetaStore {
    bin: Blob,
    pub first_id: Vec<u32>,
    pub first_time: Vec<u32>,
    off: Vec<u64>,
    pub n_docs: usize,
    pub authors: Authors,
}

impl MetaStore {
    pub fn open(dir: &Path, n_docs: usize) -> Result<Self> {
        Self::open_with(dir, n_docs, false)
    }

    /// `resident`: meta.bin in RAM; otherwise each block is read from disk when requested.
    pub fn open_with(dir: &Path, n_docs: usize, resident: bool) -> Result<Self> {
        let bin = Blob::open(&dir.join("meta.bin"), resident).context("meta.bin")?;
        let idx = std::fs::read(dir.join("meta.idx"))?;
        let nb = (idx.len() - 8) / 16;
        let mut first_id = Vec::with_capacity(nb);
        let mut first_time = Vec::with_capacity(nb);
        let mut off = Vec::with_capacity(nb + 1);
        for b in 0..nb {
            let e = &idx[b * 16..b * 16 + 16];
            first_id.push(u32::from_le_bytes(e[0..4].try_into().unwrap()));
            first_time.push(u32::from_le_bytes(e[4..8].try_into().unwrap()));
            off.push(u64::from_le_bytes(e[8..16].try_into().unwrap()));
        }
        off.push(u64::from_le_bytes(idx[nb * 16..nb * 16 + 8].try_into().unwrap()));
        let a = zstd::stream::decode_all(&std::fs::read(dir.join("authors.zst"))?[..])?;
        let authors = Authors::from_text(String::from_utf8(a)?);
        Ok(Self { bin, first_id, first_time, off, n_docs, authors })
    }

    pub fn n_blocks(&self) -> usize {
        self.first_id.len()
    }

    /// Bytes it takes in RAM (block index, authors and, if resident, meta.bin).
    pub fn ram_bytes(&self) -> usize {
        self.bin.resident_bytes()
            + (self.first_id.capacity() + self.first_time.capacity()) * 4
            + self.off.capacity() * 8
            + self.authors.ram_bytes()
    }

    pub fn blob(&self) -> &Blob {
        &self.bin
    }

    pub fn blob_mut(&mut self) -> &mut Blob {
        &mut self.bin
    }

    pub fn block(&self, b: usize) -> Vec<MetaRow> {
        let data = self.bin.get(self.off[b] as usize, (self.off[b + 1] - self.off[b]) as usize).expect("leyendo meta.bin");
        let raw = zstd::stream::decode_all(&data[..]).expect("meta zstd");
        let n = (self.n_docs - b * META_BLOCK).min(META_BLOCK);
        decode_block(&raw, n)
    }

    pub fn row(&self, doc: usize) -> MetaRow {
        self.block(doc / META_BLOCK).swap_remove(doc % META_BLOCK)
    }

    /// docnum de un id de HN (si está en este segmento).
    pub fn doc_of_id(&self, id: u32) -> Option<usize> {
        if self.first_id.is_empty() || id < self.first_id[0] {
            return None;
        }
        let b = self.first_id.partition_point(|&x| x <= id) - 1;
        let rows = self.block(b);
        rows.binary_search_by_key(&id, |r| r.id).ok().map(|k| b * META_BLOCK + k)
    }

    /// Primer docnum con time >= t (aproximado a nivel de bloque: los ids crecen con el tiempo).
    pub fn doc_at_time(&self, t: i64) -> usize {
        let b = self.first_time.partition_point(|&x| (x as i64) < t);
        if b == 0 {
            return 0;
        }
        // el bloque b-1 puede contener docs con time >= t
        let rows = self.block(b - 1);
        let k = rows.iter().position(|r| r.time >= t).unwrap_or(rows.len());
        ((b - 1) * META_BLOCK + k).min(self.n_docs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn block_roundtrip() {
        let rows: Vec<MetaRow> = (0..100)
            .map(|i| MetaRow {
                id: 1000 + i * 3,
                time: 1_700_000_000 + (i as i64) * 7 - if i == 5 { 100 } else { 0 },
                typ: (i % 5 + 1) as u8,
                dead: i % 7 == 0,
                deleted: i % 11 == 0,
                parent: if i % 2 == 0 { 0 } else { 900 + i },
                author: i * 11,
                score: if i % 3 == 0 { -3 } else { i as i32 },
                descendants: i as i32 * 2,
                poll: if i == 9 { 1000 } else { 0 },
                parts: if i == 3 { vec![1010, 1011] } else { vec![] },
            })
            .collect();
        let b = encode_block(&rows);
        assert_eq!(decode_block(&b, rows.len()), rows);
    }

    #[test]
    fn authors_like_split() {
        for text in ["", "a", "a\nbc", "a\n", "\nx\n\nyz"] {
            let a = Authors::from_text(text.to_string());
            let v: Vec<&str> = text.split('\n').collect();
            assert_eq!(a.len(), v.len(), "{text:?}");
            for (i, s) in v.iter().enumerate() {
                assert_eq!(a.get(i), *s);
            }
            assert_eq!(a.position("bc"), v.iter().position(|s| *s == "bc"));
        }
    }
}
