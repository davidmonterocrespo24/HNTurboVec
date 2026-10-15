//! Metadatos por columnas (id, time, tipo/dead, parent, autor, score, descendants, poll, parts)
//! en bloques de 1024 documentos: varints delta/zigzag + zstd. ~6 B/ítem.

use crate::varint::{put, unzz, zz};
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
    encode_block_v(rows, 1)
}

/// Row encoding `v`: 1 = ids increasing within the block (segments, id order); 2 = ids as signed
/// deltas (blocks, which keep documents in time order: 0.63 % of HN ids are out of time order).
fn encode_block_v(rows: &[MetaRow], v: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows.len() * 12);
    let mut prev = 0u32;
    for (k, r) in rows.iter().enumerate() {
        let d = match (k, v) {
            (0, _) => r.id as u64,
            (_, 1) => (r.id - prev) as u64,
            _ => zz(r.id as i64 - prev as i64),
        };
        put(&mut out, d);
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

#[cfg(test)]
fn decode_block(buf: &[u8], n: usize) -> Vec<MetaRow> {
    try_decode_block(buf, n, 1).expect("meta block")
}

/// Checked decoding of a block of `n` rows in encoding `v` (None on malformed bytes).
fn try_decode_block(buf: &[u8], n: usize, v: u8) -> Option<Vec<MetaRow>> {
    let get = |b: &[u8], p: &mut usize| crate::varint::try_get(b, p);
    let mut rows = vec![MetaRow::default(); n];
    let mut p = 0;
    let mut prev = 0u32;
    for (k, r) in rows.iter_mut().enumerate() {
        let x = get(buf, &mut p)?;
        r.id = match (k, v) {
            (0, _) => x as u32,
            (_, 1) => prev.checked_add(x as u32)?,
            _ => u32::try_from(prev as i64 + unzz(x)).ok()?,
        };
        prev = r.id;
    }
    let mut pt = 0i64;
    for (k, r) in rows.iter_mut().enumerate() {
        let x = get(buf, &mut p)?;
        r.time = if k == 0 { x as i64 } else { pt + unzz(x) };
        pt = r.time;
    }
    for r in rows.iter_mut() {
        let b = *buf.get(p)?;
        p += 1;
        r.typ = b & 7;
        r.dead = (b >> 3) & 1 == 1;
        r.deleted = (b >> 4) & 1 == 1;
    }
    for r in rows.iter_mut() {
        let x = get(buf, &mut p)?;
        r.parent = if x == 0 { 0 } else { (r.id as i64 - unzz(x)) as u32 };
    }
    for r in rows.iter_mut() {
        r.author = get(buf, &mut p)? as u32;
    }
    for r in rows.iter_mut() {
        r.score = unzz(get(buf, &mut p)?) as i32;
    }
    for r in rows.iter_mut() {
        r.descendants = unzz(get(buf, &mut p)?) as i32;
    }
    for r in rows.iter_mut() {
        let x = get(buf, &mut p)?;
        r.poll = if x == 0 { 0 } else { (r.id as i64 - unzz(x)) as u32 };
    }
    for r in rows.iter_mut() {
        let c = get(buf, &mut p)? as usize;
        if c > buf.len() {
            return None;
        }
        let mut parts = Vec::with_capacity(c);
        for _ in 0..c {
            parts.push((r.id as i64 + unzz(get(buf, &mut p)?)) as u32);
        }
        r.parts = parts;
    }
    Some(rows)
}

/// Writes the metadata of a block (P3): rows in document order (time, then id), 1,024 per zstd-19
/// frame with content checksum, an index (i64 first time, u32 "main" id, u64 offset per block,
/// then the u64 end), the flag column (with bit 5 = the id is out of time order) and the
/// out-of-order ids (u32 id, u32 doc, sorted by id) used to find them by id.
pub struct MetaBlockWriter {
    rows: Vec<MetaRow>,
    late: Vec<bool>,
    pub blocks: Vec<u8>,
    pub index: Vec<u8>,
    pub flags: Vec<u8>,
    exceptions: Vec<(u32, u32)>,
    main_id: u32,
    n: u32,
}

impl Default for MetaBlockWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl MetaBlockWriter {
    pub fn new() -> Self {
        Self { rows: Vec::new(), late: Vec::new(), blocks: Vec::new(), index: Vec::new(), flags: Vec::new(), exceptions: Vec::new(), main_id: 0, n: 0 }
    }

    /// Adds the next document; `late`: its id is out of time order (found by id through the
    /// exceptions, not through the block index).
    pub fn push(&mut self, r: MetaRow, late: bool) -> Result<()> {
        self.flags.push(flag_byte(&r) | (late as u8) << 5);
        if late {
            self.exceptions.push((r.id, self.n));
        }
        self.n += 1;
        self.rows.push(r);
        self.late.push(late);
        if self.rows.len() == META_BLOCK {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if self.rows.is_empty() {
            return Ok(());
        }
        if let Some(i) = self.late.iter().position(|&l| !l) {
            self.main_id = self.rows[i].id;
        }
        let mut c = zstd::bulk::Compressor::new(19)?;
        c.include_checksum(true)?;
        c.include_contentsize(true)?;
        let frame = c.compress(&encode_block_v(&self.rows, 2))?;
        self.index.extend_from_slice(&self.rows[0].time.to_le_bytes());
        self.index.extend_from_slice(&self.main_id.to_le_bytes());
        self.index.extend_from_slice(&(self.blocks.len() as u64).to_le_bytes());
        self.blocks.extend_from_slice(&frame);
        self.rows.clear();
        self.late.clear();
        Ok(())
    }

    /// (blocks, index, flags, exceptions) bytes.
    pub fn finish(mut self) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>)> {
        self.flush()?;
        self.index.extend_from_slice(&(self.blocks.len() as u64).to_le_bytes());
        self.exceptions.sort_unstable();
        let ex: Vec<u8> = self.exceptions.iter().flat_map(|&(id, d)| id.to_le_bytes().into_iter().chain(d.to_le_bytes())).collect();
        Ok((self.blocks, self.index, self.flags, ex))
    }
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
    let parts: Vec<Vec<u8>> = (0..meta.n_blocks()).into_par_iter().map(|b| Ok(meta.try_block(b)?.iter().map(flag_byte).collect())).collect::<Result<_>>()?;
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

    /// Name of author `i` ("" if there is no such author).
    pub fn get(&self, i: usize) -> &str {
        match (self.off.get(i), self.off.get(i + 1)) {
            (Some(&a), Some(&b)) => self.buf.get(a as usize..b as usize - 1).unwrap_or(""),
            _ => "",
        }
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
    /// per block: id of its first row (segments) or of its first in-order row (blocks)
    pub first_id: Vec<u32>,
    pub first_time: Vec<i64>,
    off: Vec<u64>,
    pub n_docs: usize,
    pub authors: Authors,
    /// row encoding (1 segments, 2 blocks)
    enc: u8,
    /// blocks: (id, doc) of the documents whose id is out of time order, sorted by id
    late: Vec<(u32, u32)>,
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
            first_time.push(u32::from_le_bytes(e[4..8].try_into().unwrap()) as i64);
            off.push(u64::from_le_bytes(e[8..16].try_into().unwrap()));
        }
        off.push(u64::from_le_bytes(idx[nb * 16..nb * 16 + 8].try_into().unwrap()));
        let a = zstd::stream::decode_all(&std::fs::read(dir.join("authors.zst"))?[..])?;
        let authors = Authors::from_text(String::from_utf8(a)?);
        Ok(Self { bin, first_id, first_time, off, n_docs, authors, enc: 1, late: Vec::new() })
    }

    /// From the parts of a block (see `MetaBlockWriter`): the blocks (in RAM or on disk), the
    /// index, the author table and the out-of-order ids.
    pub fn from_parts(bin: Blob, idx: &[u8], authors: Authors, exceptions: &[u8], n_docs: usize) -> Result<Self> {
        let bad = |m: &str| crate::block::corrupt(format!("meta/index: {m}"));
        if idx.len() < 8 || (idx.len() - 8) % 20 != 0 {
            return Err(bad("tamaño"));
        }
        let nb = (idx.len() - 8) / 20;
        if nb != n_docs.div_ceil(META_BLOCK) {
            return Err(bad("número de bloques"));
        }
        let (mut first_id, mut first_time, mut off) = (Vec::with_capacity(nb), Vec::with_capacity(nb), Vec::with_capacity(nb + 1));
        for b in 0..nb {
            let e = &idx[b * 20..b * 20 + 20];
            first_time.push(i64::from_le_bytes(e[0..8].try_into().unwrap()));
            first_id.push(u32::from_le_bytes(e[8..12].try_into().unwrap()));
            off.push(u64::from_le_bytes(e[12..20].try_into().unwrap()));
        }
        off.push(u64::from_le_bytes(idx[nb * 20..].try_into().unwrap()));
        if off.windows(2).any(|w| w[1] <= w[0]) || *off.last().unwrap() as usize != bin.len()
            || first_time.windows(2).any(|w| w[1] < w[0]) || first_id.windows(2).any(|w| w[1] < w[0])
        {
            return Err(bad("desplazamientos, fechas o ids fuera de orden"));
        }
        if exceptions.len() % 8 != 0 {
            return Err(crate::block::corrupt("meta/late: tamaño"));
        }
        let late: Vec<(u32, u32)> = exceptions
            .chunks_exact(8)
            .map(|c| (u32::from_le_bytes(c[0..4].try_into().unwrap()), u32::from_le_bytes(c[4..8].try_into().unwrap())))
            .collect();
        if late.windows(2).any(|w| w[1].0 <= w[0].0) || late.iter().any(|x| x.1 as usize >= n_docs) {
            return Err(crate::block::corrupt("meta/late: orden o rango"));
        }
        Ok(Self { bin, first_id, first_time, off, n_docs, authors, enc: 2, late })
    }

    /// Checked decoding of block `b` (a damaged frame or row is a `CorruptBlockError`).
    pub fn try_block(&self, b: usize) -> Result<Vec<MetaRow>> {
        if b >= self.n_blocks() {
            return Err(crate::block::corrupt(format!("meta block {b} fuera de rango")));
        }
        let data = self.bin.get(self.off[b] as usize, (self.off[b + 1] - self.off[b]) as usize)?;
        let n = (self.n_docs - b * META_BLOCK).min(META_BLOCK);
        let raw = unzstd(&data, MAX_BLOCK_RAW).map_err(|e| crate::block::corrupt(format!("meta block {b}: {e}")))?;
        try_decode_block(&raw, n, self.enc).ok_or_else(|| crate::block::corrupt(format!("meta block {b}: filas")))
    }

    /// Row of document `doc`.
    pub fn try_row(&self, doc: usize) -> Result<MetaRow> {
        let mut rows = self.try_block(doc / META_BLOCK)?;
        let k = doc % META_BLOCK;
        if k >= rows.len() {
            return Err(crate::block::corrupt(format!("doc {doc} fuera de rango")));
        }
        Ok(rows.swap_remove(k))
    }

    /// Document number of a HN id (None if it is not in this store).
    pub fn try_doc_of_id(&self, id: u32) -> Result<Option<usize>> {
        if let Ok(i) = self.late.binary_search_by_key(&id, |x| x.0) {
            return Ok(Some(self.late[i].1 as usize));
        }
        if self.first_id.is_empty() || id < self.first_id[0] {
            return Ok(None);
        }
        let mut b = self.first_id.partition_point(|&x| x <= id) - 1;
        // blocks: a block whose rows are all out of time order repeats the main id of the block
        // before it, so an in-order id is in the first block of a run of equal main ids
        while self.enc == 2 && b > 0 && self.first_id[b - 1] == self.first_id[b] {
            b -= 1;
        }
        let rows = self.try_block(b)?;
        Ok(if self.enc == 1 {
            rows.binary_search_by_key(&id, |r| r.id).ok().map(|k| b * META_BLOCK + k)
        } else {
            rows.iter().position(|r| r.id == id).map(|k| b * META_BLOCK + k)
        })
    }

    /// First document with time >= t (exact for blocks, which are in time order; approximate at
    /// the block level for segments, whose ids grow with time).
    pub fn try_doc_at_time(&self, t: i64) -> Result<usize> {
        let b = self.first_time.partition_point(|&x| x < t);
        if b == 0 {
            return Ok(0);
        }
        // block b-1 may hold documents with time >= t
        let rows = self.try_block(b - 1)?;
        let k = rows.iter().position(|r| r.time >= t).unwrap_or(rows.len());
        Ok(((b - 1) * META_BLOCK + k).min(self.n_docs))
    }

    pub fn n_blocks(&self) -> usize {
        self.first_id.len()
    }

    /// Where block `b` is stored: (offset, length) in the blocks file or part.
    pub fn block_range(&self, b: usize) -> (u64, u64) {
        (self.off[b], self.off[b + 1] - self.off[b])
    }

    /// Bytes it takes in RAM (block index, authors, out-of-order ids and, if resident, the blocks).
    pub fn ram_bytes(&self) -> usize {
        self.bin.resident_bytes()
            + self.first_id.capacity() * 4
            + (self.first_time.capacity() + self.off.capacity()) * 8
            + self.authors.ram_bytes()
            + self.late.capacity() * 8
    }

    pub fn blob(&self) -> &Blob {
        &self.bin
    }

    pub fn blob_mut(&mut self) -> &mut Blob {
        &mut self.bin
    }

    // Panicking forms of the above, for the build tools (the engine uses the checked ones).

    pub fn block(&self, b: usize) -> Vec<MetaRow> {
        self.try_block(b).expect("leyendo metadatos")
    }

    pub fn row(&self, doc: usize) -> MetaRow {
        self.try_row(doc).expect("leyendo metadatos")
    }

    pub fn doc_of_id(&self, id: u32) -> Option<usize> {
        self.try_doc_of_id(id).expect("leyendo metadatos")
    }

    pub fn doc_at_time(&self, t: i64) -> usize {
        self.try_doc_at_time(t).expect("leyendo metadatos")
    }
}

/// A metadata block decodes to far less than this (1,024 rows); a damaged frame that claims more
/// is an error instead of a large allocation.
const MAX_BLOCK_RAW: u64 = 16 << 20;

/// Decompresses one zstd frame of at most `cap` bytes, streaming (no allocation beyond the output).
fn unzstd(data: &[u8], cap: u64) -> std::io::Result<Vec<u8>> {
    use std::io::Read;
    let mut out = Vec::new();
    zstd::stream::read::Decoder::new(data)?.take(cap + 1).read_to_end(&mut out)?;
    if out.len() as u64 > cap {
        return Err(std::io::Error::other("bloque descomprimido demasiado grande"));
    }
    Ok(out)
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
    fn block_writer_time_order() {
        // 2,500 docs in time order; every 100th has an old id (out of time order)
        let mut rows = Vec::new();
        for i in 0..2500u32 {
            let late = i % 100 == 50;
            rows.push((
                MetaRow { id: if late { 10 + i } else { 100_000 + i * 3 }, time: 1_600_000_000 + i as i64 * 60, typ: 2, parent: 0, author: i % 7, ..Default::default() },
                late,
            ));
        }
        let mut w = MetaBlockWriter::new();
        for (r, l) in &rows {
            w.push(r.clone(), *l).unwrap();
        }
        let (blocks, index, flags, ex) = w.finish().unwrap();
        assert_eq!(flags.len(), 2500);
        assert_eq!(flags.iter().filter(|&&f| f >> 5 & 1 == 1).count(), 25);
        let m = MetaStore::from_parts(Blob::Resident(blocks), &index, Authors::from_text("a\nb\nc\nd\ne\nf\ng".into()), &ex, 2500).unwrap();
        for (d, (r, _)) in rows.iter().enumerate() {
            assert_eq!(m.row(d), *r);
            assert_eq!(m.doc_of_id(r.id), Some(d), "id {}", r.id);
        }
        assert_eq!(m.doc_of_id(100_001), None);
        assert_eq!(m.doc_at_time(1_600_000_000 + 1500 * 60), 1500);
        // a damaged frame is a typed error, not a panic
        let mut bad = m.bin.all().unwrap().into_owned();
        bad[m.off[1] as usize + 20] ^= 0x55;
        let m2 = MetaStore::from_parts(Blob::Resident(bad), &index, Authors::from_text("a".into()), &ex, 2500).unwrap();
        assert!(m2.try_block(1).unwrap_err().downcast_ref::<crate::block::CorruptBlockError>().is_some());
        assert!(m2.try_block(0).is_ok());
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
