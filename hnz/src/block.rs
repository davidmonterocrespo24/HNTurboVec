//! Compact block container (P3): one read-only file `hn.hnzb` that holds every part of an archive
//! (metadata, text, vectors, word index, models), so the archive is one file to copy and verify.
//!
//! Layout (all integers little-endian):
//! ```text
//! [0, 4096)          superblock: magic, versions, feature bits, TOC location and hash, n_docs,
//!                    writer; XXH3-64 of bytes [0, 4088) in its last 8 bytes
//! [4096, toc_off)    parts, each starting at a multiple of 4096 (zero padding between them)
//! [toc_off, EOF-64)  TOC: zstd-compressed canonical JSON {format, parts, external, manifest}
//! [EOF-64, EOF)      trailer: magic, TOC location and hash, file length; XXH3-64 of [0, 56)
//! ```
//! Every part records the XXH3-128 of its stored bytes; the TOC hash covers all of them and is the
//! snapshot id. XXH3 detects corruption, not deliberate tampering. The same inputs and binary give
//! the same bytes (no timestamps; JSON with a fixed field order and sorted maps).

use crate::blob::{Blob, FilePool};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use twox_hash::{XxHash3_128, XxHash3_64};

pub const MAGIC: [u8; 8] = *b"HNZB\r\n\x1a\n";
pub const TRAILER_MAGIC: [u8; 8] = *b"HNZBEND\0";
pub const FORMAT_MAJOR: u16 = 1;
pub const FORMAT_MINOR: u16 = 0;
pub const ALIGN: u64 = 4096;
const SB_LEN: usize = 4096;
const TRAILER_LEN: usize = 64;
/// incompatible feature bits this reader understands (it refuses any other)
const INCOMPAT_KNOWN: u64 = 0;
/// name of the container inside a block folder
pub const FILE_NAME: &str = "hn.hnzb";

/// The block's bytes are damaged (in Python: `hnz.CorruptBlockError`).
#[derive(Debug)]
pub struct CorruptBlockError(pub String);
impl std::fmt::Display for CorruptBlockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "bloque dañado: {}", self.0)
    }
}
impl std::error::Error for CorruptBlockError {}

/// The block is intact but this binary cannot read it (in Python: `hnz.IncompatibleBlockError`).
#[derive(Debug)]
pub struct IncompatibleBlockError(pub String);
impl std::fmt::Display for IncompatibleBlockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "bloque incompatible: {}", self.0)
    }
}
impl std::error::Error for IncompatibleBlockError {}

pub fn corrupt(m: impl Into<String>) -> anyhow::Error {
    anyhow::Error::new(CorruptBlockError(m.into()))
}

pub fn incompatible(m: impl Into<String>) -> anyhow::Error {
    anyhow::Error::new(IncompatibleBlockError(m.into()))
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Codec {
    Raw,
    Zstd,
}

/// How the engine uses a part: a hint for the RAM planner and for verification at open.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Access {
    /// always in RAM (read and checked at open)
    Load,
    /// scanned whole by queries (vector codes)
    Scan,
    /// read at random offsets (metadata blocks, text, postings)
    Random,
    /// only tools read it
    Cold,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PartEntry {
    pub name: String,
    pub ver: u32,
    pub off: u64,
    pub len: u64,
    pub codec: Codec,
    /// decoded size (= `len` for raw parts)
    pub raw_len: u64,
    /// "xxh3-128:<32 hex digits>" of the stored bytes
    pub hash: String,
    pub access: Access,
    pub required: bool,
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub meta: serde_json::Value,
}

/// A file that lives next to the container (the embedder IR, read by OpenVINO by path).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ExternalEntry {
    pub name: String,
    pub len: u64,
    pub hash: String,
    pub role: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct FormatInfo {
    major: u16,
    minor: u16,
    #[serde(default)]
    incompat: Vec<String>,
    #[serde(default)]
    compat: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Toc {
    format: FormatInfo,
    parts: Vec<PartEntry>,
    #[serde(default)]
    external: Vec<ExternalEntry>,
    manifest: serde_json::Value,
}

fn hex128(h: u128) -> String {
    format!("xxh3-128:{h:032x}")
}

fn le_u16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes(b[o..o + 2].try_into().unwrap())
}
fn le_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn le_u64(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}
fn le_u128(b: &[u8], o: usize) -> u128 {
    u128::from_le_bytes(b[o..o + 16].try_into().unwrap())
}

/// Options of a part being written.
pub struct PartSpec<'a> {
    pub name: &'a str,
    pub ver: u32,
    pub codec: Codec,
    pub access: Access,
    pub required: bool,
    pub meta: serde_json::Value,
}

impl<'a> PartSpec<'a> {
    /// A required part, version 1, without metadata.
    pub fn new(name: &'a str, codec: Codec, access: Access) -> Self {
        Self { name, ver: 1, codec, access, required: true, meta: serde_json::Value::Null }
    }
}

/// Writes the bytes of one part, hashing them as they go.
pub struct PartSink<'a> {
    w: &'a mut BufWriter<File>,
    hasher: XxHash3_128,
    len: u64,
}

impl Write for PartSink<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.w.write_all(buf)?;
        self.hasher.write(buf);
        self.len += buf.len() as u64;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.w.flush()
    }
}

impl PartSink<'_> {
    pub fn written(&self) -> u64 {
        self.len
    }
}

pub struct BlockWriter {
    path: PathBuf,
    tmp: PathBuf,
    file: BufWriter<File>,
    pos: u64,
    parts: Vec<PartEntry>,
    external: Vec<ExternalEntry>,
}

impl BlockWriter {
    /// Starts a block at `path` (written to `<path>.tmp` and renamed by `finish`).
    pub fn create(path: &Path) -> Result<Self> {
        let tmp = PathBuf::from(format!("{}.tmp", path.display()));
        let mut file = BufWriter::with_capacity(1 << 20, File::create(&tmp).with_context(|| format!("{}", tmp.display()))?);
        file.write_all(&[0u8; SB_LEN])?;
        Ok(Self { path: path.to_path_buf(), tmp, file, pos: SB_LEN as u64, parts: Vec::new(), external: Vec::new() })
    }

    fn pad(&mut self) -> Result<()> {
        let rem = self.pos % ALIGN;
        if rem != 0 {
            let n = (ALIGN - rem) as usize;
            self.file.write_all(&vec![0u8; n])?;
            self.pos += n as u64;
        }
        Ok(())
    }

    /// Adds a part from bytes in memory (`Codec::Zstd` compresses them, level 19).
    pub fn add(&mut self, spec: PartSpec, raw: &[u8]) -> Result<&PartEntry> {
        let stored: Cow<[u8]> = match spec.codec {
            Codec::Raw => Cow::Borrowed(raw),
            Codec::Zstd => Cow::Owned(zstd::bulk::compress(raw, 19)?),
        };
        let raw_len = raw.len() as u64;
        self.add_stream(spec, Some(raw_len), |s| Ok(s.write_all(&stored)?))
    }

    /// Adds a part written by `f` (for parts too big to hold in memory). `raw_len`: decoded size of
    /// a zstd part (None = the stored size).
    pub fn add_stream(&mut self, spec: PartSpec, raw_len: Option<u64>, f: impl FnOnce(&mut PartSink) -> Result<()>) -> Result<&PartEntry> {
        if self.parts.iter().any(|p| p.name == spec.name) {
            bail!("parte repetida: {}", spec.name);
        }
        self.pad()?;
        let off = self.pos;
        let mut sink = PartSink { w: &mut self.file, hasher: XxHash3_128::new(), len: 0 };
        f(&mut sink)?;
        let (len, hash) = (sink.len, sink.hasher.finish_128());
        self.pos += len;
        self.parts.push(PartEntry {
            name: spec.name.to_string(),
            ver: spec.ver,
            off,
            len,
            codec: spec.codec,
            raw_len: raw_len.unwrap_or(len),
            hash: hex128(hash),
            access: spec.access,
            required: spec.required,
            meta: spec.meta,
        });
        Ok(self.parts.last().unwrap())
    }

    /// Records a file kept next to the container: its name, size and hash.
    pub fn add_external(&mut self, name: &str, path: &Path, role: &str) -> Result<()> {
        let (len, hash) = hash_file(path)?;
        self.external.push(ExternalEntry { name: name.to_string(), len, hash: hex128(hash), role: role.to_string() });
        Ok(())
    }

    /// Writes the TOC, the trailer and the superblock, syncs, and renames the file into place.
    /// Returns the snapshot id (the TOC hash).
    pub fn finish(mut self, manifest: serde_json::Value, n_docs: u64, math_version: u32, writer: &str) -> Result<u128> {
        let toc = Toc {
            format: FormatInfo { major: FORMAT_MAJOR, minor: FORMAT_MINOR, incompat: Vec::new(), compat: Vec::new() },
            parts: std::mem::take(&mut self.parts),
            external: std::mem::take(&mut self.external),
            manifest,
        };
        let json = serde_json::to_vec(&toc)?;
        let stored = zstd::bulk::compress(&json, 19)?;
        let toc_hash = XxHash3_128::oneshot(&stored);
        self.pad()?;
        let toc_off = self.pos;
        self.file.write_all(&stored)?;
        self.pos += stored.len() as u64;
        let file_len = self.pos + TRAILER_LEN as u64;
        let mut t = [0u8; TRAILER_LEN];
        t[0..8].copy_from_slice(&TRAILER_MAGIC);
        t[8..16].copy_from_slice(&toc_off.to_le_bytes());
        t[16..24].copy_from_slice(&(stored.len() as u64).to_le_bytes());
        t[24..40].copy_from_slice(&toc_hash.to_le_bytes());
        t[40..48].copy_from_slice(&file_len.to_le_bytes());
        let th = XxHash3_64::oneshot(&t[..56]);
        t[56..64].copy_from_slice(&th.to_le_bytes());
        self.file.write_all(&t)?;
        let mut sb = vec![0u8; SB_LEN];
        sb[0..8].copy_from_slice(&MAGIC);
        sb[8..10].copy_from_slice(&FORMAT_MAJOR.to_le_bytes());
        sb[10..12].copy_from_slice(&FORMAT_MINOR.to_le_bytes());
        sb[12..16].copy_from_slice(&(SB_LEN as u32).to_le_bytes());
        sb[32..40].copy_from_slice(&file_len.to_le_bytes());
        sb[40..48].copy_from_slice(&toc_off.to_le_bytes());
        sb[48..56].copy_from_slice(&(stored.len() as u64).to_le_bytes());
        sb[56..64].copy_from_slice(&(json.len() as u64).to_le_bytes());
        sb[64..68].copy_from_slice(&1u32.to_le_bytes()); // TOC codec: zstd
        sb[68..72].copy_from_slice(&(ALIGN as u32).to_le_bytes());
        sb[72..88].copy_from_slice(&toc_hash.to_le_bytes());
        sb[88..96].copy_from_slice(&n_docs.to_le_bytes());
        sb[96..100].copy_from_slice(&math_version.to_le_bytes());
        let w = writer.as_bytes();
        let n = w.len().min(64);
        sb[104..104 + n].copy_from_slice(&w[..n]);
        let sh = XxHash3_64::oneshot(&sb[..SB_LEN - 8]);
        sb[SB_LEN - 8..].copy_from_slice(&sh.to_le_bytes());
        self.file.flush()?;
        let mut f = self.file.into_inner().map_err(|e| e.into_error())?;
        f.seek(SeekFrom::Start(0))?;
        f.write_all(&sb)?;
        f.sync_all()?;
        drop(f);
        std::fs::rename(&self.tmp, &self.path).with_context(|| format!("{}", self.path.display()))?;
        Ok(toc_hash)
    }
}

/// (size, XXH3-128) of a file, read in 4 MiB chunks.
pub fn hash_file(path: &Path) -> Result<(u64, u128)> {
    let mut f = File::open(path).with_context(|| format!("{}", path.display()))?;
    let mut h = XxHash3_128::new();
    let mut buf = vec![0u8; 4 << 20];
    let mut len = 0u64;
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.write(&buf[..n]);
        len += n as u64;
    }
    Ok((len, h.finish_128()))
}

/// An open block.
pub struct Block {
    /// the container file
    pub path: PathBuf,
    pool: Arc<FilePool>,
    toc: Toc,
    by_name: BTreeMap<String, usize>,
    /// TOC hash: identifies this exact content (cache keys, reports)
    pub snapshot: u128,
    pub n_docs: u64,
    pub math_version: u32,
    pub writer: String,
}

impl Block {
    /// The container path of a block given as a `.hnzb` file or as a folder containing `hn.hnzb`.
    pub fn locate(path: &Path) -> Option<PathBuf> {
        if path.is_dir() {
            let f = path.join(FILE_NAME);
            f.is_file().then_some(f)
        } else if path.is_file() && path.extension().is_some_and(|e| e == "hnzb") {
            Some(path.to_path_buf())
        } else {
            None
        }
    }

    /// Opens a block and checks its header: magic, versions, feature bits, superblock, trailer and
    /// TOC hashes, and part bounds. Part hashes are checked when a part is read whole
    /// (`read_part`, resident `blob`) or by `verify_full`.
    pub fn open(path: &Path) -> Result<Arc<Self>> {
        let file = Self::locate(path).unwrap_or_else(|| path.to_path_buf());
        let pool = FilePool::open(&file, FilePool::default_handles(), true)?;
        let flen = pool.len;
        let mut head = [0u8; 8];
        if flen >= 8 {
            pool.read_at(&mut head, 0)?;
        }
        if head != MAGIC {
            return Err(incompatible(format!("{} no es un bloque hnz", file.display())));
        }
        if flen < (SB_LEN + TRAILER_LEN) as u64 {
            return Err(corrupt(format!("{} está truncado ({flen} bytes)", file.display())));
        }
        let mut sb = vec![0u8; SB_LEN];
        pool.read_at(&mut sb, 0)?;
        if XxHash3_64::oneshot(&sb[..SB_LEN - 8]) != le_u64(&sb, SB_LEN - 8) {
            return Err(corrupt("superbloque (hash)"));
        }
        let major = le_u16(&sb, 8);
        if major != FORMAT_MAJOR {
            return Err(incompatible(format!("versión de formato {major} (este binario lee la {FORMAT_MAJOR})")));
        }
        let incompat = le_u64(&sb, 16);
        if incompat & !INCOMPAT_KNOWN != 0 {
            return Err(incompatible(format!("funciones incompatibles desconocidas {incompat:#x}")));
        }
        let file_len = le_u64(&sb, 32);
        if file_len != flen {
            return Err(corrupt(format!("tamaño {flen} bytes, el superbloque dice {file_len} (fichero truncado o ampliado)")));
        }
        let (toc_off, toc_len, toc_raw_len, toc_codec) = (le_u64(&sb, 40), le_u64(&sb, 48), le_u64(&sb, 56), le_u32(&sb, 64));
        let toc_hash = le_u128(&sb, 72);
        let mut t = [0u8; TRAILER_LEN];
        pool.read_at(&mut t, flen - TRAILER_LEN as u64)?;
        if t[0..8] != TRAILER_MAGIC || XxHash3_64::oneshot(&t[..56]) != le_u64(&t, 56) {
            return Err(corrupt("cola del fichero (trailer)"));
        }
        if le_u64(&t, 8) != toc_off || le_u64(&t, 16) != toc_len || le_u128(&t, 24) != toc_hash || le_u64(&t, 40) != file_len {
            return Err(corrupt("la cola no coincide con el superbloque"));
        }
        if toc_off % ALIGN != 0 || toc_off < SB_LEN as u64 || toc_off.checked_add(toc_len).map(|e| e + TRAILER_LEN as u64) != Some(flen) {
            return Err(corrupt("posición del índice (TOC) fuera de rango"));
        }
        let mut stored = vec![0u8; toc_len as usize];
        pool.read_at(&mut stored, toc_off)?;
        if XxHash3_128::oneshot(&stored) != toc_hash {
            return Err(corrupt("índice (TOC): hash"));
        }
        let json = match toc_codec {
            0 => stored,
            1 => zstd::bulk::decompress(&stored, toc_raw_len as usize).map_err(|e| corrupt(format!("índice (TOC): {e}")))?,
            c => return Err(incompatible(format!("códec del índice desconocido {c}"))),
        };
        let toc: Toc = serde_json::from_slice(&json).map_err(|e| corrupt(format!("índice (TOC): {e}")))?;
        if toc.format.major != FORMAT_MAJOR {
            return Err(incompatible(format!("versión de formato {} en el índice", toc.format.major)));
        }
        if !toc.format.incompat.is_empty() {
            return Err(incompatible(format!("funciones incompatibles: {:?}", toc.format.incompat)));
        }
        // parts: aligned, inside [4096, toc_off), not overlapping, unique names, well-formed hashes
        let mut by_name = BTreeMap::new();
        let mut spans: Vec<(u64, u64, &str)> = Vec::new();
        for (i, p) in toc.parts.iter().enumerate() {
            if by_name.insert(p.name.clone(), i).is_some() {
                return Err(corrupt(format!("parte repetida en el índice: {}", p.name)));
            }
            if p.off % ALIGN != 0 || p.off < SB_LEN as u64 || p.off.checked_add(p.len).is_none_or(|e| e > toc_off) {
                return Err(corrupt(format!("parte {} fuera de rango", p.name)));
            }
            if parse_hash(&p.hash).is_none() {
                return Err(corrupt(format!("parte {}: hash mal formado", p.name)));
            }
            if p.codec == Codec::Raw && p.raw_len != p.len {
                return Err(corrupt(format!("parte {}: tamaños incoherentes", p.name)));
            }
            spans.push((p.off, p.off + p.len, &p.name));
        }
        spans.sort_unstable();
        if let Some(w) = spans.windows(2).find(|w| w[1].0 < w[0].1) {
            return Err(corrupt(format!("las partes {} y {} se solapan", w[0].2, w[1].2)));
        }
        let writer = String::from_utf8_lossy(&sb[104..168]).trim_end_matches('\0').to_string();
        Ok(Arc::new(Self { path: file, pool, toc, by_name, snapshot: toc_hash, n_docs: le_u64(&sb, 88), math_version: le_u32(&sb, 96), writer }))
    }

    pub fn manifest(&self) -> &serde_json::Value {
        &self.toc.manifest
    }

    pub fn parts(&self) -> &[PartEntry] {
        &self.toc.parts
    }

    pub fn external(&self) -> &[ExternalEntry] {
        &self.toc.external
    }

    /// The folder that holds the container (and the external files).
    pub fn dir(&self) -> PathBuf {
        self.path.parent().map(|p| p.to_path_buf()).unwrap_or_default()
    }

    pub fn snapshot_hex(&self) -> String {
        format!("{:032x}", self.snapshot)
    }

    pub fn part(&self, name: &str) -> Option<&PartEntry> {
        self.by_name.get(name).map(|&i| &self.toc.parts[i])
    }

    /// A part this binary needs: missing -> `IncompatibleBlockError`.
    pub fn require(&self, name: &str) -> Result<&PartEntry> {
        self.part(name).ok_or_else(|| incompatible(format!("falta la parte {name}")))
    }

    /// The stored bytes of a part, read whole and checked against its hash.
    pub fn read_stored(&self, p: &PartEntry) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; p.len as usize];
        self.pool.read_at(&mut buf, p.off)?;
        if Some(XxHash3_128::oneshot(&buf)) != parse_hash(&p.hash) {
            return Err(corrupt(format!("parte {}: hash", p.name)));
        }
        Ok(buf)
    }

    /// A part's decoded bytes (zstd parts decompressed), checked against its hash.
    pub fn read_part(&self, name: &str) -> Result<Vec<u8>> {
        let p = self.require(name)?;
        let stored = self.read_stored(p)?;
        match p.codec {
            Codec::Raw => Ok(stored),
            Codec::Zstd => {
                let raw = zstd::bulk::decompress(&stored, p.raw_len as usize).map_err(|e| corrupt(format!("parte {}: {e}", p.name)))?;
                if raw.len() as u64 != p.raw_len {
                    return Err(corrupt(format!("parte {}: tamaño descomprimido", p.name)));
                }
                Ok(raw)
            }
        }
    }

    /// A part as a `Blob`: in RAM (read whole and checked) or read on demand from the file (raw
    /// parts only; data parts carry their own checks).
    pub fn blob(&self, name: &str, resident: bool) -> Result<Blob> {
        let p = self.require(name)?;
        if resident || p.codec != Codec::Raw {
            return Ok(Blob::Resident(self.read_part(name)?));
        }
        Ok(Blob::part(self.pool.clone(), p.off, p.len as usize))
    }

    /// Checks one part's hash, streaming (without holding it in memory).
    pub fn verify_part(&self, p: &PartEntry) -> Result<()> {
        let mut h = XxHash3_128::new();
        let mut buf = vec![0u8; (4 << 20).min(p.len as usize)];
        let mut off = 0u64;
        while off < p.len {
            let n = ((p.len - off) as usize).min(buf.len());
            self.pool.read_at(&mut buf[..n], p.off + off)?;
            h.write(&buf[..n]);
            off += n as u64;
        }
        if Some(h.finish_128()) != parse_hash(&p.hash) {
            return Err(corrupt(format!("parte {}: hash", p.name)));
        }
        Ok(())
    }

    /// `full` check: every part's hash (in parallel) and the external files next to the container.
    pub fn verify_full(&self) -> Result<()> {
        use rayon::prelude::*;
        self.toc.parts.par_iter().try_for_each(|p| self.verify_part(p))?;
        for e in &self.toc.external {
            let path = self.dir().join(&e.name);
            let (len, hash) = hash_file(&path).map_err(|_| corrupt(format!("falta el fichero externo {}", e.name)))?;
            if len != e.len || Some(hash) != parse_hash(&e.hash) {
                return Err(corrupt(format!("fichero externo {}: tamaño o hash", e.name)));
            }
        }
        Ok(())
    }
}

fn parse_hash(s: &str) -> Option<u128> {
    u128::from_str_radix(s.strip_prefix("xxh3-128:")?, 16).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("hnz_block_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn sample(path: &Path) -> u128 {
        let mut w = BlockWriter::create(path).unwrap();
        w.add(PartSpec::new("a/raw", Codec::Raw, Access::Load), b"hello block").unwrap();
        let big: Vec<u8> = (0..100_000u32).flat_map(|i| (i % 251).to_le_bytes()).collect();
        w.add(PartSpec::new("b/zstd", Codec::Zstd, Access::Load), &big).unwrap();
        w.add_stream(PartSpec::new("c/stream", Codec::Raw, Access::Random), None, |s| {
            for i in 0..1000u32 {
                s.write_all(&i.to_le_bytes())?;
            }
            Ok(())
        })
        .unwrap();
        w.finish(serde_json::json!({"profile": "test", "n": 3}), 7, 2, "hnz test").unwrap()
    }

    #[test]
    fn roundtrip_and_reproducible() {
        let d = tmpdir("rt");
        let (p1, p2) = (d.join("one.hnzb"), d.join("two.hnzb"));
        let s1 = sample(&p1);
        let s2 = sample(&p2);
        assert_eq!(s1, s2);
        assert_eq!(std::fs::read(&p1).unwrap(), std::fs::read(&p2).unwrap(), "same inputs, same bytes");
        let b = Block::open(&p1).unwrap();
        assert_eq!((b.n_docs, b.math_version, b.writer.as_str(), b.snapshot), (7, 2, "hnz test", s1));
        assert_eq!(b.manifest()["profile"], "test");
        assert!(b.parts().iter().all(|p| p.off % ALIGN == 0));
        assert_eq!(b.read_part("a/raw").unwrap(), b"hello block");
        let big: Vec<u8> = (0..100_000u32).flat_map(|i| (i % 251).to_le_bytes()).collect();
        assert_eq!(b.read_part("b/zstd").unwrap(), big);
        assert!(b.part("b/zstd").unwrap().len < big.len() as u64 / 10);
        let disk = b.blob("c/stream", false).unwrap();
        assert!(!disk.is_resident());
        assert_eq!(&disk.get(4 * 10, 4).unwrap()[..], &10u32.to_le_bytes());
        assert_eq!(disk.get(3990, 100).unwrap().len(), 10, "reads clip at the end of the part");
        assert!(b.require("missing").unwrap_err().downcast_ref::<IncompatibleBlockError>().is_some());
        b.verify_full().unwrap();
        // a folder holding hn.hnzb opens too
        std::fs::rename(&p2, d.join(FILE_NAME)).unwrap();
        assert_eq!(Block::open(&d).unwrap().snapshot, s1);
        let _ = std::fs::remove_dir_all(&d);
    }

    fn kind(e: &anyhow::Error) -> &'static str {
        if e.downcast_ref::<CorruptBlockError>().is_some() {
            "corrupt"
        } else if e.downcast_ref::<IncompatibleBlockError>().is_some() {
            "incompatible"
        } else {
            "other"
        }
    }

    #[test]
    fn corruption_matrix() {
        let d = tmpdir("bad");
        let good = d.join("good.hnzb");
        sample(&good);
        let bytes = std::fs::read(&good).unwrap();
        let b = Block::open(&good).unwrap();
        let part = |n: &str| b.part(n).unwrap().clone();
        let toc_off = le_u64(&bytes, 40) as usize;
        let check = |name: &str, mutate: &dyn Fn(&mut Vec<u8>), at_open: &str| {
            let p = d.join(format!("{name}.hnzb"));
            let mut v = bytes.clone();
            mutate(&mut v);
            std::fs::write(&p, &v).unwrap();
            match Block::open(&p) {
                Err(e) => assert_eq!(kind(&e), at_open, "{name}: {e:#}"),
                Ok(blk) => {
                    assert_eq!(at_open, "ok", "{name} opened");
                    let e = blk.verify_full().unwrap_err();
                    assert_eq!(kind(&e), "corrupt", "{name}: {e:#}");
                }
            }
        };
        check("superblock", &|v| v[200] ^= 1, "corrupt");
        check("toc", &|v| v[toc_off + 3] ^= 1, "corrupt");
        check("trailer", &|v| {
            let n = v.len();
            v[n - 30] ^= 1
        }, "corrupt");
        check("truncated", &|v| {
            let n = v.len();
            v.truncate(n - 1)
        }, "corrupt");
        check("extended", &|v| v.push(0), "corrupt");
        let a = part("c/stream");
        check("part", &|v| v[a.off as usize + 17] ^= 1, "ok");
        let z = part("b/zstd");
        check("zstd part", &|v| v[z.off as usize + 5] ^= 0x40, "ok");
        check("magic", &|v| v[0] = b'X', "incompatible");
        // a newer major version or an unknown incompatible feature, with a valid superblock hash
        let rehash = |v: &mut Vec<u8>| {
            let h = XxHash3_64::oneshot(&v[..SB_LEN - 8]);
            v[SB_LEN - 8..SB_LEN].copy_from_slice(&h.to_le_bytes());
        };
        check("major", &|v| {
            v[8] = 2;
            rehash(v)
        }, "incompatible");
        check("feature", &|v| {
            v[16] = 1;
            rehash(v)
        }, "incompatible");
        // reading a damaged part reports it by name
        let p = d.join("part2.hnzb");
        let mut v = bytes.clone();
        v[part("a/raw").off as usize] ^= 1;
        std::fs::write(&p, &v).unwrap();
        let e = Block::open(&p).unwrap().read_part("a/raw").unwrap_err();
        assert!(kind(&e) == "corrupt" && format!("{e}").contains("a/raw"), "{e}");
        let _ = std::fs::remove_dir_all(&d);
    }
}
