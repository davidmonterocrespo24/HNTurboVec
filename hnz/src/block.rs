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
        write!(f, "corrupt block: {}", self.0)
    }
}
impl std::error::Error for CorruptBlockError {}

/// The block is intact but this binary cannot read it (in Python: `hnz.IncompatibleBlockError`).
#[derive(Debug)]
pub struct IncompatibleBlockError(pub String);
impl std::fmt::Display for IncompatibleBlockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "incompatible block: {}", self.0)
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
    /// written by a newer hnz (an optional part with it is ignored, a required one refused)
    #[serde(other)]
    Unknown,
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
    /// read whole when needed (tools, the embedder's load), never by queries
    Cold,
    /// written by a newer hnz
    #[serde(other)]
    Unknown,
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
            Codec::Unknown => bail!("part {}: cannot write with an unknown codec", spec.name),
        };
        let raw_len = raw.len() as u64;
        self.add_stream(spec, Some(raw_len), |s| Ok(s.write_all(&stored)?))
    }

    /// Adds a part written by `f` (for parts too big to hold in memory). `raw_len`: decoded size of
    /// a zstd part (None = the stored size).
    pub fn add_stream(&mut self, spec: PartSpec, raw_len: Option<u64>, f: impl FnOnce(&mut PartSink) -> Result<()>) -> Result<&PartEntry> {
        if self.parts.iter().any(|p| p.name == spec.name) {
            bail!("duplicate part: {}", spec.name);
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

    /// Adds a raw part whose bytes are those of `path` (written by a builder that could not hold
    /// them in memory): the same part as `add` of those bytes.
    pub fn add_file(&mut self, spec: PartSpec, path: &Path) -> Result<&PartEntry> {
        if spec.codec != Codec::Raw {
            bail!("part {}: only an uncompressed part can be copied from a file", spec.name);
        }
        let want = std::fs::metadata(path).with_context(|| format!("{}", path.display()))?.len();
        let e = self.add_stream(spec, None, |s| {
            let mut f = std::io::BufReader::with_capacity(8 << 20, File::open(path)?);
            std::io::copy(&mut f, s)?;
            Ok(())
        })?;
        if e.len != want {
            bail!("part {}: {} bytes copied from {} ({} bytes)", e.name, e.len, path.display(), want);
        }
        Ok(e)
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
    toc_off: u64,
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
        if path.is_dir() && Self::locate(path).is_none() {
            return Err(incompatible(format!("{} is not an hnz block ({FILE_NAME} is missing)", path.display())));
        }
        let file = Self::locate(path).unwrap_or_else(|| path.to_path_buf());
        let pool = FilePool::open(&file, FilePool::default_handles(), true)?;
        let flen = pool.len;
        let mut head = [0u8; 8];
        if flen >= 8 {
            pool.read_at(&mut head, 0)?;
        }
        if head != MAGIC {
            // a damaged block still has its trailer; anything else is not a block at all
            let mut t = [0u8; TRAILER_LEN];
            if flen >= TRAILER_LEN as u64 {
                pool.read_at(&mut t, flen - TRAILER_LEN as u64)?;
                if t[0..8] == TRAILER_MAGIC && XxHash3_64::oneshot(&t[..56]) == le_u64(&t, 56) {
                    return Err(corrupt("superblock (magic)"));
                }
            }
            return Err(incompatible(format!("{} is not an hnz block", file.display())));
        }
        if flen < (SB_LEN + TRAILER_LEN) as u64 {
            return Err(corrupt(format!("{} is truncated ({flen} bytes)", file.display())));
        }
        let mut sb = vec![0u8; SB_LEN];
        pool.read_at(&mut sb, 0)?;
        if XxHash3_64::oneshot(&sb[..SB_LEN - 8]) != le_u64(&sb, SB_LEN - 8) {
            return Err(corrupt("superblock (hash)"));
        }
        let major = le_u16(&sb, 8);
        if major != FORMAT_MAJOR {
            return Err(incompatible(format!("format version {major} (this binary reads {FORMAT_MAJOR})")));
        }
        let incompat = le_u64(&sb, 16);
        if incompat & !INCOMPAT_KNOWN != 0 {
            return Err(incompatible(format!("unknown incompatible features {incompat:#x}")));
        }
        let file_len = le_u64(&sb, 32);
        if file_len != flen {
            return Err(corrupt(format!("size {flen} bytes, the superblock says {file_len} (file truncated or extended)")));
        }
        let (toc_off, toc_len, toc_raw_len, toc_codec) = (le_u64(&sb, 40), le_u64(&sb, 48), le_u64(&sb, 56), le_u32(&sb, 64));
        let toc_hash = le_u128(&sb, 72);
        let mut t = [0u8; TRAILER_LEN];
        pool.read_at(&mut t, flen - TRAILER_LEN as u64)?;
        if t[0..8] != TRAILER_MAGIC || XxHash3_64::oneshot(&t[..56]) != le_u64(&t, 56) {
            return Err(corrupt("file trailer"));
        }
        if le_u64(&t, 8) != toc_off || le_u64(&t, 16) != toc_len || le_u128(&t, 24) != toc_hash || le_u64(&t, 40) != file_len {
            return Err(corrupt("the trailer does not match the superblock"));
        }
        if toc_off % ALIGN != 0 || toc_off < SB_LEN as u64 || toc_off.checked_add(toc_len).map(|e| e + TRAILER_LEN as u64) != Some(flen) {
            return Err(corrupt("index (TOC) position out of range"));
        }
        let mut stored = vec![0u8; toc_len as usize];
        pool.read_at(&mut stored, toc_off)?;
        if XxHash3_128::oneshot(&stored) != toc_hash {
            return Err(corrupt("index (TOC): hash"));
        }
        let json = match toc_codec {
            0 => stored,
            1 => zstd::bulk::decompress(&stored, toc_raw_len as usize).map_err(|e| corrupt(format!("index (TOC): {e}")))?,
            c => return Err(incompatible(format!("unknown index codec {c}"))),
        };
        // the TOC's hash matched: bytes this binary cannot parse come from another writer
        let toc: Toc = serde_json::from_slice(&json).map_err(|e| incompatible(format!("index (TOC) unreadable by this binary: {e}")))?;
        if toc.format.major != FORMAT_MAJOR {
            return Err(incompatible(format!("format version {} in the index", toc.format.major)));
        }
        if !toc.format.incompat.is_empty() {
            return Err(incompatible(format!("incompatible features: {:?}", toc.format.incompat)));
        }
        // parts: aligned, inside [4096, toc_off), not overlapping, unique names, well-formed hashes
        let mut by_name = BTreeMap::new();
        let mut spans: Vec<(u64, u64, &str)> = Vec::new();
        let mut names = std::collections::HashSet::new();
        for (i, p) in toc.parts.iter().enumerate() {
            if !names.insert(p.name.as_str()) {
                return Err(corrupt(format!("duplicate part in the index: {}", p.name)));
            }
            // a codec or access class from a newer writer: such a required part cannot be read; an
            // optional one is never read (but still hashed by `full`)
            if p.codec == Codec::Unknown || p.access == Access::Unknown {
                if p.required {
                    return Err(incompatible(format!("part {}: unknown codec or access type", p.name)));
                }
            } else {
                by_name.insert(p.name.clone(), i);
            }
            if p.off % ALIGN != 0 || p.off < SB_LEN as u64 || p.off.checked_add(p.len).is_none_or(|e| e > toc_off) {
                return Err(corrupt(format!("part {} out of range", p.name)));
            }
            if parse_hash(&p.hash).is_none() {
                return Err(corrupt(format!("part {}: malformed hash", p.name)));
            }
            if p.codec == Codec::Raw && p.raw_len != p.len {
                return Err(corrupt(format!("part {}: inconsistent sizes", p.name)));
            }
            spans.push((p.off, p.off + p.len, &p.name));
        }
        spans.sort_unstable();
        if let Some(w) = spans.windows(2).find(|w| w[1].0 < w[0].1) {
            return Err(corrupt(format!("parts {} and {} overlap", w[0].2, w[1].2)));
        }
        // external files: plain names next to the container (never a path elsewhere), one per role
        let mut roles = std::collections::HashSet::new();
        for e in &toc.external {
            let mut c = Path::new(&e.name).components();
            let plain = matches!((c.next(), c.next()), (Some(std::path::Component::Normal(n)), None) if n.to_str() == Some(e.name.as_str()))
                && !e.name.contains([':', '/', '\\']);
            if !plain || !roles.insert(e.role.as_str()) || parse_hash(&e.hash).is_none() {
                return Err(incompatible(format!("invalid external file in the index: {:?} ({})", e.name, e.role)));
            }
        }
        let writer = String::from_utf8_lossy(&sb[104..168]).trim_end_matches('\0').to_string();
        Ok(Arc::new(Self { path: file, pool, toc, by_name, snapshot: toc_hash, toc_off, n_docs: le_u64(&sb, 88), math_version: le_u32(&sb, 96), writer }))
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

    /// Offset and length of the TOC in the container.
    pub fn toc_range(&self) -> (u64, u64) {
        (self.toc_off, self.pool.len - TRAILER_LEN as u64 - self.toc_off)
    }

    pub fn snapshot_hex(&self) -> String {
        format!("{:032x}", self.snapshot)
    }

    pub fn part(&self, name: &str) -> Option<&PartEntry> {
        self.by_name.get(name).map(|&i| &self.toc.parts[i])
    }

    /// A part this binary needs: missing -> `IncompatibleBlockError`.
    pub fn require(&self, name: &str) -> Result<&PartEntry> {
        self.part(name).ok_or_else(|| incompatible(format!("missing part {name}")))
    }

    /// The stored bytes of a part, read whole and checked against its hash.
    pub fn read_stored(&self, p: &PartEntry) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; p.len as usize];
        self.pool.read_at(&mut buf, p.off)?;
        if Some(XxHash3_128::oneshot(&buf)) != parse_hash(&p.hash) {
            return Err(corrupt(format!("part {}: hash", p.name)));
        }
        Ok(buf)
    }

    /// A part's decoded bytes (zstd parts decompressed), checked against its hash.
    pub fn read_part(&self, name: &str) -> Result<Vec<u8>> {
        let p = self.require(name)?;
        let stored = self.read_stored(p)?;
        match p.codec {
            Codec::Raw => Ok(stored),
            Codec::Unknown => Err(incompatible(format!("part {}: unknown codec", p.name))),
            Codec::Zstd => {
                let raw = zstd::bulk::decompress(&stored, p.raw_len as usize).map_err(|e| corrupt(format!("part {}: {e}", p.name)))?;
                if raw.len() as u64 != p.raw_len {
                    return Err(corrupt(format!("part {}: decompressed size", p.name)));
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
        Ok(Blob::part(self.pool.clone(), p.off, p.len as usize, parse_hash(&p.hash)))
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
            return Err(corrupt(format!("part {}: hash", p.name)));
        }
        Ok(())
    }

    /// `full` check: every part's hash (in parallel), the zero padding between parts, and the
    /// external files next to the container.
    pub fn verify_full(&self) -> Result<()> {
        use rayon::prelude::*;
        self.toc.parts.par_iter().try_for_each(|p| self.verify_part(p))?;
        let mut spans: Vec<(u64, u64)> = self.toc.parts.iter().map(|p| (p.off, p.off + p.len)).collect();
        spans.sort_unstable();
        let toc_off = self.toc_off;
        let mut gaps = Vec::new();
        let mut at = SB_LEN as u64;
        for (s, e) in spans {
            gaps.push((at, s));
            at = e;
        }
        gaps.push((at, toc_off));
        for (s, e) in gaps {
            if e > s {
                let mut buf = vec![0u8; (e - s) as usize];
                self.pool.read_at(&mut buf, s)?;
                if buf.iter().any(|&b| b != 0) {
                    return Err(corrupt(format!("nonzero padding at {s}..{e}")));
                }
            }
        }
        self.check_externals()
    }

    /// Checks the external files next to the container (size and hash).
    pub fn check_externals(&self) -> Result<()> {
        for e in &self.toc.external {
            let path = self.dir().join(&e.name);
            let (len, hash) = hash_file(&path).map_err(|_| corrupt(format!("missing external file {}", e.name)))?;
            if len != e.len || Some(hash) != parse_hash(&e.hash) {
                return Err(corrupt(format!("external file {}: size or hash", e.name)));
            }
        }
        Ok(())
    }

    /// The query embedder's files: its IR next to the container (checked against the hashes in
    /// the TOC) and its tokenizer, stored inside the block.
    pub fn embedder_files(&self) -> Result<(PathBuf, PathBuf, Vec<u8>)> {
        self.check_externals()?;
        let by_role = |r: &str| {
            self.toc.external.iter().find(|e| e.role == r).map(|e| self.dir().join(&e.name)).ok_or_else(|| incompatible(format!("missing external file of role {r}")))
        };
        Ok((by_role("query_ir")?, by_role("query_ir_weights")?, self.read_part("model/embed.tok")?))
    }

    /// Refuses a block with a required part this binary does not know, or in a version it does
    /// not read (`IncompatibleBlockError`); `known` = (name, versions read). Unknown optional parts
    /// are ignored.
    pub fn check_parts(&self, known: &[(&str, &[u32])]) -> Result<()> {
        for p in &self.toc.parts {
            match known.iter().find(|k| k.0 == p.name) {
                Some((_, vers)) if vers.contains(&p.ver) => {}
                Some((_, vers)) => return Err(incompatible(format!("part {} is at version {} (this binary reads {vers:?})", p.name, p.ver))),
                None if p.required => return Err(incompatible(format!("unknown required part {}", p.name))),
                None => {}
            }
        }
        Ok(())
    }
}

/// How much of a block is checked when it is opened.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum VerifyLevel {
    /// superblock, trailer, TOC, part bounds and definition versions (always done)
    Header,
    /// + the always-loaded parts' hashes as they load and the canaries (default)
    Quick,
    /// + every part's hash, the padding and the external files
    Full,
}

/// One versioned definition the bytes depend on, as recorded in the manifest
/// (`manifest[section][key]`), and the versions this binary reads.
pub struct Definition<'a> {
    pub section: &'a str,
    pub key: &'a str,
    pub supported: &'a [u64],
}

impl Block {
    /// Refuses a block whose bytes were made with a definition this binary does not implement
    /// (`IncompatibleBlockError`); also checks the superblock's copy of the LM math version.
    pub fn check_compat(&self, defs: &[Definition]) -> Result<()> {
        let m = self.manifest();
        for d in defs {
            let v = m.get(d.section).and_then(|s| s.get(d.key)).and_then(|v| v.as_u64());
            match v {
                Some(v) if d.supported.contains(&v) => {}
                Some(v) => return Err(incompatible(format!("{}.{} = {v} (this binary reads {:?})", d.section, d.key, d.supported))),
                None => return Err(incompatible(format!("the manifest does not give {}.{}", d.section, d.key))),
            }
        }
        if let Some(v) = m.get("text").and_then(|s| s.get("math_version")).and_then(|v| v.as_u64()) {
            if v != self.math_version as u64 {
                return Err(corrupt(format!("LM math version: superblock {}, manifest {v}", self.math_version)));
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
        // definition versions: supported, unsupported, missing
        let ok = [Definition { section: "x", key: "v", supported: &[1] }];
        assert!(b.check_compat(&[]).is_ok());
        let e = b.check_compat(&ok).unwrap_err();
        assert!(e.downcast_ref::<IncompatibleBlockError>().is_some(), "{e}");
        // a disk blob of a part checks its hash when loaded whole
        let mut blob = b.blob("c/stream", false).unwrap();
        assert_eq!(blob.make_resident().unwrap(), 4000);
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
        check("magic", &|v| v[0] = b'X', "corrupt"); // the trailer is intact: a damaged block
        check("not a block", &|v| {
            v[0] = b'X';
            let n = v.len();
            v[n - 64] = b'X'
        }, "incompatible");
        let pad_at = (part("a/raw").off + part("a/raw").len) as usize + 5;
        check("padding", &|v| v[pad_at] = 7, "ok");
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
