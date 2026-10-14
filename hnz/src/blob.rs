//! Bytes of an archive file: resident in RAM (private copy, charged to the budget) or on disk
//! (positional reads on demand, which go through the OS file cache and not through the process's
//! memory). Replaces mmap: touched mapped pages enter the process working set with nobody
//! controlling them (P2, doc/06-libreria.md).
use anyhow::{Context, Result};
use std::borrow::Cow;
use std::fs::File;
use std::path::Path;

pub enum Blob {
    Resident(Vec<u8>),
    Disk { file: File, len: usize },
}

impl Blob {
    /// Opens `path`; with `resident`, reads it whole into RAM.
    pub fn open(path: &Path, resident: bool) -> Result<Self> {
        if resident {
            return Ok(Blob::Resident(std::fs::read(path).with_context(|| format!("{}", path.display()))?));
        }
        let file = File::open(path).with_context(|| format!("{}", path.display()))?;
        let len = file.metadata()?.len() as usize;
        Ok(Blob::Disk { file, len })
    }

    pub fn len(&self) -> usize {
        match self {
            Blob::Resident(v) => v.len(),
            Blob::Disk { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes it takes in the process's RAM.
    pub fn resident_bytes(&self) -> usize {
        match self {
            Blob::Resident(v) => v.capacity(),
            Blob::Disk { .. } => 0,
        }
    }

    pub fn is_resident(&self) -> bool {
        matches!(self, Blob::Resident(_))
    }

    /// Bytes [off, off+len) (clipped at the end of the file): borrowed if resident; read if not.
    pub fn get(&self, off: usize, len: usize) -> Result<Cow<'_, [u8]>> {
        let end = (off + len).min(self.len());
        let off = off.min(end);
        match self {
            Blob::Resident(v) => Ok(Cow::Borrowed(&v[off..end])),
            Blob::Disk { file, .. } => {
                let mut buf = vec![0u8; end - off];
                read_at(file, &mut buf, off as u64)?;
                Ok(Cow::Owned(buf))
            }
        }
    }

    /// The whole content (read entirely if it is on disk).
    pub fn all(&self) -> Result<Cow<'_, [u8]>> {
        self.get(0, self.len())
    }

    /// Loads into RAM a file that was on disk (no-op if already resident). Returns the bytes loaded.
    pub fn make_resident(&mut self) -> Result<usize> {
        if self.is_resident() {
            return Ok(0);
        }
        let v = self.all()?.into_owned();
        let n = v.len();
        *self = Blob::Resident(v);
        Ok(n)
    }
}

/// Full positional read (thread-safe: each call says its own offset).
pub fn read_at(file: &File, mut buf: &mut [u8], mut off: u64) -> Result<()> {
    while !buf.is_empty() {
        #[cfg(windows)]
        let n = std::os::windows::fs::FileExt::seek_read(file, buf, off)?;
        #[cfg(unix)]
        let n = std::os::unix::fs::FileExt::read_at(file, buf, off)?;
        if n == 0 {
            anyhow::bail!("fin de fichero inesperado al leer {} bytes en {off}", buf.len());
        }
        buf = &mut buf[n..];
        off += n as u64;
    }
    Ok(())
}

/// Which segment parts are loaded into RAM; the rest is read from disk on demand. The flags and
/// the BM25 norms and dictionary (FST) are always in RAM (small, and used by every query).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Residency {
    pub vectors: bool,
    pub postings: bool,
    pub meta: bool,
    pub text: bool,
}

impl Residency {
    pub const ALL: Self = Self { vectors: true, postings: true, meta: true, text: true };
    pub const NONE: Self = Self { vectors: false, postings: false, meta: false, text: false };
    /// Engine without a budget (CLI): vectors in RAM (scanned whole by every query); the rest, with
    /// point reads from disk (sped up by the OS file cache, outside the process).
    pub const DEFAULT: Self = Self { vectors: true, postings: false, meta: false, text: false };
}
