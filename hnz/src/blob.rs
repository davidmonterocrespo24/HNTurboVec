//! Bytes of an archive file, or of one part of a block file: resident in RAM (private copy,
//! charged to the budget) or on disk (positional reads on demand, which go through the OS file
//! cache and not through the process's memory). Replaces mmap: touched mapped pages enter the
//! process working set with nobody controlling them (P2, doc/06-libreria.md).
use anyhow::{Context, Result};
use std::borrow::Cow;
use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Several read handles on one file. Windows serializes I/O on one synchronous handle, so a
/// parallel scan over one big file needs one handle per worker thread.
pub struct FilePool {
    files: Vec<File>,
    next: AtomicUsize,
    pub len: u64,
}

impl FilePool {
    /// `n` handles (at least 1). With `deny_write`, other processes cannot open the file for
    /// writing while it is open (renaming or deleting it stays possible).
    pub fn open(path: &Path, n: usize, deny_write: bool) -> Result<Arc<Self>> {
        let mut files = Vec::with_capacity(n.max(1));
        for _ in 0..n.max(1) {
            let mut oo = std::fs::OpenOptions::new();
            oo.read(true);
            #[cfg(windows)]
            if deny_write {
                use std::os::windows::fs::OpenOptionsExt;
                oo.share_mode(1 | 4); // FILE_SHARE_READ | FILE_SHARE_DELETE
            }
            #[cfg(not(windows))]
            let _ = deny_write;
            files.push(oo.open(path).with_context(|| format!("{}", path.display()))?);
        }
        let len = files[0].metadata()?.len();
        Ok(Arc::new(Self { files, next: AtomicUsize::new(0), len }))
    }

    /// Handles for parallel readers: one per worker thread plus a few for other threads.
    pub fn default_handles() -> usize {
        rayon::current_num_threads() + 4
    }

    fn pick(&self) -> &File {
        let n = self.files.len();
        let i = match rayon::current_thread_index() {
            Some(t) => t % n,
            None => self.next.fetch_add(1, Ordering::Relaxed) % n,
        };
        &self.files[i]
    }

    /// Full positional read of `buf.len()` bytes at `off`.
    pub fn read_at(&self, buf: &mut [u8], off: u64) -> Result<()> {
        read_at(self.pick(), buf, off)
    }
}

pub enum Blob {
    Resident(Vec<u8>),
    /// `len` bytes starting at `base` of a file read through `pool`
    Disk { pool: Arc<FilePool>, base: u64, len: usize },
}

impl Blob {
    /// Opens `path`; with `resident`, reads it whole into RAM.
    pub fn open(path: &Path, resident: bool) -> Result<Self> {
        if resident {
            return Ok(Blob::Resident(std::fs::read(path).with_context(|| format!("{}", path.display()))?));
        }
        let pool = FilePool::open(path, 1, false)?;
        let len = pool.len as usize;
        Ok(Blob::Disk { pool, base: 0, len })
    }

    /// A slice [base, base+len) of a file (a part of a block), read on demand.
    pub fn part(pool: Arc<FilePool>, base: u64, len: usize) -> Self {
        Blob::Disk { pool, base, len }
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

    /// Bytes [off, off+len) (clipped at the end of the file or part): borrowed if resident; read
    /// if not.
    pub fn get(&self, off: usize, len: usize) -> Result<Cow<'_, [u8]>> {
        let end = (off + len).min(self.len());
        let off = off.min(end);
        match self {
            Blob::Resident(v) => Ok(Cow::Borrowed(&v[off..end])),
            Blob::Disk { pool, base, .. } => {
                let mut buf = vec![0u8; end - off];
                pool.read_at(&mut buf, base + off as u64)?;
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
