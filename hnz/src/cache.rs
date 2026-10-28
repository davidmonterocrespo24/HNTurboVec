//! Caches of already decompressed records:
//! - `GenCache`: in RAM, with a byte limit that is charged to the budget (P2).
//! - `L2`: on disk for the LM tier (optional): cold records already decompressed are stored
//!   (re-compressed with fast zstd) in an append-only file; they survive restarts and are read in
//!   µs. Past its maximum size it is emptied (simple policy: repeated accesses refill it).

use crate::mem::Charge;
use crate::textstore::TextRec;
use crate::varint::{get, put};
use anyhow::Result;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::hash::Hash;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// RAM cache with a byte limit and two generations (approximates LRU without linked lists): new
/// entries go to the current generation; when it reaches half the limit it becomes the old one
/// (and the old one is dropped). A hit in the old generation moves the entry to the current one.
/// By `weigh` (which includes the hash table), it never exceeds `cap` bytes; its size is recorded
/// in `charge`.
pub struct GenCache<K, V> {
    cur: HashMap<K, V>,
    old: HashMap<K, V>,
    cur_bytes: u64,
    old_bytes: u64,
    cap: u64,
    weigh: fn(&V) -> u64,
    charge: Option<Charge>,
    pub hits: u64,
    pub misses: u64,
}

impl<K: Hash + Eq + Copy, V: Clone> GenCache<K, V> {
    pub fn new(cap: u64, weigh: fn(&V) -> u64, charge: Option<Charge>) -> Self {
        Self { cur: HashMap::new(), old: HashMap::new(), cur_bytes: 0, old_bytes: 0, cap, weigh, charge, hits: 0, misses: 0 }
    }

    pub fn get(&mut self, k: &K) -> Option<V> {
        if let Some(v) = self.cur.get(k) {
            self.hits += 1;
            return Some(v.clone());
        }
        match self.old.remove(k) {
            Some(v) => {
                self.hits += 1;
                let w = (self.weigh)(&v);
                self.old_bytes -= w;
                let out = v.clone();
                self.insert(*k, v, w);
                Some(out)
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// Is it there? (without counting a hit or moving it).
    pub fn contains(&self, k: &K) -> bool {
        self.cur.contains_key(k) || self.old.contains_key(k)
    }

    pub fn put(&mut self, k: K, v: V) {
        let w = (self.weigh)(&v);
        if let Some(x) = self.cur.remove(&k) {
            self.cur_bytes -= (self.weigh)(&x);
        }
        if let Some(x) = self.old.remove(&k) {
            self.old_bytes -= (self.weigh)(&x);
        }
        if w <= self.cap / 2 {
            self.insert(k, v, w);
        } else {
            self.sync();
        }
    }

    fn insert(&mut self, k: K, v: V, w: u64) {
        if self.cur_bytes + w > self.cap / 2 {
            self.old = std::mem::take(&mut self.cur);
            self.old_bytes = std::mem::take(&mut self.cur_bytes);
        }
        self.cur.insert(k, v);
        self.cur_bytes += w;
        self.sync();
    }

    fn sync(&mut self) {
        let b = self.bytes();
        if let Some(c) = self.charge.as_mut() {
            c.resize(b);
        }
    }

    /// Changes the limit; a generation larger than half of it is dropped (so the cache stays
    /// within the new limit from now on).
    pub fn set_cap(&mut self, cap: u64) {
        self.cap = cap;
        if self.old_bytes > cap / 2 {
            self.old = HashMap::new();
            self.old_bytes = 0;
        }
        if self.cur_bytes > cap / 2 {
            self.cur = HashMap::new();
            self.cur_bytes = 0;
        }
        self.sync();
    }

    pub fn clear(&mut self) {
        self.cur = HashMap::new();
        self.old = HashMap::new();
        self.cur_bytes = 0;
        self.old_bytes = 0;
        self.sync();
    }

    pub fn bytes(&self) -> u64 {
        self.cur_bytes + self.old_bytes
    }

    pub fn cap(&self) -> u64 {
        self.cap
    }

    pub fn len(&self) -> usize {
        self.cur.len() + self.old.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Bytes counted per entry of the L2's RAM index (hash table with slack).
pub const L2_ENTRY_BYTES: u64 = 96;

pub struct L2 {
    #[allow(dead_code)]
    path: PathBuf,
    cap: u64,
    inner: Mutex<Inner>,
}

struct Inner {
    file: File,
    len: u64,
    index: HashMap<(u64, u32), (u64, u32)>,
    charge: Option<Charge>,
    /// cap on index entries (its RAM: entries x `L2_ENTRY_BYTES`)
    max_entries: usize,
}

impl Inner {
    fn clear(&mut self) {
        self.index = HashMap::new();
        self.len = 0;
        let _ = self.file.set_len(0);
        self.sync();
    }

    fn sync(&mut self) {
        let b = self.index.len() as u64 * L2_ENTRY_BYTES;
        if let Some(c) = self.charge.as_mut() {
            c.resize(b);
        }
    }
}

pub fn seg_key(name: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in name.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn encode(r: &TextRec) -> Vec<u8> {
    let mut raw = Vec::with_capacity(r.title.len() + r.url.len() + r.text.len() + 8);
    put(&mut raw, r.title.len() as u64);
    put(&mut raw, r.url.len() as u64);
    raw.extend_from_slice(r.title.as_bytes());
    raw.extend_from_slice(r.url.as_bytes());
    raw.extend_from_slice(r.text.as_bytes());
    zstd::bulk::compress(&raw, 3).unwrap_or_default()
}

fn decode(b: &[u8]) -> Option<TextRec> {
    let raw = zstd::stream::decode_all(b).ok()?;
    let mut p = 0;
    let tl = get(&raw, &mut p) as usize;
    let ul = get(&raw, &mut p) as usize;
    Some(TextRec {
        title: String::from_utf8(raw.get(p..p + tl)?.to_vec()).ok()?,
        url: String::from_utf8(raw.get(p + tl..p + tl + ul)?.to_vec()).ok()?,
        text: String::from_utf8(raw.get(p + tl + ul..)?.to_vec()).ok()?,
    })
}

impl L2 {
    /// Opens (or creates) the cache and rebuilds the index by reading the file's record headers
    /// (without loading the whole file into memory).
    pub fn open(path: &Path, cap_bytes: u64) -> Result<Self> {
        Self::open_limited(path, cap_bytes, usize::MAX)
    }

    /// Like `open`, indexing at most the `max_entries` newest records (the file is append-only:
    /// later records are newer).
    pub fn open_limited(path: &Path, cap_bytes: u64, max_entries: usize) -> Result<Self> {
        // NOTE (Windows): a file opened in append mode does not allow set_len (access denied),
        // so it is opened read/write and every write seeks to the end.
        let mut oo = OpenOptions::new();
        oo.read(true).write(true).create(true);
        // exclusive access: two processes writing the same file would corrupt it
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            oo.share_mode(0);
        }
        let mut file = oo.open(path)?;
        let flen = file.metadata()?.len();
        // record: [u64 seg][u32 doc][u32 len][payload]. Two passes over the headers: count the
        // complete records, then index only the newest `max_entries`.
        let scan = |f: &mut File, skip: usize, index: Option<&mut HashMap<(u64, u32), (u64, u32)>>| -> Result<(usize, u64)> {
            f.seek(SeekFrom::Start(0))?;
            let mut rd = std::io::BufReader::with_capacity(1 << 16, f);
            let (mut n, mut p) = (0usize, 0u64);
            let mut h = [0u8; 16];
            let mut index = index;
            while p + 16 <= flen && rd.read_exact(&mut h).is_ok() {
                let seg = u64::from_le_bytes(h[0..8].try_into().unwrap());
                let doc = u32::from_le_bytes(h[8..12].try_into().unwrap());
                let len = u32::from_le_bytes(h[12..16].try_into().unwrap());
                if p + 16 + len as u64 > flen {
                    break; // incomplete tail (unexpected stop): ignored
                }
                if n >= skip {
                    if let Some(ix) = index.as_deref_mut() {
                        ix.insert((seg, doc), (p + 16, len));
                    }
                }
                rd.seek_relative(len as i64)?;
                p += 16 + len as u64;
                n += 1;
            }
            Ok((n, p))
        };
        let (n, p) = scan(&mut file, usize::MAX, None)?;
        let mut index = HashMap::new();
        scan(&mut file, n.saturating_sub(max_entries), Some(&mut index))?;
        file.set_len(p)?;
        Ok(Self { path: path.to_path_buf(), cap: cap_bytes, inner: Mutex::new(Inner { file, len: p, index, charge: None, max_entries }) })
    }

    /// Caps the RAM of the index at `bytes`; if it is over, only the newest entries stay indexed
    /// (the older records remain in the file until it is emptied).
    pub fn set_index_bytes(&self, bytes: u64) {
        if let Ok(mut g) = self.inner.lock() {
            g.max_entries = (bytes / L2_ENTRY_BYTES) as usize;
            let (n, max) = (g.index.len(), g.max_entries);
            if n > max {
                if max == 0 {
                    g.index = HashMap::new();
                } else {
                    let mut offs: Vec<u64> = g.index.values().map(|v| v.0).collect();
                    let (_, &mut cut, _) = offs.select_nth_unstable(n - max);
                    g.index.retain(|_, v| v.0 >= cut);
                    g.index.shrink_to_fit();
                }
                g.sync();
            }
        }
    }

    /// RAM cap of the index (u64::MAX = none).
    pub fn index_bytes_cap(&self) -> u64 {
        self.inner.lock().map(|g| if g.max_entries == usize::MAX { u64::MAX } else { g.max_entries as u64 * L2_ENTRY_BYTES }).unwrap_or(0)
    }

    /// Charges the RAM of the cache's index to the budget (now and as it grows).
    pub fn set_charge(&self, c: Charge) {
        if let Ok(mut g) = self.inner.lock() {
            g.charge = Some(c);
            g.sync();
        }
    }

    pub fn get(&self, seg: u64, doc: u32) -> Option<TextRec> {
        let mut g = self.inner.lock().ok()?;
        let &(off, len) = g.index.get(&(seg, doc))?;
        // read the header too and check that it is the requested record
        let mut b = vec![0u8; 16 + len as usize];
        g.file.seek(SeekFrom::Start(off.checked_sub(16)?)).ok()?;
        g.file.read_exact(&mut b).ok()?;
        let hs = u64::from_le_bytes(b[0..8].try_into().unwrap());
        let hd = u32::from_le_bytes(b[8..12].try_into().unwrap());
        let hl = u32::from_le_bytes(b[12..16].try_into().unwrap());
        if hs != seg || hd != doc || hl != len {
            return None;
        }
        decode(&b[16..])
    }

    pub fn put(&self, seg: u64, doc: u32, r: &TextRec) {
        let payload = encode(r);
        let Ok(mut g) = self.inner.lock() else { return };
        if g.index.contains_key(&(seg, doc)) || g.max_entries == 0 {
            return;
        }
        if g.len + payload.len() as u64 + 16 > self.cap || g.index.len() >= g.max_entries {
            // full (on disk or its RAM index): start over
            g.clear();
        }
        let mut rec = Vec::with_capacity(16 + payload.len());
        rec.extend_from_slice(&seg.to_le_bytes());
        rec.extend_from_slice(&doc.to_le_bytes());
        rec.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        rec.extend_from_slice(&payload);
        let at = g.len;
        if g.file.seek(SeekFrom::Start(at)).is_ok() && g.file.write_all(&rec).is_ok() {
            let off = at + 16;
            g.index.insert((seg, doc), (off, payload.len() as u32));
            g.len += rec.len() as u64;
            g.sync();
        }
    }

    pub fn entries(&self) -> usize {
        self.inner.lock().map(|g| g.index.len()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_get_and_reopen() {
        let p = std::env::temp_dir().join(format!("hnz_l2_test_{}.bin", std::process::id()));
        let _ = std::fs::remove_file(&p);
        let r1 = TextRec { title: "t".into(), url: "u".into(), text: "hola <p>mundo".into() };
        {
            let c = L2::open(&p, 1 << 20).unwrap();
            c.put(1, 5, &r1);
            assert_eq!(c.get(1, 5), Some(r1.clone()));
            assert_eq!(c.get(1, 6), None);
            assert_eq!(c.get(2, 5), None);
        }
        // persistence after reopening (and a cut tail is ignored)
        {
            let c = L2::open(&p, 1 << 20).unwrap();
            c.put(1, 6, &r1);
        }
        let n = std::fs::metadata(&p).unwrap().len();
        std::fs::OpenOptions::new().write(true).open(&p).unwrap().set_len(n - 3).unwrap();
        let c = L2::open(&p, 1 << 20).unwrap();
        assert_eq!(c.entries(), 1);
        assert_eq!(c.get(1, 5), Some(r1.clone()));
        // RAM cap of the index: 2 entries; the 3rd one empties it and starts over
        c.set_index_bytes(2 * L2_ENTRY_BYTES);
        c.put(1, 7, &r1);
        assert_eq!(c.entries(), 2);
        c.put(1, 8, &r1);
        assert_eq!(c.entries(), 1);
        assert_eq!(c.get(1, 8), Some(r1.clone()));
        c.set_index_bytes(0);
        assert_eq!(c.entries(), 0);
        drop(c);
        // reopening with a smaller cap keeps the newest records, not nothing
        let _ = std::fs::remove_file(&p);
        {
            let c = L2::open(&p, 1 << 20).unwrap();
            for d in 0..10 {
                c.put(3, d, &r1);
            }
            c.set_index_bytes(4 * L2_ENTRY_BYTES);
            assert_eq!(c.entries(), 4);
            assert!(c.get(3, 9).is_some() && c.get(3, 6).is_some() && c.get(3, 5).is_none());
        }
        let c = L2::open_limited(&p, 1 << 20, 3).unwrap();
        assert_eq!(c.entries(), 3);
        assert!(c.get(3, 9).is_some() && c.get(3, 7).is_some() && c.get(3, 6).is_none());
        drop(c);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn gencache_bytes_bounded() {
        let b = crate::mem::Budget::new(None);
        let mut c: GenCache<u32, String> = GenCache::new(1000, |s| 50 + s.len() as u64, Some(b.charge(crate::mem::Class::Cache, 0)));
        for i in 0..100u32 {
            c.put(i, "x".repeat(50));
            assert!(c.bytes() <= 1000, "{}", c.bytes());
            assert_eq!(b.used(), c.bytes());
        }
        // the newest is still there; the oldest is gone
        assert!(c.get(&99).is_some());
        assert!(c.get(&0).is_none());
        // a hit in the old generation moves it to the current one
        let old_key = (0..100u32).rev().find(|k| c.old.contains_key(k)).unwrap();
        assert!(c.get(&old_key).is_some());
        assert!(c.cur.contains_key(&old_key));
        // too big: not stored
        c.put(500, "y".repeat(600));
        assert!(!c.contains(&500));
        c.set_cap(300);
        assert!(c.bytes() <= 300);
        assert_eq!(b.used(), c.bytes());
        c.clear();
        assert_eq!(b.used(), 0);
    }
}
