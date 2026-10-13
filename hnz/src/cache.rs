//! Caché L2 en disco para el nivel LM: los registros fríos ya descomprimidos se guardan
//! (re-comprimidos con zstd rápido) en un fichero de solo-añadir; sobreviven a reinicios y se leen
//! en µs. Al superar el tamaño máximo se vacía (política simple: los accesos repetidos lo rellenan).

use crate::textstore::TextRec;
use crate::varint::{get, put};
use anyhow::Result;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub struct L2 {
    path: PathBuf,
    cap: u64,
    inner: Mutex<Inner>,
}

struct Inner {
    file: File,
    len: u64,
    index: HashMap<(u64, u32), (u64, u32)>,
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
    /// Abre (o crea) la caché y reconstruye el índice leyendo el fichero.
    pub fn open(path: &Path, cap_bytes: u64) -> Result<Self> {
        let mut file = OpenOptions::new().read(true).append(true).create(true).open(path)?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;
        let mut index = HashMap::new();
        let mut p = 0usize;
        // registro: [u64 seg][u32 doc][u32 len][payload]
        while p + 16 <= buf.len() {
            let seg = u64::from_le_bytes(buf[p..p + 8].try_into().unwrap());
            let doc = u32::from_le_bytes(buf[p + 8..p + 12].try_into().unwrap());
            let len = u32::from_le_bytes(buf[p + 12..p + 16].try_into().unwrap());
            if p + 16 + len as usize > buf.len() {
                break; // cola incompleta (corte inesperado): se ignora
            }
            index.insert((seg, doc), ((p + 16) as u64, len));
            p += 16 + len as usize;
        }
        let len = p as u64;
        file.set_len(len)?;
        Ok(Self { path: path.to_path_buf(), cap: cap_bytes, inner: Mutex::new(Inner { file, len, index }) })
    }

    pub fn get(&self, seg: u64, doc: u32) -> Option<TextRec> {
        let mut g = self.inner.lock().ok()?;
        let &(off, len) = g.index.get(&(seg, doc))?;
        let mut b = vec![0u8; len as usize];
        g.file.seek(SeekFrom::Start(off)).ok()?;
        g.file.read_exact(&mut b).ok()?;
        decode(&b)
    }

    pub fn put(&self, seg: u64, doc: u32, r: &TextRec) {
        let payload = encode(r);
        let Ok(mut g) = self.inner.lock() else { return };
        if g.index.contains_key(&(seg, doc)) {
            return;
        }
        if g.len + payload.len() as u64 + 16 > self.cap {
            // vaciar
            if let Ok(f) = OpenOptions::new().read(true).write(true).truncate(true).open(&self.path) {
                drop(f);
            }
            g.index.clear();
            g.len = 0;
            let _ = g.file.set_len(0);
        }
        let mut rec = Vec::with_capacity(16 + payload.len());
        rec.extend_from_slice(&seg.to_le_bytes());
        rec.extend_from_slice(&doc.to_le_bytes());
        rec.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        rec.extend_from_slice(&payload);
        if g.file.seek(SeekFrom::End(0)).is_ok() && g.file.write_all(&rec).is_ok() {
            let off = g.len + 16;
            g.index.insert((seg, doc), (off, payload.len() as u32));
            g.len += rec.len() as u64;
        }
    }

    pub fn entries(&self) -> usize {
        self.inner.lock().map(|g| g.index.len()).unwrap_or(0)
    }
}
