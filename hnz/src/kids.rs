//! The order of each item's `kids` (P3, user decision D3): HN lists the direct replies of an item in
//! its own ranking order. Almost always they are exactly the item's children, which the block
//! already knows through `parent`, so only their order is stored. Per document, in document order:
//! ```text
//! code 0                  no kids
//! code 1                  the children, in id order
//! code 2                  the children, in reverse id order
//! code 3, k varints       a permutation of the children (positions in id order)
//! code 4, n, n varints    an explicit list (zigzag id deltas, chained from the item's id)
//! ```
//! in zstd frames of 1,024 documents (in 2010-2015 ~77 % are in id order and the rest cost a few
//! KB per month; recent months of the dataset carry no kids at all).
use crate::varint::{put, try_get, unzz, zz};
use anyhow::Result;

pub const BLOCK: usize = 1024;

fn encode_one(out: &mut Vec<u8>, id: u32, kids: &[u32], children: &[u32]) {
    if kids.is_empty() {
        out.push(0);
        return;
    }
    let mut sorted = kids.to_vec();
    sorted.sort_unstable();
    if sorted.windows(2).all(|w| w[0] < w[1]) && sorted == children {
        if kids == children {
            out.push(1);
            return;
        }
        if kids.iter().rev().eq(children.iter()) {
            out.push(2);
            return;
        }
        out.push(3);
        for k in kids {
            put(out, children.binary_search(k).unwrap() as u64);
        }
        return;
    }
    out.push(4);
    put(out, kids.len() as u64);
    let mut prev = id as i64;
    for &k in kids {
        put(out, zz(k as i64 - prev));
        prev = k as i64;
    }
}

/// Encodes the kids of every document: `ids[d]`, `kids[d]` and `children[d]` (the sorted ids of the
/// documents whose parent is `ids[d]`). Returns (zstd frames, u64 offset per frame then the end).
pub fn encode(ids: &[u32], kids: &[Vec<u32>], children: &[Vec<u32>]) -> Result<(Vec<u8>, Vec<u8>)> {
    let (mut frames, mut idx) = (Vec::new(), Vec::new());
    for st in (0..ids.len()).step_by(BLOCK) {
        let en = (st + BLOCK).min(ids.len());
        let mut raw = Vec::new();
        for d in st..en {
            encode_one(&mut raw, ids[d], &kids[d], &children[d]);
        }
        idx.extend_from_slice(&(frames.len() as u64).to_le_bytes());
        frames.extend(zstd::bulk::compress(&raw, 19)?);
    }
    idx.extend_from_slice(&(frames.len() as u64).to_le_bytes());
    Ok((frames, idx))
}

/// `encode` one document at a time (`pack::pack_stream`): the same frames and index.
#[derive(Default)]
pub struct KidsWriter {
    raw: Vec<u8>,
    in_frame: usize,
    frames: Vec<u8>,
    idx: Vec<u8>,
}

impl KidsWriter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, id: u32, kids: &[u32], children: &[u32]) -> Result<()> {
        encode_one(&mut self.raw, id, kids, children);
        self.in_frame += 1;
        if self.in_frame == BLOCK {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        self.idx.extend_from_slice(&(self.frames.len() as u64).to_le_bytes());
        self.frames.extend(zstd::bulk::compress(&self.raw, 19)?);
        self.raw.clear();
        self.in_frame = 0;
        Ok(())
    }

    /// (zstd frames, u64 offset per frame then the end), as `encode`.
    pub fn finish(mut self) -> Result<(Vec<u8>, Vec<u8>)> {
        if self.in_frame > 0 {
            self.flush()?;
        }
        self.idx.extend_from_slice(&(self.frames.len() as u64).to_le_bytes());
        Ok((self.frames, self.idx))
    }
}

/// The children of every document of a block built month by month: (parent, id) of every
/// document, sorted, so a child in another month than its parent is found too (`children_of`
/// over the whole block gives the same lists).
#[derive(Default)]
pub struct ChildIndex {
    pairs: Vec<u64>,
}

impl ChildIndex {
    pub fn with_capacity(n: usize) -> Self {
        Self { pairs: Vec::with_capacity(n) }
    }

    pub fn add(&mut self, id: u32, parent: u32) {
        if parent != 0 {
            self.pairs.push((parent as u64) << 32 | id as u64);
        }
    }

    pub fn finish(&mut self) {
        use rayon::prelude::*;
        self.pairs.par_sort_unstable();
    }

    /// The sorted ids of the documents whose parent is `id` (into `out`).
    pub fn children(&self, id: u32, out: &mut Vec<u32>) {
        out.clear();
        let lo = self.pairs.partition_point(|&x| (x >> 32) < id as u64);
        out.extend(self.pairs[lo..].iter().take_while(|&&x| (x >> 32) == id as u64).map(|&x| x as u32));
    }
}

/// Decodes the kids of the documents of one frame (`ids` and `children` of those documents).
pub fn decode_frame(frame: &[u8], ids: &[u32], children: &[Vec<u32>]) -> Result<Vec<Vec<u32>>> {
    let bad = || crate::block::corrupt("meta/kids");
    let raw = zstd::stream::decode_all(frame).map_err(|_| bad())?;
    let mut p = 0;
    let mut out = Vec::with_capacity(ids.len());
    for (d, &id) in ids.iter().enumerate() {
        let c = &children[d];
        let code = try_get(&raw, &mut p).ok_or_else(bad)?;
        out.push(match code {
            0 => Vec::new(),
            1 => c.clone(),
            2 => c.iter().rev().copied().collect(),
            3 => (0..c.len()).map(|_| try_get(&raw, &mut p).and_then(|i| c.get(i as usize).copied())).collect::<Option<_>>().ok_or_else(bad)?,
            4 => {
                let n = try_get(&raw, &mut p).ok_or_else(bad)? as usize;
                if n > raw.len() {
                    return Err(bad());
                }
                let mut prev = id as i64;
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    prev += unzz(try_get(&raw, &mut p).ok_or_else(bad)?);
                    v.push(u32::try_from(prev).map_err(|_| bad())?);
                }
                v
            }
            _ => return Err(bad()),
        });
    }
    Ok(out)
}

/// Children of each document (sorted ids of the documents whose parent is its id).
pub fn children_of(ids: &[u32], parents: &[u32]) -> Vec<Vec<u32>> {
    let pos: std::collections::HashMap<u32, usize> = ids.iter().enumerate().map(|(d, &i)| (i, d)).collect();
    let mut ch = vec![Vec::new(); ids.len()];
    for (d, &p) in parents.iter().enumerate() {
        if let Some(&q) = pos.get(&p) {
            ch[q].push(ids[d]);
        }
    }
    for c in ch.iter_mut() {
        c.sort_unstable();
    }
    ch
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_every_case() {
        // item 1 (doc 0) has children 2, 3, 4; item 5 has children 6, 7; item 8 has child 9
        let ids = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let parents = vec![0, 1, 1, 1, 0, 5, 5, 0, 8, 0];
        let ch = children_of(&ids, &parents);
        assert_eq!(ch[0], vec![2, 3, 4]);
        let kids = vec![
            vec![3, 2, 4],     // permutation
            vec![],            // none
            vec![],
            vec![],
            vec![7, 6],        // reverse
            vec![],
            vec![],
            vec![9, 12345],    // a kid outside the block: explicit
            vec![],
            vec![11],          // no children, but kids: explicit
        ];
        let (frames, idx) = encode(&ids, &kids, &ch).unwrap();
        assert_eq!(idx.len(), 16);
        let got = decode_frame(&frames, &ids, &ch).unwrap();
        assert_eq!(got, kids);
        // in id order
        let kids2: Vec<Vec<u32>> = ch.clone();
        let (f2, _) = encode(&ids, &kids2, &ch).unwrap();
        assert_eq!(decode_frame(&f2, &ids, &ch).unwrap(), kids2);
        assert!(decode_frame(&frames[..frames.len() / 2], &ids, &ch).is_err());
    }

    #[test]
    fn streaming_writer_and_child_index_match_encode() {
        // 2,600 documents: frames of 1,024 that span a "month" boundary, children in later docs
        let n = 2600u32;
        let ids: Vec<u32> = (1..=n).collect();
        let parents: Vec<u32> = ids.iter().map(|&i| if i % 3 == 0 { 0 } else { (i * 7 + 11) % n + 1 }).collect();
        let ch = children_of(&ids, &parents);
        let kids: Vec<Vec<u32>> = ch.iter().enumerate().map(|(d, c)| if d % 5 == 0 { c.iter().rev().copied().collect() } else { c.clone() }).collect();
        let want = encode(&ids, &kids, &ch).unwrap();
        let mut ci = ChildIndex::with_capacity(ids.len());
        for (&i, &p) in ids.iter().zip(&parents) {
            ci.add(i, p);
        }
        ci.finish();
        let mut w = KidsWriter::new();
        let mut scratch = Vec::new();
        for (d, &i) in ids.iter().enumerate() {
            ci.children(i, &mut scratch);
            assert_eq!(scratch, ch[d]);
            w.push(i, &kids[d], &scratch).unwrap();
        }
        assert_eq!(w.finish().unwrap(), want);
        assert_eq!(KidsWriter::new().finish().unwrap(), encode(&[], &[], &[]).unwrap());
    }
}
