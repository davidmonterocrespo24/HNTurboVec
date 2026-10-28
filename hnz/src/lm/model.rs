//! GPT-NeoX (Pythia family) in int8 (W8A8) with deterministic arithmetic.
//! - `prefill`: every position of a sequence at once (to COMPRESS).
//! - `Batch::step`: one token per sequence for several sequences at once, with a KV cache
//!   (to DECOMPRESS the results of a query as a batch).
//! Both paths run exactly the same operations per position => same logits.

use super::kernels::{matmul, quant_row, Q8};
use super::math::{exp, gelu, layer_norm};
use anyhow::{bail, Context, Result};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Profile (accumulated ns): [layer matmuls, attention, head, rest]
pub static PROF: [AtomicU64; 4] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
#[inline]
fn prof_add(i: usize, t: std::time::Instant) {
    PROF[i].fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
}
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize, Debug, Clone)]
struct TensorInfo {
    name: String,
    kind: String,
    #[serde(default)]
    rows: usize,
    #[serde(default)]
    cols: usize,
    #[serde(default)]
    len: usize,
    offset: usize,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Header {
    pub arch: String,
    pub source: String,
    pub hidden: usize,
    pub heads: usize,
    pub layers: usize,
    pub inter: usize,
    pub vocab: usize,
    pub rotary_ndims: usize,
    pub max_pos: usize,
    pub eps: f32,
    pub bos: u32,
    tensors: Vec<TensorInfo>,
    vocab_offsets: usize,
    vocab_bytes: usize,
    vocab_bytes_len: usize,
}

/// int8 matrix: per-row scales (copied, aligned) + position of the data in the file.
struct QMat {
    rows: usize,
    cols: usize,
    scales: Vec<f32>,
    rowsum: Vec<i32>,
    off: usize,
}

struct LayerW {
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    ln2_w: Vec<f32>,
    ln2_b: Vec<f32>,
    qkv: QMat,
    qkv_b: Vec<f32>,
    o: QMat,
    o_b: Vec<f32>,
    fc1: QMat,
    fc1_b: Vec<f32>,
    fc2: QMat,
    fc2_b: Vec<f32>,
}

pub struct LmModel {
    pub h: Header,
    data: Vec<u8>,
    embed_in: QMat,
    layers: Vec<LayerW>,
    lnf_w: Vec<f32>,
    lnf_b: Vec<f32>,
    embed_out: QMat,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
    vocab_off: Vec<u32>,
    vocab_bytes: Vec<u8>,
    pub id: String,
    /// id from earlier versions (samples 1 in 97 bytes): accepted when opening old segments
    pub id_legacy: String,
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

impl LmModel {
    pub fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read(path).with_context(|| format!("LM model {}", path.display()))?;
        Self::from_bytes(data)
    }

    /// From the bytes of a `.hnzlm` file (its id depends only on these bytes).
    pub fn from_bytes(data: Vec<u8>) -> Result<Self> {
        if data.len() < 12 || &data[..6] != b"HNZLM1" {
            bail!("not a .hnzlm file");
        }
        // every range the header names must lie inside the file (an unexpected file is an error,
        // never a panic)
        let span = |a: usize, len: usize| -> Result<std::ops::Range<usize>> {
            a.checked_add(len).filter(|&e| e <= data.len()).map(|e| a..e).with_context(|| format!("LM model: range {a}+{len} is outside the file"))
        };
        let hl = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let h: Header = serde_json::from_slice(&data[span(12, hl)?])?;
        if h.arch != "gpt_neox" {
            bail!("unsupported architecture: {}", h.arch);
        }
        if h.heads == 0 || h.hidden == 0 || h.inter == 0 || h.hidden % h.heads != 0 || h.rotary_ndims > h.hidden / h.heads || h.rotary_ndims % 2 != 0 || h.vocab == 0 || h.max_pos == 0 {
            bail!("LM model: inconsistent dimensions in the header");
        }
        let base = 12 + hl + ((64 - (12 + hl) % 64) % 64);
        let find = |name: &str| -> Result<TensorInfo> {
            h.tensors.iter().find(|t| t.name == name).cloned().with_context(|| format!("missing tensor {name}"))
        };
        let at = |off: usize| base.checked_add(off).context("LM model: offset");
        let vecf = |name: &str, want: usize| -> Result<Vec<f32>> {
            let t = find(name)?;
            if t.len < want {
                bail!("LM model: {name} has {} values, expected {want}", t.len);
            }
            Ok(f32s(&data[span(at(t.offset)?, t.len.checked_mul(4).context("size")?)?]))
        };
        let q8 = |name: &str, rows: usize, cols: usize| -> Result<QMat> {
            let t = find(name)?;
            if t.kind != "q8" {
                bail!("{name} is not q8");
            }
            if t.rows < rows || t.cols != cols {
                bail!("LM model: {name} is {}x{}, expected {rows}x{cols}", t.rows, t.cols);
            }
            let s = span(at(t.offset)?, t.rows.checked_mul(4).context("size")?)?;
            let d = span(s.end, t.rows.checked_mul(t.cols).context("size")?)?;
            let rowsum = data[d.clone()].chunks_exact(t.cols).map(|r| r.iter().map(|&b| b as i8 as i32).sum()).collect();
            Ok(QMat { rows: t.rows, cols: t.cols, scales: f32s(&data[s]), rowsum, off: d.start })
        };
        let (hd, inter) = (h.hidden, h.inter);
        let mut layers = Vec::new();
        for l in 0..h.layers {
            layers.push(LayerW {
                ln1_w: vecf(&format!("l{l}.ln1.w"), hd)?,
                ln1_b: vecf(&format!("l{l}.ln1.b"), hd)?,
                ln2_w: vecf(&format!("l{l}.ln2.w"), hd)?,
                ln2_b: vecf(&format!("l{l}.ln2.b"), hd)?,
                qkv: q8(&format!("l{l}.qkv.w"), 3 * hd, hd)?,
                qkv_b: vecf(&format!("l{l}.qkv.b"), 3 * hd)?,
                o: q8(&format!("l{l}.o.w"), hd, hd)?,
                o_b: vecf(&format!("l{l}.o.b"), hd)?,
                fc1: q8(&format!("l{l}.fc1.w"), inter, hd)?,
                fc1_b: vecf(&format!("l{l}.fc1.b"), inter)?,
                fc2: q8(&format!("l{l}.fc2.w"), hd, inter)?,
                fc2_b: vecf(&format!("l{l}.fc2.b"), hd)?,
            });
        }
        let vo = span(at(h.vocab_offsets)?, (h.vocab + 1).checked_mul(4).context("size")?)?;
        let vocab_off: Vec<u32> = data[vo].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        let vocab_bytes = data[span(at(h.vocab_bytes)?, h.vocab_bytes_len)?].to_vec();
        if vocab_off.windows(2).any(|w| w[1] < w[0]) || *vocab_off.last().unwrap() as usize > vocab_bytes.len() {
            bail!("LM model: inconsistent token byte table");
        }
        let id = format!("{}#full:{:016x}", h.source, fnv_full(&data));
        let id_legacy = format!("{}#{:016x}", h.source, fnv(&data));
        Ok(Self {
            embed_in: q8("embed_in", h.vocab, hd)?,
            lnf_w: vecf("lnf.w", hd)?,
            lnf_b: vecf("lnf.b", hd)?,
            embed_out: q8("embed_out", h.vocab, hd)?,
            rope_cos: vecf("rope_cos", h.max_pos * (h.rotary_ndims / 2))?,
            rope_sin: vecf("rope_sin", h.max_pos * (h.rotary_ndims / 2))?,
            layers,
            vocab_off,
            vocab_bytes,
            id,
            id_legacy,
            h,
            data,
        })
    }

    fn q8<'a>(&'a self, m: &'a QMat) -> Q8<'a> {
        let bytes = &self.data[m.off..m.off + m.rows * m.cols];
        // u8 -> i8 (same representation, alignment 1)
        let data = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const i8, bytes.len()) };
        Q8 { rows: m.rows, cols: m.cols, scales: &m.scales, rowsum: &m.rowsum, data }
    }

    /// KV-cache bytes per position and sequence (keys + values of all layers, f32).
    pub fn kv_bytes_per_token(&self) -> u64 {
        (2 * self.h.hidden * self.h.layers * 4) as u64
    }

    /// Working memory of each active sequence in one `Batch::step` + sampling: logits and
    /// cumulative frequencies (vocabulary x 4 B each) and the activations of one position.
    pub fn step_bytes_per_seq(&self) -> u64 {
        (2 * self.h.vocab * 4 + 16 * self.h.hidden * 4) as u64
    }

    /// Bytes of a token (byte-level BPE).
    pub fn token_bytes(&self, id: u32) -> &[u8] {
        let i = id as usize;
        &self.vocab_bytes[self.vocab_off[i] as usize..self.vocab_off[i + 1] as usize]
    }

    fn embed(&self, tok: u32, out: &mut [f32]) {
        let e = self.q8(&self.embed_in);
        let r = e.row(tok as usize);
        let s = e.scales[tok as usize];
        for (o, &q) in out.iter_mut().zip(r) {
            *o = q as f32 * s;
        }
    }

    fn rope(&self, x: &mut [f32], pos: usize) {
        let half = self.h.rotary_ndims / 2;
        let c = &self.rope_cos[pos * half..(pos + 1) * half];
        let s = &self.rope_sin[pos * half..(pos + 1) * half];
        for i in 0..half {
            let a = x[i];
            let b = x[i + half];
            x[i] = a * c[i] - b * s[i];
            x[i + half] = b * c[i] + a * s[i];
        }
    }

    /// Runs one layer over `t` rows (tokens) at positions `pos[i]`, using and updating the
    /// KV cache of sequence `seq_of[i]`. `x`: t x H (updated in place).
    fn layer(&self, l: usize, x: &mut [f32], pos: &[usize], kv: &mut [&mut KvCache], seq_of: &[usize], parallel: bool) {
        let (hd, nh, it) = (self.h.hidden, self.h.heads, self.h.inter);
        let hs = hd / nh;
        let t = pos.len();
        let lw = &self.layers[l];
        let mut a = vec![0f32; t * hd];
        let mut m = vec![0f32; t * hd];
        for i in 0..t {
            layer_norm(&x[i * hd..(i + 1) * hd], &lw.ln1_w, &lw.ln1_b, self.h.eps, &mut a[i * hd..(i + 1) * hd]);
            layer_norm(&x[i * hd..(i + 1) * hd], &lw.ln2_w, &lw.ln2_b, self.h.eps, &mut m[i * hd..(i + 1) * hd]);
        }
        // --- attention ---
        let mut aq = vec![0i8; t * hd];
        let mut asx = vec![0f32; t];
        for i in 0..t {
            asx[i] = quant_row(&a[i * hd..(i + 1) * hd], &mut aq[i * hd..(i + 1) * hd]);
        }
        let mut qkv = vec![0f32; t * 3 * hd];
        let tm = std::time::Instant::now();
        matmul(&self.q8(&lw.qkv), &aq, &asx, Some(&lw.qkv_b), &mut qkv, parallel);
        prof_add(0, tm);
        let ta = std::time::Instant::now();
        for i in 0..t {
            let row = &mut qkv[i * 3 * hd..(i + 1) * 3 * hd];
            for h in 0..nh {
                self.rope(&mut row[h * hs..(h + 1) * hs], pos[i]);
                self.rope(&mut row[hd + h * hs..hd + (h + 1) * hs], pos[i]);
            }
            let c = &mut kv[seq_of[i]];
            debug_assert_eq!(c.len[l], pos[i]);
            c.push(l, &row[hd..2 * hd], &row[2 * hd..3 * hd]);
        }
        let scale = 1.0 / (hs as f32).sqrt();
        let mut att = vec![0f32; t * hd];
        {
            let kvr: Vec<&KvCache> = kv.iter().map(|c| &**c).collect();
            let one = |task: usize, o: &mut [f32]| {
                let (i, h) = (task / nh, task % nh);
                let c = kvr[seq_of[i]];
                let n = pos[i] + 1;
                let q = &qkv[i * 3 * hd + h * hs..i * 3 * hd + (h + 1) * hs];
                // s[j] = (Σ_d q[d]·k[j][d]) · scale, with each key's sum in order d = 0..hs
                // (canonical definition, MATH_VERSION 2): vectorized across the keys
                let mut sc = vec![0f32; n];
                // blocks of 32 keys with the accumulators in registers across the dims;
                // for each key the order of the sums is the same (d = 0..hs)
                let rows: Vec<&[f32]> = (0..hs).map(|d| &c.krow(l, h, d)[..n]).collect();
                let mut jb = 0;
                while jb + 32 <= n {
                    let mut acc = [0f32; 32];
                    for d in 0..hs {
                        let qd = q[d];
                        let kr: &[f32; 32] = rows[d][jb..jb + 32].try_into().unwrap();
                        for lane in 0..32 {
                            acc[lane] += qd * kr[lane];
                        }
                    }
                    sc[jb..jb + 32].copy_from_slice(&acc);
                    jb += 32;
                }
                if jb < n {
                    let rem = n - jb;
                    let mut acc = [0f32; 32];
                    for d in 0..hs {
                        let qd = q[d];
                        let kr = &rows[d][jb..n];
                        for lane in 0..rem {
                            acc[lane] += qd * kr[lane];
                        }
                    }
                    sc[jb..n].copy_from_slice(&acc[..rem]);
                }
                let mut mx = f32::NEG_INFINITY;
                for s in sc.iter_mut() {
                    *s *= scale;
                    if *s > mx {
                        mx = *s;
                    }
                }
                // exp in a separate loop (vectorizable), and the sum in the SAME sequential order as always
                for s in sc.iter_mut() {
                    *s = exp(*s - mx);
                }
                let mut tot = 0f32;
                for &s in sc.iter() {
                    tot += s;
                }
                let inv = 1.0 / tot;
                if hs == 64 {
                    // 64 accumulators in registers across the keys (same order per dim)
                    let mut acc = [0f32; 64];
                    for j in 0..n {
                        let p = sc[j] * inv;
                        let v: &[f32; 64] = c.v[l][j * hd + h * hs..j * hd + (h + 1) * hs].try_into().unwrap();
                        for d in 0..64 {
                            acc[d] += p * v[d];
                        }
                    }
                    for d in 0..64 {
                        o[d] += acc[d];
                    }
                } else {
                    for j in 0..n {
                        let p = sc[j] * inv;
                        let v = &c.v[l][j * hd + h * hs..j * hd + (h + 1) * hs];
                        for d in 0..hs {
                            o[d] += p * v[d];
                        }
                    }
                }
            };
            let work: usize = pos.iter().map(|&p| p + 1).sum::<usize>() * nh;
            if parallel && work > 2048 {
                let base = super::kernels::SendPtr(att.as_mut_ptr());
                super::pool::pool().run(t * nh, &|task| {
                    let (i, h) = (task / nh, task % nh);
                    // SAFETY: each (row, head) writes its own disjoint chunk of `att`
                    let o = unsafe { std::slice::from_raw_parts_mut(base.get().add(i * hd + h * hs), hs) };
                    one(task, o);
                });
            } else {
                for task in 0..t * nh {
                    let (i, h) = (task / nh, task % nh);
                    one(task, &mut att[i * hd + h * hs..i * hd + (h + 1) * hs]);
                }
            }
        }
        prof_add(1, ta);
        let tm = std::time::Instant::now();
        let mut attq = vec![0i8; t * hd];
        let mut attsx = vec![0f32; t];
        for i in 0..t {
            attsx[i] = quant_row(&att[i * hd..(i + 1) * hd], &mut attq[i * hd..(i + 1) * hd]);
        }
        let mut ao = vec![0f32; t * hd];
        matmul(&self.q8(&lw.o), &attq, &attsx, Some(&lw.o_b), &mut ao, parallel);
        // --- MLP ---
        let mut mq = vec![0i8; t * hd];
        let mut msx = vec![0f32; t];
        for i in 0..t {
            msx[i] = quant_row(&m[i * hd..(i + 1) * hd], &mut mq[i * hd..(i + 1) * hd]);
        }
        let mut f1 = vec![0f32; t * it];
        matmul(&self.q8(&lw.fc1), &mq, &msx, Some(&lw.fc1_b), &mut f1, parallel);
        for v in f1.iter_mut() {
            *v = gelu(*v);
        }
        let mut f1q = vec![0i8; t * it];
        let mut f1sx = vec![0f32; t];
        for i in 0..t {
            f1sx[i] = quant_row(&f1[i * it..(i + 1) * it], &mut f1q[i * it..(i + 1) * it]);
        }
        let mut f2 = vec![0f32; t * hd];
        matmul(&self.q8(&lw.fc2), &f1q, &f1sx, Some(&lw.fc2_b), &mut f2, parallel);
        prof_add(0, tm);
        // parallel residual: h = mlp + attn + h
        for i in 0..t * hd {
            x[i] = (f2[i] + ao[i]) + x[i];
        }
    }

    fn head(&self, x: &[f32], t: usize, logits: &mut [f32], parallel: bool) {
        let hd = self.h.hidden;
        let mut n = vec![0f32; t * hd];
        for i in 0..t {
            layer_norm(&x[i * hd..(i + 1) * hd], &self.lnf_w, &self.lnf_b, self.h.eps, &mut n[i * hd..(i + 1) * hd]);
        }
        let mut q = vec![0i8; t * hd];
        let mut sx = vec![0f32; t];
        for i in 0..t {
            sx[i] = quant_row(&n[i * hd..(i + 1) * hd], &mut q[i * hd..(i + 1) * hd]);
        }
        let th = std::time::Instant::now();
        matmul(&self.q8(&self.embed_out), &q, &sx, None, logits, parallel);
        prof_add(2, th);
    }

    /// Logits of every position of a sequence (position i predicts token i+1).
    pub fn prefill(&self, toks: &[u32], parallel: bool) -> Vec<f32> {
        let t = toks.len();
        let hd = self.h.hidden;
        let mut x = vec![0f32; t * hd];
        for (i, &tk) in toks.iter().enumerate() {
            self.embed(tk, &mut x[i * hd..(i + 1) * hd]);
        }
        let pos: Vec<usize> = (0..t).collect();
        let mut cache = KvCache::new(self.h.layers, hd, hd / self.h.heads);
        let seq_of = vec![0usize; t];
        // causality: row i only sees keys 0..=i -> we go layer by layer, with the cache
        // growing row by row inside `layer` (pos[i] == len before adding).
        for l in 0..self.h.layers {
            let mut refs = [&mut cache];
            self.layer(l, &mut x, &pos, &mut refs, &seq_of, parallel);
        }
        let mut logits = vec![0f32; t * self.h.vocab];
        self.head(&x, t, &mut logits, parallel);
        logits
    }

    /// Like `prefill`, but hands the logits to `f(pos0, logits)` in CHUNKS of `chunk` positions,
    /// so it never allocates T x vocab floats at once (2,000 tokens = 400 MB). Same bits: the head
    /// is computed row by row in exactly the same way.
    pub fn prefill_chunked(&self, toks: &[u32], chunk: usize, mut f: impl FnMut(usize, &[f32])) {
        let t = toks.len();
        let hd = self.h.hidden;
        let mut x = vec![0f32; t * hd];
        for (i, &tk) in toks.iter().enumerate() {
            self.embed(tk, &mut x[i * hd..(i + 1) * hd]);
        }
        let pos: Vec<usize> = (0..t).collect();
        let mut cache = KvCache::new(self.h.layers, hd, hd / self.h.heads);
        let seq_of = vec![0usize; t];
        for l in 0..self.h.layers {
            let mut refs = [&mut cache];
            self.layer(l, &mut x, &pos, &mut refs, &seq_of, false);
        }
        drop(cache);
        let mut logits = vec![0f32; chunk.min(t) * self.h.vocab];
        for p0 in (0..t).step_by(chunk) {
            let n = chunk.min(t - p0);
            self.head(&x[p0 * hd..(p0 + n) * hd], n, &mut logits[..n * self.h.vocab], false);
            f(p0, &logits[..n * self.h.vocab]);
        }
    }
}

/// Version of the LM's numeric definition. Changing it invalidates segments already compressed.
/// 1: attention scores with `dot` per key. 2: transposed keys, summed per dim in order.
pub const MATH_VERSION: u32 = 2;

/// KV cache of one sequence. Keys TRANSPOSED per layer: [head·dim][position] (capacity doubles as
/// it grows, never past `limit` positions), to score many keys at once; values [pos][hidden] with
/// the same capacity.
pub struct KvCache {
    kt: Vec<Vec<f32>>,
    cap: Vec<usize>,
    pub len: Vec<usize>,
    pub v: Vec<Vec<f32>>,
    hd: usize,
    hs: usize,
    limit: usize,
}

impl KvCache {
    pub fn new(layers: usize, hd: usize, hs: usize) -> Self {
        Self::with_limit(layers, hd, hs, usize::MAX)
    }

    /// With at most `limit` positions: memory stays within `limit` x `kv_bytes_per_token` (plus,
    /// while growing, the copy of one layer's keys). If exceeded, it grows anyway.
    pub fn with_limit(layers: usize, hd: usize, hs: usize, limit: usize) -> Self {
        Self { kt: vec![Vec::new(); layers], cap: vec![0; layers], len: vec![0; layers], v: vec![Vec::new(); layers], hd, hs, limit }
    }

    fn push(&mut self, l: usize, krow: &[f32], vrow: &[f32]) {
        let (hd, len) = (self.hd, self.len[l]);
        if len == self.cap[l] {
            let nc = (self.cap[l] * 2).max(32).min(self.limit).max(len + 1);
            let mut nk = vec![0f32; hd * nc];
            for r in 0..hd {
                nk[r * nc..r * nc + len].copy_from_slice(&self.kt[l][r * self.cap[l]..r * self.cap[l] + len]);
            }
            self.kt[l] = nk;
            self.cap[l] = nc;
            let have = self.v[l].len();
            self.v[l].reserve_exact(nc * hd - have);
        }
        let c = self.cap[l];
        for (r, &x) in krow.iter().enumerate() {
            self.kt[l][r * c + len] = x;
        }
        self.v[l].extend_from_slice(vrow);
        self.len[l] += 1;
    }

    #[inline]
    fn krow(&self, l: usize, h: usize, d: usize) -> &[f32] {
        let r = h * self.hs + d;
        &self.kt[l][r * self.cap[l]..r * self.cap[l] + self.len[l]]
    }
}

/// Batch decoding: several sequences advance one token per step.
pub struct Batch<'m> {
    m: &'m LmModel,
    pub caches: Vec<KvCache>,
    pub pos: Vec<usize>,
}

impl<'m> Batch<'m> {
    pub fn new(m: &'m LmModel, n: usize) -> Self {
        let (hd, hs) = (m.h.hidden, m.h.hidden / m.h.heads);
        Self { m, caches: (0..n).map(|_| KvCache::new(m.h.layers, hd, hs)).collect(), pos: vec![0; n] }
    }

    /// One more slot, with an empty KV cache; returns its index.
    pub fn add_slot(&mut self) -> usize {
        let (hd, hs) = (self.m.h.hidden, self.m.h.hidden / self.m.h.heads);
        self.caches.push(KvCache::with_limit(self.m.h.layers, hd, hs, 0));
        self.pos.push(0);
        self.caches.len() - 1
    }

    /// Empties slot `slot` (KV cache and position) to start another sequence in it.
    pub fn reset(&mut self, slot: usize) {
        self.reset_limited(slot, usize::MAX);
    }

    /// Like `reset`, with the slot's KV cache limited to `limit` positions (0 = free its memory).
    pub fn reset_limited(&mut self, slot: usize, limit: usize) {
        let (hd, hs) = (self.m.h.hidden, self.m.h.hidden / self.m.h.heads);
        self.caches[slot] = KvCache::with_limit(self.m.h.layers, hd, hs, limit);
        self.pos[slot] = 0;
    }

    /// Advances the `active` sequences with the tokens `toks` (one per active sequence).
    /// Returns the logits (active.len() x vocab).
    pub fn step(&mut self, active: &[usize], toks: &[u32], parallel: bool) -> Vec<f32> {
        let m = self.m;
        let hd = m.h.hidden;
        let t = active.len();
        let mut x = vec![0f32; t * hd];
        for (i, &tk) in toks.iter().enumerate() {
            m.embed(tk, &mut x[i * hd..(i + 1) * hd]);
        }
        let pos: Vec<usize> = active.iter().map(|&a| self.pos[a]).collect();
        // mutable (disjoint) references to the active caches; `active` is sorted
        let mut is_active = vec![false; self.caches.len()];
        for &a in active {
            is_active[a] = true;
        }
        let mut refs: Vec<&mut KvCache> =
            self.caches.iter_mut().enumerate().filter(|(i, _)| is_active[*i]).map(|(_, c)| c).collect();
        let seq_of: Vec<usize> = (0..t).collect();
        for l in 0..m.h.layers {
            m.layer(l, &mut x, &pos, &mut refs, &seq_of, parallel);
        }
        let mut logits = vec![0f32; t * m.h.vocab];
        m.head(&x, t, &mut logits, parallel);
        for &a in active {
            self.pos[a] += 1;
        }
        logits
    }
}

fn fnv_full(b: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &x in b {
        h ^= x as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h ^ b.len() as u64
}

fn fnv(b: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &x in b.iter().step_by(97) {
        h ^= x as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h ^ b.len() as u64
}
