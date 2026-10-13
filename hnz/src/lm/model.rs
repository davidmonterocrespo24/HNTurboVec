//! GPT-NeoX (familia Pythia) en int8 (W8A8) con aritmética determinista.
//! - `prefill`: todas las posiciones de una secuencia de una vez (para COMPRIMIR).
//! - `Batch::step`: un token por secuencia para varias secuencias a la vez con KV-cache
//!   (para DESCOMPRIMIR los resultados de una consulta en lote).
//! Ambos caminos ejecutan exactamente las mismas operaciones por posición => mismos logits.

use super::kernels::{matmul, quant_row, Q8};
use super::math::{dot, exp, gelu, layer_norm};
use anyhow::{bail, Context, Result};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Perfil (ns acumulados): [matmul capas, atención, cabeza, resto]
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

/// Matriz int8: escalas por fila (copiadas, alineadas) + posición de los datos en el fichero.
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
    base: usize,
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
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

impl LmModel {
    pub fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read(path).with_context(|| format!("modelo LM {}", path.display()))?;
        if &data[..6] != b"HNZLM1" {
            bail!("no es un .hnzlm");
        }
        let hl = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let h: Header = serde_json::from_slice(&data[12..12 + hl])?;
        if h.arch != "gpt_neox" {
            bail!("arquitectura no soportada: {}", h.arch);
        }
        let base = 12 + hl + ((64 - (12 + hl) % 64) % 64);
        let find = |name: &str| -> Result<TensorInfo> {
            h.tensors.iter().find(|t| t.name == name).cloned().with_context(|| format!("falta tensor {name}"))
        };
        let vecf = |name: &str| -> Result<Vec<f32>> {
            let t = find(name)?;
            Ok(f32s(&data[base + t.offset..base + t.offset + t.len * 4]))
        };
        let q8 = |name: &str| -> Result<QMat> {
            let t = find(name)?;
            if t.kind != "q8" {
                bail!("{name} no es q8");
            }
            let s0 = base + t.offset;
            let d0 = s0 + t.rows * 4;
            let rowsum = (0..t.rows)
                .map(|r| data[d0 + r * t.cols..d0 + (r + 1) * t.cols].iter().map(|&b| b as i8 as i32).sum())
                .collect();
            Ok(QMat { rows: t.rows, cols: t.cols, scales: f32s(&data[s0..d0]), rowsum, off: d0 })
        };
        let mut layers = Vec::new();
        for l in 0..h.layers {
            layers.push(LayerW {
                ln1_w: vecf(&format!("l{l}.ln1.w"))?,
                ln1_b: vecf(&format!("l{l}.ln1.b"))?,
                ln2_w: vecf(&format!("l{l}.ln2.w"))?,
                ln2_b: vecf(&format!("l{l}.ln2.b"))?,
                qkv: q8(&format!("l{l}.qkv.w"))?,
                qkv_b: vecf(&format!("l{l}.qkv.b"))?,
                o: q8(&format!("l{l}.o.w"))?,
                o_b: vecf(&format!("l{l}.o.b"))?,
                fc1: q8(&format!("l{l}.fc1.w"))?,
                fc1_b: vecf(&format!("l{l}.fc1.b"))?,
                fc2: q8(&format!("l{l}.fc2.w"))?,
                fc2_b: vecf(&format!("l{l}.fc2.b"))?,
            });
        }
        let vo_start = base + h.vocab_offsets;
        let vocab_off: Vec<u32> = data[vo_start..vo_start + (h.vocab + 1) * 4]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let vocab_bytes = data[base + h.vocab_bytes..base + h.vocab_bytes + h.vocab_bytes_len].to_vec();
        let id = format!("{}#{:016x}", h.source, fnv(&data));
        Ok(Self {
            embed_in: q8("embed_in")?,
            lnf_w: vecf("lnf.w")?,
            lnf_b: vecf("lnf.b")?,
            embed_out: q8("embed_out")?,
            rope_cos: vecf("rope_cos")?,
            rope_sin: vecf("rope_sin")?,
            layers,
            vocab_off,
            vocab_bytes,
            id,
            h,
            data,
            base,
        })
    }

    fn q8<'a>(&'a self, m: &'a QMat) -> Q8<'a> {
        let bytes = &self.data[m.off..m.off + m.rows * m.cols];
        // u8 -> i8 (misma representación, alineación 1)
        let data = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const i8, bytes.len()) };
        Q8 { rows: m.rows, cols: m.cols, scales: &m.scales, rowsum: &m.rowsum, data }
    }

    /// Bytes de un token (BPE a nivel de byte).
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

    /// Ejecuta una capa sobre `t` filas (tokens) con posiciones `pos[i]`, usando/actualizando la
    /// KV-cache de la secuencia `seq_of[i]`. `x`: t x H (se actualiza in situ).
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
        // --- atención ---
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
            debug_assert_eq!(c.k[l].len(), pos[i] * hd);
            c.k[l].extend_from_slice(&row[hd..2 * hd]);
            c.v[l].extend_from_slice(&row[2 * hd..3 * hd]);
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
                let mut sc = Vec::with_capacity(n);
                let mut mx = f32::NEG_INFINITY;
                for j in 0..n {
                    let s = dot(q, &c.k[l][j * hd + h * hs..j * hd + (h + 1) * hs]) * scale;
                    if s > mx {
                        mx = s;
                    }
                    sc.push(s);
                }
                let mut tot = 0f32;
                for s in sc.iter_mut() {
                    *s = exp(*s - mx);
                    tot += *s;
                }
                let inv = 1.0 / tot;
                for j in 0..n {
                    let p = sc[j] * inv;
                    let v = &c.v[l][j * hd + h * hs..j * hd + (h + 1) * hs];
                    for d in 0..hs {
                        o[d] += p * v[d];
                    }
                }
            };
            let work: usize = pos.iter().map(|&p| p + 1).sum::<usize>() * nh;
            if parallel && work > 2048 {
                let base = super::kernels::SendPtr(att.as_mut_ptr());
                super::pool::pool().run(t * nh, &|task| {
                    let (i, h) = (task / nh, task % nh);
                    // SAFETY: cada (fila, cabeza) escribe su trozo disjunto de `att`
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
        // residual paralelo: h = mlp + attn + h
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

    /// Logits de todas las posiciones de una secuencia (posición i predice el token i+1).
    pub fn prefill(&self, toks: &[u32], parallel: bool) -> Vec<f32> {
        let t = toks.len();
        let hd = self.h.hidden;
        let mut x = vec![0f32; t * hd];
        for (i, &tk) in toks.iter().enumerate() {
            self.embed(tk, &mut x[i * hd..(i + 1) * hd]);
        }
        let pos: Vec<usize> = (0..t).collect();
        let mut cache = KvCache::new(self.h.layers);
        let seq_of = vec![0usize; t];
        // causalidad: la fila i solo ve las claves 0..=i -> procesamos por capas con la cache
        // creciendo fila a fila dentro de `layer` (pos[i] == len antes de añadir).
        for l in 0..self.h.layers {
            let mut refs = [&mut cache];
            self.layer(l, &mut x, &pos, &mut refs, &seq_of, parallel);
        }
        let mut logits = vec![0f32; t * self.h.vocab];
        self.head(&x, t, &mut logits, parallel);
        logits
    }
}

/// KV-cache de una secuencia: por capa, claves y valores (pos x hidden).
pub struct KvCache {
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
}

impl KvCache {
    pub fn new(layers: usize) -> Self {
        Self { k: vec![Vec::new(); layers], v: vec![Vec::new(); layers] }
    }
}

/// Decodificación en lote: varias secuencias avanzan un token por paso.
pub struct Batch<'m> {
    m: &'m LmModel,
    pub caches: Vec<KvCache>,
    pub pos: Vec<usize>,
}

impl<'m> Batch<'m> {
    pub fn new(m: &'m LmModel, n: usize) -> Self {
        Self { m, caches: (0..n).map(|_| KvCache::new(m.h.layers)).collect(), pos: vec![0; n] }
    }

    /// Avanza las secuencias `active` con los tokens `toks` (uno por secuencia activa).
    /// Devuelve los logits (active.len() x vocab).
    pub fn step(&mut self, active: &[usize], toks: &[u32], parallel: bool) -> Vec<f32> {
        let m = self.m;
        let hd = m.h.hidden;
        let t = active.len();
        let mut x = vec![0f32; t * hd];
        for (i, &tk) in toks.iter().enumerate() {
            m.embed(tk, &mut x[i * hd..(i + 1) * hd]);
        }
        let pos: Vec<usize> = active.iter().map(|&a| self.pos[a]).collect();
        // referencias mutables (disjuntas) a las caches activas; `active` va ordenado
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

fn fnv(b: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &x in b.iter().step_by(97) {
        h ^= x as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h ^ b.len() as u64
}
