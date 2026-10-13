//! Matmul int8 x int8 -> i32 EXACTA (la suma de enteros no depende del orden), así que cualquier
//! implementación (escalar, AVX2, AVX-VNNI, varios hilos) da exactamente el mismo resultado.
//! - AVX-VNNI: `vpdpbusd` (u8 x i8, 32 MACs/instrucción). Las activaciones se desplazan +128
//!   (u8) y se corrige restando 128·Σw de la fila (exacto).
//! - AVX2: extensión a i16 + `vpmaddwd`.
//! - escalar: referencia.

use super::math::round_even;
use std::sync::OnceLock;

/// Cuantiza una fila de activaciones a int8 por token. Devuelve la escala (x ≈ q * escala).
#[inline]
pub fn quant_row(x: &[f32], q: &mut [i8]) -> f32 {
    let mut amax = 0f32;
    for &v in x {
        let a = v.abs();
        if a > amax {
            amax = a;
        }
    }
    if amax == 0.0 {
        q.fill(0);
        return 0.0;
    }
    let inv = 127.0 / amax;
    for (o, &v) in q.iter_mut().zip(x) {
        let r = round_even(v * inv);
        *o = r.clamp(-127.0, 127.0) as i8;
    }
    amax / 127.0
}

/// Matriz int8 por filas con escala por fila y suma de cada fila (para la corrección VNNI).
pub struct Q8<'a> {
    pub rows: usize,
    pub cols: usize,
    pub scales: &'a [f32],
    pub rowsum: &'a [i32],
    pub data: &'a [i8],
}

impl<'a> Q8<'a> {
    #[inline]
    pub fn row(&self, n: usize) -> &[i8] {
        &self.data[n * self.cols..(n + 1) * self.cols]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    Vnni,
    Avx2,
    Scalar,
}

pub fn kernel() -> Kernel {
    static K: OnceLock<Kernel> = OnceLock::new();
    *K.get_or_init(|| {
        if std::env::var("HNZ_KERNEL").as_deref() == Ok("scalar") {
            return Kernel::Scalar;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if std::env::var("HNZ_KERNEL").as_deref() != Ok("avx2") && is_x86_feature_detected!("avxvnni") && is_x86_feature_detected!("avx2") {
                return Kernel::Vnni;
            }
            if is_x86_feature_detected!("avx2") {
                return Kernel::Avx2;
            }
        }
        Kernel::Scalar
    })
}

#[inline]
fn dot_scalar(a: &[i8], b: &[i8]) -> i32 {
    a.iter().zip(b).map(|(&x, &y)| x as i32 * y as i32).sum()
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    #[inline(always)]
    unsafe fn hsum(v: __m256i) -> i32 {
        let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
        _mm_cvtsi128_si32(s)
    }

    /// 4 productos escalares a la vez: w · x[0..4] (u8 desplazadas). `k` múltiplo de 32.
    #[target_feature(enable = "avx2,avxvnni")]
    pub unsafe fn dot4_vnni(w: &[i8], x: [&[u8]; 4], out: &mut [i32; 4]) {
        let k = w.len();
        let (mut a0, mut a1, mut a2, mut a3) = (_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256());
        let mut i = 0;
        while i < k {
            let vw = _mm256_loadu_si256(w.as_ptr().add(i) as *const __m256i);
            a0 = _mm256_dpbusd_avx_epi32(a0, _mm256_loadu_si256(x[0].as_ptr().add(i) as *const __m256i), vw);
            a1 = _mm256_dpbusd_avx_epi32(a1, _mm256_loadu_si256(x[1].as_ptr().add(i) as *const __m256i), vw);
            a2 = _mm256_dpbusd_avx_epi32(a2, _mm256_loadu_si256(x[2].as_ptr().add(i) as *const __m256i), vw);
            a3 = _mm256_dpbusd_avx_epi32(a3, _mm256_loadu_si256(x[3].as_ptr().add(i) as *const __m256i), vw);
            i += 32;
        }
        *out = [hsum(a0), hsum(a1), hsum(a2), hsum(a3)];
    }

    #[target_feature(enable = "avx2,avxvnni")]
    pub unsafe fn dot1_vnni(w: &[i8], x: &[u8]) -> i32 {
        let k = w.len();
        let mut a = _mm256_setzero_si256();
        let mut i = 0;
        while i < k {
            a = _mm256_dpbusd_avx_epi32(
                a,
                _mm256_loadu_si256(x.as_ptr().add(i) as *const __m256i),
                _mm256_loadu_si256(w.as_ptr().add(i) as *const __m256i),
            );
            i += 32;
        }
        hsum(a)
    }

    /// AVX2: w · x (ambos i8). `k` múltiplo de 32.
    #[target_feature(enable = "avx2")]
    pub unsafe fn dot1_avx2(w: &[i8], x: &[i8]) -> i32 {
        let k = w.len();
        let mut acc = _mm256_setzero_si256();
        let mut i = 0;
        while i < k {
            let va = _mm256_loadu_si256(w.as_ptr().add(i) as *const __m256i);
            let vb = _mm256_loadu_si256(x.as_ptr().add(i) as *const __m256i);
            let a_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
            let a_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
            let b_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
            let b_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a_lo, b_lo));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a_hi, b_hi));
            i += 32;
        }
        hsum(acc)
    }

    /// Suma horizontal de 8 vectores a la vez -> [Σa0, …, Σa7].
    #[inline(always)]
    unsafe fn hsum8(a: &[__m256i; 8]) -> [i32; 8] {
        let t0 = _mm256_hadd_epi32(a[0], a[1]);
        let t1 = _mm256_hadd_epi32(a[2], a[3]);
        let t2 = _mm256_hadd_epi32(a[4], a[5]);
        let t3 = _mm256_hadd_epi32(a[6], a[7]);
        let u0 = _mm256_hadd_epi32(t0, t1);
        let u1 = _mm256_hadd_epi32(t2, t3);
        let r = _mm256_add_epi32(_mm256_permute2x128_si256(u0, u1, 0x20), _mm256_permute2x128_si256(u0, u1, 0x31));
        let mut o = [0i32; 8];
        _mm256_storeu_si256(o.as_mut_ptr() as *mut __m256i, r);
        o
    }

    /// 2 filas x 4 tokens (8 acumuladores): out[0..4] = w0·x[0..4], out[4..8] = w1·x[0..4].
    #[target_feature(enable = "avx2,avxvnni")]
    pub unsafe fn dot2x4_vnni(w0: &[i8], w1: &[i8], x: [&[u8]; 4], out: &mut [i32; 8]) {
        let k = w0.len();
        let mut a = [_mm256_setzero_si256(); 8];
        let mut i = 0;
        while i < k {
            let x0 = _mm256_loadu_si256(x[0].as_ptr().add(i) as *const __m256i);
            let x1 = _mm256_loadu_si256(x[1].as_ptr().add(i) as *const __m256i);
            let x2 = _mm256_loadu_si256(x[2].as_ptr().add(i) as *const __m256i);
            let x3 = _mm256_loadu_si256(x[3].as_ptr().add(i) as *const __m256i);
            let v0 = _mm256_loadu_si256(w0.as_ptr().add(i) as *const __m256i);
            let v1 = _mm256_loadu_si256(w1.as_ptr().add(i) as *const __m256i);
            a[0] = _mm256_dpbusd_avx_epi32(a[0], x0, v0);
            a[1] = _mm256_dpbusd_avx_epi32(a[1], x1, v0);
            a[2] = _mm256_dpbusd_avx_epi32(a[2], x2, v0);
            a[3] = _mm256_dpbusd_avx_epi32(a[3], x3, v0);
            a[4] = _mm256_dpbusd_avx_epi32(a[4], x0, v1);
            a[5] = _mm256_dpbusd_avx_epi32(a[5], x1, v1);
            a[6] = _mm256_dpbusd_avx_epi32(a[6], x2, v1);
            a[7] = _mm256_dpbusd_avx_epi32(a[7], x3, v1);
            i += 32;
        }
        *out = hsum8(&a);
    }
}

/// Calcula los acumuladores enteros de las filas n0..n1 de W contra los t tokens.
/// dst[(r - n0) * t + ti] = Σ_k w[r][k] · xq[ti][k]
fn rows_acc(w: &Q8, xq: &[i8], xu: &[u8], t: usize, n0: usize, n1: usize, dst: &mut [i32]) {
    let k = w.cols;
    let kern = if k % 32 == 0 { kernel() } else { Kernel::Scalar };
    // AVX-VNNI con bloques de 2 filas x 4 tokens
    #[cfg(target_arch = "x86_64")]
    if kern == Kernel::Vnni && std::env::var_os("HNZ_KERNEL_1ROW").is_none() {
        let xs = |ti: usize| &xu[ti * k..(ti + 1) * k];
        let mut r = n0;
        while r + 1 < n1 {
            let (w0, w1) = (w.row(r), w.row(r + 1));
            let (c0, c1) = (128 * w.rowsum[r], 128 * w.rowsum[r + 1]);
            let (b0, b1) = ((r - n0) * t, (r + 1 - n0) * t);
            let mut ti = 0;
            let mut o = [0i32; 8];
            while ti + 4 <= t {
                unsafe { x86::dot2x4_vnni(w0, w1, [xs(ti), xs(ti + 1), xs(ti + 2), xs(ti + 3)], &mut o) };
                for j in 0..4 {
                    dst[b0 + ti + j] = o[j] - c0;
                    dst[b1 + ti + j] = o[4 + j] - c1;
                }
                ti += 4;
            }
            while ti < t {
                dst[b0 + ti] = unsafe { x86::dot1_vnni(w0, xs(ti)) } - c0;
                dst[b1 + ti] = unsafe { x86::dot1_vnni(w1, xs(ti)) } - c1;
                ti += 1;
            }
            r += 2;
        }
        if r < n1 {
            rows_acc_1(w, xq, xu, t, r, r + 1, n0, kern, dst);
        }
        return;
    }
    rows_acc_1(w, xq, xu, t, n0, n1, n0, kern, dst);
}

/// Versión fila a fila (escalar, AVX2, o VNNI 1x4). `base` = primera fila de `dst`.
#[allow(clippy::too_many_arguments)]
fn rows_acc_1(w: &Q8, xq: &[i8], xu: &[u8], t: usize, n0: usize, n1: usize, base: usize, kern: Kernel, dst: &mut [i32]) {
    let k = w.cols;
    for r in n0..n1 {
        let wr = w.row(r);
        let d = &mut dst[(r - base) * t..(r - base + 1) * t];
        match kern {
            #[cfg(target_arch = "x86_64")]
            Kernel::Vnni => {
                let corr = 128 * w.rowsum[r];
                let mut ti = 0;
                let mut o = [0i32; 4];
                while ti + 4 <= t {
                    unsafe {
                        x86::dot4_vnni(
                            wr,
                            [&xu[ti * k..(ti + 1) * k], &xu[(ti + 1) * k..(ti + 2) * k], &xu[(ti + 2) * k..(ti + 3) * k], &xu[(ti + 3) * k..(ti + 4) * k]],
                            &mut o,
                        )
                    };
                    for j in 0..4 {
                        d[ti + j] = o[j] - corr;
                    }
                    ti += 4;
                }
                while ti < t {
                    d[ti] = unsafe { x86::dot1_vnni(wr, &xu[ti * k..(ti + 1) * k]) } - corr;
                    ti += 1;
                }
            }
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx2 => {
                for ti in 0..t {
                    d[ti] = unsafe { x86::dot1_avx2(wr, &xq[ti * k..(ti + 1) * k]) };
                }
            }
            _ => {
                for ti in 0..t {
                    d[ti] = dot_scalar(wr, &xq[ti * k..(ti + 1) * k]);
                }
            }
        }
    }
}

pub struct SendPtr<T>(pub *mut T);
impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SendPtr<T> {}
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> SendPtr<T> {
    #[inline]
    pub fn get(self) -> *mut T {
        self.0
    }
}

/// y[t][n] = (Σ_k xq[t][k]·w[n][k]) · sx[t] · sw[n] + bias[n]
/// `xq`: T filas de `cols` int8; `out`: T x rows. `parallel` reparte filas de W entre hilos.
pub fn matmul(w: &Q8, xq: &[i8], sx: &[f32], bias: Option<&[f32]>, out: &mut [f32], parallel: bool) {
    let t = sx.len();
    let n = w.rows;
    debug_assert_eq!(xq.len(), t * w.cols);
    debug_assert_eq!(out.len(), t * n);
    let xu: Vec<u8> = if kernel() == Kernel::Vnni { xq.iter().map(|&v| (v as i16 + 128) as u8).collect() } else { Vec::new() };
    let mut acc = vec![0i32; n * t];
    if parallel && n * t * w.cols > 1 << 18 {
        let p = super::pool::pool();
        let chunk = (n / (p.threads.max(1) * 3)).max(16);
        let nchunks = n.div_ceil(chunk);
        let base = SendPtr(acc.as_mut_ptr());
        p.run(nchunks, &|ci| {
            let n0 = ci * chunk;
            let n1 = (n0 + chunk).min(n);
            // SAFETY: cada tarea escribe un trozo disjunto [n0*t, n1*t) de `acc`
            let d = unsafe { std::slice::from_raw_parts_mut(base.get().add(n0 * t), (n1 - n0) * t) };
            rows_acc(w, xq, &xu, t, n0, n1, d);
        });
    } else {
        rows_acc(w, xq, &xu, t, 0, n, &mut acc);
    }
    for r in 0..n {
        let sw = w.scales[r];
        let b = bias.map_or(0.0, |b| b[r]);
        for ti in 0..t {
            out[ti * n + r] = acc[r * t + ti] as f32 * (sx[ti] * sw) + b;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn kernels_agree_exactly() {
        let (n, k, t) = (37, 512, 7);
        let mut x: u64 = 99;
        let mut rnd = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % 255) as i32 - 127
        };
        let w: Vec<i8> = (0..n * k).map(|_| rnd() as i8).collect();
        let xs: Vec<i8> = (0..t * k).map(|_| rnd() as i8).collect();
        let xu: Vec<u8> = xs.iter().map(|&v| (v as i16 + 128) as u8).collect();
        let rowsum: Vec<i32> = (0..n).map(|r| w[r * k..(r + 1) * k].iter().map(|&v| v as i32).sum()).collect();
        let scales = vec![1f32; n];
        let q = Q8 { rows: n, cols: k, scales: &scales, rowsum: &rowsum, data: &w };
        let mut reference = vec![0i32; n * t];
        for r in 0..n {
            for ti in 0..t {
                reference[r * t + ti] = dot_scalar(&w[r * k..(r + 1) * k], &xs[ti * k..(ti + 1) * k]);
            }
        }
        let mut got = vec![0i32; n * t];
        rows_acc(&q, &xs, &xu, t, 0, n, &mut got);
        assert_eq!(got, reference, "kernel {:?}", kernel());
    }
}
