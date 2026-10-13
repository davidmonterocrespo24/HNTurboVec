//! Matemática DETERMINISTA para el LM: solo operaciones IEEE básicas (+ - * / sqrt) en un orden
//! fijo, sin libm (exp/erf propias). Mismo binario => mismos bits en cualquier CPU x86-64.

const LN2_HI: f32 = 0.693_145_75;
const LN2_LO: f32 = 1.428_606_8e-6;
const MAGIC: f32 = 12_582_912.0; // 1.5 * 2^23: redondeo al par más cercano sumando/restando

/// Redondeo al entero más cercano (par en empates) para |x| < 2^22, sin libm.
#[inline(always)]
pub fn round_even(x: f32) -> f32 {
    (x + MAGIC) - MAGIC
}

/// e^x determinista (error relativo ~2e-7). Para x < -87 devuelve 0.
#[inline(always)]
pub fn exp(x: f32) -> f32 {
    let x = if x < -87.0 { -87.0 } else if x > 88.0 { 88.0 } else { x };
    let k = round_even(x * std::f32::consts::LOG2_E);
    let r = (x - k * LN2_HI) - k * LN2_LO;
    let p = 1.0 + r * (1.0 + r * (0.5 + r * (0.166_666_67 + r * (0.041_666_668 + r * (0.008_333_334 + r * 0.001_388_889)))));
    let bits = ((k as i32 + 127) as u32) << 23;
    p * f32::from_bits(bits)
}

/// erf(x) (Abramowitz-Stegun 7.1.26, error < 1.5e-7) con exp propia.
#[inline(always)]
pub fn erf(x: f32) -> f32 {
    let s = if x < 0.0 { -1.0 } else { 1.0 };
    let a = x * s;
    let t = 1.0 / (1.0 + 0.327_591_1 * a);
    let y = 1.0 - ((((1.061_405_4 * t - 1.453_152_1) * t + 1.421_413_8) * t - 0.284_496_74) * t + 0.254_829_6) * t * exp(-a * a);
    s * y
}

/// GELU exacta (con erf): 0.5 x (1 + erf(x / sqrt 2)).
#[inline(always)]
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// Suma con 8 acumuladores y reducción en árbol fijo (vectorizable y determinista).
#[inline]
pub fn sum(x: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let chunks = x.chunks_exact(8);
    let rem = chunks.remainder();
    for c in chunks {
        for j in 0..8 {
            acc[j] += c[j];
        }
    }
    let mut tail = 0f32;
    for &v in rem {
        tail += v;
    }
    ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7])) + tail
}

/// Producto escalar f32 determinista (8 carriles, orden fijo).
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let n = a.len() / 8 * 8;
    let mut i = 0;
    while i < n {
        for j in 0..8 {
            acc[j] += a[i + j] * b[i + j];
        }
        i += 8;
    }
    let mut tail = 0f32;
    while i < a.len() {
        tail += a[i] * b[i];
        i += 1;
    }
    ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7])) + tail
}

/// LayerNorm determinista.
pub fn layer_norm(x: &[f32], w: &[f32], b: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len() as f32;
    let mean = sum(x) / n;
    let mut d = vec![0f32; x.len()];
    for i in 0..x.len() {
        d[i] = x[i] - mean;
    }
    let var = dot(&d, &d) / n;
    let inv = 1.0 / (var + eps).sqrt();
    for i in 0..x.len() {
        out[i] = d[i] * inv * w[i] + b[i];
    }
}

/// Logits -> frecuencias enteras acumuladas (total ~2^30, todas >= 1). Determinista.
pub const FREQ_TOTAL: u32 = 1 << 30;
pub fn logits_to_cum(logits: &[f32], cum: &mut Vec<u64>) {
    let v = logits.len();
    let mut m = f32::NEG_INFINITY;
    for &l in logits {
        if l > m {
            m = l;
        }
    }
    thread_local! {
        static E: std::cell::RefCell<(Vec<f32>, Vec<u32>)> = const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
    }
    E.with(|buf| {
        let (e, f) = &mut *buf.borrow_mut();
        e.resize(v, 0.0);
        f.resize(v, 0);
        for i in 0..v {
            e[i] = exp(logits[i] - m);
        }
        let s = sum(e);
        let scale = (FREQ_TOTAL - v as u32) as f32 / s;
        // valores < 2^30: la conversión vía i32 es exacta y vectorizable
        for i in 0..v {
            f[i] = 1 + (e[i] * scale) as i32 as u32;
        }
        cum.clear();
        cum.reserve(v + 1);
        cum.push(0);
        let mut acc = 0u64;
        for &x in f.iter() {
            acc += x as u64;
            cum.push(acc);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exp_erf_accuracy() {
        for i in -800..=200 {
            let x = i as f32 * 0.1;
            let r = (x as f64).exp() as f32;
            let e = exp(x);
            assert!(((e - r) / r).abs() < 1e-6, "exp({x}) = {e} vs {r}");
        }
        for i in -40..=40 {
            let x = i as f32 * 0.1;
            // erf de referencia por serie (suficiente para |x|<=4)
            let mut s = 0f64;
            let xx = x as f64;
            let mut term = xx;
            for n in 0..200 {
                s += term / (2 * n + 1) as f64;
                term *= -xx * xx / (n + 1) as f64;
            }
            let r = (s * 2.0 / std::f64::consts::PI.sqrt()) as f32;
            assert!((erf(x) - r).abs() < 3e-7, "erf({x}) = {} vs {r}", erf(x));
        }
    }
}
