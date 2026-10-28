//! DETERMINISTIC math for the LM: only basic IEEE operations (+ - * / sqrt) in a fixed order,
//! no libm (our own exp/erf). Same binary => same bits on any x86-64 CPU.

const LN2_HI: f32 = 0.693_145_75;
const LN2_LO: f32 = 1.428_606_8e-6;
const MAGIC: f32 = 12_582_912.0; // 1.5 * 2^23: adding and then subtracting it rounds to nearest even

/// Rounds to the nearest integer (ties to even) for |x| < 2^22, without libm.
#[inline(always)]
pub fn round_even(x: f32) -> f32 {
    (x + MAGIC) - MAGIC
}

/// Deterministic e^x (relative error ~2e-7). For x < -87 it returns 0.
#[inline(always)]
pub fn exp(x: f32) -> f32 {
    let x = if x < -87.0 { -87.0 } else if x > 88.0 { 88.0 } else { x };
    let k = round_even(x * std::f32::consts::LOG2_E);
    let r = (x - k * LN2_HI) - k * LN2_LO;
    let p = 1.0 + r * (1.0 + r * (0.5 + r * (0.166_666_67 + r * (0.041_666_668 + r * (0.008_333_334 + r * 0.001_388_889)))));
    let bits = ((k as i32 + 127) as u32) << 23;
    p * f32::from_bits(bits)
}

/// erf(x) (Abramowitz-Stegun 7.1.26, error < 1.5e-7) with our own exp.
#[inline(always)]
pub fn erf(x: f32) -> f32 {
    let s = if x < 0.0 { -1.0 } else { 1.0 };
    let a = x * s;
    let t = 1.0 / (1.0 + 0.327_591_1 * a);
    let y = 1.0 - ((((1.061_405_4 * t - 1.453_152_1) * t + 1.421_413_8) * t - 0.284_496_74) * t + 0.254_829_6) * t * exp(-a * a);
    s * y
}

/// Exact GELU (with erf): 0.5 x (1 + erf(x / sqrt 2)).
#[inline(always)]
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// Sum with 8 accumulators and a fixed tree reduction (vectorizable and deterministic).
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

/// Deterministic f32 dot product (8 lanes, fixed order).
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    // same order of operations as the original version (lane j sums elements j, j+8, ...),
    // written with chunks_exact so LLVM vectorizes it without bounds checks
    let mut acc = [0f32; 8];
    let ca = a.chunks_exact(8);
    let cb = b[..a.len()].chunks_exact(8);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for j in 0..8 {
            acc[j] += x[j] * y[j];
        }
    }
    let mut tail = 0f32;
    for (x, y) in ra.iter().zip(rb) {
        tail += x * y;
    }
    ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7])) + tail
}

/// Deterministic LayerNorm.
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

thread_local! {
    /// sampling scratch of this thread (exponentials, frequencies)
    static SCRATCH: std::cell::RefCell<(Vec<f32>, Vec<u32>)> = const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

/// Frees this thread's sampling scratch when dropped (for host threads, which should not keep it).
pub struct ScratchRelease;

impl Drop for ScratchRelease {
    fn drop(&mut self) {
        let _ = SCRATCH.try_with(|b| *b.borrow_mut() = (Vec::new(), Vec::new()));
    }
}

/// Logits -> cumulative integer frequencies (total ~2^30, all >= 1). Deterministic.
pub const FREQ_TOTAL: u32 = 1 << 30;
pub fn logits_to_cum(logits: &[f32], cum: &mut Vec<u64>) {
    let v = logits.len();
    let mut m = f32::NEG_INFINITY;
    for &l in logits {
        if l > m {
            m = l;
        }
    }
    SCRATCH.with(|buf| {
        let (e, f) = &mut *buf.borrow_mut();
        e.resize(v, 0.0);
        f.resize(v, 0);
        for i in 0..v {
            e[i] = exp(logits[i] - m);
        }
        let s = sum(e);
        let scale = (FREQ_TOTAL - v as u32) as f32 / s;
        // values < 2^30: converting through i32 is exact and vectorizable
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
            // reference erf from its series (enough for |x|<=4)
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
