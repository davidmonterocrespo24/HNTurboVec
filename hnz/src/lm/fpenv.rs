//! Floating-point environment for the LM. Its arithmetic must match, bit for bit, what it computed
//! when compressing, or lossless decompression breaks. The SSE/AVX control register (MXCSR) is per
//! thread, and a host process (Python, a backend) may call hnz on a thread where some library
//! turned on flush-to-zero / denormals-are-zero or changed the rounding mode (e.g. code built with
//! fast-math). The LM entry points therefore set the default state for their duration and give the
//! caller its own back; the LM pool's threads set it once at start.

/// Default MXCSR: all exceptions masked, round to nearest, no FTZ/DAZ.
const DEFAULT: u32 = 0x1F80;
/// Control bits (DAZ, exception masks, rounding, FTZ); bits 0-5 are sticky exception flags.
const CONTROL: u32 = 0xFFC0;

#[cfg(target_arch = "x86_64")]
pub fn get() -> u32 {
    let mut x: u32 = 0;
    // SAFETY: stores the 32-bit MXCSR into a local
    unsafe { std::arch::asm!("stmxcsr [{p}]", p = in(reg) &mut x as *mut u32, options(nostack, preserves_flags)) };
    x
}

#[cfg(target_arch = "x86_64")]
fn set(x: u32) {
    // SAFETY: loads a valid MXCSR value (only reserved bits would fault; we never set them)
    unsafe { std::arch::asm!("ldmxcsr [{p}]", p = in(reg) &x as *const u32, options(nostack, preserves_flags)) };
}

#[cfg(not(target_arch = "x86_64"))]
pub fn get() -> u32 {
    DEFAULT
}

#[cfg(not(target_arch = "x86_64"))]
fn set(_: u32) {}

/// Sets the default control state on this thread (for threads the LM owns).
pub fn set_default() {
    let cur = get();
    if cur & CONTROL != DEFAULT {
        set((cur & !CONTROL) | DEFAULT);
    }
}

/// Default control state while alive; restores the caller's control bits when dropped.
pub struct FpGuard {
    old: u32,
}

impl FpGuard {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        let old = get();
        set_default();
        FpGuard { old }
    }
}

impl Drop for FpGuard {
    fn drop(&mut self) {
        let cur = get();
        if cur & CONTROL != self.old & CONTROL {
            set((cur & !CONTROL) | (self.old & CONTROL));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hint::black_box;

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn guard_restores_default_arithmetic() {
        let tiny = || black_box(1e-30f32) * black_box(1e-10f32); // a denormal (~1e-40)
        let exact = tiny();
        assert!(exact > 0.0);
        let before = get();
        // a host library turned on flush-to-zero + denormals-are-zero on this thread
        set((before & !CONTROL) | DEFAULT | 0x8040);
        assert_eq!(tiny(), 0.0);
        {
            let _g = FpGuard::new();
            assert_eq!(get() & CONTROL, DEFAULT);
            assert_eq!(tiny().to_bits(), exact.to_bits());
        }
        // the caller gets its own state back
        assert_eq!(get() & CONTROL, DEFAULT | 0x8040);
        set(before);
        assert_eq!(tiny().to_bits(), exact.to_bits());
    }
}
