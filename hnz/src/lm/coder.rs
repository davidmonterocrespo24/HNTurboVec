//! Codificador aritmético binario (estilo LZMA, rango de 32 bits, probabilidades de 16 bits).
//! Un token se codifica como 16 decisiones binarias sobre el árbol del vocabulario (0..65536),
//! con P(bit) = masa del subárbol izquierdo / masa del nodo, calculada con ENTEROS a partir de
//! las frecuencias acumuladas del LM -> totalmente determinista.
//! Al terminar se elige el valor final con más bytes a cero y se recortan (el decodificador
//! rellena con ceros), y se omite el primer byte (siempre 0 en este esquema).

const TOP: u32 = 1 << 24;
pub const VOCAB_BITS: u32 = 16;

pub struct Encoder {
    low: u64,
    range: u32,
    cache: u8,
    cache_size: u64,
    out: Vec<u8>,
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Encoder {
    pub fn new() -> Self {
        Self { low: 0, range: u32::MAX, cache: 0, cache_size: 1, out: Vec::new() }
    }

    fn shift_low(&mut self) {
        if (self.low as u32) < 0xFF00_0000 || (self.low >> 32) != 0 {
            let carry = (self.low >> 32) as u8;
            let mut temp = self.cache;
            loop {
                self.out.push(temp.wrapping_add(carry));
                temp = 0xFF;
                self.cache_size -= 1;
                if self.cache_size == 0 {
                    break;
                }
            }
            self.cache = ((self.low >> 24) & 0xFF) as u8;
        }
        self.cache_size += 1;
        self.low = (self.low & 0x00FF_FFFF) << 8;
    }

    /// Codifica `bit` con probabilidad de 0 = p0/65536 (1..=65535).
    #[inline]
    pub fn bit(&mut self, p0: u32, bit: u32) {
        let bound = (self.range >> 16) * p0;
        if bit == 0 {
            self.range = bound;
        } else {
            self.low += bound as u64;
            self.range -= bound;
        }
        while self.range < TOP {
            self.range <<= 8;
            self.shift_low();
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        // valor dentro de [low, low+range) con más bits bajos a cero
        let hi = self.low + self.range as u64 - 1;
        for shift in (0..=32).rev().step_by(8) {
            let mask = (1u64 << shift) - 1;
            let v = (self.low + mask) & !mask;
            if v <= hi {
                self.low = v;
                break;
            }
        }
        for _ in 0..5 {
            self.shift_low();
        }
        let mut out = self.out;
        out.remove(0); // primer byte siempre 0
        while out.last() == Some(&0) {
            out.pop();
        }
        out
    }
}

pub struct Decoder<'a> {
    code: u32,
    range: u32,
    data: &'a [u8],
    pos: usize,
}

impl<'a> Decoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let mut d = Self { code: 0, range: u32::MAX, data, pos: 0 };
        for _ in 0..4 {
            d.code = (d.code << 8) | d.next() as u32;
        }
        d
    }

    #[inline]
    fn next(&mut self) -> u8 {
        let b = self.data.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        b
    }

    #[inline]
    pub fn bit(&mut self, p0: u32) -> u32 {
        let bound = (self.range >> 16) * p0;
        let b = if self.code < bound {
            self.range = bound;
            0
        } else {
            self.code -= bound;
            self.range -= bound;
            1
        };
        while self.range < TOP {
            self.range <<= 8;
            self.code = (self.code << 8) | self.next() as u32;
        }
        b
    }
}

#[inline]
fn mass(cum: &[u64], lo: usize, hi: usize) -> u64 {
    let v = cum.len() - 1;
    cum[hi.min(v)] - cum[lo.min(v)]
}

#[inline]
fn p0_of(m0: u64, m: u64) -> u32 {
    let p = ((m0 as u128) << 16) / m as u128;
    (p as u32).clamp(1, 65535)
}

/// Codifica el token `tok` dada la distribución acumulada `cum` (len = vocab+1).
pub fn encode_token(e: &mut Encoder, cum: &[u64], tok: u32) {
    let mut lo = 0usize;
    let mut size = 1usize << VOCAB_BITS;
    for _ in 0..VOCAB_BITS {
        size >>= 1;
        let mid = lo + size;
        let m0 = mass(cum, lo, mid);
        let m1 = mass(cum, mid, mid + size);
        let bit = (tok as usize >= mid) as u32;
        if m0 != 0 && m1 != 0 {
            e.bit(p0_of(m0, m0 + m1), bit);
        }
        if bit == 1 {
            lo = mid;
        }
    }
}

pub fn decode_token(d: &mut Decoder, cum: &[u64]) -> u32 {
    let mut lo = 0usize;
    let mut size = 1usize << VOCAB_BITS;
    for _ in 0..VOCAB_BITS {
        size >>= 1;
        let mid = lo + size;
        let m0 = mass(cum, lo, mid);
        let m1 = mass(cum, mid, mid + size);
        let bit = if m0 == 0 {
            1
        } else if m1 == 0 {
            0
        } else {
            d.bit(p0_of(m0, m0 + m1))
        };
        if bit == 1 {
            lo = mid;
        }
    }
    lo as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip_random_distributions() {
        let mut x: u64 = 12345;
        let mut rnd = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let v = 50304usize;
        let mut dists = Vec::new();
        let mut toks = Vec::new();
        for _ in 0..300 {
            let mut cum = vec![0u64; v + 1];
            let peak = (rnd() % v as u64) as usize;
            for i in 0..v {
                let f = if i == peak { 1 << 28 } else { 1 + rnd() % 64 };
                cum[i + 1] = cum[i] + f;
            }
            let tok = if rnd() % 4 == 0 { (rnd() % v as u64) as u32 } else { peak as u32 };
            dists.push(cum);
            toks.push(tok);
        }
        let mut e = Encoder::new();
        for (c, &t) in dists.iter().zip(&toks) {
            encode_token(&mut e, c, t);
        }
        let bytes = e.finish();
        let mut d = Decoder::new(&bytes);
        for (c, &t) in dists.iter().zip(&toks) {
            assert_eq!(decode_token(&mut d, c), t);
        }
        // coste cercano a la entropía: ~0.2 bits para el pico, ~18+ bits para los demás
        assert!(bytes.len() < 300 * 3, "{} bytes", bytes.len());
    }

    #[test]
    fn empty_and_tiny() {
        let e = Encoder::new();
        let b = e.finish();
        assert!(b.len() <= 4);
    }
}
