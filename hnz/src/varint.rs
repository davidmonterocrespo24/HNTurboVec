//! Enteros de longitud variable (LEB128) y zigzag para deltas con signo.

#[inline]
pub fn put(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

#[inline]
pub fn get(buf: &[u8], pos: &mut usize) -> u64 {
    let mut v = 0u64;
    let mut shift = 0;
    loop {
        let b = buf[*pos];
        *pos += 1;
        v |= ((b & 0x7f) as u64) << shift;
        if b < 0x80 {
            return v;
        }
        shift += 7;
    }
}

#[inline]
pub fn zz(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

#[inline]
pub fn unzz(u: u64) -> i64 {
    ((u >> 1) as i64) ^ -((u & 1) as i64)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip() {
        let vals = [0u64, 1, 127, 128, 300, u32::MAX as u64, u64::MAX];
        let mut b = Vec::new();
        for &v in &vals {
            put(&mut b, v);
        }
        let mut p = 0;
        for &v in &vals {
            assert_eq!(get(&b, &mut p), v);
        }
        for v in [-5i64, 0, 7, i64::MIN / 2, i64::MAX / 2] {
            assert_eq!(unzz(zz(v)), v);
        }
    }
}
