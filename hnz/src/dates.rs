//! Fechas sin dependencias: civil <-> días desde epoch (algoritmo de H. Hinnant).

pub fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m as i64 + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// "2015", "2015-06", "2015-06-01" -> epoch (inicio del periodo). `end=true` -> inicio del
/// periodo SIGUIENTE (para rangos inclusivos: --to 2015 incluye todo 2015).
pub fn parse(s: &str, end: bool) -> Option<i64> {
    let p: Vec<&str> = s.trim().split('-').collect();
    let y: i64 = p.first()?.parse().ok()?;
    let m: u32 = p.get(1).map_or(Some(1), |x| x.parse().ok())?;
    let d: u32 = p.get(2).map_or(Some(1), |x| x.parse().ok())?;
    let days = if !end {
        days_from_civil(y, m, d)
    } else {
        match p.len() {
            1 => days_from_civil(y + 1, 1, 1),
            2 => {
                if m == 12 {
                    days_from_civil(y + 1, 1, 1)
                } else {
                    days_from_civil(y, m + 1, 1)
                }
            }
            _ => days_from_civil(y, m, d) + 1,
        }
    };
    Some(days * 86400)
}

pub fn fmt(t: i64) -> String {
    let (y, m, d) = civil_from_days(t.div_euclid(86400));
    let s = t.rem_euclid(86400);
    format!("{y:04}-{m:02}-{d:02} {:02}:{:02}", s / 3600, s % 3600 / 60)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip() {
        for z in [-1000i64, 0, 13000, 19723, 20000] {
            let (y, m, d) = civil_from_days(z);
            assert_eq!(days_from_civil(y, m, d), z);
        }
        assert_eq!(fmt(1704067200), "2024-01-01 00:00");
        assert_eq!(parse("2024", false), Some(1704067200));
        assert_eq!(parse("2023", true), Some(1704067200));
        assert_eq!(parse("2023-12", true), Some(1704067200));
        assert_eq!(parse("2023-12-31", true), Some(1704067200));
    }
}
