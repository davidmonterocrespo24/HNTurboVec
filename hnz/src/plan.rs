//! RAM planner (P2): given the `ram` budget, decides which segment parts are loaded into RAM and how
//! much is left for the LM batch, the working memory of running queries and the caches. Order by
//! benefit: vectors (every semantic search scans them whole) > metadata (every result, the edges of
//! a date range, author filters) > BM25 postings > a share for the decompressed-text cache (each hit
//! saves LM work) > compressed text (on disk it only costs one read) > the rest, to the cache.
//! Within each part, the most recent segments first.
use crate::mem::{human, BudgetError, MB};
use serde::Serialize;

/// Parts of a segment that can live in RAM or be read from disk on demand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Part {
    Vectors,
    Meta,
    Postings,
    Text,
}

impl Part {
    pub const ALL: [Part; 4] = [Part::Vectors, Part::Meta, Part::Postings, Part::Text];

    pub fn name(self) -> &'static str {
        match self {
            Part::Vectors => "vectors",
            Part::Meta => "meta",
            Part::Postings => "postings",
            Part::Text => "text",
        }
    }
}

/// A segment part that can be loaded.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub seg: usize,
    pub part: Part,
    pub bytes: u64,
    /// newest date (epoch) in the segment: for the same part, recent segments go first
    pub recency: i64,
}

#[derive(Clone, Debug, Default)]
pub struct Input {
    /// None = no limit
    pub limit: Option<u64>,
    /// models loaded or reserved to be loaded (the embedder, with its load peak)
    pub models: u64,
    /// structures that are always in RAM
    pub fixed: u64,
    pub candidates: Vec<Candidate>,
    /// what the LM batch needs for the longest record (0 = no LM-tier segments)
    pub lm_min: u64,
    /// minimum working memory for queries
    pub query_min: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Plan {
    pub limit: Option<u64>,
    /// the bare minimum: models, fixed structures and the minimums of the pools and the cache
    pub floor: u64,
    /// (segment, part) pairs loaded into RAM
    pub resident: Vec<(usize, Part)>,
    /// cap of the LM batch (KV cache, logits)
    pub lm_pool: u64,
    /// working memory for running queries
    pub query_pool: u64,
    /// caches of decompressed text
    pub cache: u64,
}

/// minimum for the decompressed-text caches
pub const CACHE_MIN: u64 = 16 * MB;
/// the working pools grow up to 5 % of the budget, with this maximum
const POOL_MAX: u64 = 256 * MB;
/// cache size without a budget (ram=None)
const CACHE_UNLIMITED: u64 = 1024 * MB;

fn take(want: u64, avail: &mut u64) -> u64 {
    let x = want.min(*avail);
    *avail -= x;
    x
}

pub fn plan(inp: &Input) -> Result<Plan, BudgetError> {
    let floor = inp.models + inp.fixed + inp.lm_min + inp.query_min + CACHE_MIN;
    let lim = inp.limit.unwrap_or(u64::MAX);
    if floor > lim {
        return Err(BudgetError(format!(
            "ram={} no alcanza: hacen falta al menos {} (modelos {}, índices fijos {}, lote del LM {}, consultas {}, caché {})",
            human(lim),
            human(floor),
            human(inp.models),
            human(inp.fixed),
            human(inp.lm_min),
            human(inp.query_min),
            human(CACHE_MIN)
        )));
    }
    let mut avail = lim - floor;
    // working pools: up to 5 % of the budget each (at most 256 MB)
    let target = |min: u64| (lim / 20).min(POOL_MAX).max(min);
    let lm_pool = if inp.lm_min > 0 { inp.lm_min + take(target(inp.lm_min) - inp.lm_min, &mut avail) } else { 0 };
    let query_pool = inp.query_min + take(target(inp.query_min) - inp.query_min, &mut avail);
    // a small cache share before loading indexes: 5 % (at most 128 MB)
    let mut cache = CACHE_MIN + take((lim / 20).min(128 * MB).saturating_sub(CACHE_MIN), &mut avail);
    let mut order = inp.candidates.clone();
    order.sort_by_key(|c| (c.part, std::cmp::Reverse(c.recency), c.seg));
    let mut resident = Vec::new();
    for c in order.iter().filter(|c| c.part != Part::Text) {
        if c.bytes <= avail {
            avail -= c.bytes;
            resident.push((c.seg, c.part));
        }
    }
    // more cache before the compressed text: up to 25 % of the budget (at most 1 GB)
    cache += take((lim / 4).min(1024 * MB).saturating_sub(cache), &mut avail);
    for c in order.iter().filter(|c| c.part == Part::Text) {
        if c.bytes <= avail {
            avail -= c.bytes;
            resident.push((c.seg, c.part));
        }
    }
    // the rest, to the cache
    let cache = if inp.limit.is_none() { CACHE_UNLIMITED } else { cache + avail };
    Ok(Plan { limit: inp.limit, floor, resident, lm_pool, query_pool, cache })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(limit: Option<u64>) -> Input {
        let mut candidates = Vec::new();
        // 3 segments (0 = the oldest): vectors 100 MB, metadata 10, postings 50, text 200
        for seg in 0..3 {
            for (part, mb) in [(Part::Vectors, 100), (Part::Meta, 10), (Part::Postings, 50), (Part::Text, 200)] {
                candidates.push(Candidate { seg, part, bytes: mb * MB, recency: 1000 + seg as i64 });
            }
        }
        Input { limit, models: 500 * MB, fixed: 20 * MB, candidates, lm_min: 60 * MB, query_min: 30 * MB }
    }

    #[test]
    fn floor_and_budget_error() {
        let i = input(Some(600 * MB));
        let e = plan(&i).unwrap_err();
        assert!(e.0.contains("656 MB"), "{}", e.0);
        // exactly the minimum: nothing in RAM, minimal pools and cache
        let p = plan(&input(Some(626 * MB))).unwrap();
        assert!(p.resident.is_empty());
        assert_eq!((p.lm_pool, p.query_pool, p.cache), (60 * MB, 30 * MB, CACHE_MIN));
    }

    #[test]
    fn priorities_and_totals() {
        // 1 GB: after the minimums (626), the pools (lm up to 51, q up to 51) and the cache (51),
        // the vectors and the metadata of all 3 segments fit, newest first
        let lim = 1024 * MB;
        let p = plan(&input(Some(lim))).unwrap();
        assert_eq!(p.resident[..4], [(2, Part::Vectors), (1, Part::Vectors), (0, Part::Vectors), (2, Part::Meta)], "{:?}", p.resident);
        assert!(!p.resident.contains(&(0, Part::Text)));
        let used: u64 = p.resident.iter().map(|&(s, part)| input(None).candidates.iter().find(|c| c.seg == s && c.part == part).unwrap().bytes).sum();
        // everything adds up to the limit
        assert_eq!(500 * MB + 20 * MB + p.lm_pool + p.query_pool + p.cache + used, lim);
        // no limit: everything in RAM
        let p = plan(&input(None)).unwrap();
        assert_eq!(p.resident.len(), 12);
        assert_eq!((p.lm_pool, p.query_pool, p.cache), (POOL_MAX, POOL_MAX, CACHE_UNLIMITED));
    }
}
