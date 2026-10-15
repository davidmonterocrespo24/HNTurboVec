//! Robustness of a compact block against damaged bytes (P3, acceptance A2): damages one byte at a
//! time (through the read-fault hook: the file itself is never modified) and runs a fixed set of
//! operations. Every outcome must be the clean result or a typed error (`CorruptBlockError`,
//! `IncompatibleBlockError`), never a panic. Silent differences are expected only where bytes are
//! read without a hash at query time (vector codes, word postings) and are counted per part.
//!
//! Two kinds of iterations:
//! - open: the block is opened (nothing or everything in RAM), the short canaries checked (as
//!   `verify="quick"`), then the operations run;
//! - query: one engine opened clean with everything on disk; the operations run.
//! Half of the damaged bytes are among those the iteration reads (traced from a clean run), the
//! other half anywhere: open: superblock, any part or the tail (padding, TOC, trailer); query: a
//! part read at query time.
//!
//! Usage (HNZ_OV_LIBS must point to OpenVINO: the queries are embedded once at the start):
//!     cargo run --release --example fuzz_block -- <block> <iterations> [seed] [log.jsonl]
use anyhow::Result;
use hnz::blob::{fault, Residency};
use hnz::block::{Block, CorruptBlockError, IncompatibleBlockError};
use hnz::engine::{Engine, EngineOptions, Mode, Query};
use hnz::mem::Budget;
use std::collections::BTreeMap;
use std::io::Write;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::Arc;

const QUERIES: [&str; 3] = ["is the iPad going to kill netbooks", "how do I deal with burnout as a software engineer", "postgres vs mysql"];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

fn open_engine(block: &Arc<Block>, res: Residency) -> Result<Engine> {
    Engine::open_block(block.clone(), EngineOptions { budget: Budget::new(None), residency: res, l2: None, cache_bytes: 0, lm_pool: u64::MAX })
}

/// The fixed operations; returns a fingerprint of everything they produced.
fn ops(eng: &Engine, qv: &[Vec<f32>], ids: &[u32]) -> Result<String> {
    let mut fp = String::new();
    let base = Query { k: 10, ..Default::default() };
    let qs = [
        // hybrid with a date range (2010)
        Query { text: QUERIES[0].into(), from: Some(1_262_304_000), to: Some(1_293_840_000), ..base.clone() },
        Query { text: QUERIES[1].into(), mode: Mode::Vector, ..base.clone() },
        Query { text: QUERIES[2].into(), mode: Mode::Keyword, ..base.clone() },
        Query { text: "kubernetes terraform ansible".into(), mode: Mode::Keyword, ..base.clone() },
        // listing: stories since 2024-01-20
        Query { text: String::new(), types: vec![1], from: Some(1_705_708_800), ..base.clone() },
    ];
    let vecs = [Some(&qv[0][..]), Some(&qv[1][..]), None, None, None];
    let mut first = Vec::new();
    for (q, v) in qs.iter().zip(vecs) {
        let hits = eng.search(q, v)?;
        fp += &format!("{:?};", hits.iter().map(|h| (h.doc, h.score.to_bits())).collect::<Vec<_>>());
        if first.is_empty() {
            first = hits;
        }
    }
    first.truncate(2);
    for r in eng.fetch(&first, false)? {
        fp += &format!("{}|{}|{}|{:?};", r.id, r.title, r.text, r.story_title);
    }
    for &id in ids {
        fp += &format!("{:?};", eng.get(id)?.map(|it| (it.title, it.url, it.text, it.by, it.time)));
    }
    Ok(fp)
}

fn kind(e: &anyhow::Error) -> &'static str {
    if e.downcast_ref::<CorruptBlockError>().is_some() {
        "corrupt"
    } else if e.downcast_ref::<IncompatibleBlockError>().is_some() {
        "incompatible"
    } else {
        "untyped"
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = Path::new(args.get(1).map(String::as_str).unwrap_or("work/p3/hn-sample.hnz"));
    let iters: u64 = args.get(2).and_then(|x| x.parse().ok()).unwrap_or(1000);
    let mut rng = Rng(args.get(3).and_then(|x| x.parse().ok()).unwrap_or(0x5eed_1234_abcd_0001));
    let mut log = std::fs::File::create(args.get(4).map(String::as_str).unwrap_or("fuzz_block.jsonl"))?;
    // silence the default panic message (each panic is logged as one line)
    std::panic::set_hook(Box::new(|_| {}));

    let block = Block::open(path)?;
    let flen = std::fs::metadata(&block.path)?.len();
    let clean = open_engine(&block, Residency::NONE)?;
    let ov = std::env::var("HNZ_OV_LIBS").map_err(|_| anyhow::anyhow!("HNZ_OV_LIBS"))?;
    let mut emb = clean.embedder(Path::new("models"), Path::new(&ov))?;
    let qv: Vec<Vec<f32>> = QUERIES.iter().map(|q| emb.embed_query(q)).collect::<Result<_>>()?;
    drop(emb);
    // a few ids to `get`: stories from the listing
    let lst = clean.search(&Query { text: String::new(), types: vec![1], k: 3, ..Default::default() }, None)?;
    let ids: Vec<u32> = lst.iter().map(|h| clean.segs[0].s.meta.row(h.doc as usize).id).collect();
    let want = ops(&clean, &qv, &ids)?;
    fault::trace(true);
    assert_eq!(ops(&clean, &qv, &ids)?, want, "the operations are not deterministic");
    let query_reads = fault::trace(true);
    {
        let b = Block::open(path)?;
        let e = open_engine(&b, Residency::NONE)?;
        e.check_canaries(&["short"])?;
        assert_eq!(ops(&e, &qv, &ids)?, want, "a reopened block gives other results");
    }
    let open_reads = fault::trace(false);
    eprintln!("traced reads: {} at query time, {} for open + query", query_reads.len(), open_reads.len());
    eprintln!("clean fingerprint: {} bytes; {} parts; file {} bytes", want.len(), block.parts().len(), flen);

    // regions: the superblock, each part, and the tail (padding, TOC and trailer after the last part)
    let mut regions: Vec<(String, u64, u64)> = vec![("superblock".into(), 0, 4096)];
    let last_end = block.parts().iter().map(|p| p.off + p.len).max().unwrap_or(4096);
    regions.extend(block.parts().iter().map(|p| (p.name.clone(), p.off, p.len)));
    regions.push(("tail (padding, TOC, trailer)".into(), last_end, flen - last_end));
    let query_parts: Vec<&(String, u64, u64)> = regions.iter().filter(|r| ["meta/blocks", "text/groups", "lex/post", "vec/codes"].contains(&r.0.as_str())).collect();

    let by_region = |reads: &[(u64, u64)]| -> Vec<Vec<(u64, u64)>> {
        let mut m: BTreeMap<usize, Vec<(u64, u64)>> = BTreeMap::new();
        for &(o, l) in reads.iter().filter(|r| r.1 > 0) {
            let k = regions.iter().position(|r| o >= r.1 && o < r.1 + r.2).unwrap_or(usize::MAX);
            m.entry(k).or_default().push((o, l));
        }
        m.into_values().collect()
    };
    let (open_by_region, query_by_region) = (by_region(&open_reads), by_region(&query_reads));
    eprintln!("regions read: {} at open + query, {} at query time", open_by_region.len(), query_by_region.len());
    let mut stats: BTreeMap<(String, String, &'static str), u64> = BTreeMap::new();
    let t0 = std::time::Instant::now();
    for i in 0..iters {
        let open_mode = i % 4 == 0; // 1 in 4 iterations reopens the block
        let off = if rng.below(2) == 0 {
            // a byte the iteration reads: a region it reads, one of its reads there, a byte of it
            let reads = if open_mode { &open_by_region } else { &query_by_region };
            let rr = &reads[rng.below(reads.len() as u64) as usize];
            let (o, l) = rr[rng.below(rr.len() as u64) as usize];
            o + rng.below(l)
        } else {
            let r = if open_mode { &regions[rng.below(regions.len() as u64) as usize] } else { query_parts[rng.below(query_parts.len() as u64) as usize] };
            r.1 + rng.below(r.2)
        };
        // the region it falls in
        let r = regions.iter().find(|r| off >= r.1 && off < r.1 + r.2).cloned().unwrap_or(("padding".into(), off, 1));
        let xor = if rng.below(2) == 0 { 1u8 << rng.below(8) } else { 1 + rng.below(255) as u8 };
        let res = if rng.below(2) == 0 { Residency::NONE } else { Residency::ALL };
        fault::set(Some((off, xor)));
        let out = catch_unwind(AssertUnwindSafe(|| -> Result<String> {
            if open_mode {
                let b = Block::open(path)?;
                let e = open_engine(&b, res)?;
                e.check_canaries(&["short"])?;
                ops(&e, &qv, &ids)
            } else {
                ops(&clean, &qv, &ids)
            }
        }));
        fault::set(None);
        let outcome: &'static str = match &out {
            Err(_) => "PANIC",
            Ok(Err(e)) => kind(e),
            Ok(Ok(fp)) if *fp == want => "same",
            Ok(Ok(_)) => "different",
        };
        let mode = if open_mode { if res.text { "open+ram" } else { "open+disk" } } else { "query" };
        *stats.entry((r.0.clone(), mode.to_string(), outcome)).or_default() += 1;
        if outcome != "same" {
            let msg = match &out {
                Ok(Err(e)) => format!("{e:#}"),
                Err(p) => p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default(),
                _ => String::new(),
            };
            writeln!(log, "{}", serde_json::json!({"i": i, "mode": mode, "region": r.0, "off": off, "xor": xor, "outcome": outcome, "msg": msg}))?;
            log.flush()?;
        }
        if (i + 1) % 250 == 0 {
            eprintln!("{} / {iters} ({:.0} s)", i + 1, t0.elapsed().as_secs_f64());
        }
    }
    let mut totals: BTreeMap<&str, u64> = BTreeMap::new();
    for ((region, mode, outcome), n) in &stats {
        println!("{region:<30} {mode:<10} {outcome:<13} {n}");
        *totals.entry(outcome).or_default() += n;
    }
    println!("totals: {totals:?} in {:.0} s", t0.elapsed().as_secs_f64());
    if totals.contains_key("PANIC") || totals.contains_key("untyped") {
        std::process::exit(1);
    }
    Ok(())
}
