//! Checks the typed LM jobs (P4) on real records of a block:
//! - Full / Snippet(64) / Title through `decode_records_pool` (now built on `decode_jobs`) give
//!   exactly what the batch loop before P4 gave (`decode_records_pool_v1`), for any pool;
//! - Prefix { body: 128, mark: 64 } gives the bytes `LmCodec::prefix` computes with the LM
//!   tokenizer, its `snippet_end` cuts exactly the Snippet(64) text, and the document prompt
//!   rebuilt from it is the one `hnz prompts` wrote (`<prompts>/<month>.prompts.jsonl`).
//! Usage: cargo run --release --example decode_check -- <block> <n> [<prompts dir>]
use anyhow::{bail, Result};
use hnz::blob::Residency;
use hnz::block::Block;
use hnz::engine::{Engine, EngineOptions};
use hnz::lm::jobs::{Decoded, Want};
use hnz::lm::store::LmCodec;
use hnz::mem::Budget;
use hnz::segment::TextSource;
use std::collections::HashMap;
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let block = Block::open(Path::new(args.get(1).map(String::as_str).unwrap_or("work/p3/hn-sample.hnz")))?;
    let n: usize = args.get(2).and_then(|x| x.parse().ok()).unwrap_or(2000);
    let eng = Engine::open_block(block.clone(), EngineOptions { budget: Budget::new(None), residency: Residency::NONE, l2: None, cache_bytes: 0, lm_pool: u64::MAX })?;
    let codec = LmCodec::from_parts(block.read_part("model/lm")?, Some(&block.read_part("model/lm.tok")?))?;
    let TextSource::Lm(lm) = &eng.segs[0].s.text else { bail!("not an LM block") };
    let meta = &eng.segs[0].s.meta;
    // the prompts `hnz prompts` wrote, by id
    let mut prompts: HashMap<u32, String> = HashMap::new();
    if let Some(dir) = args.get(3) {
        for e in std::fs::read_dir(dir)? {
            let p = e?.path();
            if p.to_string_lossy().ends_with(".prompts.jsonl") {
                for line in std::fs::read_to_string(&p)?.lines() {
                    let v: serde_json::Value = serde_json::from_str(line)?;
                    prompts.insert(v["id"].as_u64().unwrap() as u32, v["prompt"].as_str().unwrap().to_string());
                }
            }
        }
        eprintln!("{} prompts read", prompts.len());
    }
    // n documents spread over the block
    let total = eng.total_docs;
    let docs: Vec<usize> = (0..n).map(|i| (i as u64 * 2654435761 % total as u64) as usize).collect();
    let raws: Vec<Vec<u8>> = docs.iter().map(|&d| lm.raw(d).map(|r| r.into_owned())).collect::<Result<_>>()?;
    let refs: Vec<&[u8]> = raws.iter().map(|r| r.as_slice()).collect();
    let checks = vec![true; n];
    let mut bad = 0usize;
    // 1. the three text modes, new vs old, at two pools
    for (name, limit, title) in [("full", None, false), ("snippet", Some(64), false), ("title", None, true)] {
        let limits = vec![limit; n];
        let tonly = vec![title; n];
        let old = codec.decode_records_pool_v1(&refs, true, &limits, &tonly, &checks, u64::MAX)?;
        for pool in [u64::MAX, codec.min_pool()] {
            let new = codec.decode_records_pool(&refs, true, &limits, &tonly, &checks, pool)?;
            let diff = old.iter().zip(&new).filter(|(a, b)| a.0 != b.0 || a.1 != b.1).count();
            println!("{name:<8} pool {:>14}: {diff} of {n} differ", if pool == u64::MAX { "unlimited".to_string() } else { pool.to_string() });
            bad += diff;
        }
    }
    // 2. prefixes: against the tokenizer, the snippet cut, and the prompts of `hnz prompts`
    let full = codec.decode_records_pool(&refs, true, &vec![None; n], &vec![false; n], &checks, u64::MAX)?;
    let snip = codec.decode_records_pool(&refs, true, &vec![Some(64); n], &vec![false; n], &checks, u64::MAX)?;
    let wants = vec![Want::Prefix { body: hnz::prompt::PREFIX_TOKENS, mark: 64 }; n];
    let pre = codec.decode_jobs(&refs, &wants, &checks, true, u64::MAX)?;
    let (mut lm_docs, mut pdiff, mut sdiff, mut prompt_same, mut prompt_diff, mut complete) = (0, 0, 0, 0, 0, 0);
    for (j, d) in pre.iter().enumerate() {
        let Decoded::Prefix(p) = d else { continue }; // zstd and empty records come back as text
        lm_docs += 1;
        complete += p.complete as usize;
        let (t, _) = &full[j];
        let want = codec.prefix(&t.title, &t.url, &t.text);
        if want.as_ref().map(|w| (w.0.as_bytes(), w.1 as u8)) != Some((p.bytes.as_slice(), p.flags)) {
            pdiff += 1;
            if pdiff <= 3 {
                println!("prefix differs, doc {}: {:?} vs {:?}", docs[j], String::from_utf8_lossy(&p.bytes), want.map(|w| w.0));
            }
        }
        let cut = String::from_utf8_lossy(&p.bytes[..p.snippet_end]).to_string();
        let as_snippet = hnz::lm::store::snippet_of(&cut, p.flags as u64);
        if (as_snippet.title.as_str(), as_snippet.url.as_str(), as_snippet.text.as_str()) != (snip[j].0.title.as_str(), snip[j].0.url.as_str(), snip[j].0.text.as_str()) && snip[j].1 {
            sdiff += 1;
        }
        let id = meta.row(docs[j]).id;
        if let Some(want) = prompts.get(&id) {
            let got = hnz::prompt::doc_prompt(&hnz::prompt::prefix_string(&p.bytes), p.flags as u64);
            if &got == want {
                prompt_same += 1;
            } else {
                prompt_diff += 1;
                if prompt_diff <= 3 {
                    println!("prompt differs, id {id}:\n  {got:?}\n  {want:?}");
                }
            }
        }
    }
    println!("prefixes: {lm_docs} LM records ({complete} complete): {pdiff} differ from the tokenizer's, {sdiff} snippet cuts differ; prompts: {prompt_same} equal, {prompt_diff} differ");
    bad += pdiff + sdiff + prompt_diff;
    if bad > 0 {
        std::process::exit(1);
    }
    Ok(())
}
