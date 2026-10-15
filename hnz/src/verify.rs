//! `hnz verify-block` and `hnz inspect` (P3).
//!
//! `verify` checks a compact block at a chosen depth:
//! - header: superblock, trailer, TOC, part bounds, known parts and definition versions;
//! - quick: what `hnz.open` checks by default: + the always-loaded parts (read whole, so hashed)
//!   and the 4 short canaries decoded;
//! - full: + every part hash, the zero padding, the external files and every canary;
//! - deep: + the block against the sources named in its pack spec: every metadata row, record
//!   (bytes and header recomputed from the item), vector code, `kids` list and the author table;
//!   the document order and the id lookups; the word index rebuilt byte for byte; and a sample of
//!   the records (or all of them) decoded with the LM and compared with the Parquet text.
//!
//! `inspect` describes a block (parts, bytes per document) and projects its size to N documents.
use crate::blob::Residency;
use crate::block::{Block, CorruptBlockError, IncompatibleBlockError};
use crate::engine::{Engine, EngineOptions, BLOCK_DEFS, BLOCK_PARTS};
use crate::lm::store::{LmCodec, LmStore};
use crate::mem::Budget;
use crate::meta::MetaRow;
use crate::pack::{check_record, fill, sign_bits, PackSpec};
use crate::segment::TextSource;
use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use twox_hash::XxHash3_64;

/// Exit code of `hnz verify-block` for an error: 2 corrupt, 3 incompatible, 1 anything else.
pub fn exit_code(e: &anyhow::Error) -> i32 {
    if e.downcast_ref::<CorruptBlockError>().is_some() {
        2
    } else if e.downcast_ref::<IncompatibleBlockError>().is_some() {
        3
    } else {
        1
    }
}

/// Differences between the block and its sources (deep): how many, and the first few.
#[derive(Default)]
struct Diffs {
    n: u64,
    first: Vec<String>,
}

impl Diffs {
    fn add(&mut self, m: impl Into<String>) {
        self.n += 1;
        if self.first.len() < 20 {
            self.first.push(m.into());
        }
    }
}

fn secs(t: Instant) -> f64 {
    (t.elapsed().as_secs_f64() * 100.0).round() / 100.0
}

/// The engine over a block with nothing promoted to RAM (only the always-loaded parts).
fn open_engine(block: &Arc<Block>) -> Result<Engine> {
    Engine::open_block(block.clone(), EngineOptions { budget: Budget::new(None), residency: Residency::NONE, l2: None, cache_bytes: 16 << 20, lm_pool: u64::MAX })
}

/// Checks a block at `level` ("header", "quick", "full" or "deep"). Damaged bytes and blocks this
/// binary cannot read are errors (`CorruptBlockError`, `IncompatibleBlockError`); differences with
/// the sources (deep) are listed in the report, whose "ok" is then false.
/// - `spec` (deep): the pack spec; by default the one recorded in the block;
/// - `decode` (deep): records decoded and compared: a fraction ("0.02"), "all" or "none".
pub fn verify(path: &Path, level: &str, spec: Option<&Path>, decode: &str) -> Result<Value> {
    let lv = match level {
        "header" => 0,
        "quick" => 1,
        "full" => 2,
        "deep" => 3,
        o => bail!("--level debe ser header, quick, full o deep (es {o:?})"),
    };
    let decode: Option<f64> = match decode {
        "none" => None,
        "all" => Some(1.0),
        f => match f.parse::<f64>() {
            Ok(x) if x > 0.0 && x <= 1.0 => Some(x),
            _ => bail!("--decode debe ser una fracción en (0, 1], all o none (es {f:?})"),
        },
    };
    if !path.exists() {
        bail!("no existe {}", path.display());
    }
    let file = Block::locate(path).ok_or_else(|| crate::block::incompatible(format!("{} no es un bloque hnz (falta {})", path.display(), crate::block::FILE_NAME)))?;
    let t0 = Instant::now();
    let mut checks = Vec::new();
    let mut diffs = Diffs::default();
    let t = Instant::now();
    let block = Block::open(&file)?;
    block.check_parts(BLOCK_PARTS)?;
    block.check_compat(BLOCK_DEFS)?;
    checks.push(json!({"check": "header", "seconds": secs(t), "parts": block.parts().len(), "external": block.external().len()}));
    if lv >= 1 {
        let t = Instant::now();
        let eng = open_engine(&block)?;
        let n = eng.check_canaries(&["short"])?;
        checks.push(json!({"check": "quick", "seconds": secs(t), "canaries": n}));
        if lv >= 2 {
            let t = Instant::now();
            block.verify_full()?;
            let n = eng.check_canaries(&["long", "fallback", "empty"])?;
            let bytes: u64 = block.parts().iter().map(|p| p.len).sum::<u64>() + block.external().iter().map(|e| e.len).sum::<u64>();
            checks.push(json!({"check": "full", "seconds": secs(t), "bytes_hashed": bytes, "canaries": n}));
        }
        if lv >= 3 {
            deep(&block, &eng, spec, decode, &mut checks, &mut diffs)?;
        }
    }
    Ok(json!({
        "block": block.path.display().to_string(), "snapshot": block.snapshot_hex(), "level": level,
        "ok": diffs.n == 0, "differences": diffs.n, "first_differences": diffs.first,
        "checks": checks, "seconds": secs(t0),
    }))
}

/// One item of a month, checked against its document (see `deep`).
struct ItemCheck {
    doc: Option<usize>,
    diffs: Vec<String>,
    kind: Option<&'static str>,
    ntok: usize,
    has_vec: bool,
    has_kids: bool,
    index_text: String,
}

/// Deep check: the block against the sources of its pack spec. It streams: per document it keeps
/// only the id, time and parent (plus a few bits), and handles one month of items at a time
/// (≈4 GB at 49 M documents, most of it the word index being rebuilt).
fn deep(block: &Arc<Block>, eng: &Engine, spec_path: Option<&Path>, decode: Option<f64>, checks: &mut Vec<Value>, diffs: &mut Diffs) -> Result<()> {
    use crate::meta::META_BLOCK;
    let spec: PackSpec = match spec_path {
        Some(p) => serde_json::from_str(&std::fs::read_to_string(p).with_context(|| format!("{}", p.display()))?)?,
        None => serde_json::from_value(block.manifest()["build"]["spec"].clone()).context("el bloque no guarda su especificación de pack: use --spec")?,
    };
    // a problem in a source file is not a problem of the block: its error loses the block's types
    let src = |e: anyhow::Error| anyhow::anyhow!("fuente: {e:#}");
    let man = block.manifest();
    let n = block.n_docs as usize;
    let sg = &eng.segs[0];
    let meta = &sg.s.meta;
    let TextSource::Lm(lm) = &sg.s.text else { bail!("el texto del bloque no es del nivel LM") };
    let vec = sg.vec.as_ref().context("el bloque no tiene vectores")?;
    let bits = man["vec"]["bits"].as_u64().unwrap_or(0) as usize;

    // ---- metadata: every block (1,024 at a time), order, flags, late ids, index, lookups
    let t = Instant::now();
    eprintln!("deep: metadata");
    let nb = meta.n_blocks();
    let flags = block.read_part("meta/flags")?;
    let (mut ids, mut times, mut parents) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    let mut n_deleted = 0u64;
    for w in (0..nb).step_by(1024) {
        let blocks: Vec<Vec<MetaRow>> = (w..(w + 1024).min(nb)).into_par_iter().map(|b| meta.try_block(b)).collect::<Result<_>>()?;
        for r in blocks.into_iter().flatten() {
            if flags.get(ids.len()).map(|f| f & 0x1f) != Some(crate::meta::flag_byte(&r)) {
                diffs.add(format!("id {}: flag byte", r.id));
            }
            n_deleted += r.deleted as u64;
            ids.push(r.id);
            times.push(r.time);
            parents.push(r.parent);
        }
    }
    if ids.len() != n || n == 0 {
        diffs.add(format!("metadata: {} rows, the block has {n} documents", ids.len()));
        return Ok(());
    }
    for d in 1..n {
        if (times[d], ids[d]) <= (times[d - 1], ids[d - 1]) {
            diffs.add(format!("docs {} and {d} are not in (time, id) order", d - 1));
        }
    }
    let mut by_id: Vec<u32> = (0..n as u32).collect();
    by_id.par_sort_unstable_by_key(|&d| ids[d as usize]);
    for w in by_id.windows(2).filter(|w| ids[w[0] as usize] == ids[w[1] as usize]) {
        diffs.add(format!("id {} is in docs {} and {}", ids[w[0] as usize], w[0], w[1]));
    }
    // late: a time later than that of some higher id
    let mut late = vec![false; n];
    let mut smin = i64::MAX;
    for &d in by_id.iter().rev() {
        let d = d as usize;
        late[d] = times[d] > smin;
        smin = smin.min(times[d]);
    }
    for d in (0..n).filter(|&d| (flags[d] >> 5) & 1 != late[d] as u8) {
        diffs.add(format!("id {}: late flag", ids[d]));
    }
    let want_late: Vec<u8> = by_id.iter().filter(|&&d| late[d as usize]).flat_map(|&d| ids[d as usize].to_le_bytes().into_iter().chain(d.to_le_bytes())).collect();
    if block.read_part("meta/late")? != want_late {
        diffs.add("meta/late is not the list of the late documents");
    }
    drop(by_id);
    // the index: each block's first time and main id (its first in-order id, else the one before)
    let mut main = 0u32;
    for b in 0..nb {
        let (st, en) = (b * META_BLOCK, ((b + 1) * META_BLOCK).min(n));
        if let Some(d) = (st..en).find(|&d| !late[d]) {
            main = ids[d];
        }
        if meta.first_time[b] != times[st] || meta.first_id[b] != main {
            diffs.add(format!("meta/index: block {b}"));
        }
    }
    // lookups: every in-order id leads to its own block (late ids go through meta/late, above);
    // a time leads to the first document at or after it (at every block start and a sample)
    let bad: Vec<String> = (0..n)
        .into_par_iter()
        .filter(|&d| !late[d])
        .filter_map(|d| {
            let b = meta.block_of_id(ids[d]);
            (b != Some(d / META_BLOCK)).then(|| format!("id {}: the id lookup goes to block {b:?}, it is in block {}", ids[d], d / META_BLOCK))
        })
        .collect();
    bad.into_iter().for_each(|b| diffs.add(b));
    for d in (0..n).step_by(META_BLOCK).chain((0..n).step_by(997)) {
        let want = times.partition_point(|&x| x < times[d]);
        let got = meta.try_doc_at_time(times[d])?;
        if got != want {
            diffs.add(format!("time {}: the time lookup gives doc {got}, it is doc {want}", times[d]));
        }
    }
    let s = &man["snapshot"];
    let n_late = late.iter().filter(|&&l| l).count() as u64;
    let (id_min, id_max) = (*ids.iter().min().unwrap() as u64, *ids.iter().max().unwrap() as u64);
    for (k, v) in [("n_docs", n as u64), ("n_late", n_late), ("n_deleted", n_deleted), ("id_min", id_min), ("id_max", id_max)] {
        if s[k].as_u64() != Some(v) {
            diffs.add(format!("manifest snapshot.{k} = {}, the rows give {v}", s[k]));
        }
    }
    if s["time_min"].as_i64() != Some(times[0]) || s["time_max"].as_i64() != Some(times[n - 1]) {
        diffs.add("manifest snapshot.time_min / time_max");
    }
    drop(late);
    checks.push(json!({"check": "deep: metadata", "seconds": secs(t), "rows": n, "late": n_late}));

    // ---- the models, the tokenizers, the embedder and the author table against the spec's files
    for (part, file) in [("model/lm", &spec.lm_model), ("model/lm.tok", &spec.lm_tokenizer), ("model/embed.tok", &spec.embed_tokenizer)] {
        let want = std::fs::read(file).with_context(|| format!("{}", file.display())).map_err(src)?;
        if block.read_part(part)? != want {
            diffs.add(format!("{part} differs from {}", file.display()));
        }
    }
    for e in block.external() {
        let ext = if e.role == "query_ir" { "xml" } else { "bin" };
        let f = PathBuf::from(format!("{}.{ext}", spec.embed_ir.display()));
        let (len, h) = crate::block::hash_file(&f).map_err(src)?;
        if len != e.len || format!("xxh3-128:{h:032x}") != e.hash {
            diffs.add(format!("{} differs from {}", e.name, f.display()));
        }
    }
    let authors = crate::globals::load_authors(&spec.globals).map_err(src)?;
    if authors.join("\n").as_bytes() != block.read_part("meta/authors")?.as_slice() {
        diffs.add("meta/authors differs from the global author table");
    }
    drop(authors);

    // ---- every item of every month against its document
    let codec = LmCodec::from_parts(block.read_part("model/lm")?, Some(&block.read_part("model/lm.tok")?))?;
    let sampled = |d: usize| decode.is_some_and(|f| f >= 1.0 || (XxHash3_64::oneshot(&(d as u64).to_le_bytes()) % 1_000_000) < (f * 1e6) as u64);
    // children of each document (the documents whose parent is its id, by id), for the kids
    let mut pc: Vec<(u32, u32)> = (0..n).filter(|&d| parents[d] != 0).map(|d| (parents[d], ids[d])).collect();
    pc.par_sort_unstable();
    drop(parents);
    let children = |id: u32| -> Vec<u32> {
        let a = pc.partition_point(|x| x.0 < id);
        let b = pc.partition_point(|x| x.0 <= id);
        pc[a..b].iter().map(|x| x.1).collect()
    };
    let frames = block.read_part("meta/kids")?;
    let koffs: Vec<u64> = block.read_part("meta/kids.idx")?.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
    let nf = n.div_ceil(crate::kids::BLOCK);
    if koffs.len() != nf + 1 {
        diffs.add(format!("meta/kids.idx: {} offsets for {nf} frames", koffs.len()));
    }
    let fb = match block.part("text/prefix_fallback") {
        Some(_) => crate::lm::store::parse_prefix_fallback(&block.read_part("text/prefix_fallback")?).ok_or_else(|| crate::block::corrupt("text/prefix_fallback"))?,
        None => HashMap::new(),
    };
    let classes = crate::globals::load_df(&spec.globals).map_err(src)?;
    let gj: Value = serde_json::from_str(&std::fs::read_to_string(spec.globals.join("globals.json")).map_err(|e| src(e.into()))?)?;
    if gj["df_xxh3"] != man["lex"]["class_source"]["df"] {
        diffs.add("the block's term classes come from other global statistics");
    }
    let n_all = gj["n_docs"].as_u64().context("globals.json: n_docs").map_err(src)?;
    let mut lw = crate::bm25::LexWriter::new(&classes, (spec.max_df_ratio * n_all as f64).ceil() as u64, spec.max_df_ratio as f32);
    let mut pending: std::collections::BTreeMap<usize, String> = std::collections::BTreeMap::new();
    let (mut next, mut batch) = (0usize, Vec::<String>::new());
    let mut seen = vec![0u8; n];
    let mut kinds: HashMap<&'static str, u64> = HashMap::new();
    let (mut tokens, mut with_vec, mut n_kids, mut n_decoded, mut fb_found) = (0u64, 0u64, 0u64, 0usize, 0usize);
    for m in &spec.months {
        let t = Instant::now();
        eprintln!("deep: {m}");
        let parquet = fill(&spec.parquet, m);
        let items = crate::item::read_parquet(&parquet).map_err(src)?.items;
        if items.is_empty() {
            continue;
        }
        let kl = crate::item::read_kids(&parquet).map_err(src)?;
        if kl.len() != items.len() || kl.iter().zip(&items).any(|(k, it)| k.0 != it.id) {
            return Err(src(anyhow::anyhow!("{m}: the kids column does not match the items")));
        }
        let seg_dir = fill(&spec.lm_segments, m);
        let seg = LmStore::open_with(&seg_dir, false).map_err(src)?;
        if seg.info.n_docs != items.len() {
            return Err(src(anyhow::anyhow!("{m}: the month segment has {} documents and the Parquet file {} items", seg.info.n_docs, items.len())));
        }
        let seg_meta = crate::meta::MetaStore::open(&seg_dir, items.len()).map_err(src)?;
        let mut l = 0;
        for b in 0..seg_meta.n_blocks() {
            for r in seg_meta.try_block(b).map_err(src)? {
                if items.get(l).map(|x| x.id) != Some(r.id) {
                    return Err(src(anyhow::anyhow!("{m}: document {l} of the month segment is id {}, not the Parquet's", r.id)));
                }
                l += 1;
            }
        }
        let base = fill(&spec.embeddings, m);
        let (shape, descr, map, off) = crate::vecs::read_npy(&PathBuf::from(format!("{}.f16.npy", base.display()))).map_err(src)?;
        let (ishape, idescr, imap, ioff) = crate::vecs::read_npy(&PathBuf::from(format!("{}.ids.npy", base.display()))).map_err(src)?;
        if descr != "<f2" || shape.len() != 2 || shape[1] < bits || idescr != "<u4" || ishape.first() != shape.first() || map.len() < off + shape[0] * shape[1] * 2 || imap.len() < ioff + shape[0] * 4 {
            return Err(src(anyhow::anyhow!("{m}: embeddings must be float16 N x >= {bits} with N uint32 ids")));
        }
        // the files are months by item time: an item outside its month means a misread
        if let (Some(t0), Some(t1)) = (crate::dates::parse(m, false), crate::dates::parse(m, true)) {
            for it in items.iter().filter(|it| it.time < t0 || it.time >= t1) {
                diffs.add(format!("{m}: id {} has time {}, outside the month (a misread source)", it.id, it.time));
            }
        }
        let dim = shape[1];
        let emb: HashMap<u32, usize> = imap[ioff..].chunks_exact(4).enumerate().map(|(r, c)| (u32::from_le_bytes(c.try_into().unwrap()), r)).collect();
        // the document of each item; each metadata block and kids frame of the month decoded once
        let docs: Vec<Option<usize>> = items.par_iter().map(|it| meta.try_doc_of_id(it.id)).collect::<Result<_>>()?;
        let mut bl: Vec<usize> = docs.iter().flatten().map(|&d| d / META_BLOCK).collect();
        bl.sort_unstable();
        bl.dedup();
        let rows: HashMap<usize, Vec<MetaRow>> = bl.par_iter().map(|&b| Ok((b, meta.try_block(b)?))).collect::<Result<_>>()?;
        let mut fr: Vec<usize> = docs.iter().flatten().map(|&d| d / crate::kids::BLOCK).collect();
        fr.sort_unstable();
        fr.dedup();
        let kids_of: HashMap<usize, std::result::Result<Vec<Vec<u32>>, String>> = fr
            .par_iter()
            .map(|&f| {
                let (st, en) = (f * crate::kids::BLOCK, ((f + 1) * crate::kids::BLOCK).min(n));
                let r = match (koffs.get(f), koffs.get(f + 1)) {
                    (Some(&a), Some(&b)) if a <= b && b as usize <= frames.len() => {
                        let ch: Vec<Vec<u32>> = (st..en).map(|d| children(ids[d])).collect();
                        crate::kids::decode_frame(&frames[a as usize..b as usize], &ids[st..en], &ch).map_err(|e| format!("{e:#}"))
                    }
                    _ => Err("offsets".to_string()),
                };
                (f, r)
            })
            .collect();
        let got: Vec<ItemCheck> = items
            .par_iter()
            .enumerate()
            .map(|(l, it)| -> Result<ItemCheck> {
                let Some(d) = docs[l] else {
                    return Ok(ItemCheck { doc: None, diffs: vec![format!("{m}: id {} is not in the block", it.id)], kind: None, ntok: 0, has_vec: false, has_kids: false, index_text: String::new() });
                };
                let r = &rows[&(d / META_BLOCK)][d % META_BLOCK];
                let mut dd = Vec::new();
                if r.id != it.id
                    || meta.authors.get(r.author as usize) != it.by
                    || (r.time, r.typ, r.dead, r.deleted, r.parent, r.score, r.descendants, r.poll, &r.parts)
                        != (it.time, it.typ, it.dead, it.deleted, it.parent, it.score, it.descendants, it.poll, &it.parts)
                {
                    dd.push(format!("{m}: id {}: metadata differs", it.id));
                }
                let rec = lm.raw(d)?;
                if seg.raw(l).map_err(src)?.as_ref() != rec.as_ref() {
                    dd.push(format!("{m}: id {}: record bytes differ from the month segment", it.id));
                }
                let (kind, ntok) = match check_record(&codec, it, &rec) {
                    Ok((k, t)) => (Some(k), t),
                    Err(e) => {
                        dd.push(format!("{m}: {e:#}"));
                        (None, 0)
                    }
                };
                let has_vec = vec.has(d);
                match emb.get(&it.id) {
                    Some(&row) => {
                        let want = sign_bits(&map[off + row * dim * 2..off + (row * dim + bits) * 2], bits);
                        if !has_vec || vec.try_code(d)?.as_ref() != want.as_slice() {
                            dd.push(format!("{m}: id {}: vector code differs from the embedding", it.id));
                        }
                    }
                    None if has_vec => dd.push(format!("{m}: id {}: a vector code without an embedding", it.id)),
                    None => {}
                }
                if has_vec && kind == Some("fallback") {
                    let want = codec.prefix(&it.title, &it.url, &it.text).map(|(p, f)| (f as u8, p));
                    if fb.get(&(d as u32)) != want.as_ref() {
                        dd.push(format!("{m}: id {}: text/prefix_fallback", it.id));
                    }
                }
                match &kids_of[&(d / crate::kids::BLOCK)] {
                    Ok(k) if k[d % crate::kids::BLOCK] == kl[l].1 => {}
                    Ok(_) => dd.push(format!("{m}: id {}: kids differ", it.id)),
                    Err(e) => dd.push(format!("{m}: id {}: kids frame: {e}", it.id)),
                }
                Ok(ItemCheck { doc: Some(d), diffs: dd, kind, ntok, has_vec, has_kids: !kl[l].1.is_empty(), index_text: crate::segment::index_text(&it.title, &it.url, &it.text) })
            })
            .collect::<Result<_>>()?;
        let mut picked: Vec<(usize, usize)> = Vec::new();
        for (l, c) in got.into_iter().enumerate() {
            c.diffs.into_iter().for_each(|x| diffs.add(x));
            let Some(d) = c.doc else { continue };
            seen[d] = seen[d].saturating_add(1);
            if let Some(k) = c.kind {
                *kinds.entry(k).or_default() += 1;
            }
            tokens += c.ntok as u64;
            with_vec += c.has_vec as u64;
            n_kids += c.has_kids as u64;
            fb_found += (c.has_vec && c.kind == Some("fallback")) as usize;
            pending.insert(d, c.index_text);
            if sampled(d) {
                picked.push((d, l));
            }
        }
        // the word index takes the texts in document order, in batches of 65,536 like pack
        while let Some(t) = pending.remove(&next) {
            batch.push(t);
            next += 1;
            if batch.len() == 65536 {
                lw.add_batch(&batch);
                batch.clear();
            }
        }
        // the sampled records of the month, decoded with the block's LM, against the items
        for ch in picked.chunks(65536) {
            let raws: Vec<std::borrow::Cow<[u8]>> = ch.iter().map(|&(d, _)| lm.raw(d)).collect::<Result<_>>()?;
            let refs: Vec<&[u8]> = raws.iter().map(|r| r.as_ref()).collect();
            let mut bad = Vec::new();
            codec.decode_stream(&refs, true, 64, |j, r| {
                let it = &items[ch[j].1];
                if (r.title.as_str(), r.url.as_str(), r.text.as_str()) != (it.title.as_str(), it.url.as_str(), it.text.as_str()) {
                    bad.push(it.id);
                }
                Ok(())
            })?;
            bad.into_iter().for_each(|id| diffs.add(format!("{m}: id {id}: the decoded text differs from the Parquet")));
        }
        n_decoded += picked.len();
        checks.push(json!({"check": format!("deep: {m}"), "seconds": secs(t), "items": items.len(), "decoded": picked.len()}));
    }
    lw.add_batch(&batch);
    drop(batch);
    let not_once = seen.iter().filter(|&&s| s != 1).count();
    if not_once > 0 || next != n {
        diffs.add(format!("{not_once} documents are not exactly one item of the spec's months"));
    }
    let t_ = &man["text"];
    for (key, v) in [
        (&s["n_lm"], kinds.get("lm").copied().unwrap_or(0)),
        (&s["n_fallback"], kinds.get("fallback").copied().unwrap_or(0)),
        (&s["n_empty"], kinds.get("empty").copied().unwrap_or(0)),
        (&t_["lm_records"], kinds.get("lm").copied().unwrap_or(0)),
        (&t_["zstd_records"], kinds.get("fallback").copied().unwrap_or(0)),
        (&t_["tokens"], tokens),
        (&s["n_with_vector"], with_vec),
        (&man["vec"]["prefix_fallback"], fb_found as u64),
    ] {
        if !key.is_null() && key.as_u64() != Some(v) {
            diffs.add(format!("a manifest counter is {key}, the items give {v}"));
        }
    }
    if fb.len() != fb_found {
        diffs.add(format!("text/prefix_fallback has {} entries for {fb_found} documents", fb.len()));
    }
    checks.push(json!({"check": "deep: items", "kids_lists": n_kids, "decoded": n_decoded}));

    // the word index, rebuilt from the items' text in document order
    let t = Instant::now();
    eprintln!("deep: word index");
    let lp = lw.finish()?;
    for (name, bytes) in [("lex/fst", &lp.fst), ("lex/post", &lp.post), ("lex/norms", &lp.norms)] {
        if block.read_part(name)? != *bytes {
            diffs.add(format!("{name} differs from a rebuild"));
        }
    }
    // compared after the same JSON round trip (an f32 written to JSON is read back as an f64)
    let stored: crate::bm25::Bm25Info = serde_json::from_value(man["lex"]["info"].clone()).map_err(|e| crate::block::corrupt(format!("lex.info: {e}")))?;
    if serde_json::to_value(&lp.info)? != serde_json::to_value(&stored)? {
        diffs.add("manifest lex.info differs from a rebuild");
    }
    checks.push(json!({"check": "deep: word index", "seconds": secs(t), "terms": lp.info.n_terms}));
    Ok(())
}

/// Describes a block: its parts (offset, sizes, bytes per document) and TOC; with `project` the
/// sizes projected to that many documents (per-document parts scaled; the author table and the
/// models fixed; the word index exact from `lex`, the `lex.json` of `hnz lexicon`, when given);
/// with `id`, where that item's bytes are (document, metadata block, text group).
pub fn inspect(path: &Path, project: Option<u64>, lex: Option<&Path>, id: Option<u32>) -> Result<Value> {
    let block = Block::open(path)?;
    let man = block.manifest();
    let n = block.n_docs.max(1) as f64;
    let container = std::fs::metadata(&block.path)?.len();
    let external: u64 = block.external().iter().map(|e| e.len).sum();
    let in_parts: u64 = block.parts().iter().map(|p| p.len).sum();
    let r2 = |x: f64| (x * 100.0).round() / 100.0;
    let parts: Vec<Value> = block
        .parts()
        .iter()
        .map(|p| json!({"name": p.name, "ver": p.ver, "codec": p.codec, "access": p.access, "off": p.off, "bytes": p.len, "raw_bytes": p.raw_len, "bytes_per_doc": r2(p.len as f64 / n)}))
        .collect();
    let mut out = json!({
        "path": block.path.display().to_string(), "snapshot": block.snapshot_hex(), "writer": block.writer,
        "n_docs": block.n_docs, "math_version": block.math_version,
        "source": man["snapshot"], "text": man["text"], "vec": man["vec"], "lex_terms": man["lex"]["terms"], "search": man["search"],
        "parts": parts,
        "toc": {"off": block.toc_range().0, "len": block.toc_range().1},
        "external": block.external().iter().map(|e| json!({"name": e.name, "bytes": e.len, "role": e.role})).collect::<Vec<_>>(),
        "bytes": {"container": container, "in_parts": in_parts, "container_overhead": container - in_parts, "external": external,
                  "total": container + external, "per_doc": r2((container + external) as f64 / n)},
    });
    if let Some(np) = project {
        let lj: Option<Value> = match lex {
            Some(p) => Some(serde_json::from_str(&std::fs::read_to_string(p).with_context(|| format!("{}", p.display()))?)?),
            None => None,
        };
        let mut rows = serde_json::Map::new();
        let mut total = 0f64;
        for p in block.parts() {
            let exact = lj.as_ref().and_then(|l| {
                let key = match p.name.as_str() {
                    "lex/fst" => "fst",
                    "lex/post" => "post",
                    "lex/norms" => "norms_zstd",
                    _ => return None,
                };
                Some(l["bytes"][key].as_f64()? * np as f64 / l["n_docs"].as_f64()?)
            });
            let (how, b) = match exact {
                Some(b) => ("lex.json", b),
                None if p.name == "meta/authors" || p.name.starts_with("model/") => ("fixed", p.len as f64),
                None => ("scaled", p.len as f64 * np as f64 / n),
            };
            total += b;
            rows.insert(p.name.clone(), json!({"bytes": b.round(), "how": how}));
        }
        let fixed = (container - in_parts) as f64 + external as f64;
        total += fixed;
        out["projection"] = json!({
            "n_docs": np, "parts": rows, "fixed_overhead_and_external": fixed,
            "total": total.round(), "total_gb": r2(total / 1e9), "per_doc": r2(total / np as f64),
        });
    }
    if let Some(id) = id {
        let eng = open_engine(&block)?;
        let sg = &eng.segs[0];
        let TextSource::Lm(lm) = &sg.s.text else { bail!("el texto del bloque no es del nivel LM") };
        let at = |name: &str| block.part(name).map_or(0, |p| p.off);
        out["locate"] = match sg.s.meta.try_doc_of_id(id)? {
            None => json!({"id": id, "found": false}),
            Some(d) => {
                let mb = d / crate::meta::META_BLOCK;
                let (mo, ml) = sg.s.meta.block_range(mb);
                let g = d / lm.info.group;
                let (to, tl) = lm.group_range(g);
                json!({"id": id, "found": true, "doc": d,
                       "meta_block": {"index": mb, "off": at("meta/blocks") + mo, "len": ml},
                       "text_group": {"index": g, "off": at("text/groups") + to, "len": tl},
                       "has_vector": sg.vec.as_ref().is_some_and(|v| v.has(d))})
            }
        };
    }
    Ok(out)
}
