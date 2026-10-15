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

fn xxh(s: &str) -> u64 {
    XxHash3_64::oneshot(s.as_bytes())
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
    has_vec: bool,
    index_text: String,
    /// decode sample: hash of the item's title, url and text
    want: Option<u64>,
}

fn deep(block: &Arc<Block>, eng: &Engine, spec_path: Option<&Path>, decode: Option<f64>, checks: &mut Vec<Value>, diffs: &mut Diffs) -> Result<()> {
    let spec: PackSpec = match spec_path {
        Some(p) => serde_json::from_str(&std::fs::read_to_string(p).with_context(|| format!("{}", p.display()))?)?,
        None => serde_json::from_value(block.manifest()["build"]["spec"].clone()).context("el bloque no guarda su especificación de pack: use --spec")?,
    };
    let man = block.manifest();
    let n = block.n_docs as usize;
    let sg = &eng.segs[0];
    let meta = &sg.s.meta;
    let TextSource::Lm(lm) = &sg.s.text else { bail!("el texto del bloque no es del nivel LM") };
    let vec = sg.vec.as_ref().context("el bloque no tiene vectores")?;
    let bits = man["vec"]["bits"].as_u64().unwrap_or(0) as usize;

    // metadata: every row; (time, id) order; late flags; the flag column; id and time lookups
    let t = Instant::now();
    eprintln!("deep: metadata");
    let rows: Vec<MetaRow> = (0..meta.n_blocks()).into_par_iter().map(|b| meta.try_block(b)).collect::<Result<Vec<_>>>()?.into_iter().flatten().collect();
    if rows.len() != n {
        diffs.add(format!("metadata: {} rows, the block has {n} documents", rows.len()));
        return Ok(());
    }
    let mut by_id: Vec<(u32, usize)> = rows.iter().enumerate().map(|(d, r)| (r.id, d)).collect();
    by_id.sort_unstable();
    for w in by_id.windows(2).filter(|w| w[0].0 == w[1].0) {
        diffs.add(format!("id {} is in docs {} and {}", w[0].0, w[0].1, w[1].1));
    }
    let mut late = vec![false; n];
    let mut smin = i64::MAX;
    for &(_, d) in by_id.iter().rev() {
        late[d] = rows[d].time > smin;
        smin = smin.min(rows[d].time);
    }
    for d in 1..n {
        if (rows[d].time, rows[d].id) <= (rows[d - 1].time, rows[d - 1].id) {
            diffs.add(format!("docs {} and {d} are not in (time, id) order", d - 1));
        }
    }
    let flags = block.read_part("meta/flags")?;
    for (d, r) in rows.iter().enumerate() {
        if flags.get(d).copied() != Some(crate::meta::flag_byte(r) | (late[d] as u8) << 5) {
            diffs.add(format!("id {}: flag byte", r.id));
        }
    }
    let bad: Vec<String> = (0..n)
        .into_par_iter()
        .filter_map(|d| {
            let got = meta.doc_of_id(rows[d].id);
            (got != Some(d)).then(|| format!("id {}: the id lookup gives doc {got:?}, it is doc {d}", rows[d].id))
        })
        .collect();
    bad.into_iter().for_each(|b| diffs.add(b));
    for d in (0..n).step_by(997) {
        let want = rows.partition_point(|r| r.time < rows[d].time);
        if meta.doc_at_time(rows[d].time) != want {
            diffs.add(format!("time {}: the time lookup gives doc {}, it is doc {want}", rows[d].time, meta.doc_at_time(rows[d].time)));
        }
    }
    let s = &man["snapshot"];
    let n_late = late.iter().filter(|&&l| l).count() as u64;
    let n_deleted = rows.iter().filter(|r| r.deleted).count() as u64;
    let (id_min, id_max) = (by_id[0].0 as u64, by_id[n - 1].0 as u64);
    for (k, v) in [("n_late", n_late), ("n_deleted", n_deleted), ("id_min", id_min), ("id_max", id_max)] {
        if s[k].as_u64() != Some(v) {
            diffs.add(format!("manifest snapshot.{k} = {}, the rows give {v}", s[k]));
        }
    }
    if s["time_min"].as_i64() != Some(rows[0].time) || s["time_max"].as_i64() != Some(rows[n - 1].time) {
        diffs.add("manifest snapshot.time_min / time_max");
    }
    checks.push(json!({"check": "deep: metadata", "seconds": secs(t), "rows": n, "late": n_late}));

    // the global author table
    let authors = crate::globals::load_authors(&spec.globals)?;
    if authors.join("\n").as_bytes() != block.read_part("meta/authors")?.as_slice() {
        diffs.add("meta/authors differs from the global author table");
    }

    // every item of every month against its document
    let codec = LmCodec::from_parts(block.read_part("model/lm")?, Some(&block.read_part("model/lm.tok")?))?;
    let sampled = |d: usize| decode.is_some_and(|f| f >= 1.0 || (XxHash3_64::oneshot(&(d as u64).to_le_bytes()) % 1_000_000) < (f * 1e6) as u64);
    let mut seen = vec![0u8; n];
    let mut kids_of: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut texts: Vec<String> = vec![String::new(); n];
    let mut want_text: HashMap<usize, u64> = HashMap::new();
    let mut kinds: HashMap<&'static str, u64> = HashMap::new();
    let mut with_vec = 0u64;
    for m in &spec.months {
        let t = Instant::now();
        eprintln!("deep: {m}");
        let parquet = fill(&spec.parquet, m);
        let items = crate::item::read_parquet(&parquet)?.items;
        let mut kl = crate::item::read_kids(&parquet)?;
        if kl.len() != items.len() || kl.iter().zip(&items).any(|(k, it)| k.0 != it.id) {
            bail!("{m}: the kids column does not match the items");
        }
        let seg = LmStore::open_with(&fill(&spec.lm_segments, m), false)?;
        let base = fill(&spec.embeddings, m);
        let (shape, descr, map, off) = crate::vecs::read_npy(&PathBuf::from(format!("{}.f16.npy", base.display())))?;
        let (_, _, imap, ioff) = crate::vecs::read_npy(&PathBuf::from(format!("{}.ids.npy", base.display())))?;
        if descr != "<f2" || shape.len() != 2 || shape[1] < bits {
            bail!("{m}: embeddings must be float16 N x >= {bits}");
        }
        let dim = shape[1];
        let emb: HashMap<u32, usize> = imap[ioff..].chunks_exact(4).enumerate().map(|(r, c)| (u32::from_le_bytes(c.try_into().unwrap()), r)).collect();
        let got: Vec<ItemCheck> = items
            .par_iter()
            .enumerate()
            .map(|(l, it)| -> Result<ItemCheck> {
                let Ok(i) = by_id.binary_search_by_key(&it.id, |x| x.0) else {
                    return Ok(ItemCheck { doc: None, diffs: vec![format!("{m}: id {} is not in the block", it.id)], kind: None, has_vec: false, index_text: String::new(), want: None });
                };
                let d = by_id[i].1;
                let r = &rows[d];
                let mut dd = Vec::new();
                let author = (r.author as usize) < meta.authors.len();
                if !author
                    || meta.authors.get(r.author as usize) != it.by
                    || (r.time, r.typ, r.dead, r.deleted, r.parent, r.score, r.descendants, r.poll, &r.parts)
                        != (it.time, it.typ, it.dead, it.deleted, it.parent, it.score, it.descendants, it.poll, &it.parts)
                {
                    dd.push(format!("{m}: id {}: metadata differs", it.id));
                }
                let rec = lm.raw(d)?;
                if seg.raw(l)?.as_ref() != rec.as_ref() {
                    dd.push(format!("{m}: id {}: record bytes differ from the month segment", it.id));
                }
                let kind = match check_record(&codec, it, &rec) {
                    Ok((k, _)) => Some(k),
                    Err(e) => {
                        dd.push(format!("{m}: {e:#}"));
                        None
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
                let want = sampled(d).then(|| xxh(&format!("{}\0{}\0{}", it.title, it.url, it.text)));
                Ok(ItemCheck { doc: Some(d), diffs: dd, kind, has_vec, index_text: crate::segment::index_text(&it.title, &it.url, &it.text), want })
            })
            .collect::<Result<_>>()?;
        for (j, c) in got.into_iter().enumerate() {
            c.diffs.into_iter().for_each(|x| diffs.add(x));
            let Some(d) = c.doc else { continue };
            seen[d] = seen[d].saturating_add(1);
            if let Some(k) = c.kind {
                *kinds.entry(k).or_default() += 1;
            }
            with_vec += c.has_vec as u64;
            texts[d] = c.index_text;
            kids_of[d] = std::mem::take(&mut kl[j].1);
            if let Some(w) = c.want {
                want_text.insert(d, w);
            }
        }
        checks.push(json!({"check": format!("deep: {m}"), "seconds": secs(t), "items": items.len()}));
    }
    let not_once = seen.iter().filter(|&&s| s != 1).count();
    if not_once > 0 {
        diffs.add(format!("{not_once} documents are not exactly one item of the spec's months"));
    }
    for (k, key) in [("lm", "n_lm"), ("fallback", "n_fallback"), ("empty", "n_empty")] {
        let v = kinds.get(k).copied().unwrap_or(0);
        if s[key].as_u64() != Some(v) {
            diffs.add(format!("manifest snapshot.{key} = {}, the items give {v}", s[key]));
        }
    }
    if s["n_with_vector"].as_u64() != Some(with_vec) {
        diffs.add(format!("manifest snapshot.n_with_vector = {}, the embeddings give {with_vec}", s["n_with_vector"]));
    }

    // kids: every frame decoded with the children found through `parent`
    let t = Instant::now();
    eprintln!("deep: kids");
    let ids: Vec<u32> = rows.iter().map(|r| r.id).collect();
    let parents: Vec<u32> = rows.iter().map(|r| r.parent).collect();
    let children = crate::kids::children_of(&ids, &parents);
    let frames = block.read_part("meta/kids")?;
    let offs: Vec<u64> = block.read_part("meta/kids.idx")?.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
    let nf = n.div_ceil(crate::kids::BLOCK);
    if offs.len() != nf + 1 {
        diffs.add(format!("meta/kids.idx: {} offsets for {nf} frames", offs.len()));
    } else {
        let bad: Vec<String> = (0..nf)
            .into_par_iter()
            .filter_map(|f| {
                let (st, en) = (f * crate::kids::BLOCK, ((f + 1) * crate::kids::BLOCK).min(n));
                let (a, b) = (offs[f] as usize, offs[f + 1] as usize);
                if a > b || b > frames.len() {
                    return Some(format!("kids frame {f}: offsets"));
                }
                match crate::kids::decode_frame(&frames[a..b], &ids[st..en], &children[st..en]) {
                    Ok(k) => (0..en - st).find(|&i| k[i] != kids_of[st + i]).map(|i| format!("id {}: kids differ", ids[st + i])),
                    Err(e) => Some(format!("kids frame {f}: {e:#}")),
                }
            })
            .collect();
        bad.into_iter().for_each(|b| diffs.add(b));
    }
    let n_kids = kids_of.iter().filter(|k| !k.is_empty()).count();
    drop(kids_of);
    checks.push(json!({"check": "deep: kids", "seconds": secs(t), "with_kids": n_kids}));

    // the word index, rebuilt from the items' text in document order
    let t = Instant::now();
    eprintln!("deep: word index");
    let classes = crate::globals::load_df(&spec.globals)?;
    let gj: Value = serde_json::from_str(&std::fs::read_to_string(spec.globals.join("globals.json"))?)?;
    if gj["df_xxh3"] != man["lex"]["class_source"]["df"] {
        diffs.add("the block's term classes come from other global statistics");
    }
    let n_all = gj["n_docs"].as_u64().context("globals.json: n_docs")?;
    let mut lw = crate::bm25::LexWriter::new(&classes, (spec.max_df_ratio * n_all as f64).ceil() as u64, spec.max_df_ratio as f32);
    for ch in texts.chunks(65536) {
        lw.add_batch(ch);
    }
    drop(texts);
    let lp = lw.finish()?;
    for (name, bytes) in [("lex/fst", &lp.fst), ("lex/post", &lp.post), ("lex/norms", &lp.norms)] {
        if block.read_part(name)? != *bytes {
            diffs.add(format!("{name} differs from a rebuild"));
        }
    }
    if serde_json::to_value(&lp.info)? != man["lex"]["info"] {
        diffs.add("manifest lex.info differs from a rebuild");
    }
    checks.push(json!({"check": "deep: word index", "seconds": secs(t), "terms": lp.info.n_terms}));

    // records decoded with the block's LM and compared with the Parquet text
    if !want_text.is_empty() {
        let t = Instant::now();
        let mut docs: Vec<usize> = want_text.keys().copied().collect();
        docs.sort_unstable();
        eprintln!("deep: decoding {} records", docs.len());
        let mut bad = Vec::new();
        for ch in docs.chunks(65536) {
            let raws: Vec<std::borrow::Cow<[u8]>> = ch.iter().map(|&d| lm.raw(d)).collect::<Result<_>>()?;
            let refs: Vec<&[u8]> = raws.iter().map(|r| r.as_ref()).collect();
            codec.decode_stream(&refs, true, 64, |j, r| {
                if xxh(&format!("{}\0{}\0{}", r.title, r.url, r.text)) != want_text[&ch[j]] {
                    bad.push(ch[j]);
                }
                Ok(())
            })?;
        }
        for d in bad {
            diffs.add(format!("id {}: the decoded text differs from the Parquet", rows[d].id));
        }
        checks.push(json!({"check": "deep: decode", "seconds": secs(t), "records": docs.len()}));
    }
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
