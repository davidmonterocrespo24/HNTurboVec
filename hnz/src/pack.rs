//! `hnz pack` (P3): builds a compact block (`<out>/hn.hnzb` + the embedder IR next to it) from the
//! monthly Parquet files and the per-month LM segments, whose compressed records are copied byte
//! for byte (every copied record is checked against its source item: whole record, or header and
//! check byte), the prefix embeddings, the global statistics of `hnz globals` (author table, term
//! classes) and the models. Documents are in (time, id) order (E32: 0.63 % of HN ids are out of
//! time order), so a date range is a contiguous range of documents.
use crate::block::{Access, BlockWriter, Codec, PartSpec};
use crate::bm25::LexWriter;
use crate::item::{read_kids, read_parquet, Item};
use crate::lm::store::{LmCodec, LmStore, RecPlan, TextGroupWriter};
use crate::meta::{MetaBlockWriter, MetaRow, MetaStore};
use anyhow::{bail, Context, Result};
use half::f16;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;
use twox_hash::XxHash3_64;

/// What to pack (a JSON file, so a block can be rebuilt exactly; the block records it).
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct PackSpec {
    /// "YYYY-MM", in time order
    pub months: Vec<String>,
    /// path patterns with {y} (year) and {m} (YYYY-MM)
    pub parquet: String,
    pub lm_segments: String,
    /// prefix embeddings: `<pattern>.f16.npy` (N x 768) and `<pattern>.ids.npy`
    pub embeddings: String,
    pub globals: PathBuf,
    pub max_df_ratio: f64,
    pub lm_model: PathBuf,
    pub lm_tokenizer: PathBuf,
    pub embed_tokenizer: PathBuf,
    /// the embedder IR without extension (`.xml` and `.bin`)
    pub embed_ir: PathBuf,
    pub bits: usize,
    pub c1: usize,
    /// prompt definition of the embeddings (1 = experiments/embed_prefix.py, E30a)
    pub prompt_version: u32,
}

pub(crate) fn fill(pat: &str, m: &str) -> PathBuf {
    PathBuf::from(pat.replace("{y}", &m[..4]).replace("{m}", m))
}

struct Month {
    name: String,
    parquet: PathBuf,
    items: Vec<Item>,
    kids: Vec<Vec<u32>>,
    lm: LmStore,
    emb: HashMap<u32, usize>,
    npy: (Vec<usize>, memmap2::Mmap, usize),
}

fn xxh(b: &[u8]) -> String {
    format!("xxh3-64:{:016x}", XxHash3_64::oneshot(b))
}

/// A month of the spec, or None if its file has no items (nothing else is needed for it).
fn load_month(spec: &PackSpec, m: &str, codec: &LmCodec) -> Result<Option<Month>> {
    let parquet = fill(&spec.parquet, m);
    let items = read_parquet(&parquet)?.items;
    if items.is_empty() {
        eprintln!("pack: {m} has no items");
        return Ok(None);
    }
    // the dataset's files are months by item time: an item outside its month means a misread
    let (t0, t1) = (crate::dates::parse(m, false), crate::dates::parse(m, true));
    if let (Some(t0), Some(t1)) = (t0, t1) {
        if let Some(it) = items.iter().find(|it| it.time < t0 || it.time >= t1) {
            bail!("{m}: id {} has time {} ({}), outside the month", it.id, it.time, crate::dates::fmt(it.time));
        }
    }
    let kl = read_kids(&parquet)?;
    if kl.len() != items.len() || kl.iter().zip(&items).any(|(k, it)| k.0 != it.id) {
        bail!("{m}: the kids column does not match the items");
    }
    let kids = kl.into_iter().map(|x| x.1).collect();
    let seg = fill(&spec.lm_segments, m);
    let lm = LmStore::open_with(&seg, true).with_context(|| format!("{}", seg.display()))?;
    if lm.info.math != crate::lm::model::MATH_VERSION || lm.info.model_id != codec.model.id || !lm.info.rec_check || lm.info.n_docs != items.len() {
        bail!(
            "{m}: the LM segment does not match (math {}, model {}, rec_check {}, {} docs vs {} items)",
            lm.info.math,
            lm.info.model_id,
            lm.info.rec_check,
            lm.info.n_docs,
            items.len()
        );
    }
    // the segment's documents must be the Parquet items, in the same (id) order
    let meta = MetaStore::open(&seg, items.len())?;
    let mut d = 0;
    for b in 0..meta.n_blocks() {
        for r in meta.try_block(b)? {
            if items.get(d).map(|x| x.id) != Some(r.id) {
                bail!("{m}: LM segment document {d} is id {}, the Parquet has {:?}", r.id, items.get(d).map(|x| x.id));
            }
            d += 1;
        }
    }
    let base = fill(&spec.embeddings, m);
    let npy_path = PathBuf::from(format!("{}.f16.npy", base.display()));
    let ids_path = PathBuf::from(format!("{}.ids.npy", base.display()));
    let (shape, descr, map, off) = crate::vecs::read_npy(&npy_path)?;
    let (ishape, idescr, imap, ioff) = crate::vecs::read_npy(&ids_path)?;
    if descr != "<f2" || shape.len() != 2 || shape[1] < spec.bits || idescr != "<u4" || ishape[0] != shape[0] {
        bail!("{m}: embeddings must be float16 N x >= {} with uint32 ids", spec.bits);
    }
    // exactly the items with a prompt have an embedding
    let prompt: Vec<bool> = items.par_iter().map(crate::prompt::has_prompt).collect();
    let want: std::collections::HashSet<u32> = items.iter().zip(&prompt).filter(|x| *x.1).map(|x| x.0.id).collect();
    let mut emb = HashMap::with_capacity(shape[0]);
    for (row, c) in imap[ioff..].chunks_exact(4).enumerate() {
        let id = u32::from_le_bytes(c.try_into().unwrap());
        if !want.contains(&id) {
            bail!("{m}: embedding for id {id}, which is not an item of the month with a prompt");
        }
        if emb.insert(id, row).is_some() {
            bail!("{m}: two embeddings for id {id}");
        }
    }
    if emb.len() != want.len() {
        bail!("{m}: {} items have a prompt and {} have an embedding", want.len(), emb.len());
    }
    Ok(Some(Month { name: m.to_string(), parquet, items, kids, lm, emb, npy: (shape, map, off) }))
}

/// Sign bits of the first `bits` dimensions of a row (bit j of byte j/8 = dim j > 0).
fn sign_code(month: &Month, row: usize, bits: usize) -> Vec<u8> {
    let (shape, map, off) = &month.npy;
    let dim = shape[1];
    sign_bits(&map[off + row * dim * 2..off + (row * dim + bits) * 2], bits)
}

/// Sign rule 1: bit j (bit j % 8 of byte j / 8) = 1 when float16 dimension j is > 0.
pub(crate) fn sign_bits(row_f16: &[u8], bits: usize) -> Vec<u8> {
    let mut code = vec![0u8; bits / 8];
    for j in 0..bits {
        if f16::from_le_bytes([row_f16[2 * j], row_f16[2 * j + 1]]).to_f32() > 0.0 {
            code[j / 8] |= 1 << (j % 8);
        }
    }
    code
}

/// Checks a copied record against its source item (see `LmCodec::plan_record`).
pub(crate) fn check_record(codec: &LmCodec, it: &Item, rec: &[u8]) -> Result<(&'static str, usize)> {
    match codec.plan_record(&it.title, &it.url, &it.text) {
        RecPlan::Empty => {
            // what encode_record writes for an empty record: the header alone (mode 0, no tokens)
            if rec != [0u8] {
                bail!("id {}: expected an empty record", it.id);
            }
            Ok(("empty", 0))
        }
        RecPlan::Fallback(r) => {
            if r != rec {
                bail!("id {}: the zstd record differs from the item", it.id);
            }
            Ok(("fallback", 0))
        }
        RecPlan::Lm { ids, head } => {
            if !rec.starts_with(&head) || rec.len() <= head.len() {
                bail!("id {}: the LM record header or check byte differs from the item", it.id);
            }
            Ok(("lm", ids.len()))
        }
    }
}

/// Builds the block. Returns a report (sizes per part, counts).
pub fn pack(spec: &PackSpec, out: &Path) -> Result<serde_json::Value> {
    let t0 = Instant::now();
    if spec.bits == 0 || spec.bits % 64 != 0 {
        bail!("bits must be a multiple of 64");
    }
    // globals: author table, term classes
    let authors = crate::globals::load_authors(&spec.globals)?;
    let author_id: HashMap<&str, u32> = authors.iter().enumerate().map(|(i, a)| (a.as_str(), i as u32)).collect();
    let classes = crate::globals::load_df(&spec.globals)?;
    let gj: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(spec.globals.join("globals.json"))?)?;
    let n_all = gj["n_docs"].as_u64().context("globals.json: n_docs")?;
    let max_df_abs = (spec.max_df_ratio * n_all as f64).ceil() as u64;
    // the records are checked with the model and the tokenizer the block stores
    let lm_bytes = std::fs::read(&spec.lm_model).with_context(|| format!("{}", spec.lm_model.display()))?;
    let lm_tok = std::fs::read(&spec.lm_tokenizer).with_context(|| format!("{}", spec.lm_tokenizer.display()))?;
    let codec = LmCodec::from_parts(lm_bytes.clone(), Some(&lm_tok))?;
    eprintln!("pack: models and globals loaded ({:.0} s)", t0.elapsed().as_secs_f64());
    let months: Vec<Month> = spec.months.iter().map(|m| load_month(spec, m, &codec)).collect::<Result<Vec<_>>>()?.into_iter().flatten().collect();
    eprintln!("pack: {} months loaded ({:.0} s)", months.len(), t0.elapsed().as_secs_f64());

    // document order: (time, id); "late" = time later than some item with a higher id
    let mut docs: Vec<(usize, usize)> = months.iter().enumerate().flat_map(|(mi, m)| (0..m.items.len()).map(move |l| (mi, l))).collect();
    let item = |&(mi, l): &(usize, usize)| -> &Item { &months[mi].items[l] };
    let mut by_id: Vec<(u32, i64, usize)> = docs.iter().enumerate().map(|(i, d)| (item(d).id, item(d).time, i)).collect();
    by_id.sort_unstable();
    if by_id.windows(2).any(|w| w[0].0 == w[1].0) {
        bail!("an id appears in two months");
    }
    let mut late_of = vec![false; docs.len()];
    let mut smin = i64::MAX;
    for &(_, t, i) in by_id.iter().rev() {
        late_of[i] = t > smin;
        smin = smin.min(t);
    }
    let mut order: Vec<usize> = (0..docs.len()).collect();
    order.sort_by_key(|&i| (item(&docs[i]).time, item(&docs[i]).id));
    let late: Vec<bool> = order.iter().map(|&i| late_of[i]).collect();
    docs = order.iter().map(|&i| docs[i]).collect();
    let n = docs.len();
    if n == 0 {
        bail!("nothing to pack: the months of the spec have no items");
    }

    // every copied record against its item (in parallel)
    let checks: Vec<(&'static str, usize)> = docs
        .par_iter()
        .map(|d| {
            let rec = months[d.0].lm.raw(d.1)?;
            check_record(&codec, item(d), &rec).with_context(|| format!("month {}", months[d.0].name))
        })
        .collect::<Result<_>>()?;
    eprintln!("pack: {n} records checked against their items ({:.0} s)", t0.elapsed().as_secs_f64());

    // metadata, text, vectors, kids, word index, in document order
    let mut meta = MetaBlockWriter::new();
    let mut text = TextGroupWriter::new();
    let mut has = vec![0u64; n.div_ceil(64)];
    let mut codes: Vec<u8> = Vec::new();
    let (mut ids, mut parents, mut kids) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    let mut lex = LexWriter::new(&classes, max_df_abs, spec.max_df_ratio as f32);
    let mut batch: Vec<String> = Vec::new();
    let (mut n_deleted, mut n_with_vector) = (0u64, 0u64);
    // the prompt prefix of the documents with a vector whose record has no LM tokens (P4 rebuilds
    // their prompt from it instead of decoding): u32 doc, u8 flags, varint length, bytes
    let mut prefix_fb: Vec<u8> = Vec::new();
    let mut n_prefix_fb = 0u32;
    for (doc, d) in docs.iter().enumerate() {
        let it = item(d);
        let author = *author_id.get(it.by.as_str()).with_context(|| format!("id {}: author {:?} is not in the global table", it.id, it.by))?;
        n_deleted += it.deleted as u64;
        let row = MetaRow {
            id: it.id,
            time: it.time,
            typ: it.typ,
            dead: it.dead,
            deleted: it.deleted,
            parent: it.parent,
            author,
            score: it.score,
            descendants: it.descendants,
            poll: it.poll,
            parts: it.parts.clone(),
        };
        meta.push(row, late[doc])?;
        text.push(months[d.0].lm.raw(d.1)?.into_owned());
        if let Some(&row) = months[d.0].emb.get(&it.id) {
            has[doc / 64] |= 1 << (doc % 64);
            codes.extend(sign_code(&months[d.0], row, spec.bits));
            n_with_vector += 1;
            if checks[doc].0 == "fallback" {
                let (p, flags) = codec.prefix(&it.title, &it.url, &it.text).context("prefix of a record with a prompt")?;
                prefix_fb.extend_from_slice(&(doc as u32).to_le_bytes());
                prefix_fb.push(flags as u8);
                crate::varint::put(&mut prefix_fb, p.len() as u64);
                prefix_fb.extend_from_slice(p.as_bytes());
                n_prefix_fb += 1;
            }
        }
        ids.push(it.id);
        parents.push(it.parent);
        kids.push(months[d.0].kids[d.1].clone());
        batch.push(crate::segment::index_text(&it.title, &it.url, &it.text));
        if batch.len() == 65536 {
            lex.add_batch(&batch);
            batch.clear();
        }
    }
    lex.add_batch(&batch);
    let (meta_blocks, meta_index, flags, late_ex) = meta.finish()?;
    let (groups, text_idx, gsum) = text.finish();
    let children = crate::kids::children_of(&ids, &parents);
    let (kids_frames, kids_idx) = crate::kids::encode(&ids, &kids, &children)?;
    let lp = lex.finish()?;
    eprintln!("pack: parts built ({:.0} s)", t0.elapsed().as_secs_f64());

    // canaries: a few records decoded at open (short) or by `full` (long, fallback, empty)
    let pick = |want: &dyn Fn(usize) -> bool, k: usize, key: &dyn Fn(usize) -> u64| -> Vec<usize> {
        let mut c: Vec<usize> = (0..n).filter(|&d| want(d)).collect();
        c.sort_by_key(|&d| key(d));
        c.truncate(k);
        c
    };
    let hkey = |d: usize| XxHash3_64::oneshot(format!("{d}canary").as_bytes());
    let mut canaries = Vec::new();
    let short = pick(&|d| checks[d].0 == "lm" && (24..=64).contains(&checks[d].1), 4, &hkey);
    let long = pick(&|d| checks[d].0 == "lm" && checks[d].1 <= 2000, 1, &|d| (checks[d].1 as i64 - 1500).unsigned_abs());
    let fb = pick(&|d| checks[d].0 == "fallback", 1, &hkey);
    let empty = pick(&|d| checks[d].0 == "empty", 1, &hkey);
    for (kind, set) in [("short", &short), ("long", &long), ("fallback", &fb), ("empty", &empty)] {
        for &d in set.iter() {
            let it = item(&docs[d]);
            let rec = months[docs[d].0].lm.raw(docs[d].1)?;
            canaries.push(serde_json::json!({
                "doc": d, "id": it.id, "kind": kind, "ntok": checks[d].1, "rec": xxh(&rec),
                "text": xxh(format!("{}\0{}\0{}", it.title, it.url, it.text).as_bytes()),
            }));
        }
    }
    let count = |k: &str| checks.iter().filter(|c| c.0 == k).count();
    let tokens: u64 = checks.iter().map(|c| c.1 as u64).sum();

    // write the block: always-loaded parts first, then the large ones
    std::fs::create_dir_all(out)?;
    let mut w = BlockWriter::create(&out.join(crate::block::FILE_NAME))?;
    let mut fb = n_prefix_fb.to_le_bytes().to_vec();
    fb.extend_from_slice(&prefix_fb);
    let authors_txt = authors.join("\n");
    let parts: Vec<(&str, Codec, Access, &[u8])> = vec![
        ("meta/index", Codec::Zstd, Access::Load, &meta_index),
        ("meta/flags", Codec::Zstd, Access::Load, &flags),
        ("meta/late", Codec::Zstd, Access::Load, &late_ex),
        ("meta/authors", Codec::Zstd, Access::Load, authors_txt.as_bytes()),
        ("text/idx", Codec::Zstd, Access::Load, &text_idx),
        ("text/gsum", Codec::Raw, Access::Load, &gsum),
        ("text/prefix_fallback", Codec::Zstd, Access::Load, &fb),
        ("vec/has", Codec::Zstd, Access::Load, bytemuck_u64(&has)),
        ("lex/fst", Codec::Raw, Access::Load, &lp.fst),
        ("lex/norms", Codec::Zstd, Access::Load, &lp.norms),
        ("model/lm", Codec::Raw, Access::Load, &lm_bytes),
        ("vec/codes", Codec::Raw, Access::Scan, &codes),
        ("meta/blocks", Codec::Raw, Access::Random, &meta_blocks),
        ("lex/post", Codec::Raw, Access::Random, &lp.post),
        ("text/groups", Codec::Raw, Access::Random, &groups),
        ("meta/kids", Codec::Raw, Access::Cold, &kids_frames),
        ("meta/kids.idx", Codec::Zstd, Access::Cold, &kids_idx),
    ];
    for (name, codec, access, bytes) in &parts {
        w.add(PartSpec::new(name, *codec, *access), bytes)?;
    }
    let embed_tok = std::fs::read(&spec.embed_tokenizer)?;
    w.add(PartSpec::new("model/embed.tok", Codec::Zstd, Access::Cold), &embed_tok)?;
    w.add(PartSpec::new("model/lm.tok", Codec::Zstd, Access::Cold), &lm_tok)?;
    for (ext, name) in [("xml", "embed.xml"), ("bin", "embed.bin")] {
        let src = PathBuf::from(format!("{}.{ext}", spec.embed_ir.display()));
        std::fs::copy(&src, out.join(name)).with_context(|| format!("{}", src.display()))?;
        w.add_external(name, &out.join(name), if ext == "xml" { "query_ir" } else { "query_ir_weights" })?;
    }
    let files: Vec<serde_json::Value> = months
        .iter()
        .map(|m| {
            let p = m.parquet.to_string_lossy().replace('\\', "/");
            let g = gj["files"].as_array().and_then(|f| f.iter().find(|x| x["path"] == p.as_str())).cloned();
            serde_json::json!({"path": p, "rows": m.items.len(), "sha256": g.as_ref().and_then(|x| x.get("sha256").cloned()),
                               "rev": g.as_ref().and_then(|x| x.get("rev").cloned())})
        })
        .collect();
    let (t_min, t_max) = (item(&docs[0]).time, item(&docs[n - 1]).time);
    let manifest = serde_json::json!({
        "profile": "compact",
        "docnum": "documents in (time, id) order; deleted rows kept as metadata-only documents",
        "snapshot": {"dataset": "open-index/hacker-news", "files": files, "n_docs": n, "n_deleted": n_deleted,
                     "id_min": by_id[0].0, "id_max": by_id[n - 1].0, "time_min": t_min, "time_max": t_max,
                     "n_lm": count("lm"), "n_fallback": count("fallback"), "n_empty": count("empty"),
                     "n_with_vector": n_with_vector, "n_late": late.iter().filter(|&&l| l).count()},
        "meta": {"block_rows": crate::meta::META_BLOCK, "row_encoding": 2, "frame_checksum": true,
                 "authors": {"n": authors.len(), "order": "count desc, name asc", "source": gj["authors_xxh3"]}},
        "kids": {"encoding": 1, "block": crate::kids::BLOCK},
        "text": {"codec": "lm", "group": 32, "framing": 1, "rec_check": 1, "math_version": crate::lm::model::MATH_VERSION,
                 "transform_version": 1, "max_tokens": 2000, "model_id": codec.model.id, "lm_records": count("lm"),
                 "zstd_records": count("fallback"), "tokens": tokens, "gsum": "xxh3_64 low 32"},
        "vec": {"bits": spec.bits, "source_dims": 768, "sign_rule": 1, "layout": "sparse", "prompt_version": spec.prompt_version,
                "prefix_tokens": crate::prompt::PREFIX_TOKENS, "max_seq": crate::prompt::MAX_SEQ,
                "doc_prompt": "title: <title or none> | text: <prefix body as plain text>", "query_prompt": crate::prompt::QUERY_PROMPT,
                "prefix_fallback": n_prefix_fb},
        "lex": {"info": lp.info, "tokenizer_version": 1, "value_layout": 1,
                "class_source": {"n_docs": n_all, "df": gj["df_xxh3"]},
                "terms": {"frequent": lp.classes.0, "single": lp.classes.1, "list": lp.classes.2}},
        "search": {"c1": spec.c1},
        "canaries": canaries,
        "build": {"hnz": env!("CARGO_PKG_VERSION"), "spec": serde_json::to_value(spec)?},
    });
    let snap = w.finish(manifest, n as u64, crate::lm::model::MATH_VERSION, concat!("hnz ", env!("CARGO_PKG_VERSION")))?;
    let block = crate::block::Block::open(out)?;
    let sizes: serde_json::Map<String, serde_json::Value> = block.parts().iter().map(|p| (p.name.clone(), serde_json::json!(p.len))).collect();
    let total: u64 = std::fs::metadata(&block.path)?.len() + block.external().iter().map(|e| e.len).sum::<u64>();
    Ok(serde_json::json!({
        "out": out.display().to_string(), "snapshot": format!("{snap:032x}"), "n_docs": n, "n_late": late.iter().filter(|&&l| l).count(),
        "records": {"lm": count("lm"), "fallback": count("fallback"), "empty": count("empty")}, "n_with_vector": n_with_vector,
        "bytes": {"container": std::fs::metadata(&block.path)?.len(), "total_with_ir": total, "parts": sizes},
        "seconds": t0.elapsed().as_secs_f64(),
    }))
}

/// The first `k` candidates by (key, document): what a stable sort by key of the candidates in
/// document order, cut to `k`, gives (`pack`'s choice of canaries).
struct TopK {
    k: usize,
    v: Vec<(u64, usize, serde_json::Value)>,
}

impl TopK {
    fn new(k: usize) -> Self {
        Self { k, v: Vec::new() }
    }

    fn offer(&mut self, key: u64, doc: usize, make: impl FnOnce() -> serde_json::Value) {
        if self.v.len() == self.k && (key, doc) >= (self.v[self.k - 1].0, self.v[self.k - 1].1) {
            return;
        }
        let at = self.v.partition_point(|x| (x.0, x.1) < (key, doc));
        self.v.insert(at, (key, doc, make()));
        self.v.truncate(self.k);
    }
}

/// `pack` for blocks too big to hold in memory (all of HN, P5): the same block, byte for byte,
/// built month by month. A first pass reads every month's items for the children of each
/// document (a reply may be in another month than its item) and the checks of the whole spec;
/// the second one loads one month at a time (items, LM segment, embeddings), in time order, and
/// writes the two large parts (text groups, vector codes) to files in `tmp` (deleted at the end).
/// The months of the spec must be in time order.
pub fn pack_stream(spec: &PackSpec, out: &Path, tmp: &Path) -> Result<serde_json::Value> {
    use std::io::Write;
    let t0 = Instant::now();
    if spec.bits == 0 || spec.bits % 64 != 0 {
        bail!("bits must be a multiple of 64");
    }
    // months in time order (documents within a month go in (time, id) order: the global order)
    let mut prev_end = None;
    for m in &spec.months {
        let (Some(a), Some(b)) = (crate::dates::parse(m, false), crate::dates::parse(m, true)) else { bail!("month {m:?} is not YYYY-MM") };
        if prev_end.is_some_and(|e| a < e) {
            bail!("the months of the spec must be in time order and not repeat ({m})");
        }
        prev_end = Some(b);
    }
    let authors = crate::globals::load_authors(&spec.globals)?;
    let author_id: HashMap<&str, u32> = authors.iter().enumerate().map(|(i, a)| (a.as_str(), i as u32)).collect();
    let classes = crate::globals::load_df(&spec.globals)?;
    let gj: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(spec.globals.join("globals.json"))?)?;
    let n_all = gj["n_docs"].as_u64().context("globals.json: n_docs")?;
    let max_df_abs = (spec.max_df_ratio * n_all as f64).ceil() as u64;
    let lm_bytes = std::fs::read(&spec.lm_model).with_context(|| format!("{}", spec.lm_model.display()))?;
    let lm_tok = std::fs::read(&spec.lm_tokenizer).with_context(|| format!("{}", spec.lm_tokenizer.display()))?;
    let codec = LmCodec::from_parts(lm_bytes.clone(), Some(&lm_tok))?;
    eprintln!("pack: models and globals loaded ({:.0} s)", t0.elapsed().as_secs_f64());

    // pass 1: every month's (id, parent), ids unique across months, items inside their month
    let mut children = crate::kids::ChildIndex::default();
    let mut seen: Vec<u64> = Vec::new();
    let mut n = 0usize;
    for m in &spec.months {
        let items = read_parquet(&fill(&spec.parquet, m))?.items;
        let (t_a, t_b) = (crate::dates::parse(m, false).unwrap(), crate::dates::parse(m, true).unwrap());
        for it in &items {
            if it.time < t_a || it.time >= t_b {
                bail!("{m}: id {} has time {} ({}), outside the month", it.id, it.time, crate::dates::fmt(it.time));
            }
            let (w, b) = (it.id as usize / 64, it.id % 64);
            if w >= seen.len() {
                seen.resize(w + 1 + (w >> 3), 0);
            }
            if seen[w] >> b & 1 == 1 {
                bail!("an id appears in two months");
            }
            seen[w] |= 1 << b;
            children.add(it.id, it.parent);
        }
        n += items.len();
    }
    drop(seen);
    children.finish();
    if n == 0 {
        bail!("nothing to pack: the months of the spec have no items");
    }
    if n > u32::MAX as usize {
        bail!("{n} documents: more than a block can number");
    }
    eprintln!("pack: {n} documents in {} months, children indexed ({:.0} s)", spec.months.len(), t0.elapsed().as_secs_f64());

    // pass 2: one month at a time
    std::fs::create_dir_all(tmp)?;
    let (groups_path, codes_path) = (tmp.join("text.groups"), tmp.join("vec.codes"));
    let mut groups_w = std::io::BufWriter::with_capacity(8 << 20, std::fs::File::create(&groups_path)?);
    let mut codes_w = std::io::BufWriter::with_capacity(8 << 20, std::fs::File::create(&codes_path)?);
    let mut meta = MetaBlockWriter::new();
    let mut text = TextGroupWriter::new();
    let mut has = vec![0u64; n.div_ceil(64)];
    let mut kw = crate::kids::KidsWriter::new();
    let mut lex = LexWriter::new(&classes, max_df_abs, spec.max_df_ratio as f32);
    let mut batch: Vec<String> = Vec::new();
    let mut prefix_fb: Vec<u8> = Vec::new();
    let (mut n_prefix_fb, mut n_deleted, mut n_with_vector, mut n_late) = (0u32, 0u64, 0u64, 0usize);
    let (mut n_lm, mut n_fallback, mut n_empty, mut tokens) = (0usize, 0usize, 0usize, 0u64);
    let (mut id_min, mut id_max, mut max_id) = (u32::MAX, 0u32, 0u32);
    let (mut time_min, mut time_max) = (None::<i64>, 0i64);
    let hkey = |d: usize| XxHash3_64::oneshot(format!("{d}canary").as_bytes());
    let (mut c_short, mut c_long, mut c_fb, mut c_empty) = (TopK::new(4), TopK::new(1), TopK::new(1), TopK::new(1));
    let mut files: Vec<serde_json::Value> = Vec::new();
    let mut scratch: Vec<u32> = Vec::new();
    let mut base = 0usize;
    for m in &spec.months {
        let Some(month) = load_month(spec, m, &codec)? else { continue };
        let items = &month.items;
        let mut order: Vec<usize> = (0..items.len()).collect();
        order.sort_by_key(|&l| (items[l].time, items[l].id));
        let checks: Vec<(&'static str, usize)> = order
            .par_iter()
            .map(|&l| check_record(&codec, &items[l], &month.lm.raw(l)?).with_context(|| format!("month {}", month.name)))
            .collect::<Result<_>>()?;
        for (j, &l) in order.iter().enumerate() {
            let doc = base + j;
            let it = &items[l];
            let author = *author_id.get(it.by.as_str()).with_context(|| format!("id {}: author {:?} is not in the global table", it.id, it.by))?;
            n_deleted += it.deleted as u64;
            // late: a document with a higher id comes earlier in (time, id) order
            let late = it.id < max_id;
            max_id = max_id.max(it.id);
            n_late += late as usize;
            let row = MetaRow {
                id: it.id,
                time: it.time,
                typ: it.typ,
                dead: it.dead,
                deleted: it.deleted,
                parent: it.parent,
                author,
                score: it.score,
                descendants: it.descendants,
                poll: it.poll,
                parts: it.parts.clone(),
            };
            meta.push(row, late)?;
            let rec = month.lm.raw(l)?;
            let (kind, ntok) = checks[j];
            match kind {
                "lm" => n_lm += 1,
                "fallback" => n_fallback += 1,
                _ => n_empty += 1,
            }
            tokens += ntok as u64;
            if let Some(&row) = month.emb.get(&it.id) {
                has[doc / 64] |= 1 << (doc % 64);
                codes_w.write_all(&sign_code(&month, row, spec.bits))?;
                n_with_vector += 1;
                if kind == "fallback" {
                    let (p, flags) = codec.prefix(&it.title, &it.url, &it.text).context("prefix of a record with a prompt")?;
                    prefix_fb.extend_from_slice(&(doc as u32).to_le_bytes());
                    prefix_fb.push(flags as u8);
                    crate::varint::put(&mut prefix_fb, p.len() as u64);
                    prefix_fb.extend_from_slice(p.as_bytes());
                    n_prefix_fb += 1;
                }
            }
            // canaries, as `pack` picks them
            let canary = |kind: &str| {
                serde_json::json!({
                    "doc": doc, "id": it.id, "kind": kind, "ntok": ntok, "rec": xxh(&rec),
                    "text": xxh(format!("{}\0{}\0{}", it.title, it.url, it.text).as_bytes()),
                })
            };
            if kind == "lm" && (24..=64).contains(&ntok) {
                c_short.offer(hkey(doc), doc, || canary("short"));
            }
            if kind == "lm" && ntok <= 2000 {
                c_long.offer((ntok as i64 - 1500).unsigned_abs(), doc, || canary("long"));
            }
            if kind == "fallback" {
                c_fb.offer(hkey(doc), doc, || canary("fallback"));
            }
            if kind == "empty" {
                c_empty.offer(hkey(doc), doc, || canary("empty"));
            }
            text.push(rec.into_owned());
            let kids = &month.kids[l];
            if kids.is_empty() {
                kw.push(it.id, &[], &[])?;
            } else {
                children.children(it.id, &mut scratch);
                kw.push(it.id, kids, &scratch)?;
            }
            batch.push(crate::segment::index_text(&it.title, &it.url, &it.text));
            if batch.len() == 65536 {
                lex.add_batch(&batch);
                batch.clear();
            }
            id_min = id_min.min(it.id);
            id_max = id_max.max(it.id);
            time_min.get_or_insert(it.time);
            time_max = it.time;
        }
        groups_w.write_all(&text.take_groups())?;
        let p = month.parquet.to_string_lossy().replace('\\', "/");
        let g = gj["files"].as_array().and_then(|f| f.iter().find(|x| x["path"] == p.as_str())).cloned();
        files.push(serde_json::json!({"path": p, "rows": items.len(), "sha256": g.as_ref().and_then(|x| x.get("sha256").cloned()),
                                      "rev": g.as_ref().and_then(|x| x.get("rev").cloned())}));
        base += items.len();
        eprintln!("pack: {m} ({} documents, {base} of {n}, {:.0} s)", items.len(), t0.elapsed().as_secs_f64());
    }
    if base != n {
        bail!("the second pass read {base} documents, the first {n}");
    }
    drop(children);
    lex.add_batch(&batch);
    drop(batch);
    let (meta_blocks, meta_index, flags, late_ex) = meta.finish()?;
    let (tail, text_idx, gsum) = text.finish();
    groups_w.write_all(&tail)?;
    groups_w.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    codes_w.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    let (kids_frames, kids_idx) = kw.finish()?;
    let lp = lex.finish()?;
    eprintln!("pack: parts built ({:.0} s)", t0.elapsed().as_secs_f64());
    let mut canaries = Vec::new();
    for t in [c_short, c_long, c_fb, c_empty] {
        canaries.extend(t.v.into_iter().map(|x| x.2));
    }

    // the block, part by part in `pack`'s order
    std::fs::create_dir_all(out)?;
    let mut w = BlockWriter::create(&out.join(crate::block::FILE_NAME))?;
    let mut fb = n_prefix_fb.to_le_bytes().to_vec();
    fb.extend_from_slice(&prefix_fb);
    let authors_txt = authors.join("\n");
    enum Src<'a> {
        Mem(&'a [u8]),
        File(&'a Path),
    }
    let parts: Vec<(&str, Codec, Access, Src)> = vec![
        ("meta/index", Codec::Zstd, Access::Load, Src::Mem(&meta_index)),
        ("meta/flags", Codec::Zstd, Access::Load, Src::Mem(&flags)),
        ("meta/late", Codec::Zstd, Access::Load, Src::Mem(&late_ex)),
        ("meta/authors", Codec::Zstd, Access::Load, Src::Mem(authors_txt.as_bytes())),
        ("text/idx", Codec::Zstd, Access::Load, Src::Mem(&text_idx)),
        ("text/gsum", Codec::Raw, Access::Load, Src::Mem(&gsum)),
        ("text/prefix_fallback", Codec::Zstd, Access::Load, Src::Mem(&fb)),
        ("vec/has", Codec::Zstd, Access::Load, Src::Mem(bytemuck_u64(&has))),
        ("lex/fst", Codec::Raw, Access::Load, Src::Mem(&lp.fst)),
        ("lex/norms", Codec::Zstd, Access::Load, Src::Mem(&lp.norms)),
        ("model/lm", Codec::Raw, Access::Load, Src::Mem(&lm_bytes)),
        ("vec/codes", Codec::Raw, Access::Scan, Src::File(&codes_path)),
        ("meta/blocks", Codec::Raw, Access::Random, Src::Mem(&meta_blocks)),
        ("lex/post", Codec::Raw, Access::Random, Src::Mem(&lp.post)),
        ("text/groups", Codec::Raw, Access::Random, Src::File(&groups_path)),
        ("meta/kids", Codec::Raw, Access::Cold, Src::Mem(&kids_frames)),
        ("meta/kids.idx", Codec::Zstd, Access::Cold, Src::Mem(&kids_idx)),
    ];
    for (name, codec, access, src) in &parts {
        match src {
            Src::Mem(bytes) => w.add(PartSpec::new(name, *codec, *access), bytes)?,
            Src::File(path) => w.add_file(PartSpec::new(name, *codec, *access), path)?,
        };
    }
    let embed_tok = std::fs::read(&spec.embed_tokenizer)?;
    w.add(PartSpec::new("model/embed.tok", Codec::Zstd, Access::Cold), &embed_tok)?;
    w.add(PartSpec::new("model/lm.tok", Codec::Zstd, Access::Cold), &lm_tok)?;
    for (ext, name) in [("xml", "embed.xml"), ("bin", "embed.bin")] {
        let src = PathBuf::from(format!("{}.{ext}", spec.embed_ir.display()));
        std::fs::copy(&src, out.join(name)).with_context(|| format!("{}", src.display()))?;
        w.add_external(name, &out.join(name), if ext == "xml" { "query_ir" } else { "query_ir_weights" })?;
    }
    let manifest = serde_json::json!({
        "profile": "compact",
        "docnum": "documents in (time, id) order; deleted rows kept as metadata-only documents",
        "snapshot": {"dataset": "open-index/hacker-news", "files": files, "n_docs": n, "n_deleted": n_deleted,
                     "id_min": id_min, "id_max": id_max, "time_min": time_min, "time_max": time_max,
                     "n_lm": n_lm, "n_fallback": n_fallback, "n_empty": n_empty,
                     "n_with_vector": n_with_vector, "n_late": n_late},
        "meta": {"block_rows": crate::meta::META_BLOCK, "row_encoding": 2, "frame_checksum": true,
                 "authors": {"n": authors.len(), "order": "count desc, name asc", "source": gj["authors_xxh3"]}},
        "kids": {"encoding": 1, "block": crate::kids::BLOCK},
        "text": {"codec": "lm", "group": 32, "framing": 1, "rec_check": 1, "math_version": crate::lm::model::MATH_VERSION,
                 "transform_version": 1, "max_tokens": 2000, "model_id": codec.model.id, "lm_records": n_lm,
                 "zstd_records": n_fallback, "tokens": tokens, "gsum": "xxh3_64 low 32"},
        "vec": {"bits": spec.bits, "source_dims": 768, "sign_rule": 1, "layout": "sparse", "prompt_version": spec.prompt_version,
                "prefix_tokens": crate::prompt::PREFIX_TOKENS, "max_seq": crate::prompt::MAX_SEQ,
                "doc_prompt": "title: <title or none> | text: <prefix body as plain text>", "query_prompt": crate::prompt::QUERY_PROMPT,
                "prefix_fallback": n_prefix_fb},
        "lex": {"info": lp.info, "tokenizer_version": 1, "value_layout": 1,
                "class_source": {"n_docs": n_all, "df": gj["df_xxh3"]},
                "terms": {"frequent": lp.classes.0, "single": lp.classes.1, "list": lp.classes.2}},
        "search": {"c1": spec.c1},
        "canaries": canaries,
        "build": {"hnz": env!("CARGO_PKG_VERSION"), "spec": serde_json::to_value(spec)?},
    });
    let snap = w.finish(manifest, n as u64, crate::lm::model::MATH_VERSION, concat!("hnz ", env!("CARGO_PKG_VERSION")))?;
    for p in [&groups_path, &codes_path] {
        let _ = std::fs::remove_file(p);
    }
    let _ = std::fs::remove_dir(tmp);
    let block = crate::block::Block::open(out)?;
    let sizes: serde_json::Map<String, serde_json::Value> = block.parts().iter().map(|p| (p.name.clone(), serde_json::json!(p.len))).collect();
    let total: u64 = std::fs::metadata(&block.path)?.len() + block.external().iter().map(|e| e.len).sum::<u64>();
    Ok(serde_json::json!({
        "out": out.display().to_string(), "snapshot": format!("{snap:032x}"), "n_docs": n, "n_late": n_late,
        "records": {"lm": n_lm, "fallback": n_fallback, "empty": n_empty}, "n_with_vector": n_with_vector,
        "bytes": {"container": std::fs::metadata(&block.path)?.len(), "total_with_ir": total, "parts": sizes},
        "seconds": t0.elapsed().as_secs_f64(),
    }))
}

fn bytemuck_u64(v: &[u64]) -> &[u8] {
    // SAFETY: u64 has no padding; little-endian is the block's byte order (x86-64)
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 8) }
}
