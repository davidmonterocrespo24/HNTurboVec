//! Global statistics of a set of Parquet files (all of HN), needed to build blocks whose ids and
//! term classes are global (P3): the document frequency of every index term, the author table
//! (ordered by number of items, then by name; the author id is the line number) and per-file
//! counts. One pass, in parallel by file (E30e).
use crate::segment;
use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Per-file counts.
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct FileStats {
    pub path: String,
    pub rows: u64,
    pub deleted: u64,
    pub id_min: u32,
    pub id_max: u32,
    /// from the download registry (data/_descargas.json), if given
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rev: Option<String>,
}

struct FilePart {
    stats: FileStats,
    df: Vec<(String, u32)>,
    authors: Vec<(String, u64)>,
}

fn one_file(p: &Path) -> Result<FilePart> {
    let items = crate::item::read_parquet(p)?.items;
    let mut df: HashMap<String, u32> = HashMap::new();
    let mut by: HashMap<String, u64> = HashMap::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut deleted = 0u64;
    for it in &items {
        deleted += it.deleted as u64;
        *by.entry(it.by.clone()).or_insert(0) += 1;
        seen.clear();
        crate::tokenize::for_each_term(&segment::index_text(&it.title, &it.url, &it.text), |t| {
            if !seen.contains(t) {
                seen.insert(t.to_string());
            }
        });
        for t in seen.drain() {
            *df.entry(t).or_insert(0) += 1;
        }
    }
    let stats = FileStats {
        path: p.to_string_lossy().replace('\\', "/"),
        rows: items.len() as u64,
        deleted,
        id_min: items.first().map_or(0, |x| x.id),
        id_max: items.last().map_or(0, |x| x.id),
        sha256: None,
        rev: None,
    };
    Ok(FilePart { stats, df: df.into_iter().collect(), authors: by.into_iter().collect() })
}

/// Runs the pass and writes `<out>/df.fst` (term -> df), `<out>/authors.zst` (names joined by
/// '\n') and `<out>/globals.json` (totals, rare-term counts per threshold, files). `registry`:
/// the download registry to copy each file's SHA-256 from.
pub fn run(out: &Path, parquet: &[PathBuf], registry: Option<&Path>) -> Result<serde_json::Value> {
    if parquet.is_empty() {
        bail!("no Parquet files given");
    }
    std::fs::create_dir_all(out)?;
    let t0 = Instant::now();
    let reg: serde_json::Value = match registry {
        Some(r) => serde_json::from_str(&std::fs::read_to_string(r).with_context(|| format!("{}", r.display()))?)?,
        None => serde_json::Value::Null,
    };
    // at most 8 months in memory at once
    let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build()?;
    let (tx, rx) = std::sync::mpsc::sync_channel::<Result<FilePart>>(2);
    let files = parquet.to_vec();
    let worker = std::thread::spawn(move || {
        pool.install(|| {
            files.par_iter().for_each_with(tx, |tx, p| {
                let _ = tx.send(one_file(p).with_context(|| format!("{}", p.display())));
            })
        })
    });
    let mut df: HashMap<String, u64> = HashMap::new();
    let mut authors: HashMap<String, u64> = HashMap::new();
    let mut stats: Vec<FileStats> = Vec::new();
    for r in rx {
        let part = r?;
        for (t, d) in part.df {
            *df.entry(t).or_insert(0) += d as u64;
        }
        for (a, n) in part.authors {
            *authors.entry(a).or_insert(0) += n;
        }
        eprintln!("  {}/{} {} ({} rows, {:.0} s)", stats.len() + 1, parquet.len(), part.stats.path, part.stats.rows, t0.elapsed().as_secs_f64());
        stats.push(part.stats);
    }
    worker.join().map_err(|_| anyhow::anyhow!("globals worker panicked"))?;
    // files are months by item time; HN ids are NOT always in time order (0.63 % of items have a
    // time later than items with higher ids, E32), so id ranges of different months can overlap
    stats.sort_by(|a, b| a.path.cmp(&b.path));
    for s in stats.iter_mut() {
        if let Some(e) = reg.get(&s.path) {
            s.sha256 = e.get("sha256").and_then(|v| v.as_str()).map(String::from);
            s.rev = e.get("rev").and_then(|v| v.as_str()).map(String::from);
        }
    }
    // df dictionary
    let mut terms: Vec<(String, u64)> = df.into_iter().collect();
    terms.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    let mut fb = fst::MapBuilder::new(std::io::BufWriter::new(std::fs::File::create(out.join("df.fst"))?))?;
    for (t, d) in &terms {
        fb.insert(t.as_bytes(), *d)?;
    }
    fb.finish()?;
    // author table: most frequent first, then by name (bytes)
    let mut au: Vec<(String, u64)> = authors.into_iter().collect();
    au.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.as_bytes().cmp(b.0.as_bytes())));
    if au.iter().any(|a| a.0.contains('\n')) {
        bail!("an author name contains a newline");
    }
    let joined = au.iter().map(|a| a.0.as_str()).collect::<Vec<_>>().join("\n");
    std::fs::write(out.join("authors.zst"), zstd::bulk::compress(joined.as_bytes(), 19)?)?;
    // rare-term counts per threshold (fraction of all items): terms kept and their (item, term) pairs
    let n_docs: u64 = stats.iter().map(|s| s.rows).sum();
    let total_postings: u64 = terms.iter().map(|x| x.1).sum();
    let thresholds: Vec<serde_json::Value> = [0.00005f64, 0.0001, 0.0002, 0.0005, 0.001, 0.01]
        .iter()
        .map(|&r| {
            let max_df = (r * n_docs as f64).ceil() as u64;
            let (nt, np, n1) = terms.iter().filter(|x| x.1 <= max_df).fold((0u64, 0u64, 0u64), |a, x| (a.0 + 1, a.1 + x.1, a.2 + (x.1 == 1) as u64));
            serde_json::json!({"ratio": r, "max_df": max_df, "terms": nt, "postings": np, "df1_terms": n1})
        })
        .collect();
    let h = |p: &Path| crate::block::hash_file(p).map(|(_, h)| format!("xxh3-128:{h:032x}"));
    let summary = serde_json::json!({
        "n_docs": n_docs, "n_terms": terms.len(), "total_postings": total_postings, "thresholds": thresholds,
        "n_authors": au.len(), "authors_xxh3": h(&out.join("authors.zst"))?, "df_xxh3": h(&out.join("df.fst"))?,
        "df_fst_bytes": std::fs::metadata(out.join("df.fst"))?.len(), "files": stats, "seconds": t0.elapsed().as_secs_f64(),
    });
    std::fs::write(out.join("globals.json"), serde_json::to_string_pretty(&summary)?)?;
    Ok(summary)
}

/// The author table written by `run` (author id = position).
pub fn load_authors(dir: &Path) -> Result<Vec<String>> {
    let raw = zstd::stream::decode_all(&std::fs::read(dir.join("authors.zst")).context("authors.zst")?[..])?;
    Ok(String::from_utf8(raw)?.split('\n').map(String::from).collect())
}

/// The term -> df dictionary written by `run`.
pub fn load_df(dir: &Path) -> Result<fst::Map<Vec<u8>>> {
    Ok(fst::Map::new(std::fs::read(dir.join("df.fst")).context("df.fst")?)?)
}
