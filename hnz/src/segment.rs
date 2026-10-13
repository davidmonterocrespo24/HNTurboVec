//! Segmento: un trozo inmutable y autocontenido del archivo (p.ej. un año), con metadatos,
//! texto, índice BM25 y códigos vectoriales. Los docnums son locales y siguen el orden de id
//! (= orden temporal), así un rango de fechas es un rango contiguo de documentos.

use crate::item::{read_parquet, Item};
use crate::meta::{self, MetaRow, MetaStore};
use crate::textstore::{self, TextStore};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub version: u32,
    pub name: String,
    pub n_docs: usize,
    pub deleted_skipped: usize,
    pub id_min: u32,
    pub id_max: u32,
    pub time_min: i64,
    pub time_max: i64,
    /// "zstd" (caliente/templado) o "lm" (frío)
    pub text_tier: String,
    pub text_block: usize,
    pub text_level: i32,
    pub transformed_pct: f64,
    #[serde(default)]
    pub has_bm25: bool,
    #[serde(default)]
    pub vec_bits: usize,
}

pub fn read_manifest(dir: &Path) -> Result<Manifest> {
    let s = std::fs::read_to_string(dir.join("manifest.json")).with_context(|| format!("{}/manifest.json", dir.display()))?;
    Ok(serde_json::from_str(&s)?)
}

pub fn write_manifest(dir: &Path, m: &Manifest) -> Result<()> {
    std::fs::write(dir.join("manifest.json"), serde_json::to_string_pretty(m)?)?;
    Ok(())
}

pub fn load_items(parquets: &[PathBuf]) -> Result<(Vec<Item>, usize)> {
    let mut items = Vec::new();
    let mut deleted = 0;
    for p in parquets {
        let l = read_parquet(p)?;
        deleted += l.deleted;
        items.extend(l.items);
    }
    items.sort_by_key(|x| x.id);
    items.dedup_by_key(|x| x.id);
    Ok((items, deleted))
}

/// Construye meta + texto (nivel zstd) de un segmento.
pub fn build_store(items: &[Item], deleted: usize, out: &Path, name: &str, block: usize, level: i32) -> Result<Manifest> {
    if items.is_empty() {
        bail!("segmento vacío");
    }
    std::fs::create_dir_all(out)?;
    let t0 = Instant::now();
    // autores ordenados por frecuencia (los frecuentes reciben índices pequeños = varints de 1 byte)
    let mut freq: HashMap<&str, u32> = HashMap::new();
    for it in items {
        *freq.entry(it.by.as_str()).or_default() += 1;
    }
    let mut authors: Vec<(&str, u32)> = freq.into_iter().collect();
    authors.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    let aidx: HashMap<&str, u32> = authors.iter().enumerate().map(|(i, a)| (a.0, i as u32)).collect();
    let rows: Vec<MetaRow> = items
        .iter()
        .map(|it| MetaRow {
            id: it.id,
            time: it.time,
            typ: it.typ,
            dead: it.dead,
            parent: it.parent,
            author: aidx[it.by.as_str()],
            score: it.score,
            descendants: it.descendants,
            poll: it.poll,
            parts: it.parts.clone(),
        })
        .collect();
    let names: Vec<String> = authors.iter().map(|a| a.0.to_string()).collect();
    meta::write(out, &rows, &names)?;
    let t_meta = t0.elapsed();
    let recs: Vec<(&str, &str, &str)> = items.iter().map(|it| (it.title.as_str(), it.url.as_str(), it.text.as_str())).collect();
    let st = textstore::write(out, &recs, block, level, 112 * 1024)?;
    eprintln!(
        "  meta {:.1}s, texto {:.1}s; HTML transformado: moderno {:.1}%, antiguo {:.1}%, crudo {:.1}%",
        t_meta.as_secs_f64(),
        (t0.elapsed() - t_meta).as_secs_f64(),
        100.0 * st.modes[0] as f64 / items.len() as f64,
        100.0 * st.modes[1] as f64 / items.len() as f64,
        100.0 * st.modes[2] as f64 / items.len() as f64
    );
    let m = Manifest {
        version: 1,
        name: name.to_string(),
        n_docs: items.len(),
        deleted_skipped: deleted,
        id_min: items[0].id,
        id_max: items[items.len() - 1].id,
        time_min: items.iter().map(|i| i.time).min().unwrap(),
        time_max: items.iter().map(|i| i.time).max().unwrap(),
        text_tier: "zstd".into(),
        text_block: block,
        text_level: level,
        transformed_pct: 100.0 * (st.modes[0] + st.modes[1]) as f64 / items.len() as f64,
        has_bm25: false,
        vec_bits: 0,
    };
    write_manifest(out, &m)?;
    Ok(m)
}

pub struct Segment {
    pub dir: PathBuf,
    pub man: Manifest,
    pub meta: MetaStore,
    pub text: TextStore,
}

impl Segment {
    pub fn open(dir: &Path) -> Result<Self> {
        let man = read_manifest(dir)?;
        let meta = MetaStore::open(dir, man.n_docs)?;
        let text = TextStore::open(dir, man.text_block, man.n_docs)?;
        Ok(Self { dir: dir.to_path_buf(), man, meta, text })
    }

    pub fn item(&self, doc: usize) -> Result<Item> {
        let m = self.meta.row(doc);
        let t = self.text.get(doc)?;
        Ok(join(&m, t, &self.meta.authors))
    }
}

pub fn join(m: &MetaRow, t: textstore::TextRec, authors: &[String]) -> Item {
    Item {
        id: m.id,
        typ: m.typ,
        dead: m.dead,
        by: authors[m.author as usize].clone(),
        time: m.time,
        text: t.text,
        parent: m.parent,
        poll: m.poll,
        url: t.url,
        score: m.score,
        title: t.title,
        parts: m.parts.clone(),
        descendants: m.descendants,
    }
}

/// Tamaño en disco de cada componente del segmento.
pub fn sizes(dir: &Path) -> Result<Vec<(String, u64)>> {
    let mut v = Vec::new();
    for e in std::fs::read_dir(dir)? {
        let e = e?;
        if e.file_type()?.is_file() {
            v.push((e.file_name().to_string_lossy().to_string(), e.metadata()?.len()));
        }
    }
    v.sort();
    Ok(v)
}
