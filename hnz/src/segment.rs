//! Segmento: un trozo inmutable y autocontenido del archivo (p.ej. un mes o un año), con
//! metadatos, texto (nivel zstd o nivel LM), índice BM25 y códigos vectoriales. Los docnums son
//! locales y siguen el orden de id (= orden temporal): un rango de fechas es un rango contiguo.

use crate::bm25;
use crate::item::{read_parquet, Item};
use crate::lm::store::{LmCodec, LmStore};
use crate::meta::{self, MetaRow, MetaStore};
use crate::textstore::{self, TextRec, TextStore};
use crate::htmlx;
use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub version: u32,
    pub name: String,
    pub n_docs: usize,
    /// nº de filas `deleted` (desde la versión 3 se CONSERVAN como metadatos; antes se omitían)
    pub deleted_skipped: usize,
    /// identificador único de esta construcción (invalida cachés de versiones anteriores)
    #[serde(default)]
    pub build_id: u64,
    pub id_min: u32,
    pub id_max: u32,
    pub time_min: i64,
    pub time_max: i64,
    /// "zstd" (caliente/templado) o "lm" (frío)
    pub text_tier: String,
    #[serde(default)]
    pub text_block: usize,
    #[serde(default)]
    pub text_level: i32,
    /// fichero del modelo LM (nivel frío), relativo a la carpeta de modelos
    #[serde(default)]
    pub lm_model: Option<String>,
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

/// Nivel del texto de un segmento.
pub enum Tier<'a> {
    Zstd { block: usize, level: i32 },
    Lm { codec: &'a LmCodec, model_file: String },
}

/// Texto que se indexa con BM25: título + dominio de la url + texto plano.
pub fn index_text(title: &str, url: &str, html: &str) -> String {
    let host = url.split("://").nth(1).unwrap_or("").split('/').next().unwrap_or("");
    format!("{title} {host} {}", htmlx::to_plain(html))
}

/// Carpetas que el motor ignora: construcciones a medias o restos de `retier`.
pub fn is_transient_dir(p: &Path) -> bool {
    let n = p.file_name().map(|x| x.to_string_lossy().to_string()).unwrap_or_default();
    n.ends_with(".building") || n.ends_with(".retier") || n.ends_with(".old") || n.contains(".old.")
}

/// Construye un segmento completo: metadatos, texto (según nivel) e índice BM25.
/// Se construye en `<out>.building` y solo al final se sustituye `<out>` (un corte a mitad no deja
/// un segmento mezclado que el motor pudiera servir).
pub fn build(items: &[Item], deleted: usize, final_dir: &Path, name: &str, tier: Tier) -> Result<Manifest> {
    if items.is_empty() {
        bail!("segmento vacío");
    }
    let base = final_dir.file_name().context("la ruta del segmento debe terminar en un nombre de carpeta")?.to_string_lossy().to_string();
    // nunca borrar una carpeta que no sea un segmento (p.ej. --out out o --out data)
    if final_dir.exists() && !final_dir.join("manifest.json").exists() && std::fs::read_dir(final_dir)?.next().is_some() {
        bail!("{} existe y no es un segmento de hnz (no tiene manifest.json): elija otra carpeta --out", final_dir.display());
    }
    let tmp = final_dir.with_file_name(format!("{base}.building"));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp)?;
    let out: &Path = &tmp;
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
            deleted: it.deleted,
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
    let (tier_name, block, level, lm_model, transformed_pct) = match tier {
        Tier::Zstd { block, level } => {
            let st = textstore::write(out, &recs, block, level, 112 * 1024)?;
            eprintln!(
                "  texto zstd {:.1}s; HTML transformado: moderno {:.1}%, antiguo {:.1}%, crudo {:.1}%",
                (t0.elapsed() - t_meta).as_secs_f64(),
                100.0 * st.modes[0] as f64 / items.len() as f64,
                100.0 * st.modes[1] as f64 / items.len() as f64,
                100.0 * st.modes[2] as f64 / items.len() as f64
            );
            ("zstd", block, level, None, 100.0 * (st.modes[0] + st.modes[1]) as f64 / items.len() as f64)
        }
        Tier::Lm { codec, model_file } => {
            let st = crate::lm::store::write(out, codec, &recs, true)?;
            eprintln!(
                "  texto LM {:.1}s: {} registros con LM, {} con zstd de reserva, {:.1} tokens/reg, {:.1} B/reg",
                (t0.elapsed() - t_meta).as_secs_f64(),
                st.lm_records,
                st.zstd_records,
                st.tokens as f64 / items.len() as f64,
                st.payload_bytes as f64 / items.len() as f64
            );
            ("lm", 0, 0, Some(model_file), 100.0)
        }
    };
    // BM25 directamente desde los ítems (no hace falta descomprimir)
    let t1 = Instant::now();
    let mut b = bm25::Builder::new();
    for chunk in items.chunks(1 << 16) {
        let texts: Vec<String> = chunk.par_iter().map(|it| index_text(&it.title, &it.url, &it.text)).collect();
        b.add_batch(&texts);
    }
    // sin poda por df: podar por segmento (E18) hacía desaparecer términos frecuentes de un periodo
    // (p.ej. "covid" en 2020) y la búsqueda por palabras no los encontraba (revisión adversarial)
    let info = b.write_opts(out, true, 1.0)?;
    eprintln!("  BM25: {} términos, {} postings ({:.1}s)", info.n_terms, info.n_postings, t1.elapsed().as_secs_f64());
    let build_id = {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
        nanos ^ crate::cache::seg_key(name) ^ (items.len() as u64).rotate_left(17)
    };
    let m = Manifest {
        version: 3,
        name: name.to_string(),
        n_docs: items.len(),
        deleted_skipped: deleted,
        build_id,
        id_min: items[0].id,
        id_max: items[items.len() - 1].id,
        time_min: items.iter().map(|i| i.time).min().unwrap(),
        time_max: items.iter().map(|i| i.time).max().unwrap(),
        text_tier: tier_name.into(),
        text_block: block,
        text_level: level,
        lm_model,
        transformed_pct,
        has_bm25: true,
        vec_bits: 0,
    };
    write_manifest(out, &m)?;
    // sustituir el segmento final: primero se aparta el viejo con un rename (atómico) y solo
    // después se borra; si algo falla, ni el viejo ni el nuevo quedan a medias
    if final_dir.exists() {
        let old = final_dir.with_file_name(format!("{base}.old.{build_id:016x}"));
        std::fs::rename(final_dir, &old).with_context(|| format!("apartando {}", final_dir.display()))?;
        if let Err(e) = std::fs::rename(&tmp, final_dir) {
            let _ = std::fs::rename(&old, final_dir);
            return Err(anyhow::anyhow!(e).context(format!("renombrando {} -> {}", tmp.display(), final_dir.display())));
        }
        if let Err(e) = std::fs::remove_dir_all(&old) {
            eprintln!("aviso: no se pudo borrar {} ({e}); el motor lo ignora, bórrelo a mano", old.display());
        }
    } else {
        std::fs::rename(&tmp, final_dir).with_context(|| format!("renombrando {} -> {}", tmp.display(), final_dir.display()))?;
    }
    Ok(m)
}

pub enum TextSource {
    Zstd(TextStore),
    Lm(LmStore),
}

pub struct Segment {
    pub dir: PathBuf,
    pub man: Manifest,
    pub meta: MetaStore,
    pub text: TextSource,
}

/// Texto recuperado de un documento; `partial` = solo el principio (modo fragmento del nivel LM).
#[derive(Clone, Debug, Default)]
pub struct Got {
    pub rec: TextRec,
    pub partial: bool,
}

impl Segment {
    /// Everything in RAM (for the commands that walk the whole segment: verify, retier, rebm25…).
    pub fn open(dir: &Path) -> Result<Self> {
        Self::open_with(dir, crate::blob::Residency::ALL)
    }

    pub fn open_with(dir: &Path, res: crate::blob::Residency) -> Result<Self> {
        let man = read_manifest(dir)?;
        let meta = MetaStore::open_with(dir, man.n_docs, res.meta)?;
        let text = if man.text_tier == "lm" {
            TextSource::Lm(LmStore::open_with(dir, res.text)?)
        } else {
            TextSource::Zstd(TextStore::open_with(dir, man.text_block, man.n_docs, res.text)?)
        };
        Ok(Self { dir: dir.to_path_buf(), man, meta, text })
    }

    /// Bytes it takes in RAM (metadata and text).
    pub fn ram_bytes(&self) -> usize {
        self.meta.ram_bytes()
            + match &self.text {
                TextSource::Lm(l) => l.ram_bytes(),
                TextSource::Zstd(t) => t.ram_bytes(),
            }
    }

    pub fn is_lm(&self) -> bool {
        matches!(self.text, TextSource::Lm(_))
    }

    /// Texto de un documento del nivel zstd (para el nivel LM usar `Engine`, que decodifica en lote).
    pub fn zstd_text(&self, doc: usize) -> Result<TextRec> {
        match &self.text {
            TextSource::Zstd(t) => t.get(doc),
            TextSource::Lm(_) => bail!("segmento LM: use la decodificación en lote"),
        }
    }

    /// Ítem completo (decodifica con `codec` si el segmento es del nivel LM).
    pub fn item(&self, doc: usize, codec: Option<&LmCodec>) -> Result<Item> {
        let m = self.meta.row(doc);
        let t = match &self.text {
            TextSource::Zstd(t) => t.get(doc)?,
            TextSource::Lm(l) => {
                let c = codec.context("hace falta el modelo LM para este segmento")?;
                let r = l.raw(doc)?;
                c.decode_records(&[&r[..]], true, None, false, l.rec_check())?.remove(0).0
            }
        };
        Ok(join(&m, t, &self.meta.authors))
    }
}

pub fn join(m: &MetaRow, t: TextRec, authors: &crate::meta::Authors) -> Item {
    Item {
        id: m.id,
        typ: m.typ,
        dead: m.dead,
        deleted: m.deleted,
        by: authors.get(m.author as usize).to_string(),
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
