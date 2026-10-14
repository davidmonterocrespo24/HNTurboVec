//! Fachada estable para usar hnz como biblioteca (Python vía PyO3, C, HTTP): abrir un archivo,
//! buscar y recuperar ítems. Segura entre hilos: varias consultas a la vez sobre un mismo `Archive`.
//!
//! - El embedder de consultas (OpenVINO) vive en un hilo propio: sus objetos no se pueden compartir
//!   entre hilos y así no hace falta `unsafe`; las consultas le piden el vector por un canal.
//! - El LM descomprime de una consulta en una: su pool de hilos es global (fase P1; el planificador
//!   con prioridades llega con el perfil compacto).
use crate::dates;
use crate::embed::Embedder;
use crate::engine::{Engine, Mode, Query, ResultItem, MAX_K};
use crate::htmlx;
use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::Mutex;
use std::time::Instant;

// el motor se comparte entre hilos: si algún día deja de ser Send + Sync, que falle al compilar
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Engine>();
};

/// Opciones de apertura.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct OpenOptions {
    /// carpeta de modelos (por defecto `<archivo>/models` o `<archivo>/../models`)
    pub models: Option<PathBuf>,
    /// carpeta con `openvino_c.dll` (por defecto la variable `HNZ_OV_LIBS`)
    pub ov_libs: Option<PathBuf>,
    /// cargar el embedder al abrir (si no, en la primera búsqueda semántica)
    pub preload: bool,
    /// presupuesto de RAM en bytes (fase P1: se guarda e informa; se aplica en la fase P2)
    pub ram: Option<u64>,
}

/// Una búsqueda. Fechas como en la CLI: "2015", "2015-06", "2015-06-01" (ambos extremos inclusive).
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct SearchRequest {
    pub query: String,
    pub k: usize,
    pub since: Option<String>,
    pub until: Option<String>,
    /// story, comment, job, poll, pollopt (vacío = todos)
    pub types: Vec<String>,
    pub author: Option<String>,
    /// hybrid | vector | keyword
    pub mode: String,
    /// texto completo también en el nivel LM (si no, fragmento de 64 tokens)
    pub full: bool,
    pub include_dead: bool,
    pub w_bm25: f32,
    pub c_bm25: f32,
}

impl Default for SearchRequest {
    fn default() -> Self {
        Self {
            query: String::new(),
            k: 10,
            since: None,
            until: None,
            types: Vec::new(),
            author: None,
            mode: "hybrid".into(),
            full: false,
            include_dead: false,
            w_bm25: 0.15,
            c_bm25: 20.0,
        }
    }
}

#[derive(Serialize, Default, Clone, Copy)]
pub struct Timings {
    pub embedding: f64,
    pub search: f64,
    pub fetch: f64,
    pub total: f64,
}

#[derive(Serialize)]
pub struct SearchResponse {
    pub query: String,
    pub k: usize,
    pub results: Vec<ResultItem>,
    pub ms: Timings,
}

/// Error en los argumentos de una consulta (en Python: ValueError).
#[derive(Debug)]
pub struct QueryError(pub String);
impl std::fmt::Display for QueryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for QueryError {}

/// Valida una consulta y la traduce a la del motor.
pub fn make_query(r: &SearchRequest) -> Result<Query> {
    let bad = |m: String| anyhow::Error::new(QueryError(m));
    if r.k == 0 || r.k > MAX_K {
        return Err(bad(format!("k debe estar entre 1 y {MAX_K} (es {})", r.k)));
    }
    if !(r.c_bm25.is_finite() && r.c_bm25 > 0.0 && r.w_bm25.is_finite() && r.w_bm25 >= 0.0) {
        return Err(bad(format!("c_bm25 debe ser > 0 y w_bm25 >= 0 (son {} y {})", r.c_bm25, r.w_bm25)));
    }
    let mut types = Vec::new();
    for t in &r.types {
        types.push(match t.as_str() {
            "story" => 1u8,
            "comment" => 2,
            "poll" => 3,
            "pollopt" => 4,
            "job" => 5,
            o => return Err(bad(format!("tipo desconocido: {o} (story, comment, job, poll, pollopt)"))),
        });
    }
    let mode = match r.mode.as_str() {
        "hybrid" => Mode::Hybrid,
        "vector" => Mode::Vector,
        "keyword" => Mode::Keyword,
        o => return Err(bad(format!("modo desconocido: {o} (hybrid, vector, keyword)"))),
    };
    let date = |s: &Option<String>, end: bool, name: &str| -> Result<Option<i64>> {
        match s {
            Some(x) => dates::parse(x, end).map(Some).ok_or_else(|| bad(format!("fecha {name} inválida: {x:?} (2015, 2015-06 o 2015-06-01)"))),
            None => Ok(None),
        }
    };
    Ok(Query {
        text: r.query.clone(),
        k: r.k,
        from: date(&r.since, false, "since")?,
        to: date(&r.until, true, "until")?,
        mode,
        types,
        by: r.author.clone(),
        include_dead: r.include_dead,
        w_bm25: r.w_bm25,
        c_bm25: r.c_bm25,
        vec_cand: None,
    })
}

type EmbedReply = mpsc::Sender<Result<Vec<f32>>>;

/// Un archivo hnz abierto.
pub struct Archive {
    eng: Engine,
    root: PathBuf,
    models: PathBuf,
    ov_libs: Option<PathBuf>,
    ram: Option<u64>,
    /// canal al hilo del embedder (se crea al primer uso)
    emb: Mutex<Option<mpsc::Sender<(String, EmbedReply)>>>,
    /// el LM descomprime de una consulta en una (pool global; fase P1)
    lm: Mutex<()>,
}

fn default_models(root: &Path) -> Result<PathBuf> {
    for c in [root.join("models"), root.parent().map(|p| p.join("models")).unwrap_or_default()] {
        if c.is_dir() {
            return Ok(c);
        }
    }
    bail!("no se encuentran los modelos junto a {} (indique models=...)", root.display())
}

impl Archive {
    pub fn open(root: &Path, opts: OpenOptions) -> Result<Self> {
        let models = match opts.models {
            Some(m) => m,
            None => default_models(root)?,
        };
        let eng = Engine::open(root, &models).with_context(|| format!("abriendo {}", root.display()))?;
        let ov_libs = opts.ov_libs.or_else(|| std::env::var_os("HNZ_OV_LIBS").map(PathBuf::from));
        let a = Archive { eng, root: root.to_path_buf(), models, ov_libs, ram: opts.ram, emb: Mutex::new(None), lm: Mutex::new(()) };
        if opts.preload {
            a.embed("warm up")?;
        }
        Ok(a)
    }

    fn embedder(&self) -> Result<mpsc::Sender<(String, EmbedReply)>> {
        let mut g = self.emb.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(tx) = g.as_ref() {
            return Ok(tx.clone());
        }
        let libs = self.ov_libs.clone().context("falta la carpeta de OpenVINO (ov_libs o la variable HNZ_OV_LIBS)")?;
        let models = self.models.clone();
        let (tx, rx) = mpsc::channel::<(String, EmbedReply)>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<()>>();
        std::thread::Builder::new().name("hnz-embedder".into()).spawn(move || {
            let mut emb = match Embedder::new(&models, &libs) {
                Ok(e) => {
                    let _ = ready_tx.send(Ok(()));
                    e
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                    return;
                }
            };
            // termina solo cuando se cierra el archivo (se sueltan todos los emisores)
            for (text, reply) in rx {
                let _ = reply.send(emb.embed_query(&text));
            }
        })?;
        ready_rx.recv().map_err(|_| anyhow!("el hilo del embedder terminó al arrancar"))??;
        *g = Some(tx.clone());
        Ok(tx)
    }

    /// Vector de una consulta con EmbeddingGemma 2 (el mismo que usa la búsqueda semántica).
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let (rtx, rrx) = mpsc::channel();
        self.embedder()?.send((text.to_string(), rtx)).map_err(|_| anyhow!("el embedder no está disponible"))?;
        rrx.recv().map_err(|_| anyhow!("el embedder no respondió"))?
    }

    pub fn search(&self, req: &SearchRequest) -> Result<SearchResponse> {
        let q = make_query(req)?;
        let t0 = Instant::now();
        let qv = if q.mode != Mode::Keyword && !q.text.trim().is_empty() { Some(self.embed(&q.text)?) } else { None };
        let t_emb = t0.elapsed();
        let hits = self.eng.search(&q, qv.as_deref());
        let t_s = t0.elapsed();
        let results = {
            let _g = self.lm.lock().unwrap_or_else(|e| e.into_inner());
            self.eng.fetch(&hits, req.full)?
        };
        let t_all = t0.elapsed();
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
        Ok(SearchResponse {
            query: req.query.clone(),
            k: q.k,
            results,
            ms: Timings { embedding: ms(t_emb), search: ms(t_s - t_emb), fetch: ms(t_all - t_s), total: ms(t_all) },
        })
    }

    /// Un ítem completo por id de HN (texto literal: `html` es el HTML original de HN).
    pub fn get(&self, id: u32) -> Result<Option<serde_json::Value>> {
        let it = {
            let _g = self.lm.lock().unwrap_or_else(|e| e.into_inner());
            self.eng.get(id)?
        };
        Ok(match it {
            Some(it) => {
                let mut v = serde_json::to_value(&it)?;
                v["text_plain"] = serde_json::Value::String(htmlx::to_plain(&it.text));
                Some(v)
            }
            None => None,
        })
    }

    pub fn stats(&self) -> serde_json::Value {
        let segs: Vec<serde_json::Value> = self
            .eng
            .segs
            .iter()
            .map(|sg| {
                let m = &sg.s.man;
                serde_json::json!({
                    "name": m.name, "docs": m.n_docs, "tier": m.text_tier, "vec_bits": m.vec_bits,
                    "date_min": dates::fmt(m.time_min), "date_max": dates::fmt(m.time_max),
                })
            })
            .collect();
        serde_json::json!({
            "path": self.root.display().to_string(),
            "models": self.models.display().to_string(),
            "docs": self.eng.total_docs,
            "segments": segs,
            "ram_budget": self.ram,
            "embedder_loaded": self.emb.lock().map(|g| g.is_some()).unwrap_or(false),
        })
    }
}
