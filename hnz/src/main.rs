use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use hnz::bm25;
use hnz::embed::Embedder;
use hnz::engine::{Engine, Mode, Query};
use hnz::segment::{self, Segment};
use hnz::{dates, htmlx, vecs};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "hnz", about = "Hacker News comprimido con búsqueda híbrida")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Clone, Copy, ValueEnum)]
enum ModeArg {
    Hybrid,
    Vector,
    Keyword,
}

#[derive(clap::Args, Clone)]
struct ModelArgs {
    /// carpeta con gemma2_text_b1_s64.{xml,bin} y gemma2_tokenizer.json
    #[arg(long, env = "HNZ_MODELS", default_value = "models")]
    models: PathBuf,
    /// carpeta con openvino_c.dll
    #[arg(long, env = "HNZ_OV_LIBS", default_value = ".venv/Lib/site-packages/openvino/libs")]
    ov_libs: PathBuf,
}

#[derive(clap::Args, Clone)]
struct SearchArgs {
    /// nº de resultados
    #[arg(short, long, default_value_t = 10)]
    k: usize,
    /// desde (2015 | 2015-06 | 2015-06-01), inclusive
    #[arg(long)]
    from: Option<String>,
    /// hasta (2018 | 2018-12 | 2018-12-31), inclusive
    #[arg(long)]
    to: Option<String>,
    #[arg(long, value_enum, default_value = "hybrid")]
    mode: ModeArg,
    /// tipos: story,comment,job,poll
    #[arg(long, value_delimiter = ',')]
    r#type: Vec<String>,
    /// solo de este autor
    #[arg(long)]
    by: Option<String>,
    /// incluir ítems "dead"
    #[arg(long)]
    dead: bool,
    #[arg(long, default_value_t = 0.15)]
    w_bm25: f32,
    #[arg(long, default_value_t = 10.0)]
    c_bm25: f32,
}

#[derive(Subcommand)]
enum Cmd {
    /// Construye un segmento (meta + texto) a partir de Parquet mensuales
    Build {
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 16)]
        block: usize,
        #[arg(long, default_value_t = 19)]
        level: i32,
        parquet: Vec<PathBuf>,
    },
    /// Construye los índices de un segmento: BM25 y (si se dan embeddings) vectores binarios
    Index {
        #[arg(long)]
        seg: PathBuf,
        /// embeddings float16 (N x 768) de embed_month.py
        #[arg(long)]
        emb: Option<PathBuf>,
        /// ids (uint32) correspondientes a los embeddings
        #[arg(long)]
        ids: Option<PathBuf>,
        /// bits por vector (MRL: 768, 512, 256)
        #[arg(long, default_value_t = 768)]
        bits: usize,
    },
    /// Busca en todos los segmentos de una carpeta
    Search {
        #[arg(long, default_value = "out")]
        root: PathBuf,
        #[command(flatten)]
        s: SearchArgs,
        #[command(flatten)]
        m: ModelArgs,
        /// salida JSON
        #[arg(long)]
        json: bool,
        query: Vec<String>,
    },
    /// Servidor HTTP: GET /search?q=..&k=10&from=2015&to=2018&mode=hybrid  (y una página en /)
    Serve {
        #[arg(long, default_value = "out")]
        root: PathBuf,
        #[arg(long, default_value_t = 8088)]
        port: u16,
        #[command(flatten)]
        m: ModelArgs,
    },
    /// Evalúa la calidad (P@k, MRR) con consultas JSONL {"q":..., "story": id} (relevantes = su hilo)
    Eval {
        #[arg(long, default_value = "out")]
        root: PathBuf,
        #[arg(long)]
        file: PathBuf,
        #[arg(short, long, default_value_t = 10)]
        k: usize,
        #[command(flatten)]
        m: ModelArgs,
    },
    /// Muestra un ítem por id de HN
    Get {
        #[arg(long)]
        seg: PathBuf,
        id: u32,
    },
    /// Comprueba que TODOS los registros se recuperan idénticos al Parquet original
    Verify {
        #[arg(long)]
        seg: PathBuf,
        parquet: Vec<PathBuf>,
    },
    /// Tamaño por componente y bytes/ítem
    Stats {
        #[arg(long)]
        seg: PathBuf,
    },
    /// Mide la latencia de lectura aleatoria de registros
    Bench {
        #[arg(long)]
        seg: PathBuf,
        #[arg(long, default_value_t = 2000)]
        n: usize,
    },
}

fn make_query(text: &str, s: &SearchArgs) -> Result<Query> {
    let types = s
        .r#type
        .iter()
        .map(|t| match t.as_str() {
            "story" => Ok(1u8),
            "comment" => Ok(2),
            "poll" => Ok(3),
            "pollopt" => Ok(4),
            "job" => Ok(5),
            o => bail!("tipo desconocido: {o}"),
        })
        .collect::<Result<Vec<u8>>>()?;
    Ok(Query {
        text: text.to_string(),
        k: s.k,
        from: match &s.from {
            Some(f) => Some(dates::parse(f, false).context("fecha --from inválida")?),
            None => None,
        },
        to: match &s.to {
            Some(t) => Some(dates::parse(t, true).context("fecha --to inválida")?),
            None => None,
        },
        mode: match s.mode {
            ModeArg::Hybrid => Mode::Hybrid,
            ModeArg::Vector => Mode::Vector,
            ModeArg::Keyword => Mode::Keyword,
        },
        types,
        by: s.by.clone(),
        include_dead: s.dead,
        w_bm25: s.w_bm25,
        c_bm25: s.c_bm25,
        vec_cand: None,
    })
}

fn cmd_eval(root: &Path, file: &Path, k: usize, m: &ModelArgs) -> Result<()> {
    let eng = Engine::open(root)?;
    let mut emb = Embedder::new(&m.models, &m.ov_libs)?;
    let qs: Vec<(String, u32)> = std::fs::read_to_string(file)?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).unwrap();
            (v["q"].as_str().unwrap().to_string(), v["story"].as_u64().unwrap() as u32)
        })
        .collect();
    let t0 = Instant::now();
    let vecs: Vec<Vec<f32>> = qs.iter().map(|q| emb.embed_query(&q.0)).collect::<Result<_>>()?;
    eprintln!("{} consultas embebidas en {:.1} ms/consulta", qs.len(), t0.elapsed().as_secs_f64() * 1e3 / qs.len() as f64);
    let mut configs: Vec<(String, Query)> = Vec::new();
    let base = Query { k: k + 1, ..Default::default() };
    configs.push(("vector (2 etapas)".into(), Query { mode: Mode::Vector, ..base.clone() }));
    configs.push(("vector exacto (asimétrico total)".into(), Query { mode: Mode::Vector, vec_cand: Some(usize::MAX / 2), ..base.clone() }));
    configs.push(("palabras (BM25)".into(), Query { mode: Mode::Keyword, ..base.clone() }));
    for &w in &[0.05f32, 0.1, 0.15, 0.25, 0.4] {
        for &c in &[5.0f32, 10.0, 20.0] {
            configs.push((format!("híbrida w={w} c={c}"), Query { mode: Mode::Hybrid, w_bm25: w, c_bm25: c, ..base.clone() }));
        }
    }
    println!("{:<36} {:>7} {:>7} {:>9}", "configuración", "P@k", "MRR", "ms/cons.");
    for (name, tmpl) in &configs {
        let t = Instant::now();
        let (mut p, mut mrr) = (0f64, 0f64);
        for (i, (text, story)) in qs.iter().enumerate() {
            let q = Query { text: text.clone(), ..tmpl.clone() };
            let hits = eng.search(&q, Some(&vecs[i]));
            let mut rel = 0;
            let mut first = 0usize;
            let mut n = 0;
            for h in &hits {
                let row = eng.segs[h.seg].s.meta.row(h.doc as usize);
                if row.id == *story {
                    continue; // la propia story no cuenta
                }
                n += 1;
                if n > k {
                    break;
                }
                if eng.root_id(h.seg, h.doc) == *story {
                    rel += 1;
                    if first == 0 {
                        first = n;
                    }
                }
            }
            p += rel as f64 / k as f64;
            if first > 0 {
                mrr += 1.0 / first as f64;
            }
        }
        let nq = qs.len() as f64;
        println!("{:<36} {:>7.3} {:>7.3} {:>9.1}", name, p / nq, mrr / nq, t.elapsed().as_secs_f64() * 1e3 / nq);
    }
    Ok(())
}

/// Texto que se indexa con BM25: título + dominio de la url + texto plano.
fn index_text(title: &str, url: &str, html: &str) -> String {
    let host = url.split("://").nth(1).unwrap_or("").split('/').next().unwrap_or("");
    format!("{title} {host} {}", htmlx::to_plain(html))
}

fn cmd_index(seg: &Path, emb: Option<PathBuf>, ids: Option<PathBuf>, bits: usize) -> Result<()> {
    let t0 = Instant::now();
    let s = Segment::open(seg)?;
    let mut b = bm25::Builder::new();
    let nb = s.text.n_blocks();
    let chunk = 4096; // bloques por lote
    for c0 in (0..nb).step_by(chunk) {
        let texts: Vec<String> = (c0..(c0 + chunk).min(nb))
            .into_par_iter()
            .flat_map_iter(|bi| s.text.get_block(bi).expect("bloque").into_iter().map(|r| index_text(&r.title, &r.url, &r.text)))
            .collect();
        b.add_batch(&texts);
    }
    let info = b.write(seg)?;
    eprintln!(
        "BM25: {} docs, {} términos, {} postings, avgdl {:.1} ({:.1}s)",
        info.n_docs,
        info.n_terms,
        info.n_postings,
        info.avgdl,
        t0.elapsed().as_secs_f64()
    );
    let mut man = s.man.clone();
    man.has_bm25 = true;
    if let (Some(e), Some(i)) = (emb, ids) {
        let t1 = Instant::now();
        let doc_ids: Vec<u32> = (0..s.meta.n_blocks()).flat_map(|bi| s.meta.block(bi).into_iter().map(|r| r.id)).collect();
        let matched = vecs::build(seg, &doc_ids, &e, &i, bits)?;
        man.vec_bits = bits;
        eprintln!("vectores: {matched}/{} docs con embedding, {bits} bits ({:.1}s)", doc_ids.len(), t1.elapsed().as_secs_f64());
    }
    segment::write_manifest(seg, &man)?;
    Ok(())
}

fn print_results(res: &[hnz::engine::ResultItem]) {
    for r in res {
        let head = if !r.title.is_empty() {
            r.title.clone()
        } else {
            format!("re: {}", r.story_title.clone().unwrap_or_else(|| "(hilo)".into()))
        };
        println!("\n#{} [{} | {} | {} | {} pts] score {:.3} (v {:.3}, bm25 {:.2})", r.rank, r.kind, r.date, r.by, r.points, r.score, r.vscore, r.bm25);
        println!("   {head}");
        if !r.url.is_empty() {
            println!("   {}", r.url);
        }
        let t: String = r.text.chars().take(300).collect::<String>().replace('\n', " ");
        if !t.is_empty() {
            println!("   {t}{}", if r.text.chars().count() > 300 { "…" } else { "" });
        }
        println!("   {}", r.link);
    }
}

const PAGE: &str = include_str!("page.html");

fn query_param(url: &str, key: &str) -> Option<String> {
    let q = url.split_once('?')?.1;
    for kv in q.split('&') {
        let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
        if k == key {
            let v = v.replace('+', " ");
            let mut out = Vec::new();
            let b = v.as_bytes();
            let mut i = 0;
            let hex = |c: u8| (c as char).to_digit(16).map(|d| d as u8);
            while i < b.len() {
                if b[i] == b'%' && i + 2 < b.len() {
                    if let (Some(h), Some(l)) = (hex(b[i + 1]), hex(b[i + 2])) {
                        out.push(h * 16 + l);
                        i += 3;
                        continue;
                    }
                }
                out.push(b[i]);
                i += 1;
            }
            return Some(String::from_utf8_lossy(&out).to_string());
        }
    }
    None
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Build { out, block, level, parquet } => {
            let t0 = Instant::now();
            let (items, deleted) = segment::load_items(&parquet)?;
            eprintln!("leídos {} ítems ({} borrados omitidos) en {:.1}s", items.len(), deleted, t0.elapsed().as_secs_f64());
            let name = out.file_name().unwrap().to_string_lossy().to_string();
            let m = segment::build_store(&items, deleted, &out, &name, block, level)?;
            eprintln!("segmento {} listo: {} docs en {:.1}s", m.name, m.n_docs, t0.elapsed().as_secs_f64());
        }
        Cmd::Index { seg, emb, ids, bits } => cmd_index(&seg, emb, ids, bits)?,
        Cmd::Search { root, s, m, json, query } => {
            let text = query.join(" ");
            let q = make_query(&text, &s)?;
            let t0 = Instant::now();
            let eng = Engine::open(&root)?;
            let t_open = t0.elapsed();
            let mut emb = if q.mode != Mode::Keyword { Some(Embedder::new(&m.models, &m.ov_libs)?) } else { None };
            let t_load = t0.elapsed();
            let t1 = Instant::now();
            let qv = match emb.as_mut() {
                Some(e) => Some(e.embed_query(&text)?),
                None => None,
            };
            let t_emb = t1.elapsed();
            let t2 = Instant::now();
            let hits = eng.search(&q, qv.as_deref());
            let t_search = t2.elapsed();
            let t3 = Instant::now();
            let res = eng.fetch(&hits)?;
            let t_fetch = t3.elapsed();
            if json {
                println!("{}", serde_json::to_string_pretty(&res)?);
            } else {
                print_results(&res);
            }
            eprintln!(
                "\n[{} docs en {} segmentos | abrir {:.0} ms, cargar modelo {:.0} ms | embedding {:.1} ms, búsqueda {:.1} ms, recuperar+descomprimir {:.1} ms]",
                eng.total_docs,
                eng.segs.len(),
                t_open.as_secs_f64() * 1e3,
                (t_load - t_open).as_secs_f64() * 1e3,
                t_emb.as_secs_f64() * 1e3,
                t_search.as_secs_f64() * 1e3,
                t_fetch.as_secs_f64() * 1e3
            );
        }
        Cmd::Serve { root, port, m } => {
            let eng = Engine::open(&root)?;
            let emb = std::sync::Mutex::new(Embedder::new(&m.models, &m.ov_libs)?);
            let server = tiny_http::Server::http(("127.0.0.1", port)).map_err(|e| anyhow::anyhow!("{e}"))?;
            eprintln!("hnz: {} docs en {} segmentos — http://127.0.0.1:{port}/", eng.total_docs, eng.segs.len());
            for req in server.incoming_requests() {
                let url = req.url().to_string();
                let resp = if url == "/" || url.starts_with("/?") {
                    tiny_http::Response::from_string(PAGE)
                        .with_header("Content-Type: text/html; charset=utf-8".parse::<tiny_http::Header>().unwrap())
                } else if url.starts_with("/search") {
                    let body = (|| -> Result<String> {
                        let text = query_param(&url, "q").unwrap_or_default();
                        let mut s = SearchArgs {
                            k: query_param(&url, "k").and_then(|x| x.parse().ok()).unwrap_or(10),
                            from: query_param(&url, "from").filter(|x| !x.is_empty()),
                            to: query_param(&url, "to").filter(|x| !x.is_empty()),
                            mode: ModeArg::Hybrid,
                            r#type: query_param(&url, "type").filter(|x| !x.is_empty()).map(|x| x.split(',').map(String::from).collect()).unwrap_or_default(),
                            by: query_param(&url, "by").filter(|x| !x.is_empty()),
                            dead: false,
                            w_bm25: 0.15,
                            c_bm25: 10.0,
                        };
                        s.mode = match query_param(&url, "mode").as_deref() {
                            Some("vector") => ModeArg::Vector,
                            Some("keyword") => ModeArg::Keyword,
                            _ => ModeArg::Hybrid,
                        };
                        let q = make_query(&text, &s)?;
                        let t0 = Instant::now();
                        let qv = if q.mode != Mode::Keyword { Some(emb.lock().unwrap().embed_query(&text)?) } else { None };
                        let t_emb = t0.elapsed();
                        let hits = eng.search(&q, qv.as_deref());
                        let t_s = t0.elapsed();
                        let res = eng.fetch(&hits)?;
                        let t_all = t0.elapsed();
                        Ok(serde_json::to_string(&serde_json::json!({
                            "query": text, "k": q.k, "results": res,
                            "ms": {"embedding": t_emb.as_secs_f64()*1e3, "search": (t_s - t_emb).as_secs_f64()*1e3,
                                   "fetch": (t_all - t_s).as_secs_f64()*1e3, "total": t_all.as_secs_f64()*1e3}
                        }))?)
                    })()
                    .unwrap_or_else(|e| serde_json::json!({"error": e.to_string()}).to_string());
                    tiny_http::Response::from_string(body)
                        .with_header("Content-Type: application/json; charset=utf-8".parse::<tiny_http::Header>().unwrap())
                } else if let Some(id) = url.strip_prefix("/item/").and_then(|x| x.parse::<u32>().ok()) {
                    let body = match eng.get(id) {
                        Ok(Some(it)) => serde_json::to_string(&it).unwrap(),
                        _ => serde_json::json!({"error": "no encontrado"}).to_string(),
                    };
                    tiny_http::Response::from_string(body)
                        .with_header("Content-Type: application/json; charset=utf-8".parse::<tiny_http::Header>().unwrap())
                } else {
                    tiny_http::Response::from_string("not found").with_status_code(404)
                };
                let _ = req.respond(resp);
            }
        }
        Cmd::Eval { root, file, k, m } => cmd_eval(&root, &file, k, &m)?,
        Cmd::Get { seg, id } => {
            let s = Segment::open(&seg)?;
            match s.meta.doc_of_id(id) {
                Some(d) => println!("{}", serde_json::to_string_pretty(&s.item(d)?)?),
                None => bail!("id {id} no está en el segmento"),
            }
        }
        Cmd::Verify { seg, parquet } => {
            let t0 = Instant::now();
            let s = Segment::open(&seg)?;
            let (items, _) = segment::load_items(&parquet)?;
            if items.len() != s.man.n_docs {
                bail!("nº de docs distinto: parquet {} vs segmento {}", items.len(), s.man.n_docs);
            }
            let bad: usize = (0..s.text.n_blocks())
                .into_par_iter()
                .map(|b| {
                    let texts = s.text.get_block(b).expect("bloque de texto");
                    let mut bad = 0;
                    for (k, t) in texts.into_iter().enumerate() {
                        let doc = b * s.text.block + k;
                        let m = s.meta.row(doc);
                        let got = segment::join(&m, t, &s.meta.authors);
                        if got != items[doc] {
                            if bad < 3 {
                                eprintln!("DIFERENCIA en id {}:\n  orig {:?}\n  leído {:?}", items[doc].id, items[doc], got);
                            }
                            bad += 1;
                        }
                    }
                    bad
                })
                .sum();
            if bad > 0 {
                bail!("{bad} registros difieren");
            }
            println!("OK: {} registros idénticos al original ({:.1}s)", items.len(), t0.elapsed().as_secs_f64());
        }
        Cmd::Stats { seg } => {
            let m = segment::read_manifest(&seg)?;
            let sz = segment::sizes(&seg)?;
            let total: u64 = sz.iter().map(|x| x.1).sum();
            println!("segmento {}: {} docs, tier {} (bloque {}, zstd-{})", m.name, m.n_docs, m.text_tier, m.text_block, m.text_level);
            for (f, b) in &sz {
                println!("  {:<16} {:>12} B  {:>7.2} B/doc", f, b, *b as f64 / m.n_docs as f64);
            }
            println!(
                "  TOTAL            {:>12} B  {:>7.2} B/doc  -> 49,3M: {:.2} GB",
                total,
                total as f64 / m.n_docs as f64,
                total as f64 / m.n_docs as f64 * 49.3e6 / 1e9
            );
        }
        Cmd::Bench { seg, n } => {
            let s = Segment::open(&seg)?;
            let mut x: u64 = 0x9E3779B97F4A7C15;
            let docs: Vec<usize> = (0..n)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    (x % s.man.n_docs as u64) as usize
                })
                .collect();
            let t0 = Instant::now();
            let mut bytes = 0;
            for &d in &docs {
                bytes += s.item(d)?.text.len();
            }
            let dt = t0.elapsed().as_secs_f64();
            println!("{} lecturas aleatorias (1 hilo, sin caché): {:.1} µs/registro ({} B de texto)", n, dt / n as f64 * 1e6, bytes);
        }
    }
    Ok(())
}
