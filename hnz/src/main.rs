use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use hnz::bm25;
use hnz::embed::Embedder;
use hnz::engine::{Engine, Mode, Query};
use hnz::lm::store::LmCodec;
use hnz::segment::{self, Segment, TextSource, Tier};
use hnz::{dates, vecs};
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

#[derive(Clone, Copy, ValueEnum, PartialEq, Eq, Debug)]
enum TierArg {
    /// caliente/templado: zstd en bloques (lectura en µs)
    Zstd,
    /// frío: LM + codificación aritmética (máxima compresión)
    Lm,
}

#[derive(clap::Args, Clone)]
struct ModelArgs {
    /// carpeta de modelos (EmbeddingGemma 2 IR + tokenizer, LM .hnzlm)
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
    #[arg(long, default_value_t = 20.0)]
    c_bm25: f32,
    /// texto completo también para el nivel LM (por defecto: fragmento de 64 tokens)
    #[arg(long)]
    full: bool,
}

#[derive(Subcommand)]
enum Cmd {
    /// Construye un segmento (meta + texto + BM25) a partir de Parquet mensuales
    Build {
        #[arg(long)]
        out: PathBuf,
        #[arg(long, value_enum, default_value = "zstd")]
        tier: TierArg,
        #[arg(long, default_value_t = 16)]
        block: usize,
        #[arg(long, default_value_t = 19)]
        level: i32,
        /// modelo LM para --tier lm
        #[arg(long, default_value = "models/lm-pythia-70m.hnzlm")]
        lm: PathBuf,
        parquet: Vec<PathBuf>,
    },
    /// Cambia el nivel de un segmento (p.ej. zstd -> lm al envejecer) usando sus propios textos.
    /// Reconstruye en una carpeta temporal, verifica y sustituye; conserva los vectores.
    Retier {
        #[arg(long)]
        seg: PathBuf,
        #[arg(long, value_enum)]
        tier: TierArg,
        /// verificar TODOS los documentos del segmento nuevo (por defecto, una muestra de 2.000)
        #[arg(long)]
        verify_all: bool,
        /// borrar el original al terminar (solo con --verify-all; si no, queda como <seg>.old)
        #[arg(long)]
        delete_old: bool,
        /// bits de los vectores del segmento nuevo (por defecto 512 al enfriar a lm, si no los mismos)
        #[arg(long)]
        bits: Option<usize>,
        #[arg(long, default_value_t = 16)]
        block: usize,
        #[arg(long, default_value_t = 19)]
        level: i32,
        #[arg(long, default_value = "models/lm-pythia-70m.hnzlm")]
        lm: PathBuf,
    },
    /// Reconstruye el índice BM25 de un segmento con otras opciones (para medir tamaño/calidad)
    Rebm25 {
        #[arg(long)]
        seg: PathBuf,
        /// postings booleanos (sin frecuencia del término)
        #[arg(long)]
        no_tf: bool,
        /// no indexar términos presentes en más de esta fracción de documentos
        #[arg(long, default_value_t = 1.0)]
        max_df: f32,
        #[arg(long, env = "HNZ_MODELS", default_value = "models")]
        models: PathBuf,
    },
    /// Añade el índice vectorial (binario) a un segmento a partir de embeddings
    Index {
        #[arg(long)]
        seg: PathBuf,
        /// embeddings float16 (N x 768) de embed_month.py
        #[arg(long)]
        emb: PathBuf,
        /// ids (uint32) correspondientes a los embeddings
        #[arg(long)]
        ids: PathBuf,
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
    /// Servidor HTTP: GET /search?q=..&k=10&from=2015&to=2018&mode=hybrid, /item/<id>, página en /
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
    /// Prueba del nivel frío sobre un segmento zstd: comprime y descomprime (en lote) una muestra
    /// Exporta a JSONL el texto de cada ítem tal como lo ve el LM (para entrenar/evaluar un LM de HN)
    Lmdump {
        #[arg(long)]
        out: PathBuf,
        /// tomar 1 de cada N ítems (muestreo uniforme por id)
        #[arg(long, default_value_t = 1)]
        every: usize,
        parquet: Vec<PathBuf>,
    },
    Lmtest {
        #[arg(long)]
        seg: PathBuf,
        #[arg(long, default_value = "models/lm-pythia-70m.hnzlm")]
        lm: PathBuf,
        #[arg(long, default_value_t = 200)]
        n: usize,
        #[arg(long, default_value_t = 10)]
        batch: usize,
    },
    /// Muestra un ítem por id de HN
    Get {
        #[arg(long, default_value = "out")]
        root: PathBuf,
        #[command(flatten)]
        m: ModelArgs,
        id: u32,
    },
    /// Comprueba que TODOS los registros se recuperan idénticos al Parquet original
    Verify {
        #[arg(long)]
        seg: PathBuf,
        #[arg(long, env = "HNZ_MODELS", default_value = "models")]
        models: PathBuf,
        parquet: Vec<PathBuf>,
    },
    /// Tamaño por componente y bytes/ítem (de un segmento o de toda la carpeta)
    Stats {
        #[arg(long)]
        seg: Option<PathBuf>,
        #[arg(long, default_value = "out")]
        root: PathBuf,
    },
    /// Latencia de lectura aleatoria (zstd: registros sueltos; LM: lotes de 10)
    Bench {
        #[arg(long)]
        seg: PathBuf,
        #[arg(long, env = "HNZ_MODELS", default_value = "models")]
        models: PathBuf,
        #[arg(long, default_value_t = 200)]
        n: usize,
    },
}

fn make_query(text: &str, s: &SearchArgs) -> Result<Query> {
    if !(s.c_bm25.is_finite() && s.c_bm25 > 0.0 && s.w_bm25.is_finite() && s.w_bm25 >= 0.0) {
        bail!("--c-bm25 debe ser > 0 y --w-bm25 >= 0 (son {} y {})", s.c_bm25, s.w_bm25);
    }
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

fn rand_docs(n: usize, max: usize, seed: u64) -> Vec<usize> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % max as u64) as usize
        })
        .collect()
}

fn cmd_lmtest(seg: &Path, lm: &Path, n: usize, batch: usize) -> Result<()> {
    let s = Segment::open(seg)?;
    let t0 = Instant::now();
    let codec = LmCodec::load(lm)?;
    eprintln!("modelo {} cargado en {:.0} ms", codec.model.id, t0.elapsed().as_secs_f64() * 1e3);
    let docs = rand_docs(n, s.man.n_docs, 0x2545F4914F6CDD1D);
    let recs: Vec<hnz::textstore::TextRec> = docs.iter().map(|&d| s.zstd_text(d)).collect::<Result<_>>()?;
    let raw: usize = recs.iter().map(|r| r.title.len() + r.url.len() + r.text.len() + 2).sum();
    let t1 = Instant::now();
    let enc: Vec<(Vec<u8>, u64)> = recs.par_iter().map(|r| codec.encode_record(&r.title, &r.url, &r.text)).collect();
    let te = t1.elapsed().as_secs_f64();
    {
        let p: Vec<f64> = hnz::lm::model::PROF.iter().map(|x| x.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e9).collect();
        let hilos = rayon::current_num_threads() as f64;
        eprintln!(
            "  [perfil codificación, s de CPU] matmul capas {:.1}, atención {:.1}, cabeza {:.1}, resto (softmax+codificador+tokenizar) {:.1} de {:.1}",
            p[0], p[1], p[2], (te * hilos - p[0] - p[1] - p[2]).max(0.0), te * hilos
        );
    }
    let bytes: usize = enc.iter().map(|e| e.0.len()).sum();
    let toks: u64 = enc.iter().map(|e| e.1).sum();
    let nfb = enc.iter().filter(|e| (e.0[0] >> 3) & 1 == 1).count();
    println!(
        "codificar: {} registros, {:.1} B/reg crudo -> {:.1} B/reg ({:.3} bits/byte), {:.1} tokens/reg, {} con zstd de reserva | {:.0} reg/s ({:.0} tok/s) con {} hilos",
        n,
        raw as f64 / n as f64,
        bytes as f64 / n as f64,
        bytes as f64 * 8.0 / raw as f64,
        toks as f64 / n as f64,
        nfb,
        n as f64 / te,
        toks as f64 / te,
        rayon::current_num_threads()
    );
    let t2 = Instant::now();
    let mut bad = 0;
    let mut times = Vec::new();
    for (bi, chunk) in enc.chunks(batch).enumerate() {
        let tb = Instant::now();
        let refs: Vec<&[u8]> = chunk.iter().map(|e| &e.0[..]).collect();
        let out = codec.decode_records(&refs, true, None, false, true)?;
        times.push(tb.elapsed().as_secs_f64() * 1e3);
        for (j, (o, _)) in out.iter().enumerate() {
            if *o != recs[bi * batch + j] {
                if bad < 3 {
                    eprintln!("DIFERENCIA doc {}:\n  orig {:?}\n  dec  {:?}", docs[bi * batch + j], recs[bi * batch + j], o);
                }
                bad += 1;
            }
        }
    }
    times.sort_by(|a, b| a.total_cmp(b));
    println!(
        "decodificar en lotes de {}: mediana {:.0} ms/lote, p90 {:.0} ms ({:.1} s total) | errores: {}",
        batch,
        times[times.len() / 2],
        times[times.len() * 9 / 10],
        t2.elapsed().as_secs_f64(),
        bad
    );
    if bad > 0 {
        bail!("{bad} registros no coinciden");
    }
    Ok(())
}

/// Bits de los vectores: múltiplo de 64 (la etapa 1 compara palabras de 64 bits; MRL de
/// EmbeddingGemma 2: 768/512/256/128).
fn check_bits(bits: usize) -> Result<()> {
    if bits == 0 || bits % 64 != 0 {
        bail!("--bits debe ser múltiplo de 64 (p.ej. 768, 512, 256), no {bits}");
    }
    Ok(())
}

fn cmd_eval(root: &Path, file: &Path, k: usize, m: &ModelArgs) -> Result<()> {
    let eng = Engine::open(root, &m.models)?;
    let mut emb = Embedder::new(&m.models, &m.ov_libs)?;
    let raw = std::fs::read(file)?;
    // tolerar BOM UTF-8 (PowerShell) y rechazar UTF-16 con un mensaje claro
    if raw.starts_with(&[0xFF, 0xFE]) || raw.starts_with(&[0xFE, 0xFF]) {
        bail!("{} está en UTF-16: genérelo con make_eval.py --out (UTF-8)", file.display());
    }
    let text = String::from_utf8(raw)?;
    let qs: Vec<(String, u32)> = text
        .trim_start_matches('\u{feff}')
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
                if eng.root_id(h.seg, h.doc) == Some(*story) {
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

fn print_results(res: &[hnz::engine::ResultItem]) {
    for r in res {
        let head = if !r.title.is_empty() {
            r.title.clone()
        } else {
            format!("re: {}", r.story_title.clone().unwrap_or_else(|| "(hilo)".into()))
        };
        println!(
            "\n#{} [{} | {} | {} | {} pts | {}] score {:.3} (v {:.3}, bm25 {:.2})",
            r.rank, r.kind, r.date, r.by, r.points, r.tier, r.score, r.vscore, r.bm25
        );
        println!("   {head}");
        if !r.url.is_empty() {
            println!("   {}", r.url);
        }
        let t: String = r.text.chars().take(300).collect::<String>().replace('\n', " ");
        if !t.is_empty() {
            println!("   {t}{}", if r.text.chars().count() > 300 || r.partial { "…" } else { "" });
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
        Cmd::Build { out, tier, block, level, lm, parquet } => {
            let t0 = Instant::now();
            let (items, deleted) = segment::load_items(&parquet)?;
            eprintln!("leídos {} ítems (incluidas {} filas deleted, que se guardan como metadatos) en {:.1}s", items.len(), deleted, t0.elapsed().as_secs_f64());
            let name = out.file_name().unwrap().to_string_lossy().to_string();
            let codec;
            let t = match tier {
                TierArg::Zstd => Tier::Zstd { block, level },
                TierArg::Lm => {
                    codec = LmCodec::load(&lm)?;
                    Tier::Lm { codec: &codec, model_file: lm.file_name().unwrap().to_string_lossy().to_string() }
                }
            };
            let m = segment::build(&items, deleted, &out, &name, t)?;
            eprintln!("segmento {} ({}) listo: {} docs en {:.1}s", m.name, m.text_tier, m.n_docs, t0.elapsed().as_secs_f64());
        }
        Cmd::Retier { seg, tier, block, level, lm, verify_all, delete_old, bits } => {
            if let Some(b) = bits {
                check_bits(b)?;
            }
            let t0 = Instant::now();
            // ruta canónica con nombre de carpeta (p.ej. "--seg ." no sirve: no tiene nombre)
            let seg = std::fs::canonicalize(&seg).with_context(|| format!("{}", seg.display()))?;
            let name = seg.file_name().context("--seg debe ser la carpeta de un segmento")?.to_string_lossy().to_string();
            if let Ok(cwd) = std::env::current_dir().and_then(std::fs::canonicalize) {
                if cwd.starts_with(&seg) {
                    bail!("el directorio actual está dentro del segmento ({}): ejecute retier desde otra carpeta (Windows no puede renombrarlo)", cwd.display());
                }
            }
            let tmp = seg.with_file_name(format!("{name}.retier"));
            let old = seg.with_file_name(format!("{name}.old"));
            if old.exists() {
                bail!("ya existe {} de un retier anterior: revíselo y bórrelo antes de continuar", old.display());
            }
            let s = Segment::open(&seg)?;
            let models = lm.parent().unwrap_or(Path::new(".")).to_path_buf();
            let codec_old = match &s.text {
                TextSource::Lm(_) => Some(LmCodec::load(&models.join(s.man.lm_model.as_ref().context("lm_model")?))?),
                _ => None,
            };
            // reconstruir todos los ítems desde el propio segmento
            let mut items = Vec::with_capacity(s.man.n_docs);
            match &s.text {
                TextSource::Zstd(ts) => {
                    for b in 0..ts.n_blocks() {
                        for (k, t) in ts.get_block(b)?.into_iter().enumerate() {
                            items.push(segment::join(&s.meta.row(b * ts.block + k), t, &s.meta.authors));
                        }
                    }
                }
                TextSource::Lm(ls) => {
                    let c = codec_old.as_ref().unwrap();
                    ls.check_model(c)?;
                    let raws: Vec<&[u8]> = (0..s.man.n_docs).map(|d| ls.raw(d)).collect();
                    let mut texts: Vec<Option<hnz::textstore::TextRec>> = vec![None; s.man.n_docs];
                    c.decode_stream(&raws, ls.rec_check(), 64, |d, t| {
                        texts[d] = Some(t);
                        Ok(())
                    })?;
                    for (d, t) in texts.into_iter().enumerate() {
                        items.push(segment::join(&s.meta.row(d), t.context("registro sin decodificar")?, &s.meta.authors));
                    }
                }
            }
            if items.len() != s.man.n_docs {
                bail!("se leyeron {} ítems de {} esperados: se conserva el segmento", items.len(), s.man.n_docs);
            }
            eprintln!("leídos {} ítems del segmento ({:.1}s)", items.len(), t0.elapsed().as_secs_f64());
            let codec;
            let t = match tier {
                TierArg::Zstd => Tier::Zstd { block, level },
                TierArg::Lm => {
                    codec = LmCodec::load(&lm)?;
                    Tier::Lm { codec: &codec, model_file: lm.file_name().unwrap().to_string_lossy().to_string() }
                }
            };
            // construir en <seg>.retier y verificar; si algo falla, se borra y el original queda intacto
            let result = (|| -> Result<segment::Manifest> {
                let mut m = segment::build(&items, s.man.deleted_skipped, &tmp, &s.man.name, t)?;
                if s.man.vec_bits > 0 {
                    // MRL: los primeros `bits` signos son el vector truncado -> se recortan los códigos
                    let new_bits = bits.unwrap_or(if tier == TierArg::Lm { 512 } else { s.man.vec_bits }).min(s.man.vec_bits);
                    let (ob, nb) = (s.man.vec_bits / 8, new_bits / 8);
                    let codes = std::fs::read(seg.join("vec.bin"))?;
                    let cut: Vec<u8> = if nb == ob { codes } else { codes.chunks_exact(ob).flat_map(|c| c[..nb].iter().copied()).collect() };
                    std::fs::write(tmp.join("vec.bin"), cut)?;
                    std::fs::copy(seg.join("vec.has"), tmp.join("vec.has"))?;
                    m.vec_bits = new_bits;
                    segment::write_manifest(&tmp, &m)?;
                }
                // conservar el registro de meses de pipeline.py, con el nivel nuevo
                if let Ok(txt) = std::fs::read_to_string(seg.join("sources.json")) {
                    if let Ok(mut v) = serde_json::from_str::<serde_json::Value>(&txt) {
                        v["tier"] = serde_json::Value::String(m.text_tier.clone());
                        std::fs::write(tmp.join("sources.json"), v.to_string())?;
                    }
                }
                let ns = Segment::open(&tmp)?;
                let docs: Vec<usize> = if verify_all { (0..items.len()).collect() } else { rand_docs(2000.min(items.len()), items.len(), 77) };
                match &ns.text {
                    TextSource::Zstd(_) => {
                        for &d in &docs {
                            if ns.item(d, None)? != items[d] {
                                bail!("verificación fallida en el doc {d}");
                            }
                        }
                    }
                    TextSource::Lm(ls) => {
                        let c = LmCodec::load(&models.join(ns.man.lm_model.as_ref().unwrap()))?;
                        let raws: Vec<&[u8]> = docs.iter().map(|&d| ls.raw(d)).collect();
                        let mut bad = None;
                        c.decode_stream(&raws, ls.rec_check(), 64, |j, t| {
                            let d = docs[j];
                            if segment::join(&ns.meta.row(d), t, &ns.meta.authors) != items[d] {
                                bad.get_or_insert(d);
                            }
                            Ok(())
                        })?;
                        if let Some(d) = bad {
                            bail!("verificación fallida en el doc {d}");
                        }
                    }
                }
                eprintln!("verificados {} de {} documentos ({})", docs.len(), items.len(), if verify_all { "todos" } else { "muestra" });
                Ok(m)
            })();
            let m = match result {
                Ok(m) => m,
                Err(e) => {
                    let _ = std::fs::remove_dir_all(&tmp);
                    let _ = std::fs::remove_dir_all(seg.with_file_name(format!("{name}.retier.building")));
                    return Err(e.context("retier abortado; el segmento original no se ha modificado"));
                }
            };
            drop(s);
            std::fs::rename(&seg, &old)?;
            if let Err(e) = std::fs::rename(&tmp, &seg) {
                let _ = std::fs::rename(&old, &seg);
                return Err(anyhow::anyhow!(e).context("no se pudo sustituir el segmento; se ha restaurado el original"));
            }
            if delete_old && verify_all {
                std::fs::remove_dir_all(&old)?;
                eprintln!("segmento {} ahora en nivel {} ({:.1}s); original borrado", m.name, m.text_tier, t0.elapsed().as_secs_f64());
            } else {
                eprintln!(
                    "segmento {} ahora en nivel {} ({:.1}s); el original queda en {} (el motor lo ignora; bórrelo cuando quiera{})",
                    m.name,
                    m.text_tier,
                    t0.elapsed().as_secs_f64(),
                    old.display(),
                    if delete_old { ": --delete-old solo actúa con --verify-all" } else { "" }
                );
            }
        }
        Cmd::Rebm25 { seg, no_tf, max_df, models } => {
            let t0 = Instant::now();
            let s = Segment::open(&seg)?;
            let mut texts: Vec<String> = Vec::with_capacity(s.man.n_docs);
            match &s.text {
                TextSource::Zstd(ts) => {
                    for b in 0..ts.n_blocks() {
                        for r in ts.get_block(b)? {
                            texts.push(segment::index_text(&r.title, &r.url, &r.text));
                        }
                    }
                }
                TextSource::Lm(ls) => {
                    let c = LmCodec::load(&models.join(s.man.lm_model.as_ref().context("lm_model")?))?;
                    ls.check_model(&c)?;
                    let raws: Vec<&[u8]> = (0..s.man.n_docs).map(|d| ls.raw(d)).collect();
                    let mut got: Vec<String> = vec![String::new(); s.man.n_docs];
                    c.decode_stream(&raws, ls.rec_check(), 64, |d, r| {
                        got[d] = segment::index_text(&r.title, &r.url, &r.text);
                        Ok(())
                    })?;
                    texts = got;
                }
            }
            let mut b = bm25::Builder::new();
            for ch in texts.chunks(1 << 16) {
                b.add_batch(ch);
            }
            let info = b.write_opts(&seg, !no_tf, max_df)?;
            let sz: u64 = ["bm25.fst", "bm25.post", "bm25.norms"].iter().map(|f| std::fs::metadata(seg.join(f)).map(|m| m.len()).unwrap_or(0)).sum();
            eprintln!(
                "BM25 (tf={}, max_df={}): {} términos, {} postings, {:.1} B/doc ({:.1}s)",
                !no_tf,
                max_df,
                info.n_terms,
                info.n_postings,
                sz as f64 / s.man.n_docs as f64,
                t0.elapsed().as_secs_f64()
            );
        }
        Cmd::Index { seg, emb, ids, bits } => {
            check_bits(bits)?;
            let t1 = Instant::now();
            let s = Segment::open(&seg)?;
            let doc_ids: Vec<u32> = (0..s.meta.n_blocks()).flat_map(|bi| s.meta.block(bi).into_iter().map(|r| r.id)).collect();
            let matched = vecs::build(&seg, &doc_ids, &emb, &ids, bits)?;
            let mut man = s.man.clone();
            man.vec_bits = bits;
            segment::write_manifest(&seg, &man)?;
            eprintln!("vectores: {matched}/{} docs con embedding, {bits} bits ({:.1}s)", doc_ids.len(), t1.elapsed().as_secs_f64());
        }
        Cmd::Search { root, s, m, json, query } => {
            let text = query.join(" ");
            let q = make_query(&text, &s)?;
            let t0 = Instant::now();
            let eng = Engine::open(&root, &m.models)?;
            let t_open = t0.elapsed();
            let mut emb = if q.mode != Mode::Keyword && !text.trim().is_empty() { Some(Embedder::new(&m.models, &m.ov_libs)?) } else { None };
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
            let res = eng.fetch(&hits, s.full)?;
            let t_fetch = t3.elapsed();
            if json {
                println!("{}", serde_json::to_string_pretty(&res)?);
            } else {
                print_results(&res);
            }
            let n_lm = res.iter().filter(|r| r.tier == "lm").count();
            eprintln!(
                "\n[{} docs en {} segmentos | abrir {:.0} ms, cargar modelos {:.0} ms | embedding {:.1} ms, búsqueda {:.1} ms, recuperar+descomprimir {:.1} ms ({} del nivel LM{})]",
                eng.total_docs,
                eng.segs.len(),
                t_open.as_secs_f64() * 1e3,
                (t_load - t_open).as_secs_f64() * 1e3,
                t_emb.as_secs_f64() * 1e3,
                t_search.as_secs_f64() * 1e3,
                t_fetch.as_secs_f64() * 1e3,
                n_lm,
                if s.full { ", completos" } else { ", fragmentos" }
            );
        }
        Cmd::Serve { root, port, m } => {
            let eng = Engine::open(&root, &m.models)?;
            let emb = std::sync::Mutex::new(Embedder::new(&m.models, &m.ov_libs)?);
            let server = tiny_http::Server::http(("127.0.0.1", port)).map_err(|e| anyhow::anyhow!("{e}"))?;
            eprintln!("hnz: {} docs en {} segmentos — http://127.0.0.1:{port}/", eng.total_docs, eng.segs.len());
            let json_hdr = || "Content-Type: application/json; charset=utf-8".parse::<tiny_http::Header>().unwrap();
            for req in server.incoming_requests() {
                let url = req.url().to_string();
                let resp = if url == "/" || url.starts_with("/?") {
                    tiny_http::Response::from_string(PAGE)
                        .with_header("Content-Type: text/html; charset=utf-8".parse::<tiny_http::Header>().unwrap())
                } else if url.starts_with("/search") {
                    let body = (|| -> Result<String> {
                        let text = query_param(&url, "q").unwrap_or_default();
                        let mode = match query_param(&url, "mode").as_deref() {
                            Some("vector") => ModeArg::Vector,
                            Some("keyword") => ModeArg::Keyword,
                            _ => ModeArg::Hybrid,
                        };
                        let s = SearchArgs {
                            k: query_param(&url, "k").and_then(|x| x.parse::<usize>().ok()).unwrap_or(10).clamp(1, 100),
                            from: query_param(&url, "from").filter(|x| !x.is_empty()),
                            to: query_param(&url, "to").filter(|x| !x.is_empty()),
                            mode,
                            r#type: query_param(&url, "type").filter(|x| !x.is_empty()).map(|x| x.split(',').map(String::from).collect()).unwrap_or_default(),
                            by: query_param(&url, "by").filter(|x| !x.is_empty()),
                            dead: false,
                            w_bm25: 0.15,
                            c_bm25: 20.0,
                            full: query_param(&url, "full").is_some_and(|x| x == "1" || x == "true"),
                        };
                        let q = make_query(&text, &s)?;
                        let t0 = Instant::now();
                        let qv = if q.mode != Mode::Keyword && !text.trim().is_empty() { Some(emb.lock().unwrap().embed_query(&text)?) } else { None };
                        let t_emb = t0.elapsed();
                        let hits = eng.search(&q, qv.as_deref());
                        let t_s = t0.elapsed();
                        let res = eng.fetch(&hits, s.full)?;
                        let t_all = t0.elapsed();
                        Ok(serde_json::to_string(&serde_json::json!({
                            "query": text, "k": q.k, "results": res,
                            "ms": {"embedding": t_emb.as_secs_f64()*1e3, "search": (t_s - t_emb).as_secs_f64()*1e3,
                                   "fetch": (t_all - t_s).as_secs_f64()*1e3, "total": t_all.as_secs_f64()*1e3}
                        }))?)
                    })()
                    .unwrap_or_else(|e| serde_json::json!({"error": e.to_string()}).to_string());
                    tiny_http::Response::from_string(body).with_header(json_hdr())
                } else if let Some(id) = url.strip_prefix("/item/").and_then(|x| x.parse::<u32>().ok()) {
                    let t0 = Instant::now();
                    let body = match eng.get(id) {
                        Ok(Some(it)) => {
                            let mut v = serde_json::to_value(&it).unwrap();
                            v["text_plain"] = serde_json::Value::String(hnz::htmlx::to_plain(&it.text));
                            v["ms"] = serde_json::json!(t0.elapsed().as_secs_f64() * 1e3);
                            v.to_string()
                        }
                        _ => serde_json::json!({"error": "no encontrado"}).to_string(),
                    };
                    tiny_http::Response::from_string(body).with_header(json_hdr())
                } else {
                    tiny_http::Response::from_string("not found").with_status_code(404)
                };
                let _ = req.respond(resp);
            }
        }
        Cmd::Eval { root, file, k, m } => cmd_eval(&root, &file, k, &m)?,
        Cmd::Lmtest { seg, lm, n, batch } => cmd_lmtest(&seg, &lm, n, batch)?,
        Cmd::Lmdump { out, every, parquet } => {
            use std::io::Write;
            let mut w = std::io::BufWriter::new(std::fs::File::create(&out).with_context(|| format!("{}", out.display()))?);
            let (mut n, mut skipped) = (0usize, 0usize);
            for p in &parquet {
                let (items, _) = segment::load_items(std::slice::from_ref(p))?;
                for it in items.iter().filter(|it| !it.deleted && it.id as usize % every.max(1) == 0) {
                    match hnz::lm::store::lm_text(&it.title, &it.url, &it.text) {
                        Some(s) if !s.trim().is_empty() => {
                            serde_json::to_writer(&mut w, &serde_json::json!({"id": it.id, "s": s}))?;
                            w.write_all(b"\n")?;
                            n += 1;
                        }
                        _ => skipped += 1,
                    }
                }
            }
            w.flush()?;
            eprintln!("{n} registros escritos en {} ({skipped} omitidos: vacíos o con zstd de reserva)", out.display());
        }
        Cmd::Get { root, m, id } => {
            let eng = Engine::open(&root, &m.models)?;
            match eng.get(id)? {
                Some(it) => println!("{}", serde_json::to_string_pretty(&it)?),
                None => bail!("id {id} no está en el archivo"),
            }
        }
        Cmd::Verify { seg, models, parquet } => {
            let t0 = Instant::now();
            let s = Segment::open(&seg)?;
            let (items, _) = segment::load_items(&parquet)?;
            if items.len() != s.man.n_docs {
                bail!("nº de docs distinto: parquet {} vs segmento {}", items.len(), s.man.n_docs);
            }
            let bad: usize = match &s.text {
                TextSource::Zstd(ts) => {
                    let total: usize = (0..ts.n_blocks()).map(|b| (s.man.n_docs - b * ts.block).min(ts.block)).sum();
                    if ts.n_blocks() * ts.block < s.man.n_docs || total != s.man.n_docs {
                        bail!("text.idx tiene {} bloques: no cubren los {} documentos", ts.n_blocks(), s.man.n_docs);
                    }
                    (0..ts.n_blocks())
                    .into_par_iter()
                    .map(|b| {
                        let texts = ts.get_block(b).expect("bloque de texto");
                        let mut bad = 0;
                        for (k, t) in texts.into_iter().enumerate() {
                            let doc = b * ts.block + k;
                            let got = segment::join(&s.meta.row(doc), t, &s.meta.authors);
                            if got != items[doc] {
                                if bad < 3 {
                                    eprintln!("DIFERENCIA en id {}:\n  orig {:?}\n  leído {:?}", items[doc].id, items[doc], got);
                                }
                                bad += 1;
                            }
                        }
                        bad
                    })
                    .sum()
                }
                TextSource::Lm(ls) => {
                    let codec = LmCodec::load(&models.join(s.man.lm_model.as_ref().context("lm_model")?))?;
                    ls.check_model(&codec)?;
                    let mut bad = 0;
                    let mut seen = 0usize;
                    let raws: Vec<&[u8]> = (0..s.man.n_docs).map(|d| ls.raw(d)).collect();
                    // lote continuo de 64 huecos: no se espera al registro más largo de cada lote
                    codec.decode_stream(&raws, ls.rec_check(), 64, |d, t| {
                        let got = segment::join(&s.meta.row(d), t, &s.meta.authors);
                        if got != items[d] {
                            if bad < 3 {
                                eprintln!("DIFERENCIA en id {}:\n  orig {:?}\n  leído {:?}", items[d].id, items[d], got);
                            }
                            bad += 1;
                        }
                        seen += 1;
                        if seen % 20000 == 0 {
                            eprintln!("  verificados {seen}/{} ({:.0} reg/s)", s.man.n_docs, seen as f64 / t0.elapsed().as_secs_f64());
                        }
                        Ok(())
                    })?;
                    if seen != s.man.n_docs {
                        bail!("solo se decodificaron {seen} de {} registros", s.man.n_docs);
                    }
                    bad
                }
            };
            if bad > 0 {
                bail!("{bad} registros difieren");
            }
            println!("OK: {} registros idénticos al original ({:.1}s)", items.len(), t0.elapsed().as_secs_f64());
        }
        Cmd::Stats { seg, root } => {
            let dirs: Vec<PathBuf> = match seg {
                Some(s) => vec![s],
                None => {
                    let mut v: Vec<PathBuf> = std::fs::read_dir(&root)?
                        .filter_map(|e| e.ok())
                        .map(|e| e.path())
                        .filter(|p| p.join("manifest.json").exists() && !segment::is_transient_dir(p))
                        .collect();
                    v.sort();
                    v
                }
            };
            let mut all_docs = 0usize;
            let mut all_bytes = 0u64;
            let mut by_kind: std::collections::BTreeMap<&str, u64> = Default::default();
            for d in &dirs {
                let m = segment::read_manifest(d)?;
                let sz = segment::sizes(d)?;
                let total: u64 = sz.iter().map(|x| x.1).sum();
                println!("segmento {}: {} docs, nivel {}, {:.1} B/doc", m.name, m.n_docs, m.text_tier, total as f64 / m.n_docs as f64);
                for (f, b) in &sz {
                    if dirs.len() == 1 {
                        println!("  {:<16} {:>12} B  {:>7.2} B/doc", f, b, *b as f64 / m.n_docs as f64);
                    }
                    let kind = if f.starts_with("text") || f.starts_with("lm.") {
                        "texto"
                    } else if f.starts_with("meta") || f.starts_with("authors") {
                        "metadatos"
                    } else if f.starts_with("bm25") {
                        "BM25"
                    } else if f.starts_with("vec") {
                        "vectores"
                    } else {
                        "otros"
                    };
                    *by_kind.entry(kind).or_default() += b;
                }
                all_docs += m.n_docs;
                all_bytes += total;
            }
            println!("TOTAL: {} docs, {} B = {:.1} B/doc", all_docs, all_bytes, all_bytes as f64 / all_docs as f64);
            for (k, b) in by_kind {
                println!("  {:<10} {:>7.1} B/doc  -> 49,3M: {:.2} GB", k, b as f64 / all_docs as f64, b as f64 / all_docs as f64 * 49.3e6 / 1e9);
            }
            println!("  {:<10} {:>7.1} B/doc  -> 49,3M: {:.2} GB", "TOTAL", all_bytes as f64 / all_docs as f64, all_bytes as f64 / all_docs as f64 * 49.3e6 / 1e9);
        }
        Cmd::Bench { seg, models, n } => {
            let s = Segment::open(&seg)?;
            let docs = rand_docs(n, s.man.n_docs, 0x9E3779B97F4A7C15);
            match &s.text {
                TextSource::Zstd(_) => {
                    let t0 = Instant::now();
                    let mut bytes = 0;
                    for &d in &docs {
                        bytes += s.item(d, None)?.text.len();
                    }
                    let dt = t0.elapsed().as_secs_f64();
                    println!("{} lecturas aleatorias (1 hilo, sin caché): {:.1} µs/registro ({} B de texto)", n, dt / n as f64 * 1e6, bytes);
                }
                TextSource::Lm(ls) => {
                    let codec = LmCodec::load(&models.join(s.man.lm_model.as_ref().context("lm_model")?))?;
                    for (label, limit) in [("fragmento (64 tokens)", Some(64usize)), ("completo", None)] {
                        let mut times = Vec::new();
                        for c in docs.chunks(10) {
                            let raws: Vec<&[u8]> = c.iter().map(|&d| ls.raw(d)).collect();
                            let t = Instant::now();
                            codec.decode_records(&raws, true, limit, false, ls.rec_check())?;
                            times.push(t.elapsed().as_secs_f64() * 1e3);
                        }
                        times.sort_by(|a, b| a.total_cmp(b));
                        println!(
                            "nivel LM, lotes de 10 aleatorios, {label}: mediana {:.0} ms, p90 {:.0} ms",
                            times[times.len() / 2],
                            times[times.len() * 9 / 10]
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

#[allow(dead_code)]
fn unused(_: bm25::Bm25Info) {}
