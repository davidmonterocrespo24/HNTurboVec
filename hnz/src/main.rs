use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use hnz::segment::{self, Segment};
use rayon::prelude::*;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "hnz", about = "Hacker News comprimido con búsqueda híbrida")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
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
