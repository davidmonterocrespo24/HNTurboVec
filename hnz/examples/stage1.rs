//! E30d: time of the first vector stage (Hamming 256 → asymmetric table) at the scale of all of HN.
//!
//! Replicates the real 256-bit codes (MRL prefix) of the 3 sample segments up to ~49.3 M
//! documents (each copy with 8 bits flipped at random, so there are no exact duplicates), writes them
//! to `<dir>/vec.bin` and measures `VecIndex::search` over the WHOLE range with 8 and 16 threads.
//! Usage (from the project root): cargo run --release --manifest-path hnz/Cargo.toml --example stage1 -- [docs] [dir]
use hnz::vecs::{QueryLut, VecIndex};
use std::io::Write;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let target: usize = args.get(1).map(|s| s.parse()).transpose()?.unwrap_or(49_300_000);
    let dir = std::path::PathBuf::from(args.get(2).cloned().unwrap_or_else(|| "out_e30d".into()));
    let bits = 256usize;
    let nb = bits / 8;
    let mut base: Vec<u8> = Vec::new();
    for (seg, sb) in [("out/seg-2024-01", 96usize), ("out/seg-2015-06", 64), ("out/seg-2010-06", 64)] {
        let v = std::fs::read(format!("{seg}/vec.bin"))?;
        for c in v.chunks_exact(sb) {
            base.extend_from_slice(&c[..nb]);
        }
    }
    let n_base = base.len() / nb;
    let reps = target.div_ceil(n_base);
    let n = n_base * reps;
    std::fs::create_dir_all(&dir)?;
    let bin = dir.join("vec.bin");
    if std::fs::metadata(&bin).map(|m| m.len() as usize) .ok() != Some(n * nb) {
        let t0 = Instant::now();
        let mut w = std::io::BufWriter::with_capacity(1 << 24, std::fs::File::create(&bin)?);
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut c = vec![0u8; nb];
        for r in 0..reps {
            for code in base.chunks_exact(nb) {
                c.copy_from_slice(code);
                if r > 0 {
                    for _ in 0..8 {
                        x ^= x << 13;
                        x ^= x >> 7;
                        x ^= x << 17;
                        let b = (x % bits as u64) as usize;
                        c[b / 8] ^= 1 << (b % 8);
                    }
                }
                w.write_all(&c)?;
            }
        }
        w.flush()?;
        std::fs::write(dir.join("vec.has"), vec![0xFFu8; n.div_ceil(64) * 8])?;
        eprintln!("{n} codes ({:.2} GB) written in {:.0} s", (n * nb) as f64 / 1e9, t0.elapsed().as_secs_f64());
    }
    let vi = VecIndex::open(&dir, bits, n)?;
    // queries: random normalized vectors (the time does not depend on their content)
    let mut x: u64 = 12345;
    let mut rnd = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x >> 11) as f32 / (1u64 << 53) as f32 - 0.5
    };
    let queries: Vec<Vec<f32>> = (0..8).map(|_| (0..768).map(|_| rnd()).collect()).collect();
    println!("{n} documents, {bits}-bit codes ({:.2} GB)", (n * nb) as f64 / 1e9);
    // the new selection (histogram) must give exactly the same result as the heap one
    for (i, q) in queries.iter().take(3).enumerate() {
        let lut = QueryLut::new(q, nb);
        for cand in [2_000usize, 20_000] {
            let (t0, a) = (Instant::now(), vi.hamming_select_heap(&lut, 0, n, cand));
            let ta = t0.elapsed().as_secs_f64() * 1e3;
            let (t1, b) = (Instant::now(), vi.hamming_select_pub(&lut, 0, n, cand));
            let tb = t1.elapsed().as_secs_f64() * 1e3;
            assert_eq!(a, b, "query {i}, cand {cand}: the histogram selection differs");
            println!("  query {i}, {cand:>6} candidates: identical; heap {ta:6.1} ms, histogram {tb:6.1} ms");
        }
    }
    for threads in [8usize, 16] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build()?;
        for (cand, k) in [(20_000usize, 100usize), (2_000, 100)] {
            let mut ms: Vec<f64> = pool.install(|| {
                queries
                    .iter()
                    .map(|q| {
                        let lut = QueryLut::new(q, nb);
                        let t = Instant::now();
                        let r = vi.search(&lut, 0, n, cand, k).unwrap();
                        assert_eq!(r.len(), k);
                        t.elapsed().as_secs_f64() * 1e3
                    })
                    .collect()
            });
            let first = ms[0];
            ms.remove(0);
            ms.sort_by(f64::total_cmp);
            let med = ms[ms.len() / 2];
            println!(
                "  {threads:>2} threads, stage 1 = {cand:>6} candidates → top-{k}: 1st query {first:7.1} ms, median {med:7.1} ms ({:.1} ns/doc/thread)",
                med * 1e6 / n as f64 * threads as f64
            );
        }
    }
    Ok(())
}
