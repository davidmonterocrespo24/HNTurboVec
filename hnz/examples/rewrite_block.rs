//! Rewrites a compact block with changes, for the incompatibility tests (P3): every part and
//! external file is copied as is, then
//! - `--set <section>.<key>=<json>` changes a manifest value (e.g. `text.math_version=1`), or
//!   `--set /json/pointer=<json>` any value of the manifest (e.g. `/canaries/0/text="…"`);
//! - `--part <name>=<file>` replaces a part's decoded bytes with a file's, or adds a required part
//!   with that name (e.g. `model/lm=models/lm-pythia-70m.hnzlm`).
//! The TOC, hashes and trailer are all valid: only the content changes. With no change the output
//! is byte-identical to the input (the writer is deterministic). Prints the new snapshot id.
//!     cargo run --release --example rewrite_block -- <src> <dst dir> [--set k=v]... [--part name=file]...
use anyhow::{bail, Context, Result};
use hnz::block::{Access, Block, BlockWriter, Codec, PartSpec, FILE_NAME};
use std::path::{Path, PathBuf};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        bail!("usage: rewrite_block <src> <dst dir> [--set section.key=json]... [--part name=file]...");
    }
    let src = Block::open(Path::new(&args[0]))?;
    let dst = PathBuf::from(&args[1]);
    std::fs::create_dir_all(&dst)?;
    let mut manifest = src.manifest().clone();
    let mut replace: Vec<(String, Vec<u8>)> = Vec::new();
    let mut i = 2;
    while i + 1 < args.len() {
        let (k, v) = args[i + 1].split_once('=').context("expected name=value")?;
        match args[i].as_str() {
            "--set" => {
                let value: serde_json::Value = serde_json::from_str(v).with_context(|| format!("{v} is not JSON"))?;
                if k.starts_with('/') {
                    *manifest.pointer_mut(k).with_context(|| format!("no {k} in the manifest"))? = value;
                } else {
                    let (sec, key) = k.split_once('.').context("--set section.key=json")?;
                    manifest[sec][key] = value;
                }
            }
            "--part" => replace.push((k.to_string(), std::fs::read(v).with_context(|| v.to_string())?)),
            o => bail!("unknown option {o}"),
        }
        i += 2;
    }
    let mut w = BlockWriter::create(&dst.join(FILE_NAME))?;
    for p in src.parts() {
        let raw = match replace.iter().position(|r| r.0 == p.name) {
            Some(j) => replace.remove(j).1,
            None => src.read_part(&p.name)?,
        };
        w.add(PartSpec { name: &p.name, ver: p.ver, codec: p.codec, access: p.access, required: p.required, meta: p.meta.clone() }, &raw)?;
    }
    for (name, raw) in &replace {
        w.add(PartSpec::new(name, Codec::Raw, Access::Cold), raw)?;
    }
    for e in src.external() {
        let (from, to) = (src.dir().join(&e.name), dst.join(&e.name));
        if !to.exists() && std::fs::hard_link(&from, &to).is_err() {
            std::fs::copy(&from, &to)?;
        }
        w.add_external(&e.name, &to, &e.role)?;
    }
    let snap = w.finish(manifest, src.n_docs, src.math_version, &src.writer)?;
    println!("{snap:032x}");
    Ok(())
}
