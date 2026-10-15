//! Embedding de consultas con EmbeddingGemma 2 (IR de OpenVINO, CPU) desde Rust.
//! Las librerías de OpenVINO se cargan en tiempo de ejecución (openvino_c.dll del wheel de Python
//! o de una instalación de OpenVINO): se añade su carpeta al PATH antes de crear el Core.

use anyhow::{anyhow, bail, Context, Result};
use openvino::{CompiledModel, Core, DeviceType, ElementType, InferRequest, RwPropertyKey, Shape, Tensor};
use std::path::Path;
use tokenizers::Tokenizer;

pub use crate::prompt::QUERY_PROMPT;

/// CPU streams for document prompts (HNZ_EMBED_STREAMS, 1 to 16; default 1): that many documents
/// are embedded at the same time.
pub fn doc_streams() -> usize {
    std::env::var("HNZ_EMBED_STREAMS").ok().and_then(|x| x.parse().ok()).unwrap_or(1).clamp(1, 16)
}

/// Memory a document request may add on top of the model (activations of a 256-token prompt,
/// with margin; E33a measures it).
pub const DOC_REQUEST_BYTES: u64 = 64 << 20;

/// Sign disagreements between a vector's first `code.len() * 8` dimensions and a stored code
/// (bit d set = dimension d > 0, as `crate::vecs::build` writes it).
pub fn sign_flips(v: &[f32], code: &[u8]) -> u32 {
    (0..(code.len() * 8).min(v.len())).filter(|&d| (v[d] > 0.0) != (code[d / 8] >> (d % 8) & 1 == 1)).count() as u32
}

/// Share of compared signs that may differ, on average, between re-embedded canaries and their
/// stored codes before the reranking chain is taken as broken: a matching chain flips 2.35 of 256
/// on average and at most 9 (E33a, 1,200 prompts), an unrelated vector half of them.
pub const MAX_FLIP_RATE: f64 = 0.125;

/// The reranking chain (LM prefix -> prompt -> tokenizer -> IR) reproduces a block's packed
/// vectors: embeds the canaries of `Engine::embed_checks` and compares their stage-1 signs with
/// the stored codes. Returns the mean flips scaled to 256 signs (None without canaries, or with an
/// embedder that cannot embed documents); `IncompatibleBlockError` above `MAX_FLIP_RATE`.
pub fn check_chain(e: &mut Embedder, checks: &[(String, Vec<u8>)]) -> Result<Option<f64>> {
    if checks.is_empty() || !e.can_embed_docs() {
        return Ok(None);
    }
    let v = e.embed_docs(&checks.iter().map(|c| c.0.clone()).collect::<Vec<_>>())?;
    let flips: Vec<u32> = v.iter().zip(checks).map(|(v, c)| sign_flips(v, &c.1)).collect();
    let signs = checks.iter().map(|c| (c.1.len() * 8).min(v[0].len())).sum::<usize>().max(1) as f64;
    let rate = flips.iter().sum::<u32>() as f64 / signs;
    if rate > MAX_FLIP_RATE {
        return Err(crate::block::incompatible(format!(
            "el embedder no reproduce los vectores del bloque: {:.0} % de los signos distintos en {} canarios ({flips:?})",
            rate * 100.0,
            flips.len()
        )));
    }
    Ok(Some(rate * 256.0))
}

pub struct Embedder {
    // orden de campos = orden de destrucción (request antes que el modelo y el core)
    req: InferRequest,
    /// more requests for documents, run at the same time (one per CPU stream)
    doc_reqs: Vec<InferRequest>,
    _model: CompiledModel,
    _core: Core,
    tok: Tokenizer,
    seq: usize,
    /// the IR takes any shape (P4: one IR embeds queries and document prompts)
    dynamic: bool,
}

fn ov<E: std::fmt::Debug>(e: E) -> anyhow::Error {
    anyhow!("OpenVINO: {e:?}")
}

/// Carpeta desde la que se cargó OpenVINO en este proceso (solo se puede cargar una vez).
static OV_DIR: std::sync::Mutex<Option<std::path::PathBuf>> = std::sync::Mutex::new(None);

/// Carga `openvino_c.dll` de `ov_libs` por su ruta completa, una vez por proceso. Así manda
/// `ov_libs` (el buscador de openvino-rs miraría antes OPENVINO_*/INTEL_OPENVINO_DIR) y sus
/// dependencias (openvino.dll, tbb, plugins) se resuelven en la misma carpeta, que se pone la
/// primera del PATH una sola vez.
fn load_openvino(ov_libs: &Path) -> Result<()> {
    let mut g = OV_DIR.lock().unwrap_or_else(|e| e.into_inner());
    let dir = std::path::absolute(ov_libs).with_context(|| format!("carpeta de OpenVINO {}", ov_libs.display()))?;
    let same = |a: &Path, b: &Path| std::fs::canonicalize(a).ok() == std::fs::canonicalize(b).ok();
    if let Some(d) = g.as_ref() {
        if !same(d, &dir) {
            bail!("OpenVINO ya está cargado desde {} en este proceso; no se puede usar también {}", d.display(), dir.display());
        }
        return Ok(());
    }
    let dll = dir.join("openvino_c.dll");
    if !dll.exists() {
        bail!("no está {} (ov_libs debe ser la carpeta libs del paquete openvino)", dll.display());
    }
    let path = std::env::var_os("PATH").unwrap_or_default();
    if !std::env::split_paths(&path).any(|p| same(&p, &dir)) {
        let mut v = vec![dir.clone()];
        v.extend(std::env::split_paths(&path));
        std::env::set_var("PATH", std::env::join_paths(v)?);
    }
    openvino_sys::library::load_from(dll).map_err(|e| anyhow!("OpenVINO: {e}"))?;
    *g = Some(dir);
    Ok(())
}

impl Embedder {
    pub fn new(models: &Path, ov_libs: &Path) -> Result<Self> {
        let name = if models.join("gemma2_text_b1_s64_int8.xml").exists() { "gemma2_text_b1_s64_int8" } else { "gemma2_text_b1_s64" };
        let tok = std::fs::read(models.join("gemma2_tokenizer.json")).context("gemma2_tokenizer.json")?;
        Self::from_files(&models.join(format!("{name}.xml")), &models.join(format!("{name}.bin")), &tok, ov_libs)
    }

    /// From the IR files and the tokenizer's JSON (a compact block keeps the tokenizer inside).
    pub fn from_files(xml: &Path, bin: &Path, tokenizer_json: &[u8], ov_libs: &Path) -> Result<Self> {
        // the tokenizer first, so its temporary copy below does not add to OpenVINO's compile peak
        let mut tok = Tokenizer::from_bytes(tokenizer_json).map_err(|e| anyhow!("tokenizer: {e}"))?;
        // no BPE cache: Gemma's pre-tokenizer makes each whole query one BPE sequence, so the
        // per-thread cache would only keep up to 10,000 past queries in memory nobody accounts for
        let mut bpe = tok.get_model().clone();
        bpe.resize_cache(0);
        tok.with_model(bpe);
        load_openvino(ov_libs)?;
        let mut core = Core::new().map_err(ov)?;
        // f32 everywhere: the CPU plugin would choose bf16 on AMX servers, and the vectors must
        // not depend on the machine
        core.set_property(&DeviceType::CPU, &RwPropertyKey::Other("INFERENCE_PRECISION_HINT".into()), "f32").map_err(ov)?;
        // documents can be embedded several at a time (HNZ_EMBED_STREAMS CPU streams; default 1)
        let streams = doc_streams();
        if streams > 1 {
            core.set_property(&DeviceType::CPU, &RwPropertyKey::NumStreams, &streams.to_string()).map_err(ov)?;
        }
        // CPU threads and core type of the embedder (HNZ_EMBED_THREADS, HNZ_EMBED_CORES =
        // ANY_CORE / PCORE_ONLY / ECORE_ONLY; default: OpenVINO's choice): while a reranked query
        // decodes, the embedder shares the CPU with the LM
        if let Some(t) = std::env::var("HNZ_EMBED_THREADS").ok().and_then(|x| x.parse::<usize>().ok()).filter(|&t| t > 0) {
            core.set_property(&DeviceType::CPU, &RwPropertyKey::Other("INFERENCE_NUM_THREADS".into()), &t.to_string()).map_err(ov)?;
        }
        if let Ok(c) = std::env::var("HNZ_EMBED_CORES") {
            core.set_property(&DeviceType::CPU, &RwPropertyKey::Other("SCHEDULING_CORE_TYPE".into()), &c).map_err(ov)?;
        }
        let model = core.read_model_from_file(&xml.to_string_lossy(), &bin.to_string_lossy()).map_err(ov)?;
        let dynamic = model.is_dynamic();
        let mut compiled = core.compile_model(&model, DeviceType::CPU).map_err(ov)?;
        let req = compiled.create_infer_request().map_err(ov)?;
        let doc_reqs = if dynamic { (1..streams).map(|_| compiled.create_infer_request().map_err(ov)).collect::<Result<_>>()? } else { Vec::new() };
        Ok(Self { req, doc_reqs, _model: compiled, _core: core, tok, seq: 64, dynamic })
    }

    /// The IR takes any shape: it can embed document prompts (P4).
    pub fn can_embed_docs(&self) -> bool {
        self.dynamic
    }

    /// Document prompts `embed_docs` embeds at the same time (one per CPU stream).
    pub fn doc_parallelism(&self) -> usize {
        1 + self.doc_reqs.len()
    }

    /// The normalized vector (768-dim) of a document prompt (`crate::prompt::doc_prompt`), at
    /// most `prompt::MAX_SEQ` tokens, computed at its own canonical shape [1, tokens rounded up to
    /// 16]: it never depends on what else is embedded. Needs an IR with dynamic shapes.
    pub fn embed_doc(&mut self, prompt: &str) -> Result<Vec<f32>> {
        Ok(self.embed_docs(std::slice::from_ref(&prompt.to_string()))?.remove(0))
    }

    /// The vectors of several document prompts (as `embed_doc`, each at its own shape), up to one
    /// per CPU stream at the same time.
    pub fn embed_docs(&mut self, prompts: &[String]) -> Result<Vec<Vec<f32>>> {
        if !self.dynamic {
            bail!("el embedder de este bloque tiene formas fijas: no puede embeber documentos");
        }
        let mut ids: Vec<Vec<u32>> = Vec::with_capacity(prompts.len());
        for p in prompts {
            let enc = self.tok.encode(p.as_str(), true).map_err(|e| anyhow!("tokenizer: {e}"))?;
            ids.push(enc.get_ids().iter().take(crate::prompt::MAX_SEQ).copied().collect());
        }
        let mut out: Vec<Vec<f32>> = vec![Vec::new(); ids.len()];
        let mut reqs: Vec<&mut InferRequest> = std::iter::once(&mut self.req).chain(self.doc_reqs.iter_mut()).collect();
        let mut next = 0;
        while next < ids.len() {
            // one document on each request, then collect them in order
            let mut started = Vec::new();
            for (slot, r) in reqs.iter_mut().enumerate() {
                if next == ids.len() {
                    break;
                }
                let (ti, tm) = tensors(&ids[next], ids[next].len().div_ceil(16).max(1) * 16)?;
                r.set_input_tensor_by_index(0, &ti).map_err(ov)?;
                r.set_input_tensor_by_index(1, &tm).map_err(ov)?;
                r.infer_async().map_err(ov)?;
                started.push((slot, next, ti, tm));
                next += 1;
            }
            for (slot, i, _ti, _tm) in started {
                reqs[slot].wait(600_000).map_err(ov)?;
                out[i] = normalized(reqs[slot])?;
            }
        }
        Ok(out)
    }

    /// Vector normalizado (768d) de la consulta.
    pub fn embed_query(&mut self, q: &str) -> Result<Vec<f32>> {
        let enc = self.tok.encode(format!("{QUERY_PROMPT}{q}"), true).map_err(|e| anyhow!("tokenizer: {e}"))?;
        let ids = enc.get_ids().to_vec();
        self.infer(&ids, self.seq)
    }

    /// Runs the model on `ids` padded (or cut) to `len` tokens; returns the normalized vector.
    fn infer(&mut self, ids: &[u32], len: usize) -> Result<Vec<f32>> {
        let (ti, tm) = tensors(ids, len)?;
        self.req.set_input_tensor_by_index(0, &ti).map_err(ov)?;
        self.req.set_input_tensor_by_index(1, &tm).map_err(ov)?;
        self.req.infer().map_err(ov)?;
        normalized(&self.req)
    }
}

/// The input tensors of one sequence: `ids` padded (or cut) to `len` tokens, and its mask.
fn tensors(ids: &[u32], len: usize) -> Result<(Tensor, Tensor)> {
    let n = ids.len().min(len);
    let shape = Shape::new(&[1, len as i64]).map_err(ov)?;
    let mut ti = Tensor::new(ElementType::I64, &shape).map_err(ov)?;
    let mut tm = Tensor::new(ElementType::I64, &shape).map_err(ov)?;
    {
        let d = ti.get_data_mut::<i64>().map_err(ov)?;
        for (i, x) in d.iter_mut().enumerate() {
            *x = if i < n { ids[i] as i64 } else { 0 };
        }
    }
    {
        let d = tm.get_data_mut::<i64>().map_err(ov)?;
        for (i, x) in d.iter_mut().enumerate() {
            *x = (i < n) as i64;
        }
    }
    Ok((ti, tm))
}

/// A finished request's output, L2-normalized.
fn normalized(req: &InferRequest) -> Result<Vec<f32>> {
    let out = req.get_output_tensor_by_index(0).map_err(ov)?;
    let v: Vec<f32> = out.get_data::<f32>().map_err(ov)?.to_vec();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    Ok(v.iter().map(|x| x / norm).collect())
}

#[cfg(test)]
mod tests {
    #[test]
    fn sign_flips_counts_disagreements() {
        let v: Vec<f32> = (0..16).map(|d| if d % 3 == 0 { 0.5 } else { -0.5 }).collect();
        let mut code = vec![0u8; 2];
        for d in (0..16).filter(|d| d % 3 == 0) {
            code[d / 8] |= 1 << (d % 8);
        }
        assert_eq!(super::sign_flips(&v, &code), 0);
        code[0] ^= 0b101; // dimensions 0 and 2
        assert_eq!(super::sign_flips(&v, &code), 2);
        // zero is "not positive", as `vecs::build` writes it
        assert_eq!(super::sign_flips(&[0.0; 8], &[0u8]), 0);
        assert_eq!(super::sign_flips(&[0.0; 8], &[0xffu8]), 8);
    }
}
