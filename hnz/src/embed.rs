//! Embedding de consultas con EmbeddingGemma 2 (IR de OpenVINO, CPU) desde Rust.
//! Las librerías de OpenVINO se cargan en tiempo de ejecución (openvino_c.dll del wheel de Python
//! o de una instalación de OpenVINO): se añade su carpeta al PATH antes de crear el Core.

use anyhow::{anyhow, bail, Context, Result};
use openvino::{CompiledModel, Core, DeviceType, ElementType, InferRequest, Shape, Tensor};
use std::path::Path;
use tokenizers::Tokenizer;

pub const QUERY_PROMPT: &str = "task: search result | query: ";

pub struct Embedder {
    // orden de campos = orden de destrucción (request antes que el modelo y el core)
    req: InferRequest,
    _model: CompiledModel,
    _core: Core,
    tok: Tokenizer,
    seq: usize,
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
        // the tokenizer first, so its temporary copy below does not add to OpenVINO's compile peak
        let mut tok = Tokenizer::from_file(models.join("gemma2_tokenizer.json")).map_err(|e| anyhow!("tokenizer: {e}"))?;
        // no BPE cache: Gemma's pre-tokenizer makes each whole query one BPE sequence, so the
        // per-thread cache would only keep up to 10,000 past queries in memory nobody accounts for
        let mut bpe = tok.get_model().clone();
        bpe.resize_cache(0);
        tok.with_model(bpe);
        load_openvino(ov_libs)?;
        let mut core = Core::new().map_err(ov)?;
        // preferir la versión con pesos int8 (NNCF): mitad de tamaño, ~2,5x más rápida, coseno 0,9998
        let name = if models.join("gemma2_text_b1_s64_int8.xml").exists() { "gemma2_text_b1_s64_int8" } else { "gemma2_text_b1_s64" };
        let xml = models.join(format!("{name}.xml"));
        let bin = models.join(format!("{name}.bin"));
        let model = core.read_model_from_file(&xml.to_string_lossy(), &bin.to_string_lossy()).map_err(ov)?;
        let mut compiled = core.compile_model(&model, DeviceType::CPU).map_err(ov)?;
        let req = compiled.create_infer_request().map_err(ov)?;
        Ok(Self { req, _model: compiled, _core: core, tok, seq: 64 })
    }

    /// Vector normalizado (768d) de la consulta.
    pub fn embed_query(&mut self, q: &str) -> Result<Vec<f32>> {
        let enc = self.tok.encode(format!("{QUERY_PROMPT}{q}"), true).map_err(|e| anyhow!("tokenizer: {e}"))?;
        let ids = enc.get_ids();
        let n = ids.len().min(self.seq);
        let shape = Shape::new(&[1, self.seq as i64]).map_err(ov)?;
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
        self.req.set_input_tensor_by_index(0, &ti).map_err(ov)?;
        self.req.set_input_tensor_by_index(1, &tm).map_err(ov)?;
        self.req.infer().map_err(ov)?;
        let out = self.req.get_output_tensor_by_index(0).map_err(ov)?;
        let v: Vec<f32> = out.get_data::<f32>().map_err(ov)?.to_vec();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        Ok(v.iter().map(|x| x / norm).collect())
    }
}
