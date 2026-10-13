//! Embedding de consultas con EmbeddingGemma 2 (IR de OpenVINO, CPU) desde Rust.
//! Las librerías de OpenVINO se cargan en tiempo de ejecución (openvino_c.dll del wheel de Python
//! o de una instalación de OpenVINO): se añade su carpeta al PATH antes de crear el Core.

use anyhow::{anyhow, Result};
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

impl Embedder {
    pub fn new(models: &Path, ov_libs: &Path) -> Result<Self> {
        let old = std::env::var("PATH").unwrap_or_default();
        std::env::set_var("PATH", format!("{};{}", ov_libs.display(), old));
        let mut core = Core::new().map_err(ov)?;
        let xml = models.join("gemma2_text_b1_s64.xml");
        let bin = models.join("gemma2_text_b1_s64.bin");
        let model = core.read_model_from_file(&xml.to_string_lossy(), &bin.to_string_lossy()).map_err(ov)?;
        let mut compiled = core.compile_model(&model, DeviceType::CPU).map_err(ov)?;
        let req = compiled.create_infer_request().map_err(ov)?;
        let tok = Tokenizer::from_file(models.join("gemma2_tokenizer.json")).map_err(|e| anyhow!("tokenizer: {e}"))?;
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
