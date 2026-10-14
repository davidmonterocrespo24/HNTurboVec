//! Módulo nativo `hnz._hnz`: puente fino entre Python y `hnz::api`. Recibe y devuelve JSON (el
//! envoltorio `hnz/__init__.py` lo convierte a dataclasses) y suelta el GIL mientras trabaja, para que
//! varias consultas de un backend se ejecuten a la vez.
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::path::Path;
use std::sync::Arc;

fn to_py(e: anyhow::Error) -> PyErr {
    if e.downcast_ref::<hnz::api::QueryError>().is_some() {
        PyValueError::new_err(format!("{e}"))
    } else {
        PyRuntimeError::new_err(format!("{e:#}"))
    }
}

/// Archivo hnz abierto (seguro entre hilos).
#[pyclass(frozen, module = "hnz._hnz")]
struct Archive {
    inner: Arc<hnz::api::Archive>,
}

#[pymethods]
impl Archive {
    /// Búsqueda: `request` es el JSON de `hnz::api::SearchRequest`; devuelve el de `SearchResponse`.
    fn search_json(&self, py: Python<'_>, request: &str) -> PyResult<String> {
        let req: hnz::api::SearchRequest = serde_json::from_str(request).map_err(|e| PyValueError::new_err(format!("consulta inválida: {e}")))?;
        let inner = self.inner.clone();
        py.detach(move || inner.search(&req).and_then(|r| Ok(serde_json::to_string(&r)?))).map_err(to_py)
    }

    /// Un ítem por id de HN (JSON) o None si no está en el archivo.
    fn get_json(&self, py: Python<'_>, id: u32) -> PyResult<Option<String>> {
        let inner = self.inner.clone();
        py.detach(move || inner.get(id).and_then(|v| Ok(v.map(|v| v.to_string())))).map_err(to_py)
    }

    /// Vector (EmbeddingGemma 2) de un texto de consulta.
    fn embed(&self, py: Python<'_>, text: String) -> PyResult<Vec<f32>> {
        let inner = self.inner.clone();
        py.detach(move || inner.embed(&text)).map_err(to_py)
    }

    fn stats_json(&self, py: Python<'_>) -> String {
        let inner = self.inner.clone();
        py.detach(move || inner.stats().to_string())
    }
}

/// Abre un archivo hnz. `options` es el JSON de `hnz::api::OpenOptions`.
#[pyfunction]
fn open(py: Python<'_>, path: String, options: &str) -> PyResult<Archive> {
    let opts: hnz::api::OpenOptions = serde_json::from_str(options).map_err(|e| PyValueError::new_err(format!("opciones inválidas: {e}")))?;
    let a = py.detach(move || hnz::api::Archive::open(Path::new(&path), opts)).map_err(to_py)?;
    Ok(Archive { inner: Arc::new(a) })
}

/// Términos que indexa y busca el BM25 de hnz (el mismo tokenizador del motor).
#[pyfunction]
fn terms(text: &str) -> Vec<String> {
    hnz::tokenize::terms(text)
}

/// Texto que el motor indexa para un ítem: título + dominio de la url + texto plano del HTML.
#[pyfunction]
fn index_text(title: &str, url: &str, html: &str) -> String {
    hnz::segment::index_text(title, url, html)
}

#[pymodule]
fn _hnz(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(open, m)?)?;
    m.add_function(wrap_pyfunction!(terms, m)?)?;
    m.add_function(wrap_pyfunction!(index_text, m)?)?;
    m.add_class::<Archive>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("MAX_K", hnz::engine::MAX_K)?;
    Ok(())
}
