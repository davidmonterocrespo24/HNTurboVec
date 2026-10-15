//! Native module `hnz._hnz`: a thin bridge between Python and `hnz::api`. It takes and returns
//! JSON (the `hnz/__init__.py` wrapper turns it into dataclasses) and releases the GIL while it
//! works, so several queries of a backend run at the same time.
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyMemoryError, PyRuntimeError, PyTimeoutError, PyValueError};
use pyo3::prelude::*;
use std::path::Path;
use std::sync::Arc;

create_exception!(hnz, BudgetError, PyMemoryError, "The RAM budget (`ram`) is too small for what hnz needs to open the archive.");
create_exception!(
    hnz,
    BusyError,
    PyTimeoutError,
    "No turn for the operation within `timeout` (too many concurrent operations or not enough working memory)."
);

create_exception!(hnz, BlockError, PyException, "A compact block (hn.hnzb) cannot be used.");
create_exception!(hnz, CorruptBlockError, BlockError, "The block's bytes are damaged (a hash, a checksum or a length does not match).");
create_exception!(
    hnz,
    IncompatibleBlockError,
    BlockError,
    "The block is intact but this version of hnz cannot read it (format, versions, LM model, or canaries that decode differently)."
);

fn to_py(e: anyhow::Error) -> PyErr {
    if e.downcast_ref::<hnz::block::CorruptBlockError>().is_some() {
        return CorruptBlockError::new_err(format!("{e:#}"));
    }
    if e.downcast_ref::<hnz::block::IncompatibleBlockError>().is_some() {
        return IncompatibleBlockError::new_err(format!("{e:#}"));
    }
    if e.downcast_ref::<hnz::api::QueryError>().is_some() {
        PyValueError::new_err(format!("{e}"))
    } else if e.downcast_ref::<hnz::api::BudgetError>().is_some() {
        BudgetError::new_err(format!("{e:#}"))
    } else if e.downcast_ref::<hnz::api::BusyError>().is_some() {
        BusyError::new_err(format!("{e}"))
    } else {
        PyRuntimeError::new_err(format!("{e:#}"))
    }
}

/// An open hnz archive (thread-safe).
#[pyclass(frozen, module = "hnz._hnz")]
struct Archive {
    inner: Arc<hnz::api::Archive>,
}

#[pymethods]
impl Archive {
    /// Search: `request` is the JSON of `hnz::api::SearchRequest`; returns that of `SearchResponse`.
    fn search_json(&self, py: Python<'_>, request: &str) -> PyResult<String> {
        let req: hnz::api::SearchRequest = serde_json::from_str(request).map_err(|e| PyValueError::new_err(format!("consulta inválida: {e}")))?;
        let inner = self.inner.clone();
        py.detach(move || inner.search(&req).and_then(|r| Ok(serde_json::to_string(&r)?))).map_err(to_py)
    }

    /// An item by HN id (JSON), or None if it is not in the archive.
    fn get_json(&self, py: Python<'_>, id: u32) -> PyResult<Option<String>> {
        let inner = self.inner.clone();
        py.detach(move || inner.get(id).and_then(|v| Ok(v.map(|v| v.to_string())))).map_err(to_py)
    }

    /// Vector (EmbeddingGemma 2) of a query text.
    fn embed(&self, py: Python<'_>, text: String) -> PyResult<Vec<f32>> {
        let inner = self.inner.clone();
        py.detach(move || inner.embed(&text)).map_err(to_py)
    }

    fn stats_json(&self, py: Python<'_>) -> String {
        let inner = self.inner.clone();
        py.detach(move || inner.stats().to_string())
    }

    fn memory_json(&self, py: Python<'_>) -> String {
        let inner = self.inner.clone();
        py.detach(move || inner.memory().to_string())
    }
}

/// Opens an hnz archive. `options` is the JSON of `hnz::api::OpenOptions`.
#[pyfunction]
fn open(py: Python<'_>, path: String, options: &str) -> PyResult<Archive> {
    let opts: hnz::api::OpenOptions = serde_json::from_str(options).map_err(|e| PyValueError::new_err(format!("opciones inválidas: {e}")))?;
    let a = py.detach(move || hnz::api::Archive::open(Path::new(&path), opts)).map_err(to_py)?;
    Ok(Archive { inner: Arc::new(a) })
}

/// Checks a compact block ("header", "quick" or "full"); returns the report (JSON).
#[pyfunction]
fn verify_block(py: Python<'_>, path: String, level: String) -> PyResult<String> {
    if !["header", "quick", "full"].contains(&level.as_str()) {
        return Err(PyValueError::new_err(format!("level debe ser header, quick o full (es {level:?})")));
    }
    py.detach(move || hnz::verify::verify(Path::new(&path), &level, None, "none").map(|r| r.to_string())).map_err(to_py)
}

/// Describes a compact block (JSON): parts, sizes, a projection to `project` documents, where an
/// item's bytes are (`id`).
#[pyfunction]
#[pyo3(signature = (path, project=None, lex=None, id=None))]
fn inspect(py: Python<'_>, path: String, project: Option<u64>, lex: Option<String>, id: Option<u32>) -> PyResult<String> {
    py.detach(move || hnz::verify::inspect(Path::new(&path), project, lex.as_deref().map(Path::new), id).map(|r| r.to_string())).map_err(to_py)
}

/// Terms that hnz's BM25 indexes and searches (the engine's own tokenizer).
#[pyfunction]
fn terms(text: &str) -> Vec<String> {
    hnz::tokenize::terms(text)
}

/// Text the engine indexes for an item: title + url domain + plain text of the HTML.
#[pyfunction]
fn index_text(title: &str, url: &str, html: &str) -> String {
    hnz::segment::index_text(title, url, html)
}

#[pymodule]
fn _hnz(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(open, m)?)?;
    m.add_function(wrap_pyfunction!(terms, m)?)?;
    m.add_function(wrap_pyfunction!(index_text, m)?)?;
    m.add_function(wrap_pyfunction!(verify_block, m)?)?;
    m.add_function(wrap_pyfunction!(inspect, m)?)?;
    m.add_class::<Archive>()?;
    m.add("BudgetError", m.py().get_type::<BudgetError>())?;
    m.add("BusyError", m.py().get_type::<BusyError>())?;
    m.add("BlockError", m.py().get_type::<BlockError>())?;
    m.add("CorruptBlockError", m.py().get_type::<CorruptBlockError>())?;
    m.add("IncompatibleBlockError", m.py().get_type::<IncompatibleBlockError>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("MAX_K", hnz::engine::MAX_K)?;
    Ok(())
}
