use pyo3::prelude::*;

/// Adds two numbers together.
pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

/// A Python-callable version of the add function.
#[pyfunction]
fn add_py(left: u64, right: u64) -> PyResult<u64> {
    Ok(add(left, right))
}

/// A Python module implemented in Rust.
#[pymodule]
fn rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add_py, m)?)?;
    Ok(())
}
