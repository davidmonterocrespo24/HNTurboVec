//! Nivel frío: LM (Pythia/GPT-NeoX, int8 W8A8 determinista) + codificación aritmética.
pub mod coder;
pub mod fpenv;
pub mod jobs;
pub mod kernels;
pub mod math;
pub mod model;
pub mod pool;
pub mod service;
pub mod store;
