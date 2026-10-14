//! hnz: Hacker News comprimido al máximo, con búsqueda híbrida y recuperación literal.
pub mod api;
pub mod bm25;
pub mod cache;
pub mod dates;
pub mod embed;
pub mod engine;
pub mod htmlx;
pub mod item;
pub mod lm;
pub mod meta;
pub mod segment;
pub mod textstore;
pub mod tokenize;
pub mod varint;
pub mod vecs;
