//! Almacén de texto del NIVEL FRÍO: cada registro (título, url, texto) se comprime con el LM +
//! codificador aritmético, de forma independiente (acceso aleatorio perfecto).
//!
//! Representación que ve el LM (E15): "título\nurl\ncuerpo", donde el cuerpo es el HTML de HN en
//! forma compacta y "natural": párrafos como "\n\n", enlaces como <url>, <url>k> (truncado) o
//! <url|texto>, y etiquetas <i>, <pre><code> literales. Si no es representable sin ambigüedad se
//! usan los marcadores de control; y si el registro no es apto (muy largo, etc.) se guarda con zstd.

use super::coder::{decode_token, encode_token, Decoder, Encoder};
use super::math::logits_to_cum;
use super::model::{Batch, LmModel};
use crate::htmlx::{self, Mode};
use crate::textstore::TextRec;
use crate::varint::{put, try_get};
use anyhow::{anyhow, bail, Context, Result};
use crate::blob::Blob;
use std::borrow::Cow;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::path::Path;
use tokenizers::Tokenizer;

const GROUP: usize = 32;
/// máximo de tokens por registro (y nunca más que el contexto del modelo - 1)
const MAX_TOKENS: usize = 2000;
/// token cap when decoding only the title (HN titles have <= 80 characters: <= 321 tokens with the
/// newline, even if every byte were a token)
pub(super) const TITLE_TOKENS: usize = 400;

/// Forma compacta (marcadores de htmlx) -> forma "natural" para el LM. None si sería ambigua.
pub fn to_lm(c: &str) -> Option<String> {
    if c.contains(['<', '>', '\n']) {
        return None;
    }
    let mut out = String::with_capacity(c.len() + 16);
    let mut it = c.char_indices().peekable();
    while let Some((i, ch)) = it.next() {
        match ch {
            '\u{05}' => out.push_str("\n\n"),
            '\u{06}' => out.push_str("<i>"),
            '\u{07}' => out.push_str("</i>"),
            '\u{0e}' => out.push_str("<pre><code>"),
            '\u{0f}' => out.push_str("</code></pre>"),
            '\u{01}' | '\u{03}' | '\u{11}' => {
                let rest = &c[i + 1..];
                let j = rest.find('\u{02}')?;
                let url = &rest[..j];
                if url.contains(['|', '\u{04}']) {
                    return None;
                }
                let after = &rest[j + 1..];
                let consumed;
                match ch {
                    '\u{01}' => {
                        out.push('<');
                        out.push_str(url);
                        out.push('>');
                        consumed = j + 1;
                    }
                    _ => {
                        let k = after.find('\u{04}')?;
                        let inner = &after[..k];
                        if inner.chars().any(|x| (x as u32) < 0x20) {
                            return None;
                        }
                        out.push('<');
                        out.push_str(url);
                        if ch == '\u{11}' {
                            out.push('>');
                            out.push_str(inner);
                            out.push('>');
                        } else {
                            out.push('|');
                            out.push_str(inner);
                            out.push('>');
                        }
                        consumed = j + 1 + k + 1;
                    }
                }
                // saltar los caracteres consumidos
                let end = i + 1 + consumed;
                while let Some(&(p, _)) = it.peek() {
                    if p < end {
                        it.next();
                    } else {
                        break;
                    }
                }
            }
            c if (c as u32) < 0x20 => return None,
            _ => out.push(ch),
        }
    }
    Some(out)
}

/// Inverso de `to_lm`.
pub fn from_lm(s: &str) -> Option<String> {
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        let rest = &s[i..];
        if rest.starts_with("\n\n") {
            out.push('\u{05}');
            i += 2;
        } else if rest.starts_with("<i>") {
            out.push('\u{06}');
            i += 3;
        } else if rest.starts_with("</i>") {
            out.push('\u{07}');
            i += 4;
        } else if rest.starts_with("<pre><code>") {
            out.push('\u{0e}');
            i += 11;
        } else if rest.starts_with("</code></pre>") {
            out.push('\u{0f}');
            i += 13;
        } else if rest.starts_with('<') {
            let body = &rest[1..];
            let e = body.find(['>', '|'])?;
            let url = &body[..e];
            if body.as_bytes()[e] == b'|' {
                let t = body[e + 1..].find('>')?;
                out.push('\u{03}');
                out.push_str(url);
                out.push('\u{02}');
                out.push_str(&body[e + 1..e + 1 + t]);
                out.push('\u{04}');
                i += 1 + e + 1 + t + 1;
            } else {
                let after = &body[e + 1..];
                let nd = after.bytes().take_while(|b| b.is_ascii_digit()).count();
                if nd > 0 && after.as_bytes().get(nd) == Some(&b'>') {
                    out.push('\u{11}');
                    out.push_str(url);
                    out.push('\u{02}');
                    out.push_str(&after[..nd]);
                    out.push('\u{04}');
                    i += 1 + e + 1 + nd + 1;
                } else {
                    out.push('\u{01}');
                    out.push_str(url);
                    out.push('\u{02}');
                    i += 1 + e + 1;
                }
            }
        } else if rest.starts_with('\n') {
            return None;
        } else {
            let ch = rest.chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    Some(out)
}

/// Texto exacto que codifica el LM para un registro (None si va con zstd de reserva). Sirve para
/// entrenar o evaluar modelos con la misma representación que usa el nivel frío.
pub fn lm_text(title: &str, url: &str, text: &str) -> Option<String> {
    lm_string(title, url, text).map(|(s, _)| s)
}

/// Cadena que ve el LM + banderas. None si el registro no es apto (se guardará con zstd).
fn lm_string(title: &str, url: &str, text: &str) -> Option<(String, u64)> {
    if title.contains('\n') || url.contains('\n') {
        return None;
    }
    let (mode, compact) = htmlx::transform(text);
    let (body, lmmap) = if mode == Mode::Raw {
        (text.to_string(), 0u64)
    } else {
        match to_lm(&compact) {
            Some(l) if from_lm(&l).as_deref() == Some(compact.as_str()) => (l, 1),
            _ => (compact, 0),
        }
    };
    Some((format!("{title}\n{url}\n{body}"), (lmmap << 2) | mode as u64))
}

fn parse_lm_string(s: &str, flags: u64) -> Result<TextRec> {
    let mut parts = s.splitn(3, '\n');
    let title = parts.next().unwrap_or("").to_string();
    let url = parts.next().unwrap_or("").to_string();
    let body = parts.next().unwrap_or("");
    let mode = Mode::from_bits(flags & 3);
    let compact = if (flags >> 2) & 1 == 1 { from_lm(body).ok_or_else(|| anyhow!("cuerpo LM inválido"))? } else { body.to_string() };
    Ok(TextRec { title, url, text: if mode == Mode::Raw { compact } else { htmlx::restore(mode, &compact) } })
}

/// Convierte un prefijo del cuerpo (representación LM, compacta o HTML) en texto legible.
fn plain_prefix(body: &str, flags: u64) -> String {
    let mode = Mode::from_bits(flags & 3);
    if mode == Mode::Raw {
        return htmlx::to_plain(body);
    }
    let lmmap = (flags >> 2) & 1 == 1;
    let mut out = String::with_capacity(body.len());
    let mut i = 0;
    while i < body.len() {
        let rest = &body[i..];
        if lmmap && rest.starts_with("\n\n") {
            out.push('\n');
            i += 2;
        } else if let Some(t) = ["<i>", "</i>", "<pre><code>", "</code></pre>"].iter().find(|t| lmmap && rest.starts_with(**t)) {
            i += t.len();
        } else if lmmap && rest.starts_with('<') {
            // enlace <url>, <url>k>, <url|texto>: mostrar la url
            let b = &rest[1..];
            let e = b.find(['>', '|']).unwrap_or(b.len());
            out.push_str(&b[..e]);
            let mut j = 1 + e;
            if b.as_bytes().get(e) == Some(&b'|') {
                j += b[e..].find('>').map_or(b.len() - e, |x| x + 1);
            } else if e < b.len() {
                j += 1;
                let after = &b[e + 1..];
                let nd = after.bytes().take_while(|c| c.is_ascii_digit()).count();
                if nd > 0 && after.as_bytes().get(nd) == Some(&b'>') {
                    j += nd + 1;
                }
            }
            i += j.min(rest.len());
        } else {
            let ch = rest.chars().next().unwrap();
            if (ch as u32) >= 0x20 || ch == '\n' {
                out.push(ch);
            } else if ch == '\u{05}' {
                out.push('\n');
            }
            i += ch.len_utf8();
        }
    }
    out
}

/// The snippet of a cut LM string (title, url and the readable start of the body), as the
/// decoder returns it when it stops early.
pub fn snippet_of(s: &str, flags: u64) -> TextRec {
    partial_rec(s, flags)
}

pub(super) fn partial_rec(s: &str, flags: u64) -> TextRec {
    let mut parts = s.splitn(3, '\n');
    let title = parts.next().unwrap_or("").to_string();
    let url = parts.next().unwrap_or("").to_string();
    let body = parts.next().unwrap_or("");
    TextRec { title, url, text: plain_prefix(body, flags) }
}

pub struct LmCodec {
    pub model: LmModel,
    /// needed only to compress (decompression works from the model's own token bytes)
    tok: Option<Tokenizer>,
}

/// See `LmCodec::plan_record`.
pub enum RecPlan {
    /// one header byte
    Empty,
    /// the whole record (stored with zstd)
    Fallback(Vec<u8>),
    /// the LM token ids and the first bytes of the record (header varint + check byte)
    Lm { ids: Vec<u32>, head: Vec<u8> },
}

pub struct EncStats {
    pub lm_records: usize,
    pub zstd_records: usize,
    pub tokens: u64,
    pub payload_bytes: u64,
}

impl LmCodec {
    pub fn load(model_path: &Path) -> Result<Self> {
        let model = LmModel::load(model_path)?;
        if model.h.vocab > 1 << super::coder::VOCAB_BITS {
            bail!("vocabulario de {} tokens: el codificador admite hasta {}", model.h.vocab, 1 << super::coder::VOCAB_BITS);
        }
        let tp = model_path.with_extension("tokenizer.json");
        let tok = Tokenizer::from_file(&tp).map_err(|e| anyhow!("tokenizer {}: {e}", tp.display()))?;
        Ok(Self { model, tok: Some(tok) })
    }

    /// From the bytes of the model (and of its tokenizer, needed only to compress).
    pub fn from_parts(model: Vec<u8>, tokenizer_json: Option<&[u8]>) -> Result<Self> {
        let model = LmModel::from_bytes(model)?;
        if model.h.vocab > 1 << super::coder::VOCAB_BITS {
            bail!("vocabulario de {} tokens: el codificador admite hasta {}", model.h.vocab, 1 << super::coder::VOCAB_BITS);
        }
        let tok = match tokenizer_json {
            Some(j) => Some(Tokenizer::from_bytes(j).map_err(|e| anyhow!("tokenizer: {e}"))?),
            None => None,
        };
        Ok(Self { model, tok })
    }

    fn tokenizer(&self) -> &Tokenizer {
        self.tok.as_ref().expect("este codec se cargó sin tokenizador: solo puede descomprimir")
    }

    /// Comprime un registro (un hilo). Devuelve el registro serializado y su nº de tokens.
    /// Formato: varint(cabecera) [+ 1 byte de comprobación si es LM con tokens] + payload, con
    /// cabecera = ntok<<4 | codec<<3 | lmmap<<2 | modo (codec 1 = zstd de reserva).
    pub fn encode_record(&self, title: &str, url: &str, text: &str) -> (Vec<u8>, u64) {
        let _fp = super::fpenv::FpGuard::new();
        let (ids, head) = match self.plan_record(title, url, text) {
            RecPlan::Empty => {
                let mut out = Vec::new();
                put(&mut out, Mode::Modern as u64);
                return (out, 0);
            }
            RecPlan::Fallback(out) => return (out, 0),
            RecPlan::Lm { ids, head } => (ids, head),
        };
        let mut seq = Vec::with_capacity(ids.len() + 1);
        seq.push(self.model.h.bos);
        seq.extend_from_slice(&ids[..ids.len() - 1]);
        let mut enc = Encoder::new();
        let v = self.model.h.vocab;
        let mut cum = Vec::new();
        // cabeza de vocabulario por trozos de 64 posiciones (~13 MB en vez de T x 200 KB)
        self.model.prefill_chunked(&seq, 64, |p0, logits| {
            for r in 0..logits.len() / v {
                logits_to_cum(&logits[r * v..(r + 1) * v], &mut cum);
                encode_token(&mut enc, &cum, ids[p0 + r]);
            }
        });
        let mut out = head;
        out.extend(enc.finish());
        (out, ids.len() as u64)
    }

    /// The prefix of an item's LM string (see `crate::prompt`: up to the end of the url plus
    /// `PREFIX_TOKENS` body tokens) and the record's text flags, found with the LM tokenizer; for an
    /// LM record it is exactly what the decoder emits when asked for that many body tokens. None if
    /// the item has no LM string.
    pub fn prefix(&self, title: &str, url: &str, text: &str) -> Option<(String, u64)> {
        use crate::prompt::{prefix_cut, prefix_string, PREFIX_TOKENS};
        let (s, flags) = lm_string(title, url, text)?;
        let ids = self.tokenizer().encode(s.as_str(), false).map(|e| e.get_ids().to_vec()).unwrap_or_default();
        let v = self.model.h.vocab as u32;
        if ids.is_empty() || ids.iter().any(|&i| i >= v) {
            // not tokenizable for this model (such records are stored with zstd): the title, the
            // url and the first PREFIX_TOKENS x 4 characters of the body
            let mut parts = s.splitn(3, '\n');
            let (t, u, b) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""), parts.next().unwrap_or(""));
            return Some((format!("{t}\n{u}\n{}", b.chars().take(PREFIX_TOKENS * 4).collect::<String>()), flags));
        }
        let cut = prefix_cut(&ids, |i| self.model.token_bytes(i), PREFIX_TOKENS);
        let bytes: Vec<u8> = ids[..cut].iter().flat_map(|&i| self.model.token_bytes(i).iter().copied()).collect();
        Some((prefix_string(&bytes), flags))
    }

    /// What `encode_record` does with a record, without the arithmetic coding: the whole record if
    /// it is empty or stored with zstd, or the LM token ids and the record's header (varint + check
    /// byte). Pack and verify use it to check copied records against the source items.
    pub fn plan_record(&self, title: &str, url: &str, text: &str) -> RecPlan {
        let fallback = || {
            let mut raw = Vec::new();
            put(&mut raw, title.len() as u64);
            put(&mut raw, url.len() as u64);
            raw.extend_from_slice(title.as_bytes());
            raw.extend_from_slice(url.as_bytes());
            raw.extend_from_slice(text.as_bytes());
            let mut out = Vec::new();
            put(&mut out, 1 << 3 | Mode::Raw as u64);
            out.extend(zstd::bulk::compress(&raw, 19).expect("zstd"));
            RecPlan::Fallback(out)
        };
        // registro vacío (p.ej. una fila deleted): solo 1 byte de cabecera, sin LM
        if title.is_empty() && url.is_empty() && text.is_empty() {
            return RecPlan::Empty;
        }
        let Some((s, flags)) = lm_string(title, url, text) else {
            return fallback();
        };
        let ids: Vec<u32> = match self.tokenizer().encode(s.as_str(), false) {
            Ok(e) => e.get_ids().to_vec(),
            Err(_) => return fallback(),
        };
        let bytes: Vec<u8> = ids.iter().flat_map(|&i| self.model.token_bytes(i).iter().copied()).collect();
        if ids.is_empty()
            || ids.len() > MAX_TOKENS.min(self.model.h.max_pos.saturating_sub(1))
            || bytes != s.as_bytes()
            || ids.iter().any(|&i| i as usize >= self.model.h.vocab)
        {
            return fallback();
        }
        let mut head = Vec::new();
        put(&mut head, (ids.len() as u64) << 4 | flags);
        head.push(check_byte(s.as_bytes()));
        RecPlan::Lm { ids, head }
    }

    /// Descomprime varios registros EN LOTE (los pesos se leen una vez por paso para todos).
    /// - `limit`: máximo de tokens del CUERPO por registro (modo fragmento; título y url completos).
    /// - `title_only`: parar en el primer salto de línea (solo el título).
    /// - `has_check`: los registros llevan byte de comprobación (segmentos `rec_check`).
    /// Devuelve (texto, parcial).
    pub fn decode_records(&self, recs: &[&[u8]], parallel: bool, limit: Option<usize>, title_only: bool, has_check: bool) -> Result<Vec<(TextRec, bool)>> {
        let n = recs.len();
        self.decode_records_each(recs, parallel, &vec![limit; n], &vec![title_only; n], &vec![has_check; n])
    }

    /// Como `decode_records` pero con límite, modo "solo título" y comprobación POR REGISTRO, para
    /// decodificar en un único lote cosas distintas (fragmentos de resultados + títulos de stories).
    pub fn decode_records_each(
        &self,
        recs: &[&[u8]],
        parallel: bool,
        limits: &[Option<usize>],
        title_only: &[bool],
        has_check: &[bool],
    ) -> Result<Vec<(TextRec, bool)>> {
        self.decode_records_pool(recs, parallel, limits, title_only, has_check, u64::MAX)
    }

    /// Maximum memory a record of `ntok` tokens can use in the batch: its full KV cache, x 7/6 for
    /// the moment a layer grows (old and new keys, and values being moved, at once), + its logits
    /// and activations in each step.
    pub fn record_cost(&self, ntok: usize) -> u64 {
        ntok as u64 * self.model.kv_bytes_per_token() * 7 / 6 + self.model.step_bytes_per_seq()
    }

    /// Fixed memory of a batch besides its records: the sampling scratch of every thread that
    /// samples (exponentials, frequencies and cumulative frequencies: vocabulary x 16 B each).
    pub fn batch_overhead(&self) -> u64 {
        (super::pool::pool().threads as u64 + 1) * self.model.h.vocab as u64 * 16
    }

    /// The smallest pool that can decode any record.
    /// Most tokens an LM record can have (longer ones are stored with zstd).
    pub fn max_tokens(&self) -> usize {
        MAX_TOKENS.min(self.model.h.max_pos.saturating_sub(1))
    }

    pub fn min_pool(&self) -> u64 {
        self.max_record_cost() + self.batch_overhead()
    }

    /// Cost of the longest possible record (the LM batch never needs less).
    pub fn max_record_cost(&self) -> u64 {
        self.record_cost(MAX_TOKENS.max(self.model.h.max_pos))
    }

    /// Like `decode_records_each` with a memory cap for the batch (`pool` bytes): records join the
    /// batch while the sum of their bounds (`record_cost`) fits, and each one that finishes frees
    /// its KV cache and makes room for the next (continuous batching, see `super::jobs`). The
    /// result is the same with any cap: each sequence is computed identically in any batch.
    pub fn decode_records_pool(
        &self,
        recs: &[&[u8]],
        parallel: bool,
        limits: &[Option<usize>],
        title_only: &[bool],
        has_check: &[bool],
        pool: u64,
    ) -> Result<Vec<(TextRec, bool)>> {
        use super::jobs::{Decoded, Want};
        let wants: Vec<Want> = (0..recs.len())
            .map(|i| if title_only[i] { Want::Title } else { limits[i].map_or(Want::Full, Want::Snippet) })
            .collect();
        self.decode_jobs(recs, &wants, has_check, parallel, pool)?
            .into_iter()
            .map(|d| match d {
                Decoded::Text(t, p) => Ok((t, p)),
                Decoded::Prefix(_) => anyhow::bail!("decodificación inesperada"),
            })
            .collect()
    }

    /// The batch loop before P4 (in-order admission); kept for the tests that compare both.
    #[doc(hidden)]
    pub fn decode_records_pool_v1(
        &self,
        recs: &[&[u8]],
        parallel: bool,
        limits: &[Option<usize>],
        title_only: &[bool],
        has_check: &[bool],
        pool: u64,
    ) -> Result<Vec<(TextRec, bool)>> {
        // the caller's thread also runs part of the math (a host may have changed its MXCSR), and
        // must not keep the sampling scratch afterwards (a host may have many threads)
        let _fp = super::fpenv::FpGuard::new();
        let _scratch = super::math::ScratchRelease;
        let pool = pool.saturating_sub(self.batch_overhead());
        let n = recs.len();
        let mut out: Vec<Option<(TextRec, bool)>> = vec![None; n];
        let mut lm_idx = Vec::new();
        let mut decs = Vec::new();
        let mut heads = Vec::new();
        for (i, r) in recs.iter().enumerate() {
            let h = parse_head(r, has_check[i], self.max_tokens())?;
            if h.zstd {
                out[i] = Some((decode_fallback(h.payload)?, false));
            } else if h.ntok == 0 {
                out[i] = Some((TextRec::default(), false));
            } else {
                lm_idx.push(i);
                decs.push(Decoder::new(h.payload));
                heads.push(h);
            }
        }
        let k = lm_idx.len();
        let mut bytes: Vec<Vec<u8>> = vec![Vec::new(); k];
        let mut produced = vec![0usize; k];
        if k > 0 {
            let mut nl = vec![0usize; k];
            let mut body = vec![0usize; k];
            let mut done = vec![false; k];
            let mut batch = Batch::new(&self.model, k);
            let mut last = vec![self.model.h.bos; k];
            let prof = std::env::var("HNZ_PROF").is_ok();
            let (mut t_model, mut t_code) = (0f64, 0f64);
            let mut steps = 0;
            // tokens each record may decode (bound of its KV cache) and its cost
            let bound: Vec<usize> = (0..k).map(|j| if title_only[lm_idx[j]] { heads[j].ntok.min(TITLE_TOKENS) } else { heads[j].ntok }).collect();
            let cost: Vec<u64> = bound.iter().map(|&b| self.record_cost(b)).collect();
            let (mut next, mut in_use) = (0usize, 0u64);
            let mut active: Vec<usize> = Vec::new();
            loop {
                // admit in order while it fits (always at least one: the cap never blocks)
                while next < k && (active.is_empty() || in_use + cost[next] <= pool) {
                    batch.reset_limited(next, bound[next]);
                    in_use += cost[next];
                    active.push(next);
                    next += 1;
                }
                if active.is_empty() {
                    break;
                }
                steps += 1;
                let inp: Vec<u32> = active.iter().map(|&j| last[j]).collect();
                let t0 = std::time::Instant::now();
                let logits = batch.step(&active, &inp, parallel);
                let t1 = std::time::Instant::now();
                let got = self.sample(&logits, &active, &mut decs, parallel);
                for (r, &j) in active.iter().enumerate() {
                    let t = got[r];
                    let tb = self.model.token_bytes(t);
                    bytes[j].extend_from_slice(tb);
                    produced[j] += 1;
                    last[j] = t;
                    if nl[j] >= 2 {
                        body[j] += 1;
                    }
                    nl[j] += tb.iter().filter(|&&b| b == b'\n').count();
                    let i = lm_idx[j];
                    if (title_only[i] && nl[j] >= 1) || limits[i].is_some_and(|l| nl[j] >= 2 && body[j] >= l) || produced[j] >= bound[j] {
                        done[j] = true;
                    }
                }
                // finished ones free their KV cache and make room for the next ones
                active.retain(|&j| {
                    if done[j] || produced[j] >= heads[j].ntok {
                        batch.reset_limited(j, 0);
                        in_use -= cost[j];
                        false
                    } else {
                        true
                    }
                });
                if prof {
                    t_model += (t1 - t0).as_secs_f64();
                    t_code += t1.elapsed().as_secs_f64();
                }
            }
            if prof {
                let p: Vec<f64> = super::model::PROF.iter().map(|x| x.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e6).collect();
                eprintln!(
                    "  [perfil] {} pasos: modelo {:.1} ms (matmul capas {:.1}, atención {:.1}, cabeza {:.1}), softmax+decodificador {:.1} ms",
                    steps, t_model * 1e3, p[0], p[1], p[2], t_code * 1e3
                );
            }
        }
        for (j, &i) in lm_idx.iter().enumerate() {
            if produced[j] == heads[j].ntok {
                out[i] = Some((finish_record(std::mem::take(&mut bytes[j]), &heads[j])?, false));
            } else {
                let s = String::from_utf8_lossy(&bytes[j]).to_string();
                out[i] = Some((partial_rec(&s, heads[j].flags), true));
            }
        }
        Ok(out.into_iter().map(|x| x.unwrap()).collect())
    }

    /// Softmax -> frecuencias -> decodificador aritmético para cada fila activa (en paralelo).
    fn sample(&self, logits: &[f32], active: &[usize], decs: &mut [Decoder], parallel: bool) -> Vec<u32> {
        let v = self.model.h.vocab;
        let got: Vec<std::sync::atomic::AtomicU32> = (0..active.len()).map(|_| std::sync::atomic::AtomicU32::new(0)).collect();
        let dec_ptr = super::kernels::SendPtr(decs.as_mut_ptr());
        let run = |r: usize| {
            let j = active[r];
            let mut cum = Vec::new();
            logits_to_cum(&logits[r * v..(r + 1) * v], &mut cum);
            // SAFETY: `active` no repite índices: cada tarea usa un decodificador distinto
            let d = unsafe { &mut *dec_ptr.get().add(j) };
            got[r].store(decode_token(d, &cum), std::sync::atomic::Ordering::Relaxed);
        };
        if parallel {
            super::pool::pool().run(active.len(), &run);
        } else {
            (0..active.len()).for_each(run);
        }
        got.into_iter().map(|a| a.into_inner()).collect()
    }

    /// Descompresión masiva con LOTE CONTINUO: `slots` registros a la vez y, cuando uno termina,
    /// su hueco lo ocupa el siguiente (no se espera al más largo del lote). Llama a
    /// `sink(índice, texto)` por cada registro. Para verify / retier / rebm25.
    pub fn decode_stream(&self, recs: &[&[u8]], has_check: bool, slots: usize, mut sink: impl FnMut(usize, TextRec) -> Result<()>) -> Result<()> {
        let _fp = super::fpenv::FpGuard::new();
        let slots = slots.max(1);
        let mut batch = Batch::new(&self.model, slots);
        let mut slot_rec: Vec<Option<usize>> = vec![None; slots];
        let mut heads: Vec<Option<RecHead>> = (0..slots).map(|_| None).collect();
        let mut decs: Vec<Decoder> = (0..slots).map(|_| Decoder::new(&[])).collect();
        let mut bytes: Vec<Vec<u8>> = vec![Vec::new(); slots];
        let mut produced = vec![0usize; slots];
        let mut last = vec![self.model.h.bos; slots];
        let mut next = 0usize;
        loop {
            // rellenar huecos libres (los registros sin LM se resuelven al momento)
            for s in 0..slots {
                while slot_rec[s].is_none() && next < recs.len() {
                    let i = next;
                    next += 1;
                    let h = parse_head(recs[i], has_check, self.max_tokens())?;
                    if h.zstd {
                        sink(i, decode_fallback(h.payload)?)?;
                    } else if h.ntok == 0 {
                        sink(i, TextRec::default())?;
                    } else {
                        batch.reset(s);
                        decs[s] = Decoder::new(h.payload);
                        heads[s] = Some(h);
                        bytes[s].clear();
                        produced[s] = 0;
                        last[s] = self.model.h.bos;
                        slot_rec[s] = Some(i);
                    }
                }
            }
            let active: Vec<usize> = (0..slots).filter(|&s| slot_rec[s].is_some()).collect();
            if active.is_empty() {
                break;
            }
            let inp: Vec<u32> = active.iter().map(|&s| last[s]).collect();
            let logits = batch.step(&active, &inp, true);
            let got = self.sample(&logits, &active, &mut decs, true);
            for (r, &s) in active.iter().enumerate() {
                let t = got[r];
                bytes[s].extend_from_slice(self.model.token_bytes(t));
                produced[s] += 1;
                last[s] = t;
                let h = heads[s].as_ref().unwrap();
                if produced[s] == h.ntok {
                    let rec = finish_record(std::mem::take(&mut bytes[s]), h)?;
                    let i = slot_rec[s].take().unwrap();
                    heads[s] = None;
                    sink(i, rec)?;
                }
            }
        }
        Ok(())
    }
}

/// Cabecera de un registro del nivel LM.
pub(super) struct RecHead<'a> {
    pub ntok: usize,
    /// lmmap<<2 | modo
    pub flags: u64,
    pub zstd: bool,
    pub check: Option<u8>,
    pub payload: &'a [u8],
}

/// The header of a record; a damaged one (unreadable, or more tokens than any record can have,
/// which would make the decoder run on and on) is a `CorruptBlockError`.
pub(super) fn parse_head(r: &[u8], has_check: bool, max_tok: usize) -> Result<RecHead<'_>> {
    let bad = || crate::block::corrupt("cabecera de registro dañada");
    let mut p = 0;
    let h = try_get(r, &mut p).ok_or_else(bad)?;
    let zstd = (h >> 3) & 1 == 1;
    let ntok = (h >> 4) as usize;
    if ntok > max_tok {
        return Err(bad());
    }
    let mut check = None;
    if !zstd && ntok > 0 && has_check {
        check = Some(*r.get(p).ok_or_else(bad)?);
        p += 1;
    }
    Ok(RecHead { ntok, flags: h & 7, zstd, check, payload: &r[p.min(r.len())..] })
}

pub(super) fn decode_fallback(payload: &[u8]) -> Result<TextRec> {
    let bad = || crate::block::corrupt("registro zstd dañado");
    let raw = zstd::stream::decode_all(payload).map_err(|_| bad())?;
    let mut q = 0;
    let tl = try_get(&raw, &mut q).ok_or_else(bad)? as usize;
    let ul = try_get(&raw, &mut q).ok_or_else(bad)? as usize;
    let (t_end, u_end) = (q.checked_add(tl).ok_or_else(bad)?, q.saturating_add(tl).checked_add(ul).ok_or_else(bad)?);
    if u_end > raw.len() {
        return Err(bad());
    }
    Ok(TextRec {
        title: String::from_utf8(raw[q..t_end].to_vec()).map_err(|_| bad())?,
        url: String::from_utf8(raw[t_end..u_end].to_vec()).map_err(|_| bad())?,
        text: String::from_utf8(raw[u_end..].to_vec()).map_err(|_| bad())?,
    })
}

/// Byte de comprobación de un registro LM (detecta una divergencia del modelo o del binario).
pub fn check_byte(b: &[u8]) -> u8 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &x in b {
        h ^= x as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    (h ^ (h >> 8) ^ (h >> 16) ^ (h >> 24) ^ (h >> 32)) as u8
}

fn finish_record(bytes: Vec<u8>, h: &RecHead) -> Result<TextRec> {
    finish_bytes(bytes, h.check, h.flags)
}

/// The text of a whole decoded record: its check byte verified, then the LM string parsed.
pub(super) fn finish_bytes(bytes: Vec<u8>, check: Option<u8>, flags: u64) -> Result<TextRec> {
    verify_check(&bytes, check)?;
    let s = String::from_utf8(bytes).context("UTF-8 inválido al descomprimir (¿modelo distinto?)")?;
    parse_lm_string(&s, flags)
}

/// The check byte of a whole decoded record.
pub(super) fn verify_check(bytes: &[u8], check: Option<u8>) -> Result<()> {
    if let Some(c) = check {
        if check_byte(bytes) != c {
            // the record's bytes passed their group checksum: this binary decodes them differently
            return Err(crate::block::incompatible("registro corrupto: la comprobación no coincide (¿modelo LM o binario distinto del que comprimió?)"));
        }
    }
    Ok(())
}

/// The text of a whole LM string (title, url and body on three lines, with the record's flags).
pub fn text_of(bytes: Vec<u8>, flags: u64) -> Result<TextRec> {
    let s = String::from_utf8(bytes).context("UTF-8 inválido al descomprimir (¿modelo distinto?)")?;
    parse_lm_string(&s, flags)
}

#[derive(Serialize, Deserialize)]
pub struct LmInfo {
    /// versión de la definición numérica con la que se comprimió (ver model::MATH_VERSION)
    #[serde(default = "math_v1")]
    pub math: u32,
    pub model_id: String,
    pub n_docs: usize,
    pub group: usize,
    pub lm_records: usize,
    pub zstd_records: usize,
    pub tokens: u64,
    /// los registros LM llevan 1 byte de comprobación tras la cabecera (desde la versión 3)
    #[serde(default)]
    pub rec_check: bool,
}

/// Comprime todos los registros (en paralelo, un registro por hilo) y escribe lm.bin/lm.idx/lm.json.
pub fn write(dir: &Path, codec: &LmCodec, recs: &[(&str, &str, &str)], progress: bool) -> Result<EncStats> {
    let done = std::sync::atomic::AtomicUsize::new(0);
    let t0 = std::time::Instant::now();
    let enc: Vec<(Vec<u8>, u64)> = recs
        .par_iter()
        .map(|r| {
            let x = codec.encode_record(r.0, r.1, r.2);
            let d = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if progress && d % 5000 == 0 {
                let el = t0.elapsed().as_secs_f64();
                eprintln!("  LM: {d}/{} registros ({:.0} reg/s, faltan ~{:.0} min)", recs.len(), d as f64 / el, (recs.len() - d) as f64 / (d as f64 / el) / 60.0);
            }
            x
        })
        .collect();
    let mut bin = File::create(dir.join("lm.bin"))?;
    let mut idx = Vec::new();
    let mut off = 0u64;
    let mut st = EncStats { lm_records: 0, zstd_records: 0, tokens: 0, payload_bytes: 0 };
    for g in enc.chunks(GROUP) {
        idx.extend_from_slice(&off.to_le_bytes());
        let mut buf = Vec::new();
        for r in g {
            put(&mut buf, r.0.len() as u64);
        }
        for r in g {
            buf.extend_from_slice(&r.0);
            st.payload_bytes += r.0.len() as u64;
            if r.1 > 0 || (r.0[0] >> 3) & 1 == 0 {
                st.lm_records += 1;
            } else {
                st.zstd_records += 1;
            }
            st.tokens += r.1;
        }
        bin.write_all(&buf)?;
        off += buf.len() as u64;
    }
    idx.extend_from_slice(&off.to_le_bytes());
    std::fs::write(dir.join("lm.idx"), idx)?;
    let info = LmInfo {
        math: super::model::MATH_VERSION,
        model_id: codec.model.id.clone(),
        n_docs: recs.len(),
        group: GROUP,
        lm_records: st.lm_records,
        zstd_records: st.zstd_records,
        tokens: st.tokens,
        rec_check: true,
    };
    std::fs::write(dir.join("lm.json"), serde_json::to_string_pretty(&info)?)?;
    Ok(st)
}

fn math_v1() -> u32 {
    1
}

pub struct LmStore {
    bin: Blob,
    off: Vec<u64>,
    pub info: LmInfo,
    /// blocks: low 32 bits of the XXH3-64 of each group, checked on every read
    gsum: Option<Vec<u32>>,
}

/// Checksum of a text group in a block.
pub fn group_sum(group: &[u8]) -> u32 {
    twox_hash::XxHash3_64::oneshot(group) as u32
}

/// Writes the text of a block (P3): records in document order, in groups of `GROUP` (the record
/// bytes are copied as they are), the u64 offset of each group (then the end) and a checksum per
/// group.
pub struct TextGroupWriter {
    pending: Vec<Vec<u8>>,
    pub groups: Vec<u8>,
    pub idx: Vec<u8>,
    pub gsum: Vec<u8>,
    pub n: usize,
}

impl Default for TextGroupWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl TextGroupWriter {
    pub fn new() -> Self {
        Self { pending: Vec::new(), groups: Vec::new(), idx: Vec::new(), gsum: Vec::new(), n: 0 }
    }

    pub fn push(&mut self, rec: Vec<u8>) {
        self.pending.push(rec);
        self.n += 1;
        if self.pending.len() == GROUP {
            self.flush();
        }
    }

    fn flush(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let mut g = Vec::new();
        for r in &self.pending {
            put(&mut g, r.len() as u64);
        }
        for r in &self.pending {
            g.extend_from_slice(r);
        }
        self.idx.extend_from_slice(&(self.groups.len() as u64).to_le_bytes());
        self.gsum.extend_from_slice(&group_sum(&g).to_le_bytes());
        self.groups.extend_from_slice(&g);
        self.pending.clear();
    }

    pub fn finish(mut self) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        self.flush();
        self.idx.extend_from_slice(&(self.groups.len() as u64).to_le_bytes());
        (self.groups, self.idx, self.gsum)
    }
}

impl LmStore {
    pub fn open(dir: &Path) -> Result<Self> {
        Self::open_with(dir, true)
    }

    pub fn open_with(dir: &Path, resident: bool) -> Result<Self> {
        let bin = Blob::open(&dir.join("lm.bin"), resident).context("lm.bin")?;
        let idx = std::fs::read(dir.join("lm.idx"))?;
        let off = idx.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        let info: LmInfo = serde_json::from_str(&std::fs::read_to_string(dir.join("lm.json"))?)?;
        Ok(Self { bin, off, info, gsum: None })
    }

    /// From the parts of a block: the groups (in RAM or on disk), their offsets and checksums.
    pub fn from_parts(bin: Blob, idx: &[u8], gsum: &[u8], info: LmInfo) -> Result<Self> {
        let bad = |m: &str| crate::block::corrupt(format!("text: {m}"));
        if info.group == 0 {
            return Err(crate::block::incompatible("text: grupo de 0 registros"));
        }
        let ng = info.n_docs.div_ceil(info.group);
        if idx.len() != (ng + 1) * 8 || gsum.len() != ng * 4 {
            return Err(bad("tamaño del índice"));
        }
        let off: Vec<u64> = idx.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        if off.windows(2).any(|w| w[1] < w[0]) || *off.last().unwrap() as usize != bin.len() {
            return Err(bad("desplazamientos"));
        }
        let gsum = gsum.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        Ok(Self { bin, off, info, gsum: Some(gsum) })
    }

    pub fn ram_bytes(&self) -> usize {
        self.bin.resident_bytes() + self.off.capacity() * 8 + self.gsum.as_ref().map_or(0, |g| g.capacity() * 4)
    }

    pub fn blob(&self) -> &Blob {
        &self.bin
    }

    pub fn blob_mut(&mut self) -> &mut Blob {
        &mut self.bin
    }

    /// Compressed bytes of record `doc` (borrowed if lm.bin is resident; read if not).
    pub fn raw(&self, doc: usize) -> Result<Cow<'_, [u8]>> {
        if doc >= self.info.n_docs || doc / self.info.group + 1 >= self.off.len() {
            bail!("registro {doc} fuera de rango ({} documentos)", self.info.n_docs);
        }
        let g = doc / self.info.group;
        let k = doc % self.info.group;
        let data = self.bin.get(self.off[g] as usize, (self.off[g + 1] - self.off[g]) as usize)?;
        let corrupt = || crate::block::corrupt(format!("text group {g} (docs {}..)", g * self.info.group));
        if let Some(gs) = &self.gsum {
            if group_sum(&data) != gs[g] {
                return Err(corrupt());
            }
        }
        let n = (self.info.n_docs - g * self.info.group).min(self.info.group);
        let mut p = 0;
        let lens: Vec<usize> = (0..n).map(|_| crate::varint::try_get(&data, &mut p).map(|x| x as usize)).collect::<Option<_>>().ok_or_else(corrupt)?;
        let start = p + lens[..k].iter().sum::<usize>();
        let end = start + lens[k];
        if end > data.len() {
            return Err(corrupt());
        }
        Ok(match data {
            Cow::Borrowed(b) => Cow::Borrowed(&b[start..end]),
            Cow::Owned(v) => Cow::Owned(v[start..end].to_vec()),
        })
    }

    pub fn rec_check(&self) -> bool {
        self.info.rec_check
    }

    /// Where group `g` is stored: (offset, length) in the text file or part.
    pub fn group_range(&self, g: usize) -> (u64, u64) {
        (self.off[g], self.off[g + 1] - self.off[g])
    }

    pub fn check_model(&self, codec: &LmCodec) -> Result<()> {
        if self.info.math != super::model::MATH_VERSION {
            bail!(
                "el segmento se comprimió con la versión numérica {} del LM y este binario usa la {}: hay que reconstruirlo desde el Parquet (hnz build --tier lm)",
                self.info.math,
                super::model::MATH_VERSION
            );
        }
        if self.info.model_id != codec.model.id && self.info.model_id != codec.model.id_legacy {
            bail!("el segmento se comprimió con {} y el modelo cargado es {}", self.info.model_id, codec.model.id);
        }
        Ok(())
    }
}

/// The prompt prefixes of `text/prefix_fallback`: doc -> (flags, prefix).
pub fn parse_prefix_fallback(raw: &[u8]) -> Option<std::collections::HashMap<u32, (u8, String)>> {
    let n = u32::from_le_bytes(raw.get(..4)?.try_into().ok()?) as usize;
    let mut p = 4;
    let mut m = std::collections::HashMap::with_capacity(n);
    for _ in 0..n {
        let doc = u32::from_le_bytes(raw.get(p..p + 4)?.try_into().ok()?);
        let flags = *raw.get(p + 4)?;
        p += 5;
        let len = crate::varint::try_get(raw, &mut p)? as usize;
        let s = String::from_utf8(raw.get(p..p.checked_add(len)?)?.to_vec()).ok()?;
        p += len;
        m.insert(doc, (flags, s));
    }
    (p == raw.len()).then_some(m)
}
