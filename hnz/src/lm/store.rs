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
use crate::varint::{get, put};
use anyhow::{anyhow, bail, Context, Result};
use memmap2::Mmap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::path::Path;
use tokenizers::Tokenizer;

const GROUP: usize = 32;
/// máximo de tokens por registro (y nunca más que el contexto del modelo - 1)
const MAX_TOKENS: usize = 2000;

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

fn partial_rec(s: &str, flags: u64) -> TextRec {
    let mut parts = s.splitn(3, '\n');
    let title = parts.next().unwrap_or("").to_string();
    let url = parts.next().unwrap_or("").to_string();
    let body = parts.next().unwrap_or("");
    TextRec { title, url, text: plain_prefix(body, flags) }
}

pub struct LmCodec {
    pub model: LmModel,
    tok: Tokenizer,
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
        Ok(Self { model, tok })
    }

    /// Comprime un registro (un hilo). Devuelve el registro serializado y su nº de tokens.
    /// Formato: varint(cabecera) [+ 1 byte de comprobación si es LM con tokens] + payload, con
    /// cabecera = ntok<<4 | codec<<3 | lmmap<<2 | modo (codec 1 = zstd de reserva).
    pub fn encode_record(&self, title: &str, url: &str, text: &str) -> (Vec<u8>, u64) {
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
            (out, 0u64)
        };
        // registro vacío (p.ej. una fila deleted): solo 1 byte de cabecera, sin LM
        if title.is_empty() && url.is_empty() && text.is_empty() {
            let mut out = Vec::new();
            put(&mut out, Mode::Modern as u64);
            return (out, 0);
        }
        let Some((s, flags)) = lm_string(title, url, text) else {
            return fallback();
        };
        let ids: Vec<u32> = match self.tok.encode(s.as_str(), false) {
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
        let mut out = Vec::new();
        put(&mut out, (ids.len() as u64) << 4 | flags);
        out.push(check_byte(s.as_bytes()));
        out.extend(enc.finish());
        (out, ids.len() as u64)
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
        let n = recs.len();
        let mut out: Vec<Option<(TextRec, bool)>> = vec![None; n];
        let mut lm_idx = Vec::new();
        let mut decs = Vec::new();
        let mut heads = Vec::new();
        for (i, r) in recs.iter().enumerate() {
            let h = parse_head(r, has_check[i]);
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
            loop {
                let active: Vec<usize> = (0..k).filter(|&j| !done[j] && produced[j] < heads[j].ntok).collect();
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
                    if (title_only[i] && nl[j] >= 1) || limits[i].is_some_and(|l| nl[j] >= 2 && body[j] >= l) {
                        done[j] = true;
                    }
                }
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
                    let h = parse_head(recs[i], has_check);
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
struct RecHead<'a> {
    ntok: usize,
    /// lmmap<<2 | modo
    flags: u64,
    zstd: bool,
    check: Option<u8>,
    payload: &'a [u8],
}

fn parse_head(r: &[u8], has_check: bool) -> RecHead<'_> {
    let mut p = 0;
    let h = get(r, &mut p);
    let zstd = (h >> 3) & 1 == 1;
    let ntok = (h >> 4) as usize;
    let mut check = None;
    if !zstd && ntok > 0 && has_check {
        check = r.get(p).copied();
        p += 1;
    }
    RecHead { ntok, flags: h & 7, zstd, check, payload: &r[p.min(r.len())..] }
}

fn decode_fallback(payload: &[u8]) -> Result<TextRec> {
    let raw = zstd::stream::decode_all(payload)?;
    let mut q = 0;
    let tl = get(&raw, &mut q) as usize;
    let ul = get(&raw, &mut q) as usize;
    Ok(TextRec {
        title: String::from_utf8(raw[q..q + tl].to_vec())?,
        url: String::from_utf8(raw[q + tl..q + tl + ul].to_vec())?,
        text: String::from_utf8(raw[q + tl + ul..].to_vec())?,
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
    if let Some(c) = h.check {
        if check_byte(&bytes) != c {
            bail!("registro corrupto: la comprobación no coincide (¿modelo LM o binario distinto del que comprimió?)");
        }
    }
    let s = String::from_utf8(bytes).context("UTF-8 inválido al descomprimir (¿modelo distinto?)")?;
    parse_lm_string(&s, h.flags)
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
    bin: Mmap,
    off: Vec<u64>,
    pub info: LmInfo,
}

impl LmStore {
    pub fn open(dir: &Path) -> Result<Self> {
        let bin = unsafe { Mmap::map(&File::open(dir.join("lm.bin")).context("lm.bin")?)? };
        let idx = std::fs::read(dir.join("lm.idx"))?;
        let off = idx.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        let info: LmInfo = serde_json::from_str(&std::fs::read_to_string(dir.join("lm.json"))?)?;
        Ok(Self { bin, off, info })
    }

    /// Bytes comprimidos del registro `doc`.
    pub fn raw(&self, doc: usize) -> &[u8] {
        let g = doc / self.info.group;
        let k = doc % self.info.group;
        let data = &self.bin[self.off[g] as usize..self.off[g + 1] as usize];
        let n = (self.info.n_docs - g * self.info.group).min(self.info.group);
        let mut p = 0;
        let lens: Vec<usize> = (0..n).map(|_| get(data, &mut p) as usize).collect();
        let start = p + lens[..k].iter().sum::<usize>();
        &data[start..start + lens[k]]
    }

    pub fn rec_check(&self) -> bool {
        self.info.rec_check
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
