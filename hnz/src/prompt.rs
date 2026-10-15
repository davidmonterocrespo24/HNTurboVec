//! The text the embedding model sees (prompt version 2). Rust is the reference: the document
//! vectors of a block are computed from these prompts (`hnz prompts` writes their Gemma token ids
//! and `scripts/embed_prompts.py` embeds them), and the reranker (P4) rebuilds the same prompts
//! at query time from the decoded prefix of each candidate.
//!
//! A document's **prefix** is its LM string (`title\nurl\nbody`, the representation the LM
//! compresses) up to the token that completes the second newline, plus `PREFIX_TOKENS` tokens of
//! the body: exactly where the LM decoder stops when asked for `PREFIX_TOKENS` body tokens, so the
//! reranker needs no tokenizer. Its prompt is EmbeddingGemma's document format,
//! `title: <title or "none"> | text: <the prefix body as plain text>`, cut at `MAX_SEQ` Gemma
//! tokens. The URL is not part of the prompt (as in E30a).
use crate::item::Item;
use crate::lm::store::lm_text;

/// Version of everything below; recorded in a block's manifest (`vec.prompt_version`).
pub const PROMPT_VERSION: u32 = 2;
/// Prompt of a query (EmbeddingGemma's retrieval-query format).
pub const QUERY_PROMPT: &str = "task: search result | query: ";
/// Body tokens (of the LM) in a document's prefix.
pub const PREFIX_TOKENS: usize = 128;
/// Gemma tokens of a document prompt (the longest sequence of the document IRs).
pub const MAX_SEQ: usize = 256;

/// Whether an item has a document vector: not deleted, representable as an LM string, and not
/// blank (the same rule as `hnz lmdump`, which produced the E30a embeddings).
pub fn has_prompt(it: &Item) -> bool {
    !it.deleted && lm_text(&it.title, &it.url, &it.text).is_some_and(|s| !s.trim().is_empty())
}

/// Number of tokens of the prefix: up to and including the token that brings the count of newline
/// bytes to 2, then `body` more tokens (or all of them).
pub fn prefix_cut<'a>(ids: &[u32], token_bytes: impl Fn(u32) -> &'a [u8], body: usize) -> usize {
    let mut nl = 0;
    for (i, &t) in ids.iter().enumerate() {
        nl += token_bytes(t).iter().filter(|&&b| b == b'\n').count();
        if nl >= 2 {
            return (i + 1 + body).min(ids.len());
        }
    }
    ids.len()
}

/// The prefix string from its bytes: UTF-8, with a character split by the cut dropped.
pub fn prefix_string(bytes: &[u8]) -> String {
    let s = String::from_utf8_lossy(bytes);
    s.trim_end_matches('\u{FFFD}').to_string()
}

/// The document prompt of a prefix (`flags`: the record's text mode, see `htmlx::Mode`, and
/// whether the body uses the LM's link and tag notation).
pub fn doc_prompt(prefix: &str, flags: u64) -> String {
    let mut parts = prefix.splitn(3, '\n');
    let title = collapse(parts.next().unwrap_or(""));
    let _url = parts.next();
    let body = body_text(parts.next().unwrap_or(""), flags);
    format!("title: {} | text: {body}", if title.is_empty() { "none" } else { title.as_str() })
}

/// Words separated by single spaces.
fn collapse(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Plain text of a body prefix in the LM representation: tags and the markers of links and
/// paragraphs removed (a link with its own text becomes "text (url)"), entities decoded (they are
/// already decoded except in raw HTML), whitespace collapsed. A marker cut by the end of the prefix
/// keeps what it shows.
pub fn body_text(body: &str, flags: u64) -> String {
    use crate::htmlx::Mode;
    let mode = Mode::from_bits(flags & 3);
    if mode == Mode::Raw {
        // HN's HTML as it is: drop the tags (link texts stay), decode the entities
        return collapse(&crate::htmlx::to_plain(body));
    }
    let lm_notation = (flags >> 2) & 1 == 1;
    let mut out = String::with_capacity(body.len());
    let mut i = 0;
    while i < body.len() {
        let rest = &body[i..];
        if lm_notation {
            if let Some(t) = ["<i>", "</i>", "<pre><code>", "</code></pre>"].iter().find(|t| rest.starts_with(**t)) {
                i += t.len();
                continue;
            }
            if let Some(b) = rest.strip_prefix('<') {
                // <url>, <url>k> (a truncated link), <url|text>
                let e = b.find(['>', '|']).unwrap_or(b.len());
                let url = &b[..e];
                let mut used = 1 + e;
                if b.as_bytes().get(e) == Some(&b'|') {
                    let t = b[e + 1..].find('>').unwrap_or(b.len() - e - 1);
                    let text = &b[e + 1..e + 1 + t];
                    push_link(&mut out, text, url);
                    used += 1 + t + (e + 1 + t < b.len()) as usize;
                } else {
                    push_link(&mut out, "", url);
                    if e < b.len() {
                        used += 1;
                        let after = &b[e + 1..];
                        let nd = after.bytes().take_while(|c| c.is_ascii_digit()).count();
                        if nd > 0 && after.as_bytes().get(nd) == Some(&b'>') {
                            used += nd + 1;
                        }
                    }
                }
                i += used.min(rest.len());
                continue;
            }
        } else if let Some((n, text, url)) = compact_link(rest) {
            push_link(&mut out, text, url);
            i += n;
            continue;
        }
        let ch = rest.chars().next().unwrap();
        match ch {
            // the compact form's paragraph mark; other markers (italics, code) vanish
            '\u{05}' => out.push(' '),
            c if (c as u32) < 0x20 && c != '\n' && c != '\t' => {}
            c => out.push(c),
        }
        i += ch.len_utf8();
    }
    collapse(&out)
}

/// A link of the compact form at the start of `s` (\x01 U \x02 | \x03 U \x02 T \x04 |
/// \x11 U \x02 k \x04): (bytes used, its text, its url). The end of the string closes a cut link.
fn compact_link(s: &str) -> Option<(usize, &str, &str)> {
    let kind = s.chars().next()?;
    if !matches!(kind, '\u{01}' | '\u{03}' | '\u{11}') {
        return None;
    }
    let r = &s[1..];
    let j = r.find('\u{02}').unwrap_or(r.len());
    let url = &r[..j];
    if kind == '\u{01}' || j == r.len() {
        return Some((1 + (j + 1).min(r.len()), "", url));
    }
    let a = &r[j + 1..];
    let k = a.find('\u{04}').unwrap_or(a.len());
    let text = if kind == '\u{03}' { &a[..k] } else { "" };
    Some((1 + j + 1 + (k + 1).min(a.len()), text, url))
}

/// `hnz prompts`: the prompts of the items with a vector in each Parquet file, as Gemma token ids
/// (`encode(prompt, true)`, at most `MAX_SEQ`) for `scripts/embed_prompts.py`:
/// `<out>/<file stem>.ids.npy` (u32 HN ids, in the file's order), `.len.npy` (u16 tokens per
/// prompt) and `.tok.npy` (u32, all the token ids one prompt after another); with `text`, also
/// `.prompts.jsonl`. `lm`: an LM model with its tokenizer next to it (`.tokenizer.json`).
pub fn dump(parquet: &[std::path::PathBuf], out: &std::path::Path, lm: &std::path::Path, gemma_tokenizer: &std::path::Path, text: bool) -> anyhow::Result<serde_json::Value> {
    use anyhow::{anyhow, Context};
    use rayon::prelude::*;
    use std::io::Write;
    let codec = crate::lm::store::LmCodec::load(lm)?;
    let gtok = tokenizers::Tokenizer::from_file(gemma_tokenizer).map_err(|e| anyhow!("{}: {e}", gemma_tokenizer.display()))?;
    std::fs::create_dir_all(out)?;
    let mut report = Vec::new();
    for p in parquet {
        let t0 = std::time::Instant::now();
        let stem = p.file_stem().context("file name")?.to_string_lossy().to_string();
        let items = crate::item::read_parquet(p)?.items;
        let rows: Vec<(u32, String, Vec<u32>)> = items
            .par_iter()
            .filter(|it| has_prompt(it))
            .map(|it| {
                let (prefix, flags) = codec.prefix(&it.title, &it.url, &it.text).context("prefix")?;
                let prompt = doc_prompt(&prefix, flags);
                let enc = gtok.encode(prompt.as_str(), true).map_err(|e| anyhow!("gemma tokenizer: {e}"))?;
                let ids: Vec<u32> = enc.get_ids().iter().take(MAX_SEQ).copied().collect();
                Ok((it.id, prompt, ids))
            })
            .collect::<anyhow::Result<_>>()?;
        let n = rows.len();
        let cut = rows.iter().filter(|r| r.2.len() == MAX_SEQ).count();
        let tokens: usize = rows.iter().map(|r| r.2.len()).sum();
        let ids: Vec<u8> = rows.iter().flat_map(|r| r.0.to_le_bytes()).collect();
        let lens: Vec<u8> = rows.iter().flat_map(|r| (r.2.len() as u16).to_le_bytes()).collect();
        let toks: Vec<u8> = rows.iter().flat_map(|r| r.2.iter().flat_map(|t| t.to_le_bytes())).collect();
        write_npy(&out.join(format!("{stem}.ids.npy")), "<u4", n, &ids)?;
        write_npy(&out.join(format!("{stem}.len.npy")), "<u2", n, &lens)?;
        write_npy(&out.join(format!("{stem}.tok.npy")), "<u4", tokens, &toks)?;
        if text {
            let mut w = std::io::BufWriter::new(std::fs::File::create(out.join(format!("{stem}.prompts.jsonl")))?);
            for r in &rows {
                serde_json::to_writer(&mut w, &serde_json::json!({"id": r.0, "prompt": r.1}))?;
                w.write_all(b"\n")?;
            }
            w.flush()?;
        }
        eprintln!("prompts {stem}: {n} of {} items, {:.0} Gemma tokens on average, {cut} cut at {MAX_SEQ} ({:.0} s)", items.len(), tokens as f64 / n.max(1) as f64, t0.elapsed().as_secs_f64());
        report.push(serde_json::json!({"file": p.display().to_string(), "items": items.len(), "prompts": n, "tokens": tokens, "cut": cut}));
    }
    Ok(serde_json::json!({"prompt_version": PROMPT_VERSION, "prefix_tokens": PREFIX_TOKENS, "max_seq": MAX_SEQ, "files": report}))
}

/// A one-dimensional .npy file (format 1.0).
fn write_npy(path: &std::path::Path, descr: &str, n: usize, data: &[u8]) -> anyhow::Result<()> {
    let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({n},), }}");
    // magic (6) + version (2) + header length (2) + header + '\n', a multiple of 64
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut buf = Vec::with_capacity(10 + header.len() + data.len());
    buf.extend_from_slice(b"\x93NUMPY\x01\x00");
    buf.extend_from_slice(&(header.len() as u16).to_le_bytes());
    buf.extend_from_slice(header.as_bytes());
    buf.extend_from_slice(data);
    std::fs::write(path, buf)?;
    Ok(())
}

fn push_link(out: &mut String, text: &str, url: &str) {
    out.push(' ');
    if text.is_empty() || text == url {
        out.push_str(url);
    } else {
        out.push_str(text);
        out.push_str(" (");
        out.push_str(url);
        out.push(')');
    }
    out.push(' ');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompts_of_each_body_form() {
        // LM notation (flags: mode 0, lm notation bit 2)
        let p = doc_prompt("Show HN: a thing\nhttps://x.com\nHello <i>world</i>\n\nsee <https://a.b/c|the docs> and <https://d.e>", 4);
        assert_eq!(p, "title: Show HN: a thing | text: Hello world see the docs (https://a.b/c) and https://d.e");
        // a truncated link <url>k>
        assert_eq!(body_text("go to <https://a.b/c/d>12> now", 4), "go to https://a.b/c/d now");
        // compact form with control markers (no LM notation)
        let c = "a\u{05}b \u{03}https://u\u{02}text\u{04} c \u{01}https://v\u{02} d \u{11}https://w/x\u{02}9\u{04}";
        assert_eq!(body_text(c, 0), "a b text (https://u) c https://v d https://w/x");
        // raw HTML (mode 2)
        assert_eq!(body_text("<p>One &amp; <a href=\"https://z\">two</a>", 2), "One & two");
        // empty title, cut link at the end, cut tag
        assert_eq!(doc_prompt("\n\nbody <https://a.b|lin", 4), "title: none | text: body lin (https://a.b)");
        assert_eq!(doc_prompt("t\n\nx <a hre", 2), "title: t | text: x");
        assert_eq!(doc_prompt("", 0), "title: none | text: ");
    }

    #[test]
    fn cut_after_title_and_url() {
        let toks: Vec<&[u8]> = vec![b"Ti", b"tle\n", b"ht", b"tp\n", b"a", b" b", b" c"];
        let tb = |i: u32| toks[i as usize];
        let ids: Vec<u32> = (0..toks.len() as u32).collect();
        assert_eq!(prefix_cut(&ids, tb, 2), 6);
        assert_eq!(prefix_cut(&ids, tb, 128), 7);
        // both newlines in one token (empty url)
        let toks2: Vec<&[u8]> = vec![b"T", b"\n\n", b"x", b"y"];
        assert_eq!(prefix_cut(&[0, 1, 2, 3], |i| toks2[i as usize], 1), 3);
        // a character split by the cut
        assert_eq!(prefix_string("añ".as_bytes()[..2].as_ref()), "a");
    }
}
