//! REVERSIBLE transform of HN's HTML into almost plain text.
//!
//! HN escapes a few characters and uses few tags (`<p>`, `<i>`, `<pre><code>`, links
//! `<a href=".." rel="nofollow">`). Turning them into control characters and unescaping entities
//! makes the text shorter and more predictable (better for zstd and, above all, for the LM).
//! There are two escaping "dialects" depending on the era, and long links are shown truncated
//! with "..." (30-40 % of links), which is stored as (url, length).
//! Every record is checked: if `decode(encode(x)) != x` the original is stored as is.

const ANCHOR_SAME: char = '\u{01}'; // <a href=U rel=nofollow>U</a>          ->  \x01 U \x02
const SEP: char = '\u{02}';
const ANCHOR_DIFF: char = '\u{03}'; // <a href=U rel=nofollow>T</a>          ->  \x03 U \x02 T \x04
const END: char = '\u{04}';
const ANCHOR_TRUNC: char = '\u{11}'; // <a href=U rel=nofollow>U[..k]...</a> ->  \x11 U \x02 k \x04
const TAGS: [(&str, char); 5] = [
    ("<p>", '\u{05}'),
    ("<i>", '\u{06}'),
    ("</i>", '\u{07}'),
    ("<pre><code>", '\u{0e}'),
    ("</code></pre>", '\u{0f}'),
];

/// HN's escaping dialect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dialect {
    /// Current HN: `&amp; &lt; &gt; &quot; &#x27; &#x2F;`
    Modern = 0,
    /// Old HN (~2007-2012): `&#38; &#60; &#62;`; quotes, apostrophes and slashes not escaped
    Legacy = 1,
}

#[inline]
fn is_marker(c: char) -> bool {
    matches!(c, '\u{01}'..='\u{07}' | '\u{0e}' | '\u{0f}' | '\u{11}')
}

#[inline]
fn escape_char(c: char, out: &mut String, d: Dialect) {
    match (d, c) {
        (Dialect::Modern, '&') => out.push_str("&amp;"),
        (Dialect::Modern, '<') => out.push_str("&lt;"),
        (Dialect::Modern, '>') => out.push_str("&gt;"),
        (Dialect::Modern, '"') => out.push_str("&quot;"),
        (Dialect::Modern, '\'') => out.push_str("&#x27;"),
        (Dialect::Modern, '/') => out.push_str("&#x2F;"),
        (Dialect::Legacy, '&') => out.push_str("&#38;"),
        (Dialect::Legacy, '<') => out.push_str("&#60;"),
        (Dialect::Legacy, '>') => out.push_str("&#62;"),
        _ => out.push(c),
    }
}

/// Escapes the way HN does.
pub fn escape(s: &str, out: &mut String, d: Dialect) {
    for c in s.chars() {
        escape_char(c, out, d);
    }
}

/// Unescapes the entities HN produces (`&amp; &lt; &gt; &quot; &apos; &#xHH; &#DD;`).
pub fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        if s.as_bytes()[i] == b'&' {
            if let Some(end) = s[i..].find(';') {
                if end <= 10 {
                    let ent = &s[i + 1..i + end];
                    let ch = match ent {
                        "amp" => Some('&'),
                        "lt" => Some('<'),
                        "gt" => Some('>'),
                        "quot" => Some('"'),
                        "apos" => Some('\''),
                        _ if ent.starts_with("#x") || ent.starts_with("#X") => {
                            u32::from_str_radix(&ent[2..], 16).ok().and_then(char::from_u32)
                        }
                        _ if ent.starts_with('#') => ent[1..].parse::<u32>().ok().and_then(char::from_u32),
                        _ => None,
                    };
                    if let Some(c) = ch {
                        out.push(c);
                        i += end + 1;
                        continue;
                    }
                }
            }
        }
        let c = s[i..].chars().next().unwrap();
        out.push(c);
        i += c.len_utf8();
    }
    out
}

fn push_text(seg: &str, out: &mut String) -> Option<()> {
    let mut t = seg.to_string();
    for (tag, m) in TAGS {
        if t.contains(tag) {
            t = t.replace(tag, m.encode_utf8(&mut [0u8; 4]));
        }
    }
    if t.contains('<') || t.contains('>') {
        return None;
    }
    out.push_str(&unescape(&t));
    Some(())
}

/// HN HTML -> compact form (independent of the dialect). `None` if it cannot be represented.
pub fn encode(s: &str) -> Option<String> {
    if s.chars().any(is_marker) {
        return None;
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    loop {
        match rest.find("<a href=\"") {
            None => {
                push_text(rest, &mut out)?;
                break;
            }
            Some(i) => {
                push_text(&rest[..i], &mut out)?;
                let after = &rest[i + 9..];
                let q = after.find('"')?;
                let href = &after[..q];
                let tail = after[q..].strip_prefix("\" rel=\"nofollow\">")?;
                let e = tail.find("</a>")?;
                let txt = &tail[..e];
                if txt == href {
                    out.push(ANCHOR_SAME);
                    out.push_str(&unescape(href));
                    out.push(SEP);
                } else if txt.len() > 3 && txt.ends_with("...") && href.starts_with(&txt[..txt.len() - 3]) {
                    out.push(ANCHOR_TRUNC);
                    out.push_str(&unescape(href));
                    out.push(SEP);
                    out.push_str(&(txt.len() - 3).to_string());
                    out.push(END);
                } else {
                    out.push(ANCHOR_DIFF);
                    out.push_str(&unescape(href));
                    out.push(SEP);
                    push_text(txt, &mut out)?;
                    out.push(END);
                }
                rest = &tail[e + 4..];
            }
        }
    }
    Some(out)
}

fn decode_text(seg: &str, out: &mut String, d: Dialect) {
    for c in seg.chars() {
        match c {
            '\u{05}' => out.push_str("<p>"),
            '\u{06}' => out.push_str("<i>"),
            '\u{07}' => out.push_str("</i>"),
            '\u{0e}' => out.push_str("<pre><code>"),
            '\u{0f}' => out.push_str("</code></pre>"),
            _ => escape_char(c, out, d),
        }
    }
}

fn find_from(s: &str, from: usize, c: char) -> usize {
    let from = from.min(s.len());
    s[from..].find(c).map(|p| from + p).unwrap_or(s.len())
}

/// Compact form -> original HN HTML.
pub fn decode(s: &str, d: Dialect) -> String {
    let mut out = String::with_capacity(s.len() + s.len() / 6 + 16);
    let mut i = 0;
    while i < s.len() {
        let c = s[i..].chars().next().unwrap();
        if c == ANCHOR_SAME || c == ANCHOR_DIFF || c == ANCHOR_TRUNC {
            let j = find_from(s, i + 1, SEP);
            let mut u = String::new();
            escape(&s[i + 1..j], &mut u, d);
            out.push_str("<a href=\"");
            out.push_str(&u);
            out.push_str("\" rel=\"nofollow\">");
            if c == ANCHOR_SAME {
                out.push_str(&u);
                out.push_str("</a>");
                i = (j + 1).min(s.len());
            } else {
                let k = find_from(s, j + 1, END);
                let inner = &s[(j + 1).min(k)..k];
                if c == ANCHOR_DIFF {
                    decode_text(inner, &mut out, d);
                } else {
                    let n: usize = inner.parse().unwrap_or(0).min(u.len());
                    out.push_str(u.get(..n).unwrap_or(&u));
                    out.push_str("...");
                }
                out.push_str("</a>");
                i = (k + 1).min(s.len());
            }
        } else {
            let next = s[i..]
                .find(|ch| ch == ANCHOR_SAME || ch == ANCHOR_DIFF || ch == ANCHOR_TRUNC)
                .map(|p| i + p)
                .unwrap_or(s.len());
            decode_text(&s[i..next], &mut out, d);
            i = next;
        }
    }
    out
}

/// Storage mode of a text record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Modern = 0,
    Legacy = 1,
    Raw = 2,
}

impl Mode {
    pub fn from_bits(b: u64) -> Mode {
        match b {
            0 => Mode::Modern,
            1 => Mode::Legacy,
            _ => Mode::Raw,
        }
    }
}

/// Returns (mode, string to store). Checks the round trip in the chosen dialect.
pub fn transform(s: &str) -> (Mode, String) {
    if s.is_empty() {
        return (Mode::Modern, String::new());
    }
    if let Some(e) = encode(s) {
        if decode(&e, Dialect::Modern) == s {
            return (Mode::Modern, e);
        }
        if decode(&e, Dialect::Legacy) == s {
            return (Mode::Legacy, e);
        }
    }
    (Mode::Raw, s.to_string())
}

/// Inverse of `transform`.
pub fn restore(mode: Mode, s: &str) -> String {
    match mode {
        Mode::Modern => decode(s, Dialect::Modern),
        Mode::Legacy => decode(s, Dialect::Legacy),
        Mode::Raw => s.to_string(),
    }
}

/// Readable plain text (for indexing and snippets): no tags and no entities.
pub fn to_plain(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut tag = String::new();
    for c in html.chars() {
        if in_tag {
            if c == '>' {
                in_tag = false;
                let t = tag.to_ascii_lowercase();
                if t == "p" || t.starts_with("pre") {
                    out.push('\n');
                }
                tag.clear();
            } else {
                tag.push(c);
            }
        } else if c == '<' {
            in_tag = true;
        } else {
            out.push(c);
        }
    }
    unescape(&out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip_modern_and_legacy() {
        let cases = [
            ("Hello &#x27;world&#x27; &amp; friends &gt; you<p>Second para with <i>it</i>", Mode::Modern),
            ("See <a href=\"https:&#x2F;&#x2F;example.com&#x2F;a?b=1&amp;c=2\" rel=\"nofollow\">https:&#x2F;&#x2F;example.com&#x2F;a?b=1&amp;c=2</a> ok", Mode::Modern),
            ("<a href=\"https:&#x2F;&#x2F;x.com\" rel=\"nofollow\">click here</a> and <pre><code>  let x = 1;\n</code></pre>", Mode::Modern),
            ("<a href=\"https:&#x2F;&#x2F;example.com&#x2F;very&#x2F;long&#x2F;path\" rel=\"nofollow\">https:&#x2F;&#x2F;example.com&#x2F;ve...</a>", Mode::Modern),
            ("it's \"quoted\" a/b &#62; c &#38; d<p><a href=\"http://validator.w3.org/check?uri=http%3A%2F%2Fwww.x.com\" rel=\"nofollow\">http://validator.w3.org/check?uri=http%3A...</a>", Mode::Legacy),
            ("plain text no entities", Mode::Modern),
        ];
        for (c, want) in cases {
            let (m, e) = transform(c);
            assert_eq!(m, want, "unexpected mode for {c}");
            assert_eq!(restore(m, &e), c);
            assert!(e.len() <= c.len());
        }
    }
    #[test]
    fn fallback_on_weird() {
        let c = "has <b>bold</b> tag";
        let (m, e) = transform(c);
        assert_eq!(m, Mode::Raw);
        assert_eq!(restore(m, &e), c);
    }
}
