//! Tokenizador para BM25 (idéntico al indexar y al consultar).
//! Minúsculas, secuencias alfanuméricas (Unicode), sin apóstrofos ("don't" -> "dont"),
//! casos especiales de HN: "c++", "c#", "f#". Sin stopwords, sin números largos.

pub const STOPWORDS: &[&str] = &[
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be",
    "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "could", "did", "do",
    "does", "doing", "dont", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off",
    "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should",
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would", "you", "your", "yours",
    "yourself", "yourselves", "s", "t", "im", "ive", "youre", "thats", "isnt", "doesnt", "didnt", "cant", "wont",
    "also", "get", "got", "like", "one", "would", "really", "thing", "things", "lot", "even", "much", "make", "http",
    "https", "www", "com", "amp", "quot", "nofollow", "href",
];

fn is_stop(t: &str) -> bool {
    use std::collections::HashSet;
    use std::sync::OnceLock;
    static SET: OnceLock<HashSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| STOPWORDS.iter().copied().collect()).contains(t)
}

/// Llama a `f` con cada término indexable.
pub fn for_each_term(text: &str, mut f: impl FnMut(&str)) {
    let mut tok = String::with_capacity(32);
    let mut chars = text.chars().peekable();
    loop {
        let c = chars.next();
        match c {
            Some(c) if c.is_alphanumeric() => {
                for l in c.to_lowercase() {
                    tok.push(l);
                }
            }
            Some('\'') | Some('’') if !tok.is_empty() && chars.peek().map_or(false, |n| n.is_alphabetic()) => {
                // apóstrofo dentro de palabra: se elimina (don't -> dont)
            }
            other => {
                if !tok.is_empty() {
                    // c++ / c# / f#
                    if other == Some('+') && chars.peek() == Some(&'+') && tok.len() <= 2 {
                        chars.next();
                        tok.push_str("++");
                    } else if other == Some('#') && tok.len() <= 2 {
                        tok.push('#');
                    }
                    emit(&tok, &mut f);
                    tok.clear();
                }
                if other.is_none() {
                    break;
                }
            }
        }
    }
}

fn emit(tok: &str, f: &mut impl FnMut(&str)) {
    let n = tok.chars().count();
    if n == 0 || n > 32 {
        return;
    }
    let all_digits = tok.bytes().all(|b| b.is_ascii_digit());
    if all_digits && n > 4 {
        return;
    }
    // letras sueltas: solo nombres de lenguajes (C, R); el resto es ruido
    if n == 1 && !all_digits && !matches!(tok, "c" | "r") {
        return;
    }
    if is_stop(tok) {
        return;
    }
    f(tok);
}

pub fn terms(text: &str) -> Vec<String> {
    let mut v = Vec::new();
    for_each_term(text, |t| v.push(t.to_string()));
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn basic() {
        assert_eq!(terms("I don't like C++ and C# in 2024, see Rust's async!"), vec!["c++", "c#", "2024", "see", "rusts", "async"]);
        assert_eq!(terms("Ünïcode Straße 1234567"), vec!["ünïcode", "straße"]);
        assert_eq!(terms("PHP vs HTML vs .NET in C and R, not x"), vec!["php", "vs", "html", "vs", "net", "c", "r"]);
    }
}
