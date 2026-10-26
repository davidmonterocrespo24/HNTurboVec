//! The HTTP server of `hnz serve` (the CLI and the wheel's `hnz` command) and `Archive.serve`
//! (Python), on the library's archive: the same RAM budget, admission control, reranking and JSON
//! as `hnz.open(…)`.
//!
//! `GET /search?q=…&k=10&from=2015&to=2018&type=comment&by=pg&mode=hybrid&deadline_ms=3000` answers
//! the JSON of `search(…, raw=True)`; `/item/<id>`, `/stats` and `/memory` answer theirs; `/` is a
//! search page. A bad query answers 400, no turn in time (`BusyError`) 503, anything else 500.
//!
//! A small HTTP/1.1 server of its own (tiny_http, used before, could be crashed by one request
//! declaring a huge Content-Length, and set no time limits): GET and HEAD only, no request bodies,
//! one request per connection (`Connection: close`), the request head within 16 KiB and 10 s, a
//! response written within 30 s, at most 4 connections per worker waiting in its queue (the rest
//! wait in the OS backlog). Stopping releases the port at once; requests in progress finish and
//! the ones still queued answer 503. Meant for this machine or a private network: on the internet,
//! put it behind a reverse proxy.
use crate::api::{Archive, BusyError, QueryError, SearchRequest};
use anyhow::{anyhow, Context, Result};
use std::io::{ErrorKind, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const PAGE: &str = include_str!("page.html");

/// How often `serve` asks its `stop` callback whether to stop.
const POLL: Duration = Duration::from_millis(200);
/// The accept loop's nap when no connection is waiting.
const IDLE: Duration = Duration::from_millis(5);
/// A request's head (request line and headers): at most this size, received within this time.
const HEAD_BYTES: usize = 16 << 10;
const HEAD_TIME: Duration = Duration::from_secs(10);
/// The same while stopping, for the requests still queued (they only get a 503).
const STOP_HEAD_TIME: Duration = Duration::from_secs(1);
/// A response is written within this time (a client that does not read loses it).
const WRITE_TIME: Duration = Duration::from_secs(30);
/// After answering, how long the client's remaining input is read and dropped, so that closing
/// the socket does not reset the connection before the client has read the answer.
const LINGER_TIME: Duration = Duration::from_millis(200);
const LINGER_BYTES: usize = 64 << 10;
/// Connections waiting for a worker, per worker.
const QUEUE_PER_WORKER: usize = 4;
/// At most this many worker threads.
pub const MAX_THREADS: usize = 256;

/// An HTTP response: status, content type and body.
pub struct Response {
    pub status: u16,
    pub content_type: &'static str,
    pub body: Vec<u8>,
}

impl Response {
    fn json(status: u16, body: String) -> Self {
        Self { status, content_type: "application/json; charset=utf-8", body: body.into_bytes() }
    }

    fn error(status: u16, msg: &str) -> Self {
        Self::json(status, serde_json::json!({"error": msg}).to_string())
    }
}

/// Serves `arch` on `host:port` with `threads` workers (1 to `MAX_THREADS`) until `stop` (asked
/// every 200 ms on the calling thread) returns true or an error. The port is released before it
/// returns; requests in progress finish first.
pub fn serve(arch: Arc<Archive>, host: &str, port: u16, threads: usize, mut stop: impl FnMut() -> Result<bool>) -> Result<()> {
    let listener = TcpListener::bind((host, port)).with_context(|| format!("no se pudo escuchar en {host}:{port}"))?;
    // polled, so that this thread also asks `stop` (in Python, the thread that gets Ctrl+C)
    listener.set_nonblocking(true)?;
    let n = threads.clamp(1, MAX_THREADS);
    let (tx, rx) = sync_channel::<TcpStream>(QUEUE_PER_WORKER * n);
    let rx = Arc::new(Mutex::new(rx));
    let stopping = Arc::new(AtomicBool::new(false));
    let mut workers = Vec::with_capacity(n);
    let mut out = Ok(());
    for i in 0..n {
        let (arch, rx, stopping) = (arch.clone(), rx.clone(), stopping.clone());
        match std::thread::Builder::new().name(format!("hnz-http-{i}")).spawn(move || worker(&arch, &rx, &stopping)) {
            Ok(w) => workers.push(w),
            Err(e) => {
                out = Err(anyhow::Error::new(e).context("no se pudo crear un hilo del servidor HTTP"));
                break;
            }
        }
    }
    drop(rx);
    if out.is_ok() {
        out = accept_loop(&listener, &tx, &mut stop);
    }
    drop(listener); // the port is free from here
    stopping.store(true, Ordering::SeqCst);
    drop(tx); // each worker ends once the queue is empty
    for w in workers {
        let _ = w.join();
    }
    out
}

/// Accepts connections into the workers' queue until `stop` says so.
fn accept_loop(listener: &TcpListener, tx: &SyncSender<TcpStream>, stop: &mut impl FnMut() -> Result<bool>) -> Result<()> {
    let mut held: Option<TcpStream> = None; // accepted, waiting for room in the queue
    let mut asked = Instant::now();
    loop {
        if asked.elapsed() >= POLL {
            if stop()? {
                return Ok(());
            }
            asked = Instant::now();
        }
        if let Some(s) = held.take() {
            match tx.try_send(s) {
                Ok(()) => {}
                Err(TrySendError::Full(s)) => {
                    // every worker busy and the queue full: stop accepting for now
                    held = Some(s);
                    std::thread::sleep(IDLE);
                    continue;
                }
                Err(TrySendError::Disconnected(_)) => return Err(anyhow!("los hilos del servidor HTTP terminaron")),
            }
        }
        match listener.accept() {
            // on Windows an accepted socket inherits the listener's non-blocking mode
            Ok((s, _)) => {
                if s.set_nonblocking(false).is_ok() {
                    held = Some(s);
                }
            }
            Err(e) if e.kind() == ErrorKind::WouldBlock => std::thread::sleep(IDLE),
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            // a client gone before its accept, or no resources right now: carry on
            Err(_) => std::thread::sleep(Duration::from_millis(50)),
        }
    }
}

fn worker(arch: &Archive, rx: &Mutex<Receiver<TcpStream>>, stopping: &AtomicBool) {
    loop {
        // one worker waits on the queue at a time, the others on the lock
        let next = match rx.lock() {
            Ok(q) => q.recv(),
            Err(_) => return,
        };
        let Ok(s) = next else {
            return; // the queue is closed and empty: `serve` is stopping
        };
        // a panic answers nothing (the connection closes) instead of taking the worker down
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| connection(arch, s, stopping.load(Ordering::SeqCst))));
    }
}

/// One connection: its request, the answer, the close.
fn connection(arch: &Archive, mut s: TcpStream, stopping: bool) {
    let (resp, head_only) = match read_head(&mut s, if stopping { STOP_HEAD_TIME } else { HEAD_TIME }) {
        Err(_) => return, // the client left, or sent too slowly: nobody to answer
        Ok(None) => (Response::error(431, "cabecera de la petición demasiado grande"), false),
        Ok(Some(head)) => match parse_head(&head) {
            Err(r) => (r, false),
            Ok((_, head_only)) if stopping => (Response::error(503, "el servidor se está deteniendo"), head_only),
            Ok((target, head_only)) => (respond(arch, &target), head_only),
        },
    };
    let deadline = Instant::now() + WRITE_TIME;
    if write_response(&mut s, &resp, head_only, deadline).is_ok() {
        linger(&mut s);
    }
}

/// The request head (without the blank line that ends it), received within `limit`; None if it
/// is larger than `HEAD_BYTES`.
fn read_head(s: &mut TcpStream, limit: Duration) -> std::io::Result<Option<Vec<u8>>> {
    let t0 = Instant::now();
    let mut head = Vec::with_capacity(1024);
    let mut chunk = [0u8; 4096];
    loop {
        let left = limit.saturating_sub(t0.elapsed());
        if left.is_zero() {
            return Err(ErrorKind::TimedOut.into());
        }
        s.set_read_timeout(Some(left))?;
        let n = s.read(&mut chunk)?;
        if n == 0 {
            return Err(ErrorKind::UnexpectedEof.into());
        }
        let from = head.len().saturating_sub(3); // a terminator split across two reads
        head.extend_from_slice(&chunk[..n]);
        if let Some(i) = head_end(&head[from..]) {
            head.truncate(from + i);
            // the limit holds however the head arrived (its end may come in the read that passes it)
            return Ok((head.len() <= HEAD_BYTES).then_some(head));
        }
        if head.len() > HEAD_BYTES {
            return Ok(None);
        }
    }
}

/// Where the head ends: "\r\n\r\n", or a bare "\n\n".
fn head_end(b: &[u8]) -> Option<usize> {
    (0..b.len()).find(|&i| b[i..].starts_with(b"\r\n\r\n") || b[i..].starts_with(b"\n\n"))
}

/// The request's target and whether it is a HEAD, or the error to answer.
fn parse_head(head: &[u8]) -> std::result::Result<(String, bool), Response> {
    let bad = || Response::error(400, "petición HTTP inválida");
    let text = std::str::from_utf8(head).map_err(|_| bad())?;
    let mut lines = text.split('\n').map(|l| l.strip_suffix('\r').unwrap_or(l));
    let mut parts = lines.next().unwrap_or("").split(' ');
    let (Some(method), Some(target), Some(version), None) = (parts.next(), parts.next(), parts.next(), parts.next()) else {
        return Err(bad());
    };
    if !version.starts_with("HTTP/1.") || !target.starts_with('/') {
        return Err(bad());
    }
    if method != "GET" && method != "HEAD" {
        return Err(Response::error(405, "solo se aceptan GET y HEAD"));
    }
    for line in lines {
        let Some((name, value)) = line.split_once(':') else { continue };
        let name = name.trim();
        // a body is never read: refuse it rather than leave it in the connection
        if name.eq_ignore_ascii_case("transfer-encoding") || (name.eq_ignore_ascii_case("content-length") && value.trim() != "0") {
            return Err(Response::error(400, "las peticiones GET y HEAD no llevan cuerpo"));
        }
    }
    Ok((target.to_string(), method == "HEAD"))
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        431 => "Request Header Fields Too Large",
        503 => "Service Unavailable",
        _ => "Internal Server Error",
    }
}

fn write_response(s: &mut TcpStream, r: &Response, head_only: bool, deadline: Instant) -> std::io::Result<()> {
    let mut head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nConnection: close\r\n",
        r.status,
        reason(r.status),
        r.content_type,
        r.body.len()
    );
    if r.status == 405 {
        head.push_str("Allow: GET, HEAD\r\n");
    }
    head.push_str("\r\n");
    write_until(s, head.as_bytes(), deadline)?;
    if !head_only {
        write_until(s, &r.body, deadline)?;
    }
    s.flush()
}

/// `write_all` that gives up at `deadline` (a socket timeout bounds each write, not their sum).
fn write_until(s: &mut TcpStream, mut data: &[u8], deadline: Instant) -> std::io::Result<()> {
    while !data.is_empty() {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err(ErrorKind::TimedOut.into());
        }
        s.set_write_timeout(Some(left))?;
        match s.write(&data[..data.len().min(64 << 10)]) {
            Ok(0) => return Err(ErrorKind::WriteZero.into()),
            Ok(n) => data = &data[n..],
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Ends the answer (FIN) and reads what the client still sends, briefly, before the close: closing
/// a socket with unread input resets the connection, and the client may lose the answer.
fn linger(s: &mut TcpStream) {
    if s.shutdown(Shutdown::Write).is_err() || s.set_read_timeout(Some(LINGER_TIME)).is_err() {
        return;
    }
    let t0 = Instant::now();
    let (mut buf, mut read) = ([0u8; 4096], 0);
    while read < LINGER_BYTES && t0.elapsed() < LINGER_TIME {
        match s.read(&mut buf) {
            Ok(0) | Err(_) => return,
            Ok(n) => read += n,
        }
    }
}

/// One request (`target` = path and query): the JSON of the library's calls, or the search page.
pub fn respond(arch: &Archive, target: &str) -> Response {
    // errors: 400 for a bad query, 503 when busy, 500 otherwise
    let fail = |e: anyhow::Error| {
        let status = if e.downcast_ref::<QueryError>().is_some() {
            400
        } else if e.downcast_ref::<BusyError>().is_some() {
            503
        } else {
            500
        };
        Response::error(status, &format!("{e:#}"))
    };
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let params = params(query);
    match path {
        "/" => Response { status: 200, content_type: "text/html; charset=utf-8", body: PAGE.as_bytes().to_vec() },
        "/search" => {
            // a key's first non-empty value
            let p = |k: &str| params.iter().find(|(key, v)| key == k && !v.is_empty()).map(|(_, v)| v.clone());
            let num = |k: &str| -> Result<Option<f64>> {
                match p(k) {
                    Some(x) => Ok(Some(x.parse::<f64>().map_err(|_| anyhow::Error::new(QueryError(format!("{k} no es un número: {x:?}"))))?)),
                    None => Ok(None),
                }
            };
            let req = (|| -> Result<SearchRequest> {
                Ok(SearchRequest {
                    query: p("q").unwrap_or_default(),
                    k: num("k")?.map_or(10, |x| x as usize),
                    since: p("from").or_else(|| p("since")),
                    until: p("to").or_else(|| p("until")),
                    types: p("type").or_else(|| p("types")).map(|x| x.split(',').map(String::from).collect()).unwrap_or_default(),
                    author: p("by").or_else(|| p("author")),
                    mode: p("mode").unwrap_or_else(|| "hybrid".into()),
                    full: p("full").is_some_and(|x| x == "1" || x == "true"),
                    include_dead: p("dead").is_some_and(|x| x == "1" || x == "true"),
                    w_bm25: num("w_bm25")?.map(|x| x as f32),
                    c_bm25: num("c_bm25")?.map(|x| x as f32),
                    rerank: match p("rerank").as_deref() {
                        Some("1") | Some("true") => Some(true),
                        Some("0") | Some("false") => Some(false),
                        _ => None,
                    },
                    deadline_ms: num("deadline_ms")?,
                    rerank_depth: num("depth")?.map(|x| x as usize),
                    ..Default::default()
                })
            })();
            match req.and_then(|r| arch.search(&r)).and_then(|r| Ok(serde_json::to_string(&r)?)) {
                Ok(b) => Response::json(200, b),
                Err(e) => fail(e),
            }
        }
        "/stats" => Response::json(200, arch.stats().to_string()),
        "/memory" => Response::json(200, arch.memory().to_string()),
        _ => match path.strip_prefix("/item/") {
            Some(id) => {
                let t0 = Instant::now();
                match id.parse::<u32>() {
                    Ok(id) => match arch.get(id) {
                        Ok(Some(mut v)) => {
                            v["ms"] = serde_json::json!(t0.elapsed().as_secs_f64() * 1e3);
                            Response::json(200, v.to_string())
                        }
                        Ok(None) => Response::error(404, "no encontrado"),
                        Err(e) => fail(e),
                    },
                    Err(_) => Response::error(400, "id inválido"),
                }
            }
            None => Response::error(404, "no encontrado"),
        },
    }
}

/// The query string's (key, value) pairs, percent-decoded ('+' is a space), in order.
fn params(query: &str) -> Vec<(String, String)> {
    query
        .split('&')
        .filter(|kv| !kv.is_empty())
        .map(|kv| {
            let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
            (decode(k), decode(v))
        })
        .collect()
}

fn decode(s: &str) -> String {
    let b = s.as_bytes();
    let hex = |c: u8| (c as char).to_digit(16).map(|d| d as u8);
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < b.len() => match (hex(b[i + 1]), hex(b[i + 2])) {
                (Some(h), Some(l)) => {
                    out.push(h * 16 + l);
                    i += 3;
                    continue;
                }
                _ => out.push(b'%'),
            },
            c => out.push(c),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::{decode, head_end, params, parse_head};

    #[test]
    fn query_strings_are_decoded() {
        let p = params("q=rust+async%20io&k=3&by=&x&&q=second");
        let get = |k: &str| p.iter().find(|(key, _)| key == k).map(|(_, v)| v.as_str());
        assert_eq!(get("q"), Some("rust async io"));
        assert_eq!(get("k"), Some("3"));
        assert_eq!(get("by"), Some(""));
        assert_eq!(get("x"), Some(""));
        assert_eq!(get("mode"), None);
        assert_eq!(p.len(), 5);
        // UTF-8 escapes; '%2B' is a plus; a '%' that starts no escape stays; keys are decoded too
        assert_eq!(decode("caf%C3%A9+100%"), "café 100%");
        assert_eq!(decode("1%2B1%"), "1+1%");
        assert_eq!(decode("%zz%4"), "%zz%4");
        assert_eq!(params("ty%70e=story")[0].0, "type");
        assert_eq!(decode("%FF"), "\u{FFFD}");
    }

    #[test]
    fn request_heads() {
        assert_eq!(head_end(b"GET / HTTP/1.1\r\nHost: x\r\n\r\nrest"), Some(23));
        assert_eq!(head_end(b"GET / HTTP/1.0\n\n"), Some(14));
        assert_eq!(head_end(b"GET / HTTP/1.1\r\nHost: x\r\n"), None);
        assert_eq!(parse_head(b"GET /search?q=a HTTP/1.1\r\nHost: x").ok(), Some(("/search?q=a".to_string(), false)));
        assert_eq!(parse_head(b"HEAD /stats HTTP/1.0").ok(), Some(("/stats".to_string(), true)));
        assert_eq!(parse_head(b"GET / HTTP/1.1\r\nContent-Length: 0").ok(), Some(("/".to_string(), false)));
        let status = |h: &[u8]| parse_head(h).err().map(|r| r.status);
        assert_eq!(status(b"POST /search HTTP/1.1\r\nContent-Length: 9223372036854775807"), Some(405));
        assert_eq!(status(b"GET /search HTTP/1.1\r\nContent-Length: 18446744073709551615"), Some(400));
        assert_eq!(status(b"GET /search HTTP/1.1\r\nTransfer-Encoding: chunked"), Some(400));
        assert_eq!(status(b"GET http://x/search HTTP/1.1"), Some(400));
        assert_eq!(status(b"GET /search"), Some(400));
        assert_eq!(status(b"GET /a b HTTP/1.1"), Some(400));
        assert_eq!(status(b"GET / SPDY/3"), Some(400));
        assert_eq!(status(b"\xff\xfe"), Some(400));
    }
}
