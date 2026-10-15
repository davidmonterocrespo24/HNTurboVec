//! Continuous-batch LM decoding of typed jobs (P4). Records join the batch while their memory
//! bound fits the pool (first fit: a record that does not fit no longer blocks smaller ones behind
//! it) and leave as soon as they are done, freeing their KV cache. A job says what it wants of its
//! record: all of it, a snippet, the title, or the raw prefix the reranker builds the document
//! prompt from (`crate::prompt`). Each sequence is computed identically in any batch, so a result
//! never depends on what else is being decoded or in which order records were admitted.
use super::coder::{decode_token, Decoder};
use super::math::logits_to_cum;
use super::model::Batch;
use super::store::{decode_fallback, finish_bytes, parse_head, partial_rec, verify_check, LmCodec, TITLE_TOKENS};
use crate::textstore::TextRec;
use anyhow::Result;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, Ordering};

/// What a job wants of its record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Want {
    /// the whole record
    Full,
    /// title, url and the first `n` body tokens, as plain text (a snippet)
    Snippet(usize),
    /// the title only
    Title,
    /// the raw LM bytes up to the end of the url plus `body` body tokens (the prompt prefix);
    /// `mark`: also report where a `Snippet(mark)` decode would have stopped
    Prefix { body: usize, mark: usize },
}

/// A decoded prompt prefix.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PrefixOut {
    /// the LM string's bytes, `title\nurl\nbody…` (all of it when `complete`)
    pub bytes: Vec<u8>,
    /// the record's text flags (mode | lmmap << 2)
    pub flags: u8,
    /// the whole record was decoded (and its check byte verified)
    pub complete: bool,
    /// how many of `bytes` a `Snippet(mark)` decode would have produced
    pub snippet_end: usize,
    pub tokens: u32,
}

/// The result of a job.
#[derive(Clone, Debug)]
pub enum Decoded {
    /// (text, partial), as `decode_records_pool` returns it; a zstd or empty record is always
    /// returned whole this way, whatever was wanted
    Text(TextRec, bool),
    Prefix(PrefixOut),
}

/// A record being decoded.
pub(crate) struct Seq {
    pub tag: u64,
    pub want: Want,
    /// someone waits for it now (the service steps these first)
    pub interactive: bool,
    /// order of admission into the batch (the service restarts the youngest bulk one for room)
    pub admitted: u64,
    /// memory it may use in the batch (`LmCodec::record_cost` of its bound)
    pub cost: u64,
    payload: Vec<u8>,
    dec: Decoder<'static>,
    ntok: usize,
    flags: u64,
    check: Option<u8>,
    bound: usize,
    slot: usize,
    bytes: Vec<u8>,
    produced: usize,
    nl: usize,
    body: usize,
    snippet_end: Option<usize>,
    last: u32,
}

impl Seq {
    /// Back to its start, out of any batch (an evicted sequence decodes again from scratch).
    fn restart(&mut self, bos: u32) {
        self.dec = Decoder::owned(self.payload.clone());
        self.slot = usize::MAX;
        self.bytes.clear();
        (self.produced, self.nl, self.body, self.snippet_end, self.last) = (0, 0, 0, None, bos);
    }
}

/// What a job's record needs: nothing more (stored with zstd, or empty), or decoding.
pub(crate) enum Prep {
    Done(Decoded),
    Seq(Box<Seq>),
}

impl LmCodec {
    /// Reads a job's record header.
    pub(crate) fn prepare(&self, tag: u64, rec: &[u8], has_check: bool, want: Want) -> Result<Prep> {
        let h = parse_head(rec, has_check, self.max_tokens())?;
        if h.zstd {
            return Ok(Prep::Done(Decoded::Text(decode_fallback(h.payload)?, false)));
        }
        if h.ntok == 0 {
            return Ok(Prep::Done(Decoded::Text(TextRec::default(), false)));
        }
        let bound = if want == Want::Title { h.ntok.min(TITLE_TOKENS) } else { h.ntok };
        Ok(Prep::Seq(Box::new(Seq {
            tag,
            want,
            interactive: false,
            admitted: 0,
            cost: self.record_cost(bound),
            payload: h.payload.to_vec(),
            dec: Decoder::owned(h.payload.to_vec()),
            ntok: h.ntok,
            flags: h.flags,
            check: h.check,
            bound,
            slot: usize::MAX,
            bytes: Vec::new(),
            produced: 0,
            nl: 0,
            body: 0,
            snippet_end: None,
            last: self.model.h.bos,
        })))
    }

    /// Decodes `recs`, each as `wants[i]` says, in one continuous batch within `pool` bytes.
    pub fn decode_jobs(&self, recs: &[&[u8]], wants: &[Want], has_check: &[bool], parallel: bool, pool: u64) -> Result<Vec<Decoded>> {
        // the caller's thread also runs part of the math (a host may have changed its MXCSR), and
        // must not keep the sampling scratch afterwards (a host may have many threads)
        let _fp = super::fpenv::FpGuard::new();
        let _scratch = super::math::ScratchRelease;
        let n = recs.len();
        let mut out: Vec<Option<Decoded>> = (0..n).map(|_| None).collect();
        let mut queue: VecDeque<Box<Seq>> = VecDeque::new();
        for i in 0..n {
            match self.prepare(i as u64, recs[i], has_check[i], wants[i])? {
                Prep::Done(d) => out[i] = Some(d),
                Prep::Seq(s) => queue.push_back(s),
            }
        }
        let mut run = Runner::new(self, pool);
        loop {
            // first fit, oldest first
            let mut i = 0;
            while i < queue.len() {
                if run.fits(&queue[i]) {
                    let s = queue.remove(i).unwrap();
                    run.start(s);
                } else {
                    i += 1;
                }
            }
            if run.running.is_empty() {
                break;
            }
            for (tag, r) in run.step(parallel, &|_| true) {
                out[tag as usize] = Some(r?);
            }
        }
        Ok(out.into_iter().map(|x| x.expect("every job is decoded")).collect())
    }
}

/// A continuous batch of sequences within a memory pool.
pub(crate) struct Runner<'m> {
    codec: &'m LmCodec,
    batch: Batch<'m>,
    free: Vec<usize>,
    pub running: Vec<Box<Seq>>,
    pub in_use: u64,
    /// memory for the sequences (the pool minus the batch's fixed sampling scratch)
    pub pool: u64,
    raw_pool: u64,
}

impl<'m> Runner<'m> {
    pub fn new(codec: &'m LmCodec, pool: u64) -> Self {
        Self { codec, batch: Batch::new(&codec.model, 0), free: Vec::new(), running: Vec::new(), in_use: 0, pool: pool.saturating_sub(codec.batch_overhead()), raw_pool: pool }
    }

    pub fn set_pool(&mut self, pool: u64) {
        self.pool = pool.saturating_sub(self.codec.batch_overhead());
        self.raw_pool = pool;
    }

    /// The pool it was given (before the batch's fixed part).
    pub fn pool_bytes(&self) -> u64 {
        self.raw_pool
    }

    /// The sequence fits in the pool now (the first one always does: the cap never blocks).
    pub fn fits(&self, s: &Seq) -> bool {
        self.running.is_empty() || self.in_use + s.cost <= self.pool
    }

    pub fn start(&mut self, mut s: Box<Seq>) {
        let slot = self.free.pop().unwrap_or_else(|| self.batch.add_slot());
        self.batch.reset_limited(slot, s.bound);
        s.slot = slot;
        self.in_use += s.cost;
        self.running.push(s);
    }

    fn release(&mut self, s: &Seq) {
        self.batch.reset_limited(s.slot, 0);
        self.free.push(s.slot);
        self.in_use -= s.cost;
    }

    /// Takes running sequence `i` out of the batch, restarted (its KV cache freed).
    pub fn evict(&mut self, i: usize) -> Box<Seq> {
        let mut s = self.running.swap_remove(i);
        self.release(&s);
        s.restart(self.codec.model.h.bos);
        s
    }

    /// Takes every running sequence out (e.g. after a failed step), restarted.
    pub fn evict_all(&mut self) -> Vec<Box<Seq>> {
        let mut v = Vec::new();
        while !self.running.is_empty() {
            v.push(self.evict(self.running.len() - 1));
        }
        v
    }

    /// One step of the running sequences `pick` selects; returns the jobs that finished.
    pub fn step(&mut self, parallel: bool, pick: &dyn Fn(&Seq) -> bool) -> Vec<(u64, Result<Decoded>)> {
        let mut idx: Vec<usize> = (0..self.running.len()).filter(|&i| pick(&self.running[i])).collect();
        if idx.is_empty() {
            return Vec::new();
        }
        // the batch takes its active slots in increasing order
        idx.sort_by_key(|&i| self.running[i].slot);
        let active: Vec<usize> = idx.iter().map(|&i| self.running[i].slot).collect();
        let inp: Vec<u32> = idx.iter().map(|&i| self.running[i].last).collect();
        let logits = self.batch.step(&active, &inp, parallel);
        let got = self.sample(&logits, &idx, parallel);
        let model = &self.codec.model;
        let mut done = vec![false; self.running.len()];
        for (r, &i) in idx.iter().enumerate() {
            let s = &mut self.running[i];
            let t = got[r];
            let tb = model.token_bytes(t);
            s.bytes.extend_from_slice(tb);
            s.produced += 1;
            s.last = t;
            if s.nl >= 2 {
                s.body += 1;
            }
            s.nl += tb.iter().filter(|&&b| b == b'\n').count();
            let fin = match s.want {
                Want::Full => false,
                Want::Title => s.nl >= 1,
                Want::Snippet(l) => s.nl >= 2 && s.body >= l,
                Want::Prefix { body, mark } => {
                    if s.snippet_end.is_none() && s.nl >= 2 && s.body >= mark {
                        s.snippet_end = Some(s.bytes.len());
                    }
                    s.nl >= 2 && s.body >= body
                }
            };
            done[i] = fin || s.produced >= s.bound;
        }
        // finished ones leave the batch at once (from the end, so swap_remove keeps indices valid)
        let mut out = Vec::new();
        for i in (0..done.len()).rev() {
            if done[i] {
                let s = self.running.swap_remove(i);
                self.release(&s);
                out.push((s.tag, finish(*s)));
            }
        }
        out
    }

    /// Softmax -> frequencies -> arithmetic decoder for each active row (in parallel).
    fn sample(&mut self, logits: &[f32], idx: &[usize], parallel: bool) -> Vec<u32> {
        let v = self.codec.model.h.vocab;
        let got: Vec<AtomicU32> = (0..idx.len()).map(|_| AtomicU32::new(0)).collect();
        let seqs = super::kernels::SendPtr(self.running.as_mut_ptr());
        let run = |r: usize| {
            let mut cum = Vec::new();
            logits_to_cum(&logits[r * v..(r + 1) * v], &mut cum);
            // SAFETY: `idx` has no repeated index: each task uses a different sequence
            let s = unsafe { &mut *seqs.get().add(idx[r]) };
            got[r].store(decode_token(&mut s.dec, &cum), Ordering::Relaxed);
        };
        if parallel {
            super::pool::pool().run(idx.len(), &run);
        } else {
            (0..idx.len()).for_each(run);
        }
        got.into_iter().map(|a| a.into_inner()).collect()
    }
}

/// The result of a sequence that is done.
fn finish(s: Seq) -> Result<Decoded> {
    let complete = s.produced == s.ntok;
    match s.want {
        Want::Prefix { .. } => {
            if complete {
                verify_check(&s.bytes, s.check)?;
            }
            let snippet_end = s.snippet_end.unwrap_or(s.bytes.len());
            Ok(Decoded::Prefix(PrefixOut { bytes: s.bytes, flags: s.flags as u8, complete, snippet_end, tokens: s.produced as u32 }))
        }
        _ if complete => Ok(Decoded::Text(finish_bytes(s.bytes, s.check, s.flags)?, false)),
        _ => Ok(Decoded::Text(partial_rec(&String::from_utf8_lossy(&s.bytes), s.flags), true)),
    }
}
