//! Repeatable communication-fault injection for uncertainty-aware abstention.
//!
//! A depth stream can lose, repeat, reorder, or corrupt its events on the
//! way from the exchange. This module makes those failures repeatable: it
//! injects dropped, duplicate, delayed, and book-crossing depth events
//! into a deterministic stream with frozen seeds, replays the result
//! through the production sync machine, and records one integrity state
//! per observation.
//!
//! The rules are simple and strict:
//!
//! - A signal (top-5 order-flow imbalance) is emitted only from a valid
//!   book. While the book is invalid the observation abstains: no touch
//!   quote and no signal are recorded.
//! - Every injected fault is written to the fault and coverage ledger
//!   with its kind, frozen seed, event index, and a traceable reason.
//!   Faults are never relabeled as clean data, including the idempotent
//!   duplicate redelivery, which changes nothing but is still recorded.
//! - The ledger is the experimental prerequisite for integrity-conditioned
//!   abstention work: it preserves exactly which observations were
//!   faulted, why, and what was abstained.

use crate::booksync::{BookSync, DepthEvent, DepthOutcome, MergeOutcome, SnapshotData, TouchQuote};
use crate::direction_manifest::sha256_hex;
use crate::orderbook::OrderBook;

/// Number of deterministic mutation traces in the acceptance set.
pub const TRACE_COUNT: usize = 20;

/// Frozen base seed for the acceptance traces. Trace `i` uses
/// `BASE_SEED + i`, so the whole set replays identically every run.
pub const BASE_SEED: u64 = 20261212;

/// Number of depth events in each trace stream.
pub const N_EVENTS: usize = 40;

/// Update id of the opening snapshot; the generated stream starts at
/// `SNAPSHOT_ID + 1`.
pub const SNAPSHOT_ID: i64 = 100;

/// Deterministic xorshift64 generator (same family as the replay tests).
/// Frozen seeds are the whole point: a trace replays bit-identically.
#[derive(Debug, Clone)]
pub struct SeededRng(u64);

impl SeededRng {
    pub fn new(seed: u64) -> Self {
        // A zero state would stick, so only zero maps to a fixed nonzero
        // state. Every other seed is kept exactly, so distinct seeds
        // always start distinct.
        Self(if seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            seed
        })
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut s = self.0;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        self.0 = s;
        s
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (self.next_u64() as f64 / u64::MAX as f64) * (hi - lo)
    }
}

/// One of the four communication faults the harness injects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultKind {
    /// The event never arrives; the next event opens a sequence gap.
    Dropped,
    /// The exact event is delivered twice in a row.
    Duplicate,
    /// The event arrives after its successor, out of order.
    Delayed,
    /// The event carries a bid above the best ask and would cross the book.
    SnapshotCrossing,
}

impl FaultKind {
    pub fn name(self) -> &'static str {
        match self {
            FaultKind::Dropped => "dropped",
            FaultKind::Duplicate => "duplicate",
            FaultKind::Delayed => "delayed",
            FaultKind::SnapshotCrossing => "snapshot-crossing",
        }
    }

    pub fn all() -> [FaultKind; 4] {
        [
            FaultKind::Dropped,
            FaultKind::Duplicate,
            FaultKind::Delayed,
            FaultKind::SnapshotCrossing,
        ]
    }
}

/// One frozen fault specification: which fault, where, under which seed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaultSpec {
    pub seed: u64,
    pub kind: FaultKind,
    pub event_index: usize,
}

/// Integrity state of one replayed observation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntegrityState {
    /// The book is synced and uncrossed; a signal may be emitted.
    Valid,
    /// The book cannot be trusted; the observation must abstain.
    Invalid { reason: String },
}

impl IntegrityState {
    pub fn is_valid(&self) -> bool {
        matches!(self, IntegrityState::Valid)
    }

    pub fn reason(&self) -> Option<&str> {
        match self {
            IntegrityState::Valid => None,
            IntegrityState::Invalid { reason } => Some(reason),
        }
    }
}

/// One replayed observation: the integrity state, the touch quote when
/// valid, and the signal when valid. An invalid observation carries
/// neither a quote nor a signal.
#[derive(Debug, Clone)]
pub struct Observation {
    pub event_index: usize,
    pub event_ms: i64,
    pub update_id: i64,
    pub integrity: IntegrityState,
    pub touch: Option<(f64, f64, f64, f64)>,
    pub signal: Option<f64>,
}

/// The signal for one observation: top-5 order-flow imbalance, emitted
/// only from a valid book. An invalid book abstains and yields `None`.
pub fn signal_for(book: &OrderBook, integrity: &IntegrityState) -> Option<f64> {
    match integrity {
        IntegrityState::Valid => book.imbalance(5),
        IntegrityState::Invalid { .. } => None,
    }
}

/// One ledger entry: a single injected fault and what it cost.
#[derive(Debug, Clone)]
pub struct LedgerEntry {
    pub trace_id: usize,
    pub seed: u64,
    pub kind: FaultKind,
    pub event_index: usize,
    pub reason: String,
    /// True when at least one observation abstained because of this fault.
    pub abstained: bool,
}

/// The fault and coverage ledger: every injected fault, preserved as the
/// experimental prerequisite for integrity-conditioned abstention work.
#[derive(Debug, Default, Clone)]
pub struct FaultLedger {
    entries: Vec<LedgerEntry>,
}

impl FaultLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, entry: LedgerEntry) {
        self.entries.push(entry);
    }

    pub fn entries(&self) -> &[LedgerEntry] {
        &self.entries
    }

    pub fn total_faults(&self) -> usize {
        self.entries.len()
    }

    /// How many faults of this kind the ledger holds.
    pub fn coverage(&self, kind: FaultKind) -> usize {
        self.entries.iter().filter(|e| e.kind == kind).count()
    }

    /// How many recorded faults forced at least one abstention.
    pub fn abstentions(&self) -> usize {
        self.entries.iter().filter(|e| e.abstained).count()
    }

    /// Every recorded reason, in ledger order.
    pub fn reasons(&self) -> Vec<String> {
        self.entries.iter().map(|e| e.reason.clone()).collect()
    }

    /// Deterministic serialization of the ledger, one fault per line.
    pub fn canonical_text(&self) -> String {
        let mut out = String::new();
        for e in &self.entries {
            out.push_str(&format!(
                "trace={} seed={} kind={} event={} abstained={} reason={}\n",
                e.trace_id,
                e.seed,
                e.kind.name(),
                e.event_index,
                e.abstained,
                e.reason
            ));
        }
        out
    }

    /// Digest of the canonical ledger text; identical ledgers hash alike.
    pub fn digest(&self) -> String {
        sha256_hex(self.canonical_text().as_bytes())
    }
}

/// A clean depth stream plus the reference snapshots used to rebuild a
/// fresh snapshot whenever a gap forces a resync. `after[i]` is the
/// reference book state after clean event `i`.
#[derive(Debug, Clone)]
pub struct CleanStream {
    pub snapshot: SnapshotData,
    pub events: Vec<DepthEvent>,
    pub after: Vec<SnapshotData>,
}

const BID_PRICES: [f64; 3] = [100.0, 99.9, 99.8];
const ASK_PRICES: [f64; 3] = [100.1, 100.2, 100.3];

/// Build a deterministic clean stream: a gapless in-order sequence of
/// level updates that never crosses the book. The same seed always
/// builds the same stream.
pub fn build_clean_stream(n_events: usize, seed: u64) -> CleanStream {
    let mut rng = SeededRng::new(seed);
    let snapshot = SnapshotData {
        last_update_id: SNAPSHOT_ID,
        bids: vec![(100.0, 5.0), (99.9, 3.0), (99.8, 2.0)],
        asks: vec![(100.1, 4.0), (100.2, 2.0), (100.3, 1.0)],
    };
    let mut book = OrderBook::new();
    book.from_snapshot(&snapshot.bids, &snapshot.asks, snapshot.last_update_id);
    let mut events = Vec::with_capacity(n_events);
    let mut after = Vec::with_capacity(n_events);
    for i in 0..n_events {
        let u = SNAPSHOT_ID + 1 + i as i64;
        let qty = (rng.range(0.5, 9.5) * 10.0).round() / 10.0;
        let (bids, asks) = if rng.next_u64().is_multiple_of(2) {
            let p = BID_PRICES[rng.below(BID_PRICES.len())];
            (vec![(p, qty)], vec![])
        } else {
            let p = ASK_PRICES[rng.below(ASK_PRICES.len())];
            (vec![], vec![(p, qty)])
        };
        let ev = DepthEvent {
            event_ms: 1_700_000_000_000 + u,
            u_first: u,
            u_last: u,
            bids,
            asks,
        };
        assert_eq!(
            book.apply_diff(ev.u_first, ev.u_last, &ev.bids, &ev.asks),
            crate::orderbook::SyncResult::Applied,
            "clean fixture stream must stay gapless"
        );
        assert!(
            !book.is_crossed(),
            "clean fixture stream must stay uncrossed"
        );
        after.push(SnapshotData {
            last_update_id: book.last_id,
            bids: book.bid_levels(),
            asks: book.ask_levels(),
        });
        events.push(ev);
    }
    CleanStream {
        snapshot,
        events,
        after,
    }
}

/// The reference snapshot for a given update id: the opening snapshot
/// for ids at or below it, otherwise the reference state after the
/// clean event with that id.
fn snapshot_for(stream: &CleanStream, update_id: i64) -> &SnapshotData {
    if update_id <= SNAPSHOT_ID {
        return &stream.snapshot;
    }
    let idx = (update_id - SNAPSHOT_ID - 1) as usize;
    stream.after.get(idx).unwrap_or_else(|| {
        stream
            .after
            .last()
            .expect("clean stream always holds at least one snapshot")
    })
}

/// Inject one fault into a copy of the clean event stream,
/// deterministically. The fault placement comes from the spec, so the
/// same spec always produces the same delivered stream.
pub fn inject_fault(stream: &CleanStream, spec: FaultSpec) -> Vec<DepthEvent> {
    let mut delivered = stream.events.clone();
    if spec.event_index >= delivered.len() {
        return delivered;
    }
    match spec.kind {
        FaultKind::Dropped => {
            delivered.remove(spec.event_index);
        }
        FaultKind::Duplicate => {
            let copy = delivered[spec.event_index].clone();
            delivered.insert(spec.event_index + 1, copy);
        }
        FaultKind::Delayed => {
            if spec.event_index + 1 < delivered.len() {
                delivered.swap(spec.event_index, spec.event_index + 1);
            }
        }
        FaultKind::SnapshotCrossing => {
            let pre = if spec.event_index == 0 {
                &stream.snapshot
            } else {
                &stream.after[spec.event_index - 1]
            };
            let best_ask = pre
                .asks
                .iter()
                .map(|(p, _)| *p)
                .fold(f64::INFINITY, f64::min);
            let cross_bid = best_ask + 0.05;
            delivered[spec.event_index].bids = vec![(cross_bid, 1.0)];
        }
    }
    delivered
}

/// Digest of a touch-quote sequence: the reference digest for a replay.
/// Canonical bytes are the event time followed by the four touch values.
pub fn digest_touch_quotes(quotes: &[TouchQuote]) -> String {
    let mut buf = Vec::with_capacity(quotes.len() * 40);
    for (ms, (bp, bq, ap, aq)) in quotes {
        buf.extend_from_slice(&ms.to_le_bytes());
        for v in [*bp, *bq, *ap, *aq] {
            buf.extend_from_slice(&v.to_bits().to_le_bytes());
        }
    }
    sha256_hex(&buf)
}

/// Replay one delivered stream through the production sync machine and
/// record one observation per event. A gap forces an immediate resync
/// from the reference snapshot at the gap event, mirroring the
/// recorder fetching a fresh snapshot after a resync.
fn replay(stream: &CleanStream, delivered: &[DepthEvent]) -> Vec<Observation> {
    let mut sync = BookSync::new();
    let mut quotes = Vec::new();
    let mut synced = matches!(
        sync.merge_snapshot(&stream.snapshot, &mut quotes),
        MergeOutcome::Synced { .. }
    );
    let mut observations = Vec::with_capacity(delivered.len());
    for (event_index, raw) in delivered.iter().enumerate() {
        let ev = raw.clone();
        if !synced {
            let mut retry = Vec::new();
            synced = matches!(
                sync.merge_snapshot(snapshot_for(stream, ev.u_last), &mut retry),
                MergeOutcome::Synced { .. }
            );
        }
        let (integrity, touch) = match sync.on_depth(ev.clone()) {
            DepthOutcome::Applied | DepthOutcome::Stale => {
                if sync.book().is_crossed() {
                    (
                        IntegrityState::Invalid {
                            reason: "book crossed after applied update".to_string(),
                        },
                        None,
                    )
                } else {
                    (IntegrityState::Valid, sync.book().touch())
                }
            }
            DepthOutcome::Buffered => (
                IntegrityState::Invalid {
                    reason: "event buffered while syncing".to_string(),
                },
                None,
            ),
            DepthOutcome::GapResync { reason } => {
                let mut retry = Vec::new();
                synced = matches!(
                    sync.merge_snapshot(snapshot_for(stream, ev.u_last), &mut retry),
                    MergeOutcome::Synced { .. }
                );
                (IntegrityState::Invalid { reason }, None)
            }
        };
        let signal = signal_for(sync.book(), &integrity);
        observations.push(Observation {
            event_index,
            event_ms: ev.event_ms,
            update_id: ev.u_last,
            integrity,
            touch,
            signal,
        });
    }
    observations
}

/// Outcome of one deterministic mutation trace.
#[derive(Debug)]
pub struct TraceResult {
    pub trace_id: usize,
    pub seed: u64,
    pub kind: FaultKind,
    pub event_index: usize,
    pub observations: Vec<Observation>,
    /// Signals emitted while the book was invalid; the acceptance bar
    /// is exactly zero.
    pub invalid_signal_count: usize,
    pub signals_emitted: usize,
    /// Observations that abstained because the book was invalid.
    pub abstentions: usize,
    /// Every traceable reason recorded during the trace.
    pub reasons: Vec<String>,
}

impl TraceResult {
    /// Digest of the valid observations' touch quotes. For the
    /// idempotent duplicate this equals the clean reference digest.
    pub fn valid_touch_digest(&self) -> String {
        let quotes: Vec<TouchQuote> = self
            .observations
            .iter()
            .filter(|o| o.integrity.is_valid())
            .filter_map(|o| o.touch.map(|t| (o.event_ms, t)))
            .collect();
        digest_touch_quotes(&quotes)
    }
}

/// Run one deterministic mutation trace: build the clean stream, inject
/// the fault for this seed and kind, replay, and record the fault in a
/// fresh ledger. The fault placement is drawn from the frozen seed, so
/// the same inputs always produce the same trace.
pub fn run_trace(
    trace_id: usize,
    kind: FaultKind,
    seed: u64,
    n_events: usize,
) -> (TraceResult, FaultLedger) {
    let stream = build_clean_stream(n_events, seed);
    let mut placement = SeededRng::new(seed ^ 0xf00d);
    let event_index = placement.below(n_events.saturating_sub(1).max(1));
    let spec = FaultSpec {
        seed,
        kind,
        event_index,
    };
    let delivered = inject_fault(&stream, spec);
    let observations = replay(&stream, &delivered);
    let invalid_signal_count = observations
        .iter()
        .filter(|o| !o.integrity.is_valid() && o.signal.is_some())
        .count();
    let signals_emitted = observations.iter().filter(|o| o.signal.is_some()).count();
    let abstentions = observations
        .iter()
        .filter(|o| !o.integrity.is_valid())
        .count();
    let reasons = if kind == FaultKind::Duplicate {
        let ev = &stream.events[event_index];
        vec![format!(
            "duplicate redelivery of update id range [{}, {}]",
            ev.u_first, ev.u_last
        )]
    } else {
        observations
            .iter()
            .filter_map(|o| o.integrity.reason().map(str::to_string))
            .collect()
    };
    let mut ledger = FaultLedger::new();
    ledger.record(LedgerEntry {
        trace_id,
        seed,
        kind,
        event_index,
        reason: reasons
            .first()
            .cloned()
            .unwrap_or_else(|| format!("no gap observed for {} fault", kind.name())),
        abstained: abstentions > 0,
    });
    (
        TraceResult {
            trace_id,
            seed,
            kind,
            event_index,
            observations,
            invalid_signal_count,
            signals_emitted,
            abstentions,
            reasons,
        },
        ledger,
    )
}

/// Run the full acceptance set: 20 deterministic mutation traces, five
/// per fault kind, with frozen seeds `BASE_SEED + trace_id`, plus the
/// combined fault and coverage ledger.
pub fn run_twenty_traces() -> (Vec<TraceResult>, FaultLedger) {
    let mut results = Vec::with_capacity(TRACE_COUNT);
    let mut ledger = FaultLedger::new();
    for trace_id in 0..TRACE_COUNT {
        let kind = FaultKind::all()[trace_id % FaultKind::all().len()];
        let seed = BASE_SEED + trace_id as u64;
        let (trace, trace_ledger) = run_trace(trace_id, kind, seed, N_EVENTS);
        for entry in trace_ledger.entries() {
            ledger.record(entry.clone());
        }
        results.push(trace);
    }
    (results, ledger)
}

/// The valid reference digest: replay the clean stream with no fault
/// and digest its touch quotes. Fault injection must leave this
/// unchanged.
pub fn clean_reference_digest(n_events: usize, seed: u64) -> String {
    let stream = build_clean_stream(n_events, seed);
    let observations = replay(&stream, &stream.events);
    let quotes: Vec<TouchQuote> = observations
        .iter()
        .filter(|o| o.integrity.is_valid())
        .filter_map(|o| o.touch.map(|t| (o.event_ms, t)))
        .collect();
    digest_touch_quotes(&quotes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n6_seeded_rng_is_deterministic() {
        let mut a = SeededRng::new(42);
        let mut b = SeededRng::new(42);
        let seq_a: Vec<u64> = (0..8).map(|_| a.next_u64()).collect();
        let seq_b: Vec<u64> = (0..8).map(|_| b.next_u64()).collect();
        assert_eq!(seq_a, seq_b);
        let mut c = SeededRng::new(43);
        assert_ne!(seq_a[0], c.next_u64());
    }

    #[test]
    fn n6_signal_for_abstains_when_invalid() {
        let mut book = OrderBook::new();
        book.from_snapshot(&[(100.0, 5.0)], &[(100.1, 4.0)], 1);
        assert!(signal_for(&book, &IntegrityState::Valid).is_some());
        assert_eq!(
            signal_for(
                &book,
                &IntegrityState::Invalid {
                    reason: "gap".to_string(),
                }
            ),
            None
        );
    }

    #[test]
    fn n6_ledger_digest_is_deterministic() {
        let (_, first) = run_twenty_traces();
        let (_, second) = run_twenty_traces();
        assert_eq!(first.digest(), second.digest());
        assert_eq!(first.total_faults(), 20);
        assert_eq!(first.canonical_text(), second.canonical_text());
    }

    #[test]
    fn n6_clean_stream_is_gapless() {
        let stream = build_clean_stream(N_EVENTS, BASE_SEED);
        let mut book = OrderBook::new();
        book.from_snapshot(&stream.snapshot.bids, &stream.snapshot.asks, SNAPSHOT_ID);
        for ev in &stream.events {
            assert_eq!(
                book.apply_diff(ev.u_first, ev.u_last, &ev.bids, &ev.asks),
                crate::orderbook::SyncResult::Applied
            );
        }
        assert!(!book.is_crossed());
    }
}
