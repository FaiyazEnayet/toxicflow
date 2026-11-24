//! Typed L1/L2 as-of event-sourced microstructure replay engine.
//!
//! The journal (`storage::JournalRecord`) is the raw event record of a
//! session. This module replays it as a typed engine instead of a
//! one-shot rebuild:
//!
//! - Every observation carries two times: the exchange-stamped event
//!   time ([`EventTime::exchange_ms`], the `E`/`T` field on the wire,
//!   absent for REST snapshots) and the receive time, which the journal
//!   preserves only as receipt order, so the receipt identity is the
//!   journal sequence ([`ReceiptId`]).
//! - Every observation carries its exchange sequence identity
//!   ([`SequenceId`]): a depth update covers `[u_first, u_last]`, a
//!   snapshot carries its `last_update_id`.
//! - The reducer has typed validity ([`Validity`]): it is `Valid` only
//!   while the book is synced and uncrossed. While it is invalid it
//!   emits no features at all, and every stretch of invalidity is
//!   recorded as a [`ResyncInterval`] from the observation that lost
//!   sync until the snapshot that restored it.
//! - Every derived feature carries its lineage ([`Lineage`]): the
//!   observation IDs the pre-event book state was built from. By
//!   construction every cited ID is strictly earlier than the
//!   feature's own as-of receipt ID; the triggering event is named by
//!   the feature's as-of ID itself, never repeated in the lineage.
//! - L2 unavailability is logged explicitly, with the observation ID,
//!   in the outcome log: no depth level beyond the touch is ever
//!   trusted while the reducer is invalid.
//!
//! The reference reducer is [`crate::storage::reconstruct_session`]:
//! on any journal the reference accepts, this engine reproduces its
//! trades and quotes exactly. The difference is what happens on a
//! journal the reference rejects: instead of failing the whole
//! replay, the engine marks the stretch invalid, logs it, records
//! the resync interval, and continues from the next snapshot, so the
//! valid stretches still yield their features with full lineage.

use crate::metrics::{Quote, Trade};
use crate::orderbook::{OrderBook, SyncResult};
use crate::storage::JournalRecord;

/// Top-of-book depth levels the L2 imbalance feature covers.
pub const IMBALANCE_LEVELS: usize = 5;

/// Receipt identity of one journal observation: its sequence number.
/// Receipt order is sequence order; the journal keeps no wall-clock
/// receipt timestamp, so the sequence is the receive time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReceiptId(pub u64);

/// One observation time: the exchange-stamped event time when the
/// exchange stamped one, plus the receipt identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EventTime {
    /// Exchange event time in milliseconds (`E` for depth, `T` for
    /// trades). `None` for snapshots, which are REST-fetched at
    /// receipt and carry no exchange event time.
    pub exchange_ms: Option<i64>,
    /// Receipt identity: the journal sequence number.
    pub receipt: ReceiptId,
}

impl EventTime {
    pub fn observed(exchange_ms: Option<i64>, receipt: ReceiptId) -> Self {
        Self {
            exchange_ms,
            receipt,
        }
    }
}

/// Exchange sequence identity of one observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceId {
    /// First update id covered (`U` for depth, `last_update_id` for a
    /// snapshot).
    pub first: i64,
    /// Last update id covered (`u` for depth, `last_update_id` for a
    /// snapshot).
    pub last: i64,
}

impl SequenceId {
    pub fn for_depth(u_first: i64, u_last: i64) -> Self {
        Self {
            first: u_first,
            last: u_last,
        }
    }

    pub fn for_snapshot(last_update_id: i64) -> Self {
        Self {
            first: last_update_id,
            last: last_update_id,
        }
    }
}

/// Why the reducer is not valid. Typed so callers match on the cause
/// instead of parsing a message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidReason {
    /// No snapshot has been applied yet.
    AwaitingSnapshot,
    /// A depth update arrived before the first snapshot.
    DepthBeforeSnapshot,
    /// A depth update skips past the book.
    SequenceGap {
        expected_first: i64,
        found_first: i64,
    },
    /// A depth update is older than the book.
    StaleUpdate { u_last: i64, book_last_id: i64 },
    /// The book would be crossed.
    CrossedBook,
}

impl InvalidReason {
    pub fn describe(&self) -> String {
        match self {
            InvalidReason::AwaitingSnapshot => "awaiting first snapshot".to_string(),
            InvalidReason::DepthBeforeSnapshot => "depth update before first snapshot".to_string(),
            InvalidReason::SequenceGap {
                expected_first,
                found_first,
            } => {
                format!("sequence gap: expected first update {expected_first}, found {found_first}")
            }
            InvalidReason::StaleUpdate {
                u_last,
                book_last_id,
            } => format!("stale update: u_last {u_last} is older than book last_id {book_last_id}"),
            InvalidReason::CrossedBook => "book is crossed".to_string(),
        }
    }
}

/// Typed validity of the reducer. Features are emitted only while
/// `Valid`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Validity {
    Valid,
    Invalid(InvalidReason),
}

/// Lineage of one derived feature: the observation IDs the
/// pre-event book state was built from. Every ID here is strictly
/// earlier than the feature's own as-of receipt ID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lineage {
    pub sources: Vec<u64>,
}

/// One L1 touch feature at an as-of point: best bid/ask, mid, spread.
#[derive(Debug, Clone, PartialEq)]
pub struct TouchFeature {
    pub as_of: ReceiptId,
    pub exchange_ms: i64,
    pub sequence: SequenceId,
    pub bid_px: f64,
    pub bid_sz: f64,
    pub ask_px: f64,
    pub ask_sz: f64,
    pub mid: f64,
    pub spread: f64,
    pub lineage: Lineage,
}

impl TouchFeature {
    pub fn event_time(&self) -> EventTime {
        EventTime::observed(Some(self.exchange_ms), self.as_of)
    }
}

/// One L2 top-of-book imbalance feature at an as-of point.
#[derive(Debug, Clone, PartialEq)]
pub struct ImbalanceFeature {
    pub as_of: ReceiptId,
    pub exchange_ms: i64,
    pub sequence: SequenceId,
    pub levels: usize,
    pub imbalance: f64,
    pub lineage: Lineage,
}

impl ImbalanceFeature {
    pub fn event_time(&self) -> EventTime {
        EventTime::observed(Some(self.exchange_ms), self.as_of)
    }
}

/// One interval where the reducer was not valid: from the observation
/// that lost sync until the snapshot that restored it (`None` when
/// the journal ends still invalid).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResyncInterval {
    pub start_seq: u64,
    pub end_seq: Option<u64>,
    pub reason: InvalidReason,
}

/// Outcome of replaying a whole journal.
#[derive(Debug)]
pub struct ReplayOutcome {
    /// Deterministic digest of the reduced state. Replaying the same
    /// records in another process reproduces it exactly.
    pub digest: String,
    /// Trades, in journal order. Trades are observations, not derived
    /// signals, so they are kept even while the reducer is invalid.
    pub trades: Vec<Trade>,
    /// Touch quotes, one per applied depth update with a touch.
    pub quotes: Vec<Quote>,
    /// L1 touch features, one per applied depth update with a touch.
    pub touches: Vec<TouchFeature>,
    /// L2 imbalance features, one per applied depth update.
    pub imbalances: Vec<ImbalanceFeature>,
    /// Every invalid stretch, in order.
    pub resyncs: Vec<ResyncInterval>,
    /// Explicit diagnostics, including every L2 unavailability with
    /// its observation ID.
    pub log: Vec<String>,
    /// Validity at the end of the journal.
    pub validity: Validity,
    /// Features that would have been emitted while invalid. Must stay
    /// zero: the emitter counts instead of emitting.
    pub features_while_invalid: usize,
    /// Journal records folded.
    pub records_applied: usize,
}

/// As-of event reducer over journal records.
pub struct ReplayEngine {
    book: OrderBook,
    synced: bool,
    after_snapshot: bool,
    validity: Validity,
    /// Snapshot seq plus applied depth seqs since the snapshot, in
    /// order. The next feature's lineage is this list as it stands
    /// before its triggering event, so every cited ID is earlier.
    lineage_sources: Vec<u64>,
    open_resync: Option<(u64, InvalidReason)>,
    touches: Vec<TouchFeature>,
    imbalances: Vec<ImbalanceFeature>,
    trades: Vec<Trade>,
    quotes: Vec<Quote>,
    resyncs: Vec<ResyncInterval>,
    log: Vec<String>,
    features_while_invalid: usize,
    records_applied: usize,
}

impl Default for ReplayEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ReplayEngine {
    pub fn new() -> Self {
        Self {
            book: OrderBook::new(),
            synced: false,
            after_snapshot: false,
            validity: Validity::Invalid(InvalidReason::AwaitingSnapshot),
            lineage_sources: Vec::new(),
            open_resync: None,
            touches: Vec::new(),
            imbalances: Vec::new(),
            trades: Vec::new(),
            quotes: Vec::new(),
            resyncs: Vec::new(),
            log: Vec::new(),
            features_while_invalid: 0,
            records_applied: 0,
        }
    }

    pub fn validity(&self) -> &Validity {
        &self.validity
    }

    /// Latest L1 touch feature at or before `seq`, if any.
    pub fn touch_as_of(&self, seq: u64) -> Option<&TouchFeature> {
        let idx = self.touches.partition_point(|f| f.as_of.0 <= seq);
        if idx == 0 {
            None
        } else {
            Some(&self.touches[idx - 1])
        }
    }

    /// Latest L2 imbalance feature at or before `seq`, if any. An L2
    /// query over an invalid stretch answers with the last valid
    /// feature, never with a value built from an untrusted book.
    pub fn imbalance_as_of(&self, seq: u64) -> Option<&ImbalanceFeature> {
        let idx = self.imbalances.partition_point(|f| f.as_of.0 <= seq);
        if idx == 0 {
            None
        } else {
            Some(&self.imbalances[idx - 1])
        }
    }

    fn open_resync(&mut self, seq: u64, reason: InvalidReason) {
        if self.open_resync.is_none() {
            self.log
                .push(format!("L2 unavailable: {} (seq {seq})", reason.describe()));
            self.open_resync = Some((seq, reason));
        }
    }

    fn close_resync(&mut self, seq: u64) {
        if let Some((start, reason)) = self.open_resync.take() {
            self.resyncs.push(ResyncInterval {
                start_seq: start,
                end_seq: Some(seq),
                reason,
            });
        }
    }

    fn invalidate(&mut self, seq: u64, reason: InvalidReason) {
        self.validity = Validity::Invalid(reason.clone());
        self.synced = false;
        self.after_snapshot = false;
        self.open_resync(seq, reason);
    }

    /// Fold one journal record, in sequence order.
    pub fn apply(&mut self, seq: u64, record: &JournalRecord) {
        self.records_applied += 1;
        match record {
            JournalRecord::SessionStart { .. } => {
                self.book = OrderBook::new();
                self.synced = false;
                self.after_snapshot = false;
                self.lineage_sources.clear();
                self.validity = Validity::Invalid(InvalidReason::AwaitingSnapshot);
                self.open_resync(seq, InvalidReason::AwaitingSnapshot);
            }
            JournalRecord::Snapshot {
                last_update_id,
                bids,
                asks,
            } => {
                let mut cand = OrderBook::new();
                cand.from_snapshot(bids, asks, *last_update_id);
                if cand.is_crossed() {
                    self.invalidate(seq, InvalidReason::CrossedBook);
                    return;
                }
                self.book = cand;
                self.synced = true;
                self.after_snapshot = true;
                self.validity = Validity::Valid;
                self.lineage_sources = vec![seq];
                self.close_resync(seq);
            }
            JournalRecord::Depth {
                time_ms,
                u_first,
                u_last,
                bids,
                asks,
            } => {
                if !self.synced {
                    self.log.push(format!(
                        "L2 unavailable: depth update before first snapshot (seq {seq})"
                    ));
                    // The first cause of invalidity stands until a
                    // snapshot restores validity: a depth arriving
                    // after a crossed snapshot must not relabel the
                    // crossed book as merely awaiting a snapshot.
                    self.open_resync(seq, InvalidReason::DepthBeforeSnapshot);
                    return;
                }
                // Apply to a candidate first: a diff that would cross
                // the book must not commit.
                let mut cand = self.book.clone();
                let mut outcome = cand.apply_diff(*u_first, *u_last, bids, asks);
                if outcome == SyncResult::Gap && self.after_snapshot {
                    cand = self.book.clone();
                    outcome = cand.apply_diff_baseline(*u_first, *u_last, bids, asks);
                }
                self.after_snapshot = false;
                match outcome {
                    SyncResult::Applied => {
                        if cand.is_crossed() {
                            self.invalidate(seq, InvalidReason::CrossedBook);
                            return;
                        }
                        self.book = cand;
                        self.emit_touch(seq, *time_ms, *u_first, *u_last);
                    }
                    SyncResult::Stale => {
                        self.invalidate(
                            seq,
                            InvalidReason::StaleUpdate {
                                u_last: *u_last,
                                book_last_id: self.book.last_id,
                            },
                        );
                    }
                    SyncResult::Gap => {
                        self.invalidate(
                            seq,
                            InvalidReason::SequenceGap {
                                expected_first: self.book.last_id + 1,
                                found_first: *u_first,
                            },
                        );
                    }
                }
            }
            JournalRecord::Trade {
                time_ms,
                price,
                qty,
                is_buyer_maker,
            } => {
                self.trades.push(Trade {
                    time_ms: *time_ms,
                    price: *price,
                    qty: *qty,
                    is_buyer_maker: *is_buyer_maker,
                });
            }
            JournalRecord::Overflow { dropped } => {
                self.log.push(format!(
                    "journal overflow: {dropped} records dropped before seq {seq}; lineage after this point is incomplete"
                ));
            }
            JournalRecord::SessionEnd { .. } => {}
        }
    }

    /// Emit the L1/L2 features for an applied depth update. The guard
    /// is the point of the engine: while the reducer is invalid,
    /// nothing is emitted and the attempt is counted.
    fn emit_touch(&mut self, seq: u64, time_ms: i64, u_first: i64, u_last: i64) {
        if !matches!(self.validity, Validity::Valid) {
            self.features_while_invalid += 1;
            return;
        }
        let Some((bp, bq, ap, aq)) = self.book.touch() else {
            return;
        };
        let lineage = Lineage {
            sources: self.lineage_sources.clone(),
        };
        let sequence = SequenceId::for_depth(u_first, u_last);
        let mid = (bp + ap) / 2.0;
        self.touches.push(TouchFeature {
            as_of: ReceiptId(seq),
            exchange_ms: time_ms,
            sequence,
            bid_px: bp,
            bid_sz: bq,
            ask_px: ap,
            ask_sz: aq,
            mid,
            spread: ap - bp,
            lineage: lineage.clone(),
        });
        self.quotes.push(Quote {
            time_ms,
            bid_px: bp,
            bid_sz: bq,
            ask_px: ap,
            ask_sz: aq,
        });
        if let Some(imb) = self.book.imbalance(IMBALANCE_LEVELS) {
            self.imbalances.push(ImbalanceFeature {
                as_of: ReceiptId(seq),
                exchange_ms: time_ms,
                sequence,
                levels: IMBALANCE_LEVELS,
                imbalance: imb,
                lineage,
            });
        }
        self.lineage_sources.push(seq);
    }

    /// Finish the replay: close out the digest and hand back the
    /// outcome. An interval still open at the end stays open
    /// (`end_seq` is `None`).
    pub fn finish(mut self) -> ReplayOutcome {
        if let Some((start, reason)) = self.open_resync.take() {
            self.resyncs.push(ResyncInterval {
                start_seq: start,
                end_seq: None,
                reason,
            });
        }
        let mut outcome = ReplayOutcome {
            digest: String::new(),
            trades: std::mem::take(&mut self.trades),
            quotes: std::mem::take(&mut self.quotes),
            touches: std::mem::take(&mut self.touches),
            imbalances: std::mem::take(&mut self.imbalances),
            resyncs: std::mem::take(&mut self.resyncs),
            log: std::mem::take(&mut self.log),
            validity: self.validity.clone(),
            features_while_invalid: self.features_while_invalid,
            records_applied: self.records_applied,
        };
        outcome.digest = replay_digest(&outcome);
        outcome
    }
}

/// Replay a whole journal in sequence order.
pub fn replay(records: &[JournalRecord]) -> ReplayOutcome {
    let mut engine = ReplayEngine::new();
    for (seq, record) in records.iter().enumerate() {
        engine.apply(seq as u64, record);
    }
    engine.finish()
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

fn bits(v: f64) -> String {
    format!("{:016x}", v.to_bits())
}

/// Deterministic digest of the reduced state. The fold covers every
/// emitted feature (with bit-exact floats and lineage), every trade
/// and quote, and every resync interval, in order. Nothing in it
/// depends on wall-clock time, hash-map order, or process state, so
/// two clean processes replaying the same journal agree exactly.
pub fn replay_digest(outcome: &ReplayOutcome) -> String {
    let mut h = FNV_OFFSET;
    for f in &outcome.touches {
        h = fnv1a(
            h,
            format!(
                "touch {} {} {} {} {} {} {} {} {:?}\n",
                f.as_of.0,
                f.exchange_ms,
                bits(f.bid_px),
                bits(f.bid_sz),
                bits(f.ask_px),
                bits(f.ask_sz),
                bits(f.mid),
                bits(f.spread),
                f.lineage.sources,
            )
            .as_bytes(),
        );
    }
    for f in &outcome.imbalances {
        h = fnv1a(
            h,
            format!(
                "imbalance {} {} {} {} {:?}\n",
                f.as_of.0,
                f.exchange_ms,
                f.levels,
                bits(f.imbalance),
                f.lineage.sources,
            )
            .as_bytes(),
        );
    }
    for t in &outcome.trades {
        h = fnv1a(
            h,
            format!(
                "trade {} {} {} {}\n",
                t.time_ms,
                bits(t.price),
                bits(t.qty),
                t.is_buyer_maker
            )
            .as_bytes(),
        );
    }
    for q in &outcome.quotes {
        h = fnv1a(
            h,
            format!(
                "quote {} {} {} {} {}\n",
                q.time_ms,
                bits(q.bid_px),
                bits(q.bid_sz),
                bits(q.ask_px),
                bits(q.ask_sz),
            )
            .as_bytes(),
        );
    }
    for r in &outcome.resyncs {
        h = fnv1a(
            h,
            format!("resync {} {:?} {:?}\n", r.start_seq, r.end_seq, r.reason,).as_bytes(),
        );
    }
    format!("{h:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::JournalRecord;

    fn snap(id: i64) -> JournalRecord {
        JournalRecord::Snapshot {
            last_update_id: id,
            bids: vec![(100.0, 5.0), (99.9, 3.0)],
            asks: vec![(100.1, 4.0), (100.2, 2.0)],
        }
    }

    fn depth(time_ms: i64, u_first: i64, u_last: i64) -> JournalRecord {
        JournalRecord::Depth {
            time_ms,
            u_first,
            u_last,
            bids: vec![(100.0, 6.0)],
            asks: vec![],
        }
    }

    #[test]
    fn a1_validity_transitions_on_snapshot_and_gap() {
        let mut e = ReplayEngine::new();
        assert!(matches!(
            e.validity(),
            Validity::Invalid(InvalidReason::AwaitingSnapshot)
        ));
        e.apply(
            0,
            &JournalRecord::SessionStart {
                symbol: "BTCUSDT".to_string(),
            },
        );
        assert!(matches!(e.validity(), Validity::Invalid(_)));
        e.apply(1, &snap(100));
        assert!(matches!(e.validity(), Validity::Valid));
        e.apply(2, &depth(1_700_000_000_000, 101, 101));
        assert!(matches!(e.validity(), Validity::Valid));
        // update 102 never arrives: 103 is a gap.
        e.apply(3, &depth(1_700_000_000_100, 103, 103));
        assert!(matches!(
            e.validity(),
            Validity::Invalid(InvalidReason::SequenceGap { .. })
        ));
        e.apply(4, &snap(102));
        assert!(matches!(e.validity(), Validity::Valid));
    }

    #[test]
    fn a1_stale_update_invalidates_and_logs() {
        let mut e = ReplayEngine::new();
        e.apply(
            0,
            &JournalRecord::SessionStart {
                symbol: "BTCUSDT".to_string(),
            },
        );
        e.apply(1, &snap(100));
        e.apply(2, &depth(1_700_000_000_000, 101, 101));
        e.apply(3, &depth(1_700_000_000_100, 90, 95));
        assert!(matches!(
            e.validity(),
            Validity::Invalid(InvalidReason::StaleUpdate { .. })
        ));
        let out = e.finish();
        assert_eq!(out.features_while_invalid, 0);
        assert!(out.log.iter().any(|l| l.contains("L2 unavailable")));
        assert!(out.log.iter().any(|l| l.contains("seq 3")));
    }

    #[test]
    fn a1_baseline_after_snapshot_matches_reference() {
        // The first depth after a snapshot may start past the snapshot
        // id: it is the new baseline, exactly as the reference reducer
        // treats it. Both must agree on trades and quotes.
        let records = vec![
            JournalRecord::SessionStart {
                symbol: "BTCUSDT".to_string(),
            },
            snap(100),
            depth(1_700_000_000_000, 115, 116),
            JournalRecord::Trade {
                time_ms: 1_700_000_000_050,
                price: 100.05,
                qty: 0.5,
                is_buyer_maker: true,
            },
            JournalRecord::SessionEnd { records: 4 },
        ];
        let out = replay(&records);
        let (ref_trades, ref_quotes) = crate::storage::reconstruct_session(&records).unwrap();
        assert_eq!(out.trades, ref_trades);
        assert_eq!(out.quotes, ref_quotes);
        assert_eq!(out.touches.len(), 1);
        assert_eq!(out.touches[0].lineage.sources, vec![1]);
    }

    #[test]
    fn a1_digest_stable_across_replays_of_same_records() {
        let records = vec![
            JournalRecord::SessionStart {
                symbol: "BTCUSDT".to_string(),
            },
            snap(100),
            depth(1_700_000_000_000, 101, 101),
            depth(1_700_000_000_100, 102, 102),
            JournalRecord::SessionEnd { records: 4 },
        ];
        assert_eq!(replay(&records).digest, replay(&records).digest);
        // One changed quantity changes the digest.
        let mut changed = records.clone();
        if let JournalRecord::Depth { bids, .. } = &mut changed[2] {
            bids[0].1 = 6.5;
        }
        assert_ne!(replay(&records).digest, replay(&changed).digest);
    }

    #[test]
    fn a1_overflow_is_logged_not_silent() {
        let records = vec![
            JournalRecord::SessionStart {
                symbol: "BTCUSDT".to_string(),
            },
            snap(100),
            JournalRecord::Overflow { dropped: 3 },
            depth(1_700_000_000_000, 101, 101),
            JournalRecord::SessionEnd { records: 4 },
        ];
        let out = replay(&records);
        assert!(out.log.iter().any(|l| l.contains("overflow")));
        assert!(out.log.iter().any(|l| l.contains("seq 2")));
        assert_eq!(out.touches.len(), 1);
        assert!(matches!(out.validity, Validity::Valid));
    }

    #[test]
    fn a1_event_time_and_sequence_ids_are_typed() {
        let records = vec![
            JournalRecord::SessionStart {
                symbol: "BTCUSDT".to_string(),
            },
            snap(100),
            depth(1_700_000_000_000, 101, 102),
            JournalRecord::SessionEnd { records: 3 },
        ];
        let out = replay(&records);
        let f = &out.touches[0];
        assert_eq!(f.sequence, SequenceId::for_depth(101, 102));
        assert_eq!(
            f.event_time(),
            EventTime::observed(Some(1_700_000_000_000), ReceiptId(2))
        );
        assert_eq!(f.as_of, ReceiptId(2));
    }
}
