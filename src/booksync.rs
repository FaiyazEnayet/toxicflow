//! Shared book-synchronization state machine for the live paths.
//!
//! Both the recorder (`recorder::Session`) and the terminal dashboard
//! (`toxicflow live`) must keep a local book in sync with the exchange
//! diff-depth stream, and both used to carry private copies of the
//! procedure with the same failure shape: state changed before
//! validation finished, and malformed input was silently read as zero.
//!
//! This module is the single implementation:
//!
//! - Wire values enter as typed [`DepthEvent`] / [`SnapshotData`] values, validated at parse time (finite prices > 0, quantities >= 0, update ids in range). Invalid input is rejected with a diagnostic; it is never applied as zeros.
//! - Snapshot merges are transactional. The snapshot and the buffered diffs are applied to a candidate book with a candidate quote buffer; both commit only after every buffered event applies cleanly and the result is not crossed. A failed merge leaves the committed book and the emitted quotes untouched and keeps the buffered events for the next attempt.
//! - A diff that would cross the synced book is treated as a gap: nothing commits, the machine drops back to syncing, and the event is kept as the head of the resync buffer.
//! - The buffer is bounded, and every gap or overflow is recorded as an explicit diagnostic instead of failing silently.

use crate::orderbook::{OrderBook, SyncResult};

/// Maximum depth events held while a snapshot merge is outstanding.
/// At the 100 ms stream cadence this covers several minutes of
/// buffering; beyond it the oldest events are dropped (they are stale
/// relative to any snapshot fetched afterwards anyway) and the drop is
/// recorded as a diagnostic.
pub const MAX_BUFFERED_EVENTS: usize = 4096;

/// A validated depthUpdate event.
#[derive(Debug, Clone, PartialEq)]
pub struct DepthEvent {
    pub event_ms: i64,
    pub u_first: i64,
    pub u_last: i64,
    pub bids: Vec<(f64, f64)>,
    pub asks: Vec<(f64, f64)>,
}

/// A validated REST depth snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct SnapshotData {
    pub last_update_id: i64,
    pub bids: Vec<(f64, f64)>,
    pub asks: Vec<(f64, f64)>,
}

/// Parse one price/quantity string pair under the wire rules: both
/// must parse as finite numbers, the price strictly positive, the
/// quantity non-negative (zero removes a level, so it is legitimate).
pub fn parse_level_pair(price: &str, qty: &str) -> Result<(f64, f64), String> {
    let p: f64 = price
        .parse()
        .map_err(|_| format!("invalid price {price:?}"))?;
    let q: f64 = qty
        .parse()
        .map_err(|_| format!("invalid quantity {qty:?}"))?;
    if !p.is_finite() || p <= 0.0 {
        return Err(format!("price {price:?} is not finite and positive"));
    }
    if !q.is_finite() || q < 0.0 {
        return Err(format!("quantity {qty:?} is not finite and non-negative"));
    }
    Ok((p, q))
}

/// Parse the `b`/`a` level arrays of a depth message. Every entry must
/// be a [price, quantity] pair of strings; anything else is an error,
/// not a zero.
pub fn parse_levels_json(v: &serde_json::Value) -> Result<Vec<(f64, f64)>, String> {
    let arr = v
        .as_array()
        .ok_or_else(|| "levels are not an array".to_string())?;
    let mut out = Vec::with_capacity(arr.len());
    for x in arr {
        let p = x[0]
            .as_str()
            .ok_or_else(|| "level price is not a string".to_string())?;
        let q = x[1]
            .as_str()
            .ok_or_else(|| "level quantity is not a string".to_string())?;
        out.push(parse_level_pair(p, q)?);
    }
    Ok(out)
}

fn checked_id(v: &serde_json::Value, field: &str) -> Result<i64, String> {
    let id = v[field]
        .as_i64()
        .ok_or_else(|| format!("missing or non-integer {field}"))?;
    if id < 0 {
        return Err(format!("{field} = {id} is negative"));
    }
    Ok(id)
}

impl DepthEvent {
    /// Validate a raw depthUpdate payload into a typed event.
    pub fn from_json(v: &serde_json::Value) -> Result<Self, String> {
        let u_first = checked_id(v, "U")?;
        let u_last = checked_id(v, "u")?;
        if u_first < 1 || u_last < u_first {
            return Err(format!(
                "update id range [{u_first}, {u_last}] is out of order"
            ));
        }
        let event_ms = v["E"].as_i64().unwrap_or(0);
        if event_ms < 0 {
            return Err(format!("event time {event_ms} is negative"));
        }
        Ok(Self {
            event_ms,
            u_first,
            u_last,
            bids: parse_levels_json(&v["b"])?,
            asks: parse_levels_json(&v["a"])?,
        })
    }
}

impl SnapshotData {
    /// Validate raw snapshot levels (string pairs, as the REST API
    /// returns them) into a typed snapshot.
    pub fn from_pairs(
        last_update_id: i64,
        bids: &[(String, String)],
        asks: &[(String, String)],
    ) -> Result<Self, String> {
        if last_update_id < 0 {
            return Err(format!("lastUpdateId {last_update_id} is negative"));
        }
        let conv = |lv: &[(String, String)]| -> Result<Vec<(f64, f64)>, String> {
            lv.iter().map(|(p, q)| parse_level_pair(p, q)).collect()
        };
        Ok(Self {
            last_update_id,
            bids: conv(bids)?,
            asks: conv(asks)?,
        })
    }

    /// Validate a raw REST depth response body into a typed snapshot.
    pub fn from_json(v: &serde_json::Value) -> Result<Self, String> {
        let last_update_id = checked_id(v, "lastUpdateId")?;
        Ok(Self {
            last_update_id,
            bids: parse_levels_json(&v["bids"])?,
            asks: parse_levels_json(&v["asks"])?,
        })
    }
}

/// One committed quote observation: (event time, (bid px, bid sz,
/// ask px, ask sz)).
pub type TouchQuote = (i64, (f64, f64, f64, f64));

/// What one depth event did to the state machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DepthOutcome {
    /// Applied to the synced book.
    Applied,
    /// Older than the book; ignored.
    Stale,
    /// Held in the buffer while (re)syncing.
    Buffered,
    /// The event does not continue the committed book (sequence gap,
    /// an already-consumed id range carrying different content, or a
    /// diff that would cross the book). Nothing was committed; the
    /// machine is syncing again with this event as the buffer head,
    /// and `reason` says which case fired.
    GapResync { reason: String },
}

/// What a snapshot merge attempt did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MergeOutcome {
    /// Candidate book and quotes committed; `applied` counts the
    /// buffered events replayed on top of the snapshot.
    Synced { applied: usize },
    /// Nothing committed: the snapshot cannot continue the buffered
    /// stream, or the merged result would be crossed. The caller
    /// should fetch a fresher snapshot; the buffer is retained.
    Refetch,
}

/// The shared sync state machine: one committed book, one bounded
/// buffer of events awaiting a snapshot, and gap diagnostics.
pub struct BookSync {
    book: OrderBook,
    synced: bool,
    buffer: Vec<DepthEvent>,
    /// Content of the most recently applied event, used to tell an
    /// idempotent redelivery apart from a conflicting reuse of an
    /// already-consumed id range.
    last_applied: Option<DepthEvent>,
    gaps: u32,
    last_gap: Option<String>,
}

impl Default for BookSync {
    fn default() -> Self {
        Self::new()
    }
}

impl BookSync {
    pub fn new() -> Self {
        Self {
            book: OrderBook::new(),
            synced: false,
            buffer: Vec::new(),
            last_applied: None,
            gaps: 0,
            last_gap: None,
        }
    }

    pub fn is_synced(&self) -> bool {
        self.synced
    }

    pub fn book(&self) -> &OrderBook {
        &self.book
    }

    pub fn buffered_len(&self) -> usize {
        self.buffer.len()
    }

    /// (number of gap/resync events, most recent diagnostic).
    pub fn gap_diagnostics(&self) -> (u32, Option<&str>) {
        (self.gaps, self.last_gap.as_deref())
    }

    /// Feed one validated depth event.
    pub fn on_depth(&mut self, ev: DepthEvent) -> DepthOutcome {
        if !self.synced {
            self.push_buffer(ev);
            return DepthOutcome::Buffered;
        }
        // A redelivery of the last applied event is idempotent; the
        // same id range carrying different content conflicts with the
        // committed history and can only be resolved by a resync.
        if let Some(last) = &self.last_applied {
            if ev.u_first == last.u_first && ev.u_last == last.u_last {
                if ev.bids == last.bids && ev.asks == last.asks {
                    return DepthOutcome::Applied;
                }
                return self.gap(ev, "conflicting content for an applied update id range");
            }
        }
        // Apply to a candidate first: a diff that crosses the book
        // must not commit.
        let mut cand = self.book.clone();
        match cand.apply_diff(ev.u_first, ev.u_last, &ev.bids, &ev.asks) {
            SyncResult::Applied => {
                if cand.is_crossed() {
                    return self.gap(ev, "diff would cross the book");
                }
                self.book = cand;
                self.last_applied = Some(ev);
                DepthOutcome::Applied
            }
            SyncResult::Stale => DepthOutcome::Stale,
            SyncResult::Gap => {
                self.gap(ev, "sequence gap between the book and the event update ids")
            }
        }
    }

    /// Attempt the transactional snapshot merge. On `Synced`, returns
    /// the quotes produced by replaying the buffered events (one per
    /// event whose post-state has a touch) alongside via `quotes_out`;
    /// on `Refetch` nothing is committed and `quotes_out` is cleared.
    pub fn merge_snapshot(
        &mut self,
        snap: &SnapshotData,
        quotes_out: &mut Vec<TouchQuote>,
    ) -> MergeOutcome {
        quotes_out.clear();
        // The first buffered event not fully covered by the snapshot
        // must continue it: U <= lastUpdateId + 1.
        let first_live = self.buffer.iter().find(|e| e.u_last > snap.last_update_id);
        if let Some(f) = first_live {
            if snap.last_update_id + 1 < f.u_first {
                return MergeOutcome::Refetch; // snapshot too old
            }
        }
        let mut cand = OrderBook::new();
        cand.from_snapshot(&snap.bids, &snap.asks, snap.last_update_id);
        if cand.is_crossed() {
            return MergeOutcome::Refetch; // a crossed snapshot never syncs
        }
        let mut cand_quotes: Vec<TouchQuote> = Vec::new();
        let mut applied = 0usize;
        for e in &self.buffer {
            if e.u_last <= snap.last_update_id {
                continue; // stale: fully covered by the snapshot
            }
            match cand.apply_diff(e.u_first, e.u_last, &e.bids, &e.asks) {
                SyncResult::Applied => {
                    applied += 1;
                    if let Some(t) = cand.touch() {
                        cand_quotes.push((e.event_ms, t));
                    }
                }
                SyncResult::Stale => {}
                SyncResult::Gap => return MergeOutcome::Refetch,
            }
        }
        if cand.is_crossed() {
            return MergeOutcome::Refetch; // merged state would be crossed
        }
        // Commit: book, quotes, and sync state change together.
        self.last_applied = self
            .buffer
            .iter()
            .rev()
            .find(|e| e.u_last > snap.last_update_id)
            .cloned();
        self.book = cand;
        self.buffer.clear();
        self.synced = true;
        *quotes_out = cand_quotes;
        MergeOutcome::Synced { applied }
    }

    fn gap(&mut self, ev: DepthEvent, reason: &str) -> DepthOutcome {
        self.gaps += 1;
        self.last_gap = Some(format!(
            "{reason} (event U={}, u={}, book last_id={})",
            ev.u_first, ev.u_last, self.book.last_id
        ));
        self.synced = false;
        self.buffer.clear();
        self.buffer.push(ev);
        DepthOutcome::GapResync {
            reason: reason.to_string(),
        }
    }

    fn push_buffer(&mut self, ev: DepthEvent) {
        if self.buffer.len() >= MAX_BUFFERED_EVENTS {
            // Bounded buffer: drop the oldest events, which any
            // snapshot fetched now would supersede anyway, and say so.
            let drop = self.buffer.len() + 1 - MAX_BUFFERED_EVENTS;
            self.buffer.drain(..drop);
            self.gaps += 1;
            self.last_gap = Some(format!(
                "event buffer overflow: dropped {drop} oldest buffered events"
            ));
        }
        self.buffer.push(ev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap() -> SnapshotData {
        SnapshotData::from_pairs(
            100,
            &[
                ("100.0".to_string(), "5.0".to_string()),
                ("99.9".to_string(), "3.0".to_string()),
            ],
            &[
                ("100.1".to_string(), "4.0".to_string()),
                ("100.2".to_string(), "2.0".to_string()),
            ],
        )
        .unwrap()
    }

    fn ev(u_first: i64, u_last: i64, bids: &[(f64, f64)], asks: &[(f64, f64)]) -> DepthEvent {
        DepthEvent {
            event_ms: 1_700_000_000_000 + u_first,
            u_first,
            u_last,
            bids: bids.to_vec(),
            asks: asks.to_vec(),
        }
    }

    #[test]
    fn merge_commits_quotes_with_the_book() {
        let mut s = BookSync::new();
        s.on_depth(ev(101, 102, &[(100.0, 6.0)], &[]));
        s.on_depth(ev(103, 103, &[(99.9, 0.0)], &[(100.2, 9.0)]));
        let mut q = Vec::new();
        assert_eq!(
            s.merge_snapshot(&snap(), &mut q),
            MergeOutcome::Synced { applied: 2 }
        );
        assert_eq!(q.len(), 2);
        assert_eq!(s.book().best_bid(), Some((100.0, 6.0)));
        assert!(s.is_synced());
    }

    #[test]
    fn failed_merge_commits_nothing() {
        let mut s = BookSync::new();
        s.on_depth(ev(101, 102, &[(100.0, 6.0)], &[]));
        // update 103 never arrived; 104 cannot apply
        s.on_depth(ev(104, 104, &[(100.0, 8.0)], &[]));
        let mut q = Vec::new();
        assert_eq!(s.merge_snapshot(&snap(), &mut q), MergeOutcome::Refetch);
        assert!(q.is_empty());
        assert!(s.book().is_empty());
        assert_eq!(s.book().last_id, -1);
        assert_eq!(s.buffered_len(), 2);
        assert!(!s.is_synced());
    }
}
