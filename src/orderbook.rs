//! Local limit order book with Binance diff-depth sync.
//!
//! Follows the exchange's documented procedure: buffer diff events, take a
//! REST snapshot, discard stale diffs, verify the [U, u] continuity, then
//! apply. A gap (U > last_id + 1) forces a full resync.
//!
//! Prices are stored as integer ticks (price * 1e8) so levels can live in a
//! `BTreeMap` and best bid/ask are O(log n).

use std::collections::BTreeMap;

const TICK_SCALE: f64 = 1e8;

fn to_ticks(price: f64) -> i64 {
    (price * TICK_SCALE).round() as i64
}

fn from_ticks(ticks: i64) -> f64 {
    ticks as f64 / TICK_SCALE
}

/// Outcome of applying one depthUpdate event.
#[derive(Debug, PartialEq, Eq)]
pub enum SyncResult {
    /// Book updated normally.
    Applied,
    /// Event was older than the book; ignored.
    Stale,
    /// Events were missed; caller must resync from a fresh snapshot.
    Gap,
}

#[derive(Debug, Default)]
pub struct OrderBook {
    bids: BTreeMap<i64, f64>,
    asks: BTreeMap<i64, f64>,
    pub last_id: i64,
}

impl OrderBook {
    pub fn new() -> Self {
        Self {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            last_id: -1,
        }
    }

    /// Load a REST snapshot. `bids`/`asks` are (price, qty) pairs.
    pub fn from_snapshot(&mut self, bids: &[(f64, f64)], asks: &[(f64, f64)], last_id: i64) {
        self.bids = bids
            .iter()
            .filter(|(_, q)| *q > 0.0)
            .map(|(p, q)| (to_ticks(*p), *q))
            .collect();
        self.asks = asks
            .iter()
            .filter(|(_, q)| *q > 0.0)
            .map(|(p, q)| (to_ticks(*p), *q))
            .collect();
        self.last_id = last_id;
    }

    /// Apply one depthUpdate. `bids`/`asks` are (price, qty) level updates;
    /// a zero quantity removes the level.
    pub fn apply_diff(
        &mut self,
        u_first: i64,
        u_last: i64,
        bids: &[(f64, f64)],
        asks: &[(f64, f64)],
    ) -> SyncResult {
        if u_last < self.last_id {
            return SyncResult::Stale;
        }
        if u_first > self.last_id + 1 {
            return SyncResult::Gap;
        }
        for (book, levels) in [(&mut self.bids, bids), (&mut self.asks, asks)] {
            for (price, qty) in levels {
                let t = to_ticks(*price);
                if *qty == 0.0 {
                    book.remove(&t);
                } else {
                    book.insert(t, *qty);
                }
            }
        }
        self.last_id = u_last;
        SyncResult::Applied
    }

    pub fn best_bid(&self) -> Option<(f64, f64)> {
        self.bids.iter().next_back().map(|(t, q)| (from_ticks(*t), *q))
    }

    pub fn best_ask(&self) -> Option<(f64, f64)> {
        self.asks.iter().next().map(|(t, q)| (from_ticks(*t), *q))
    }

    pub fn midprice(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some((b, _)), Some((a, _))) => Some((b + a) / 2.0),
            _ => None,
        }
    }

    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some((b, _)), Some((a, _))) => Some(a - b),
            _ => None,
        }
    }

    /// (bid_px, bid_sz, ask_px, ask_sz) at the touch.
    pub fn touch(&self) -> Option<(f64, f64, f64, f64)> {
        match (self.best_bid(), self.best_ask()) {
            (Some((bp, bq)), Some((ap, aq))) => Some((bp, bq, ap, aq)),
            _ => None,
        }
    }

    /// (bid volume - ask volume) / (bid volume + ask volume), top `n` levels.
    pub fn imbalance(&self, n: usize) -> Option<f64> {
        let bv: f64 = self.bids.iter().rev().take(n).map(|(_, q)| q).sum();
        let av: f64 = self.asks.iter().take(n).map(|(_, q)| q).sum();
        if bv + av == 0.0 {
            None
        } else {
            Some((bv - av) / (bv + av))
        }
    }

    pub fn len(&self) -> usize {
        self.bids.len() + self.asks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bids.is_empty() && self.asks.is_empty()
    }

    /// True when the book holds both sides and best bid >= best ask.
    ///
    /// A crossed book cannot arise from a correctly synchronized exchange
    /// stream; it signals stale data, a missed update, or a bad snapshot.
    pub fn is_crossed(&self) -> bool {
        match (self.best_bid(), self.best_ask()) {
            (Some((b, _)), Some((a, _))) => b >= a,
            _ => false,
        }
    }

    /// Bid levels as (price, qty), ascending by price.
    pub fn bid_levels(&self) -> Vec<(f64, f64)> {
        self.bids.iter().map(|(t, q)| (from_ticks(*t), *q)).collect()
    }

    /// Ask levels as (price, qty), ascending by price.
    pub fn ask_levels(&self) -> Vec<(f64, f64)> {
        self.asks.iter().map(|(t, q)| (from_ticks(*t), *q)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn book() -> OrderBook {
        let mut b = OrderBook::new();
        b.from_snapshot(
            &[(100.0, 5.0), (99.9, 3.0)],
            &[(100.1, 4.0), (100.2, 2.0)],
            10,
        );
        b
    }

    #[test]
    fn best_quotes_and_mid() {
        let b = book();
        assert_eq!(b.best_bid(), Some((100.0, 5.0)));
        assert_eq!(b.best_ask(), Some((100.1, 4.0)));
        assert!((b.midprice().unwrap() - 100.05).abs() < 1e-9);
        assert!((b.spread().unwrap() - 0.1).abs() < 1e-9);
    }

    #[test]
    fn apply_diff_updates_and_removes() {
        let mut b = book();
        // update 100.0 qty, remove 99.9, add 99.8
        let r = b.apply_diff(11, 11, &[(100.0, 7.0), (99.9, 0.0), (99.8, 1.0)], &[]);
        assert_eq!(r, SyncResult::Applied);
        assert_eq!(b.best_bid(), Some((100.0, 7.0)));
        assert_eq!(b.bids.len(), 2);
        assert_eq!(b.last_id, 11);
    }

    #[test]
    fn stale_event_ignored() {
        let mut b = book();
        let r = b.apply_diff(5, 9, &[(100.0, 99.0)], &[]);
        assert_eq!(r, SyncResult::Stale);
        assert_eq!(b.best_bid(), Some((100.0, 5.0)));
    }

    #[test]
    fn gap_detected() {
        let mut b = book();
        let r = b.apply_diff(12, 13, &[(100.0, 99.0)], &[]);
        assert_eq!(r, SyncResult::Gap);
        // book untouched
        assert_eq!(b.best_bid(), Some((100.0, 5.0)));
    }

    #[test]
    fn imbalance_sign() {
        let b = book();
        // bids 5+3=8, asks 4+2=6 -> (8-6)/14
        let imb = b.imbalance(5).unwrap();
        assert!((imb - 2.0 / 14.0).abs() < 1e-9);
    }

    #[test]
    fn empty_book_quotes_none() {
        let b = OrderBook::new();
        assert_eq!(b.midprice(), None);
        assert_eq!(b.touch(), None);
        assert_eq!(b.imbalance(5), None);
    }

    #[test]
    fn exact_next_update_applies() {
        let mut b = book();
        let r = b.apply_diff(11, 11, &[(100.0, 8.0)], &[]);
        assert_eq!(r, SyncResult::Applied);
        assert_eq!(b.last_id, 11);
        assert_eq!(b.best_bid(), Some((100.0, 8.0)));
    }

    #[test]
    fn overlapping_redelivery_is_idempotent() {
        let mut b = book();
        // same range re-applied: state must not change
        let r = b.apply_diff(9, 10, &[(100.0, 5.0)], &[(100.1, 4.0)]);
        assert_eq!(r, SyncResult::Applied);
        assert_eq!(b.last_id, 10);
        assert_eq!(b.best_bid(), Some((100.0, 5.0)));
        assert_eq!(b.best_ask(), Some((100.1, 4.0)));
    }

    #[test]
    fn partially_overlapping_event_applies() {
        let mut b = book();
        // U <= last_id < u: newer tail must be honored
        let r = b.apply_diff(10, 12, &[(100.0, 9.0)], &[]);
        assert_eq!(r, SyncResult::Applied);
        assert_eq!(b.last_id, 12);
        assert_eq!(b.best_bid(), Some((100.0, 9.0)));
    }

    #[test]
    fn duplicate_update_id_applies_idempotently() {
        let mut b = book();
        let r = b.apply_diff(10, 10, &[(100.0, 5.0)], &[]);
        assert_eq!(r, SyncResult::Applied);
        assert_eq!(b.last_id, 10);
        assert_eq!(b.best_bid(), Some((100.0, 5.0)));
    }

    #[test]
    fn one_past_boundary_is_gap() {
        let mut b = book();
        // U == last_id + 2 misses exactly one update
        let r = b.apply_diff(12, 12, &[], &[]);
        assert_eq!(r, SyncResult::Gap);
        assert_eq!(b.last_id, 10);
    }

    #[test]
    fn fully_stale_range_ignored() {
        let mut b = book();
        // u < last_id: entirely in the past
        let r = b.apply_diff(7, 9, &[(100.0, 1.0)], &[(100.1, 1.0)]);
        assert_eq!(r, SyncResult::Stale);
        assert_eq!(b.best_bid(), Some((100.0, 5.0)));
        assert_eq!(b.best_ask(), Some((100.1, 4.0)));
    }

    #[test]
    fn snapshot_replaces_book_entirely() {
        let mut b = book();
        b.from_snapshot(&[(200.0, 1.0)], &[(200.5, 1.0)], 500);
        assert_eq!(b.best_bid(), Some((200.0, 1.0)));
        assert_eq!(b.best_ask(), Some((200.5, 1.0)));
        assert_eq!(b.last_id, 500);
        assert_eq!(b.len(), 2);
        // old levels are gone
        assert_eq!(b.bid_levels(), vec![(200.0, 1.0)]);
    }

    #[test]
    fn snapshot_drops_zero_and_negative_qty() {
        let mut b = OrderBook::new();
        b.from_snapshot(
            &[(100.0, 5.0), (99.9, 0.0), (99.8, -2.0)],
            &[(100.1, 0.0)],
            1,
        );
        assert_eq!(b.bid_levels(), vec![(100.0, 5.0)]);
        assert_eq!(b.ask_levels(), Vec::<(f64, f64)>::new());
    }

    #[test]
    fn crossed_book_detected() {
        let mut b = OrderBook::new();
        // bid above ask: crossed
        b.from_snapshot(&[(100.2, 5.0)], &[(100.1, 4.0)], 1);
        assert!(b.is_crossed());
        // a diff that uncrosses clears the flag
        b.apply_diff(2, 2, &[(100.2, 0.0), (99.9, 5.0)], &[]);
        assert!(!b.is_crossed());
    }

    #[test]
    fn crossed_book_edge_cases() {
        let b = OrderBook::new();
        assert!(!b.is_crossed()); // empty: not crossed, just empty
        let mut one_sided = OrderBook::new();
        one_sided.from_snapshot(&[(100.0, 5.0)], &[], 1);
        assert!(!one_sided.is_crossed());
        // locked market (bid == ask) counts as crossed
        let mut locked = OrderBook::new();
        locked.from_snapshot(&[(100.0, 5.0)], &[(100.0, 4.0)], 1);
        assert!(locked.is_crossed());
    }

    #[test]
    fn level_accessors_cover_whole_book() {
        let b = book();
        assert_eq!(b.bid_levels(), vec![(99.9, 3.0), (100.0, 5.0)]);
        assert_eq!(b.ask_levels(), vec![(100.1, 4.0), (100.2, 2.0)]);
    }

    #[test]
    fn removing_absent_level_is_noop() {
        let mut b = book();
        let before = b.bid_levels();
        let r = b.apply_diff(11, 11, &[(77.7, 0.0)], &[(88.8, 0.0)]);
        assert_eq!(r, SyncResult::Applied);
        assert_eq!(b.bid_levels(), before);
    }

    #[test]
    fn book_emptied_by_diffs_stays_usable() {
        let mut b = book();
        b.apply_diff(11, 11, &[(100.0, 0.0), (99.9, 0.0)], &[(100.1, 0.0), (100.2, 0.0)]);
        assert!(b.is_empty());
        assert_eq!(b.best_bid(), None);
        assert_eq!(b.midprice(), None);
        assert!(!b.is_crossed());
        // and can be repopulated
        let r = b.apply_diff(12, 12, &[(100.0, 5.0)], &[(100.1, 4.0)]);
        assert_eq!(r, SyncResult::Applied);
        assert_eq!(b.best_bid(), Some((100.0, 5.0)));
    }
}
