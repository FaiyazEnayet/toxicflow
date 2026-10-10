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
}
