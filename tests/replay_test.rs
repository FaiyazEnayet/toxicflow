//! Market-data replay tests: the local book must track a simulated
//! exchange feed through drops, duplicates, resyncs, and bursts.
//!
//! A deterministic event generator drives two books in parallel:
//! - `ReferenceBook`: a deliberately simple, independent implementation
//!   (HashMap keyed on raw f64 bits, no tick conversion) fed the clean,
//!   in-order stream. This is ground truth.
//! - `OrderBook`: the production implementation, fed the same stream under
//!   adverse delivery (drops, duplicates, stale redelivery, resyncs).
//!
//! After each scenario the two must agree exactly on every level.
//!
//! Note on out-of-order delivery: the exchange stream arrives over a single
//! TCP connection, so the protocol assumes in-order delivery and handles
//! redelivery via Stale/overlap logic. A genuinely reordered stream (newer
//! event applied before an older one carrying unique updates) is outside the
//! documented sync procedure; the tests below lock in the actual behavior
//! (Stale classification, no corruption, resync restores convergence).

use std::collections::{HashMap, HashSet};

use toxicflow::orderbook::{OrderBook, SyncResult};

/// Deterministic xorshift PRNG (same family as the other integration test).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut s = self.0;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        self.0 = s;
        s
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (self.next() as f64 / u64::MAX as f64) * (hi - lo)
    }
}

/// Independent ground truth: absolute level updates keyed on raw f64 bits.
/// No tick conversion, no BTreeMap — a different code path from OrderBook.
type BookSnapshot = (Vec<(f64, f64)>, Vec<(f64, f64)>, i64);

#[derive(Default)]
struct ReferenceBook {
    bids: HashMap<u64, f64>,
    asks: HashMap<u64, f64>,
    last_id: i64,
}

impl ReferenceBook {
    fn apply(&mut self, u_first: i64, u_last: i64, bids: &[(f64, f64)], asks: &[(f64, f64)]) {
        assert_eq!(
            u_first,
            self.last_id + 1,
            "reference consumes a gapless in-order stream"
        );
        for (p, q) in bids {
            if *q == 0.0 {
                self.bids.remove(&p.to_bits());
            } else {
                self.bids.insert(p.to_bits(), *q);
            }
        }
        for (p, q) in asks {
            if *q == 0.0 {
                self.asks.remove(&p.to_bits());
            } else {
                self.asks.insert(p.to_bits(), *q);
            }
        }
        self.last_id = u_last;
    }

    fn snapshot(&self) -> BookSnapshot {
        (
            self.bids
                .iter()
                .map(|(b, q)| (f64::from_bits(*b), *q))
                .collect(),
            self.asks
                .iter()
                .map(|(b, q)| (f64::from_bits(*b), *q))
                .collect(),
            self.last_id,
        )
    }
}

struct Event {
    u_first: i64,
    u_last: i64,
    bids: Vec<(f64, f64)>,
    asks: Vec<(f64, f64)>,
}

struct Stream {
    init_bids: Vec<(f64, f64)>,
    init_asks: Vec<(f64, f64)>,
    events: Vec<Event>,
}

/// Deterministic stream of depthUpdate-style events.
///
/// Price grid: bids at 99.50..99.99, asks at 100.10..100.59, 0.01 apart —
/// far above the 1e-8 tick resolution, so no two prices share a tick.
/// Bids always stay below asks, so the stream never crosses by construction.
fn generate_stream(seed: u64, n_events: usize) -> Stream {
    let mut rng = Rng(seed);
    let bid_grid: Vec<f64> = (0..50).map(|i| 99.50 + i as f64 * 0.01).collect();
    let ask_grid: Vec<f64> = (0..50).map(|i| 100.10 + i as f64 * 0.01).collect();

    let mut init_bids = Vec::new();
    let mut init_asks = Vec::new();
    for &p in &bid_grid[30..50] {
        init_bids.push((p, 1.0 + rng.range(0.0, 9.0)));
    }
    for &p in &ask_grid[0..20] {
        init_asks.push((p, 1.0 + rng.range(0.0, 9.0)));
    }

    let mut events = Vec::new();
    let mut next_id = 1i64;
    for _ in 0..n_events {
        // like a 100ms aggregation window: 1-3 update ids per event
        let span = 1 + rng.below(3) as i64;
        let u_first = next_id;
        let u_last = next_id + span - 1;
        next_id += span;

        let n_touch = 1 + rng.below(5) as usize;
        let mut bids = Vec::new();
        let mut asks = Vec::new();
        let mut used = HashSet::new();
        for _ in 0..n_touch {
            let on_bid = rng.below(2) == 0;
            let pool = if on_bid { &bid_grid } else { &ask_grid };
            let p = pool[rng.below(pool.len() as u64) as usize];
            if !used.insert(p.to_bits()) {
                continue; // one update per price per event
            }
            let q = if rng.below(8) == 0 {
                0.0 // level removed
            } else {
                0.5 + rng.range(0.0, 10.0)
            };
            if on_bid {
                bids.push((p, q));
            } else {
                asks.push((p, q));
            }
        }
        events.push(Event {
            u_first,
            u_last,
            bids,
            asks,
        });
    }
    Stream {
        init_bids,
        init_asks,
        events,
    }
}

/// Fresh production book loaded from the stream's initial snapshot.
fn synced_book(s: &Stream) -> OrderBook {
    let mut b = OrderBook::new();
    b.from_snapshot(&s.init_bids, &s.init_asks, 0);
    b
}

/// Fresh reference book loaded from the stream's initial snapshot.
fn synced_truth(s: &Stream) -> ReferenceBook {
    let mut t = ReferenceBook::default();
    for (p, q) in &s.init_bids {
        t.bids.insert(p.to_bits(), *q);
    }
    for (p, q) in &s.init_asks {
        t.asks.insert(p.to_bits(), *q);
    }
    t.last_id = 0;
    t
}

/// Exact state comparison: every level, both sides, plus the update id.
/// Prices compare within 1e-9 (tick round-trip); quantities are exact.
fn assert_books_match(book: &OrderBook, truth: &ReferenceBook) {
    let mut bb = book.bid_levels();
    bb.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut tb: Vec<(f64, f64)> = truth
        .bids
        .iter()
        .map(|(k, q)| (f64::from_bits(*k), *q))
        .collect();
    tb.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    assert_eq!(bb.len(), tb.len(), "bid level count diverged");
    for (x, y) in bb.iter().zip(tb.iter()) {
        assert!(
            (x.0 - y.0).abs() < 1e-9,
            "bid price diverged: {x:?} vs {y:?}"
        );
        assert_eq!(x.1, y.1, "bid qty diverged at {}", x.0);
    }

    let mut ba = book.ask_levels();
    ba.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut ta: Vec<(f64, f64)> = truth
        .asks
        .iter()
        .map(|(k, q)| (f64::from_bits(*k), *q))
        .collect();
    ta.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    assert_eq!(ba.len(), ta.len(), "ask level count diverged");
    for (x, y) in ba.iter().zip(ta.iter()) {
        assert!(
            (x.0 - y.0).abs() < 1e-9,
            "ask price diverged: {x:?} vs {y:?}"
        );
        assert_eq!(x.1, y.1, "ask qty diverged at {}", x.0);
    }
    assert_eq!(book.last_id, truth.last_id, "update id diverged");
}

#[test]
fn replay_clean_stream_converges_exactly() {
    let s = generate_stream(7, 3_000);
    let mut book = synced_book(&s);
    let mut truth = synced_truth(&s);
    for e in &s.events {
        truth.apply(e.u_first, e.u_last, &e.bids, &e.asks);
        assert_eq!(
            book.apply_diff(e.u_first, e.u_last, &e.bids, &e.asks),
            SyncResult::Applied
        );
        assert!(!book.is_crossed(), "clean stream must never cross");
    }
    assert_books_match(&book, &truth);
}

#[test]
fn replay_duplicate_events_are_idempotent() {
    let s = generate_stream(8, 3_000);
    let mut book = synced_book(&s);
    let mut truth = synced_truth(&s);
    let mut rng = Rng(99);
    for (i, e) in s.events.iter().enumerate() {
        truth.apply(e.u_first, e.u_last, &e.bids, &e.asks);
        assert_eq!(
            book.apply_diff(e.u_first, e.u_last, &e.bids, &e.asks),
            SyncResult::Applied
        );
        // immediate redelivery: same content, must not change state
        if rng.below(5) == 0 {
            let before = book.bid_levels();
            assert_eq!(
                book.apply_diff(e.u_first, e.u_last, &e.bids, &e.asks),
                SyncResult::Applied
            );
            assert_eq!(book.bid_levels(), before);
        }
        // redelivery of a much older event: must be Stale, state untouched
        if i > 0 && i % 37 == 0 {
            let old = &s.events[i - 1];
            let before_bid = book.best_bid();
            let before_ask = book.best_ask();
            assert_eq!(
                book.apply_diff(old.u_first, old.u_last, &old.bids, &old.asks),
                SyncResult::Stale
            );
            assert_eq!(book.best_bid(), before_bid);
            assert_eq!(book.best_ask(), before_ask);
        }
    }
    assert_books_match(&book, &truth);
}

#[test]
fn replay_dropped_events_resync_to_exact_state() {
    let s = generate_stream(9, 3_000);
    let mut book = synced_book(&s);
    // truth plays the exchange: it sees every event, dropped or not
    let mut truth = synced_truth(&s);
    let mut rng = Rng(100);
    let mut resyncs = 0;
    let mut i = 0;
    while i < s.events.len() {
        // randomly drop 1-3 events, then let the next one signal the gap
        if rng.below(20) == 0 {
            let drop = 1 + rng.below(3) as usize;
            let gap_idx = i + drop;
            if gap_idx < s.events.len() {
                // the exchange saw the dropped events; the book did not
                for e in &s.events[i..gap_idx] {
                    truth.apply(e.u_first, e.u_last, &e.bids, &e.asks);
                }
                let ge = &s.events[gap_idx];
                assert_eq!(
                    book.apply_diff(ge.u_first, ge.u_last, &ge.bids, &ge.asks),
                    SyncResult::Gap,
                    "dropped events must be detected"
                );
                // resync protocol: fresh snapshot from the exchange's
                // current state, then the stream continues cleanly
                let (sb, sa, sid) = truth.snapshot();
                book.from_snapshot(&sb, &sa, sid);
                resyncs += 1;
                assert_eq!(
                    book.apply_diff(ge.u_first, ge.u_last, &ge.bids, &ge.asks),
                    SyncResult::Applied
                );
                truth.apply(ge.u_first, ge.u_last, &ge.bids, &ge.asks);
                i = gap_idx + 1;
                continue;
            }
        }
        let e = &s.events[i];
        truth.apply(e.u_first, e.u_last, &e.bids, &e.asks);
        assert_eq!(
            book.apply_diff(e.u_first, e.u_last, &e.bids, &e.asks),
            SyncResult::Applied
        );
        i += 1;
    }
    assert!(resyncs > 20, "expected many resyncs, got {resyncs}");
    assert_books_match(&book, &truth);
}

#[test]
fn replay_reconnect_midstream_snapshot_converges() {
    let s = generate_stream(11, 3_000);
    let mut book = synced_book(&s);
    let mut truth = synced_truth(&s);
    let cut = s.events.len() / 2;
    for e in &s.events[..cut] {
        truth.apply(e.u_first, e.u_last, &e.bids, &e.asks);
        assert_eq!(
            book.apply_diff(e.u_first, e.u_last, &e.bids, &e.asks),
            SyncResult::Applied
        );
    }
    // simulate a dropped connection: fresh snapshot, then resume the stream
    let (sb, sa, sid) = truth.snapshot();
    let mut book2 = OrderBook::new();
    book2.from_snapshot(&sb, &sa, sid);
    for e in &s.events[cut..] {
        truth.apply(e.u_first, e.u_last, &e.bids, &e.asks);
        assert_eq!(
            book2.apply_diff(e.u_first, e.u_last, &e.bids, &e.asks),
            SyncResult::Applied
        );
        assert!(!book2.is_crossed());
    }
    assert_books_match(&book2, &truth);
}

#[test]
fn replay_burst_of_extreme_activity() {
    // 20k events, up to 5 level updates each: the book must keep up and
    // stay exactly in sync, with no spurious crossed state.
    let s = generate_stream(12, 20_000);
    let mut book = synced_book(&s);
    let mut truth = synced_truth(&s);
    for e in &s.events {
        truth.apply(e.u_first, e.u_last, &e.bids, &e.asks);
        assert_eq!(
            book.apply_diff(e.u_first, e.u_last, &e.bids, &e.asks),
            SyncResult::Applied
        );
        assert!(!book.is_crossed());
    }
    assert_books_match(&book, &truth);
    assert!(!book.is_empty());
}

#[test]
fn replay_adversarial_mix_converges() {
    // The gauntlet: duplicates, stale redeliveries, and drop-induced
    // resyncs, all in one stream, driven by a second PRNG.
    let s = generate_stream(13, 5_000);
    let mut book = synced_book(&s);
    let mut truth = synced_truth(&s);
    let mut rng = Rng(2024);
    let mut i = 0;
    let mut resyncs = 0;
    while i < s.events.len() {
        let roll = rng.below(100);
        if roll < 5 {
            // drop 1-2 events -> gap -> resync
            let gap_idx = i + 1 + rng.below(2) as usize;
            if gap_idx < s.events.len() {
                // exchange saw the dropped events
                for e in &s.events[i..gap_idx] {
                    truth.apply(e.u_first, e.u_last, &e.bids, &e.asks);
                }
                let ge = &s.events[gap_idx];
                assert_eq!(
                    book.apply_diff(ge.u_first, ge.u_last, &ge.bids, &ge.asks),
                    SyncResult::Gap
                );
                let (sb, sa, sid) = truth.snapshot();
                book.from_snapshot(&sb, &sa, sid);
                resyncs += 1;
                assert_eq!(
                    book.apply_diff(ge.u_first, ge.u_last, &ge.bids, &ge.asks),
                    SyncResult::Applied
                );
                truth.apply(ge.u_first, ge.u_last, &ge.bids, &ge.asks);
                i = gap_idx + 1;
                continue;
            }
        }
        let e = &s.events[i];
        truth.apply(e.u_first, e.u_last, &e.bids, &e.asks);
        assert_eq!(
            book.apply_diff(e.u_first, e.u_last, &e.bids, &e.asks),
            SyncResult::Applied
        );
        if roll < 25 {
            // duplicate redelivery: idempotent
            assert_eq!(
                book.apply_diff(e.u_first, e.u_last, &e.bids, &e.asks),
                SyncResult::Applied
            );
        } else if roll < 35 && i > 5 {
            // stale redelivery from well in the past
            let old = &s.events[i - 5];
            assert_eq!(
                book.apply_diff(old.u_first, old.u_last, &old.bids, &old.asks),
                SyncResult::Stale
            );
        }
        assert!(!book.is_crossed());
        i += 1;
    }
    assert!(resyncs > 0);
    assert_books_match(&book, &truth);
}

#[test]
fn out_of_order_arrival_is_classified_not_corrupting() {
    // Deliver event i+1 before event i. Event i's updates are missing from
    // the book's point of view, so the later event must signal Gap (not
    // silently apply): the protocol detects the inconsistency instead of
    // corrupting state, and a resync restores exact convergence.
    let s = generate_stream(14, 500);
    let mut book = synced_book(&s);
    let mut truth = synced_truth(&s);
    for e in &s.events[..100] {
        truth.apply(e.u_first, e.u_last, &e.bids, &e.asks);
        assert_eq!(
            book.apply_diff(e.u_first, e.u_last, &e.bids, &e.asks),
            SyncResult::Applied
        );
    }
    let a = &s.events[100];
    let b = &s.events[101];
    truth.apply(a.u_first, a.u_last, &a.bids, &a.asks);
    truth.apply(b.u_first, b.u_last, &b.bids, &b.asks);
    // b arrives while a's updates are still missing: Gap, book untouched
    let before_bid = book.best_bid();
    assert_eq!(
        book.apply_diff(b.u_first, b.u_last, &b.bids, &b.asks),
        SyncResult::Gap
    );
    assert_eq!(book.best_bid(), before_bid);
    // resync from the exchange state recovers exactly
    let (sb, sa, sid) = truth.snapshot();
    book.from_snapshot(&sb, &sa, sid);
    // a now arrives late: fully covered by the snapshot, Stale, ignored
    assert_eq!(
        book.apply_diff(a.u_first, a.u_last, &a.bids, &a.asks),
        SyncResult::Stale
    );
    for e in &s.events[102..] {
        truth.apply(e.u_first, e.u_last, &e.bids, &e.asks);
        assert_eq!(
            book.apply_diff(e.u_first, e.u_last, &e.bids, &e.asks),
            SyncResult::Applied
        );
    }
    assert_books_match(&book, &truth);
}
