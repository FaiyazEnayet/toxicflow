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

// =====================================================================
// E2: transactional book sync / fail-closed recorder.
//
// The record path (recorder::Session) and the live dashboard path now
// share one sync state machine (booksync::BookSync). These tests drive
// captured-event fixtures through both and pin the transactional
// contract: nothing is committed or emitted until validation passes, an
// invalid message is rejected instead of being read as zeros, and a
// snapshot or diff sequence that would cross the book never syncs.
// =====================================================================

use toxicflow::booksync::{
    BookSync, DepthEvent, DepthOutcome, MergeOutcome, SnapshotData, MAX_BUFFERED_EVENTS,
};
use toxicflow::metrics::Quote;
use toxicflow::recorder::Session;

fn s(v: &str) -> String {
    v.to_string()
}

fn depth_json(
    u_first: i64,
    u_last: i64,
    bids: &[(&str, &str)],
    asks: &[(&str, &str)],
) -> serde_json::Value {
    let pairs = |lv: &[(&str, &str)]| {
        lv.iter()
            .map(|(p, q)| vec![p.to_string(), q.to_string()])
            .collect::<Vec<_>>()
    };
    serde_json::json!({
        "e": "depthUpdate",
        "E": 1_700_000_000_000i64 + u_first,
        "U": u_first,
        "u": u_last,
        "b": pairs(bids),
        "a": pairs(asks),
    })
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

fn snap_ok() -> SnapshotData {
    SnapshotData::from_pairs(
        100,
        &[(s("100.0"), s("5.0")), (s("99.9"), s("3.0"))],
        &[(s("100.1"), s("4.0")), (s("100.2"), s("2.0"))],
    )
    .expect("fixture snapshot is valid")
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

#[test]
fn stage3_toxicflow_2_invalid_messages_never_parse_as_zero() {
    // non-numeric price in a depth event
    let bad = depth_json(101, 101, &[("not-a-price", "1.0")], &[]);
    assert!(DepthEvent::from_json(&bad).is_err());
    // non-positive price
    let bad = depth_json(101, 101, &[("0", "1.0")], &[]);
    assert!(DepthEvent::from_json(&bad).is_err());
    let bad = depth_json(101, 101, &[("-5.0", "1.0")], &[]);
    assert!(DepthEvent::from_json(&bad).is_err());
    // negative quantity (a zero quantity is a legitimate level removal)
    let bad = depth_json(101, 101, &[("100.0", "-1.0")], &[]);
    assert!(DepthEvent::from_json(&bad).is_err());
    // non-finite quantity spelled as a float string
    let bad = depth_json(101, 101, &[("100.0", "NaN")], &[]);
    assert!(DepthEvent::from_json(&bad).is_err());
    // update ids out of range: u < U, and negative ids
    let bad = depth_json(105, 104, &[], &[]);
    assert!(DepthEvent::from_json(&bad).is_err());
    let bad = depth_json(-1, 104, &[], &[]);
    assert!(DepthEvent::from_json(&bad).is_err());
    // ids missing entirely (the old code defaulted them to 0)
    let mut missing = depth_json(101, 101, &[], &[]);
    missing.as_object_mut().unwrap().remove("U");
    assert!(DepthEvent::from_json(&missing).is_err());
    // a well-formed event still parses, and a zero-qty removal is kept
    let ok = DepthEvent::from_json(&depth_json(101, 101, &[("100.0", "0")], &[])).unwrap();
    assert_eq!(ok.bids, vec![(100.0, 0.0)]);

    // snapshots validate the same way
    assert!(SnapshotData::from_pairs(-1, &[], &[]).is_err());
    assert!(SnapshotData::from_pairs(100, &[(s("abc"), s("1.0"))], &[]).is_err());
    assert!(SnapshotData::from_pairs(100, &[(s("100.0"), s("-2.0"))], &[]).is_err());
    assert!(snap_ok().last_update_id == 100);
}

#[test]
fn stage3_toxicflow_2_gap_mid_merge_leaves_book_and_quotes_unchanged() {
    let rt = runtime();
    rt.block_on(async {
        let mut sess = Session::new();
        // buffer two events, then a third whose predecessor never arrived
        sess.on_depth(&depth_json(101, 102, &[("100.0", "6.0")], &[]), "BTCUSDT");
        sess.on_depth(&depth_json(103, 103, &[("99.9", "0")], &[]), "BTCUSDT");
        sess.on_depth(&depth_json(105, 105, &[("100.0", "8.0")], &[]), "BTCUSDT");
        assert_eq!(sess.buffered_len(), 3);

        // the merge must fail as a whole: no half-applied book, and not
        // one quote emitted for the failed attempt
        assert!(!sess.merge_snapshot_data(&snap_ok()));
        assert!(!sess.is_synced());
        assert_eq!(sess.book().last_id, -1);
        assert!(sess.book().is_empty());
        assert_eq!(sess.quotes().len(), 0);
        // buffered events survive for the next attempt
        assert_eq!(sess.buffered_len(), 3);
    });
}

#[test]
fn stage3_toxicflow_2_successful_merge_commits_book_and_quotes_together() {
    let rt = runtime();
    rt.block_on(async {
        let mut sess = Session::new();
        sess.on_depth(&depth_json(101, 102, &[("100.0", "6.0")], &[]), "BTCUSDT");
        sess.on_depth(
            &depth_json(103, 103, &[("99.9", "0")], &[("100.2", "9.0")]),
            "BTCUSDT",
        );
        assert!(sess.merge_snapshot_data(&snap_ok()));
        assert!(sess.is_synced());
        assert_eq!(sess.book().best_bid(), Some((100.0, 6.0)));
        assert_eq!(sess.book().last_id, 103);
        // exactly one quote per committed event, none for the snapshot
        assert_eq!(sess.quotes().len(), 2);
        assert_eq!(sess.quotes()[1].bid_px, 100.0);
        assert_eq!(sess.quotes()[1].ask_px, 100.1);
    });
}

#[test]
fn stage3_toxicflow_2_out_of_order_duplicate_and_conflicting_ids() {
    let mut sync = BookSync::new();
    // while syncing, events buffer in arrival order
    assert_eq!(
        sync.on_depth(ev(101, 102, &[(100.0, 6.0)], &[])),
        DepthOutcome::Buffered
    );
    assert_eq!(
        sync.merge_snapshot(&snap_ok(), &mut Vec::new()),
        MergeOutcome::Synced { applied: 1 }
    );
    assert!(sync.is_synced());

    // duplicate redelivery of the just-applied event: idempotent
    let before = sync.book().bid_levels();
    assert_eq!(
        sync.on_depth(ev(101, 102, &[(100.0, 6.0)], &[])),
        DepthOutcome::Applied
    );
    assert_eq!(sync.book().bid_levels(), before);

    // an older event arriving late is classified Stale, state untouched
    assert_eq!(
        sync.on_depth(ev(90, 95, &[(100.0, 1.0)], &[])),
        DepthOutcome::Stale
    );
    assert_eq!(sync.book().bid_levels(), before);

    // conflicting ids: the exact id range of the last applied event
    // redelivered with different content cannot be reconciled with the
    // committed book, so it is treated as a gap: resync, with the
    // event kept as buffer head and an explicit diagnostic recorded
    let out = sync.on_depth(ev(101, 102, &[(100.0, 42.0)], &[]));
    assert!(matches!(out, DepthOutcome::GapResync { .. }), "got {out:?}");
    assert!(!sync.is_synced());
    assert_eq!(sync.buffered_len(), 1);
    let (n_gaps, diag) = sync.gap_diagnostics();
    assert!(n_gaps >= 1);
    assert!(diag.is_some());

    // a genuinely out-of-order arrival (successor before predecessor) is
    // also a gap, never a silent apply
    let mut sync2 = BookSync::new();
    sync2.on_depth(ev(101, 101, &[], &[]));
    assert_eq!(
        sync2.merge_snapshot(&snap_ok(), &mut Vec::new()),
        MergeOutcome::Synced { applied: 1 }
    );
    let out = sync2.on_depth(ev(103, 103, &[(100.0, 7.0)], &[]));
    assert!(matches!(out, DepthOutcome::GapResync { .. }), "got {out:?}");
    assert!(!sync2.is_synced());
}

#[test]
fn stage3_toxicflow_2_crossed_or_invalid_snapshot_never_syncs() {
    // crossed snapshot: best bid above best ask
    let crossed =
        SnapshotData::from_pairs(100, &[(s("100.2"), s("5.0"))], &[(s("100.1"), s("4.0"))])
            .unwrap();
    let mut sync = BookSync::new();
    sync.on_depth(ev(101, 101, &[(100.0, 6.0)], &[]));
    assert_eq!(
        sync.merge_snapshot(&crossed, &mut Vec::new()),
        MergeOutcome::Refetch
    );
    assert!(!sync.is_synced());
    assert!(sync.book().is_empty());

    // a valid snapshot whose buffered diffs cross the book mid-merge is
    // likewise refused as a whole
    let mut sync = BookSync::new();
    sync.on_depth(ev(101, 101, &[(100.3, 1.0)], &[]));
    assert_eq!(
        sync.merge_snapshot(&snap_ok(), &mut Vec::new()),
        MergeOutcome::Refetch
    );
    assert!(!sync.is_synced());
    assert!(sync.book().is_empty());

    // a diff that would cross an already-synced book forces a resync
    // instead of committing a crossed state
    let mut sync = BookSync::new();
    sync.on_depth(ev(101, 101, &[], &[]));
    assert_eq!(
        sync.merge_snapshot(&snap_ok(), &mut Vec::new()),
        MergeOutcome::Synced { applied: 1 }
    );
    let out = sync.on_depth(ev(102, 102, &[(100.2, 1.0)], &[]));
    assert!(matches!(out, DepthOutcome::GapResync { .. }), "got {out:?}");
    assert!(!sync.is_synced());
    assert!(!sync.book().is_crossed());
}

#[test]
fn stage3_toxicflow_2_stale_snapshot_is_refetched_not_stalled() {
    // the live path's old failure mode: a snapshot older than the first
    // buffered event must be a visible Refetch (so the caller fetches a
    // fresher one), never a silent dead end
    let mut sync = BookSync::new();
    sync.on_depth(ev(201, 202, &[(100.0, 6.0)], &[]));
    assert_eq!(
        sync.merge_snapshot(&snap_ok(), &mut Vec::new()),
        MergeOutcome::Refetch
    );
    assert!(!sync.is_synced());
    // the buffered event is retained, and a snapshot that continues the
    // stream (lastUpdateId = 200) merges cleanly
    let fresh = SnapshotData::from_pairs(200, &[(s("100.0"), s("5.0"))], &[(s("100.1"), s("4.0"))])
        .unwrap();
    assert_eq!(
        sync.merge_snapshot(&fresh, &mut Vec::new()),
        MergeOutcome::Synced { applied: 1 }
    );
    assert!(sync.is_synced());
    assert_eq!(sync.book().best_bid(), Some((100.0, 6.0)));
}

#[test]
fn stage3_toxicflow_2_buffer_is_bounded_with_diagnostics() {
    let mut sync = BookSync::new();
    for i in 0..(MAX_BUFFERED_EVENTS as i64 + 50) {
        let id = 1000 + i;
        sync.on_depth(ev(id, id, &[(100.0, 1.0)], &[]));
    }
    assert!(sync.buffered_len() <= MAX_BUFFERED_EVENTS);
    let (_, diag) = sync.gap_diagnostics();
    assert!(diag.is_some(), "overflow must leave an explicit diagnostic");
}

#[test]
fn stage3_toxicflow_2_oracle_matches_every_committed_state() {
    // Independent oracle: the ReferenceBook from the replay suite above,
    // fed the clean stream; the sync engine, driven through snapshot
    // merge and live diffs, must match it after every committed step.
    let s = generate_stream(21, 400);
    let mut sync = BookSync::new();
    let mut truth = synced_truth(&s);

    let snap = SnapshotData {
        last_update_id: 0,
        bids: s.init_bids.clone(),
        asks: s.init_asks.clone(),
    };
    // buffer the first ten events, then merge the snapshot under them
    for e in &s.events[..10] {
        sync.on_depth(ev(e.u_first, e.u_last, &e.bids, &e.asks));
    }
    let out = sync.merge_snapshot(&snap, &mut Vec::new());
    assert!(matches!(out, MergeOutcome::Synced { .. }), "got {out:?}");
    for e in &s.events[..10] {
        truth.apply(e.u_first, e.u_last, &e.bids, &e.asks);
    }
    assert_books_match(sync.book(), &truth);

    for e in &s.events[10..] {
        truth.apply(e.u_first, e.u_last, &e.bids, &e.asks);
        assert_eq!(
            sync.on_depth(ev(e.u_first, e.u_last, &e.bids, &e.asks)),
            DepthOutcome::Applied
        );
        assert_books_match(sync.book(), &truth);
    }
}

#[test]
fn stage3_toxicflow_2_live_and_record_paths_agree() {
    // The same captured events through both paths: Session (record) and
    // a bare BookSync driven the way live_dashboard drives it. Committed
    // books must agree after every step, and the record path's quotes
    // must number exactly the events the live path committed, ending at
    // the live book's touch.
    let rt = runtime();
    rt.block_on(async {
        let s = generate_stream(22, 300);
        let mut sess = Session::new();
        let mut live = BookSync::new();
        let mut committed_events = 0usize;

        let snap = SnapshotData {
            last_update_id: 0,
            bids: s.init_bids.clone(),
            asks: s.init_asks.clone(),
        };
        let to_json = |e: &Event| {
            let pairs = |lv: &[(f64, f64)]| {
                lv.iter()
                    .map(|(p, q)| vec![p.to_string(), q.to_string()])
                    .collect::<Vec<_>>()
            };
            serde_json::json!({
                "e": "depthUpdate",
                "E": 1_700_000_000_000i64 + e.u_first,
                "U": e.u_first,
                "u": e.u_last,
                "b": pairs(&e.bids),
                "a": pairs(&e.asks),
            })
        };

        for e in &s.events[..10] {
            sess.on_depth(&to_json(e), "BTCUSDT");
            live.on_depth(ev(e.u_first, e.u_last, &e.bids, &e.asks));
        }
        assert!(sess.merge_snapshot_data(&snap));
        match live.merge_snapshot(&snap, &mut Vec::new()) {
            MergeOutcome::Synced { applied } => committed_events += applied,
            other => panic!("live merge failed: {other:?}"),
        }
        assert_eq!(sess.book().bid_levels(), live.book().bid_levels());
        assert_eq!(sess.book().ask_levels(), live.book().ask_levels());

        for e in &s.events[10..] {
            sess.on_depth(&to_json(e), "BTCUSDT");
            let out = live.on_depth(ev(e.u_first, e.u_last, &e.bids, &e.asks));
            assert_eq!(out, DepthOutcome::Applied);
            committed_events += 1;
            assert_eq!(sess.book().bid_levels(), live.book().bid_levels());
            assert_eq!(sess.book().ask_levels(), live.book().ask_levels());
            assert_eq!(sess.book().last_id, live.book().last_id);
        }

        assert_eq!(sess.quotes().len(), committed_events);
        let (bp, bq, ap, aq) = live.book().touch().unwrap();
        let last: &Quote = sess.quotes().last().unwrap();
        assert_eq!(
            (last.bid_px, last.bid_sz, last.ask_px, last.ask_sz),
            (bp, bq, ap, aq)
        );
    });
}
