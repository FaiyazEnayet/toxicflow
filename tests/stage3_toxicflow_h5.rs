//! Stage 3 toxicflow H5: document the distinction between the internal
//! 1e-8 price scale and the symbol exchange tick-size filter; validate
//! tick precision, finite prices and nonnegative sizes before converting.
//!
//! The internal 1e-8 scale is representation precision only. The
//! exchange tick-size filter is a separate per-symbol rule. These tests
//! pin both: decimal-to-ticks-to-decimal round-trips exactly for the
//! listed instrument tick/lot fixtures, while off-tick, nonfinite and
//! negative quantities fail and are never accepted into the book. They
//! also pin the replay digest of one valid reference event tape, which
//! must be unchanged by the validation.

use toxicflow::orderbook::{BookValidity, OrderBook, TickError, TickFilter};
use toxicflow::replay::replay;
use toxicflow::storage::JournalRecord;

/// Listed instrument tick/lot fixtures: (filter, on-tick prices).
fn fixtures() -> Vec<(TickFilter, Vec<f64>)> {
    vec![
        (
            TickFilter::btc_usdt(),
            vec![0.01, 100.0, 99.9, 100.1, 100.2, 67890.12, 0.02],
        ),
        (TickFilter::eth_usdt(), vec![0.01, 2500.5, 2500.51, 1234.56]),
        (
            TickFilter::finest(),
            vec![0.00000001, 0.00000005, 1.00000001, 100.00000001],
        ),
        (
            TickFilter::new(0.0001, 0.01, 1.0).expect("low-price fixture"),
            vec![0.0001, 0.0002, 0.001, 0.1234],
        ),
    ]
}

#[test]
fn stage3_toxicflow_h5_roundtrip_exact_for_fixtures() {
    for (filter, prices) in fixtures() {
        for p in prices {
            let ticks = filter
                .to_ticks_checked(p)
                .expect("fixture price must convert");
            assert_eq!(
                TickFilter::from_ticks(ticks),
                p,
                "decimal->ticks->decimal must round-trip exactly for {p} under {filter:?}"
            );
        }
    }
}

#[test]
fn stage3_toxicflow_h5_roundtrip_exact_through_book() {
    // What goes into the book comes back out bit-identical.
    let mut b = OrderBook::new().with_tick_filter(TickFilter::btc_usdt());
    b.from_snapshot(
        &[(100.0, 5.0), (99.9, 3.0), (0.01, 1.0)],
        &[(100.1, 4.0), (100.2, 2.0), (67890.12, 0.5)],
        10,
    );
    assert_eq!(b.bid_levels(), vec![(0.01, 1.0), (99.9, 3.0), (100.0, 5.0)]);
    assert_eq!(
        b.ask_levels(),
        vec![(100.1, 4.0), (100.2, 2.0), (67890.12, 0.5)]
    );
    assert_eq!(b.invalid_rejected(), 0);
}

#[test]
fn stage3_toxicflow_h5_scale_is_not_tick_size() {
    // 100.005 stores exactly at the 1e-8 scale, so the finest filter
    // accepts it; the BTCUSDT 0.01 filter rejects it as off-tick.
    // That is the whole distinction: the scale is representation, the
    // tick size is the venue rule.
    let ticks = TickFilter::finest()
        .to_ticks_checked(100.005)
        .expect("100.005 is representable at 1e-8");
    assert_eq!(TickFilter::from_ticks(ticks), 100.005);
    assert_eq!(
        TickFilter::btc_usdt().to_ticks_checked(100.005),
        Err(TickError::OffTick {
            price: 100.005,
            tick_size: 0.01
        })
    );
    for p in [0.015, 99.995] {
        assert!(matches!(
            TickFilter::btc_usdt().to_ticks_checked(p),
            Err(TickError::OffTick { .. })
        ));
    }
}

#[test]
fn stage3_toxicflow_h5_arithmetic_price_converts() {
    // Regression: the fault injector builds the crossing price as best
    // ask + 0.05, which in f64 is 100.15000000000000968, one ulp off
    // the decimal 100.15. It names the 100.15 tick and must convert,
    // not fail as unrepresentable.
    let cross_bid = 100.1 + 0.05;
    assert_ne!(cross_bid, 100.15);
    let ticks = TickFilter::btc_usdt()
        .to_ticks_checked(cross_bid)
        .expect("arithmetic price must convert");
    assert_eq!(TickFilter::from_ticks(ticks), 100.15);
    // And it crosses a book whose best ask is 100.1.
    let mut b = OrderBook::new().with_tick_filter(TickFilter::btc_usdt());
    b.from_snapshot(&[(100.0, 5.0)], &[(100.1, 4.0)], 10);
    b.apply_diff(11, 11, &[(cross_bid, 1.0)], &[]);
    assert!(b.is_crossed());
    assert_eq!(b.invalid_rejected(), 0);
}

#[test]
fn stage3_toxicflow_h5_off_tick_rejected_in_book() {
    let mut b = OrderBook::new().with_tick_filter(TickFilter::btc_usdt());
    b.from_snapshot(&[(100.0, 5.0)], &[(100.1, 4.0)], 10);
    let before = b.bid_levels();
    let r = b.apply_diff(11, 11, &[(100.005, 1.0), (100.0, 7.0)], &[]);
    assert_eq!(r, toxicflow::orderbook::SyncResult::Applied);
    // The off-tick level never entered the book; the valid level did.
    assert_eq!(b.bid_levels(), vec![(100.0, 7.0)]);
    assert_eq!(b.invalid_rejected(), 1);
    assert_eq!(
        b.last_rejection(),
        Some(TickError::OffTick {
            price: 100.005,
            tick_size: 0.01
        })
    );
    // A snapshot level that is off-tick is dropped and counted too.
    let mut s = OrderBook::new().with_tick_filter(TickFilter::btc_usdt());
    s.from_snapshot(&[(100.005, 5.0), (100.0, 5.0)], &[(100.1, 4.0)], 1);
    assert_eq!(s.bid_levels(), vec![(100.0, 5.0)]);
    assert_eq!(s.invalid_rejected(), 1);
    assert_ne!(before, b.bid_levels(), "valid update still applies");
}

#[test]
fn stage3_toxicflow_h5_nonfinite_and_negative_rejected() {
    let btc = TickFilter::btc_usdt();
    for p in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -100.0, 0.0] {
        assert!(
            btc.to_ticks_checked(p).is_err(),
            "price {p} must fail conversion"
        );
    }
    assert_eq!(
        btc.to_ticks_checked(f64::NAN),
        Err(TickError::NonFinitePrice)
    );
    assert_eq!(
        btc.to_ticks_checked(-100.0),
        Err(TickError::NonPositivePrice)
    );
    for q in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
        assert!(btc.check_qty(q).is_err(), "quantity {q} must fail");
    }
    assert_eq!(btc.check_qty(f64::NAN), Err(TickError::NonFiniteQty));
    assert_eq!(btc.check_qty(-1.0), Err(TickError::NegativeQty));
    // Zero quantity is allowed: in a diff it removes the level.
    assert!(btc.check_qty(0.0).is_ok());

    // The unfiltered book rejects the same inputs: no symbol is
    // needed to know a price is NaN or a size is negative.
    let mut b = OrderBook::new();
    b.from_snapshot(&[(100.0, 5.0)], &[(100.1, 4.0)], 10);
    let before = (b.bid_levels(), b.ask_levels());
    let r = b.apply_diff(
        11,
        11,
        &[(f64::NAN, 1.0), (100.0, -1.0), (100.0, 7.0)],
        &[(f64::INFINITY, 1.0)],
    );
    assert_eq!(r, toxicflow::orderbook::SyncResult::Applied);
    assert_eq!(b.bid_levels(), vec![(100.0, 7.0)]);
    assert_eq!(b.ask_levels(), vec![(100.1, 4.0)]);
    assert_eq!(b.invalid_rejected(), 3);
    assert_ne!(before.0, b.bid_levels(), "valid update still applies");
}

#[test]
fn stage3_toxicflow_h5_unrepresentable_and_overflow_rejected() {
    // Nine decimal places cannot be stored at the 1e-8 scale, even
    // under the finest filter.
    assert_eq!(
        TickFilter::finest().to_ticks_checked(100.000000001),
        Err(TickError::UnrepresentablePrice)
    );
    assert_eq!(
        TickFilter::finest().to_ticks_checked(0.000000001),
        Err(TickError::UnrepresentablePrice)
    );
    // Beyond 2^53 the scaled value is no longer an exact integer.
    assert_eq!(
        TickFilter::btc_usdt().to_ticks_checked(1e10),
        Err(TickError::PriceOverflow)
    );
    // A quantity below one satoshi is unrepresentable the same way.
    assert_eq!(
        TickFilter::btc_usdt().check_qty(0.000000001),
        Err(TickError::UnrepresentableQty)
    );
}

#[test]
fn stage3_toxicflow_h5_lot_step_and_min_qty() {
    let btc = TickFilter::btc_usdt();
    assert_eq!(
        btc.check_qty(0.0005),
        Err(TickError::OffLotStep {
            qty: 0.0005,
            step_size: 0.001
        })
    );
    assert_eq!(
        btc.check_qty(0.0015),
        Err(TickError::OffLotStep {
            qty: 0.0015,
            step_size: 0.001
        })
    );
    assert!(btc.check_qty(0.001).is_ok());
    assert!(btc.check_qty(0.002).is_ok());
    // A low-price symbol with a 1.0 minimum: on-step but below min.
    let low = TickFilter::new(0.0001, 0.01, 1.0).expect("low-price fixture");
    assert_eq!(
        low.check_qty(0.5),
        Err(TickError::BelowMinQty {
            qty: 0.5,
            min_qty: 1.0
        })
    );
    assert!(low.check_qty(1.0).is_ok());
    assert_eq!(
        low.check_qty(1.005),
        Err(TickError::OffLotStep {
            qty: 1.005,
            step_size: 0.01
        })
    );

    // The old behaviour inserted a negative quantity from a diff as a
    // live level; now it is rejected and the book is unchanged.
    let mut b = OrderBook::new().with_tick_filter(TickFilter::btc_usdt());
    b.from_snapshot(&[(100.0, 5.0)], &[(100.1, 4.0)], 10);
    b.apply_diff(11, 11, &[(100.0, -2.0)], &[]);
    assert_eq!(b.bid_levels(), vec![(100.0, 5.0)]);
    assert_eq!(b.invalid_rejected(), 1);
    assert_eq!(b.last_rejection(), Some(TickError::NegativeQty));
}

#[test]
fn stage3_toxicflow_h5_valid_levels_still_apply() {
    // A valid tape builds the identical book with and without a
    // filter: validation must not change behaviour on valid input.
    let build = |filtered: bool| {
        let mut b = if filtered {
            OrderBook::new().with_tick_filter(TickFilter::btc_usdt())
        } else {
            OrderBook::new()
        };
        b.from_snapshot(
            &[(100.0, 5.0), (99.9, 3.0)],
            &[(100.1, 4.0), (100.2, 2.0)],
            10,
        );
        b.apply_diff(11, 11, &[(100.0, 7.0), (99.9, 0.0), (99.8, 1.0)], &[]);
        b.apply_diff(12, 12, &[], &[(100.1, 0.0), (100.3, 6.0)]);
        b
    };
    let plain = build(false);
    let filtered = build(true);
    assert_eq!(plain.bid_levels(), filtered.bid_levels());
    assert_eq!(plain.ask_levels(), filtered.ask_levels());
    assert_eq!(plain.last_id, filtered.last_id);
    assert_eq!(plain.validity(), filtered.validity());
    assert_eq!(filtered.invalid_rejected(), 0);
}

#[test]
fn stage3_toxicflow_h5_validity_metrics() {
    let mut b = OrderBook::new().with_tick_filter(TickFilter::btc_usdt());
    b.from_snapshot(
        &[(100.0, 5.0), (99.9, 3.0)],
        &[(100.1, 4.0), (100.2, 2.0)],
        10,
    );
    b.apply_diff(11, 11, &[(100.0, 7.0)], &[]);
    b.apply_diff(5, 9, &[(100.0, 99.0)], &[]);
    b.apply_diff(13, 13, &[(100.0, 99.0)], &[]);
    let v = b.validity();
    assert!(!v.crossed);
    assert!(!v.locked);
    assert!(!v.one_sided);
    assert!(!v.empty);
    assert_eq!(v.n_bid_levels, 2);
    assert_eq!(v.n_ask_levels, 2);
    // Spread 100.1 - 100.0 = 0.1 = 10,000,000 internal ticks.
    assert_eq!(v.spread_ticks, Some(10_000_000));
    assert_eq!(v.invalid_rejected, 0);
    assert_eq!(v.gaps_seen, 1);
    assert_eq!(v.stale_seen, 1);
    assert_eq!(v.applied_events, 1);
    // One id applied (11), one missed in the gap (12): coverage 1/2.
    assert!((v.id_coverage - 0.5).abs() < 1e-12);
    assert_eq!(v.last_rejection, None);
    assert!(!b.is_crossed(), "flag and metrics must agree");
}

#[test]
fn stage3_toxicflow_h5_validity_flags() {
    // Crossed and locked are distinct: a locked book is crossed, a
    // crossed book need not be locked.
    let mut crossed = OrderBook::new();
    crossed.from_snapshot(&[(100.2, 5.0)], &[(100.1, 4.0)], 1);
    assert!(crossed.validity().crossed);
    assert!(!crossed.validity().locked);
    assert_eq!(crossed.validity().spread_ticks, Some(-10_000_000));

    let mut locked = OrderBook::new();
    locked.from_snapshot(&[(100.0, 5.0)], &[(100.0, 4.0)], 1);
    assert!(locked.validity().crossed);
    assert!(locked.validity().locked);
    assert_eq!(locked.validity().spread_ticks, Some(0));

    let mut one_sided = OrderBook::new();
    one_sided.from_snapshot(&[(100.0, 5.0)], &[], 1);
    let v = one_sided.validity();
    assert!(v.one_sided);
    assert!(!v.crossed);
    assert_eq!(v.spread_ticks, None);

    let v = OrderBook::new().validity();
    assert!(v.empty);
    assert!(!v.crossed);
    assert_eq!(v.spread_ticks, None);
    // Nothing applied and nothing missed: coverage is 1.0 by default.
    assert!((v.id_coverage - 1.0).abs() < 1e-12);
    // BookValidity is a plain value: the same book reports the same.
    let _ = BookValidity {
        crossed: false,
        locked: false,
        one_sided: false,
        empty: true,
        n_bid_levels: 0,
        n_ask_levels: 0,
        spread_ticks: None,
        invalid_rejected: 0,
        gaps_seen: 0,
        stale_seen: 0,
        applied_events: 0,
        id_coverage: 1.0,
        last_rejection: None,
    };
}

#[test]
fn stage3_toxicflow_h5_filter_rejects_invalid_filter() {
    for (tick, step, min) in [
        (0.0, 0.001, 0.001),
        (-0.01, 0.001, 0.001),
        (f64::NAN, 0.001, 0.001),
        (0.01, 0.0, 0.001),
        (0.01, 0.001, -1.0),
        (0.000000001, 0.001, 0.001),
        (0.01, f64::INFINITY, 0.001),
    ] {
        assert_eq!(
            TickFilter::new(tick, step, min),
            Err(TickError::InvalidFilter),
            "filter ({tick}, {step}, {min}) must be rejected"
        );
    }
    assert!(TickFilter::new(0.01, 0.001, 0.001).is_ok());
    // Attaching a filter is visible on the book; a plain book has none.
    let b = OrderBook::new().with_tick_filter(TickFilter::btc_usdt());
    assert_eq!(b.tick_filter(), Some(TickFilter::btc_usdt()));
    assert_eq!(OrderBook::new().tick_filter(), None);
}

/// One valid reference event tape, the A1 tape: session start, one
/// snapshot, two depth updates, session end. Its replay digest was
/// recorded on the pre-change code and must be unchanged by the
/// validation: valid input behaves exactly as before.
fn reference_tape() -> Vec<JournalRecord> {
    vec![
        JournalRecord::SessionStart {
            symbol: "BTCUSDT".to_string(),
        },
        JournalRecord::Snapshot {
            last_update_id: 100,
            bids: vec![(100.0, 5.0), (99.9, 3.0)],
            asks: vec![(100.1, 4.0), (100.2, 2.0)],
        },
        JournalRecord::Depth {
            time_ms: 1_700_000_000_000,
            u_first: 101,
            u_last: 101,
            bids: vec![(100.0, 6.0)],
            asks: vec![],
        },
        JournalRecord::Depth {
            time_ms: 1_700_000_000_100,
            u_first: 102,
            u_last: 102,
            bids: vec![(100.0, 6.0)],
            asks: vec![],
        },
        JournalRecord::SessionEnd { records: 4 },
    ]
}

#[test]
fn stage3_toxicflow_h5_reducer_digest_unchanged() {
    let out = replay(&reference_tape());
    assert_eq!(out.digest, "17b16df5de9c6b92");
    assert_eq!(out.trades.len(), 0);
    assert_eq!(out.quotes.len(), 2);
    // Deterministic: a second replay of the same tape agrees.
    assert_eq!(replay(&reference_tape()).digest, out.digest);
}
