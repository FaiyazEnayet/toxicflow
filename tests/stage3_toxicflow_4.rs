//! Stage 3 toxicflow H4: `ofi_bars` and `build_dataset` require a valid
//! bar size, a nonzero prediction horizon, monotonic inputs, and never
//! cast an out-of-range negative trade index to `usize`.
//!
//! Invalid configuration or input returns a typed error instead of an
//! empty frame. Valid frames reproduce the archived reference values
//! below (captured from main at 15277ed77505a42f638ea00e4081c08bcf0c1dec
//! before the validation landed) within 1e-10, and prefix frames are
//! unchanged by appended input.

use toxicflow::metrics::{ofi_bars, ofi_events, BarError, Quote, Trade};
use toxicflow::predict::{build_dataset, DatasetError};

/// Deterministic 600 s session: one quote a second, mid drifting up
/// 0.01 a quote, spread alternating 0.2 / 0.1, one 0.1 trade a second
/// with alternating sides. Sixty 10 s bars, 54 dataset rows.
fn h4_quotes_trades() -> (Vec<Quote>, Vec<Trade>) {
    let mut quotes = Vec::new();
    let mut trades = Vec::new();
    for i in 0..600 {
        let t = i as i64 * 1000;
        let mid = 100.0 + 0.01 * i as f64;
        let spread = if i % 3 == 0 { 0.2 } else { 0.1 };
        let bq = 5.0 + ((i * 7) % 5) as f64;
        let aq = 4.0 + ((i * 11) % 4) as f64;
        quotes.push(Quote {
            time_ms: t,
            bid_px: mid - spread / 2.0,
            bid_sz: bq,
            ask_px: mid + spread / 2.0,
            ask_sz: aq,
        });
        trades.push(Trade {
            time_ms: t + 100,
            price: mid,
            qty: 0.1,
            is_buyer_maker: i % 2 == 0,
        });
    }
    (quotes, trades)
}

fn h4_quote(t: i64) -> Quote {
    Quote {
        time_ms: t,
        bid_px: 99.95,
        bid_sz: 6.0,
        ask_px: 100.05,
        ask_sz: 4.0,
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-10
}

#[test]
fn stage3_toxicflow_4_ofi_bars_rejects_zero_and_negative_bar_ms() {
    let qs = vec![h4_quote(0), h4_quote(10_000)];
    let e = ofi_events(&qs);
    assert_eq!(
        ofi_bars(&qs, &e, 0),
        Err(BarError::InvalidBarSize { bar_ms: 0 })
    );
    assert_eq!(
        ofi_bars(&qs, &e, -10_000),
        Err(BarError::InvalidBarSize { bar_ms: -10_000 })
    );
}

#[test]
fn stage3_toxicflow_4_ofi_bars_rejects_empty_input() {
    assert_eq!(ofi_bars(&[], &[], 10_000), Err(BarError::EmptyInput));
}

#[test]
fn stage3_toxicflow_4_ofi_bars_rejects_non_monotonic_quotes() {
    // A later quote earlier than the grid start used to make the bar
    // offset negative and cast it silently to `usize`, which wrapped
    // to a huge index and tried to allocate that many bars.
    let qs = vec![h4_quote(20_000), h4_quote(5_000)];
    let e = ofi_events(&qs);
    assert_eq!(
        ofi_bars(&qs, &e, 10_000),
        Err(BarError::NonMonotonic {
            index: 1,
            previous: 20_000,
            current: 5_000,
        })
    );
}

#[test]
fn stage3_toxicflow_4_ofi_bars_rejects_mismatched_events() {
    let qs = vec![h4_quote(0), h4_quote(10_000)];
    let e = vec![0.0];
    assert_eq!(
        ofi_bars(&qs, &e, 10_000),
        Err(BarError::MismatchedEvents {
            quotes: 2,
            events: 1,
        })
    );
}

#[test]
fn stage3_toxicflow_4_ofi_bars_valid_prefix_matches_reference() {
    let (quotes, _) = h4_quotes_trades();
    let events = ofi_events(&quotes);
    let bars = ofi_bars(&quotes, &events, 10_000).expect("valid fixture");
    assert_eq!(bars.len(), 60);
    let expected: [(i64, f64, f64, f64, f64); 3] = [
        (0, 41.0, 100.09, 0.2, 0.06666666666666667),
        (10_000, 39.0, 100.19, 0.1, 0.23076923076923078),
        (20_000, 48.0, 100.29, 0.1, 0.06666666666666667),
    ];
    for (got, want) in bars.iter().take(3).zip(expected.iter()) {
        assert_eq!(got.0, want.0);
        assert!(close(got.1, want.1), "ofi {} vs {}", got.1, want.1);
        assert!(close(got.2, want.2), "mid {} vs {}", got.2, want.2);
        assert!(close(got.3, want.3), "spread {} vs {}", got.3, want.3);
        assert!(close(got.4, want.4), "imbalance {} vs {}", got.4, want.4);
    }
    // Prefix invariance, exact: appending later quotes leaves the
    // already-emitted bars untouched.
    let prefix = ofi_bars(&quotes[..300], &events[..300], 10_000).expect("valid prefix");
    assert_eq!(prefix.len(), 30);
    assert_eq!(prefix, bars[..30].to_vec());
}

#[test]
fn stage3_toxicflow_4_build_dataset_rejects_zero_horizon() {
    let (quotes, trades) = h4_quotes_trades();
    assert_eq!(
        build_dataset(&quotes, &trades, 10_000, 0).unwrap_err(),
        DatasetError::InvalidHorizon { horizon: 0 }
    );
}

#[test]
fn stage3_toxicflow_4_build_dataset_rejects_empty_input_and_bad_bar() {
    let (quotes, trades) = h4_quotes_trades();
    assert_eq!(
        build_dataset(&[], &[], 10_000, 2).unwrap_err(),
        DatasetError::Bars(BarError::EmptyInput)
    );
    assert_eq!(
        build_dataset(&quotes, &trades, 0, 2).unwrap_err(),
        DatasetError::Bars(BarError::InvalidBarSize { bar_ms: 0 })
    );
}

#[test]
fn stage3_toxicflow_4_build_dataset_rejects_trade_before_first_bar() {
    // A trade earlier than the first bar start used to cast its
    // negative bar index silently to `usize` and vanish from the
    // frame with no trace.
    let mut quotes = Vec::new();
    for i in 0..500 {
        quotes.push(h4_quote(100_000 + i * 1000));
    }
    let trades = vec![Trade {
        time_ms: 90_000,
        price: 100.0,
        qty: 0.1,
        is_buyer_maker: false,
    }];
    assert_eq!(
        build_dataset(&quotes, &trades, 10_000, 2).unwrap_err(),
        DatasetError::TradeBeforeStart {
            index: 0,
            time_ms: 90_000,
            bar_start: 100_000,
        }
    );
}

#[test]
fn stage3_toxicflow_4_build_dataset_rejects_underflowing_trade_timestamp() {
    // time_ms - t0 underflows i64 here; checked arithmetic must
    // report it instead of wrapping.
    let mut quotes = Vec::new();
    for i in 0..500 {
        quotes.push(h4_quote(1_000_000 + i * 1000));
    }
    let trades = vec![Trade {
        time_ms: i64::MIN,
        price: 100.0,
        qty: 0.1,
        is_buyer_maker: false,
    }];
    assert_eq!(
        build_dataset(&quotes, &trades, 10_000, 2).unwrap_err(),
        DatasetError::TimestampOverflow { time_ms: i64::MIN }
    );
}

#[test]
fn stage3_toxicflow_4_build_dataset_rejects_non_monotonic_trades() {
    let (quotes, _) = h4_quotes_trades();
    let trades = vec![
        Trade {
            time_ms: 100,
            price: 100.0,
            qty: 0.1,
            is_buyer_maker: false,
        },
        Trade {
            time_ms: 50,
            price: 100.0,
            qty: 0.1,
            is_buyer_maker: false,
        },
    ];
    assert_eq!(
        build_dataset(&quotes, &trades, 10_000, 2).unwrap_err(),
        DatasetError::NonMonotonicTrades {
            index: 1,
            previous: 100,
            current: 50,
        }
    );
}

#[test]
fn stage3_toxicflow_4_build_dataset_thin_data_is_ok_empty_not_error() {
    // Thin data is not invalid input: fewer than 40 bars still
    // yields an empty frame with Ok, as before.
    let quotes = vec![h4_quote(0), h4_quote(10_000)];
    let ds = build_dataset(&quotes, &[], 10_000, 2).expect("thin data is valid");
    assert!(ds.is_empty());
}

#[test]
fn stage3_toxicflow_4_build_dataset_valid_prefix_matches_reference() {
    let (quotes, trades) = h4_quotes_trades();
    let ds = build_dataset(&quotes, &trades, 10_000, 2).expect("valid fixture");
    assert_eq!(ds.len(), 54);
    // Archived reference values (pre-change run at 15277ed), 1e-10.
    let expected: [(i64, f64, f64, f64, f64, f64, u8); 3] = [
        (
            40_000,
            35.0,
            -1.1942195516537522,
            0.06666666666666667,
            -0.8164965809110596,
            0.0,
            1,
        ),
        (
            50_000,
            52.0,
            1.6588485934905437,
            0.23076923076923078,
            -0.7071067811775476,
            0.0,
            1,
        ),
        (
            60_000,
            38.0,
            -0.6289432567831733,
            0.06666666666666667,
            1.1547005383559181,
            0.0,
            1,
        ),
    ];
    for (got, want) in ds.iter().take(3).zip(expected.iter()) {
        assert_eq!(got.time_ms, want.0);
        assert!(close(got.ofi, want.1), "ofi {} vs {}", got.ofi, want.1);
        assert!(
            close(got.ofi_z, want.2),
            "ofi_z {} vs {}",
            got.ofi_z,
            want.2
        );
        assert!(
            close(got.imbalance, want.3),
            "imbalance {} vs {}",
            got.imbalance,
            want.3
        );
        assert!(
            close(got.spread_z, want.4),
            "spread_z {} vs {}",
            got.spread_z,
            want.4
        );
        assert!(
            close(got.signed_vol, want.5),
            "signed_vol {} vs {}",
            got.signed_vol,
            want.5
        );
        assert_eq!(got.target, want.6);
    }
    // Prefix invariance, exact: the first 450 quotes (45 bars) give
    // the same first 39 rows as the full 600-quote frame. A shorter
    // prefix cannot be used here: fewer than 40 bars is thin data
    // and yields no rows at all.
    let ds_prefix = build_dataset(&quotes[..450], &trades[..450], 10_000, 2).expect("valid prefix");
    assert_eq!(ds_prefix.len(), 39);
    for (a, b) in ds_prefix.iter().zip(ds.iter()) {
        assert_eq!(a.time_ms, b.time_ms);
        assert_eq!(a.ofi, b.ofi);
        assert_eq!(a.ofi_z, b.ofi_z);
        assert_eq!(a.imbalance, b.imbalance);
        assert_eq!(a.spread_z, b.spread_z);
        assert_eq!(a.signed_vol, b.signed_vol);
        assert_eq!(a.target, b.target);
    }
}
