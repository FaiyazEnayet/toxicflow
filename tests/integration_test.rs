//! End-to-end pipeline on fully synthetic market data.

use toxicflow::metrics::{
    kyle_lambda, ofi_bars, ofi_events, vpin, volume_buckets, FlowBar, Quote, Trade,
};
use toxicflow::orderbook::{OrderBook, SyncResult};
use toxicflow::{predict, strategy};

fn rng_state() -> u64 {
    12345
}
fn rnd(s: &mut u64) -> f64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    (*s as f64) / (u64::MAX as f64)
}

/// Synthetic session: first half balanced noise, second half one-sided
/// informed buying that pushes the price up.
fn synth_session(n_secs: usize) -> (Vec<Quote>, Vec<Trade>) {
    let mut s = rng_state();
    let mut quotes = Vec::new();
    let mut trades = Vec::new();
    let mut mid = 100.0;
    for i in 0..n_secs {
        let t = i as i64 * 1000;
        let toxic = i > n_secs / 2;
        quotes.push(Quote {
            time_ms: t,
            bid_px: mid - 0.05,
            bid_sz: 5.0,
            ask_px: mid + 0.05,
            ask_sz: 5.0,
        });
        for k in 0..4 {
            let buy = rnd(&mut s) < if toxic { 0.9 } else { 0.5 };
            trades.push(Trade {
                time_ms: t + k as i64 * 200,
                price: mid,
                qty: 0.05,
                is_buyer_maker: !buy,
            });
        }
        mid += if toxic { 0.04 } else { 0.0 } + (rnd(&mut s) - 0.5) * 0.02;
    }
    (quotes, trades)
}

#[test]
fn book_sync_end_to_end() {
    let mut b = OrderBook::new();
    b.from_snapshot(&[(100.0, 5.0)], &[(100.1, 4.0)], 100);
    assert_eq!(b.apply_diff(101, 102, &[(100.0, 6.0)], &[]), SyncResult::Applied);
    assert_eq!(b.apply_diff(103, 105, &[], &[(100.1, 0.0)]), SyncResult::Applied);
    assert_eq!(b.best_ask(), None); // removed
    assert_eq!(b.apply_diff(200, 201, &[], &[]), SyncResult::Gap);
}

#[test]
fn full_pipeline_toxic_second_half() {
    let (quotes, trades) = synth_session(1200);
    assert_eq!(trades.len(), 4800);

    // VPIN rises in the toxic half
    let bk = volume_buckets(&trades, 50);
    assert_eq!(bk.len(), 50);
    let vp = vpin(&bk, 20);
    let first_half: f64 = vp[..10].iter().map(|(_, v)| v).sum::<f64>() / 10.0;
    let last_half: f64 = vp[vp.len() - 10..].iter().map(|(_, v)| v).sum::<f64>() / 10.0;
    assert!(last_half > first_half, "{last_half} vs {first_half}");

    // OFI bars exist and Kyle's lambda is computable
    let ev = ofi_events(&quotes);
    let bars = ofi_bars(&quotes, &ev, 10_000);
    assert_eq!(bars.len(), 120);
    let fb: Vec<FlowBar> = bars
        .windows(2)
        .map(|w| FlowBar {
            time_ms: w[1].0,
            signed_vol: 0.1,
            dmid: w[1].2 - w[0].2,
            gap_before: false,
        })
        .collect();
    let lam = kyle_lambda(&fb, 30);
    assert!(lam.iter().any(|x| x.is_some()));

    // prediction runs end to end and returns a sane report
    // (directional edge is covered by the unit tests on cleaner synthetics)
    let ds = predict::build_dataset(&quotes, &trades, 10_000, 2);
    let rep = predict::walk_forward(&ds, 3, 2).expect("eval");
    assert!((0.3..=1.0).contains(&rep.accuracy));
    assert!((0.0..=1.0).contains(&rep.baseline));
    assert!(!rep.calibration.is_empty());

    // gated MM avoids the toxic half
    let cfg = strategy::MmConfig {
        vpin_threshold: 0.55,
        ..Default::default()
    };
    let r = strategy::run_mm(&quotes, &trades, &cfg).expect("mm");
    assert!(r.seconds_gated > 0);
    assert!(r.pnl_gated > r.pnl_open);
}
