//! Toxicity-gated market-making DEMO.
//!
//! This is a toy, not a strategy. The fill model is naive on purpose:
//! each second we post `size` on both sides at the touch, and we assume we
//! get filled by incoming market orders up to our size. There is no queue
//! position, no latency, no fees. Its only purpose is to show the
//! *direction* of the effect: standing down when VPIN is high avoids
//! trading into informed flow in the simulator.
//!
//! The simulation is causal: the bucket volume is frozen from a leading
//! calibration window (never from whole-session volume), the quote posted
//! for second `i` is the last quote observed strictly before second `i`
//! starts, and the gate for second `i` uses only VPIN completed strictly
//! before second `i` starts. Appending future trades cannot change earlier
//! bucket thresholds, posted quotes, gate decisions or PnL.
//!
//! Do not trade this.

use std::collections::BTreeMap;

use crate::metrics::{vpin, volume_buckets_sized, Quote, Trade};

pub struct MmConfig {
    /// quote size in base units per side per second
    pub size: f64,
    /// stand down when trailing VPIN exceeds this
    pub vpin_threshold: f64,
    pub vpin_buckets: usize,
    pub vpin_window: usize,
    /// leading seconds used to size volume buckets; frozen afterwards
    pub calibration_secs: usize,
}

impl Default for MmConfig {
    fn default() -> Self {
        Self {
            size: 0.002,
            vpin_threshold: 0.7,
            vpin_buckets: 50,
            vpin_window: 20,
            calibration_secs: 300,
        }
    }
}

#[derive(Debug)]
pub struct MmReport {
    pub seconds: usize,
    pub seconds_gated: usize,
    pub pnl_open: f64,
    pub pnl_gated: f64,
    pub turnover: f64,
    pub inventory_end: f64,
}

/// Run the quoting simulation. Returns marked-to-mid PnL with the gate on
/// and off, in quote currency.
///
/// Causal discipline: the volume-bucket size is frozen from the leading
/// `calibration_secs` (later trades cannot move earlier bucket thresholds);
/// second `i` is quoted at the last quote observed strictly before second
/// `i` starts (second 0 is never quoted); the gate for second `i` reads
/// only VPIN completed strictly before second `i` starts.
pub fn run_mm(
    quotes: &[Quote],
    trades: &[Trade],
    cfg: &MmConfig,
) -> Option<MmReport> {
    if quotes.is_empty() || trades.is_empty() {
        return None;
    }
    // per-second bars: last touch + signed trade volume
    let t0 = quotes[0].time_ms / 1000 * 1000;
    let t1 = quotes.last().map(|q| q.time_ms).unwrap_or(t0);
    let n = ((t1 - t0) / 1000 + 1).max(1) as usize;
    if n < 30 {
        return None;
    }
    let mut sec_bid = vec![f64::NAN; n];
    let mut sec_ask = vec![f64::NAN; n];
    let mut buy_v = vec![0.0; n];
    let mut sell_v = vec![0.0; n];
    for q in quotes {
        let i = ((q.time_ms - t0) / 1000) as usize;
        if i < n {
            sec_bid[i] = q.bid_px;
            sec_ask[i] = q.ask_px;
        }
    }
    for t in trades {
        let i = ((t.time_ms - t0) / 1000) as usize;
        if i < n {
            if t.is_buyer_maker {
                sell_v[i] += t.qty;
            } else {
                buy_v[i] += t.qty;
            }
        }
    }
    // forward-fill missing quote seconds
    let mut lb = f64::NAN;
    let mut la = f64::NAN;
    for i in 0..n {
        if sec_bid[i].is_nan() {
            sec_bid[i] = lb;
            sec_ask[i] = la;
        } else {
            lb = sec_bid[i];
            la = sec_ask[i];
        }
    }
    // Bucket size from the calibration window only: frozen before the
    // session, so later trades cannot move earlier bucket thresholds.
    let calib_end = t0 + cfg.calibration_secs as i64 * 1000;
    let calib_vol: f64 = trades
        .iter()
        .filter(|t| t.time_ms < calib_end)
        .map(|t| t.qty)
        .sum();
    if calib_vol <= 0.0 {
        return None;
    }
    let buckets = volume_buckets_sized(trades, calib_vol / cfg.vpin_buckets as f64);
    // trailing VPIN sampled per second; a bucket completed at time t is
    // visible to the gate only for seconds starting strictly after t.
    let vp = vpin(&buckets, cfg.vpin_window);
    let mut vpin_by_sec = BTreeMap::new();
    for (t, v) in &vp {
        vpin_by_sec.insert(t / 1000, *v);
    }
    let vpin_at = |sec: usize| -> f64 {
        let key = (t0 / 1000 + sec as i64) as i64;
        vpin_by_sec.range(..key).next_back().map(|(_, v)| *v).unwrap_or(0.0)
    };

    let mut cash_open = 0.0;
    let mut cash_gated = 0.0;
    let mut inv_open = 0.0;
    let mut inv_gated = 0.0;
    let mut turnover = 0.0;
    let mut gated_secs = 0usize;
    for i in 0..n {
        // decision delay: second i quotes at the last quote seen strictly
        // before second i starts; second 0 has no prior quote and is skipped.
        if i == 0 || sec_bid[i - 1].is_nan() {
            continue;
        }
        let (bid, ask) = (sec_bid[i - 1], sec_ask[i - 1]);
        let fill_b = cfg.size.min(sell_v[i]); // we buy: pay the bid
        let fill_a = cfg.size.min(buy_v[i]); // we sell: earn the ask
        // ungated leg always quotes
        cash_open -= fill_b * bid;
        cash_open += fill_a * ask;
        inv_open += fill_b - fill_a;
        // gated leg stands down in toxic regimes
        if vpin_at(i) > cfg.vpin_threshold {
            gated_secs += 1;
            continue;
        }
        cash_gated -= fill_b * bid;
        cash_gated += fill_a * ask;
        inv_gated += fill_b - fill_a;
        turnover += (fill_b + fill_a) * (bid + ask) / 2.0;
    }
    let last_mid = (sec_bid[n - 1] + sec_ask[n - 1]) / 2.0;
    Some(MmReport {
        seconds: n,
        seconds_gated: gated_secs,
        pnl_open: cash_open + inv_open * last_mid,
        pnl_gated: cash_gated + inv_gated * last_mid,
        turnover,
        inventory_end: inv_gated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth(n: usize, toxic_half: bool) -> (Vec<Quote>, Vec<Trade>) {
        let mut quotes = Vec::new();
        let mut trades = Vec::new();
        let mut mid = 100.0;
        let mut s: u64 = 7;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s as f64) / (u64::MAX as f64)
        };
        for i in 0..n {
            let t = i as i64 * 1000;
            // second half: one-sided informed buying pushes price up
            let informed = toxic_half && i > n / 2;
            let dir = if informed { 1.0 } else { 0.0 };
            quotes.push(Quote {
                time_ms: t,
                bid_px: mid - 0.05,
                bid_sz: 5.0,
                ask_px: mid + 0.05,
                ask_sz: 5.0,
            });
            for k in 0..4 {
                let buy = rnd() < 0.5 + 0.45 * dir;
                trades.push(Trade {
                    time_ms: t + k * 200,
                    price: mid,
                    qty: 0.05,
                    is_buyer_maker: !buy,
                });
            }
            mid += dir * 0.05 + (rnd() - 0.5) * 0.02;
        }
        (quotes, trades)
    }

    #[test]
    fn gate_avoids_toxic_half() {
        let (q, t) = synth(600, true);
        let cfg = MmConfig { vpin_threshold: 0.55, ..MmConfig::default() };
        let r = run_mm(&q, &t, &cfg).expect("sim failed");
        assert!(r.seconds_gated > 0, "gate never triggered");
        // informed buying lifts price: quoting into it loses vs standing down
        assert!(
            r.pnl_gated > r.pnl_open,
            "gated {:.4} should beat open {:.4}",
            r.pnl_gated,
            r.pnl_open
        );
    }

    #[test]
    fn no_gate_triggers_on_clean_flow() {
        let (q, t) = synth(600, false);
        let cfg = MmConfig { vpin_threshold: 0.95, ..MmConfig::default() };
        let r = run_mm(&q, &t, &cfg).expect("sim failed");
        assert_eq!(r.seconds_gated, 0);
        assert!((r.pnl_gated - r.pnl_open).abs() < 1e-9);
    }

    fn quote_at(t: i64, bid: f64, ask: f64) -> Quote {
        Quote { time_ms: t, bid_px: bid, bid_sz: 5.0, ask_px: ask, ask_sz: 5.0 }
    }

    #[test]
    fn hand_computed_two_trade_fixture() {
        // Decision delay, hand-computed: second 0 is never quoted (no prior
        // quote); second 1 fills at second 0's quote (bid 99 / ask 101),
        // not at the quote printed inside second 1 (bid 199 / ask 201).
        // Trade A (second 0) is never filled; trade B (second 1, buyer
        // takes) sells 0.01 at 101. Mark at the last mid, 110.
        let quotes = vec![
            quote_at(0, 99.0, 101.0),
            quote_at(1_200, 199.0, 201.0),
            quote_at(60_000, 109.0, 111.0),
        ];
        let trades = vec![
            Trade { time_ms: 500, price: 100.0, qty: 0.01, is_buyer_maker: true },
            Trade { time_ms: 1_500, price: 100.0, qty: 0.01, is_buyer_maker: false },
        ];
        let cfg = MmConfig {
            size: 0.01,
            vpin_threshold: 2.0,
            calibration_secs: 30,
            ..MmConfig::default()
        };
        let r = run_mm(&quotes, &trades, &cfg).expect("sim failed");
        assert_eq!(r.seconds, 61);
        assert_eq!(r.seconds_gated, 0);
        assert!((r.inventory_end + 0.01).abs() < 1e-9, "inv {}", r.inventory_end);
        assert!((r.turnover - 1.0).abs() < 1e-9, "turnover {}", r.turnover);
        // cash +1.01 from the fill, marked short 0.01 at 110
        assert!((r.pnl_open + 0.09).abs() < 1e-9, "pnl_open {}", r.pnl_open);
        assert!((r.pnl_gated + 0.09).abs() < 1e-9, "pnl_gated {}", r.pnl_gated);
    }

    #[test]
    fn appended_future_trades_do_not_change_history() {
        // Trades appended after the last quote extend no simulated second
        // and sit outside the calibration window: bucket thresholds,
        // posted quotes, gate decisions and PnL must be bit-identical.
        let (q, t) = synth(600, true);
        let mut t2 = t.clone();
        for i in 0..2400 {
            t2.push(Trade {
                time_ms: 600_000 + i as i64 * 250,
                price: 120.0,
                qty: 0.05,
                is_buyer_maker: i % 2 == 0,
            });
        }
        let cfg = MmConfig { vpin_threshold: 0.55, ..MmConfig::default() };
        let r1 = run_mm(&q, &t, &cfg).expect("sim failed");
        let r2 = run_mm(&q, &t2, &cfg).expect("sim failed");
        assert_eq!(r1.seconds, r2.seconds);
        assert_eq!(r1.seconds_gated, r2.seconds_gated);
        assert!((r1.pnl_open - r2.pnl_open).abs() < 1e-12, "{} vs {}", r1.pnl_open, r2.pnl_open);
        assert!((r1.pnl_gated - r2.pnl_gated).abs() < 1e-12, "{} vs {}", r1.pnl_gated, r2.pnl_gated);
        assert!((r1.turnover - r2.turnover).abs() < 1e-12);
        assert!((r1.inventory_end - r2.inventory_end).abs() < 1e-12);
    }
}
