//! Toxicity-gated market-making DEMO.
//!
//! This is a toy, not a strategy. The fill model is naive on purpose:
//! each second we post `size` on both sides at the touch, and we assume we
//! get filled by incoming market orders up to our size. There is no queue
//! position, no latency, no fees. Its only purpose is to show the
//! *direction* of the effect: standing down when VPIN is high avoids
//! trading into informed flow in the simulator.
//!
//! Do not trade this.

use std::collections::BTreeMap;

use crate::metrics::{vpin, volume_buckets, Quote, Trade};

pub struct MmConfig {
    /// quote size in base units per side per second
    pub size: f64,
    /// stand down when trailing VPIN exceeds this
    pub vpin_threshold: f64,
    pub vpin_buckets: usize,
    pub vpin_window: usize,
}

impl Default for MmConfig {
    fn default() -> Self {
        Self {
            size: 0.002,
            vpin_threshold: 0.7,
            vpin_buckets: 50,
            vpin_window: 20,
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
    let mut bid = vec![f64::NAN; n];
    let mut ask = vec![f64::NAN; n];
    let mut buy_v = vec![0.0; n];
    let mut sell_v = vec![0.0; n];
    for q in quotes {
        let i = ((q.time_ms - t0) / 1000) as usize;
        if i < n {
            bid[i] = q.bid_px;
            ask[i] = q.ask_px;
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
        if bid[i].is_nan() {
            bid[i] = lb;
            ask[i] = la;
        } else {
            lb = bid[i];
            la = ask[i];
        }
    }
    // trailing VPIN sampled per second
    let buckets = volume_buckets(trades, cfg.vpin_buckets);
    let vp = vpin(&buckets, cfg.vpin_window);
    let mut vpin_by_sec = BTreeMap::new();
    for (t, v) in &vp {
        vpin_by_sec.insert(t / 1000, *v);
    }
    let vpin_at = |sec: usize| -> f64 {
        let key = (t0 / 1000 + sec as i64) as i64;
        vpin_by_sec.range(..=key).next_back().map(|(_, v)| *v).unwrap_or(0.0)
    };

    let mut cash_open = 0.0;
    let mut cash_gated = 0.0;
    let mut inv_open = 0.0;
    let mut inv_gated = 0.0;
    let mut turnover = 0.0;
    let mut gated_secs = 0usize;
    for i in 0..n {
        if bid[i].is_nan() {
            continue;
        }
        let fill_b = cfg.size.min(sell_v[i]); // we buy: pay the bid
        let fill_a = cfg.size.min(buy_v[i]); // we sell: earn the ask
        // ungated leg always quotes
        cash_open -= fill_b * bid[i];
        cash_open += fill_a * ask[i];
        inv_open += fill_b - fill_a;
        // gated leg stands down in toxic regimes
        if vpin_at(i) > cfg.vpin_threshold {
            gated_secs += 1;
            continue;
        }
        cash_gated -= fill_b * bid[i];
        cash_gated += fill_a * ask[i];
        inv_gated += fill_b - fill_a;
        turnover += (fill_b + fill_a) * (bid[i] + ask[i]) / 2.0;
    }
    let last_mid = (bid[n - 1] + ask[n - 1]) / 2.0;
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
}
