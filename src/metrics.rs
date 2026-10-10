//! Toxicity metrics as pure functions.
//!
//! VPIN: Easley, Lopez de Prado, O'Hara. Trades are grouped into
//! equal-volume buckets; each bucket's imbalance is |buy - sell| / V; VPIN
//! is the rolling mean over a window of buckets. Direction comes from the
//! exchange's own aggressor flag, so no tick rule is needed.
//!
//! Kyle's lambda: rolling OLS slope of midprice changes on signed order
//! flow (Glosten-Harris/Hasbrouck style price impact).
//!
//! OFI: Cont, Kukanov, Stoikov (2014). From L2 quote changes at the touch:
//!   e = 1(bid up)*bid_sz - 1(bid down)*prev_bid_sz
//!       - 1(ask down)*ask_sz + 1(ask up)*prev_ask_sz

/// One aggregated trade.
#[derive(Debug, Clone, Copy)]
pub struct Trade {
    pub time_ms: i64,
    pub price: f64,
    pub qty: f64,
    /// true = the buyer was the maker, i.e. seller-initiated
    pub is_buyer_maker: bool,
}

impl Trade {
    pub fn price(&self) -> f64 {
        self.price
    }
    /// +qty for buyer-initiated, -qty for seller-initiated.
    pub fn signed_qty(&self) -> f64 {
        if self.is_buyer_maker {
            -self.qty
        } else {
            self.qty
        }
    }
}

/// One touch quote (best bid/ask with sizes).
#[derive(Debug, Clone, Copy)]
pub struct Quote {
    pub time_ms: i64,
    pub bid_px: f64,
    pub bid_sz: f64,
    pub ask_px: f64,
    pub ask_sz: f64,
}

impl Quote {
    pub fn mid(&self) -> f64 {
        (self.bid_px + self.ask_px) / 2.0
    }
    pub fn spread(&self) -> f64 {
        self.ask_px - self.bid_px
    }
    pub fn imbalance(&self) -> f64 {
        let tot = self.bid_sz + self.ask_sz;
        if tot == 0.0 {
            0.0
        } else {
            (self.bid_sz - self.ask_sz) / tot
        }
    }
}

/// One equal-volume bucket of trade flow.
#[derive(Debug, Clone, Copy)]
pub struct Bucket {
    pub time_ms: i64,
    pub buy_vol: f64,
    pub sell_vol: f64,
    pub imbalance: f64,
}

/// Split signed trade flow into buckets of exactly `bucket_vol` base units.
/// A trade straddling a boundary is split proportionally. A trailing
/// partially-filled bucket is dropped.
pub fn volume_buckets_sized(trades: &[Trade], bucket_vol: f64) -> Vec<Bucket> {
    if trades.is_empty() || bucket_vol <= 0.0 {
        return Vec::new();
    }
    let mut sorted: Vec<&Trade> = trades.iter().collect();
    sorted.sort_by_key(|t| t.time_ms);
    let v = bucket_vol;

    let mut out = Vec::new();
    let (mut buy, mut sell, mut filled) = (0.0, 0.0, 0.0);
    let mut end_time = 0i64;
    for t in sorted {
        let (mut rb, mut rs) = if t.is_buyer_maker {
            (0.0, t.qty)
        } else {
            (t.qty, 0.0)
        };
        while rb + rs > 1e-12 {
            let take = (v - filled).min(rb + rs);
            let frac = take / (rb + rs);
            buy += rb * frac;
            sell += rs * frac;
            filled += take;
            rb *= 1.0 - frac;
            rs *= 1.0 - frac;
            end_time = end_time.max(t.time_ms);
            if filled >= v - 1e-9 {
                out.push(Bucket {
                    time_ms: end_time,
                    buy_vol: buy,
                    sell_vol: sell,
                    imbalance: (buy - sell).abs() / v,
                });
                buy = 0.0;
                sell = 0.0;
                filled = 0.0;
            }
        }
    }
    out
}

/// Split signed trade flow into `n_buckets` equal-volume buckets.
/// A trade straddling a boundary is split proportionally. A trailing
/// partially-filled bucket is dropped.
pub fn volume_buckets(trades: &[Trade], n_buckets: usize) -> Vec<Bucket> {
    if trades.is_empty() || n_buckets == 0 {
        return Vec::new();
    }
    let total: f64 = trades.iter().map(|t| t.qty).sum();
    if total <= 0.0 {
        return Vec::new();
    }
    volume_buckets_sized(trades, total / n_buckets as f64)
}

/// VPIN: rolling mean of bucket imbalance over `window` buckets.
/// Returns (bucket end time, vpin).
pub fn vpin(buckets: &[Bucket], window: usize) -> Vec<(i64, f64)> {
    let w = window.max(1);
    buckets
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let start = i.saturating_sub(w - 1);
            let n = (i - start + 1) as f64;
            let sum: f64 = buckets[start..=i].iter().map(|x| x.imbalance).sum();
            (b.time_ms, sum / n)
        })
        .collect()
}

/// One bar of signed flow and the resulting midprice change.
#[derive(Debug, Clone, Copy)]
pub struct FlowBar {
    pub time_ms: i64,
    pub signed_vol: f64,
    pub dmid: f64,
}

/// Rolling OLS slope of dmid on signed_vol: lambda and R^2 per bar.
/// `None` where the window is too short or flow variance is zero.
pub fn kyle_lambda(bars: &[FlowBar], window: usize) -> Vec<Option<(f64, f64)>> {
    bars.iter()
        .enumerate()
        .map(|(i, _)| {
            let start = i.saturating_sub(window - 1);
            let xs: Vec<f64> = bars[start..=i].iter().map(|b| b.signed_vol).collect();
            let ys: Vec<f64> = bars[start..=i].iter().map(|b| b.dmid).collect();
            let n = xs.len();
            if n < 10 {
                return None;
            }
            let mx = xs.iter().sum::<f64>() / n as f64;
            let my = ys.iter().sum::<f64>() / n as f64;
            let mut sxx = 0.0;
            let mut sxy = 0.0;
            let mut syy = 0.0;
            for k in 0..n {
                let dx = xs[k] - mx;
                let dy = ys[k] - my;
                sxx += dx * dx;
                sxy += dx * dy;
                syy += dy * dy;
            }
            if sxx <= 0.0 || syy <= 0.0 {
                return None;
            }
            let lambda = sxy / sxx;
            let r2 = (sxy * sxy) / (sxx * syy);
            Some((lambda, r2))
        })
        .collect()
}

/// Per-event OFI contribution, Cont-Kukanov-Stoikov (2014).
/// First event contributes 0 (no previous quote).
pub fn ofi_events(quotes: &[Quote]) -> Vec<f64> {
    let mut out = Vec::with_capacity(quotes.len());
    for (i, q) in quotes.iter().enumerate() {
        if i == 0 {
            out.push(0.0);
            continue;
        }
        let p = &quotes[i - 1];
        let mut e = 0.0;
        if q.bid_px > p.bid_px {
            e += q.bid_sz;
        } else if q.bid_px < p.bid_px {
            e -= p.bid_sz;
        } else {
            e += q.bid_sz - p.bid_sz;
        }
        if q.ask_px < p.ask_px {
            e -= q.ask_sz;
        } else if q.ask_px > p.ask_px {
            e += p.ask_sz;
        } else {
            e -= q.ask_sz - p.ask_sz;
        }
        out.push(e);
    }
    out
}

/// Aggregate per-event OFI into fixed-length bars (bar_ms milliseconds).
/// Returns (bar start ms, ofi sum, last mid, last spread, last imbalance).
pub fn ofi_bars(quotes: &[Quote], events: &[f64], bar_ms: i64) -> Vec<(i64, f64, f64, f64, f64)> {
    if quotes.is_empty() || bar_ms <= 0 {
        return Vec::new();
    }
    let t0 = quotes[0].time_ms / bar_ms * bar_ms;
    let mut bars: Vec<(i64, f64, f64, f64, f64)> = Vec::new();
    for (q, e) in quotes.iter().zip(events.iter()) {
        let b = q.time_ms / bar_ms * bar_ms - t0;
        let idx = b as usize / bar_ms as usize;
        while bars.len() <= idx {
            bars.push((t0 + bars.len() as i64 * bar_ms, 0.0, q.mid(), q.spread(), q.imbalance()));
        }
        let row = &mut bars[idx];
        row.1 += e;
        row.2 = q.mid();
        row.3 = q.spread();
        row.4 = q.imbalance();
    }
    bars
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_sided(n: usize) -> Vec<Trade> {
        (0..n)
            .map(|i| Trade { time_ms: i as i64, price: 0.0, qty: 1.0, is_buyer_maker: false })
            .collect()
    }

    fn balanced(n: usize) -> Vec<Trade> {
        (0..n)
            .map(|i| Trade {
                time_ms: i as i64,
                price: 0.0,
                qty: 1.0,
                is_buyer_maker: i % 2 == 0,
            })
            .collect()
    }

    #[test]
    fn vpin_high_on_toxic_flow() {
        let toxic = vpin(&volume_buckets(&one_sided(1000), 50), 50);
        let clean = vpin(&volume_buckets(&balanced(1000), 50), 50);
        let vt = toxic.last().unwrap().1;
        let vc = clean.last().unwrap().1;
        assert!((vt - 1.0).abs() < 1e-9, "one-sided flow must give VPIN 1, got {vt}");
        assert!(vc < 0.3, "balanced flow must give low VPIN, got {vc}");
        assert!(vt > vc);
    }

    #[test]
    fn vpin_partial_window() {
        let b = volume_buckets(&one_sided(100), 10);
        let v = vpin(&b, 50);
        assert_eq!(v.len(), 10);
        assert!((v[0].1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn vpin_empty() {
        assert!(volume_buckets(&[], 50).is_empty());
        assert!(vpin(&[], 50).is_empty());
    }

    #[test]
    fn bucket_boundary_split() {
        // 3 trades of qty 1, 2 buckets of V=1.5: first bucket gets 1.5
        let t: Vec<Trade> = (0..3)
            .map(|i| Trade { time_ms: i, price: 0.0, qty: 1.0, is_buyer_maker: false })
            .collect();
        let b = volume_buckets(&t, 2);
        assert_eq!(b.len(), 2);
        assert!((b[0].buy_vol - 1.5).abs() < 1e-9);
        assert!((b[1].buy_vol - 1.5).abs() < 1e-9);
    }

    #[test]
    fn kyle_lambda_sign_and_scale() {
        // dmid = 2 * signed_vol exactly
        let bars: Vec<FlowBar> = (0..60)
            .map(|i| {
                let x = (i as f64 - 30.0) / 10.0;
                FlowBar { time_ms: i, signed_vol: x, dmid: 2.0 * x }
            })
            .collect();
        let lam = kyle_lambda(&bars, 60);
        let (l, r2) = lam.last().unwrap().unwrap();
        assert!((l - 2.0).abs() < 1e-9, "got {l}");
        assert!((r2 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn kyle_lambda_none_on_flat_flow() {
        let bars: Vec<FlowBar> = (0..60)
            .map(|i| FlowBar { time_ms: i, signed_vol: 1.0, dmid: i as f64 * 0.01 })
            .collect();
        assert!(kyle_lambda(&bars, 60).last().unwrap().is_none());
    }

    fn q(t: i64, bp: f64, bq: f64, ap: f64, aq: f64) -> Quote {
        Quote { time_ms: t, bid_px: bp, bid_sz: bq, ask_px: ap, ask_sz: aq }
    }

    #[test]
    fn ofi_known_sequence() {
        let qs = vec![
            q(0, 100.0, 5.0, 100.1, 4.0),
            q(1, 100.1, 3.0, 100.1, 4.0), // bid up -> +3.0
            q(2, 100.1, 3.0, 100.0, 6.0), // ask down -> -6.0
            q(3, 100.1, 4.0, 100.0, 6.0), // bid same -> +1.0
        ];
        let e = ofi_events(&qs);
        assert_eq!(e[0], 0.0);
        assert!((e[1] - 3.0).abs() < 1e-9);
        assert!((e[2] + 6.0).abs() < 1e-9);
        assert!((e[3] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ofi_bars_aggregate() {
        let qs: Vec<Quote> = (0..100)
            .map(|i| q(i * 1000, 100.0, 5.0, 100.1, 4.0))
            .collect();
        let e = ofi_events(&qs);
        let bars = ofi_bars(&qs, &e, 10_000);
        assert_eq!(bars.len(), 10);
        assert!(bars.iter().all(|b| b.1.abs() < 1e-9));
    }
}
