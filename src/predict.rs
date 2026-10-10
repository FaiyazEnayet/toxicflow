//! Short-horizon return prediction from order-flow features.
//!
//! L2 logistic regression implemented by hand (full-batch gradient
//! descent). Features per bar: OFI, rolling z-scored OFI, touch imbalance,
//! spread, signed trade volume. Target: sign of the midprice move over the
//! next `horizon` bars.
//!
//! Evaluation is walk-forward with a purge gap: training bars always end at
//! least `horizon` bars before the test block starts, so no target window
//! overlaps the training set. Time series are never shuffled. We report
//! accuracy against a majority-class baseline, the Brier score, and a
//! calibration table (predicted-probability deciles vs realized frequency).

use crate::metrics::{ofi_bars, ofi_events, Quote, Trade};

pub const N_FEATURES: usize = 5;

#[derive(Debug, Clone)]
pub struct Bar {
    pub ofi: f64,
    pub ofi_z: f64,
    pub imbalance: f64,
    pub spread_z: f64,
    pub signed_vol: f64,
    pub target: u8,
}

impl Bar {
    fn features(&self) -> [f64; N_FEATURES] {
        [self.ofi, self.ofi_z, self.imbalance, self.spread_z, self.signed_vol]
    }
}

fn rolling_mean_sd(xs: &[f64], window: usize) -> (Vec<f64>, Vec<f64>) {
    let n = xs.len();
    let mut mean = vec![f64::NAN; n];
    let mut sd = vec![f64::NAN; n];
    let mut sum = 0.0;
    let mut sum2 = 0.0;
    for i in 0..n {
        sum += xs[i];
        sum2 += xs[i] * xs[i];
        if i >= window {
            sum -= xs[i - window];
            sum2 -= xs[i - window] * xs[i - window];
        }
        let k = (i + 1).min(window) as f64;
        if k >= 5.0 {
            let m = sum / k;
            let v = (sum2 / k - m * m).max(0.0);
            mean[i] = m;
            sd[i] = v.sqrt();
        }
    }
    (mean, sd)
}

/// Build the feature/target frame. Returns empty vec when data is thin.
pub fn build_dataset(
    quotes: &[Quote],
    trades: &[Trade],
    bar_ms: i64,
    horizon: usize,
) -> Vec<Bar> {
    let events = ofi_events(quotes);
    let bars = ofi_bars(quotes, &events, bar_ms);
    if bars.len() < 40 {
        return Vec::new();
    }
    let n = bars.len();
    let ofi: Vec<f64> = bars.iter().map(|b| b.1).collect();
    let mids: Vec<f64> = bars.iter().map(|b| b.2).collect();
    let spreads: Vec<f64> = bars.iter().map(|b| b.3).collect();
    let imbs: Vec<f64> = bars.iter().map(|b| b.4).collect();

    // signed trade volume per bar
    let t0 = bars[0].0;
    let mut signed = vec![0.0; n];
    for t in trades {
        let idx = ((t.time_ms - t0) / bar_ms) as usize;
        if idx < n {
            signed[idx] += t.signed_qty();
        }
    }

    let (ofi_m, ofi_s) = rolling_mean_sd(&ofi, 30);
    let (sp_m, sp_s) = rolling_mean_sd(&spreads, 30);
    let mut out = Vec::new();
    for i in 0..n {
        if i + horizon >= n || ofi_s[i].is_nan() || sp_s[i].is_nan() {
            continue;
        }
        let fwd = mids[i + horizon] / mids[i] - 1.0;
        out.push(Bar {
            ofi: ofi[i],
            ofi_z: (ofi[i] - ofi_m[i]) / (ofi_s[i] + 1e-12),
            imbalance: imbs[i],
            spread_z: (spreads[i] - sp_m[i]) / (sp_s[i] + 1e-12),
            signed_vol: signed[i],
            target: u8::from(fwd > 0.0),
        });
    }
    out
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z.clamp(-30.0, 30.0)).exp())
}

/// Full-batch gradient descent for L2 logistic regression.
/// Returns weights with the intercept appended last.
pub fn fit_logistic(
    xs: &[[f64; N_FEATURES]],
    y: &[u8],
    l2: f64,
    iters: usize,
    lr: f64,
) -> [f64; N_FEATURES + 1] {
    let n = xs.len() as f64;
    let d = N_FEATURES;
    let mut mu = [0.0; N_FEATURES];
    let mut sd = [0.0; N_FEATURES];
    for row in xs {
        for j in 0..d {
            mu[j] += row[j];
        }
    }
    for j in 0..d {
        mu[j] /= n;
    }
    for row in xs {
        for j in 0..d {
            let z = row[j] - mu[j];
            sd[j] += z * z;
        }
    }
    for j in 0..d {
        sd[j] = (sd[j] / n).sqrt() + 1e-12;
    }
    let mut w = [0.0; N_FEATURES + 1];
    for _ in 0..iters {
        let mut grad = [0.0; N_FEATURES + 1];
        for (row, &yi) in xs.iter().zip(y.iter()) {
            let mut z = w[d];
            for j in 0..d {
                z += w[j] * (row[j] - mu[j]) / sd[j];
            }
            let err = sigmoid(z) - yi as f64;
            for j in 0..d {
                grad[j] += err * (row[j] - mu[j]) / sd[j];
            }
            grad[d] += err;
        }
        for j in 0..d {
            w[j] -= lr * (grad[j] / n + l2 * w[j] / n);
        }
        w[d] -= lr * grad[d] / n;
    }
    // fold standardization back into raw-space weights
    let mut raw = [0.0; N_FEATURES + 1];
    let mut b = w[d];
    for j in 0..d {
        raw[j] = w[j] / sd[j];
        b -= w[j] * mu[j] / sd[j];
    }
    raw[d] = b;
    raw
}

pub fn predict_proba(x: &[f64; N_FEATURES], w: &[f64; N_FEATURES + 1]) -> f64 {
    let mut z = w[N_FEATURES];
    for j in 0..N_FEATURES {
        z += w[j] * x[j];
    }
    sigmoid(z)
}

#[derive(Debug)]
pub struct EvalReport {
    pub n_bars: usize,
    pub n_folds: usize,
    pub accuracy: f64,
    pub accuracy_std: f64,
    pub baseline: f64,
    pub brier: f64,
    /// (mean predicted prob, realized rate, n) per decile
    pub calibration: Vec<(f64, f64, usize)>,
}

/// Purged walk-forward evaluation.
pub fn walk_forward(bars: &[Bar], n_splits: usize, horizon: usize) -> Option<EvalReport> {
    let n = bars.len();
    if n < 60 {
        return None;
    }
    let fold = n / (n_splits + 1);
    if fold < 10 {
        return None;
    }
    let mut accs = Vec::new();
    let mut briers = Vec::new();
    let mut probs_all = Vec::new();
    let mut y_all = Vec::new();
    for k in 1..=n_splits {
        let a = k * fold;
        let b = ((k + 1) * fold).min(n);
        let train_end = a.saturating_sub(horizon); // purge gap
        if train_end < 30 || b - a < 10 {
            continue;
        }
        let xs: Vec<[f64; N_FEATURES]> = bars[..train_end].iter().map(|x| x.features()).collect();
        let ys: Vec<u8> = bars[..train_end].iter().map(|x| x.target).collect();
        let w = fit_logistic(&xs, &ys, 1.0, 2000, 0.5);
        for bar in &bars[a..b] {
            let p = predict_proba(&bar.features(), &w);
            let pred = u8::from(p > 0.5);
            accs.push(f64::from(pred == bar.target));
            briers.push((p - bar.target as f64).powi(2));
            probs_all.push(p);
            y_all.push(bar.target);
        }
    }
    if accs.is_empty() {
        return None;
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let acc = mean(&accs);
    let std = (accs.iter().map(|x| (x - acc).powi(2)).sum::<f64>() / accs.len() as f64).sqrt();
    let pos_rate = y_all.iter().map(|&y| y as f64).sum::<f64>() / y_all.len() as f64;
    let baseline = pos_rate.max(1.0 - pos_rate);

    let mut order: Vec<usize> = (0..probs_all.len()).collect();
    order.sort_by(|&i, &j| probs_all[i].partial_cmp(&probs_all[j]).unwrap());
    let mut calibration = Vec::new();
    for chunk in order.chunks(order.len() / 10.max(1)) {
        if chunk.is_empty() {
            continue;
        }
        let mp = chunk.iter().map(|&i| probs_all[i]).sum::<f64>() / chunk.len() as f64;
        let rl = chunk.iter().map(|&i| y_all[i] as f64).sum::<f64>() / chunk.len() as f64;
        calibration.push((mp, rl, chunk.len()));
    }
    Some(EvalReport {
        n_bars: n,
        n_folds: n_splits,
        accuracy: acc,
        accuracy_std: std,
        baseline,
        brier: mean(&briers),
        calibration,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::Quote;

    fn synth_quotes(n: usize, predictable: bool) -> (Vec<Quote>, Vec<Trade>) {
        // quotes with OFI that predicts the next move when `predictable`
        let mut quotes = Vec::new();
        let mut trades = Vec::new();
        let mut mid = 100.0;
        // deterministic PRNG (xorshift)
        let mut s: u64 = 42;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s as f64) / (u64::MAX as f64) - 0.5
        };
        for i in 0..n {
            let t = i as i64 * 1000;
            // regimes last 60s, longer than the 20s prediction horizon
            let drift = if predictable {
                if i % 120 < 60 { 0.02 } else { -0.02 }
            } else {
                0.0
            };
            // OFI-implied: bid stack grows before up moves
            let bq = 5.0 + if drift > 0.0 { 4.0 } else { rnd() };
            let aq = 5.0 + if drift < 0.0 { 4.0 } else { rnd() };
            quotes.push(Quote {
                time_ms: t,
                bid_px: mid - 0.05,
                bid_sz: bq.max(0.5),
                ask_px: mid + 0.05,
                ask_sz: aq.max(0.5),
            });
            mid += drift + rnd() * 0.01;
            // a trade each second, direction follows drift when predictable
            let buy = if predictable { drift > 0.0 } else { rnd() > 0.0 };
            trades.push(Trade {
                time_ms: t + 100,
                price: mid,
                qty: 0.1,
                is_buyer_maker: !buy,
            });
        }
        (quotes, trades)
    }

    #[test]
    fn logistic_separates() {
        let xs = vec![[10.0, 0.0, 0.0, 0.0, 0.0], [-10.0, 0.0, 0.0, 0.0, 0.0]];
        let y = vec![1, 0];
        let w = fit_logistic(&xs, &y, 0.01, 2000, 0.5);
        assert!(predict_proba(&[10.0, 0.0, 0.0, 0.0, 0.0], &w) > 0.9);
        assert!(predict_proba(&[-10.0, 0.0, 0.0, 0.0, 0.0], &w) < 0.1);
    }

    #[test]
    fn beats_coin_on_predictable_flow() {
        let (q, t) = synth_quotes(1200, true);
        let bars = build_dataset(&q, &t, 10_000, 2);
        assert!(bars.len() > 60, "only {} bars", bars.len());
        let rep = walk_forward(&bars, 4, 2).expect("eval failed");
        assert!(
            rep.accuracy > rep.baseline + 0.03,
            "acc {:.3} vs baseline {:.3}",
            rep.accuracy,
            rep.baseline
        );
    }

    #[test]
    fn coin_on_noise() {
        let (q, t) = synth_quotes(1200, false);
        let bars = build_dataset(&q, &t, 10_000, 2);
        let rep = walk_forward(&bars, 4, 2).expect("eval failed");
        assert!(
            (rep.accuracy - rep.baseline).abs() < 0.08,
            "acc {:.3} vs baseline {:.3}",
            rep.accuracy,
            rep.baseline
        );
    }
}
