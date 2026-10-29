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

/// Feature-frame version, frozen at build_dataset's semantics.
/// v1 (implicit, pre-N1) computed rolling variance from sliding sums of
/// squares (E[x^2] - E[x]^2), which cancels catastrophically when the
/// mean dwarfs the variance and leaves a non-zero residue on constant
/// windows; it had no degenerate-spread marker. v2 uses a numerically
/// stable two-pass rolling variance, flags zero-variance spread windows
/// explicitly, and pins their z-score to exactly 0. v3 (N2) censors
/// every supervised row whose trailing feature window or forward
/// target window touches a bar with no observed quote, so no row
/// trains on carried-forward bars or crosses a feed gap.
pub const FEATURE_VERSION: &str = "v3";
pub const LEGACY_FEATURE_VERSION: &str = "v1";

#[derive(Debug, Clone)]
pub struct Bar {
    pub ofi: f64,
    pub ofi_z: f64,
    pub imbalance: f64,
    pub spread_z: f64,
    /// True when the rolling spread window behind `spread_z` had zero
    /// variance (e.g. an economically constant one-tick spread). The
    /// z-score of a zero-variance window is defined as exactly 0 and the
    /// row is flagged here instead of emitting a garbage ratio.
    pub spread_degenerate: bool,
    pub signed_vol: f64,
    pub target: u8,
}

impl Bar {
    fn features(&self) -> [f64; N_FEATURES] {
        [
            self.ofi,
            self.ofi_z,
            self.imbalance,
            self.spread_z,
            self.signed_vol,
        ]
    }
}

/// Trailing rolling mean / population sd over the last `window` values
/// (all values so far while the window fills). Entries are NaN until the
/// window holds at least 5 values.
///
/// Numerics (N1): each window is computed independently with a two-pass
/// mean-subtracted sum of squares, never E[x^2] - E[x]^2 over sliding
/// sums. The sliding form cancels catastrophically when the mean dwarfs
/// the variance and drifts as entries are added/removed, so a window of
/// identical values could report sd ~1e-9 instead of 0. Here a window of
/// bit-identical values yields sd exactly 0. Because every window is a
/// pure function of its own slice, results are prefix-invariant bitwise.
fn rolling_mean_sd(xs: &[f64], window: usize) -> (Vec<f64>, Vec<f64>) {
    let n = xs.len();
    let mut mean = vec![f64::NAN; n];
    let mut sd = vec![f64::NAN; n];
    for i in 0..n {
        let k = (i + 1).min(window);
        if k < 5 {
            continue;
        }
        let w = &xs[i + 1 - k..=i];
        let first = w[0];
        if w.iter().all(|&x| x == first) {
            mean[i] = first;
            sd[i] = 0.0;
            continue;
        }
        let m = w.iter().sum::<f64>() / k as f64;
        let v = w.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / k as f64;
        mean[i] = m;
        sd[i] = v.sqrt();
    }
    (mean, sd)
}

/// Build the feature/target frame. Returns empty vec when data is thin.
pub fn build_dataset(quotes: &[Quote], trades: &[Trade], bar_ms: i64, horizon: usize) -> Vec<Bar> {
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
    // N2 gap censoring: a row is supervised only if every bar feeding
    // its features (trailing 30-bar rolling windows) and every bar in
    // its forward target window was actually observed. Carried-forward
    // bars are never promoted to fresh observations.
    let status = crate::metrics::bar_status(quotes, bar_ms);
    let mut out = Vec::new();
    for i in 0..n {
        if i + horizon >= n || ofi_s[i].is_nan() || sp_s[i].is_nan() {
            continue;
        }
        if status.len() == n {
            let lo = i.saturating_sub(29);
            let features_observed = status[lo..=i].iter().all(|s| s.observed);
            let target_observed = status[i + 1..=i + horizon].iter().all(|s| s.observed);
            if !features_observed || !target_observed {
                continue;
            }
        }
        let fwd = mids[i + horizon] / mids[i] - 1.0;
        // A zero-variance spread window (constant spread, e.g. one tick
        // for the whole window) has no meaningful z-score: pin it to
        // exactly 0 and flag the row instead of dividing by a residue.
        let spread_degenerate = sp_s[i] == 0.0;
        out.push(Bar {
            ofi: ofi[i],
            ofi_z: (ofi[i] - ofi_m[i]) / (ofi_s[i] + 1e-12),
            imbalance: imbs[i],
            spread_z: if spread_degenerate {
                0.0
            } else {
                (spreads[i] - sp_m[i]) / (sp_s[i] + 1e-12)
            },
            spread_degenerate,
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
    for m in mu.iter_mut().take(d) {
        *m /= n;
    }
    for row in xs {
        for j in 0..d {
            let z = row[j] - mu[j];
            sd[j] += z * z;
        }
    }
    for s in sd.iter_mut().take(d) {
        *s = (*s / n).sqrt() + 1e-12;
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
    /// Feature-frame version the evaluated bars were built with
    /// (see FEATURE_VERSION).
    pub feature_version: &'static str,
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
    for chunk in order.chunks(order.len() / 10) {
        if chunk.is_empty() {
            continue;
        }
        let mp = chunk.iter().map(|&i| probs_all[i]).sum::<f64>() / chunk.len() as f64;
        let rl = chunk.iter().map(|&i| y_all[i] as f64).sum::<f64>() / chunk.len() as f64;
        calibration.push((mp, rl, chunk.len()));
    }
    Some(EvalReport {
        feature_version: FEATURE_VERSION,
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
                if i % 120 < 60 {
                    0.02
                } else {
                    -0.02
                }
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
            let buy = if predictable {
                drift > 0.0
            } else {
                rnd() > 0.0
            };
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

    #[test]
    fn build_dataset_prefix_invariant_to_appended_quote() {
        // 50 quotes every 10s, then one far-future quote after a gap.
        // Rows built from the prefix alone must match the rows built when
        // the future quote is appended: no feature may see the future.
        let mut quotes: Vec<Quote> = (0..50)
            .map(|i| {
                let mid = 100.0 + i as f64 * 0.01;
                Quote {
                    time_ms: i as i64 * 10_000,
                    bid_px: mid - 0.05,
                    bid_sz: 6.0,
                    ask_px: mid + 0.05,
                    ask_sz: 4.0,
                }
            })
            .collect();
        let prefix = quotes.clone();
        quotes.push(Quote {
            time_ms: 600_000,
            bid_px: 119.95,
            bid_sz: 2.0,
            ask_px: 120.05,
            ask_sz: 8.0,
        });
        let trades: Vec<Trade> = Vec::new();
        let ds_prefix = build_dataset(&prefix, &trades, 10_000, 2);
        let ds_full = build_dataset(&quotes, &trades, 10_000, 2);
        assert!(ds_prefix.len() >= 40, "only {} rows", ds_prefix.len());
        // N2: the appended quote sits after a 10-bar hole (bars 50..=59
        // unobserved), so it adds no supervised rows: every candidate
        // past the prefix has a target window touching the hole and is
        // censored. Before N2 this asserted strictly more rows, which
        // encoded the pre-censoring semantics.
        assert_eq!(ds_full.len(), ds_prefix.len());
        for (i, (a, b)) in ds_prefix.iter().zip(ds_full.iter()).enumerate() {
            assert_eq!(
                a.features(),
                b.features(),
                "row {i} changed after appending a future quote"
            );
            assert_eq!(
                a.target, b.target,
                "row {i} target changed after appending a future quote"
            );
        }
    }

    // ======================= N1 fixtures =======================
    // Stage 3 N1: stable one-tick spread numerics. These fixtures pin the
    // rolling-variance numerics and the degenerate-spread handling.

    /// Exact copy of the pre-N1 sliding sum / sum-of-squares implementation,
    /// kept only so the fixtures can record old-vs-new numbers on identical
    /// inputs. Not used by any production path.
    fn legacy_rolling_mean_sd(xs: &[f64], window: usize) -> (Vec<f64>, Vec<f64>) {
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

    #[test]
    fn n1_known_variance_fixture() {
        // xs = 1e6 + (i mod 30). Every trailing window of length k contains
        // residues {0..k-1} (growing phase) or a rotation of {0..29} (full
        // windows), so the population variance is (k^2 - 1) / 12 in closed
        // form. The 1e6 offset is what exposes the legacy formula: at that
        // scale E[x^2] - E[x]^2 cancels ~10 significant digits.
        let n = 300;
        let xs: Vec<f64> = (0..n).map(|i| 1e6 + (i % 30) as f64).collect();
        let (_, sd) = rolling_mean_sd(&xs, 30);
        let mut worst = 0.0f64;
        for i in 4..n {
            let k = (i + 1).min(30) as f64;
            let var_true = (k * k - 1.0) / 12.0;
            let var_got = sd[i] * sd[i];
            let rel = (var_got - var_true).abs() / var_true;
            worst = worst.max(rel);
            assert!(
                rel <= 1e-10,
                "i={i} k={k}: var {var_got} vs analytic {var_true} (rel err {rel:e})"
            );
        }
        // Record the legacy error on the same fixture: it must NOT meet the
        // 1e-10 bar (this is the before/after record for the fixture).
        let (_, sd_old) = legacy_rolling_mean_sd(&xs, 30);
        let mut worst_old = 0.0f64;
        for i in 4..n {
            let k = (i + 1).min(30) as f64;
            let var_true = (k * k - 1.0) / 12.0;
            let rel = (sd_old[i] * sd_old[i] - var_true).abs() / var_true;
            worst_old = worst_old.max(rel);
        }
        eprintln!("n1 fixture: rolling variance rel err new={worst:e} legacy={worst_old:e}");
        assert!(
            worst_old > 1e-10,
            "legacy formula unexpectedly met the bar (rel err {worst_old:e}); fixture too weak"
        );
    }

    fn n1_quotes(n: usize, spread_at: impl Fn(usize) -> (f64, f64)) -> Vec<Quote> {
        // One quote per 10s bar at a constant mid of 100; sizes vary via a
        // deterministic xorshift so OFI is non-trivial.
        let mut s: u64 = 0x9e3779b97f4a7c15;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 1000) as f64 / 1000.0
        };
        (0..n)
            .map(|i| {
                let (bid, ask) = spread_at(i);
                Quote {
                    time_ms: i as i64 * 10_000,
                    bid_px: bid,
                    bid_sz: 4.0 + 3.0 * rnd(),
                    ask_px: ask,
                    ask_sz: 4.0 + 3.0 * rnd(),
                }
            })
            .collect()
    }

    #[test]
    fn n1_constant_one_tick_spread_z_zero() {
        // Bid/ask pinned at 99.99 / 100.01 for the whole session: the bar
        // spread is one bit-identical value, so every rolling spread window
        // has zero variance and z must be exactly 0, never a garbage ratio
        // of two rounding residues. 2000 bars so any sliding-sum drift in
        // the statistic has room to show up.
        let quotes = n1_quotes(2000, |_| (99.99, 100.01));
        let ds = build_dataset(&quotes, &[], 10_000, 2);
        assert!(ds.len() > 1000, "only {} rows", ds.len());
        for (i, row) in ds.iter().enumerate() {
            assert!(row.spread_z.is_finite(), "row {i}: spread_z not finite");
            assert_eq!(row.spread_z, 0.0, "row {i}: spread_z must be exactly 0");
            assert!(
                row.spread_degenerate,
                "row {i}: constant-spread window must be flagged degenerate"
            );
        }
    }

    #[test]
    fn n1_spread_diagnostic_extremes() {
        // Constructed diagnostic at archived scale (200 bars): bars 0..50
        // alternate a 4-cent and 2-cent spread, bars 50..200 sit on a
        // constant 2-cent spread. Rows whose 30-bar window lies fully in
        // the constant region must have sd exactly 0 (hence z exactly 0)
        // under the current code; the legacy extremes on the same rows are
        // recorded for the before/after record.
        let quotes = n1_quotes(200, |i| {
            if i < 50 && i % 2 == 0 {
                (99.98, 100.02)
            } else {
                (99.99, 100.01)
            }
        });
        let events = ofi_events(&quotes);
        let bars = ofi_bars(&quotes, &events, 10_000);
        let spreads: Vec<f64> = bars.iter().map(|b| b.3).collect();
        let (m_new, s_new) = rolling_mean_sd(&spreads, 30);
        let (m_old, s_old) = legacy_rolling_mean_sd(&spreads, 30);
        let (mut max_new, mut max_old, mut max_sd_old) = (0.0f64, 0.0f64, 0.0f64);
        for i in 79..spreads.len() {
            assert_eq!(
                s_new[i], 0.0,
                "bar {i}: sd of a constant window must be exactly 0"
            );
            let z_new = (spreads[i] - m_new[i]) / (s_new[i] + 1e-12);
            max_new = max_new.max(z_new.abs());
            if s_old[i].is_finite() {
                max_sd_old = max_sd_old.max(s_old[i]);
                let z_old = (spreads[i] - m_old[i]) / (s_old[i] + 1e-12);
                max_old = max_old.max(z_old.abs());
            }
        }
        eprintln!(
            "n1 diagnostic: constant region: max |spread_z| new={max_new:e} legacy={max_old:e}; \
             legacy sd residue up to {max_sd_old:e} (new sd is exactly 0)"
        );
        assert_eq!(max_new, 0.0);
        // Dataset level: trailing rows (windows fully in the constant
        // region) are all flagged degenerate with z exactly 0.
        let ds = build_dataset(&quotes, &[], 10_000, 2);
        assert!(ds.len() > 100, "only {} rows", ds.len());
        for (i, row) in ds.iter().enumerate().skip(100) {
            assert!(row.spread_degenerate, "row {i} should be degenerate");
            assert_eq!(row.spread_z, 0.0, "row {i} spread_z");
        }
    }

    #[test]
    fn n1_prefix_invariance_of_features() {
        // Features for a prefix computed standalone must equal the same
        // rows computed from the full series: 0 violations. Spread varies
        // bar to bar (2/3/4 cents) so the rolling stats are genuinely
        // exercised.
        let full = n1_quotes(240, |i| match i % 3 {
            0 => (99.99, 100.01),
            1 => (99.985, 100.015),
            _ => (99.98, 100.02),
        });
        let prefix = full[..120].to_vec();
        let ds_pre = build_dataset(&prefix, &[], 10_000, 2);
        let ds_full = build_dataset(&full, &[], 10_000, 2);
        assert!(ds_pre.len() > 60, "only {} prefix rows", ds_pre.len());
        assert!(ds_full.len() > ds_pre.len());
        let mut violations = 0;
        for (a, b) in ds_pre.iter().zip(ds_full.iter()) {
            if a.features() != b.features()
                || a.target != b.target
                || a.spread_degenerate != b.spread_degenerate
            {
                violations += 1;
            }
        }
        assert_eq!(
            violations, 0,
            "prefix rows changed when the series was extended"
        );
    }

    #[test]
    fn n1_logistic_probabilities_finite() {
        // Fit on a mix of varying-spread and constant-spread fixture rows;
        // every fitted weight and predicted probability must be finite.
        let varying = n1_quotes(240, |i| match i % 3 {
            0 => (99.99, 100.01),
            1 => (99.985, 100.015),
            _ => (99.98, 100.02),
        });
        let constant = n1_quotes(200, |_| (99.99, 100.01));
        let mut bars = build_dataset(&varying, &[], 10_000, 2);
        bars.extend(build_dataset(&constant, &[], 10_000, 2));
        assert!(bars.len() > 100, "only {} rows", bars.len());
        let xs: Vec<[f64; N_FEATURES]> = bars.iter().map(|b| b.features()).collect();
        let ys: Vec<u8> = bars.iter().map(|b| b.target).collect();
        let w = fit_logistic(&xs, &ys, 1.0, 2000, 0.5);
        assert!(w.iter().all(|x| x.is_finite()), "non-finite weights: {w:?}");
        for (i, x) in xs.iter().enumerate() {
            let p = predict_proba(x, &w);
            assert!(p.is_finite() && (0.0..=1.0).contains(&p), "row {i}: p={p}");
        }
        // The eval report carries the frozen feature version (v3; v2
        // was the N1 frame, the pre-N1 frame was v1).
        let rep = walk_forward(&bars, 3, 2).expect("eval on fixture bars");
        assert_eq!(rep.feature_version, FEATURE_VERSION);
        assert_eq!(FEATURE_VERSION, "v3");
        assert_eq!(LEGACY_FEATURE_VERSION, "v1");
    }

    /// Two 60-bar quote segments on a 10 s grid. `b_start` is the first
    /// post-gap quote time: 1_510_100 leaves exactly 91 unobserved bars
    /// (a 920.1 s hole after the bar-59 quote); 600_000 closes the gap
    /// entirely (contiguous bars 0..=119).
    fn gap_fixture(b_start: i64) -> (Vec<Quote>, Vec<Trade>) {
        let mut quotes = Vec::new();
        let mut trades = Vec::new();
        for k in 0..60i64 {
            let level = 100.0 + 0.01 * k as f64;
            quotes.push(Quote {
                time_ms: k * 10_000,
                bid_px: level - 0.05,
                bid_sz: 5.0,
                ask_px: level + 0.05,
                ask_sz: 5.0,
            });
            trades.push(Trade {
                time_ms: k * 10_000 + 100,
                price: level,
                qty: 0.1,
                is_buyer_maker: k % 2 == 0,
            });
        }
        for k in 0..60i64 {
            let g = if b_start == 600_000 { 60 + k } else { 151 + k };
            let level = 100.0 + 0.01 * g as f64;
            let t = b_start + k * 10_000;
            quotes.push(Quote {
                time_ms: t,
                bid_px: level - 0.05,
                bid_sz: 5.0,
                ask_px: level + 0.05,
                ask_sz: 5.0,
            });
            trades.push(Trade {
                time_ms: t + 100,
                price: level,
                qty: 0.1,
                is_buyer_maker: k % 2 == 0,
            });
        }
        (quotes, trades)
    }

    #[test]
    fn n2_dataset_censors_rows_crossing_gap() {
        // A row survives only if its actual trailing feature window
        // (up to 30 bars; shorter at the series start, where the
        // rolling stats first become defined at i = 4) and its 3-bar
        // target window are fully observed.
        // Segment A (bars 0..=59) yields i in 4..=56 (53 rows);
        // segment B (bars 151..=210) yields i in 180..=207 (28 rows):
        // B rows before 180 have feature windows reaching back into
        // the 91-bar hole. Any row crossing the hole breaks the count.
        let (quotes, trades) = gap_fixture(1_510_100);
        let ds = build_dataset(&quotes, &trades, 10_000, 3);
        assert_eq!(ds.len(), 81, "gapped dataset row count");

        // Control: identical quotes with the hole closed. 120 bars
        // yield i in 4..=116 (113 rows), so the censor removes exactly
        // the gap-crossing and gap-adjacent rows.
        let (quotes, trades) = gap_fixture(600_000);
        let ds = build_dataset(&quotes, &trades, 10_000, 3);
        assert_eq!(ds.len(), 113, "control dataset row count");
    }
}
