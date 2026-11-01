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
//!
//! Persistence (N4): `walk_forward_detailed` also returns every fold's
//! fitted coefficients and the per-row out-of-sample probabilities, and
//! `write_prediction_ledger` serializes them as `fold_models.json` plus
//! a `predictions.csv` ledger. An independent loader
//! (`load_fold_models` / `load_predictions` /
//! `verify_prediction_ledger`) reconstructs each row's probability
//! from the stored weights and feature values and re-derives the
//! headline scores, so an evaluation stays auditable after the process
//! that produced it exits. Coefficients from runs before this
//! serialization existed were held in memory only; the serialized file
//! labels them UNRECOVERABLE and no historical checkpoint recreation
//! is claimed.

use crate::metrics::{ofi_bars, ofi_events, Quote, Trade};

pub const N_FEATURES: usize = 5;

/// Model feature names, in `Bar::features` order.
pub const FEATURE_NAMES: [&str; N_FEATURES] =
    ["ofi", "ofi_z", "imbalance", "spread_z", "signed_vol"];

/// Fit hyperparameters used by the walk-forward evaluation.
pub const FIT_L2: f64 = 1.0;
pub const FIT_ITERS: usize = 2000;
pub const FIT_LR: f64 = 0.5;

/// Schema tag written into every `fold_models.json`.
pub const FOLD_MODELS_SCHEMA: &str = "toxicflow.fold-models/v1";

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
    /// Start time of the OFI bar this row was built from (N4: the
    /// prediction ledger is timestamped, so rows carry their bar time).
    pub time_ms: i64,
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
            time_ms: bars[i].0,
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

/// Minimal SHA-256 (FIPS 180-4), present so input hashes for the
/// prediction ledger need no extra dependency. Pinned by the
/// known-answer fixtures in the test module (`n4_sha256_known_answers`).
fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let bitlen = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_be_bytes());
    for chunk in msg.as_chunks::<64>().0 {
        let mut w = [0u32; 64];
        for (i, c) in chunk.as_chunks::<4>().0.iter().take(16).enumerate() {
            w[i] = u32::from_be_bytes([c[0], c[1], c[2], c[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}

/// Content hash over dataset rows in order. Canonical encoding per
/// row: bar time (i64 LE), the 5 feature values as f64 bit patterns
/// (LE), the target byte, and the degenerate-spread flag byte. Any
/// change to a row's time, features, or label changes the hash.
fn hash_bars(bars: &[Bar]) -> String {
    let mut buf = Vec::with_capacity(bars.len() * 50);
    for b in bars {
        buf.extend_from_slice(&b.time_ms.to_le_bytes());
        for v in b.features() {
            buf.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        buf.push(b.target);
        buf.push(u8::from(b.spread_degenerate));
    }
    sha256_hex(&buf)
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

/// One fold's persisted model record (N4): everything an independent
/// loader needs to reconstruct that fold's out-of-sample predictions
/// and to audit its train/test geometry.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FoldModel {
    /// Sequential fold index over the folds that were actually fit.
    pub fold: usize,
    /// Training rows are the dataset range [train_start, train_end).
    pub train_start: usize,
    /// Training cutoff: number of leading dataset rows trained on.
    pub train_end: usize,
    /// Bar timestamp of the last training row.
    pub cutoff_time_ms: i64,
    /// Purge gap in bars between the training cutoff and the test
    /// block: test_start == train_end + purge_bars by construction.
    pub purge_bars: usize,
    /// Evaluated rows are the dataset range [test_start, test_end).
    pub test_start: usize,
    pub test_end: usize,
    pub n_train: usize,
    pub n_test: usize,
    /// Majority-class rate over this fold's training labels.
    pub train_majority_rate: f64,
    pub train_majority_label: u8,
    /// Raw-space intercept (the fold standardization is folded into
    /// the raw-space weights; see the file-level preprocessing note).
    pub intercept: f64,
    /// Raw-space weights in FEATURE_NAMES order.
    pub weights: [f64; N_FEATURES],
    /// SHA-256 over this fold's training rows (see `hash_bars`).
    pub train_input_sha256: String,
}

/// One row of the out-of-sample probability ledger.
#[derive(Debug, Clone, PartialEq)]
pub struct PredictionRow {
    pub fold: usize,
    /// Index into the evaluated dataset slice.
    pub row: usize,
    /// Start time of the bar the row was built from.
    pub time_ms: i64,
    pub features: [f64; N_FEATURES],
    pub target: u8,
    pub proba: f64,
}

/// The headline scores as saved inside `fold_models.json`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SavedReport {
    pub accuracy: f64,
    pub accuracy_std: f64,
    pub baseline: f64,
    pub brier: f64,
    /// (mean predicted prob, realized rate, n) per decile
    pub calibration: Vec<(f64, f64, usize)>,
}

/// Serialized form of `fold_models.json`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FoldModelsFile {
    pub schema: String,
    pub created_unix_ms: i64,
    pub feature_version: String,
    pub feature_names: Vec<String>,
    /// How the saved weights consume a raw feature row.
    pub preprocessing: String,
    pub bar_ms: i64,
    pub horizon_bars: usize,
    pub n_splits: usize,
    pub fit_l2: f64,
    pub fit_iters: usize,
    pub fit_lr: f64,
    pub n_bars: usize,
    /// Exact out-of-sample denominator: total evaluated rows.
    pub n_oos: usize,
    /// SHA-256 over every dataset row fed to the evaluation.
    pub dataset_sha256: String,
    /// Coefficients of runs from before this serialization existed
    /// were held in memory only and cannot be recreated.
    pub past_in_memory_coefficients: String,
    pub report: SavedReport,
    pub folds: Vec<FoldModel>,
}

impl FoldModelsFile {
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("fold models serialize")
    }
}

/// A walk-forward evaluation with its full persistence detail (N4).
pub struct WalkForwardDetail {
    pub report: EvalReport,
    pub n_bars: usize,
    pub dataset_sha256: String,
    pub folds: Vec<FoldModel>,
    pub predictions: Vec<PredictionRow>,
}

/// Purged walk-forward evaluation.
pub fn walk_forward(bars: &[Bar], n_splits: usize, horizon: usize) -> Option<EvalReport> {
    walk_forward_detailed(bars, n_splits, horizon).map(|d| d.report)
}

/// Purged walk-forward evaluation that also returns the per-fold
/// model records and the per-row out-of-sample probability ledger
/// (N4). The headline report is computed from exactly the returned
/// predictions, in the same order the in-memory evaluation always
/// used, so its numbers are unchanged by the persistence detail.
pub fn walk_forward_detailed(
    bars: &[Bar],
    n_splits: usize,
    horizon: usize,
) -> Option<WalkForwardDetail> {
    let n = bars.len();
    if n < 60 {
        return None;
    }
    let fold = n / (n_splits + 1);
    if fold < 10 {
        return None;
    }
    let mut folds: Vec<FoldModel> = Vec::new();
    let mut predictions: Vec<PredictionRow> = Vec::new();
    for k in 1..=n_splits {
        let a = k * fold;
        let b = ((k + 1) * fold).min(n);
        let train_end = a.saturating_sub(horizon); // purge gap
        if train_end < 30 || b - a < 10 {
            continue;
        }
        let xs: Vec<[f64; N_FEATURES]> = bars[..train_end].iter().map(|x| x.features()).collect();
        let ys: Vec<u8> = bars[..train_end].iter().map(|x| x.target).collect();
        let w = fit_logistic(&xs, &ys, FIT_L2, FIT_ITERS, FIT_LR);
        let pos = ys.iter().map(|&y| usize::from(y)).sum::<usize>();
        let neg = ys.len() - pos;
        let (majority_label, majority_n) = if pos >= neg { (1, pos) } else { (0, neg) };
        let fold_idx = folds.len();
        folds.push(FoldModel {
            fold: fold_idx,
            train_start: 0,
            train_end,
            cutoff_time_ms: bars[train_end - 1].time_ms,
            purge_bars: horizon,
            test_start: a,
            test_end: b,
            n_train: train_end,
            n_test: b - a,
            train_majority_rate: majority_n as f64 / ys.len() as f64,
            train_majority_label: majority_label,
            intercept: w[N_FEATURES],
            weights: [w[0], w[1], w[2], w[3], w[4]],
            train_input_sha256: hash_bars(&bars[..train_end]),
        });
        for (off, bar) in bars[a..b].iter().enumerate() {
            let p = predict_proba(&bar.features(), &w);
            predictions.push(PredictionRow {
                fold: fold_idx,
                row: a + off,
                time_ms: bar.time_ms,
                features: bar.features(),
                target: bar.target,
                proba: p,
            });
        }
    }
    if predictions.is_empty() {
        return None;
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let accs: Vec<f64> = predictions
        .iter()
        .map(|r| f64::from(u8::from(r.proba > 0.5) == r.target))
        .collect();
    let briers: Vec<f64> = predictions
        .iter()
        .map(|r| (r.proba - r.target as f64).powi(2))
        .collect();
    let acc = mean(&accs);
    let std = (accs.iter().map(|x| (x - acc).powi(2)).sum::<f64>() / accs.len() as f64).sqrt();
    let pos_rate =
        predictions.iter().map(|r| f64::from(r.target)).sum::<f64>() / predictions.len() as f64;
    let baseline = pos_rate.max(1.0 - pos_rate);

    let mut order: Vec<usize> = (0..predictions.len()).collect();
    order.sort_by(|&i, &j| {
        predictions[i]
            .proba
            .partial_cmp(&predictions[j].proba)
            .unwrap()
    });
    let mut calibration = Vec::new();
    for chunk in order.chunks(order.len() / 10) {
        if chunk.is_empty() {
            continue;
        }
        let mp = chunk.iter().map(|&i| predictions[i].proba).sum::<f64>() / chunk.len() as f64;
        let rl = chunk
            .iter()
            .map(|&i| f64::from(predictions[i].target))
            .sum::<f64>()
            / chunk.len() as f64;
        calibration.push((mp, rl, chunk.len()));
    }
    Some(WalkForwardDetail {
        report: EvalReport {
            feature_version: FEATURE_VERSION,
            n_bars: n,
            n_folds: n_splits,
            accuracy: acc,
            accuracy_std: std,
            baseline,
            brier: mean(&briers),
            calibration,
        },
        n_bars: n,
        dataset_sha256: hash_bars(bars),
        folds,
        predictions,
    })
}

/// Assemble the serializable `fold_models.json` content for a
/// detailed evaluation. `bar_ms` and `created_unix_ms` are supplied
/// by the caller (the evaluation itself is time-free and
/// deterministic).
pub fn build_fold_models_file(
    detail: &WalkForwardDetail,
    bar_ms: i64,
    created_unix_ms: i64,
) -> FoldModelsFile {
    let horizon_bars = detail.folds.first().map(|f| f.purge_bars).unwrap_or(0);
    FoldModelsFile {
        schema: FOLD_MODELS_SCHEMA.to_string(),
        created_unix_ms,
        feature_version: FEATURE_VERSION.to_string(),
        feature_names: FEATURE_NAMES.iter().map(|s| s.to_string()).collect(),
        preprocessing: "Features are the raw per-bar values in feature_names order. \
            fit_logistic standardizes each feature with the training fold's mean and \
            population sd (sd floor 1e-12) during gradient descent, then folds that \
            standardization back into the saved raw-space weights, so a saved model \
            scores a raw feature row directly: proba = sigmoid(intercept + \
            sum_j weights[j] * x[j]), with the sigmoid argument clamped to [-30, 30]."
            .to_string(),
        bar_ms,
        horizon_bars,
        n_splits: detail.report.n_folds,
        fit_l2: FIT_L2,
        fit_iters: FIT_ITERS,
        fit_lr: FIT_LR,
        n_bars: detail.n_bars,
        n_oos: detail.predictions.len(),
        dataset_sha256: detail.dataset_sha256.clone(),
        past_in_memory_coefficients: "UNRECOVERABLE".to_string(),
        report: SavedReport {
            accuracy: detail.report.accuracy,
            accuracy_std: detail.report.accuracy_std,
            baseline: detail.report.baseline,
            brier: detail.report.brier,
            calibration: detail.report.calibration.clone(),
        },
        folds: detail.folds.clone(),
    }
}

/// The ledger CSV text. Floats use Rust's Display formatting, which
/// round-trips f64 exactly, so a parsed row is bit-identical to the
/// row that was written (pinned by `n4_predictions_csv_float_roundtrip`).
pub fn predictions_csv(rows: &[PredictionRow]) -> String {
    let mut s =
        String::from("fold,row,time_ms,ofi,ofi_z,imbalance,spread_z,signed_vol,target,proba\n");
    for r in rows {
        s.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{}\n",
            r.fold,
            r.row,
            r.time_ms,
            r.features[0],
            r.features[1],
            r.features[2],
            r.features[3],
            r.features[4],
            r.target,
            r.proba
        ));
    }
    s
}

/// Write `fold_models.json` and `predictions.csv` for a detailed
/// evaluation into `dir` (created if missing). Returns both paths.
pub fn write_prediction_ledger(
    dir: &std::path::Path,
    detail: &WalkForwardDetail,
    bar_ms: i64,
    created_unix_ms: i64,
) -> std::io::Result<(std::path::PathBuf, std::path::PathBuf)> {
    std::fs::create_dir_all(dir)?;
    let models = build_fold_models_file(detail, bar_ms, created_unix_ms);
    let json_path = dir.join("fold_models.json");
    let csv_path = dir.join("predictions.csv");
    std::fs::write(&json_path, models.to_json())?;
    std::fs::write(&csv_path, predictions_csv(&detail.predictions))?;
    Ok((json_path, csv_path))
}

/// Independent loader: parse a `fold_models.json` document. This and
/// `load_predictions` read only the serialized files, never the
/// producing process's memory.
pub fn load_fold_models(json: &str) -> Result<FoldModelsFile, String> {
    serde_json::from_str(json).map_err(|e| format!("fold_models.json does not parse: {e}"))
}

/// Independent loader: parse a `predictions.csv` ledger.
pub fn load_predictions(csv: &str) -> Result<Vec<PredictionRow>, String> {
    let mut lines = csv.lines();
    let header = lines.next().ok_or("predictions.csv is empty")?;
    if header != "fold,row,time_ms,ofi,ofi_z,imbalance,spread_z,signed_vol,target,proba" {
        return Err(format!("predictions.csv header mismatch: {header}"));
    }
    let mut rows = Vec::new();
    for (ln, line) in lines.enumerate() {
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        if f.len() != 10 {
            return Err(format!(
                "predictions.csv line {}: expected 10 fields, got {}",
                ln + 2,
                f.len()
            ));
        }
        let parse = |i: usize| -> Result<f64, String> {
            f[i].parse::<f64>()
                .map_err(|e| format!("predictions.csv line {}: field {}: {e}", ln + 2, i))
        };
        rows.push(PredictionRow {
            fold: f[0]
                .parse()
                .map_err(|e| format!("predictions.csv line {}: fold: {e}", ln + 2))?,
            row: f[1]
                .parse()
                .map_err(|e| format!("predictions.csv line {}: row: {e}", ln + 2))?,
            time_ms: f[2]
                .parse()
                .map_err(|e| format!("predictions.csv line {}: time_ms: {e}", ln + 2))?,
            features: [parse(3)?, parse(4)?, parse(5)?, parse(6)?, parse(7)?],
            target: f[8]
                .parse()
                .map_err(|e| format!("predictions.csv line {}: target: {e}", ln + 2))?,
            proba: parse(9)?,
        });
    }
    Ok(rows)
}

/// Outcome of a successful ledger verification.
#[derive(Debug)]
pub struct LedgerVerification {
    pub n_rows: usize,
    /// Largest |reconstructed - saved| probability over all rows.
    pub max_proba_err: f64,
    /// Accuracy recomputed from the reconstructed probabilities.
    pub accuracy: f64,
    /// Brier score recomputed from the reconstructed probabilities.
    pub brier: f64,
}

/// Verify a loaded ledger against its fold models, using only the
/// two serialized artifacts: every row's probability is recomputed
/// from the fold's stored weights and the row's stored features, the
/// row accounting must reproduce the saved out-of-sample denominator
/// exactly, each fold's test block must start exactly `purge_bars`
/// after its training cutoff (so no evaluated row was trained on),
/// and the recomputed accuracy and Brier score must match the saved
/// report. All tolerances are 1e-8.
pub fn verify_prediction_ledger(
    models: &FoldModelsFile,
    rows: &[PredictionRow],
) -> Result<LedgerVerification, String> {
    if models.schema != FOLD_MODELS_SCHEMA {
        return Err(format!("unknown fold models schema: {}", models.schema));
    }
    if models.folds.is_empty() {
        return Err("fold models contain no fold records".to_string());
    }
    if models.feature_names.len() != N_FEATURES {
        return Err(format!(
            "feature_names has {} entries, expected {N_FEATURES}",
            models.feature_names.len()
        ));
    }
    for f in &models.folds {
        if f.test_start < f.train_end {
            return Err(format!(
                "fold {}: test block starts at {} inside the training range ending {}",
                f.fold, f.test_start, f.train_end
            ));
        }
        if f.test_start != f.train_end + f.purge_bars {
            return Err(format!(
                "fold {}: test block starts at {}, expected cutoff {} + purge {}",
                f.fold, f.test_start, f.train_end, f.purge_bars
            ));
        }
        if f.test_end - f.test_start != f.n_test {
            return Err(format!(
                "fold {}: test range holds {} rows but n_test says {}",
                f.fold,
                f.test_end - f.test_start,
                f.n_test
            ));
        }
        if f.train_end - f.train_start != f.n_train {
            return Err(format!(
                "fold {}: train range holds {} rows but n_train says {}",
                f.fold,
                f.train_end - f.train_start,
                f.n_train
            ));
        }
    }
    let summed: usize = models.folds.iter().map(|f| f.n_test).sum();
    if summed != models.n_oos {
        return Err(format!(
            "fold test counts sum to {summed}, saved OOS denominator is {}",
            models.n_oos
        ));
    }
    if rows.len() != models.n_oos {
        return Err(format!(
            "ledger holds {} rows, saved OOS denominator is {}",
            rows.len(),
            models.n_oos
        ));
    }
    let mut per_fold = vec![0usize; models.folds.len()];
    let mut max_err = 0.0f64;
    let mut acc_sum = 0.0f64;
    let mut brier_sum = 0.0f64;
    for r in rows {
        let f = models
            .folds
            .get(r.fold)
            .ok_or_else(|| format!("ledger row {} references unknown fold {}", r.row, r.fold))?;
        if f.fold != r.fold {
            return Err(format!(
                "fold record at position {} is labeled fold {}",
                r.fold, f.fold
            ));
        }
        if r.row < f.test_start || r.row >= f.test_end {
            return Err(format!(
                "ledger row {} lies outside fold {} test range [{}, {})",
                r.row, r.fold, f.test_start, f.test_end
            ));
        }
        if !r.proba.is_finite() || !(0.0..=1.0).contains(&r.proba) {
            return Err(format!(
                "ledger row {}: saved proba {} not in [0, 1]",
                r.row, r.proba
            ));
        }
        let w = [
            f.weights[0],
            f.weights[1],
            f.weights[2],
            f.weights[3],
            f.weights[4],
            f.intercept,
        ];
        let p = predict_proba(&r.features, &w);
        let err = (p - r.proba).abs();
        if !err.is_finite() {
            return Err(format!("ledger row {}: reconstruction not finite", r.row));
        }
        max_err = max_err.max(err);
        per_fold[r.fold] += 1;
        acc_sum += f64::from(u8::from(p > 0.5) == r.target);
        brier_sum += (p - r.target as f64).powi(2);
    }
    for (i, f) in models.folds.iter().enumerate() {
        if per_fold[i] != f.n_test {
            return Err(format!(
                "fold {}: ledger holds {} rows, fold record says {}",
                f.fold, per_fold[i], f.n_test
            ));
        }
    }
    if max_err > 1e-8 {
        return Err(format!(
            "max proba reconstruction error {max_err:e} exceeds 1e-8"
        ));
    }
    let accuracy = acc_sum / rows.len() as f64;
    let brier = brier_sum / rows.len() as f64;
    if (accuracy - models.report.accuracy).abs() > 1e-8 {
        return Err(format!(
            "recomputed accuracy {accuracy} differs from saved {} by more than 1e-8",
            models.report.accuracy
        ));
    }
    if (brier - models.report.brier).abs() > 1e-8 {
        return Err(format!(
            "recomputed brier {brier} differs from saved {} by more than 1e-8",
            models.report.brier
        ));
    }
    Ok(LedgerVerification {
        n_rows: rows.len(),
        max_proba_err: max_err,
        accuracy,
        brier,
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

    // ======================= N4 fixtures =======================
    // Stage 3 N4: per-fold logistic weights and the exact OOS
    // prediction ledger must survive the process that produced them,
    // so an independent loader can reconstruct every score.

    #[test]
    fn n4_sha256_known_answers() {
        // NIST / FIPS 180-4 known-answer vectors for the in-tree
        // SHA-256 used for input hashes.
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        assert_eq!(
            sha256_hex(&vec![b'a'; 1_000_000]),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn n4_dataset_rows_carry_bar_timestamps() {
        // The ledger is timestamped, so every dataset row must carry
        // the start time of the OFI bar it was built from: on the
        // 10 s grid and strictly increasing in row order.
        let (q, t) = synth_quotes(600, true);
        let ds = build_dataset(&q, &t, 10_000, 2);
        assert!(ds.len() > 40, "only {} rows", ds.len());
        for (i, row) in ds.iter().enumerate() {
            assert_eq!(row.time_ms % 10_000, 0, "row {i} off the bar grid");
            if i > 0 {
                assert!(
                    row.time_ms > ds[i - 1].time_ms,
                    "row {i} timestamp not increasing"
                );
            }
        }
    }

    #[test]
    fn n4_fold_ledger_roundtrip_reconstructs() {
        // 2400 one-second quotes on 10 s bars, horizon 2: enough rows
        // that all four walk-forward folds clear the 30-row training
        // minimum, so the standard evaluation yields four fold records.
        let (q, t) = synth_quotes(2400, true);
        let bars = build_dataset(&q, &t, 10_000, 2);
        let detail = walk_forward_detailed(&bars, 4, 2).expect("detailed eval");
        assert_eq!(detail.folds.len(), 4, "expected four fold records");
        assert_eq!(detail.report.n_folds, 4);

        let models = build_fold_models_file(&detail, 10_000, 1_700_000_000_000);
        assert_eq!(models.n_oos, detail.predictions.len());
        assert_eq!(
            models.folds.iter().map(|f| f.n_test).sum::<usize>(),
            models.n_oos,
            "fold test counts must sum to the saved OOS denominator"
        );
        for f in &models.folds {
            assert_eq!(f.test_end - f.test_start, f.n_test);
            assert_eq!(
                f.test_start,
                f.train_end + f.purge_bars,
                "fold {} test block must start exactly purge bars after the cutoff",
                f.fold
            );
            assert!(
                f.train_end > f.train_start,
                "fold {} empty train range",
                f.fold
            );
        }

        // Serialize, then reconstruct with the independent loader only.
        let json = models.to_json();
        let csv = predictions_csv(&detail.predictions);
        let loaded_models = load_fold_models(&json).expect("load fold models");
        let loaded_rows = load_predictions(&csv).expect("load predictions");
        assert_eq!(loaded_rows.len(), models.n_oos);
        let check =
            verify_prediction_ledger(&loaded_models, &loaded_rows).expect("ledger verification");
        eprintln!(
            "n4 fixture: {} rows, max proba reconstruction err {:e}",
            check.n_rows, check.max_proba_err
        );
        assert!(
            check.max_proba_err <= 1e-8,
            "reconstruction err {} exceeds 1e-8",
            check.max_proba_err
        );
        assert!((check.accuracy - detail.report.accuracy).abs() <= 1e-8);
        assert!((check.brier - detail.report.brier).abs() <= 1e-8);
        // The detailed path must not change the evaluation itself:
        // same headline numbers as the in-memory walk_forward.
        let rep = walk_forward(&bars, 4, 2).expect("plain eval");
        assert_eq!(rep.accuracy, detail.report.accuracy);
        assert_eq!(rep.brier, detail.report.brier);
    }

    #[test]
    fn n4_ledger_detects_tampering() {
        let (q, t) = synth_quotes(2400, true);
        let bars = build_dataset(&q, &t, 10_000, 2);
        let detail = walk_forward_detailed(&bars, 4, 2).expect("detailed eval");
        let models = build_fold_models_file(&detail, 10_000, 1_700_000_000_000);
        let rows = load_predictions(&predictions_csv(&detail.predictions)).unwrap();

        // A shifted weight must break reconstruction beyond 1e-8.
        let mut bad = models.clone();
        bad.folds[0].weights[0] += 0.01;
        assert!(
            verify_prediction_ledger(&bad, &rows).is_err(),
            "tampered weight went undetected"
        );

        // A dropped ledger row must break the exact denominator.
        let short = &rows[..rows.len() - 1];
        assert!(
            verify_prediction_ledger(&models, short).is_err(),
            "dropped row went undetected"
        );

        // A schema the loader does not know must be refused, never
        // silently reinterpreted.
        let mut wrong_schema = models.clone();
        wrong_schema.schema = "toxicflow.fold-models/v999".to_string();
        assert!(verify_prediction_ledger(&wrong_schema, &rows).is_err());
    }

    #[test]
    fn n4_past_coefficients_marked_unrecoverable() {
        let (q, t) = synth_quotes(2400, true);
        let bars = build_dataset(&q, &t, 10_000, 2);
        let detail = walk_forward_detailed(&bars, 4, 2).expect("detailed eval");
        let models = build_fold_models_file(&detail, 10_000, 1_700_000_000_000);
        assert_eq!(models.past_in_memory_coefficients, "UNRECOVERABLE");
        assert!(models.to_json().contains("UNRECOVERABLE"));
    }

    #[test]
    fn n4_predictions_csv_float_roundtrip() {
        // The CSV is an exact ledger: every float written must parse
        // back to the identical bit pattern.
        let rows = vec![
            PredictionRow {
                fold: 0,
                row: 7,
                time_ms: 1_700_000_000_000,
                features: [0.1, -3.89, 1e-7, 123456.789, -0.0],
                target: 1,
                proba: 0.809,
            },
            PredictionRow {
                fold: 3,
                row: 8,
                time_ms: 1_700_000_010_000,
                features: [-14.07, 9.70, 0.0, 2.5e-12, 42.0],
                target: 0,
                proba: 1.0 / 3.0,
            },
        ];
        let back = load_predictions(&predictions_csv(&rows)).expect("csv roundtrip");
        assert_eq!(back.len(), rows.len());
        for (a, b) in rows.iter().zip(back.iter()) {
            assert_eq!(a.fold, b.fold);
            assert_eq!(a.row, b.row);
            assert_eq!(a.time_ms, b.time_ms);
            assert_eq!(a.target, b.target);
            for j in 0..N_FEATURES {
                assert_eq!(
                    a.features[j].to_bits(),
                    b.features[j].to_bits(),
                    "feature {j} changed in the CSV roundtrip"
                );
            }
            assert_eq!(a.proba.to_bits(), b.proba.to_bits());
        }
    }
}
