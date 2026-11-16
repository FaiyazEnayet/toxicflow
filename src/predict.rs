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
//!
//! Statistics (TF6): the headline baseline is the training-majority
//! predictor — each fold commits to its own training majority label
//! and is scored on that fold's test rows — never a rate computed
//! from the test outcomes themselves. The test-outcome majority rate
//! is still reported, labeled `test_majority_rate`, as a descriptive
//! reference only. `accuracy_std` is the population standard
//! deviation of the per-row 0/1 correctness values (for Bernoulli
//! outcomes exactly sqrt(acc*(1-acc))); it describes the dispersion
//! of the rows and is NOT the uncertainty of the accuracy estimate.
//! Uncertainty is reported as bootstrap intervals: a moving-block
//! bootstrap over the out-of-sample rows within a session
//! (`accuracy_ci`), and, for the multi-session pilot
//! (`evaluate_session` / `summarize_session_pilot`), a bootstrap over
//! sessions, because the independent recording session — not the
//! serially dependent bar — is the effective sample unit.
//!
//! Statistics (E4): `accuracy_std` is the population standard
//! deviation of the per-fold accuracies (inter-fold variability),
//! with `accuracy_se_folds` alongside it. The pre-E4 per-row value
//! is kept as `accuracy_std_rows` and reproduced by
//! `historical_metrics` next to the repaired numbers, flagged as an
//! incomparable protocol. The report also states its denominators
//! explicitly: total labeled rows (`n_labeled`), held-out
//! predictions scored (`n_oos`), held-out class counts
//! (`n_pos_oos` / `n_neg_oos`), and rows never scored
//! (`n_unscored`), plus a per-fold Brier score and log loss table.

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

/// Schema tag written into every session pilot report (TF6).
pub const SESSION_PILOT_SCHEMA: &str = "toxicflow.session-pilot/v1";

/// Independent sessions the session pilot is required to cover
/// before its intervals are read as generalization evidence (TF6).
pub const PILOT_REQUIRED_SESSIONS: usize = 10;

/// Bootstrap resamples behind every reported interval (TF6). Fixed,
/// with a fixed seed, so a report is a deterministic function of the
/// rows it summarizes.
pub const BOOTSTRAP_RESAMPLES: usize = 2000;

/// Seed for the within-session block bootstrap (TF6).
pub const BOOTSTRAP_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// Seed for the across-session bootstrap in the pilot summary (TF6).
pub const PILOT_BOOTSTRAP_SEED: u64 = 0xD1B5_4A32_D192_ED03;

/// Moving-block length, in bars, for within-session bootstrap
/// intervals (TF6): the rolling feature windows span 30 bars, so
/// resampled blocks are at least as long as the dependence the
/// features themselves carry.
pub const ACCURACY_BLOCK_BARS: usize = 30;

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

/// One fold's held-out scores (E4): accuracy, Brier score, log loss,
/// and class counts, all scored on exactly that fold's held-out rows.
/// Serialized in `fold_models.json` so the per-fold table survives
/// the producing process, and re-verified by
/// `verify_prediction_ledger`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FoldMetrics {
    pub fold: usize,
    pub n_test: usize,
    pub n_pos: usize,
    pub n_neg: usize,
    pub accuracy: f64,
    /// Accuracy of the training-majority predictor on this fold's
    /// held-out rows.
    pub baseline_accuracy: f64,
    pub brier: f64,
    pub log_loss: f64,
}

#[derive(Debug)]
pub struct EvalReport {
    /// Feature-frame version the evaluated bars were built with
    /// (see FEATURE_VERSION).
    pub feature_version: &'static str,
    /// Total labeled dataset rows fed to the evaluation. This is the
    /// labeled total, not the scored total: only `n_oos` of these
    /// rows are held-out predictions. Kept under its historical name
    /// for compatibility; `n_labeled` carries the same value with
    /// the explicit name.
    pub n_bars: usize,
    /// Total labeled dataset rows fed to the evaluation (same value
    /// as `n_bars`).
    pub n_labeled: usize,
    /// Held-out predictions scored: the exact denominator behind
    /// accuracy, Brier score, log loss, and the calibration bins.
    pub n_oos: usize,
    /// Held-out rows with a positive target (the denominator for
    /// the positive class frequency).
    pub n_pos_oos: usize,
    /// Held-out rows with a negative target (the denominator for
    /// the negative class frequency).
    pub n_neg_oos: usize,
    /// Labeled rows the evaluation never scores: training rows and
    /// purge-gap rows. The prediction path scores every held-out
    /// row, so these are the only rows it abstains on, and the
    /// count is stated here instead of left implicit.
    pub n_unscored: usize,
    pub n_folds: usize,
    pub accuracy: f64,
    /// Population standard deviation of the per-fold accuracies
    /// (E4): inter-fold variability of the accuracy estimate. This
    /// replaced the pre-E4 meaning of this field, the population
    /// standard deviation of the per-row 0/1 correctness values
    /// (exactly sqrt(accuracy * (1 - accuracy)) for Bernoulli
    /// outcomes). That historical value is kept as
    /// `accuracy_std_rows`; the two come from different protocols
    /// and must not be compared. Use `accuracy_ci` for uncertainty.
    pub accuracy_std: f64,
    /// The pre-E4 `accuracy_std`: population standard deviation of
    /// the per-row 0/1 correctness values. Row dispersion, not an
    /// uncertainty. Kept so the old number stays visible and
    /// reproducible instead of being silently replaced.
    pub accuracy_std_rows: f64,
    /// Standard error of the accuracy across folds:
    /// `accuracy_std / sqrt(number of folds fit)`.
    pub accuracy_se_folds: f64,
    /// Out-of-sample accuracy of the training-majority predictor:
    /// each fold predicts its own training majority label for every
    /// test row. This is the headline baseline (TF6); it is computed
    /// without ever reading a test outcome.
    pub baseline: f64,
    /// Majority-class rate computed from the test outcomes
    /// themselves. Descriptive reference only (TF6): it peeks at the
    /// answers, so it is never the bar a model has to beat. Before
    /// TF6 this value was reported as `baseline`.
    pub test_majority_rate: f64,
    /// Balanced accuracy: mean of the per-class correct rates at
    /// the 0.5 threshold. NaN when a class is absent from the
    /// evaluated rows.
    pub balanced_accuracy: f64,
    /// Moving-block bootstrap 95% interval for `accuracy` over the
    /// out-of-sample rows in time order (TF6): (2.5%, 97.5%)
    /// percentiles of the resampled means.
    pub accuracy_ci: (f64, f64),
    pub brier: f64,
    /// Mean log loss (negative log likelihood) over the held-out
    /// predictions, with probabilities clamped to [1e-15, 1-1e-15].
    pub log_loss: f64,
    /// (mean predicted prob, realized rate, n) per decile
    pub calibration: Vec<(f64, f64, usize)>,
    /// Per-fold scores, one entry per fold actually fit.
    pub per_fold: Vec<FoldMetrics>,
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
    /// Population standard deviation of the per-fold accuracies
    /// (E4). Files written before E4 stored the per-row Bernoulli
    /// dispersion here; that value now lives in
    /// `accuracy_std_rows`, and the two are not comparable.
    pub accuracy_std: f64,
    /// The pre-E4 `accuracy_std` (per-row dispersion). Defaults to
    /// 0.0 when loading files written before E4, which did not
    /// record it separately.
    #[serde(default)]
    pub accuracy_std_rows: f64,
    /// Standard error of the accuracy across folds. Defaults to 0.0
    /// when loading files written before E4.
    #[serde(default)]
    pub accuracy_se_folds: f64,
    /// Training-majority predictor accuracy (TF6). Files written
    /// before TF6 stored the test-outcome majority rate here; the
    /// ledger verifier never checks this field, and TF6 files are
    /// identified by the presence of `test_majority_rate`.
    pub baseline: f64,
    pub brier: f64,
    /// Mean log loss over the held-out predictions. Defaults to 0.0
    /// when loading files written before E4.
    #[serde(default)]
    pub log_loss: f64,
    /// Balanced accuracy (TF6). Defaults to 0.0 when loading files
    /// written before TF6, which did not record it.
    #[serde(default)]
    pub balanced_accuracy: f64,
    /// Test-outcome majority rate, descriptive only (TF6). Defaults
    /// to 0.0 when loading files written before TF6; in those files
    /// the same value sits in `baseline`.
    #[serde(default)]
    pub test_majority_rate: f64,
    /// Total labeled dataset rows fed to the evaluation. Defaults to
    /// 0 when loading files written before E4.
    #[serde(default)]
    pub n_labeled: usize,
    /// Held-out predictions scored. Defaults to 0 when loading files
    /// written before E4; in those files the same value sits in
    /// `FoldModelsFile::n_oos`.
    #[serde(default)]
    pub n_oos: usize,
    /// Held-out rows with a positive target. Defaults to 0 when
    /// loading files written before E4.
    #[serde(default)]
    pub n_pos_oos: usize,
    /// Held-out rows with a negative target. Defaults to 0 when
    /// loading files written before E4.
    #[serde(default)]
    pub n_neg_oos: usize,
    /// Labeled rows never scored (training and purge-gap rows).
    /// Defaults to 0 when loading files written before E4.
    #[serde(default)]
    pub n_unscored: usize,
    /// (mean predicted prob, realized rate, n) per decile
    pub calibration: Vec<(f64, f64, usize)>,
    /// Per-fold scores, one entry per fold actually fit. Empty when
    /// loading files written before E4.
    #[serde(default)]
    pub per_fold: Vec<FoldMetrics>,
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
    /// Per-fold scores (E4), parallel to `folds`.
    pub per_fold: Vec<FoldMetrics>,
    /// Per-row correctness of the training-majority predictor (TF6),
    /// parallel to `predictions`: 1.0 when the row's fold committed
    /// to a training majority label equal to the row's target. The
    /// headline `baseline` is the mean of this vector, so the pairing
    /// between model and baseline is exact, row by row.
    pub baseline_correct: Vec<f64>,
}

/// Calibration table: mean predicted probability vs realized rate
/// over rank bins of the out-of-sample predictions.
///
/// The bins partition the probability-sorted predictions by rank, not
/// by a fixed chunk size: with `k = min(10, n)` bins, bin `j` holds
/// sorted positions `floor(j*n/k) .. floor((j+1)*n/k)`, so there are
/// always exactly `k` nonempty bins, every prediction lands in
/// exactly one bin, and bin sizes differ by at most one row. (The
/// previous `chunks(n / 10)` partition produced an eleventh,
/// nearly empty bin whenever `n` was not a multiple of 10: at
/// n = 832 the table was ten bins of 83 rows plus a final 2-row bin.)
/// The sort is stable, so tied probabilities keep ledger order.
/// Empty input yields no bins. The tuple schema is
/// (mean predicted probability, realized rate, row count).
fn calibration_bins(predictions: &[PredictionRow]) -> Vec<(f64, f64, usize)> {
    let mut order: Vec<usize> = (0..predictions.len()).collect();
    order.sort_by(|&i, &j| {
        predictions[i]
            .proba
            .partial_cmp(&predictions[j].proba)
            .unwrap()
    });
    let n = order.len();
    if n == 0 {
        return Vec::new();
    }
    let k = n.min(10);
    let mut calibration = Vec::with_capacity(k);
    for j in 0..k {
        let lo = j * n / k;
        let hi = (j + 1) * n / k;
        let chunk = &order[lo..hi];
        let mp = chunk.iter().map(|&i| predictions[i].proba).sum::<f64>() / chunk.len() as f64;
        let rl = chunk
            .iter()
            .map(|&i| f64::from(predictions[i].target))
            .sum::<f64>()
            / chunk.len() as f64;
        calibration.push((mp, rl, chunk.len()));
    }
    calibration
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
    let acc = mean(&accs);
    let std = (accs.iter().map(|x| (x - acc).powi(2)).sum::<f64>() / accs.len() as f64).sqrt();
    let pos_rate =
        predictions.iter().map(|r| f64::from(r.target)).sum::<f64>() / predictions.len() as f64;
    // TF6: the headline baseline is a predictor, not a prevalence.
    // Each row is scored against the majority label its own fold
    // committed to from training data alone. The old headline value
    // (majority rate over the test outcomes) is kept only as the
    // descriptive `test_majority_rate`.
    let baseline_correct: Vec<f64> = predictions
        .iter()
        .map(|r| f64::from(folds[r.fold].train_majority_label == r.target))
        .collect();
    let baseline = mean(&baseline_correct);
    let test_majority_rate = pos_rate.max(1.0 - pos_rate);
    let balanced_accuracy = balanced_accuracy(&predictions);
    let accuracy_ci = block_bootstrap_mean_ci(
        &accs,
        ACCURACY_BLOCK_BARS,
        BOOTSTRAP_RESAMPLES,
        BOOTSTRAP_SEED,
    );

    let calibration = calibration_bins(&predictions);
    // E4: fold-level statistics and explicit denominators. The
    // headline `accuracy_std` is inter-fold variability (the
    // population SD of the per-fold accuracies); the historical
    // per-row value is kept as `accuracy_std_rows`.
    let per_fold = per_fold_metrics(&folds, &predictions, &baseline_correct);
    let fold_accs: Vec<f64> = per_fold.iter().map(|f| f.accuracy).collect();
    let fold_mean = mean(&fold_accs);
    let fold_std = (fold_accs
        .iter()
        .map(|a| (a - fold_mean).powi(2))
        .sum::<f64>()
        / fold_accs.len() as f64)
        .sqrt();
    let fold_se = fold_std / (fold_accs.len() as f64).sqrt();
    let n_pos_oos = predictions.iter().filter(|r| r.target == 1).count();
    let n_oos = predictions.len();
    Some(WalkForwardDetail {
        report: EvalReport {
            feature_version: FEATURE_VERSION,
            n_bars: n,
            n_labeled: n,
            n_oos,
            n_pos_oos,
            n_neg_oos: n_oos - n_pos_oos,
            n_unscored: n - n_oos,
            n_folds: n_splits,
            accuracy: acc,
            accuracy_std: fold_std,
            accuracy_std_rows: std,
            accuracy_se_folds: fold_se,
            baseline,
            test_majority_rate,
            balanced_accuracy,
            accuracy_ci,
            brier: brier_score(&predictions),
            log_loss: log_loss(&predictions),
            calibration,
            per_fold: per_fold.clone(),
        },
        n_bars: n,
        dataset_sha256: hash_bars(bars),
        folds,
        predictions,
        per_fold,
        baseline_correct,
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
            accuracy_std_rows: detail.report.accuracy_std_rows,
            accuracy_se_folds: detail.report.accuracy_se_folds,
            baseline: detail.report.baseline,
            brier: detail.report.brier,
            log_loss: detail.report.log_loss,
            balanced_accuracy: detail.report.balanced_accuracy,
            test_majority_rate: detail.report.test_majority_rate,
            n_labeled: detail.report.n_labeled,
            n_oos: detail.report.n_oos,
            n_pos_oos: detail.report.n_pos_oos,
            n_neg_oos: detail.report.n_neg_oos,
            n_unscored: detail.report.n_unscored,
            calibration: detail.report.calibration.clone(),
            per_fold: detail.report.per_fold.clone(),
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
    // E4 accumulators for the per-fold recomputation below, plus the
    // (fold, time) keys that must be unique across the ledger.
    let mut seen_keys = std::collections::HashSet::new();
    let mut fold_acc = vec![0.0f64; models.folds.len()];
    let mut fold_base = vec![0.0f64; models.folds.len()];
    let mut fold_brier = vec![0.0f64; models.folds.len()];
    let mut fold_ll = vec![0.0f64; models.folds.len()];
    let mut fold_npos = vec![0usize; models.folds.len()];
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
        if !seen_keys.insert((r.fold, r.time_ms)) {
            return Err(format!(
                "duplicate (fold, time) key: fold {} time {} appears twice in the ledger",
                r.fold, r.time_ms
            ));
        }
        fold_acc[r.fold] += f64::from(u8::from(p > 0.5) == r.target);
        fold_base[r.fold] += f64::from(f.train_majority_label == r.target);
        fold_brier[r.fold] += (p - r.target as f64).powi(2);
        let pc = p.clamp(1e-15, 1.0 - 1e-15);
        fold_ll[r.fold] += if r.target == 1 {
            -pc.ln()
        } else {
            -(1.0 - pc).ln()
        };
        fold_npos[r.fold] += usize::from(r.target == 1);
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
    // E4: the calibration bins must conserve the held-out
    // denominator in at most 10 bins.
    let cal_sum: usize = models.report.calibration.iter().map(|c| c.2).sum();
    if cal_sum != models.n_oos {
        return Err(format!(
            "calibration bins sum to {cal_sum}, saved OOS denominator is {}",
            models.n_oos
        ));
    }
    if models.report.calibration.len() > 10 {
        return Err(format!(
            "calibration table has {} bins, expected at most 10",
            models.report.calibration.len()
        ));
    }
    // E4: when the file carries a per-fold table, its scores must
    // recompute from the reconstructed probabilities. Files written
    // before E4 carry no per-fold table and skip this check.
    if !models.report.per_fold.is_empty() {
        if models.report.per_fold.len() != models.folds.len() {
            return Err(format!(
                "per-fold table holds {} entries for {} folds",
                models.report.per_fold.len(),
                models.folds.len()
            ));
        }
        for (i, fm) in models.report.per_fold.iter().enumerate() {
            let f = &models.folds[i];
            if fm.fold != f.fold || fm.n_test != f.n_test || fm.n_pos != fold_npos[i] {
                return Err(format!(
                    "per-fold entry {i} does not match fold {} (n_test {} vs {}, n_pos {} vs {})",
                    f.fold, fm.n_test, f.n_test, fm.n_pos, fold_npos[i]
                ));
            }
            let d = per_fold[i].max(1) as f64;
            let check = |name: &str, saved: f64, got: f64| -> Result<(), String> {
                if (saved - got).abs() > 1e-8 {
                    return Err(format!(
                        "fold {i}: recomputed {name} {got} differs from saved {saved} by more than 1e-8"
                    ));
                }
                Ok(())
            };
            check("accuracy", fm.accuracy, fold_acc[i] / d)?;
            check("baseline accuracy", fm.baseline_accuracy, fold_base[i] / d)?;
            check("brier", fm.brier, fold_brier[i] / d)?;
            check("log loss", fm.log_loss, fold_ll[i] / d)?;
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

// ---- TF6: defensible statistics and the multi-session pilot ------

/// Balanced accuracy of the 0.5-threshold predictions in `rows`:
/// the mean of the per-class correct rates. Returns NaN when either
/// class is absent, because a per-class rate does not exist then;
/// callers must surface that instead of a fabricated number.
pub fn balanced_accuracy(rows: &[PredictionRow]) -> f64 {
    let (mut tp, mut fn_, mut tn, mut fp) = (0usize, 0usize, 0usize, 0usize);
    for r in rows {
        match (r.target, r.proba > 0.5) {
            (1, true) => tp += 1,
            (1, false) => fn_ += 1,
            (0, false) => tn += 1,
            (0, true) => fp += 1,
            _ => unreachable!(),
        }
    }
    let (pos, neg) = (tp + fn_, tn + fp);
    if pos == 0 || neg == 0 {
        return f64::NAN;
    }
    0.5 * (tp as f64 / pos as f64 + tn as f64 / neg as f64)
}

/// Brier score of the probabilities in `rows`: mean squared error
/// against the 0/1 targets.
pub fn brier_score(rows: &[PredictionRow]) -> f64 {
    if rows.is_empty() {
        return f64::NAN;
    }
    rows.iter()
        .map(|r| (r.proba - f64::from(r.target)).powi(2))
        .sum::<f64>()
        / rows.len() as f64
}

/// Log loss (mean negative log likelihood) of the probabilities in
/// `rows` (E4). Probabilities are clamped to [1e-15, 1-1e-15] so a
/// certain wrong prediction costs a large finite number instead of
/// infinity. Returns NaN on empty input, never a fabricated 0.
pub fn log_loss(rows: &[PredictionRow]) -> f64 {
    if rows.is_empty() {
        return f64::NAN;
    }
    rows.iter()
        .map(|r| {
            let p = r.proba.clamp(1e-15, 1.0 - 1e-15);
            if r.target == 1 {
                -p.ln()
            } else {
                -(1.0 - p).ln()
            }
        })
        .sum::<f64>()
        / rows.len() as f64
}

/// Per-fold scores for a detailed evaluation (E4): accuracy, Brier
/// score, log loss, class counts, and the training-majority baseline
/// accuracy of each fold, computed from that fold's held-out rows.
/// `baseline_correct` is parallel to `predictions` (see
/// `WalkForwardDetail`).
fn per_fold_metrics(
    folds: &[FoldModel],
    predictions: &[PredictionRow],
    baseline_correct: &[f64],
) -> Vec<FoldMetrics> {
    let mut acc = vec![0.0f64; folds.len()];
    let mut base = vec![0.0f64; folds.len()];
    let mut brier = vec![0.0f64; folds.len()];
    let mut ll = vec![0.0f64; folds.len()];
    let mut n = vec![0usize; folds.len()];
    let mut n_pos = vec![0usize; folds.len()];
    for (r, b) in predictions.iter().zip(baseline_correct.iter()) {
        if r.fold >= folds.len() {
            continue;
        }
        let i = r.fold;
        n[i] += 1;
        n_pos[i] += usize::from(r.target == 1);
        acc[i] += f64::from(u8::from(r.proba > 0.5) == r.target);
        base[i] += b;
        brier[i] += (r.proba - f64::from(r.target)).powi(2);
        let p = r.proba.clamp(1e-15, 1.0 - 1e-15);
        ll[i] += if r.target == 1 {
            -p.ln()
        } else {
            -(1.0 - p).ln()
        };
    }
    folds
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let d = n[i].max(1) as f64;
            let div = |x: f64| if n[i] == 0 { f64::NAN } else { x / d };
            FoldMetrics {
                fold: i,
                n_test: n[i],
                n_pos: n_pos[i],
                n_neg: n[i] - n_pos[i],
                accuracy: div(acc[i]),
                baseline_accuracy: div(base[i]),
                brier: div(brier[i]),
                log_loss: div(ll[i]),
            }
        })
        .collect()
}

// ---- E4: historical protocol reproduction ----------------------
//
// The pre-E4 protocol reported different numbers from the same
// evaluation: the headline baseline was the test-outcome majority
// rate, `accuracy_std` was the per-row Bernoulli dispersion, and
// the calibration table was built by fixed chunks. Those numbers
// are reproduced here from the same detail so a reproduction can
// reprint both protocols side by side, but they are never
// comparable to the repaired numbers: `comparable_to_repaired` is
// always false, and `reproduction_summary` says so in plain text.

/// Metrics under the pre-E4 protocol, recomputed from the same
/// evaluation detail.
#[derive(Debug, Clone)]
pub struct HistoricalMetrics {
    pub protocol: &'static str,
    pub comparable_to_repaired: bool,
    /// The historical headline baseline: the test-outcome majority
    /// rate. Descriptive only, never the bar the model had to beat.
    pub baseline: f64,
    /// The historical `accuracy_std`: per-row dispersion, exactly
    /// sqrt(accuracy * (1 - accuracy)).
    pub accuracy_std: f64,
    /// The historical calibration table (fixed-chunk partition).
    pub calibration: Vec<(f64, f64, usize)>,
}

/// The pre-E4 calibration partition: fixed chunks of `n / 10` rows,
/// which leaves an eleventh nearly-empty bin whenever `n` is not a
/// multiple of 10. Kept only so a reproduction can reprint the
/// historical table next to the repaired one.
fn historical_calibration_bins(predictions: &[PredictionRow]) -> Vec<(f64, f64, usize)> {
    let n = predictions.len();
    if n == 0 {
        return Vec::new();
    }
    let chunk = (n / 10).max(1);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        predictions[i]
            .proba
            .partial_cmp(&predictions[j].proba)
            .unwrap()
    });
    let mut out = Vec::new();
    for c in order.chunks(chunk) {
        let mp = c.iter().map(|&i| predictions[i].proba).sum::<f64>() / c.len() as f64;
        let rl = c
            .iter()
            .map(|&i| f64::from(predictions[i].target))
            .sum::<f64>()
            / c.len() as f64;
        out.push((mp, rl, c.len()));
    }
    out
}

/// Recompute the historical protocol metrics from a detailed
/// evaluation.
pub fn historical_metrics(detail: &WalkForwardDetail) -> HistoricalMetrics {
    HistoricalMetrics {
        protocol: "historical-pre-e4",
        comparable_to_repaired: false,
        baseline: detail.report.test_majority_rate,
        accuracy_std: detail.report.accuracy_std_rows,
        calibration: historical_calibration_bins(&detail.predictions),
    }
}

/// Reprint the historical and the repaired metrics from one
/// evaluation, side by side, with the protocols flagged
/// incomparable. Both blocks name their denominators; the
/// historical block is a record of what the old protocol reported,
/// not a number the repaired block can be judged against.
pub fn reproduction_summary(detail: &WalkForwardDetail) -> String {
    let hist = historical_metrics(detail);
    let rep = &detail.report;
    let mut s = String::new();
    s.push_str("reproduction: historical and repaired metrics\n");
    s.push_str(
        "INCOMPARABLE PROTOCOL: the historical block and the repaired block measure \
         different things; do not compare them as before/after values of one metric.\n",
    );
    s.push_str(&format!(
        "historical (protocol {}): baseline {:.4} (test-outcome majority rate, {} held-out \
         predictions), accuracy_std {:.4} (per-row dispersion), calibration bins {}\n",
        hist.protocol,
        hist.baseline,
        rep.n_oos,
        hist.accuracy_std,
        hist.calibration.len()
    ));
    s.push_str(&format!(
        "repaired (protocol e4): accuracy {:.4} vs trained-majority baseline {:.4}, \
         accuracy_std {:.4} (SD of {} per-fold accuracies), SE {:.4}, brier {:.4}, log loss {:.4}, \
         {} labeled rows, {} held-out predictions ({} positive / {} negative), {} rows never \
         scored, calibration bins {}\n",
        rep.accuracy,
        rep.baseline,
        rep.accuracy_std,
        rep.per_fold.len(),
        rep.accuracy_se_folds,
        rep.brier,
        rep.log_loss,
        rep.n_labeled,
        rep.n_oos,
        rep.n_pos_oos,
        rep.n_neg_oos,
        rep.n_unscored,
        rep.calibration.len()
    ));
    s
}

/// SplitMix64: a tiny deterministic PRNG for bootstrap resampling,
/// present so intervals need no extra dependency and a report is a
/// pure function of its rows and the published seed.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Moving-block bootstrap interval for the mean of `values`:
/// resamples circular blocks of `block_len` consecutive values
/// (wrapping at the end) until the resample holds `values.len()`
/// entries, and returns the (2.5%, 97.5%) percentiles of the
/// resampled means over `resamples` draws. Blocks preserve the
/// short-range serial dependence that an IID resample would
/// pretend away; with `block_len == 1` this reduces to the ordinary
/// bootstrap of the mean. Deterministic in `seed`.
pub fn block_bootstrap_mean_ci(
    values: &[f64],
    block_len: usize,
    resamples: usize,
    seed: u64,
) -> (f64, f64) {
    let n = values.len();
    if n == 0 || resamples == 0 {
        return (f64::NAN, f64::NAN);
    }
    let block = block_len.max(1).min(n);
    let blocks_per = n.div_ceil(block);
    let mut rng = SplitMix64(seed);
    let mut means = Vec::with_capacity(resamples);
    for _ in 0..resamples {
        let mut sum = 0.0;
        let mut count = 0usize;
        for _ in 0..blocks_per {
            let start = rng.next_below(n);
            for j in 0..block {
                if count == n {
                    break;
                }
                sum += values[(start + j) % n];
                count += 1;
            }
        }
        means.push(sum / n as f64);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lo = means[((0.025 * resamples as f64) as usize).min(resamples - 1)];
    let hi = means[((0.975 * resamples as f64) as usize).min(resamples - 1)];
    (lo, hi)
}

/// One independent session's paired walk-forward result (TF6). The
/// model and the training-majority baseline are scored on exactly
/// the same out-of-sample rows of this session, so `paired_diff` is
/// a per-session paired comparison, not two unrelated averages.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionEval {
    /// Caller-supplied session identifier (recording name).
    pub session: String,
    /// Bar start time of the first evaluated dataset row's session.
    pub first_bar_ms: i64,
    pub last_bar_ms: i64,
    pub n_bars: usize,
    /// Exact out-of-sample denominator for this session.
    pub n_oos: usize,
    pub n_folds: usize,
    pub accuracy: f64,
    pub balanced_accuracy: f64,
    pub brier: f64,
    /// Training-majority predictor accuracy on the same rows.
    pub train_majority_accuracy: f64,
    /// Test-outcome majority rate: descriptive reference only.
    pub test_majority_rate: f64,
    /// accuracy - train_majority_accuracy, paired on the same rows.
    pub paired_diff: f64,
    /// Block bootstrap interval for this session's accuracy.
    pub accuracy_ci: (f64, f64),
    /// Block bootstrap interval for this session's paired diff.
    pub paired_diff_ci: (f64, f64),
    /// SHA-256 over the session's dataset rows (see `hash_bars`).
    pub dataset_sha256: String,
}

/// Evaluate one session's bars with the same purged walk-forward
/// protocol as the single-session report and summarize it as a
/// paired session result (TF6). Returns None under the same
/// conditions as `walk_forward_detailed`.
pub fn evaluate_session(
    session: &str,
    bars: &[Bar],
    n_splits: usize,
    horizon: usize,
) -> Option<SessionEval> {
    let detail = walk_forward_detailed(bars, n_splits, horizon)?;
    let rows = &detail.predictions;
    let diffs: Vec<f64> = rows
        .iter()
        .zip(detail.baseline_correct.iter())
        .map(|(r, b)| f64::from(u8::from(r.proba > 0.5) == r.target) - b)
        .collect();
    let paired_diff = diffs.iter().sum::<f64>() / diffs.len() as f64;
    Some(SessionEval {
        session: session.to_string(),
        first_bar_ms: bars.first().map(|b| b.time_ms).unwrap_or(0),
        last_bar_ms: bars.last().map(|b| b.time_ms).unwrap_or(0),
        n_bars: detail.n_bars,
        n_oos: rows.len(),
        n_folds: detail.folds.len(),
        accuracy: detail.report.accuracy,
        balanced_accuracy: detail.report.balanced_accuracy,
        brier: detail.report.brier,
        train_majority_accuracy: detail.report.baseline,
        test_majority_rate: detail.report.test_majority_rate,
        paired_diff,
        accuracy_ci: detail.report.accuracy_ci,
        paired_diff_ci: block_bootstrap_mean_ci(
            &diffs,
            ACCURACY_BLOCK_BARS,
            BOOTSTRAP_RESAMPLES,
            BOOTSTRAP_SEED,
        ),
        dataset_sha256: detail.dataset_sha256.clone(),
    })
}

/// The multi-session pilot report (TF6). Aggregation is over
/// sessions: session means are unweighted, and the headline
/// interval resamples sessions, because one independent recording
/// session — not one bar — is the effective sample unit. Bar counts
/// are reported per session for accounting and are never pooled
/// into a significance claim.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionPilot {
    pub schema: String,
    pub bar_ms: i64,
    pub horizon_bars: usize,
    pub n_splits: usize,
    /// The unit one independent observation is counted in.
    pub effective_sample_unit: String,
    /// Sessions the pilot is required to cover (contract target).
    pub required_sessions: usize,
    pub n_sessions: usize,
    /// True only when `n_sessions >= required_sessions`.
    pub session_requirement_met: bool,
    pub sessions: Vec<SessionEval>,
    /// Unweighted mean of the per-session accuracies.
    pub mean_accuracy: f64,
    /// Unweighted mean of the per-session training-majority
    /// accuracies, over the same sessions.
    pub mean_train_majority_accuracy: f64,
    /// Unweighted mean of the per-session paired differences.
    pub mean_paired_diff: f64,
    /// Bootstrap interval for `mean_paired_diff`, resampling whole
    /// sessions with replacement.
    pub paired_diff_session_ci: (f64, f64),
    /// Plain-language limits of this report, written into the
    /// artifact so they travel with the numbers.
    pub notes: Vec<String>,
}

impl SessionPilot {
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("session pilot serializes")
    }
}

/// Aggregate per-session results into the pilot report (TF6). The
/// session bootstrap resamples the observed sessions' paired
/// differences with replacement; with few sessions the interval is
/// coarse by construction and the report says so.
pub fn summarize_session_pilot(
    sessions: Vec<SessionEval>,
    bar_ms: i64,
    horizon_bars: usize,
    n_splits: usize,
) -> SessionPilot {
    let n_sessions = sessions.len();
    let mean_of = |f: &dyn Fn(&SessionEval) -> f64| -> f64 {
        if n_sessions == 0 {
            f64::NAN
        } else {
            sessions.iter().map(f).sum::<f64>() / n_sessions as f64
        }
    };
    let diffs: Vec<f64> = sessions.iter().map(|s| s.paired_diff).collect();
    let paired_diff_session_ci = if n_sessions == 0 {
        (f64::NAN, f64::NAN)
    } else {
        block_bootstrap_mean_ci(&diffs, 1, BOOTSTRAP_RESAMPLES, PILOT_BOOTSTRAP_SEED)
    };
    let session_requirement_met = n_sessions >= PILOT_REQUIRED_SESSIONS;
    let mut notes = vec![
        "Effective sample unit: one independent recording session. Bars inside a \
         session are serially dependent and are never counted as independent samples."
            .to_string(),
        "test_majority_rate is computed from the test outcomes themselves and is a \
         descriptive reference only; the baseline every session is paired against is \
         the training-majority predictor, which never reads a test outcome."
            .to_string(),
    ];
    if !session_requirement_met {
        notes.push(format!(
            "This pilot covers {n_sessions} independent sessions, below the \
             {PILOT_REQUIRED_SESSIONS}-session target. Read the intervals as \
             preliminary: the session count, not the bar count, is the sample size."
        ));
    }
    SessionPilot {
        schema: SESSION_PILOT_SCHEMA.to_string(),
        bar_ms,
        horizon_bars,
        n_splits,
        effective_sample_unit: "independent recording session".to_string(),
        required_sessions: PILOT_REQUIRED_SESSIONS,
        n_sessions,
        session_requirement_met,
        mean_accuracy: mean_of(&|s: &SessionEval| s.accuracy),
        mean_train_majority_accuracy: mean_of(&|s: &SessionEval| s.train_majority_accuracy),
        mean_paired_diff: mean_of(&|s: &SessionEval| s.paired_diff),
        paired_diff_session_ci,
        sessions,
        notes,
    }
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
        for (i, &sd_i) in sd.iter().enumerate().skip(4) {
            let k = (i + 1).min(30) as f64;
            let var_true = (k * k - 1.0) / 12.0;
            let var_got = sd_i * sd_i;
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
        for (i, &sd_old_i) in sd_old.iter().enumerate().skip(4) {
            let k = (i + 1).min(30) as f64;
            let var_true = (k * k - 1.0) / 12.0;
            let rel = (sd_old_i * sd_old_i - var_true).abs() / var_true;
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

    // ---- TF6: defensible walk-forward statistics -------------------
    //
    // Known-answer fixtures for the repaired statistics. Every
    // expected value below is computed by hand in the comments, never
    // derived from the code under test.

    fn tf6_row(target: u8, proba: f64) -> PredictionRow {
        PredictionRow {
            fold: 0,
            row: 0,
            time_ms: 0,
            features: [0.0; N_FEATURES],
            target,
            proba,
        }
    }

    #[test]
    fn stage3_toxicflow_6_row_statistics_known_answers() {
        // targets:  1    1    0    0    1    0
        // probas:   0.9  0.4  0.3  0.8  0.6  0.45
        // preds:    1    0    0    1    1    0
        // correct:  1    0    1    0    1    1   -> accuracy 4/6
        // positives: preds 1,0,1 -> TPR 2/3; negatives: preds 0,1,0
        // -> TNR 2/3; balanced accuracy 2/3.
        // Brier: (0.01 + 0.36 + 0.09 + 0.64 + 0.16 + 0.2025) / 6
        //      = 1.4625 / 6 = 0.24375.
        let rows = vec![
            tf6_row(1, 0.9),
            tf6_row(1, 0.4),
            tf6_row(0, 0.3),
            tf6_row(0, 0.8),
            tf6_row(1, 0.6),
            tf6_row(0, 0.45),
        ];
        let ba = balanced_accuracy(&rows);
        assert!((ba - 2.0 / 3.0).abs() < 1e-12, "balanced accuracy {ba}");
        let br = brier_score(&rows);
        assert!((br - 0.24375).abs() < 1e-12, "brier {br}");
        // A single-class row set has no defined balanced accuracy:
        // it must come back NaN, never a fabricated 0 or 1.
        let one_class = vec![tf6_row(1, 0.9), tf6_row(1, 0.8)];
        assert!(balanced_accuracy(&one_class).is_nan());
    }

    /// Bars whose training prefixes are majority-1 while the
    /// out-of-sample rows are majority-0: the fixture that separates
    /// a training-majority baseline from a test-outcome prevalence.
    ///
    /// Bars 0..100: target 0 iff i % 10 == 9 (90 ones, 10 zeros).
    /// Bars 100..200: target 1 iff i % 10 == 0 (10 ones, 90 zeros).
    /// With 4 splits and horizon 2 the OOS rows are 40..200: 64 ones
    /// in 160 rows. Every fold's training prefix stays majority-1
    /// (fold train ends 38/78/118/158 hold 35/71/92/96 ones), so the
    /// training-majority predictor labels every OOS row 1 and scores
    /// exactly 64/160 = 0.4, while the test-outcome majority rate is
    /// 96/160 = 0.6.
    fn tf6_skewed_bars() -> Vec<Bar> {
        (0..200)
            .map(|i| {
                let target = if i < 100 {
                    u8::from(i % 10 != 9)
                } else {
                    u8::from(i % 10 == 0)
                };
                Bar {
                    time_ms: i as i64 * 10_000,
                    ofi: (i % 7) as f64 - 3.0,
                    ofi_z: 0.0,
                    imbalance: 0.0,
                    spread_z: 0.0,
                    spread_degenerate: false,
                    signed_vol: 0.0,
                    target,
                }
            })
            .collect()
    }

    #[test]
    fn stage3_toxicflow_6_baseline_uses_training_majority_not_test_outcomes() {
        let bars = tf6_skewed_bars();
        let detail = walk_forward_detailed(&bars, 4, 2).expect("detailed eval");
        let rep = &detail.report;
        // The headline baseline is the training-majority predictor
        // scored out of sample: exactly 0.4 on this fixture.
        assert!(
            (rep.baseline - 0.4).abs() < 1e-12,
            "baseline {} (expected the training-majority accuracy 0.4)",
            rep.baseline
        );
        // The test-outcome majority rate survives only as a labeled
        // descriptive reference: exactly 0.6 here.
        assert!(
            (rep.test_majority_rate - 0.6).abs() < 1e-12,
            "test_majority_rate {}",
            rep.test_majority_rate
        );
        // The baseline must be reproducible from the saved fold
        // records and ledger rows alone: each row scored against its
        // own fold's training majority label.
        let mut correct = 0usize;
        for (i, r) in detail.predictions.iter().enumerate() {
            let label = detail.folds[r.fold].train_majority_label;
            correct += usize::from(label == r.target);
            assert_eq!(detail.baseline_correct[i], f64::from(label == r.target));
        }
        let recomputed = correct as f64 / detail.predictions.len() as f64;
        assert!((recomputed - rep.baseline).abs() < 1e-12);
        // accuracy_std is inter-fold variability now (E4): the
        // population SD of the per-fold accuracies. The historical
        // per-row value survives separately as accuracy_std_rows,
        // which for Bernoulli outcomes is sqrt(acc*(1-acc)) exactly.
        // It is dispersion of the rows, not an interval.
        let expect_rows = (rep.accuracy * (1.0 - rep.accuracy)).sqrt();
        assert!(
            (rep.accuracy_std_rows - expect_rows).abs() < 1e-12,
            "accuracy_std_rows {} vs bernoulli sd {expect_rows}",
            rep.accuracy_std_rows
        );
        let accs: Vec<f64> = rep.per_fold.iter().map(|f| f.accuracy).collect();
        let m = accs.iter().sum::<f64>() / accs.len() as f64;
        let expect_folds =
            (accs.iter().map(|a| (a - m).powi(2)).sum::<f64>() / accs.len() as f64).sqrt();
        assert!(
            (rep.accuracy_std - expect_folds).abs() < 1e-12,
            "accuracy_std {} vs fold sd {expect_folds}",
            rep.accuracy_std
        );
        // The uncertainty interval is a separate object: ordered,
        // inside [0, 1], and not the +- accuracy_std pseudo-interval.
        let (lo, hi) = rep.accuracy_ci;
        assert!(lo <= rep.accuracy && rep.accuracy <= hi, "ci ({lo}, {hi})");
        assert!(0.0 <= lo && hi <= 1.0);
    }

    #[test]
    fn stage3_toxicflow_6_block_bootstrap_known_answers() {
        // A constant series has no resampling uncertainty: the
        // interval collapses onto the constant, for any block length.
        let flat = vec![0.5; 50];
        let (lo, hi) = block_bootstrap_mean_ci(&flat, 10, 500, 42);
        assert_eq!((lo, hi), (0.5, 0.5));
        // Deterministic: same seed, same interval, bit for bit.
        let alt: Vec<f64> = (0..100).map(|i| f64::from(i % 2)).collect();
        let a = block_bootstrap_mean_ci(&alt, 1, 2000, 7);
        let b = block_bootstrap_mean_ci(&alt, 1, 2000, 7);
        assert_eq!(a, b);
        // 100 fair coin flips: the block bootstrap of the mean must
        // bracket 0.5 well inside the trivial (0, 1) range.
        assert!(a.0 > 0.3 && a.0 < 0.5 && a.1 > 0.5 && a.1 < 0.7, "ci {a:?}");
        // A different seed may move the endpoints but never out of
        // the convex hull of the data.
        let c = block_bootstrap_mean_ci(&alt, 5, 2000, 99);
        assert!(0.0 <= c.0 && c.0 <= c.1 && c.1 <= 1.0);
    }

    #[test]
    fn stage3_toxicflow_6_session_pilot_pairs_and_counts_sessions() {
        // The pilot machinery runs on the skewed fixture: the session
        // result must carry the same training-majority pairing.
        let bars = tf6_skewed_bars();
        let s = evaluate_session("fixture-skew", &bars, 4, 2).expect("session eval");
        assert!((s.train_majority_accuracy - 0.4).abs() < 1e-12);
        assert!((s.test_majority_rate - 0.6).abs() < 1e-12);
        assert!((s.paired_diff - (s.accuracy - 0.4)).abs() < 1e-12);
        assert_eq!(s.n_oos, 160);

        // Aggregation counts sessions, not bars: two hand-built
        // session results with diffs +0.02 and -0.04 give a mean
        // paired diff of exactly -0.01, and the requirement flag
        // stays off below 10 sessions.
        let mk = |name: &str, acc: f64, diff: f64| SessionEval {
            session: name.to_string(),
            first_bar_ms: 0,
            last_bar_ms: 0,
            n_bars: 100,
            n_oos: 80,
            n_folds: 4,
            accuracy: acc,
            balanced_accuracy: acc,
            brier: 0.25,
            train_majority_accuracy: acc - diff,
            test_majority_rate: 0.6,
            paired_diff: diff,
            accuracy_ci: (acc - 0.05, acc + 0.05),
            paired_diff_ci: (diff - 0.05, diff + 0.05),
            dataset_sha256: String::new(),
        };
        let pilot = summarize_session_pilot(
            vec![mk("a", 0.55, 0.02), mk("b", 0.50, -0.04)],
            10_000,
            2,
            4,
        );
        assert_eq!(pilot.n_sessions, 2);
        assert_eq!(pilot.required_sessions, PILOT_REQUIRED_SESSIONS);
        assert!(!pilot.session_requirement_met);
        assert!((pilot.mean_paired_diff - (-0.01)).abs() < 1e-12);
        assert!((pilot.mean_accuracy - 0.525).abs() < 1e-12);
        assert!(pilot.effective_sample_unit.contains("session"));
        // Session bootstrap endpoints stay inside the observed range
        // of session diffs.
        let (lo, hi) = pilot.paired_diff_session_ci;
        assert!(-0.04 <= lo && lo <= hi && hi <= 0.02, "ci ({lo}, {hi})");
        // The pilot report serializes with its schema tag.
        assert!(pilot.to_json().contains(SESSION_PILOT_SCHEMA));
    }

    fn calib_rows(probas: &[f64], targets: &[u8]) -> Vec<PredictionRow> {
        probas
            .iter()
            .zip(targets.iter())
            .enumerate()
            .map(|(i, (&proba, &target))| PredictionRow {
                fold: 0,
                row: i,
                time_ms: i as i64 * 10_000,
                features: [0.0; N_FEATURES],
                target,
                proba,
            })
            .collect()
    }

    #[test]
    fn calibration_bins_empty_input_yields_no_bins() {
        assert!(calibration_bins(&[]).is_empty());
    }

    #[test]
    fn calibration_bins_fewer_rows_than_bins_gives_one_row_per_bin() {
        // n = 7 < 10, so k = 7 bins of one row each, sorted by proba.
        let rows = calib_rows(&[0.7, 0.1, 0.4, 0.2, 0.6, 0.3, 0.5], &[1, 0, 1, 0, 1, 0, 1]);
        let bins = calibration_bins(&rows);
        assert_eq!(
            bins,
            vec![
                (0.1, 0.0, 1),
                (0.2, 0.0, 1),
                (0.3, 0.0, 1),
                (0.4, 1.0, 1),
                (0.5, 1.0, 1),
                (0.6, 1.0, 1),
                (0.7, 1.0, 1),
            ]
        );
    }

    #[test]
    fn calibration_bins_twenty_rows_split_into_ten_pairs() {
        // n = 20: k = 10 bins of exactly 2 rows. Sorted probas are
        // 0.04, 0.08, ..., 0.80; the first ten sorted rows have
        // target 0 and the last ten target 1. Input is reversed so
        // the sort has to do the work.
        let probas: Vec<f64> = (0..20).map(|m| (20 - m) as f64 * 0.04).collect();
        let targets: Vec<u8> = (0..20).map(|m| u8::from(m < 10)).collect();
        let rows = calib_rows(&probas, &targets);
        let bins = calibration_bins(&rows);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins.iter().map(|b| b.2).sum::<usize>(), 20);
        for (j, b) in bins.iter().enumerate() {
            assert_eq!(b.2, 2, "bin {j} size");
            let want_mp = (4 * j + 3) as f64 * 0.02;
            assert!((b.0 - want_mp).abs() < 1e-12, "bin {j} mean proba {}", b.0);
            let want_rl = if j < 5 { 0.0 } else { 1.0 };
            assert_eq!(b.1, want_rl, "bin {j} realized rate");
        }
    }

    #[test]
    fn calibration_bins_832_rows_make_ten_balanced_bins() {
        // n = 832 (the archived main session's scored rows): the old
        // chunks(n / 10) partition made 11 bins (ten of 83 rows plus
        // a 2-row remainder). The rank partition must make exactly 10
        // bins, conserve every row, and keep sizes within one row.
        let probas: Vec<f64> = (0..832).rev().map(|i| i as f64 / 1000.0).collect();
        let targets: Vec<u8> = (0..832).rev().map(|i| u8::from(i % 3 == 0)).collect();
        let rows = calib_rows(&probas, &targets);
        let bins = calibration_bins(&rows);
        assert_eq!(bins.len(), 10);
        let sizes: Vec<usize> = bins.iter().map(|b| b.2).collect();
        assert_eq!(sizes, vec![83, 83, 83, 83, 84, 83, 83, 83, 83, 84]);
        assert_eq!(sizes.iter().sum::<usize>(), 832);
        // Conservation of outcomes: the count-weighted realized rates
        // return the input's positive count (278 rows with i % 3 == 0).
        let positives: f64 = bins.iter().map(|b| b.1 * b.2 as f64).sum();
        assert!((positives - 278.0).abs() < 1e-9, "positives {positives}");
        // Conservation of probability mass.
        let mass: f64 = bins.iter().map(|b| b.0 * b.2 as f64).sum();
        let want_mass: f64 = (0..832).map(|i| i as f64 / 1000.0).sum();
        assert!(
            (mass - want_mass).abs() < 1e-9,
            "mass {mass} vs {want_mass}"
        );
    }

    #[test]
    fn calibration_bins_tied_probabilities_keep_ledger_order() {
        // All probabilities equal: the stable sort must keep the
        // input (ledger) order, so the bins walk the targets in
        // order: five 1s then fifteen 0s, in pairs.
        let rows = calib_rows(
            &[0.5; 20],
            &[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        );
        let bins = calibration_bins(&rows);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[0], (0.5, 1.0, 2));
        assert_eq!(bins[1], (0.5, 1.0, 2));
        assert_eq!(bins[2], (0.5, 0.5, 2));
        for b in &bins[3..] {
            assert_eq!(*b, (0.5, 0.0, 2));
        }
    }

    #[test]
    fn calibration_bins_hand_calculated_rates() {
        // n = 10: one row per bin. In sorted-proba order the targets
        // are [0,0,1,0,1,1,0,1,1,1]; input is reversed.
        let probas: Vec<f64> = (0..10).rev().map(|m| 0.05 + 0.1 * m as f64).collect();
        let targets_sorted: [u8; 10] = [0, 0, 1, 0, 1, 1, 0, 1, 1, 1];
        let targets: Vec<u8> = targets_sorted.iter().rev().copied().collect();
        let rows = calib_rows(&probas, &targets);
        let bins = calibration_bins(&rows);
        assert_eq!(bins.len(), 10);
        for (j, b) in bins.iter().enumerate() {
            assert_eq!(b.2, 1);
            assert!(
                (b.0 - (0.05 + 0.1 * j as f64)).abs() < 1e-12,
                "bin {j} mp {}",
                b.0
            );
            assert_eq!(b.1, f64::from(targets_sorted[j]), "bin {j} rate");
        }
    }

    #[test]
    fn calibration_rank_partition_leaves_scores_and_predictions_unchanged() {
        // End to end: 1040 bars, 4 splits, horizon 2 gives 4 folds of
        // 208 = 832 OOS predictions, the archived main session's
        // scored-row count. Accuracy, Brier and the prediction vector
        // below were captured from the pre-fix code on this fixture;
        // the partition change is reporting-only, so they must not
        // move, while the calibration table goes from the old 11 bins
        // (sizes 83 x10 plus a 2-row remainder) to 10 balanced bins.
        let bars: Vec<Bar> = (0..1040)
            .map(|i| Bar {
                time_ms: i as i64 * 10_000,
                ofi: ((i * 37 % 101) as f64 - 50.0) / 25.0,
                ofi_z: ((i * 53 % 89) as f64 - 44.0) / 22.0,
                imbalance: ((i * 29 % 41) as f64 - 20.0) / 20.0,
                spread_z: 0.0,
                spread_degenerate: false,
                signed_vol: ((i * 17 % 31) as f64 - 15.0) / 15.0,
                target: u8::from((i * 37 % 101) + (i * 11 % 7) > 55),
            })
            .collect();
        let detail = walk_forward_detailed(&bars, 4, 2).expect("detailed eval");
        assert_eq!(detail.predictions.len(), 832);
        assert!(
            (detail.report.accuracy - 0.985_576_923_076_923_1).abs() < 1e-12,
            "accuracy moved: {}",
            detail.report.accuracy
        );
        assert!(
            (detail.report.brier - 0.02152350977877481).abs() < 1e-12,
            "brier moved: {}",
            detail.report.brier
        );
        let fingerprint: f64 = detail
            .predictions
            .iter()
            .enumerate()
            .map(|(i, r)| r.proba * (i as f64 + 1.0))
            .sum();
        assert!(
            (fingerprint - 164_617.492_914_319_6).abs() < 1e-6,
            "prediction vector moved: {fingerprint}"
        );
        let cal = &detail.report.calibration;
        assert_eq!(cal.len(), 10);
        let sizes: Vec<usize> = cal.iter().map(|c| c.2).collect();
        assert_eq!(sizes, vec![83, 83, 83, 83, 84, 83, 83, 83, 83, 84]);
        assert_eq!(sizes.iter().sum::<usize>(), 832);
    }

    // ======================= E4 fixtures =======================
    // Stage 3 E4: repair out-of-sample statistics. The historical
    // protocol reported a test-outcome majority baseline (fixed by
    // TF6), a per-row accuracy_std, chunk-based calibration bins
    // (fixed by the rank-bin change), and an unexplained n_bars.
    // These fixtures pin the repaired statistics: fold-level
    // variability, explicit labeled/held-out counts, per-fold Brier
    // and log loss, and a reproduction that reprints both protocols
    // flagged incomparable.

    /// The 1040-bar fixture from the calibration test: 4 splits,
    /// horizon 2, exactly 832 held-out predictions.
    fn e4_832_bars() -> Vec<Bar> {
        (0..1040)
            .map(|i| Bar {
                time_ms: i as i64 * 10_000,
                ofi: ((i * 37 % 101) as f64 - 50.0) / 25.0,
                ofi_z: ((i * 53 % 89) as f64 - 44.0) / 22.0,
                imbalance: ((i * 29 % 41) as f64 - 20.0) / 20.0,
                spread_z: 0.0,
                spread_degenerate: false,
                signed_vol: ((i * 17 % 31) as f64 - 15.0) / 15.0,
                target: u8::from((i * 37 % 101) + (i * 11 % 7) > 55),
            })
            .collect()
    }

    #[test]
    fn stage3_toxicflow_4_log_loss_known_answers() {
        // (1, 0.5) and (0, 0.5) each cost ln 2; (1, 0.9) and
        // (0, 0.1) each cost -ln 0.9. Mean over the four rows:
        // (2 ln 2 - 2 ln 0.9) / 4 = 0.3992538481088858.
        let rows = vec![
            tf6_row(1, 0.5),
            tf6_row(0, 0.5),
            tf6_row(1, 0.9),
            tf6_row(0, 0.1),
        ];
        let ll = log_loss(&rows);
        assert!(
            (ll - 0.399_253_848_108_885_8).abs() < 1e-12,
            "log loss {ll}"
        );
        // A certain correct prediction costs ~0 (clamped, never -inf).
        assert!(log_loss(&[tf6_row(1, 1.0)]) < 1e-12);
        assert!(log_loss(&[tf6_row(0, 0.0)]) < 1e-12);
        // No rows, no defined loss: NaN, never a fabricated 0.
        assert!(log_loss(&[]).is_nan());
    }

    #[test]
    fn stage3_toxicflow_4_per_fold_metrics_and_explicit_counts() {
        // Skewed fixture: 200 labeled rows, 4 folds of 40 held-out
        // rows each (160 scored, 40 never scored). Held-out class
        // counts per fold are 36/4, 20/20, 4/36, 4/36 positives, so
        // the pooled held-out classes are 64 positive / 96 negative
        // and every fold's training majority is 1, giving per-fold
        // baseline accuracies 0.9, 0.5, 0.1, 0.1 and pooled 0.4.
        let bars = tf6_skewed_bars();
        let detail = walk_forward_detailed(&bars, 4, 2).expect("detailed eval");
        let rep = &detail.report;
        assert_eq!(rep.n_labeled, 200);
        assert_eq!(rep.n_bars, 200);
        assert_eq!(rep.n_oos, 160);
        assert_eq!(rep.n_unscored, 40);
        assert_eq!(rep.n_oos + rep.n_unscored, rep.n_labeled);
        assert_eq!(rep.n_pos_oos, 64);
        assert_eq!(rep.n_neg_oos, 96);
        assert_eq!(rep.n_pos_oos + rep.n_neg_oos, rep.n_oos);

        assert_eq!(rep.per_fold.len(), 4);
        let pos = [36usize, 20, 4, 4];
        let base = [0.9f64, 0.5, 0.1, 0.1];
        for (i, fm) in rep.per_fold.iter().enumerate() {
            assert_eq!(fm.fold, i);
            assert_eq!(fm.n_test, 40, "fold {i} n_test");
            assert_eq!(fm.n_pos, pos[i], "fold {i} n_pos");
            assert_eq!(fm.n_neg, 40 - pos[i], "fold {i} n_neg");
            assert!(
                (fm.baseline_accuracy - base[i]).abs() < 1e-12,
                "fold {i} baseline {}",
                fm.baseline_accuracy
            );
            // Per-fold Brier and log loss recomputed from the ledger
            // rows of that fold must match the stored fold metrics.
            let rows: Vec<&PredictionRow> =
                detail.predictions.iter().filter(|r| r.fold == i).collect();
            let brier: f64 = rows
                .iter()
                .map(|r| (r.proba - f64::from(r.target)).powi(2))
                .sum::<f64>()
                / rows.len() as f64;
            assert!(
                (fm.brier - brier).abs() < 1e-12,
                "fold {i} brier {} vs {brier}",
                fm.brier
            );
            let owned: Vec<PredictionRow> = rows.iter().map(|r| (*r).clone()).collect();
            assert!(
                (fm.log_loss - log_loss(&owned)).abs() < 1e-12,
                "fold {i} log loss {}",
                fm.log_loss
            );
        }
        assert_eq!(
            rep.per_fold.iter().map(|f| f.n_test).sum::<usize>(),
            rep.n_oos
        );
        assert_eq!(
            rep.per_fold.iter().map(|f| f.n_pos).sum::<usize>(),
            rep.n_pos_oos
        );

        // accuracy_std is inter-fold variability now: the population
        // SD of the four per-fold accuracies, recomputed here from
        // the stored fold table. The historical per-row value
        // survives separately as accuracy_std_rows.
        let accs: Vec<f64> = rep.per_fold.iter().map(|f| f.accuracy).collect();
        let m = accs.iter().sum::<f64>() / accs.len() as f64;
        let fold_sd =
            (accs.iter().map(|a| (a - m).powi(2)).sum::<f64>() / accs.len() as f64).sqrt();
        assert!(
            (rep.accuracy_std - fold_sd).abs() < 1e-12,
            "accuracy_std {} vs fold sd {fold_sd}",
            rep.accuracy_std
        );
        assert!(
            (rep.accuracy_se_folds - fold_sd / 2.0).abs() < 1e-12,
            "fold SE {}",
            rep.accuracy_se_folds
        );
        let expect_rows = (rep.accuracy * (1.0 - rep.accuracy)).sqrt();
        assert!(
            (rep.accuracy_std_rows - expect_rows).abs() < 1e-12,
            "accuracy_std_rows {} vs {expect_rows}",
            rep.accuracy_std_rows
        );
        // Pooled log loss matches the ledger recomputation.
        assert!(
            (rep.log_loss - log_loss(&detail.predictions)).abs() < 1e-12,
            "pooled log loss {}",
            rep.log_loss
        );
    }

    #[test]
    fn stage3_toxicflow_4_future_label_change_leaves_earlier_baseline() {
        // Flip every label at index 80 or later. Folds 0 and 1
        // train only on rows below index 78, so their committed
        // training-majority labels and their fitted probabilities
        // must be bit-identical; a later label cannot reach back
        // into an earlier fold's baseline.
        let bars = tf6_skewed_bars();
        let before = walk_forward_detailed(&bars, 4, 2).expect("detailed eval");
        let mut changed = bars.clone();
        for b in changed.iter_mut().skip(80) {
            b.target ^= 1;
        }
        let after = walk_forward_detailed(&changed, 4, 2).expect("detailed eval");
        for k in 0..2usize {
            assert_eq!(
                after.folds[k].train_majority_label, before.folds[k].train_majority_label,
                "fold {k} baseline label moved after a future label change"
            );
            let p0: Vec<&PredictionRow> =
                before.predictions.iter().filter(|r| r.fold == k).collect();
            let p1: Vec<&PredictionRow> =
                after.predictions.iter().filter(|r| r.fold == k).collect();
            assert_eq!(p0.len(), p1.len(), "fold {k} prediction count moved");
            for (a, b) in p0.iter().zip(p1.iter()) {
                assert_eq!(a.row, b.row);
                assert_eq!(
                    a.proba.to_bits(),
                    b.proba.to_bits(),
                    "fold {k} row {} proba moved after a future label change",
                    a.row
                );
            }
        }
    }

    #[test]
    fn stage3_toxicflow_4_training_label_horizon_never_overlaps_test() {
        // For every fold, the last training row's forward target
        // window (horizon bars) must end strictly before the test
        // block starts: (train_end - 1) + horizon < test_start.
        for (bars, horizon) in [(tf6_skewed_bars(), 2usize), (e4_832_bars(), 2usize)] {
            let detail = walk_forward_detailed(&bars, 4, horizon).expect("detailed eval");
            for f in &detail.folds {
                assert_eq!(
                    f.test_start,
                    f.train_end + f.purge_bars,
                    "fold {} test block must start exactly purge bars after the cutoff",
                    f.fold
                );
                assert!(
                    f.purge_bars >= horizon,
                    "fold {} purge {} shorter than horizon {horizon}",
                    f.fold,
                    f.purge_bars
                );
                assert!(
                    f.train_end - 1 + horizon < f.test_start,
                    "fold {}: last training row {} target window ends at {} but test starts at {}",
                    f.fold,
                    f.train_end - 1,
                    f.train_end - 1 + horizon,
                    f.test_start
                );
            }
        }
    }

    #[test]
    fn stage3_toxicflow_4_oos_predictions_have_unique_fold_time_keys() {
        for bars in [tf6_skewed_bars(), e4_832_bars()] {
            let detail = walk_forward_detailed(&bars, 4, 2).expect("detailed eval");
            let mut keys: Vec<(usize, i64)> = detail
                .predictions
                .iter()
                .map(|r| (r.fold, r.time_ms))
                .collect();
            keys.sort_unstable();
            assert!(
                keys.windows(2).all(|w| w[0] != w[1]),
                "duplicate (fold, time) key in held-out predictions"
            );
            let mut rows: Vec<(usize, usize)> =
                detail.predictions.iter().map(|r| (r.fold, r.row)).collect();
            rows.sort_unstable();
            assert!(
                rows.windows(2).all(|w| w[0] != w[1]),
                "duplicate (fold, row) key in held-out predictions"
            );
            // Every held-out row lies inside its own fold's test
            // block, never in a training range.
            for r in &detail.predictions {
                let f = &detail.folds[r.fold];
                assert!(
                    r.row >= f.test_start && r.row < f.test_end,
                    "row {} outside fold {} test range",
                    r.row,
                    r.fold
                );
            }
        }
    }

    #[test]
    fn stage3_toxicflow_4_calibration_bins_conserve_oos_predictions() {
        for bars in [tf6_skewed_bars(), e4_832_bars()] {
            let detail = walk_forward_detailed(&bars, 4, 2).expect("detailed eval");
            let cal = &detail.report.calibration;
            assert!(cal.len() <= 10, "{} bins, expected at most 10", cal.len());
            assert!(!cal.is_empty());
            let total: usize = cal.iter().map(|c| c.2).sum();
            assert_eq!(
                total, detail.report.n_oos,
                "bin counts sum to {total}, held-out predictions are {}",
                detail.report.n_oos
            );
            let (lo, hi) = (
                cal.iter().map(|c| c.2).min().unwrap(),
                cal.iter().map(|c| c.2).max().unwrap(),
            );
            assert!(
                hi - lo <= 1,
                "approximately equal-count bins differ by more than one row: {lo}..{hi}"
            );
        }
    }

    #[test]
    fn stage3_toxicflow_4_historical_and_repaired_metrics_reprinted() {
        // Skewed fixture: the historical protocol's headline baseline
        // is the test-outcome majority rate 0.6 while the repaired
        // training-majority baseline is 0.4. The two numbers come
        // from different protocols and must never be compared as if
        // one replaced the other.
        let bars = tf6_skewed_bars();
        let detail = walk_forward_detailed(&bars, 4, 2).expect("detailed eval");
        let hist = historical_metrics(&detail);
        assert_eq!(hist.protocol, "historical-pre-e4");
        assert!(!hist.comparable_to_repaired);
        assert!(
            (hist.baseline - 0.6).abs() < 1e-12,
            "historical baseline {}",
            hist.baseline
        );
        assert!(
            (detail.report.baseline - 0.4).abs() < 1e-12,
            "repaired baseline {}",
            detail.report.baseline
        );
        assert!(
            (hist.accuracy_std - detail.report.accuracy_std_rows).abs() < 1e-12,
            "historical accuracy_std must equal the kept per-row value"
        );
        let summary = reproduction_summary(&detail);
        assert!(
            summary.contains("INCOMPARABLE"),
            "reproduction must flag the protocols incomparable:\n{summary}"
        );
        assert!(summary.contains("historical"), "summary:\n{summary}");
        assert!(summary.contains("repaired"), "summary:\n{summary}");
        assert!(summary.contains("0.6000"), "summary:\n{summary}");
        assert!(summary.contains("0.4000"), "summary:\n{summary}");

        // 832 held-out predictions: the historical chunk partition
        // made 11 bins (ten of 83 plus a 2-row remainder) while the
        // repaired rank partition makes exactly 10.
        let detail = walk_forward_detailed(&e4_832_bars(), 4, 2).expect("detailed eval");
        let hist = historical_metrics(&detail);
        assert_eq!(hist.calibration.len(), 11);
        let sizes: Vec<usize> = hist.calibration.iter().map(|c| c.2).collect();
        assert_eq!(
            sizes,
            vec![83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 2],
            "historical bin sizes {sizes:?}"
        );
        assert_eq!(sizes.iter().sum::<usize>(), 832);
        assert_eq!(detail.report.calibration.len(), 10);
        assert_eq!(
            detail.report.calibration.iter().map(|c| c.2).sum::<usize>(),
            832
        );
    }

    #[test]
    fn stage3_toxicflow_4_ledger_verifier_checks_per_fold_and_bins() {
        let bars = tf6_skewed_bars();
        let detail = walk_forward_detailed(&bars, 4, 2).expect("detailed eval");
        let models = build_fold_models_file(&detail, 10_000, 1_700_000_000_000);
        assert_eq!(models.report.per_fold.len(), 4);
        assert_eq!(models.report.n_oos, 160);
        assert_eq!(models.report.n_labeled, 200);
        assert_eq!(models.n_oos, detail.predictions.len());
        let json = models.to_json();
        let csv = predictions_csv(&detail.predictions);
        let loaded_models = load_fold_models(&json).expect("load fold models");
        let loaded_rows = load_predictions(&csv).expect("load predictions");
        verify_prediction_ledger(&loaded_models, &loaded_rows).expect("valid ledger");

        // A tampered per-fold Brier score must fail verification.
        let mut bad = loaded_models.clone();
        bad.report.per_fold[0].brier += 0.01;
        assert!(
            verify_prediction_ledger(&bad, &loaded_rows).is_err(),
            "tampered per-fold brier went undetected"
        );

        // A tampered per-fold log loss must fail verification.
        let mut bad = loaded_models.clone();
        bad.report.per_fold[1].log_loss += 0.01;
        assert!(
            verify_prediction_ledger(&bad, &loaded_rows).is_err(),
            "tampered per-fold log loss went undetected"
        );

        // A duplicated (fold, time) key must fail verification.
        let mut dup = loaded_rows.clone();
        dup[1].time_ms = dup[0].time_ms;
        assert!(
            verify_prediction_ledger(&loaded_models, &dup).is_err(),
            "duplicated (fold, time) key went undetected"
        );

        // A calibration table whose bins do not sum to the held-out
        // denominator must fail verification.
        let mut bad = loaded_models.clone();
        bad.report.calibration = vec![(0.5, 0.5, 1)];
        assert!(
            verify_prediction_ledger(&bad, &loaded_rows).is_err(),
            "calibration bins not summing to n_oos went undetected"
        );
    }
}
