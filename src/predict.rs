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
//!
//! Targets (P1): the binary-sign target above is preserved exactly
//! as the baseline. Two further tracks share its decision rows and
//! feature frame: a next-nonzero-mid-change target (sign of the
//! first nonzero mid change within a frozen cutoff, censored when no
//! move is observed) and an explicit three-class {down, flat, up}
//! target over a frozen horizon and flat-band grid. Every row
//! carries its decision timestamp and every label traces to a
//! concrete future mid observation; censored rows are counted per
//! track and per reason, never dropped silently.

use crate::metrics::{
    ofi_bars, ofi_events, touch_bars, volume_buckets_sized, vpin, BarError, Quote, Trade,
};

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

/// Trailing bars behind every rolling feature window in
/// `build_dataset` (30 bars = 300 s on the 10 s grid). Nonzero by
/// construction: a zero window would leave every rolling statistic
/// undefined.
pub const ROLLING_WINDOW_BARS: usize = 30;

/// What went wrong while building the feature/target frame. Invalid
/// configuration or input is a typed error, never a silently empty
/// frame. Thin data (fewer than 40 bars) is not an error: it still
/// yields `Ok` with no rows, as before.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DatasetError {
    /// The underlying bar grid could not be built.
    Bars(BarError),
    /// The prediction horizon is the forward lookback window; zero
    /// bars ahead is not a prediction.
    InvalidHorizon { horizon: usize },
    /// Trade timestamps must be nondecreasing.
    NonMonotonicTrades {
        index: usize,
        previous: i64,
        current: i64,
    },
    /// A trade falls before the first bar start, so its bar index
    /// would be negative. It previously cast silently to `usize`
    /// and vanished from the frame with no trace.
    TradeBeforeStart {
        index: usize,
        time_ms: i64,
        bar_start: i64,
    },
    /// A trade timestamp could not be placed on the bar grid
    /// without overflowing or underflowing i64.
    TimestampOverflow { time_ms: i64 },
}

impl From<BarError> for DatasetError {
    fn from(e: BarError) -> Self {
        DatasetError::Bars(e)
    }
}

impl std::fmt::Display for DatasetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatasetError::Bars(e) => write!(f, "{e}"),
            DatasetError::InvalidHorizon { horizon } => {
                write!(f, "prediction horizon must be nonzero, got {horizon}")
            }
            DatasetError::NonMonotonicTrades {
                index,
                previous,
                current,
            } => write!(
                f,
                "trade timestamps are not nondecreasing at index {index}: {previous} then {current}"
            ),
            DatasetError::TradeBeforeStart {
                index,
                time_ms,
                bar_start,
            } => write!(
                f,
                "trade {index} at {time_ms} falls before the first bar start {bar_start}"
            ),
            DatasetError::TimestampOverflow { time_ms } => {
                write!(f, "trade timestamp overflows i64 near {time_ms}")
            }
        }
    }
}

impl std::error::Error for DatasetError {}

/// Build the feature/target frame. Thin data (fewer than 40 bars)
/// yields `Ok` with no rows. Returns `Err` on an invalid bar grid
/// (see `BarError`), a zero prediction horizon, non-monotonic trades,
/// a trade before the first bar start, or a trade timestamp that
/// underflows i64 when placed on the grid.
pub fn build_dataset(
    quotes: &[Quote],
    trades: &[Trade],
    bar_ms: i64,
    horizon: usize,
) -> Result<Vec<Bar>, DatasetError> {
    if horizon == 0 {
        return Err(DatasetError::InvalidHorizon { horizon });
    }
    let events = ofi_events(quotes);
    let bars = ofi_bars(quotes, &events, bar_ms)?;
    if bars.len() < 40 {
        return Ok(Vec::new());
    }
    let n = bars.len();
    let ofi: Vec<f64> = bars.iter().map(|b| b.1).collect();
    let mids: Vec<f64> = bars.iter().map(|b| b.2).collect();
    let spreads: Vec<f64> = bars.iter().map(|b| b.3).collect();
    let imbs: Vec<f64> = bars.iter().map(|b| b.4).collect();

    // signed trade volume per bar. Trades must arrive in
    // nondecreasing time order and on or after the first bar start:
    // a negative bar index is an error here, never a silent `usize`
    // cast. Trades after the last bar have no bar to land in and
    // are dropped, as before.
    let t0 = bars[0].0;
    let mut signed = vec![0.0; n];
    let mut prev_trade: Option<i64> = None;
    for (i, t) in trades.iter().enumerate() {
        if let Some(prev) = prev_trade {
            if t.time_ms < prev {
                return Err(DatasetError::NonMonotonicTrades {
                    index: i,
                    previous: prev,
                    current: t.time_ms,
                });
            }
        }
        prev_trade = Some(t.time_ms);
        let offset = t
            .time_ms
            .checked_sub(t0)
            .ok_or(DatasetError::TimestampOverflow { time_ms: t.time_ms })?;
        if offset < 0 {
            return Err(DatasetError::TradeBeforeStart {
                index: i,
                time_ms: t.time_ms,
                bar_start: t0,
            });
        }
        let idx = (offset / bar_ms) as usize;
        if idx < n {
            signed[idx] += t.signed_qty();
        }
    }

    let (ofi_m, ofi_s) = rolling_mean_sd(&ofi, ROLLING_WINDOW_BARS);
    let (sp_m, sp_s) = rolling_mean_sd(&spreads, ROLLING_WINDOW_BARS);
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
            let lo = i.saturating_sub(ROLLING_WINDOW_BARS - 1);
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
    Ok(out)
}

// ---- P2: normalized flow features (feature frame v4) ------------

/// Number of features in the P2 (v4) frame: the five v3 features
/// plus ten normalized-flow features.
pub const P2_N_FEATURES: usize = 15;

/// P2 feature names, in `P2Bar::features` order.
pub const P2_FEATURE_NAMES: [&str; P2_N_FEATURES] = [
    "ofi",
    "ofi_z",
    "imbalance",
    "spread_z",
    "signed_vol",
    "signed_flow_w",
    "ofi_depth",
    "signed_flow_lag3",
    "signed_flow_lag10",
    "ofi_lag3",
    "ofi_lag10",
    "quote_rate",
    "trade_rate",
    "realized_vol",
    "micro_disp",
];

/// Feature-frame version of `build_p2_dataset`. v4 adds the ten
/// normalized-flow features to the v3 frame; the first five columns
/// reproduce `build_dataset` bit-identically on shared rows (pinned
/// by `stage3_toxicflow_2_p2_v3_columns_bit_identical`), so the v3
/// baseline is preserved exactly inside the new frame.
pub const P2_FEATURE_VERSION: &str = "v4";

/// Trailing window behind the signed-flow sum, in bars (6 bars =
/// 60 s on the 10 s grid).
pub const P2_FLOW_WINDOW_BARS: usize = 6;

/// Trailing window behind the depth, rate and volatility means, in
/// bars (30 bars = 300 s on the 10 s grid). P2 rows start at bar 29
/// so every trailing window is full.
pub const P2_TRAIL_WINDOW_BARS: usize = 30;

/// Prespecified stationary lag scales, in bars (30 s and 100 s on
/// the 10 s grid).
pub const P2_LAG_BARS: [usize; 2] = [3, 10];

/// Column subsets for the baseline ladder (see the P2 grid):
/// B0 is the empty list (training-majority predictor), B3 is the
/// v3 frame, B4 adds the flow-normalization group, B5 is the full
/// v4 frame.
pub const P2_B0_COLS: &[usize] = &[];
pub const P2_B3_COLS: &[usize] = &[0, 1, 2, 3, 4];
pub const P2_B4_COLS: &[usize] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
pub const P2_B5_COLS: &[usize] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
/// The ten P2 columns (indices 5..15): the block the permutation
/// and shift controls act on.
pub const P2_NEW_COLS: &[usize] = &[5, 6, 7, 8, 9, 10, 11, 12, 13, 14];

/// Seed for the permutation control's deterministic shuffle.
pub const P2_PERM_SEED: u64 = 0xA5A5_5A5A_F00D_BA5E;

/// Shift lag for the shift control, in bars.
pub const P2_SHIFT_LAG: usize = 5;

/// One v4 decision row.
#[derive(Debug, Clone)]
pub struct P2Bar {
    /// Start time of the OFI bar this row was built from.
    pub time_ms: i64,
    pub features: [f64; P2_N_FEATURES],
    pub target: u8,
    pub spread_degenerate: bool,
}

impl P2Bar {
    /// The five v3 columns, in `Bar::features` order.
    pub fn v3_features(&self) -> [f64; N_FEATURES] {
        [
            self.features[0],
            self.features[1],
            self.features[2],
            self.features[3],
            self.features[4],
        ]
    }
}

/// Trailing mean over the last `window` values (all values so far
/// while the window fills). A pure function of the slice ending at
/// i, so results are prefix-invariant bitwise.
fn trailing_mean(xs: &[f64], window: usize) -> Vec<f64> {
    xs.iter()
        .enumerate()
        .map(|(i, _)| {
            let k = (i + 1).min(window);
            xs[i + 1 - k..=i].iter().sum::<f64>() / k as f64
        })
        .collect()
}

/// Trailing sum over the last `window` values. A pure function of
/// the slice ending at i, so results are prefix-invariant bitwise.
fn trailing_sum(xs: &[f64], window: usize) -> Vec<f64> {
    xs.iter()
        .enumerate()
        .map(|(i, _)| {
            let k = (i + 1).min(window);
            xs[i + 1 - k..=i].iter().sum::<f64>()
        })
        .collect()
}

/// Trailing realized volatility: sqrt of the sum of squared log mid
/// returns over the trailing `window` bars. Only the `window` - 1
/// returns strictly inside the window are summed, so no return
/// spans the bar before the window (which may sit across a feed
/// gap). A pure function of mids at or before i.
fn trailing_realized_vol(mids: &[f64], window: usize) -> Vec<f64> {
    mids.iter()
        .enumerate()
        .map(|(i, _)| {
            let k = (i + 1).min(window);
            let lo = i + 1 - k;
            let mut s = 0.0;
            for j in lo + 1..=i {
                let r = (mids[j] / mids[j - 1]).ln();
                s += r * r;
            }
            s.sqrt()
        })
        .collect()
}

/// Build the v4 feature/target frame. Returns empty vec when data
/// is thin (fewer than 40 bars). Rows start at bar 29 so every
/// trailing 30-bar window is full, and share `build_dataset`'s gap
/// censoring: a row is supervised only if every bar feeding its
/// features and every bar in its forward target window was actually
/// observed. Every feature uses touch-accessible fields only, so the
/// frame is computable from archived quote data.
pub fn build_p2_dataset(
    quotes: &[Quote],
    trades: &[Trade],
    bar_ms: i64,
    horizon: usize,
) -> Vec<P2Bar> {
    // H4: `ofi_bars` requires time-ordered quotes. Sort stably so a
    // delayed message still lands in its own bar, keeping its arrival
    // order inside the bar; already-ordered input is unchanged.
    let mut ordered: Vec<Quote> = quotes.to_vec();
    ordered.sort_by_key(|q| q.time_ms);
    let quotes = &ordered[..];
    let events = ofi_events(quotes);
    // H4: an invalid bar grid yields no rows here, the same thin-data
    // outcome as before; `ofi_bars` itself now reports the cause.
    let bars = ofi_bars(quotes, &events, bar_ms).unwrap_or_default();
    if bars.len() < 40 {
        return Vec::new();
    }
    let touch = touch_bars(quotes, bar_ms);
    let n = bars.len();
    let ofi: Vec<f64> = bars.iter().map(|b| b.1).collect();
    let mids: Vec<f64> = bars.iter().map(|b| b.2).collect();
    let spreads: Vec<f64> = bars.iter().map(|b| b.3).collect();
    let imbs: Vec<f64> = bars.iter().map(|b| b.4).collect();

    let t0 = bars[0].0;
    let mut signed = vec![0.0; n];
    let mut trade_counts = vec![0usize; n];
    for t in trades {
        let idx = ((t.time_ms - t0) / bar_ms) as usize;
        if idx < n {
            signed[idx] += t.signed_qty();
            trade_counts[idx] += 1;
        }
    }
    let quote_counts: Vec<f64> = touch.iter().map(|b| b.n_quotes as f64).collect();
    let trade_counts_f: Vec<f64> = trade_counts.iter().map(|&c| c as f64).collect();
    let depths: Vec<f64> = touch.iter().map(|b| b.depth_mean).collect();

    let (ofi_m, ofi_s) = rolling_mean_sd(&ofi, 30);
    let (sp_m, sp_s) = rolling_mean_sd(&spreads, 30);
    let depth_mean = trailing_mean(&depths, P2_TRAIL_WINDOW_BARS);
    let quote_rate = trailing_mean(&quote_counts, P2_TRAIL_WINDOW_BARS);
    let trade_rate = trailing_mean(&trade_counts_f, P2_TRAIL_WINDOW_BARS);
    let signed_flow_w = trailing_sum(&signed, P2_FLOW_WINDOW_BARS);
    let realized_vol = trailing_realized_vol(&mids, P2_TRAIL_WINDOW_BARS);

    let status = crate::metrics::bar_status(quotes, bar_ms);
    let mut out = Vec::new();
    for i in 0..n {
        if i < P2_TRAIL_WINDOW_BARS - 1 || i + horizon >= n || ofi_s[i].is_nan() || sp_s[i].is_nan()
        {
            continue;
        }
        if status.len() == n {
            let lo = i.saturating_sub(P2_TRAIL_WINDOW_BARS - 1);
            let features_observed = status[lo..=i].iter().all(|s| s.observed);
            let target_observed = status[i + 1..=i + horizon].iter().all(|s| s.observed);
            if !features_observed || !target_observed {
                continue;
            }
        }
        let fwd = mids[i + horizon] / mids[i] - 1.0;
        let spread_degenerate = sp_s[i] == 0.0;
        let mut f = [0.0; P2_N_FEATURES];
        f[0] = ofi[i];
        f[1] = (ofi[i] - ofi_m[i]) / (ofi_s[i] + 1e-12);
        f[2] = imbs[i];
        f[3] = if spread_degenerate {
            0.0
        } else {
            (spreads[i] - sp_m[i]) / (sp_s[i] + 1e-12)
        };
        f[4] = signed[i];
        f[5] = signed_flow_w[i];
        f[6] = if depth_mean[i] == 0.0 {
            0.0
        } else {
            ofi[i] / depth_mean[i]
        };
        f[7] = signed[i - P2_LAG_BARS[0]];
        f[8] = signed[i - P2_LAG_BARS[1]];
        f[9] = ofi[i - P2_LAG_BARS[0]];
        f[10] = ofi[i - P2_LAG_BARS[1]];
        f[11] = quote_rate[i];
        f[12] = trade_rate[i];
        f[13] = realized_vol[i];
        f[14] = touch[i].micro_disp;
        out.push(P2Bar {
            time_ms: bars[i].0,
            features: f,
            target: u8::from(fwd > 0.0),
            spread_degenerate,
        });
    }
    out
}

/// How the P2 feature block is treated at scoring time. The fitted
/// model is identical under every control: only the scored test
/// rows change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum P2Control {
    /// Score each row on its own features.
    Off,
    /// Within each fold's test block, the P2 block of each row is
    /// replaced by the P2 block of another test row, drawn by a
    /// fixed-seed shuffle. Breaks feature-target alignment while
    /// preserving the block's joint distribution.
    Permute,
    /// Within each fold's test block, the P2 block of each row is
    /// replaced by the P2 block of the test row `P2_SHIFT_LAG` bars
    /// earlier. The first `P2_SHIFT_LAG` rows of each block are
    /// unscored.
    Shift,
}

/// One scored P2 row.
#[derive(Debug, Clone)]
pub struct P2Prediction {
    pub fold: usize,
    pub row: usize,
    pub time_ms: i64,
    pub target: u8,
    pub proba: f64,
}

/// One fold's held-out scores for a P2 evaluation.
#[derive(Debug, Clone)]
pub struct P2FoldMetrics {
    pub fold: usize,
    pub n_train: usize,
    pub n_test: usize,
    /// Rows actually scored (fewer than `n_test` under Shift).
    pub n_scored: usize,
    pub accuracy: f64,
    pub baseline_accuracy: f64,
    pub brier: f64,
    pub log_loss: f64,
}

/// A P2 walk-forward evaluation on one column subset.
#[derive(Debug, Clone)]
pub struct P2Eval {
    pub n_labeled: usize,
    /// Held-out predictions scored: the exact denominator behind
    /// accuracy, Brier score and log loss.
    pub n_oos: usize,
    pub n_pos_oos: usize,
    pub n_neg_oos: usize,
    /// Labeled rows the evaluation never scores: training rows,
    /// purge-gap rows, and (under Shift) warm-up rows.
    pub n_unscored: usize,
    pub accuracy: f64,
    /// Out-of-sample accuracy of the training-majority predictor on
    /// the same scored rows.
    pub baseline: f64,
    pub brier: f64,
    pub log_loss: f64,
    pub balanced_accuracy: f64,
    pub per_fold: Vec<P2FoldMetrics>,
    pub predictions: Vec<P2Prediction>,
    /// (raw-space weights in `cols` order, intercept) per fold.
    /// The standardization behind each fold is learned from that
    /// fold's training rows only.
    pub fold_models: Vec<(Vec<f64>, f64)>,
}

/// Full-batch gradient descent for L2 logistic regression on a
/// column subset of the P2 frame. Standardization (mean and
/// population sd, sd floor 1e-12) is learned from the supplied rows
/// only, exactly as in `fit_logistic`; callers pass training rows.
/// Returns (raw-space weights in `cols` order, intercept).
fn fit_logistic_p2_cols(
    xs: &[[f64; P2_N_FEATURES]],
    y: &[u8],
    cols: &[usize],
    l2: f64,
    iters: usize,
    lr: f64,
) -> (Vec<f64>, f64) {
    let n = xs.len() as f64;
    let d = cols.len();
    let mut mu = vec![0.0; d];
    let mut sd = vec![0.0; d];
    for row in xs {
        for (k, &j) in cols.iter().enumerate() {
            mu[k] += row[j];
        }
    }
    for m in mu.iter_mut() {
        *m /= n;
    }
    for row in xs {
        for (k, &j) in cols.iter().enumerate() {
            let z = row[j] - mu[k];
            sd[k] += z * z;
        }
    }
    for s in sd.iter_mut() {
        *s = (*s / n).sqrt() + 1e-12;
    }
    let mut w = vec![0.0; d + 1];
    for _ in 0..iters {
        let mut grad = vec![0.0; d + 1];
        for (row, &yi) in xs.iter().zip(y.iter()) {
            let mut z = w[d];
            for (k, &j) in cols.iter().enumerate() {
                z += w[k] * (row[j] - mu[k]) / sd[k];
            }
            let err = sigmoid(z) - yi as f64;
            for (k, &j) in cols.iter().enumerate() {
                grad[k] += err * (row[j] - mu[k]) / sd[k];
            }
            grad[d] += err;
        }
        for k in 0..d {
            w[k] -= lr * (grad[k] / n + l2 * w[k] / n);
        }
        w[d] -= lr * grad[d] / n;
    }
    let mut raw = vec![0.0; d];
    let mut b = w[d];
    for k in 0..d {
        raw[k] = w[k] / sd[k];
        b -= w[k] * mu[k] / sd[k];
    }
    (raw, b)
}

fn predict_proba_p2_cols(
    x: &[f64; P2_N_FEATURES],
    cols: &[usize],
    w: &[f64],
    intercept: f64,
) -> f64 {
    let mut z = intercept;
    for (k, &j) in cols.iter().enumerate() {
        z += w[k] * x[j];
    }
    sigmoid(z)
}

/// Deterministic Fisher-Yates shuffle of 0..n with a xorshift
/// seeded by `seed`. Scored row j takes its P2 block from donor
/// row perm[j].
fn p2_permutation(n: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut s: u64 = seed | 1;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    for i in (1..n).rev() {
        let j = (rnd() % (i as u64 + 1)) as usize;
        idx.swap(i, j);
    }
    idx
}

/// Purged walk-forward evaluation of one P2 column subset, with the
/// same fold geometry as `walk_forward_detailed` (fold =
/// n / (n_splits + 1), purge gap = `horizon` bars). An empty column
/// list is the B0 training-majority predictor: every row scores the
/// training positive rate, and the reported accuracy equals the
/// training-majority accuracy by construction.
pub fn walk_forward_p2(
    bars: &[P2Bar],
    cols: &[usize],
    control: P2Control,
    n_splits: usize,
    horizon: usize,
) -> Option<P2Eval> {
    let n = bars.len();
    if n < 60 {
        return None;
    }
    let fold = n / (n_splits + 1);
    if fold < 10 {
        return None;
    }
    let mut per_fold: Vec<P2FoldMetrics> = Vec::new();
    let mut predictions: Vec<P2Prediction> = Vec::new();
    let mut fold_models: Vec<(Vec<f64>, f64)> = Vec::new();
    for k in 1..=n_splits {
        let a = k * fold;
        let b = ((k + 1) * fold).min(n);
        let train_end = a.saturating_sub(horizon);
        if train_end < 30 || b - a < 10 {
            continue;
        }
        let train = &bars[..train_end];
        let ys: Vec<u8> = train.iter().map(|x| x.target).collect();
        let pos = ys.iter().map(|&y| usize::from(y)).sum::<usize>();
        let (majority_label, _majority_n) = if pos * 2 >= ys.len() {
            (1u8, pos)
        } else {
            (0u8, ys.len() - pos)
        };
        let (w, intercept) = if cols.is_empty() {
            let p = (pos as f64 / ys.len() as f64).clamp(1e-6, 1.0 - 1e-6);
            (Vec::new(), (p / (1.0 - p)).ln())
        } else {
            let xs: Vec<[f64; P2_N_FEATURES]> = train.iter().map(|x| x.features).collect();
            fit_logistic_p2_cols(&xs, &ys, cols, FIT_L2, FIT_ITERS, FIT_LR)
        };
        let test = &bars[a..b];
        let mut scored: Vec<(usize, [f64; P2_N_FEATURES])> = Vec::new();
        match control {
            P2Control::Off => {
                for (off, row) in test.iter().enumerate() {
                    scored.push((a + off, row.features));
                }
            }
            P2Control::Permute => {
                let perm = p2_permutation(test.len(), P2_PERM_SEED ^ (k as u64));
                for (off, row) in test.iter().enumerate() {
                    let mut f = row.features;
                    let donor = test[perm[off]].features;
                    for &j in P2_NEW_COLS {
                        f[j] = donor[j];
                    }
                    scored.push((a + off, f));
                }
            }
            P2Control::Shift => {
                for (off, row) in test.iter().enumerate() {
                    if off < P2_SHIFT_LAG {
                        continue;
                    }
                    let mut f = row.features;
                    let donor = test[off - P2_SHIFT_LAG].features;
                    for &j in P2_NEW_COLS {
                        f[j] = donor[j];
                    }
                    scored.push((a + off, f));
                }
            }
        }
        let fold_idx = per_fold.len();
        let mut acc_sum = 0.0f64;
        let mut base_sum = 0.0f64;
        let mut brier_sum = 0.0f64;
        let mut ll_sum = 0.0f64;
        for (row_idx, f) in &scored {
            let bar = &bars[*row_idx];
            let p = if cols.is_empty() {
                sigmoid(intercept)
            } else {
                predict_proba_p2_cols(f, cols, &w, intercept)
            };
            predictions.push(P2Prediction {
                fold: fold_idx,
                row: *row_idx,
                time_ms: bar.time_ms,
                target: bar.target,
                proba: p,
            });
            acc_sum += f64::from(u8::from(p > 0.5) == bar.target);
            base_sum += f64::from(majority_label == bar.target);
            brier_sum += (p - f64::from(bar.target)).powi(2);
            let pc = p.clamp(1e-15, 1.0 - 1e-15);
            ll_sum += if bar.target == 1 {
                -pc.ln()
            } else {
                -(1.0 - pc).ln()
            };
        }
        let m = scored.len().max(1) as f64;
        per_fold.push(P2FoldMetrics {
            fold: fold_idx,
            n_train: train_end,
            n_test: b - a,
            n_scored: scored.len(),
            accuracy: acc_sum / m,
            baseline_accuracy: base_sum / m,
            brier: brier_sum / m,
            log_loss: ll_sum / m,
        });
        fold_models.push((w, intercept));
    }
    if predictions.is_empty() {
        return None;
    }
    let n_oos = predictions.len();
    let acc = predictions
        .iter()
        .map(|r| f64::from(u8::from(r.proba > 0.5) == r.target))
        .sum::<f64>()
        / n_oos as f64;
    let (mut tp, mut fn_, mut tn, mut fp) = (0usize, 0usize, 0usize, 0usize);
    let mut brier_sum = 0.0f64;
    let mut ll_sum = 0.0f64;
    let mut n_pos_oos = 0usize;
    for r in predictions.iter() {
        match (r.target, r.proba > 0.5) {
            (1, true) => tp += 1,
            (1, false) => fn_ += 1,
            (0, false) => tn += 1,
            (0, true) => fp += 1,
            _ => unreachable!(),
        }
        brier_sum += (r.proba - f64::from(r.target)).powi(2);
        let pc = r.proba.clamp(1e-15, 1.0 - 1e-15);
        ll_sum += if r.target == 1 {
            -pc.ln()
        } else {
            -(1.0 - pc).ln()
        };
        n_pos_oos += usize::from(r.target == 1);
    }
    // The training-majority accuracy over the scored rows: each
    // row's fold majority label is the label that fold committed to
    // from training data alone. It is recovered from the per-fold
    // baseline accuracies weighted by scored rows.
    let mut base_correct = 0.0f64;
    for f in &per_fold {
        base_correct += f.baseline_accuracy * f.n_scored as f64;
    }
    let base_sum = base_correct / n_oos as f64;
    let balanced = if tp + fn_ == 0 || tn + fp == 0 {
        f64::NAN
    } else {
        0.5 * (tp as f64 / (tp + fn_) as f64 + tn as f64 / (tn + fp) as f64)
    };
    Some(P2Eval {
        n_labeled: n,
        n_oos,
        n_pos_oos,
        n_neg_oos: n_oos - n_pos_oos,
        n_unscored: n - n_oos,
        accuracy: acc,
        baseline: base_sum,
        brier: brier_sum / n_oos as f64,
        log_loss: ll_sum / n_oos as f64,
        balanced_accuracy: balanced,
        per_fold,
        predictions,
        fold_models,
    })
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

// ---- P1: repaired target semantics -----------------------------------
//
// The original task labels a row by the sign of the forward return
// over the next `horizon` bars, so a zero return and a negative
// return share class 0 and an unchanged market can inflate headline
// accuracy. P1 keeps that task exactly as it is (see
// `build_dataset`; `P1Dataset::baseline_bars` reproduces it row for
// row) and adds two separate labeled tracks over the same decision
// rows:
//
// * next-nonzero: the sign of the first nonzero mid change within
//   `cutoff_bars` bars after the decision bar, censored when no
//   move is observed inside the cutoff;
// * three-class: down (0), flat (1), or up (2) over the horizon,
//   with flat an explicit band |fwd| <= `flat_eps` instead of a side
//   effect of the sign.
//
// Decision rows are fixed by decision-time information only: the
// trailing 30-bar feature window must be fully observed, exactly as
// in `build_dataset`. Labels and censoring are then computed from
// future bar mids, and every decision row is accounted for in
// exactly one of labeled or censored, per track, with the reason
// counted. Nothing is ever included or excluded because of its
// outcome: the censored rows are reported, not hidden.
//
// The selective-population risk is stated, not wished away: the
// nonzero track trains and scores only on rows where a move occurs
// within the cutoff, so its accuracy is conditional on a move
// occurring, and the censored share is reported next to it. The
// label rule itself is definable at decision time (a rule over
// future observations with an explicit cutoff and censoring rule),
// which is what keeps this from being a future-eligibility leak:
// the population is all decision rows, and the labeled subset is an
// explicit, counted function of the rule.
//
// Every label traces to a concrete future mid: each row records the
// target mid and the target bar time it came from, and for the
// nonzero track the bar of the first move.
//
// The design grid is frozen (`P1_HORIZONS_BARS`,
// `P1_FLAT_EPS`, `P1_NONZERO_CUTOFF_BARS`, and
// `experiments/p1/grid.json`, which must match
// `preregistered_p1_grid`). All cells are reported; no test outcome
// selects a configuration. The default cell is horizon 3 bars with
// flat band 1e-5, the baseline horizon.

/// Frozen P1 design grid: horizons in bars for the three-class track.
pub const P1_HORIZONS_BARS: [usize; 3] = [1, 3, 6];
/// Frozen P1 design grid: flat-band half-widths as relative mid moves.
pub const P1_FLAT_EPS: [f64; 3] = [0.0, 1e-5, 3e-5];
/// Frozen next-nonzero cutoff in bars.
pub const P1_NONZERO_CUTOFF_BARS: usize = 10;
/// Default P1 cell: the baseline horizon of 3 bars.
pub const P1_DEFAULT_HORIZON: usize = 3;
/// Default P1 cell: flat band of 1e-5 relative mid move.
pub const P1_DEFAULT_FLAT_EPS: f64 = 1e-5;
/// Schema tag for `experiments/p1/grid.json`.
pub const P1_GRID_SCHEMA: &str = "toxicflow.p1-grid/v1";

/// Three-class label: down.
pub const P1_DOWN: u8 = 0;
/// Three-class label: flat.
pub const P1_FLAT: u8 = 1;
/// Three-class label: up.
pub const P1_UP: u8 = 2;

/// One frozen P1 design cell: a three-class horizon and flat band.
/// The nonzero track uses the same horizon for its baseline
/// comparison and the frozen `P1_NONZERO_CUTOFF_BARS` cutoff.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct P1Config {
    pub horizon_bars: usize,
    pub flat_eps: f64,
}

impl P1Config {
    /// Stable id, e.g. `h3-e1e-5` for horizon 3, band 1e-5.
    pub fn id(&self) -> String {
        let eps = if self.flat_eps == 0.0 {
            "e0".to_string()
        } else {
            format!("e{:e}", self.flat_eps)
        };
        format!("h{}-{eps}", self.horizon_bars)
    }

    /// True for the default cell (horizon 3, band 1e-5).
    pub fn is_default(&self) -> bool {
        self.horizon_bars == P1_DEFAULT_HORIZON && self.flat_eps == P1_DEFAULT_FLAT_EPS
    }
}

/// The default P1 cell.
pub const P1_DEFAULT_CONFIG: P1Config = P1Config {
    horizon_bars: P1_DEFAULT_HORIZON,
    flat_eps: P1_DEFAULT_FLAT_EPS,
};

/// The frozen P1 design grid, in grid-file order: horizons 1, 3, 6
/// crossed with flat bands 0, 1e-5, 3e-5.
pub fn preregistered_p1_grid() -> Vec<P1Config> {
    let mut grid = Vec::with_capacity(9);
    for &h in P1_HORIZONS_BARS.iter() {
        for &e in P1_FLAT_EPS.iter() {
            grid.push(P1Config {
                horizon_bars: h,
                flat_eps: e,
            });
        }
    }
    grid
}

/// Parse the `configs` array of a P1 grid file
/// (`experiments/p1/grid.json`). Each entry carries `horizon_bars`
/// and `flat_eps`; the entry order is preserved.
pub fn p1_grid_from_json(text: &str) -> Result<Vec<P1Config>, String> {
    let v: serde_json::Value =
        serde_json::from_str(text).map_err(|e| format!("p1 grid.json does not parse: {e}"))?;
    let configs = v
        .get("configs")
        .and_then(|c| c.as_array())
        .ok_or("p1 grid.json has no configs array")?;
    let mut out = Vec::with_capacity(configs.len());
    for (i, c) in configs.iter().enumerate() {
        let horizon = c
            .get("horizon_bars")
            .and_then(|h| h.as_u64())
            .ok_or(format!("p1 grid config {i} has no horizon_bars"))?;
        let eps = c
            .get("flat_eps")
            .and_then(|e| e.as_f64())
            .ok_or(format!("p1 grid config {i} has no flat_eps"))?;
        out.push(P1Config {
            horizon_bars: horizon as usize,
            flat_eps: eps,
        });
    }
    Ok(out)
}

// ---- P2: preregistered ablation grid ------------------------------

/// One frozen P2 design cell: a feature-column subset plus the
/// control applied at scoring time. `cols` names features from
/// `P2_FEATURE_NAMES`; `control` is one of "none", "permute",
/// "shift5".
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct P2Config {
    pub id: String,
    pub kind: String,
    pub cols: Vec<String>,
    pub control: String,
}

impl P2Config {
    /// Column indices for this config, in `P2_FEATURE_NAMES` order.
    pub fn col_indices(&self) -> Vec<usize> {
        self.cols
            .iter()
            .map(|name| {
                P2_FEATURE_NAMES
                    .iter()
                    .position(|n| n == name)
                    .unwrap_or_else(|| panic!("P2 config {} names unknown feature {name}", self.id))
            })
            .collect()
    }

    /// The scoring control for this config.
    pub fn control(&self) -> P2Control {
        match self.control.as_str() {
            "permute" => P2Control::Permute,
            "shift5" => P2Control::Shift,
            _ => P2Control::Off,
        }
    }
}

fn p2_cfg(id: &str, kind: &str, cols: &[&str], control: &str) -> P2Config {
    P2Config {
        id: id.to_string(),
        kind: kind.to_string(),
        cols: cols.iter().map(|s| s.to_string()).collect(),
        control: control.to_string(),
    }
}

/// The frozen P2 design grid, in grid-file order: the B0-B5
/// baseline ladder, single-feature additions on B3, leave-one-out
/// from B5, and permutation / 5-bar shift controls on the P2 block.
/// The B0-B5 ladder is defined here because no B0-B5 definition
/// exists in the workbook or roadmap text: B0 is the
/// training-majority predictor, B1 is OFI only, B2 is book state,
/// B3 is the v3 frame, B4 adds the flow-normalization group, and B5
/// is the full v4 frame.
pub fn preregistered_p2_grid() -> Vec<P2Config> {
    const V3: &[&str] = &["ofi", "ofi_z", "imbalance", "spread_z", "signed_vol"];
    const FLOW: &[&str] = &[
        "signed_flow_w",
        "ofi_depth",
        "signed_flow_lag3",
        "signed_flow_lag10",
        "ofi_lag3",
        "ofi_lag10",
    ];
    const ACT: &[&str] = &["quote_rate", "trade_rate", "realized_vol", "micro_disp"];
    const ALL: &[&str] = &[
        "ofi",
        "ofi_z",
        "imbalance",
        "spread_z",
        "signed_vol",
        "signed_flow_w",
        "ofi_depth",
        "signed_flow_lag3",
        "signed_flow_lag10",
        "ofi_lag3",
        "ofi_lag10",
        "quote_rate",
        "trade_rate",
        "realized_vol",
        "micro_disp",
    ];
    let mut grid = Vec::with_capacity(30);
    grid.push(p2_cfg("b0-majority", "baseline", &[], "none"));
    grid.push(p2_cfg("b1-ofi", "baseline", &["ofi"], "none"));
    grid.push(p2_cfg(
        "b2-book",
        "baseline",
        &["ofi", "ofi_z", "imbalance", "spread_z"],
        "none",
    ));
    grid.push(p2_cfg("b3-v3", "baseline", V3, "none"));
    let mut b4_cols: Vec<&str> = V3.to_vec();
    b4_cols.extend_from_slice(FLOW);
    grid.push(p2_cfg("b4-flow", "baseline", &b4_cols, "none"));
    grid.push(p2_cfg("b5-full", "default", ALL, "none"));
    // Single-feature additions on B3.
    for name in FLOW.iter().chain(ACT.iter()) {
        let mut cols: Vec<&str> = V3.to_vec();
        cols.push(name);
        grid.push(p2_cfg(&format!("add-{name}"), "add-one", &cols, "none"));
    }
    // Leave-one-out from B5.
    for name in FLOW.iter().chain(ACT.iter()) {
        let cols: Vec<&str> = ALL.iter().filter(|&&c| c != *name).copied().collect();
        grid.push(p2_cfg(
            &format!("drop-{name}"),
            "leave-one-out",
            &cols,
            "none",
        ));
    }
    // Controls on the P2 block.
    grid.push(p2_cfg("b5-permute", "control", ALL, "permute"));
    grid.push(p2_cfg("b5-shift5", "control", ALL, "shift5"));
    grid.push(p2_cfg("b4-permute", "control", &b4_cols, "permute"));
    grid.push(p2_cfg("b4-shift5", "control", &b4_cols, "shift5"));
    grid
}

/// Parse the `configs` array of a P2 grid file
/// (`experiments/p2/grid.json`). Each entry carries `id`, `kind`,
/// `cols` (feature names) and `control`; the entry order is
/// preserved.
pub fn p2_grid_from_json(text: &str) -> Result<Vec<P2Config>, String> {
    let v: serde_json::Value =
        serde_json::from_str(text).map_err(|e| format!("p2 grid.json does not parse: {e}"))?;
    let configs = v
        .get("configs")
        .and_then(|c| c.as_array())
        .ok_or("p2 grid.json has no configs array")?;
    let mut out = Vec::with_capacity(configs.len());
    for (i, c) in configs.iter().enumerate() {
        let id = c
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or(format!("p2 grid config {i} has no id"))?;
        let kind = c
            .get("kind")
            .and_then(|v| v.as_str())
            .ok_or(format!("p2 grid config {i} has no kind"))?;
        let cols = c
            .get("cols")
            .and_then(|v| v.as_array())
            .ok_or(format!("p2 grid config {i} has no cols"))?
            .iter()
            .map(|v| {
                v.as_str()
                    .map(|s| s.to_string())
                    .ok_or(format!("p2 grid config {i} has a non-string col"))
            })
            .collect::<Result<Vec<String>, String>>()?;
        let control = c
            .get("control")
            .and_then(|v| v.as_str())
            .ok_or(format!("p2 grid config {i} has no control"))?;
        out.push(P2Config {
            id: id.to_string(),
            kind: kind.to_string(),
            cols,
            control: control.to_string(),
        });
    }
    Ok(out)
}

// ---- P3: conditional toxicity pilot (feature frame v5) ------------

/// Number of features in the P3 (v5) frame: the fifteen v4 features
/// plus the two-column VPIN block.
pub const P3_N_FEATURES: usize = 17;

/// P3 feature names, in `P3Bar::features` order.
pub const P3_FEATURE_NAMES: [&str; P3_N_FEATURES] = [
    "ofi",
    "ofi_z",
    "imbalance",
    "spread_z",
    "signed_vol",
    "signed_flow_w",
    "ofi_depth",
    "signed_flow_lag3",
    "signed_flow_lag10",
    "ofi_lag3",
    "ofi_lag10",
    "quote_rate",
    "trade_rate",
    "realized_vol",
    "micro_disp",
    "vpin",
    "vpin_lag3",
];

/// Feature-frame version of `build_p3_dataset`. v5 adds the VPIN
/// block to the v4 frame; the first five columns reproduce
/// `build_dataset` bit-identically on shared rows (pinned by
/// `stage3_toxicflow_3_p3_v3_columns_bit_identical`), so the v3
/// baseline is preserved exactly inside the new frame.
pub const P3_FEATURE_VERSION: &str = "v5";

/// Frozen bucket volume behind the VPIN block, in BTC. A fixed
/// absolute volume, never derived from a session total, so the
/// bucketing carries no session-level lookahead. 15 BTC matches the
/// export convention of 50 buckets over the roughly 750 BTC main
/// session; the six frozen test sessions span 1,162 to 6,665 BTC,
/// so one session yields 77 to 444 buckets.
pub const P3_BUCKET_VOL_BTC: f64 = 15.0;

/// VPIN window, in completed buckets. A row is scored only once a
/// full window of buckets has completed before its bar close, so
/// the statistic is comparable across rows.
pub const P3_VPIN_WINDOW: usize = 20;

/// Lag of the second VPIN column, in bars (30 s on the 10 s grid).
pub const P3_VPIN_LAG_BARS: usize = 3;

/// Seed for the permutation control's deterministic shuffle.
pub const P3_PERM_SEED: u64 = 0x5EED_5EED_0000_00A3;

/// Shift lag for the shift control, in bars.
pub const P3_SHIFT_LAG: usize = 5;

/// Column subsets for the P3 cells: B0 is the empty list
/// (training-majority predictor), B3 is the v3 frame, B5 is the
/// full v4 frame (the controls-only model), VPIN is the VPIN block
/// alone, and the controls+VPIN frame is all seventeen columns.
pub const P3_B0_COLS: &[usize] = &[];
pub const P3_B3_COLS: &[usize] = &[0, 1, 2, 3, 4];
pub const P3_B5_COLS: &[usize] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
pub const P3_B3_VPIN_COLS: &[usize] = &[0, 1, 2, 3, 4, 15, 16];
pub const P3_VPIN_COLS: &[usize] = &[15, 16];
pub const P3_CONTROLS_VPIN_COLS: &[usize] =
    &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

/// One v5 decision row.
#[derive(Debug, Clone)]
pub struct P3Bar {
    /// Start time of the OFI bar this row was built from.
    pub time_ms: i64,
    pub features: [f64; P3_N_FEATURES],
    /// Direction label: sign of the forward mid return.
    pub target_direction: u8,
    /// Absolute forward mid return, for the train-median
    /// absolute-return outcome.
    pub fwd_abs: f64,
    /// Spread-widening label: spread at the horizon bar strictly
    /// above the spread at the decision bar.
    pub target_spread_wide: u8,
    /// Conditional adverse-markout label: 1 when the forward return
    /// runs against the bar's net aggressive flow side. `None`
    /// exactly when the bar's net signed flow is zero, so those
    /// rows are excluded from that outcome only.
    pub target_adverse: Option<u8>,
    pub spread_degenerate: bool,
    /// Trailing 30-bar mean trade count per bar (regime key).
    pub trade_rate: f64,
    /// Trailing 30-bar mean touch depth (regime key).
    pub depth_mean: f64,
}

/// The three P3 labels from one forward return, the bar's net
/// signed flow, and the spreads at the decision and horizon bars.
/// A zero forward return is direction 0; equal spreads are not a
/// widening; a zero forward return is never adverse to either side.
pub fn p3_labels(
    fwd: f64,
    signed_vol: f64,
    spread_now: f64,
    spread_fwd: f64,
) -> (u8, u8, Option<u8>) {
    let direction = u8::from(fwd > 0.0);
    let spread_wide = u8::from(spread_fwd > spread_now);
    let adverse = if signed_vol == 0.0 {
        None
    } else {
        Some(u8::from(
            (signed_vol > 0.0 && fwd < 0.0) || (signed_vol < 0.0 && fwd > 0.0),
        ))
    };
    (direction, spread_wide, adverse)
}

/// Median of a slice (NaN on an empty slice). Copies before
/// sorting, so callers' order is untouched.
pub fn p3_median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut xs = values.to_vec();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = xs.len();
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        (xs[n / 2 - 1] + xs[n / 2]) / 2.0
    }
}

/// Tertile cut points of a slice: the sorted values at indices
/// n/3 and 2n/3 (NaN, NaN on an empty slice). A row is tertile 0
/// at or below the first cut, tertile 1 at or below the second,
/// tertile 2 above it; ties at a cut land in the lower tertile.
pub fn p3_tertile_cuts(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mut xs = values.to_vec();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    (xs[xs.len() / 3], xs[2 * xs.len() / 3])
}

/// Tertile of one value against train-only cut points.
pub fn p3_tertile(x: f64, cut1: f64, cut2: f64) -> usize {
    if x <= cut1 {
        0
    } else if x <= cut2 {
        1
    } else {
        2
    }
}

/// Build the v5 feature/target frame. Returns empty vec when data
/// is thin (fewer than 40 bars). Rows start at bar 29 so every
/// trailing 30-bar window is full, share `build_dataset`'s gap
/// censoring, and additionally require a full 20-bucket VPIN window
/// completed before the bar close, plus a defined VPIN 3 bars
/// earlier for the lag column. The first five columns reproduce
/// `build_dataset` bit-identically on shared rows.
pub fn build_p3_dataset(
    quotes: &[Quote],
    trades: &[Trade],
    bar_ms: i64,
    horizon: usize,
) -> Vec<P3Bar> {
    // H4: `ofi_bars` requires time-ordered quotes. Sort stably so a
    // delayed message still lands in its own bar, keeping its arrival
    // order inside the bar; already-ordered input is unchanged.
    let mut ordered: Vec<Quote> = quotes.to_vec();
    ordered.sort_by_key(|q| q.time_ms);
    let quotes = &ordered[..];
    let events = ofi_events(quotes);
    // H4: an invalid bar grid yields no rows here, the same thin-data
    // outcome as before; `ofi_bars` itself now reports the cause.
    let bars = ofi_bars(quotes, &events, bar_ms).unwrap_or_default();
    if bars.len() < 40 {
        return Vec::new();
    }
    let touch = touch_bars(quotes, bar_ms);
    let n = bars.len();
    let ofi: Vec<f64> = bars.iter().map(|b| b.1).collect();
    let mids: Vec<f64> = bars.iter().map(|b| b.2).collect();
    let spreads: Vec<f64> = bars.iter().map(|b| b.3).collect();
    let imbs: Vec<f64> = bars.iter().map(|b| b.4).collect();

    let t0 = bars[0].0;
    let mut signed = vec![0.0; n];
    let mut trade_counts = vec![0usize; n];
    for t in trades {
        let idx = ((t.time_ms - t0) / bar_ms) as usize;
        if idx < n {
            signed[idx] += t.signed_qty();
            trade_counts[idx] += 1;
        }
    }
    let quote_counts: Vec<f64> = touch.iter().map(|b| b.n_quotes as f64).collect();
    let trade_counts_f: Vec<f64> = trade_counts.iter().map(|&c| c as f64).collect();
    let depths: Vec<f64> = touch.iter().map(|b| b.depth_mean).collect();

    let (ofi_m, ofi_s) = rolling_mean_sd(&ofi, 30);
    let (sp_m, sp_s) = rolling_mean_sd(&spreads, 30);
    let depth_mean = trailing_mean(&depths, P2_TRAIL_WINDOW_BARS);
    let quote_rate = trailing_mean(&quote_counts, P2_TRAIL_WINDOW_BARS);
    let trade_rate = trailing_mean(&trade_counts_f, P2_TRAIL_WINDOW_BARS);
    let signed_flow_w = trailing_sum(&signed, P2_FLOW_WINDOW_BARS);
    let realized_vol = trailing_realized_vol(&mids, P2_TRAIL_WINDOW_BARS);

    // VPIN alignment: buckets of the frozen volume, in time order.
    // `vpin` returns one entry per completed bucket; entry k holds
    // the mean imbalance over the last P3_VPIN_WINDOW buckets once
    // k + 1 reaches the window. A bar's VPIN is the entry of the
    // last bucket completed strictly before the bar close, and is
    // defined only once a full window has completed, so no row ever
    // reads a bucket that completes after its close.
    let buckets = volume_buckets_sized(trades, P3_BUCKET_VOL_BTC);
    let vp = vpin(&buckets, P3_VPIN_WINDOW);
    let vpin_at = |bar_end: i64| -> Option<f64> {
        let k = vp.partition_point(|(t, _)| *t < bar_end);
        if k >= P3_VPIN_WINDOW {
            Some(vp[k - 1].1)
        } else {
            None
        }
    };

    let status = crate::metrics::bar_status(quotes, bar_ms);
    let mut out = Vec::new();
    for i in 0..n {
        if i < P2_TRAIL_WINDOW_BARS - 1 || i + horizon >= n || ofi_s[i].is_nan() || sp_s[i].is_nan()
        {
            continue;
        }
        if status.len() == n {
            let lo = i.saturating_sub(P2_TRAIL_WINDOW_BARS - 1);
            let features_observed = status[lo..=i].iter().all(|s| s.observed);
            let target_observed = status[i + 1..=i + horizon].iter().all(|s| s.observed);
            if !features_observed || !target_observed {
                continue;
            }
        }
        if i < P3_VPIN_LAG_BARS {
            continue;
        }
        let bar_end = bars[i].0 + bar_ms;
        let Some(v) = vpin_at(bar_end) else {
            continue;
        };
        let lag_end = bars[i - P3_VPIN_LAG_BARS].0 + bar_ms;
        let Some(v_lag) = vpin_at(lag_end) else {
            continue;
        };
        let fwd = mids[i + horizon] / mids[i] - 1.0;
        let spread_degenerate = sp_s[i] == 0.0;
        let (direction, spread_wide, adverse) =
            p3_labels(fwd, signed[i], spreads[i], spreads[i + horizon]);
        let mut f = [0.0; P3_N_FEATURES];
        f[0] = ofi[i];
        f[1] = (ofi[i] - ofi_m[i]) / (ofi_s[i] + 1e-12);
        f[2] = imbs[i];
        f[3] = if spread_degenerate {
            0.0
        } else {
            (spreads[i] - sp_m[i]) / (sp_s[i] + 1e-12)
        };
        f[4] = signed[i];
        f[5] = signed_flow_w[i];
        f[6] = if depth_mean[i] == 0.0 {
            0.0
        } else {
            ofi[i] / depth_mean[i]
        };
        f[7] = signed[i - P2_LAG_BARS[0]];
        f[8] = signed[i - P2_LAG_BARS[1]];
        f[9] = ofi[i - P2_LAG_BARS[0]];
        f[10] = ofi[i - P2_LAG_BARS[1]];
        f[11] = quote_rate[i];
        f[12] = trade_rate[i];
        f[13] = realized_vol[i];
        f[14] = touch[i].micro_disp;
        f[15] = v;
        f[16] = v_lag;
        out.push(P3Bar {
            time_ms: bars[i].0,
            features: f,
            target_direction: direction,
            fwd_abs: fwd.abs(),
            target_spread_wide: spread_wide,
            target_adverse: adverse,
            spread_degenerate,
            trade_rate: trade_rate[i],
            depth_mean: depth_mean[i],
        });
    }
    out
}

/// How the P3 VPIN block is treated at scoring time. The fitted
/// model is identical under every control: only the scored test
/// rows change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum P3Control {
    /// Score each row on its own features.
    Off,
    /// Within each fold's test block, the VPIN block of each row is
    /// replaced by the VPIN block of another test row, drawn by a
    /// fixed-seed shuffle. Breaks feature-target alignment while
    /// preserving the block's joint distribution.
    Permute,
    /// Within each fold's test block, the VPIN block of each row is
    /// replaced by the VPIN block of the test row `P3_SHIFT_LAG`
    /// bars earlier. The first `P3_SHIFT_LAG` rows of each block
    /// are unscored.
    Shift,
}

/// One P3 outcome. The absolute-return outcome resolves its labels
/// per fold from the training median of `fwd_abs`; the other three
/// outcomes carry fixed labels built with the dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum P3Outcome {
    /// Sign of the 3-bar forward mid return.
    Direction,
    /// 3-bar absolute forward return above the training median.
    AbsRet,
    /// Spread at the horizon bar above the spread at the decision bar.
    SpreadWide,
    /// Forward return against the bar's net aggressive flow side.
    AdverseMarkout,
}

impl P3Outcome {
    /// Stable id used in result packets.
    pub fn id(self) -> &'static str {
        match self {
            P3Outcome::Direction => "direction",
            P3Outcome::AbsRet => "absret",
            P3Outcome::SpreadWide => "spread_wide",
            P3Outcome::AdverseMarkout => "adverse_markout",
        }
    }
}

/// One scored P3 row.
#[derive(Debug, Clone)]
pub struct P3Prediction {
    pub fold: usize,
    pub row: usize,
    pub time_ms: i64,
    pub target: u8,
    pub proba: f64,
}

/// One fold's held-out scores for a P3 evaluation, with the
/// train-only thresholds behind it: the activity and depth tertile
/// cuts and, for the absolute-return outcome, the training median
/// of `fwd_abs` (NaN for the other outcomes).
#[derive(Debug, Clone)]
pub struct P3FoldMetrics {
    pub fold: usize,
    pub n_train: usize,
    pub n_test: usize,
    /// Rows actually scored (fewer than `n_test` under Shift).
    pub n_scored: usize,
    pub accuracy: f64,
    pub baseline_accuracy: f64,
    pub brier: f64,
    pub log_loss: f64,
    pub activity_cut1: f64,
    pub activity_cut2: f64,
    pub depth_cut1: f64,
    pub depth_cut2: f64,
    pub absret_median: f64,
}

/// A P3 walk-forward evaluation on one column subset and one
/// outcome.
#[derive(Debug, Clone)]
pub struct P3Eval {
    pub outcome: P3Outcome,
    pub n_labeled: usize,
    /// Held-out predictions scored: the exact denominator behind
    /// accuracy, Brier score and log loss.
    pub n_oos: usize,
    pub n_pos_oos: usize,
    pub n_neg_oos: usize,
    /// Labeled rows the evaluation never scores: training rows,
    /// purge-gap rows, and (under Shift) warm-up rows.
    pub n_unscored: usize,
    pub accuracy: f64,
    /// Out-of-sample accuracy of the training-majority predictor on
    /// the same scored rows.
    pub baseline: f64,
    pub brier: f64,
    pub log_loss: f64,
    pub balanced_accuracy: f64,
    pub per_fold: Vec<P3FoldMetrics>,
    pub predictions: Vec<P3Prediction>,
    /// (raw-space weights in `cols` order, intercept) per fold.
    /// The standardization behind each fold is learned from that
    /// fold's training rows only.
    pub fold_models: Vec<(Vec<f64>, f64)>,
}

/// Full-batch gradient descent for L2 logistic regression on a
/// column subset of the P3 frame. Standardization (mean and
/// population sd, sd floor 1e-12) is learned from the supplied rows
/// only, exactly as in `fit_logistic`; callers pass training rows.
/// Returns (raw-space weights in `cols` order, intercept).
fn fit_logistic_p3_cols(
    xs: &[[f64; P3_N_FEATURES]],
    y: &[u8],
    cols: &[usize],
    l2: f64,
    iters: usize,
    lr: f64,
) -> (Vec<f64>, f64) {
    let n = xs.len() as f64;
    let d = cols.len();
    let mut mu = vec![0.0; d];
    let mut sd = vec![0.0; d];
    for row in xs {
        for (k, &j) in cols.iter().enumerate() {
            mu[k] += row[j];
        }
    }
    for m in mu.iter_mut() {
        *m /= n;
    }
    for row in xs {
        for (k, &j) in cols.iter().enumerate() {
            let z = row[j] - mu[k];
            sd[k] += z * z;
        }
    }
    for s in sd.iter_mut() {
        *s = (*s / n).sqrt() + 1e-12;
    }
    let mut w = vec![0.0; d + 1];
    for _ in 0..iters {
        let mut grad = vec![0.0; d + 1];
        for (row, &yi) in xs.iter().zip(y.iter()) {
            let mut z = w[d];
            for (k, &j) in cols.iter().enumerate() {
                z += w[k] * (row[j] - mu[k]) / sd[k];
            }
            let err = sigmoid(z) - yi as f64;
            for (k, &j) in cols.iter().enumerate() {
                grad[k] += err * (row[j] - mu[k]) / sd[k];
            }
            grad[d] += err;
        }
        for k in 0..d {
            w[k] -= lr * (grad[k] / n + l2 * w[k] / n);
        }
        w[d] -= lr * grad[d] / n;
    }
    let mut raw = vec![0.0; d];
    let mut b = w[d];
    for k in 0..d {
        raw[k] = w[k] / sd[k];
        b -= w[k] * mu[k] / sd[k];
    }
    (raw, b)
}

fn predict_proba_p3_cols(
    x: &[f64; P3_N_FEATURES],
    cols: &[usize],
    w: &[f64],
    intercept: f64,
) -> f64 {
    let mut z = intercept;
    for (k, &j) in cols.iter().enumerate() {
        z += w[k] * x[j];
    }
    sigmoid(z)
}

/// Purged walk-forward evaluation of one P3 column subset on one
/// outcome, with the same fold geometry as `walk_forward_p2` (fold
/// = n / (n_splits + 1), purge gap = `horizon` bars). An empty
/// column list is the B0 training-majority predictor: every row
/// scores the training positive rate. The adverse-markout outcome
/// scores only rows whose bar carries nonzero net signed flow;
/// every other outcome scores the full row set, so all configs
/// compared on one outcome share the same folds and observations.
pub fn walk_forward_p3(
    bars: &[P3Bar],
    cols: &[usize],
    control: P3Control,
    n_splits: usize,
    horizon: usize,
    outcome: P3Outcome,
) -> Option<P3Eval> {
    let rows: Vec<&P3Bar> = match outcome {
        P3Outcome::AdverseMarkout => bars.iter().filter(|r| r.target_adverse.is_some()).collect(),
        _ => bars.iter().collect(),
    };
    // Original indices into `bars`, so predictions stay addressable
    // against the unfiltered row set even when the outcome filters
    // rows out (adverse markout drops zero-flow rows).
    let orig: Vec<usize> = match outcome {
        P3Outcome::AdverseMarkout => bars
            .iter()
            .enumerate()
            .filter(|(_, r)| r.target_adverse.is_some())
            .map(|(i, _)| i)
            .collect(),
        _ => (0..bars.len()).collect(),
    };
    let n = rows.len();
    if n < 60 {
        return None;
    }
    let fold = n / (n_splits + 1);
    if fold < 10 {
        return None;
    }
    let mut per_fold: Vec<P3FoldMetrics> = Vec::new();
    let mut predictions: Vec<P3Prediction> = Vec::new();
    let mut fold_models: Vec<(Vec<f64>, f64)> = Vec::new();
    for k in 1..=n_splits {
        let a = k * fold;
        let b = ((k + 1) * fold).min(n);
        let train_end = a.saturating_sub(horizon);
        if train_end < 30 || b - a < 10 {
            continue;
        }
        let train = &rows[..train_end];
        let test = &rows[a..b];
        // Train-only thresholds: the absolute-return median and the
        // activity/depth tertile cuts are learned from the training
        // rows of this fold only.
        let absret_median = match outcome {
            P3Outcome::AbsRet => p3_median(&train.iter().map(|r| r.fwd_abs).collect::<Vec<_>>()),
            _ => f64::NAN,
        };
        let (activity_cut1, activity_cut2) =
            p3_tertile_cuts(&train.iter().map(|r| r.trade_rate).collect::<Vec<_>>());
        let (depth_cut1, depth_cut2) =
            p3_tertile_cuts(&train.iter().map(|r| r.depth_mean).collect::<Vec<_>>());
        let label = |r: &P3Bar| -> u8 {
            match outcome {
                P3Outcome::Direction => r.target_direction,
                P3Outcome::AbsRet => u8::from(r.fwd_abs > absret_median),
                P3Outcome::SpreadWide => r.target_spread_wide,
                P3Outcome::AdverseMarkout => r.target_adverse.unwrap_or(0),
            }
        };
        let ys: Vec<u8> = train.iter().map(|r| label(r)).collect();
        let pos = ys.iter().map(|&y| usize::from(y)).sum::<usize>();
        let (majority_label, _majority_n) = if pos * 2 >= ys.len() {
            (1u8, pos)
        } else {
            (0u8, ys.len() - pos)
        };
        let (w, intercept) = if cols.is_empty() {
            let p = (pos as f64 / ys.len() as f64).clamp(1e-6, 1.0 - 1e-6);
            (Vec::new(), (p / (1.0 - p)).ln())
        } else {
            let xs: Vec<[f64; P3_N_FEATURES]> = train.iter().map(|x| x.features).collect();
            fit_logistic_p3_cols(&xs, &ys, cols, FIT_L2, FIT_ITERS, FIT_LR)
        };
        // (filtered position, original row index, scored features).
        let mut scored: Vec<(usize, usize, [f64; P3_N_FEATURES])> = Vec::new();
        match control {
            P3Control::Off => {
                for (off, row) in test.iter().enumerate() {
                    scored.push((a + off, orig[a + off], row.features));
                }
            }
            P3Control::Permute => {
                let perm = p2_permutation(test.len(), P3_PERM_SEED ^ (k as u64));
                for (off, row) in test.iter().enumerate() {
                    let mut f = row.features;
                    let donor = test[perm[off]].features;
                    for &j in P3_VPIN_COLS {
                        f[j] = donor[j];
                    }
                    scored.push((a + off, orig[a + off], f));
                }
            }
            P3Control::Shift => {
                for (off, row) in test.iter().enumerate() {
                    if off < P3_SHIFT_LAG {
                        continue;
                    }
                    let mut f = row.features;
                    let donor = test[off - P3_SHIFT_LAG].features;
                    for &j in P3_VPIN_COLS {
                        f[j] = donor[j];
                    }
                    scored.push((a + off, orig[a + off], f));
                }
            }
        }
        let fold_idx = per_fold.len();
        let mut acc_sum = 0.0f64;
        let mut base_sum = 0.0f64;
        let mut brier_sum = 0.0f64;
        let mut ll_sum = 0.0f64;
        for (pos, row_idx, f) in &scored {
            let bar = &rows[*pos];
            let target = label(bar);
            let p = if cols.is_empty() {
                sigmoid(intercept)
            } else {
                predict_proba_p3_cols(f, cols, &w, intercept)
            };
            predictions.push(P3Prediction {
                fold: fold_idx,
                row: *row_idx,
                time_ms: bar.time_ms,
                target,
                proba: p,
            });
            acc_sum += f64::from(u8::from(p > 0.5) == target);
            base_sum += f64::from(majority_label == target);
            brier_sum += (p - f64::from(target)).powi(2);
            let pc = p.clamp(1e-15, 1.0 - 1e-15);
            ll_sum += if target == 1 {
                -pc.ln()
            } else {
                -(1.0 - pc).ln()
            };
        }
        let m = scored.len().max(1) as f64;
        per_fold.push(P3FoldMetrics {
            fold: fold_idx,
            n_train: train_end,
            n_test: b - a,
            n_scored: scored.len(),
            accuracy: acc_sum / m,
            baseline_accuracy: base_sum / m,
            brier: brier_sum / m,
            log_loss: ll_sum / m,
            activity_cut1,
            activity_cut2,
            depth_cut1,
            depth_cut2,
            absret_median,
        });
        fold_models.push((w, intercept));
    }
    if predictions.is_empty() {
        return None;
    }
    let n_oos = predictions.len();
    let acc = predictions
        .iter()
        .map(|r| f64::from(u8::from(r.proba > 0.5) == r.target))
        .sum::<f64>()
        / n_oos as f64;
    let (mut tp, mut fn_, mut tn, mut fp) = (0usize, 0usize, 0usize, 0usize);
    let mut brier_sum = 0.0f64;
    let mut ll_sum = 0.0f64;
    let mut n_pos_oos = 0usize;
    for r in predictions.iter() {
        match (r.target, r.proba > 0.5) {
            (1, true) => tp += 1,
            (1, false) => fn_ += 1,
            (0, false) => tn += 1,
            (0, true) => fp += 1,
            _ => unreachable!(),
        }
        brier_sum += (r.proba - f64::from(r.target)).powi(2);
        let pc = r.proba.clamp(1e-15, 1.0 - 1e-15);
        ll_sum += if r.target == 1 {
            -pc.ln()
        } else {
            -(1.0 - pc).ln()
        };
        n_pos_oos += usize::from(r.target == 1);
    }
    // The training-majority accuracy over the scored rows: each
    // row's fold majority label is the label that fold committed to
    // from training data alone. It is recovered from the per-fold
    // baseline accuracies weighted by scored rows.
    let mut base_correct = 0.0f64;
    for f in &per_fold {
        base_correct += f.baseline_accuracy * f.n_scored as f64;
    }
    let base_sum = base_correct / n_oos as f64;
    let balanced = if tp + fn_ == 0 || tn + fp == 0 {
        f64::NAN
    } else {
        0.5 * (tp as f64 / (tp + fn_) as f64 + tn as f64 / (tn + fp) as f64)
    };
    Some(P3Eval {
        outcome,
        n_labeled: n,
        n_oos,
        n_pos_oos,
        n_neg_oos: n_oos - n_pos_oos,
        n_unscored: n - n_oos,
        accuracy: acc,
        baseline: base_sum,
        brier: brier_sum / n_oos as f64,
        log_loss: ll_sum / n_oos as f64,
        balanced_accuracy: balanced,
        per_fold,
        predictions,
        fold_models,
    })
}

// ---- P3: preregistered pilot grid ---------------------------------

/// One frozen P3 design cell: a feature-column subset plus the
/// control applied at scoring time. `cols` names features from
/// `P3_FEATURE_NAMES`; `control` is one of "none", "permute",
/// "shift5".
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct P3Config {
    pub id: String,
    pub kind: String,
    pub cols: Vec<String>,
    pub control: String,
}

impl P3Config {
    /// Column indices for this config, in `P3_FEATURE_NAMES` order.
    pub fn col_indices(&self) -> Vec<usize> {
        self.cols
            .iter()
            .map(|name| {
                P3_FEATURE_NAMES
                    .iter()
                    .position(|n| n == name)
                    .unwrap_or_else(|| panic!("P3 config {} names unknown feature {name}", self.id))
            })
            .collect()
    }

    /// The scoring control for this config.
    pub fn control(&self) -> P3Control {
        match self.control.as_str() {
            "permute" => P3Control::Permute,
            "shift5" => P3Control::Shift,
            _ => P3Control::Off,
        }
    }
}

fn p3_cfg(id: &str, kind: &str, cols: &[&str], control: &str) -> P3Config {
    P3Config {
        id: id.to_string(),
        kind: kind.to_string(),
        cols: cols.iter().map(|s| s.to_string()).collect(),
        control: control.to_string(),
    }
}

/// The frozen P3 design grid, in grid-file order: the reference
/// ladder (majority, v3 frame, v4 controls-only frame), the paired
/// VPIN cells (v3 plus VPIN, VPIN only, controls plus VPIN, the
/// default cell), and the permutation / 5-bar shift controls on the
/// VPIN block.
pub fn preregistered_p3_grid() -> Vec<P3Config> {
    const V3: &[&str] = &["ofi", "ofi_z", "imbalance", "spread_z", "signed_vol"];
    const V4: &[&str] = &[
        "ofi",
        "ofi_z",
        "imbalance",
        "spread_z",
        "signed_vol",
        "signed_flow_w",
        "ofi_depth",
        "signed_flow_lag3",
        "signed_flow_lag10",
        "ofi_lag3",
        "ofi_lag10",
        "quote_rate",
        "trade_rate",
        "realized_vol",
        "micro_disp",
    ];
    const VPIN: &[&str] = &["vpin", "vpin_lag3"];
    const ALL: &[&str] = &[
        "ofi",
        "ofi_z",
        "imbalance",
        "spread_z",
        "signed_vol",
        "signed_flow_w",
        "ofi_depth",
        "signed_flow_lag3",
        "signed_flow_lag10",
        "ofi_lag3",
        "ofi_lag10",
        "quote_rate",
        "trade_rate",
        "realized_vol",
        "micro_disp",
        "vpin",
        "vpin_lag3",
    ];
    let mut grid = Vec::with_capacity(8);
    grid.push(p3_cfg("b0-majority", "baseline", &[], "none"));
    grid.push(p3_cfg("b3-v3", "baseline", V3, "none"));
    grid.push(p3_cfg("b5-controls", "baseline", V4, "none"));
    let mut b3_vpin: Vec<&str> = V3.to_vec();
    b3_vpin.extend_from_slice(VPIN);
    grid.push(p3_cfg("b3-vpin", "paired", &b3_vpin, "none"));
    grid.push(p3_cfg("vpin-only", "paired", VPIN, "none"));
    grid.push(p3_cfg("controls-vpin", "default", ALL, "none"));
    grid.push(p3_cfg("controls-vpin-permute", "control", ALL, "permute"));
    grid.push(p3_cfg("controls-vpin-shift5", "control", ALL, "shift5"));
    grid
}

/// Parse the `configs` array of a P3 grid file
/// (`experiments/p3/grid.json`). Each entry carries `id`, `kind`,
/// `cols` (feature names) and `control`; the entry order is
/// preserved.
pub fn p3_grid_from_json(text: &str) -> Result<Vec<P3Config>, String> {
    let v: serde_json::Value =
        serde_json::from_str(text).map_err(|e| format!("p3 grid.json does not parse: {e}"))?;
    let configs = v
        .get("configs")
        .and_then(|c| c.as_array())
        .ok_or("p3 grid.json has no configs array")?;
    let mut out = Vec::with_capacity(configs.len());
    for (i, c) in configs.iter().enumerate() {
        let id = c
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or(format!("p3 grid config {i} has no id"))?;
        let kind = c
            .get("kind")
            .and_then(|v| v.as_str())
            .ok_or(format!("p3 grid config {i} has no kind"))?;
        let cols = c
            .get("cols")
            .and_then(|v| v.as_array())
            .ok_or(format!("p3 grid config {i} has no cols"))?
            .iter()
            .map(|v| {
                v.as_str()
                    .map(|s| s.to_string())
                    .ok_or(format!("p3 grid config {i} has a non-string col"))
            })
            .collect::<Result<Vec<String>, String>>()?;
        let control = c
            .get("control")
            .and_then(|v| v.as_str())
            .ok_or(format!("p3 grid config {i} has no control"))?;
        out.push(P3Config {
            id: id.to_string(),
            kind: kind.to_string(),
            cols,
            control: control.to_string(),
        });
    }
    Ok(out)
}

/// One P1 decision row: the shared feature frame plus all three
/// labels. A label is `None` exactly when the row is censored for
/// that track; the reason is counted in `P1Counts`.
#[derive(Debug, Clone)]
pub struct P1Row {
    /// Index of the decision bar on the OFI bar grid.
    pub decision_bar: usize,
    /// Decision timestamp: the start time of the decision bar.
    pub decision_time_ms: i64,
    /// Feature values in `FEATURE_NAMES` order.
    pub features: [f64; N_FEATURES],
    pub spread_degenerate: bool,
    /// Preserved baseline label (sign of the forward return).
    pub baseline: Option<u8>,
    pub baseline_target_mid: Option<f64>,
    pub baseline_target_time_ms: Option<i64>,
    /// Next-nonzero label: 0 down, 1 up.
    pub nonzero: Option<u8>,
    pub nonzero_target_mid: Option<f64>,
    pub nonzero_target_time_ms: Option<i64>,
    /// Bar of the first nonzero mid change.
    pub nonzero_move_bar: Option<usize>,
    /// Three-class label: 0 down, 1 flat, 2 up.
    pub three: Option<u8>,
    pub three_target_mid: Option<f64>,
    pub three_target_time_ms: Option<i64>,
}

/// Reconciled sample counts for a P1 dataset. Every track satisfies
/// labeled + censored == `n_decision`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct P1Counts {
    /// OFI bars on the grid.
    pub n_bars: usize,
    /// Candidate decision rows (feature window defined and observed).
    pub n_decision: usize,
    pub baseline_labeled: usize,
    pub baseline_censored: usize,
    /// Labeled baseline rows whose forward return is exactly zero:
    /// the rows the old binary rule lumps with negatives.
    pub baseline_zero_fwd: usize,
    pub nonzero_labeled: usize,
    pub nonzero_up: usize,
    pub nonzero_down: usize,
    /// Complete observed cutoff window with no nonzero move.
    pub nonzero_censored_no_move: usize,
    /// Window runs past the last bar before any move is observed.
    pub nonzero_censored_incomplete: usize,
    /// Window up to the first move touches an unobserved bar, or a
    /// move-free window touches one.
    pub nonzero_censored_gap: usize,
    pub three_labeled: usize,
    pub three_down: usize,
    pub three_flat: usize,
    pub three_up: usize,
    /// Target window runs past the last bar.
    pub three_censored_incomplete: usize,
    /// Target window touches an unobserved bar.
    pub three_censored_gap: usize,
}

/// A P1 dataset: decision rows with all three labels, plus the
/// reconciled counts and the frozen cell that produced them.
#[derive(Debug, Clone)]
pub struct P1Dataset {
    pub rows: Vec<P1Row>,
    pub counts: P1Counts,
    pub config: P1Config,
    pub cutoff_bars: usize,
    /// Feature-frame version of the shared frame (see
    /// `FEATURE_VERSION`): the P1 tracks reuse the v3 frame, so the
    /// baseline comparison is over identical features.
    pub feature_version: &'static str,
}

impl P1Dataset {
    /// The preserved baseline frame: exactly the rows
    /// `build_dataset` returns for the same quotes and horizon,
    /// with the same features and targets (pinned by test).
    pub fn baseline_bars(&self) -> Vec<Bar> {
        self.rows
            .iter()
            .filter_map(|r| {
                r.baseline.map(|t| Bar {
                    time_ms: r.decision_time_ms,
                    ofi: r.features[0],
                    ofi_z: r.features[1],
                    imbalance: r.features[2],
                    spread_z: r.features[3],
                    spread_degenerate: r.spread_degenerate,
                    signed_vol: r.features[4],
                    target: t,
                })
            })
            .collect()
    }

    /// The next-nonzero labeled rows as binary bars (target 0 down,
    /// 1 up), in decision order.
    pub fn nonzero_bars(&self) -> Vec<Bar> {
        self.rows
            .iter()
            .filter_map(|r| {
                r.nonzero.map(|t| Bar {
                    time_ms: r.decision_time_ms,
                    ofi: r.features[0],
                    ofi_z: r.features[1],
                    imbalance: r.features[2],
                    spread_z: r.features[3],
                    spread_degenerate: r.spread_degenerate,
                    signed_vol: r.features[4],
                    target: t,
                })
            })
            .collect()
    }

    /// The three-class labeled rows, in decision order.
    pub fn three_class_rows(&self) -> Vec<ClassRow> {
        self.rows
            .iter()
            .filter_map(|r| {
                r.three.map(|t| ClassRow {
                    time_ms: r.decision_time_ms,
                    features: r.features,
                    target: t,
                })
            })
            .collect()
    }
}

/// Build the P1 dataset: the shared v3 feature frame (the same
/// computation as `build_dataset`, row for row) with all three
/// labels attached. `config` sets the three-class horizon and flat
/// band and the baseline horizon; `cutoff_bars` is the next-nonzero
/// cutoff. Thin data (fewer than 40 bars) yields no decision rows,
/// mirroring `build_dataset`.
pub fn build_p1_dataset(
    quotes: &[Quote],
    trades: &[Trade],
    bar_ms: i64,
    config: P1Config,
    cutoff_bars: usize,
) -> P1Dataset {
    let horizon = config.horizon_bars;
    let eps = config.flat_eps;
    let empty = |n_bars: usize| P1Dataset {
        rows: Vec::new(),
        counts: P1Counts {
            n_bars,
            ..Default::default()
        },
        config,
        cutoff_bars,
        feature_version: FEATURE_VERSION,
    };
    // H4: `ofi_bars` requires time-ordered quotes. Sort stably so a
    // delayed message still lands in its own bar, keeping its arrival
    // order inside the bar; already-ordered input is unchanged.
    let mut ordered: Vec<Quote> = quotes.to_vec();
    ordered.sort_by_key(|q| q.time_ms);
    let quotes = &ordered[..];
    let events = ofi_events(quotes);
    // H4: an invalid bar grid yields no rows here, the same thin-data
    // outcome as before; `ofi_bars` itself now reports the cause.
    let bars = ofi_bars(quotes, &events, bar_ms).unwrap_or_default();
    if bars.len() < 40 {
        return empty(bars.len());
    }
    let n = bars.len();
    let ofi: Vec<f64> = bars.iter().map(|b| b.1).collect();
    let mids: Vec<f64> = bars.iter().map(|b| b.2).collect();
    let spreads: Vec<f64> = bars.iter().map(|b| b.3).collect();
    let imbs: Vec<f64> = bars.iter().map(|b| b.4).collect();
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
    let status = crate::metrics::bar_status(quotes, bar_ms);
    let window_observed = |lo: usize, hi: usize| -> bool {
        status.len() == n && status[lo..=hi].iter().all(|s| s.observed)
    };
    let mut rows = Vec::new();
    let mut counts = P1Counts {
        n_bars: n,
        ..Default::default()
    };
    for i in 0..n {
        if ofi_s[i].is_nan() || sp_s[i].is_nan() {
            continue;
        }
        if status.len() == n {
            let lo = i.saturating_sub(29);
            if !status[lo..=i].iter().all(|s| s.observed) {
                continue;
            }
        }
        counts.n_decision += 1;
        let spread_degenerate = sp_s[i] == 0.0;
        let features = [
            ofi[i],
            (ofi[i] - ofi_m[i]) / (ofi_s[i] + 1e-12),
            imbs[i],
            if spread_degenerate {
                0.0
            } else {
                (spreads[i] - sp_m[i]) / (sp_s[i] + 1e-12)
            },
            signed[i],
        ];
        // Baseline and three-class share one target window.
        let mut baseline = None;
        let mut baseline_target_mid = None;
        let mut baseline_target_time_ms = None;
        let mut three = None;
        let mut three_target_mid = None;
        let mut three_target_time_ms = None;
        if i + horizon >= n {
            counts.baseline_censored += 1;
            counts.three_censored_incomplete += 1;
        } else if !window_observed(i + 1, i + horizon) {
            counts.baseline_censored += 1;
            counts.three_censored_gap += 1;
        } else {
            let fwd = mids[i + horizon] / mids[i] - 1.0;
            baseline = Some(u8::from(fwd > 0.0));
            baseline_target_mid = Some(mids[i + horizon]);
            baseline_target_time_ms = Some(bars[i + horizon].0);
            counts.baseline_labeled += 1;
            if fwd == 0.0 {
                counts.baseline_zero_fwd += 1;
            }
            three = Some(if fwd < -eps {
                P1_DOWN
            } else if fwd > eps {
                P1_UP
            } else {
                P1_FLAT
            });
            three_target_mid = Some(mids[i + horizon]);
            three_target_time_ms = Some(bars[i + horizon].0);
            counts.three_labeled += 1;
            match three {
                Some(P1_DOWN) => counts.three_down += 1,
                Some(P1_FLAT) => counts.three_flat += 1,
                Some(P1_UP) => counts.three_up += 1,
                _ => unreachable!(),
            }
        }
        // Next-nonzero: scan forward for the first bar whose mid
        // differs from the decision bar's mid.
        let scan_end = (i + cutoff_bars).min(n - 1);
        let mut nonzero = None;
        let mut nonzero_target_mid = None;
        let mut nonzero_target_time_ms = None;
        let mut nonzero_move_bar = None;
        let first_move = (i + 1..=scan_end).find(|&j| mids[j] != mids[i]);
        match first_move {
            Some(j) => {
                if window_observed(i + 1, j) {
                    nonzero = Some(u8::from(mids[j] > mids[i]));
                    nonzero_target_mid = Some(mids[j]);
                    nonzero_target_time_ms = Some(bars[j].0);
                    nonzero_move_bar = Some(j);
                    counts.nonzero_labeled += 1;
                    if mids[j] > mids[i] {
                        counts.nonzero_up += 1;
                    } else {
                        counts.nonzero_down += 1;
                    }
                } else {
                    counts.nonzero_censored_gap += 1;
                }
            }
            None => {
                if i + cutoff_bars >= n {
                    counts.nonzero_censored_incomplete += 1;
                } else if !window_observed(i + 1, i + cutoff_bars) {
                    counts.nonzero_censored_gap += 1;
                } else {
                    counts.nonzero_censored_no_move += 1;
                }
            }
        }
        rows.push(P1Row {
            decision_bar: i,
            decision_time_ms: bars[i].0,
            features,
            spread_degenerate,
            baseline,
            baseline_target_mid,
            baseline_target_time_ms,
            nonzero,
            nonzero_target_mid,
            nonzero_target_time_ms,
            nonzero_move_bar,
            three,
            three_target_mid,
            three_target_time_ms,
        });
    }
    P1Dataset {
        rows,
        counts,
        config,
        cutoff_bars,
        feature_version: FEATURE_VERSION,
    }
}

/// One three-class row: shared features with a down/flat/up label.
#[derive(Debug, Clone, Copy)]
pub struct ClassRow {
    pub time_ms: i64,
    pub features: [f64; N_FEATURES],
    /// 0 down, 1 flat, 2 up.
    pub target: u8,
}

/// Full-batch gradient descent for L2 multinomial logistic
/// regression (softmax), mirroring `fit_logistic`: features are
/// standardized with the training fold's mean and population sd
/// (sd floor 1e-12) during descent, then folded back into raw-space
/// weights, so a saved model scores a raw feature row directly.
/// Returns one intercept-appended weight vector per class.
pub fn fit_softmax(
    xs: &[[f64; N_FEATURES]],
    y: &[u8],
    l2: f64,
    iters: usize,
    lr: f64,
) -> [[f64; N_FEATURES + 1]; 3] {
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
    let mut w = [[0.0; N_FEATURES + 1]; 3];
    for _ in 0..iters {
        let mut grad = [[0.0; N_FEATURES + 1]; 3];
        for (row, &yi) in xs.iter().zip(y.iter()) {
            if yi > 2 {
                continue;
            }
            let mut z = [0.0; 3];
            for c in 0..3 {
                z[c] = w[c][d];
                for j in 0..d {
                    z[c] += w[c][j] * (row[j] - mu[j]) / sd[j];
                }
            }
            let mx = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut denom = 0.0;
            let mut p = [0.0; 3];
            for c in 0..3 {
                p[c] = (z[c] - mx).exp();
                denom += p[c];
            }
            for c in 0..3 {
                let err = p[c] / denom - f64::from(u8::from(c == yi as usize));
                for j in 0..d {
                    grad[c][j] += err * (row[j] - mu[j]) / sd[j];
                }
                grad[c][d] += err;
            }
        }
        for c in 0..3 {
            for j in 0..d {
                w[c][j] -= lr * (grad[c][j] / n + l2 * w[c][j] / n);
            }
            w[c][d] -= lr * grad[c][d] / n;
        }
    }
    let mut raw = [[0.0; N_FEATURES + 1]; 3];
    for c in 0..3 {
        let mut b = w[c][d];
        for j in 0..d {
            raw[c][j] = w[c][j] / sd[j];
            b -= w[c][j] * mu[j] / sd[j];
        }
        raw[c][d] = b;
    }
    raw
}

/// Softmax class probabilities for a raw feature row under
/// `fit_softmax` weights.
pub fn predict_softmax(x: &[f64; N_FEATURES], w: &[[f64; N_FEATURES + 1]; 3]) -> [f64; 3] {
    let mut z = [0.0; 3];
    for c in 0..3 {
        z[c] = w[c][N_FEATURES];
        for j in 0..N_FEATURES {
            z[c] += w[c][j] * x[j];
        }
        z[c] = z[c].clamp(-30.0, 30.0);
    }
    let mx = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut denom = 0.0;
    let mut p = [0.0; 3];
    for c in 0..3 {
        p[c] = (z[c] - mx).exp();
        denom += p[c];
    }
    [p[0] / denom, p[1] / denom, p[2] / denom]
}

/// One fold's three-class held-out scores.
#[derive(Debug, Clone)]
pub struct ThreeClassFoldMetrics {
    pub fold: usize,
    pub n_train: usize,
    pub train_down: usize,
    pub train_flat: usize,
    pub train_up: usize,
    /// Train-majority class (argmax of the train-only counts,
    /// lowest index on ties): the baseline commits to this label
    /// without ever reading a test outcome.
    pub train_majority_label: u8,
    pub n_test: usize,
    pub n_down: usize,
    pub n_flat: usize,
    pub n_up: usize,
    pub accuracy: f64,
    /// Accuracy of the training-majority predictor on this fold.
    pub baseline_accuracy: f64,
    /// Multiclass Brier score: mean over rows of the squared error
    /// summed over the three one-hot classes.
    pub brier: f64,
    pub log_loss: f64,
    /// Confusion counts [true][predicted].
    pub confusion: [[usize; 3]; 3],
}

/// Purged walk-forward evaluation of the three-class track, with
/// the same fold geometry as `walk_forward_detailed`: the purge gap
/// is `horizon` rows, so no training target window overlaps the
/// test block. Train priors are train-only by construction.
#[derive(Debug, Clone)]
pub struct ThreeClassReport {
    /// Total labeled rows fed to the evaluation.
    pub n_labeled: usize,
    /// Held-out predictions scored.
    pub n_oos: usize,
    pub n_down_oos: usize,
    pub n_flat_oos: usize,
    pub n_up_oos: usize,
    /// Labeled rows never scored (training and purge-gap rows).
    pub n_unscored: usize,
    pub n_folds: usize,
    pub accuracy: f64,
    /// Out-of-sample accuracy of the training-majority predictor.
    pub baseline: f64,
    /// Mean per-class recall over the held-out rows. NaN when a
    /// class is absent from the held-out rows.
    pub balanced_accuracy: f64,
    pub brier: f64,
    pub log_loss: f64,
    /// Confusion counts [true][predicted] over all held-out rows.
    pub confusion: [[usize; 3]; 3],
    pub per_fold: Vec<ThreeClassFoldMetrics>,
}

pub fn walk_forward_three_class(
    rows: &[ClassRow],
    n_splits: usize,
    horizon: usize,
) -> Option<ThreeClassReport> {
    let n = rows.len();
    if n < 60 {
        return None;
    }
    let fold = n / (n_splits + 1);
    if fold < 10 {
        return None;
    }
    let mut per_fold: Vec<ThreeClassFoldMetrics> = Vec::new();
    let mut confusion = [[0usize; 3]; 3];
    let (mut acc_sum, mut base_sum, mut brier_sum, mut ll_sum) = (0.0, 0.0, 0.0, 0.0);
    let (mut n_oos, mut n_down_oos, mut n_flat_oos, mut n_up_oos) =
        (0usize, 0usize, 0usize, 0usize);
    for k in 1..=n_splits {
        let a = k * fold;
        let b = ((k + 1) * fold).min(n);
        let train_end = a.saturating_sub(horizon);
        if train_end < 30 || b - a < 10 {
            continue;
        }
        let mut train_counts = [0usize; 3];
        for r in &rows[..train_end] {
            if r.target <= 2 {
                train_counts[r.target as usize] += 1;
            }
        }
        let mut majority = 0u8;
        for c in 1..3 {
            if train_counts[c] > train_counts[majority as usize] {
                majority = c as u8;
            }
        }
        let xs: Vec<[f64; N_FEATURES]> = rows[..train_end].iter().map(|r| r.features).collect();
        let ys: Vec<u8> = rows[..train_end].iter().map(|r| r.target).collect();
        let w = fit_softmax(&xs, &ys, FIT_L2, FIT_ITERS, FIT_LR);
        let mut fold_conf = [[0usize; 3]; 3];
        let (mut f_acc, mut f_base, mut f_brier, mut f_ll) = (0.0, 0.0, 0.0, 0.0);
        let (mut f_down, mut f_flat, mut f_up) = (0usize, 0usize, 0usize);
        for r in &rows[a..b] {
            if r.target > 2 {
                continue;
            }
            let p = predict_softmax(&r.features, &w);
            let mut pred = 0usize;
            for c in 1..3 {
                if p[c] > p[pred] {
                    pred = c;
                }
            }
            let t = r.target as usize;
            fold_conf[t][pred] += 1;
            confusion[t][pred] += 1;
            f_acc += f64::from(pred == t);
            f_base += f64::from(majority as usize == t);
            let mut brier = 0.0;
            for (c, &pc) in p.iter().enumerate() {
                let onehot = f64::from(c == t);
                brier += (pc - onehot).powi(2);
            }
            f_brier += brier;
            let pc = p[t].clamp(1e-15, 1.0 - 1e-15);
            f_ll += -pc.ln();
            match t {
                0 => f_down += 1,
                1 => f_flat += 1,
                _ => f_up += 1,
            }
        }
        let n_test = f_down + f_flat + f_up;
        if n_test == 0 {
            continue;
        }
        let d = n_test as f64;
        acc_sum += f_acc;
        base_sum += f_base;
        brier_sum += f_brier;
        ll_sum += f_ll;
        n_oos += n_test;
        n_down_oos += f_down;
        n_flat_oos += f_flat;
        n_up_oos += f_up;
        per_fold.push(ThreeClassFoldMetrics {
            fold: per_fold.len(),
            n_train: train_end,
            train_down: train_counts[0],
            train_flat: train_counts[1],
            train_up: train_counts[2],
            train_majority_label: majority,
            n_test,
            n_down: f_down,
            n_flat: f_flat,
            n_up: f_up,
            accuracy: f_acc / d,
            baseline_accuracy: f_base / d,
            brier: f_brier / d,
            log_loss: f_ll / d,
            confusion: fold_conf,
        });
    }
    if n_oos == 0 {
        return None;
    }
    let d = n_oos as f64;
    let mut balanced = 0.0;
    for (c, row) in confusion.iter().enumerate() {
        let row_sum: usize = row.iter().sum();
        if row_sum == 0 {
            balanced = f64::NAN;
            break;
        }
        balanced += row[c] as f64 / row_sum as f64;
    }
    if balanced.is_finite() {
        balanced /= 3.0;
    }
    Some(ThreeClassReport {
        n_labeled: n,
        n_oos,
        n_down_oos,
        n_flat_oos,
        n_up_oos,
        n_unscored: n - n_oos,
        n_folds: per_fold.len(),
        accuracy: acc_sum / d,
        baseline: base_sum / d,
        balanced_accuracy: balanced,
        brier: brier_sum / d,
        log_loss: ll_sum / d,
        confusion,
        per_fold,
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
    fn p1_softmax_separates_three_classes() {
        let xs = vec![
            [10.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [-10.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let y = vec![2u8, 1, 0];
        let w = fit_softmax(&xs, &y, 0.01, 2000, 0.5);
        let p_up = predict_softmax(&[10.0, 0.0, 0.0, 0.0, 0.0], &w);
        let p_flat = predict_softmax(&[0.0, 0.0, 0.0, 0.0, 0.0], &w);
        let p_down = predict_softmax(&[-10.0, 0.0, 0.0, 0.0, 0.0], &w);
        for p in [p_up, p_flat, p_down] {
            assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            assert!(p.iter().all(|&v| (0.0..=1.0).contains(&v)));
        }
        assert!(p_up[2] > 0.9);
        assert!(p_flat[1] > 0.9);
        assert!(p_down[0] > 0.9);
    }

    #[test]
    fn p1_thin_data_has_no_decision_rows() {
        let quotes: Vec<Quote> = (0..10)
            .map(|i| Quote {
                time_ms: i * 10_000 + 1_000,
                bid_px: 99.95,
                bid_sz: 5.0,
                ask_px: 100.05,
                ask_sz: 5.0,
            })
            .collect();
        let ds = build_p1_dataset(&quotes, &[], 10_000, P1_DEFAULT_CONFIG, 10);
        assert!(ds.rows.is_empty());
        assert_eq!(ds.counts.n_bars, 10);
        assert_eq!(ds.counts.n_decision, 0);
    }

    #[test]
    fn beats_coin_on_predictable_flow() {
        let (q, t) = synth_quotes(1200, true);
        let bars = build_dataset(&q, &t, 10_000, 2).expect("valid fixture");
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
        let bars = build_dataset(&q, &t, 10_000, 2).expect("valid fixture");
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
        let ds_prefix = build_dataset(&prefix, &trades, 10_000, 2).expect("valid fixture");
        let ds_full = build_dataset(&quotes, &trades, 10_000, 2).expect("valid fixture");
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
        let ds = build_dataset(&quotes, &[], 10_000, 2).expect("valid fixture");
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
        let bars = ofi_bars(&quotes, &events, 10_000).expect("valid fixture");
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
        let ds = build_dataset(&quotes, &[], 10_000, 2).expect("valid fixture");
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
        let ds_pre = build_dataset(&prefix, &[], 10_000, 2).expect("valid fixture");
        let ds_full = build_dataset(&full, &[], 10_000, 2).expect("valid fixture");
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
        let mut bars = build_dataset(&varying, &[], 10_000, 2).expect("valid fixture");
        bars.extend(build_dataset(&constant, &[], 10_000, 2).expect("valid fixture"));
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
        let ds = build_dataset(&quotes, &trades, 10_000, 3).expect("valid fixture");
        assert_eq!(ds.len(), 81, "gapped dataset row count");

        // Control: identical quotes with the hole closed. 120 bars
        // yield i in 4..=116 (113 rows), so the censor removes exactly
        // the gap-crossing and gap-adjacent rows.
        let (quotes, trades) = gap_fixture(600_000);
        let ds = build_dataset(&quotes, &trades, 10_000, 3).expect("valid fixture");
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
        let ds = build_dataset(&q, &t, 10_000, 2).expect("valid fixture");
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
        let bars = build_dataset(&q, &t, 10_000, 2).expect("valid fixture");
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
        let bars = build_dataset(&q, &t, 10_000, 2).expect("valid fixture");
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
        let bars = build_dataset(&q, &t, 10_000, 2).expect("valid fixture");
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

    // ---- Stage 3 P2: normalized flow features ----------------------

    fn p2_q(t: i64, mid: f64, bq: f64, aq: f64) -> Quote {
        Quote {
            time_ms: t,
            bid_px: mid - 0.05,
            bid_sz: bq,
            ask_px: mid + 0.05,
            ask_sz: aq,
        }
    }

    fn p2_col(name: &str) -> usize {
        P2_FEATURE_NAMES
            .iter()
            .position(|n| *n == name)
            .unwrap_or_else(|| panic!("unknown P2 feature {name}"))
    }

    fn p2_row(t: i64, features: [f64; P2_N_FEATURES], target: u8) -> P2Bar {
        P2Bar {
            time_ms: t,
            features,
            target,
            spread_degenerate: false,
        }
    }

    /// 40 bars, one quote per bar, constant prices, sizes alternating
    /// between A on even bars and B on odd bars.
    fn ofi_depth_session(b_a: f64, a_a: f64, b_b: f64, a_b: f64) -> Vec<Quote> {
        (0..40)
            .map(|i| {
                let (bq, aq) = if i % 2 == 0 { (b_a, a_a) } else { (b_b, a_b) };
                p2_q(i * 10_000, 100.05, bq, aq)
            })
            .collect()
    }

    #[test]
    fn stage3_toxicflow_2_p2_ofi_depth_ten_analytic_fixtures() {
        // (A bid, A ask, B bid, B ask, OFI at odd bar 31, ofi_depth).
        // Bar 31 compares B sizes against A sizes, so its OFI is
        // (bB - bA) - (aB - aA); its trailing 30-bar mean depth holds
        // 15 A bars and 15 B bars, so the mean depth is (dA + dB)/2.
        let cases: [(f64, f64, f64, f64, f64, f64); 10] = [
            (4.0, 6.0, 6.0, 4.0, 4.0, 0.8),
            (2.0, 8.0, 6.0, 4.0, 8.0, 1.6),
            (1.0, 9.0, 9.0, 1.0, 16.0, 3.2),
            (3.0, 3.0, 3.0, 9.0, -6.0, -1.3333333333333333),
            (10.0, 0.0, 0.0, 10.0, -20.0, -4.0),
            (0.5, 1.5, 2.5, 0.5, 3.0, 2.4),
            (7.0, 3.0, 7.0, 7.0, -4.0, -0.6666666666666666),
            (12.0, 4.0, 4.0, 12.0, -16.0, -2.0),
            (5.0, 5.0, 15.0, 5.0, 10.0, 1.3333333333333333),
            (2.0, 6.0, 6.0, 2.0, 8.0, 2.0),
        ];
        let mut max_err = 0.0f64;
        for (b_a, a_a, b_b, a_b, want_ofi, want_depth) in cases {
            let qs = ofi_depth_session(b_a, a_a, b_b, a_b);
            let ds = build_p2_dataset(&qs, &[], 10_000, 3);
            let row = ds
                .iter()
                .find(|r| r.time_ms == 310_000)
                .expect("row at bar 31");
            let ofi_err = (row.features[p2_col("ofi")] - want_ofi).abs();
            let depth_err = (row.features[p2_col("ofi_depth")] - want_depth).abs();
            max_err = max_err.max(ofi_err).max(depth_err);
            assert!(
                ofi_err <= 1e-10,
                "case A=({b_a},{a_a}) B=({b_b},{a_b}): OFI {} vs {want_ofi}",
                row.features[p2_col("ofi")]
            );
            assert!(
                depth_err <= 1e-10,
                "case A=({b_a},{a_a}) B=({b_b},{a_b}): OFI/depth {} vs {want_depth}",
                row.features[p2_col("ofi_depth")]
            );
        }
        assert!(
            max_err <= 1e-10,
            "max abs error over 10 analytic OFI/depth fixtures is {max_err:e}"
        );
    }

    #[test]
    fn stage3_toxicflow_2_p2_signed_flow_window_and_lags_known_answer() {
        // Constant quotes (OFI 0); bar i carries one buy trade of i
        // base units, so signed_vol[i] = i. At bar 35 the trailing
        // 6-bar signed flow is 30+31+32+33+34+35 = 195, the 3-bar
        // lag is 32 and the 10-bar lag is 25.
        let qs: Vec<Quote> = (0..40)
            .map(|i| p2_q(i * 10_000, 100.05, 5.0, 5.0))
            .collect();
        let trades: Vec<Trade> = (0..40)
            .map(|i| Trade {
                time_ms: i * 10_000 + 100,
                price: 100.05,
                qty: i as f64,
                is_buyer_maker: false,
            })
            .collect();
        let ds = build_p2_dataset(&qs, &trades, 10_000, 3);
        let row = ds
            .iter()
            .find(|r| r.time_ms == 350_000)
            .expect("row at bar 35");
        assert!((row.features[p2_col("signed_flow_w")] - 195.0).abs() < 1e-9);
        assert!((row.features[p2_col("signed_flow_lag3")] - 32.0).abs() < 1e-9);
        assert!((row.features[p2_col("signed_flow_lag10")] - 25.0).abs() < 1e-9);
        assert!(row.features[p2_col("ofi_lag3")].abs() < 1e-9);
        assert!(row.features[p2_col("ofi_lag10")].abs() < 1e-9);
        assert!(row.features[p2_col("ofi")].abs() < 1e-9);
    }

    #[test]
    fn stage3_toxicflow_2_p2_realized_vol_known_answer() {
        // Mids double every bar, so each log return is ln 2 and the
        // trailing 30-bar window holds 29 clean within-window
        // returns: realized vol = ln 2 * sqrt(29).
        let qs: Vec<Quote> = (0..40)
            .map(|i| p2_q(i * 10_000, 100.0 * 2f64.powi(i as i32), 5.0, 5.0))
            .collect();
        let ds = build_p2_dataset(&qs, &[], 10_000, 3);
        let row = ds
            .iter()
            .find(|r| r.time_ms == 350_000)
            .expect("row at bar 35");
        let want = 2f64.ln() * (29f64).sqrt();
        assert!(
            (row.features[p2_col("realized_vol")] - want).abs() <= 1e-10,
            "realized vol {} vs {want}",
            row.features[p2_col("realized_vol")]
        );
        // Constant mids give exactly zero realized volatility.
        let flat: Vec<Quote> = (0..40)
            .map(|i| p2_q(i * 10_000, 100.05, 5.0, 5.0))
            .collect();
        let ds_flat = build_p2_dataset(&flat, &[], 10_000, 3);
        let row_flat = ds_flat
            .iter()
            .find(|r| r.time_ms == 350_000)
            .expect("flat row at bar 35");
        assert!(row_flat.features[p2_col("realized_vol")].abs() <= 1e-10);
    }

    #[test]
    fn stage3_toxicflow_2_p2_quote_trade_rates_known_answer() {
        // Bar i holds (i%3 + 1) quotes and (i%2 + 1) trades. Bars
        // 6..=35 cover 10 full quote cycles (mean 2.0) and 15 of
        // each trade count (mean 1.5).
        let mut qs = Vec::new();
        let mut trades = Vec::new();
        for i in 0..40 {
            for j in 0..(i % 3 + 1) {
                qs.push(p2_q(i * 10_000 + j * 100, 100.05, 5.0, 5.0));
            }
            for k in 0..(i % 2 + 1) {
                trades.push(Trade {
                    time_ms: i * 10_000 + 500 + k * 100,
                    price: 100.05,
                    qty: 0.1,
                    is_buyer_maker: false,
                });
            }
        }
        let ds = build_p2_dataset(&qs, &trades, 10_000, 3);
        let row = ds
            .iter()
            .find(|r| r.time_ms == 350_000)
            .expect("row at bar 35");
        assert!((row.features[p2_col("quote_rate")] - 2.0).abs() < 1e-9);
        assert!((row.features[p2_col("trade_rate")] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn stage3_toxicflow_2_p2_micro_disp_known_answer() {
        // Sizes (8, 2): microprice (100.1*8 + 100.0*2)/10 = 100.08
        // against mid 100.05 and spread 0.1, displacement +0.3.
        let qs: Vec<Quote> = (0..40)
            .map(|i| p2_q(i * 10_000, 100.05, 8.0, 2.0))
            .collect();
        let ds = build_p2_dataset(&qs, &[], 10_000, 3);
        for row in &ds {
            assert!((row.features[p2_col("micro_disp")] - 0.3).abs() < 1e-12);
        }
        // Zero spread pins the displacement to exactly 0.
        let locked: Vec<Quote> = (0..40)
            .map(|i| Quote {
                time_ms: i * 10_000,
                bid_px: 100.0,
                bid_sz: 5.0,
                ask_px: 100.0,
                ask_sz: 5.0,
            })
            .collect();
        let ds_locked = build_p2_dataset(&locked, &[], 10_000, 3);
        for row in &ds_locked {
            assert_eq!(row.features[p2_col("micro_disp")], 0.0);
            assert_eq!(row.features[p2_col("ofi_depth")], 0.0);
        }
        // Zero sizes pin both the displacement and OFI/depth to 0.
        let empty: Vec<Quote> = (0..40)
            .map(|i| p2_q(i * 10_000, 100.05, 0.0, 0.0))
            .collect();
        let ds_empty = build_p2_dataset(&empty, &[], 10_000, 3);
        for row in &ds_empty {
            assert_eq!(row.features[p2_col("micro_disp")], 0.0);
            assert_eq!(row.features[p2_col("ofi_depth")], 0.0);
        }
    }

    /// Deterministic synthetic quote/trade stream: one quote per
    /// second with a random-walk mid and random sizes, one trade per
    /// second. Seeded xorshift, no randomness at test time.
    fn p2_synth_stream(seed: u64, n_bars: usize) -> (Vec<Quote>, Vec<Trade>) {
        let mut s: u64 = seed;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s as f64) / (u64::MAX as f64)
        };
        let mut quotes = Vec::new();
        let mut trades = Vec::new();
        let mut mid = 100.0;
        for sec in 0..(n_bars * 10) {
            let t = sec as i64 * 1000;
            quotes.push(p2_q(t, mid, 1.0 + 9.0 * rnd(), 1.0 + 9.0 * rnd()));
            trades.push(Trade {
                time_ms: t + 100,
                price: mid,
                qty: 0.01 + 0.5 * rnd(),
                is_buyer_maker: rnd() < 0.5,
            });
            mid += (rnd() - 0.5) * 0.04;
        }
        (quotes, trades)
    }

    #[test]
    fn stage3_toxicflow_2_p2_prefix_extension_bit_identical() {
        // 10 synthetic prefix-extension cases: building the frame on
        // a time-prefix of the stream must leave every shared row
        // bit-identical (0 future-induced changes).
        for case in 0..10 {
            let n_bars = 80 + case * 7;
            let k = n_bars / 2;
            let (quotes, trades) = p2_synth_stream(0x9E3779B97F4A7C15 + case as u64, n_bars);
            let cut_ms = k as i64 * 10_000;
            let pre_q: Vec<Quote> = quotes
                .iter()
                .filter(|q| q.time_ms < cut_ms)
                .copied()
                .collect();
            let pre_t: Vec<Trade> = trades
                .iter()
                .filter(|t| t.time_ms < cut_ms)
                .copied()
                .collect();
            let full = build_p2_dataset(&quotes, &trades, 10_000, 3);
            let pre = build_p2_dataset(&pre_q, &pre_t, 10_000, 3);
            // Rows whose target window ends before the cut exist in
            // both frames; later rows cannot be compared.
            let last_shared_ms = (k as i64 - 4) * 10_000;
            let mut compared = 0;
            for row in full.iter().filter(|r| r.time_ms <= last_shared_ms) {
                let other = pre
                    .iter()
                    .find(|r| r.time_ms == row.time_ms)
                    .unwrap_or_else(|| panic!("case {case}: missing prefix row {}", row.time_ms));
                for j in 0..P2_N_FEATURES {
                    assert_eq!(
                        row.features[j].to_bits(),
                        other.features[j].to_bits(),
                        "case {case}: feature {} changed after extension at {}",
                        P2_FEATURE_NAMES[j],
                        row.time_ms
                    );
                }
                assert_eq!(row.target, other.target);
                compared += 1;
            }
            assert!(compared > 0, "case {case}: no shared rows compared");
        }
    }

    #[test]
    fn stage3_toxicflow_2_p2_v3_columns_bit_identical() {
        // The v4 frame's first five columns must reproduce
        // `build_dataset` row for row on shared rows, so the v3
        // baseline is preserved exactly inside the new frame.
        let (q, t) = synth_quotes(1200, true);
        let v3 = build_dataset(&q, &t, 10_000, 3).expect("valid fixture");
        let p2 = build_p2_dataset(&q, &t, 10_000, 3);
        assert!(!p2.is_empty());
        let mut compared = 0;
        for row in &p2 {
            let other = v3
                .iter()
                .find(|b| b.time_ms == row.time_ms)
                .unwrap_or_else(|| panic!("no v3 row at {}", row.time_ms));
            for (j, v) in row.v3_features().iter().enumerate() {
                assert_eq!(
                    v.to_bits(),
                    other.features()[j].to_bits(),
                    "v3 column {j} differs at {}",
                    row.time_ms
                );
            }
            assert_eq!(row.target, other.target);
            assert_eq!(row.spread_degenerate, other.spread_degenerate);
            compared += 1;
        }
        assert!(compared > 0);
    }

    #[test]
    fn stage3_toxicflow_2_p2_scaler_excludes_test_rows() {
        // Perturbing only test-block rows must leave every fold's
        // fitted weights bit-identical (standardization is learned
        // in training only), while the scored probabilities move.
        let (q, t) = synth_quotes(1200, true);
        let ds = build_p2_dataset(&q, &t, 10_000, 3);
        let base = walk_forward_p2(&ds, P2_B5_COLS, P2Control::Off, 4, 3).expect("eval failed");
        let n = ds.len();
        let fold = n / 5;
        let (a, b) = (4 * fold, n.min(5 * fold));
        let mut perturbed = ds.clone();
        for row in &mut perturbed[a..b] {
            for v in row.features.iter_mut() {
                *v *= 1000.0;
            }
        }
        let rerun =
            walk_forward_p2(&perturbed, P2_B5_COLS, P2Control::Off, 4, 3).expect("eval failed");
        assert_eq!(base.fold_models.len(), rerun.fold_models.len());
        for (i, ((w0, b0), (w1, b1))) in base
            .fold_models
            .iter()
            .zip(rerun.fold_models.iter())
            .enumerate()
        {
            assert_eq!(w0.len(), w1.len());
            for (j, (x, y)) in w0.iter().zip(w1.iter()).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "fold {i} weight {j} moved when only test rows changed"
                );
            }
            assert_eq!(
                b0.to_bits(),
                b1.to_bits(),
                "fold {i} intercept moved when only test rows changed"
            );
        }
        let max_p_diff = base
            .predictions
            .iter()
            .zip(rerun.predictions.iter())
            .map(|(x, y)| (x.proba - y.proba).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_p_diff > 0.01,
            "perturbed test rows left every probability unchanged; the check is vacuous"
        );
    }

    #[test]
    fn stage3_toxicflow_2_p2_controls_on_hand_built_frame() {
        // 400 rows, alternating targets, one P2 column equal to the
        // target and every other column zero: the clean model is
        // near-perfect, the permutation control breaks the alignment,
        // and a 5-bar shift (odd lag) flips it.
        let rows: Vec<P2Bar> = (0..400)
            .map(|i| {
                let target = (i % 2) as u8;
                let mut f = [0.0; P2_N_FEATURES];
                f[p2_col("signed_flow_w")] = target as f64;
                p2_row(i as i64 * 10_000, f, target)
            })
            .collect();
        let clean =
            walk_forward_p2(&rows, P2_B5_COLS, P2Control::Off, 4, 3).expect("clean eval failed");
        assert!(
            clean.accuracy > 0.95,
            "clean accuracy {} should be near-perfect",
            clean.accuracy
        );
        let perm =
            walk_forward_p2(&rows, P2_B5_COLS, P2Control::Permute, 4, 3).expect("perm eval failed");
        assert!(
            (0.35..=0.65).contains(&perm.accuracy),
            "permuted accuracy {} should sit near chance",
            perm.accuracy
        );
        assert!(
            perm.accuracy < clean.accuracy - 0.3,
            "permutation did not collapse the signal: {} vs {}",
            perm.accuracy,
            clean.accuracy
        );
        let shift =
            walk_forward_p2(&rows, P2_B5_COLS, P2Control::Shift, 4, 3).expect("shift eval failed");
        assert!(
            shift.accuracy < 0.1,
            "shifted accuracy {} should be near zero on alternating targets",
            shift.accuracy
        );
        assert!(shift.n_oos < clean.n_oos, "shift must drop warm-up rows");
        // B0 is the training-majority predictor by construction, and
        // B3 sees only zero columns here, so both sit at chance.
        let b0 = walk_forward_p2(&rows, &[], P2Control::Off, 4, 3).expect("b0 eval failed");
        assert!(
            (b0.accuracy - 0.5).abs() < 0.05,
            "b0 accuracy {}",
            b0.accuracy
        );
        assert!(
            (b0.baseline - b0.accuracy).abs() < 1e-12,
            "b0 must equal its own baseline"
        );
        let b3 = walk_forward_p2(&rows, P2_B3_COLS, P2Control::Off, 4, 3).expect("b3 eval failed");
        assert!(
            (b3.accuracy - 0.5).abs() < 0.05,
            "b3 accuracy {}",
            b3.accuracy
        );
    }

    #[test]
    fn stage3_toxicflow_2_p2_thin_data_has_no_rows() {
        let qs: Vec<Quote> = (0..10)
            .map(|i| p2_q(i * 10_000 + 1_000, 100.05, 5.0, 5.0))
            .collect();
        assert!(build_p2_dataset(&qs, &[], 10_000, 3).is_empty());
    }

    #[test]
    fn stage3_toxicflow_2_p2_preregistered_grid_shape() {
        let grid = preregistered_p2_grid();
        assert_eq!(grid.len(), 30, "grid must hold 30 configs");
        let ids: Vec<&str> = grid.iter().map(|c| c.id.as_str()).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(ids.len(), sorted.len(), "config ids must be unique");
        let defaults: Vec<&P2Config> = grid.iter().filter(|c| c.kind == "default").collect();
        assert_eq!(defaults.len(), 1, "exactly one default config");
        assert_eq!(defaults[0].id, "b5-full");
        assert_eq!(defaults[0].cols.len(), P2_N_FEATURES);
        let b0 = grid.iter().find(|c| c.id == "b0-majority").expect("b0");
        assert!(b0.cols.is_empty());
        let b3 = grid.iter().find(|c| c.id == "b3-v3").expect("b3");
        assert_eq!(b3.cols.len(), 5);
        let b4 = grid.iter().find(|c| c.id == "b4-flow").expect("b4");
        assert_eq!(b4.cols.len(), 11);
        for c in &grid {
            for name in &c.cols {
                assert!(
                    P2_FEATURE_NAMES.contains(&name.as_str()),
                    "config {} names unknown feature {name}",
                    c.id
                );
            }
        }
    }

    #[test]
    fn stage3_toxicflow_3_p3_labels() {
        // Buy flow, mid up: not adverse. Buy flow, mid down: adverse.
        assert_eq!(p3_labels(0.01, 5.0, 0.10, 0.10), (1, 0, Some(0)));
        assert_eq!(p3_labels(-0.01, 5.0, 0.10, 0.10), (0, 0, Some(1)));
        // Sell flow, mid down: not adverse. Sell flow, mid up: adverse.
        assert_eq!(p3_labels(-0.01, -5.0, 0.10, 0.10), (0, 0, Some(0)));
        assert_eq!(p3_labels(0.01, -5.0, 0.10, 0.10), (1, 0, Some(1)));
        // Zero forward return is never adverse; zero flow is excluded.
        assert_eq!(p3_labels(0.0, 5.0, 0.10, 0.10), (0, 0, Some(0)));
        assert_eq!(p3_labels(0.01, 0.0, 0.10, 0.10), (1, 0, None));
        // Spread widening is a strict increase.
        assert_eq!(p3_labels(0.01, 5.0, 0.10, 0.11), (1, 1, Some(0)));
        assert_eq!(p3_labels(0.01, 5.0, 0.10, 0.10), (1, 0, Some(0)));
        assert_eq!(p3_labels(0.01, 5.0, 0.10, 0.09), (1, 0, Some(0)));
    }

    #[test]
    fn stage3_toxicflow_3_p3_median_and_tertiles() {
        assert_eq!(p3_median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(p3_median(&[4.0, 1.0, 2.0, 3.0]), 2.5);
        assert!(p3_median(&[]).is_nan());
        assert_eq!(
            p3_tertile_cuts(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
            (4.0, 7.0)
        );
        assert_eq!(p3_tertile(4.0, 4.0, 7.0), 0);
        assert_eq!(p3_tertile(5.0, 4.0, 7.0), 1);
        assert_eq!(p3_tertile(7.0, 4.0, 7.0), 1);
        assert_eq!(p3_tertile(8.0, 4.0, 7.0), 2);
    }

    #[test]
    fn stage3_toxicflow_3_p3_preregistered_grid_shape() {
        let grid = preregistered_p3_grid();
        assert_eq!(grid.len(), 8, "grid must hold 8 configs");
        let ids: Vec<&str> = grid.iter().map(|c| c.id.as_str()).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(ids.len(), sorted.len(), "config ids must be unique");
        let defaults: Vec<&P3Config> = grid.iter().filter(|c| c.kind == "default").collect();
        assert_eq!(defaults.len(), 1, "exactly one default config");
        assert_eq!(defaults[0].id, "controls-vpin");
        assert_eq!(defaults[0].cols.len(), P3_N_FEATURES);
        let b0 = grid.iter().find(|c| c.id == "b0-majority").expect("b0");
        assert!(b0.cols.is_empty());
        let vpin = grid
            .iter()
            .find(|c| c.id == "vpin-only")
            .expect("vpin-only");
        assert_eq!(vpin.cols, vec!["vpin".to_string(), "vpin_lag3".to_string()]);
        for c in &grid {
            for name in &c.cols {
                assert!(
                    P3_FEATURE_NAMES.contains(&name.as_str()),
                    "config {} names unknown feature {name}",
                    c.id
                );
            }
        }
    }
}
