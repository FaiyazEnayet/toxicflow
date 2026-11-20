//! Censored next-nonzero price-move hazard baseline (Stage 3 N5).
//!
//! The target is the waiting time from a decision bar to the first
//! nonzero mid change, with up and down moves as competing events.
//! It is a separate target from P1's binary/three-class direction
//! labels: P1 stays the historical direction task, and no outcome of
//! this module is ever compared against a P1 accuracy or baseline.
//!
//! Time conventions (all times in milliseconds on the OFI bar grid):
//!
//! * `t_observe_ms` is the start of the decision bar. The features
//!   behind the decision are observed over `[t_observe, t_decision)`.
//! * `t_decision_ms` is the end of the decision bar: the decision
//!   instant. A row is a decision row exactly when P1 admits it
//!   (feature window defined and fully observed).
//! * For an event at move bar `j`, `t_outcome_ms` is the start of bar
//!   `j` (the same timestamp P1 records as the nonzero target time),
//!   and the discrete waiting time is `wait_bars = j - i`. The move
//!   is first visible at the start of the move bar, so
//!   `t_outcome - t_decision = (wait_bars - 1) * bar_ms`.
//! * For a censored row, `t_censor_ms` is the end of the last fully
//!   observed at-risk bar, so `t_censor - t_decision =
//!   wait_bars * bar_ms`. A row censored by a gap contributes the
//!   bars up to (not including) the first unobserved bar.
//!
//! Every row satisfies `t_observe <= t_decision <= t_outcome` for
//! events and `t_observe <= t_decision <= t_censor` for censored
//! rows, or carries an explicit [`CensorReason`]. No outcome window
//! crosses a feed gap: a move whose window touches an unobserved bar
//! is censored, never labeled.
//!
//! The baseline is a constant-hazard (exponential) competing-risk
//! model: one hazard per cause, fit by maximum likelihood on train
//! rows only, in closed form (`lambda_c = n_c / total time at
//! risk`). The waiting-time log-likelihood of a row is
//! `ln(lambda_c) - lambda_total * wait` for an event of cause `c`,
//! and `-lambda_total * wait` for a censored row. Calibration
//! compares the fitted survival `exp(-lambda_total * t)` against a
//! Kaplan-Meier estimate on held-out rows at fixed horizons. The
//! final holdout is never used for tuning: the baseline has no
//! tuned parameters, and the walk-forward fits each fold on its
//! train rows only with a purge of one cutoff length.

use crate::metrics::{bar_status, ofi_bars, ofi_events, Quote};
use crate::predict::P1_NONZERO_CUTOFF_BARS;

/// The hazard target shares P1's frozen next-nonzero cutoff.
pub const HAZARD_CUTOFF_BARS: usize = P1_NONZERO_CUTOFF_BARS;

/// Bar length of the frozen design grid, in milliseconds.
pub const HAZARD_BAR_MS: i64 = 10_000;

/// Walk-forward splits of the frozen design grid.
pub const HAZARD_N_SPLITS: usize = 4;

/// Horizons (bars) of the frozen calibration table.
pub const HAZARD_CALIBRATION_HORIZONS_BARS: [usize; 5] = [1, 2, 3, 5, 10];

/// The six real recorded BTCUSDT sessions of the frozen design grid
/// (the same test sessions as the P1 grid).
pub const HAZARD_TEST_SESSIONS: [&str; 6] = [
    "session-20261111-163501-utc",
    "session-20261111-203511-utc",
    "session-20261112-003521-utc",
    "session-20261112-043531-utc",
    "session-20261112-083541-utc",
    "session-20261112-123552-utc",
];

pub const HAZARD_GRID_SCHEMA: &str = "toxicflow.n5-grid/v1";
pub const HAZARD_RESULTS_SCHEMA: &str = "toxicflow.n5-results/v1";

/// Why a hazard row is censored. Every censored row carries exactly
/// one reason; reasons are counted, never dropped silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CensorReason {
    /// Complete observed cutoff window with no nonzero move.
    NoMove,
    /// The cutoff window runs past the last bar before any move is
    /// observed.
    Incomplete,
    /// The window up to the first move (or the full window) touches
    /// an unobserved bar, so the move time is not trustworthy.
    Gap,
}

impl CensorReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            CensorReason::NoMove => "no_move",
            CensorReason::Incomplete => "incomplete",
            CensorReason::Gap => "gap",
        }
    }
}

/// One hazard decision row: the waiting time to the first nonzero
/// mid change, or the censoring time with its reason.
#[derive(Debug, Clone, Copy)]
pub struct HazardRow {
    /// Index of the decision bar on the OFI bar grid.
    pub decision_bar: usize,
    /// Start of the decision bar.
    pub t_observe_ms: i64,
    /// End of the decision bar: the decision instant.
    pub t_decision_ms: i64,
    /// Start of the move bar, for events.
    pub t_outcome_ms: Option<i64>,
    /// End of the last fully observed at-risk bar. Equals
    /// `t_outcome_ms` for events.
    pub t_censor_ms: i64,
    /// Bars from the decision bar to the move bar (events) or to the
    /// end of observation (censored).
    pub wait_bars: usize,
    pub event: bool,
    /// 0 down, 1 up, for events; `None` for censored rows.
    pub direction: Option<u8>,
    pub censor: Option<CensorReason>,
    /// Bar of the first nonzero mid change, for events.
    pub move_bar: Option<usize>,
}

/// Reconciled sample counts for a hazard dataset. Every decision row
/// is either an event or censored: `n_events + n_censored ==
/// n_decision`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HazardCounts {
    /// OFI bars on the grid.
    pub n_bars: usize,
    /// Candidate decision rows (feature window defined and observed).
    pub n_decision: usize,
    pub n_events: usize,
    pub n_up: usize,
    pub n_down: usize,
    pub n_censored: usize,
    /// Complete observed cutoff window with no nonzero move.
    pub censored_no_move: usize,
    /// Window runs past the last bar before any move is observed.
    pub censored_incomplete: usize,
    /// Window up to the first move (or the full window) touches an
    /// unobserved bar.
    pub censored_gap: usize,
}

/// A hazard dataset: decision rows with waiting times and censoring,
/// plus the reconciled counts and the frozen cutoff that produced
/// them.
#[derive(Debug, Clone)]
pub struct HazardDataset {
    pub rows: Vec<HazardRow>,
    pub counts: HazardCounts,
    pub cutoff_bars: usize,
    pub bar_ms: i64,
}

/// Build the hazard dataset from touch quotes on a fixed bar grid.
/// The labeling mirrors P1's next-nonzero scan exactly (same
/// decision rows, same move bars, same censoring rules), pinned by
/// the `n5_p1_nonzero_agreement` fixture; the hazard rows add the
/// waiting time, the event/censoring timestamps, and the per-row
/// censoring reason P1 only counts. Thin data (fewer than 40 bars)
/// yields no decision rows, mirroring `build_dataset`.
pub fn build_hazard_dataset(quotes: &[Quote], bar_ms: i64, cutoff_bars: usize) -> HazardDataset {
    let empty = |n_bars: usize| HazardDataset {
        rows: Vec::new(),
        counts: HazardCounts {
            n_bars,
            ..Default::default()
        },
        cutoff_bars,
        bar_ms,
    };
    if bar_ms <= 0 {
        return empty(0);
    }
    let events = ofi_events(quotes);
    // H4: an invalid bar grid yields no decision rows, the same
    // thin-data outcome as before.
    let bars = match ofi_bars(quotes, &events, bar_ms) {
        Ok(b) => b,
        Err(_) => return empty(0),
    };
    if bars.len() < 40 {
        return empty(bars.len());
    }
    let n = bars.len();
    let mids: Vec<f64> = bars.iter().map(|b| b.2).collect();
    let status = bar_status(quotes, bar_ms);
    let window_observed = |lo: usize, hi: usize| -> bool {
        status.len() == n && status[lo..=hi].iter().all(|s| s.observed)
    };
    let first_unobserved = |lo: usize, hi: usize| -> Option<usize> {
        if status.len() != n {
            return None;
        }
        (lo..=hi).find(|&g| !status[g].observed)
    };
    let mut rows = Vec::new();
    let mut counts = HazardCounts {
        n_bars: n,
        ..Default::default()
    };
    for i in 0..n {
        // Decision rows start at bar 4: the shared v3 feature frame
        // needs a 5-bar rolling window, so bars 0..4 have no defined
        // frame. This is the same gate as P1's NaN rolling-window
        // check (`rolling_mean_sd` leaves the first four entries
        // NaN), restated so the hazard builder needs no features.
        if i < 4 {
            continue;
        }
        if status.len() == n {
            let lo = i.saturating_sub(29);
            if !status[lo..=i].iter().all(|s| s.observed) {
                continue;
            }
        }
        counts.n_decision += 1;
        let t_observe = bars[i].0;
        let t_decision = bars[i].0 + bar_ms;
        // Next-nonzero: scan forward for the first bar whose mid
        // differs from the decision bar's mid.
        let scan_end = (i + cutoff_bars).min(n - 1);
        let first_move = (i + 1..=scan_end).find(|&j| mids[j] != mids[i]);
        match first_move {
            Some(j) => {
                if window_observed(i + 1, j) {
                    let up = mids[j] > mids[i];
                    rows.push(HazardRow {
                        decision_bar: i,
                        t_observe_ms: t_observe,
                        t_decision_ms: t_decision,
                        t_outcome_ms: Some(bars[j].0),
                        t_censor_ms: bars[j].0,
                        wait_bars: j - i,
                        event: true,
                        direction: Some(u8::from(up)),
                        censor: None,
                        move_bar: Some(j),
                    });
                    counts.n_events += 1;
                    if up {
                        counts.n_up += 1;
                    } else {
                        counts.n_down += 1;
                    }
                } else {
                    // A move is visible but its timing crosses an
                    // unobserved bar: censor at the first gap bar.
                    let g = first_unobserved(i + 1, j).unwrap_or(i + 1);
                    rows.push(gap_row(i, t_observe, t_decision, g, &bars));
                    counts.n_censored += 1;
                    counts.censored_gap += 1;
                }
            }
            None => {
                if i + cutoff_bars >= n {
                    rows.push(HazardRow {
                        decision_bar: i,
                        t_observe_ms: t_observe,
                        t_decision_ms: t_decision,
                        t_outcome_ms: None,
                        t_censor_ms: bars[n - 1].0 + bar_ms,
                        wait_bars: n - 1 - i,
                        event: false,
                        direction: None,
                        censor: Some(CensorReason::Incomplete),
                        move_bar: None,
                    });
                    counts.n_censored += 1;
                    counts.censored_incomplete += 1;
                } else if !window_observed(i + 1, i + cutoff_bars) {
                    let g = first_unobserved(i + 1, i + cutoff_bars).unwrap_or(i + 1);
                    rows.push(gap_row(i, t_observe, t_decision, g, &bars));
                    counts.n_censored += 1;
                    counts.censored_gap += 1;
                } else {
                    rows.push(HazardRow {
                        decision_bar: i,
                        t_observe_ms: t_observe,
                        t_decision_ms: t_decision,
                        t_outcome_ms: None,
                        t_censor_ms: bars[i + cutoff_bars].0 + bar_ms,
                        wait_bars: cutoff_bars,
                        event: false,
                        direction: None,
                        censor: Some(CensorReason::NoMove),
                        move_bar: None,
                    });
                    counts.n_censored += 1;
                    counts.censored_no_move += 1;
                }
            }
        }
    }
    HazardDataset {
        rows,
        counts,
        cutoff_bars,
        bar_ms,
    }
}

/// A gap-censored row: at risk through the bar before the first
/// unobserved bar `g`, so `wait_bars = g - 1 - decision_bar` and the
/// censoring time is the start of bar `g`.
fn gap_row(
    decision_bar: usize,
    t_observe_ms: i64,
    t_decision_ms: i64,
    first_gap_bar: usize,
    bars: &[(i64, f64, f64, f64, f64)],
) -> HazardRow {
    HazardRow {
        decision_bar,
        t_observe_ms,
        t_decision_ms,
        t_outcome_ms: None,
        t_censor_ms: bars[first_gap_bar].0,
        wait_bars: first_gap_bar - 1 - decision_bar,
        event: false,
        direction: None,
        censor: Some(CensorReason::Gap),
        move_bar: None,
    }
}

/// Constant-hazard (exponential) competing-risk baseline, fit by
/// maximum likelihood on train rows only. The MLE is closed form:
/// `lambda_c = n_c / total time at risk`, per cause.
#[derive(Debug, Clone, Copy)]
pub struct ConstantHazard {
    pub lambda_up_per_bar: f64,
    pub lambda_down_per_bar: f64,
    pub lambda_total_per_bar: f64,
    pub n_events_up: usize,
    pub n_events_down: usize,
    pub n_censored: usize,
    pub total_time_at_risk_bars: f64,
    pub n_train: usize,
}

pub fn fit_constant_hazard(rows: &[HazardRow]) -> ConstantHazard {
    let mut n_up = 0usize;
    let mut n_down = 0usize;
    let mut n_censored = 0usize;
    let mut total_time = 0.0f64;
    for r in rows {
        total_time += r.wait_bars as f64;
        if r.event {
            match r.direction {
                Some(1) => n_up += 1,
                Some(0) => n_down += 1,
                _ => {}
            }
        } else {
            n_censored += 1;
        }
    }
    let (lambda_up, lambda_down) = if total_time > 0.0 {
        (n_up as f64 / total_time, n_down as f64 / total_time)
    } else {
        (0.0, 0.0)
    };
    ConstantHazard {
        lambda_up_per_bar: lambda_up,
        lambda_down_per_bar: lambda_down,
        lambda_total_per_bar: lambda_up + lambda_down,
        n_events_up: n_up,
        n_events_down: n_down,
        n_censored,
        total_time_at_risk_bars: total_time,
        n_train: rows.len(),
    }
}

/// Waiting-time log-likelihood of `rows` under a constant hazard.
/// An event of cause `c` at wait `w` contributes
/// `ln(lambda_c) - lambda_total * w`; a censored row at wait `w`
/// contributes `-lambda_total * w`. An event of a cause the model
/// rates impossible (`lambda_c == 0`) makes the likelihood
/// `-inf`, which is the honest value, not a numerical error.
pub fn waiting_time_loglik(rows: &[HazardRow], model: &ConstantHazard) -> f64 {
    let mut ll = 0.0;
    for r in rows {
        if r.event {
            let lambda_c = match r.direction {
                Some(1) => model.lambda_up_per_bar,
                Some(0) => model.lambda_down_per_bar,
                _ => return f64::NEG_INFINITY,
            };
            if lambda_c <= 0.0 {
                return f64::NEG_INFINITY;
            }
            ll += lambda_c.ln() - model.lambda_total_per_bar * r.wait_bars as f64;
        } else {
            ll -= model.lambda_total_per_bar * r.wait_bars as f64;
        }
    }
    ll
}

/// Kaplan-Meier survival at `horizon_bars`: the product over event
/// times `t <= horizon` of `(1 - d_t / n_t)`, where `n_t` counts
/// rows with `wait >= t` (rows censored at `t` are still at risk at
/// `t`) and `d_t` counts events at `t`. Empty input survives with
/// probability 1.
pub fn kaplan_meier_survival(rows: &[HazardRow], horizon_bars: usize) -> f64 {
    let mut event_times: Vec<usize> = rows
        .iter()
        .filter(|r| r.event && r.wait_bars <= horizon_bars)
        .map(|r| r.wait_bars)
        .collect();
    event_times.sort_unstable();
    event_times.dedup();
    let mut s = 1.0;
    for t in event_times {
        let at_risk = rows.iter().filter(|r| r.wait_bars >= t).count();
        if at_risk == 0 {
            continue;
        }
        let d = rows.iter().filter(|r| r.event && r.wait_bars == t).count();
        s *= 1.0 - d as f64 / at_risk as f64;
    }
    s
}

/// One calibration row: fitted vs empirical survival at a horizon.
#[derive(Debug, Clone, Copy)]
pub struct CalibrationRow {
    pub horizon_bars: usize,
    pub n_rows: usize,
    /// Mean over rows of `exp(-lambda_i * horizon)`, where
    /// `lambda_i` is the total hazard scored against that row.
    pub predicted_survival: f64,
    /// Kaplan-Meier survival at the horizon.
    pub observed_survival: f64,
    /// Events with `wait <= horizon`.
    pub n_events_to_horizon: usize,
    /// Censored rows with `wait < horizon`.
    pub n_censored_to_horizon: usize,
}

/// Calibration table at `horizons`. `lambdas[i]` is the total hazard
/// scored against `rows[i]` (one fold's fit for a single-fold table,
/// each row's own fold fit for a pooled table).
pub fn calibration_table(
    rows: &[HazardRow],
    lambdas: &[f64],
    horizons: &[usize],
) -> Vec<CalibrationRow> {
    assert_eq!(rows.len(), lambdas.len());
    horizons
        .iter()
        .map(|&h| {
            let predicted = if rows.is_empty() {
                1.0
            } else {
                lambdas.iter().map(|l| (-l * h as f64).exp()).sum::<f64>() / rows.len() as f64
            };
            CalibrationRow {
                horizon_bars: h,
                n_rows: rows.len(),
                predicted_survival: predicted,
                observed_survival: kaplan_meier_survival(rows, h),
                n_events_to_horizon: rows.iter().filter(|r| r.event && r.wait_bars <= h).count(),
                n_censored_to_horizon: rows.iter().filter(|r| !r.event && r.wait_bars < h).count(),
            }
        })
        .collect()
}

/// One walk-forward fold of the hazard baseline. The model is fit on
/// the train rows only; the test rows are scored against it.
#[derive(Debug, Clone)]
pub struct HazardFold {
    pub fold: usize,
    pub train_end: usize,
    pub test_start: usize,
    pub test_end: usize,
    pub purge_bars: usize,
    pub model: ConstantHazard,
    pub n_test: usize,
    pub test_loglik: f64,
    pub test_mean_loglik: f64,
    pub calibration: Vec<CalibrationRow>,
}

/// Purged walk-forward evaluation of the hazard baseline. Fold
/// geometry mirrors `walk_forward_detailed`: fold `k` tests rows
/// `[k*fold, (k+1)*fold)` and trains on rows `[0, k*fold - purge)`,
/// so no train row's outcome window (at most `purge_bars` past its
/// decision bar) reaches a test row's decision bar. The pooled
/// log-likelihood is the sum of the fold test log-likelihoods, and
/// the pooled calibration scores every held-out row against its own
/// fold's fit.
#[derive(Debug, Clone)]
pub struct HazardWalkForward {
    pub folds: Vec<HazardFold>,
    pub n_rows: usize,
    pub n_test: usize,
    pub pooled_loglik: f64,
    pub pooled_mean_loglik: f64,
    pub pooled_calibration: Vec<CalibrationRow>,
}

pub fn walk_forward_hazard(
    rows: &[HazardRow],
    n_splits: usize,
    purge_bars: usize,
    horizons: &[usize],
) -> Option<HazardWalkForward> {
    let n = rows.len();
    if n < 60 {
        return None;
    }
    let fold = n / (n_splits + 1);
    if fold < 10 {
        return None;
    }
    let mut folds: Vec<HazardFold> = Vec::new();
    let mut pooled_rows: Vec<HazardRow> = Vec::new();
    let mut pooled_lambdas: Vec<f64> = Vec::new();
    let mut pooled_loglik = 0.0f64;
    for k in 1..=n_splits {
        let a = k * fold;
        let b = ((k + 1) * fold).min(n);
        let train_end = a.saturating_sub(purge_bars);
        if train_end < 30 || b - a < 10 {
            continue;
        }
        let model = fit_constant_hazard(&rows[..train_end]);
        let test = &rows[a..b];
        let ll = waiting_time_loglik(test, &model);
        let lambdas = vec![model.lambda_total_per_bar; test.len()];
        let calibration = calibration_table(test, &lambdas, horizons);
        pooled_loglik += ll;
        pooled_rows.extend_from_slice(test);
        pooled_lambdas.extend_from_slice(&lambdas);
        folds.push(HazardFold {
            fold: folds.len(),
            train_end,
            test_start: a,
            test_end: b,
            purge_bars,
            model,
            n_test: test.len(),
            test_loglik: ll,
            test_mean_loglik: ll / test.len() as f64,
            calibration,
        });
    }
    if folds.is_empty() {
        return None;
    }
    let n_test = pooled_rows.len();
    Some(HazardWalkForward {
        folds,
        n_rows: n,
        n_test,
        pooled_loglik,
        pooled_mean_loglik: pooled_loglik / n_test as f64,
        pooled_calibration: calibration_table(&pooled_rows, &pooled_lambdas, horizons),
    })
}

/// The frozen design grid: bar length, cutoff, splits, calibration
/// horizons, and test sessions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HazardGrid {
    pub bar_ms: i64,
    pub cutoff_bars: usize,
    pub n_splits: usize,
    pub calibration_horizons_bars: Vec<usize>,
    pub test_sessions: Vec<String>,
}

pub fn preregistered_hazard_grid() -> HazardGrid {
    HazardGrid {
        bar_ms: HAZARD_BAR_MS,
        cutoff_bars: HAZARD_CUTOFF_BARS,
        n_splits: HAZARD_N_SPLITS,
        calibration_horizons_bars: HAZARD_CALIBRATION_HORIZONS_BARS.to_vec(),
        test_sessions: HAZARD_TEST_SESSIONS.iter().map(|s| s.to_string()).collect(),
    }
}

/// Parse the design fields of an N5 grid file
/// (`experiments/n5/grid.json`).
pub fn hazard_grid_from_json(text: &str) -> Result<HazardGrid, String> {
    let v: serde_json::Value =
        serde_json::from_str(text).map_err(|e| format!("n5 grid.json does not parse: {e}"))?;
    let bar_ms = v
        .get("bar_ms")
        .and_then(|b| b.as_i64())
        .ok_or("n5 grid.json has no bar_ms")?;
    let cutoff = v
        .get("cutoff_bars")
        .and_then(|c| c.as_u64())
        .ok_or("n5 grid.json has no cutoff_bars")?;
    let n_splits = v
        .get("n_splits")
        .and_then(|s| s.as_u64())
        .ok_or("n5 grid.json has no n_splits")?;
    let horizons = v
        .get("calibration_horizons_bars")
        .and_then(|h| h.as_array())
        .ok_or("n5 grid.json has no calibration_horizons_bars")?;
    let mut calibration_horizons_bars = Vec::with_capacity(horizons.len());
    for (i, h) in horizons.iter().enumerate() {
        calibration_horizons_bars.push(
            h.as_u64()
                .ok_or(format!("n5 grid horizon {i} is not a number"))? as usize,
        );
    }
    let sessions = v
        .get("test_sessions")
        .and_then(|s| s.as_array())
        .ok_or("n5 grid.json has no test_sessions")?;
    let mut test_sessions = Vec::with_capacity(sessions.len());
    for (i, s) in sessions.iter().enumerate() {
        test_sessions.push(
            s.as_str()
                .ok_or(format!("n5 grid test session {i} is not a string"))?
                .to_string(),
        );
    }
    Ok(HazardGrid {
        bar_ms,
        cutoff_bars: cutoff as usize,
        n_splits: n_splits as usize,
        calibration_horizons_bars,
        test_sessions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One synthetic row with the N5 time identities baked in.
    fn row(wait: usize, event: bool, direction: Option<u8>) -> HazardRow {
        HazardRow {
            decision_bar: 0,
            t_observe_ms: 0,
            t_decision_ms: 10_000,
            t_outcome_ms: if event {
                Some(10_000 + (wait as i64 - 1) * 10_000)
            } else {
                None
            },
            t_censor_ms: if event {
                10_000 + (wait as i64 - 1) * 10_000
            } else {
                10_000 + wait as i64 * 10_000
            },
            wait_bars: wait,
            event,
            direction,
            censor: if event {
                None
            } else {
                Some(CensorReason::NoMove)
            },
            move_bar: if event { Some(wait) } else { None },
        }
    }

    fn oracle_model() -> ConstantHazard {
        ConstantHazard {
            lambda_up_per_bar: 0.10,
            lambda_down_per_bar: 0.05,
            lambda_total_per_bar: 0.15,
            n_events_up: 2,
            n_events_down: 1,
            n_censored: 2,
            total_time_at_risk_bars: 15.0,
            n_train: 5,
        }
    }

    fn oracle_rows() -> Vec<HazardRow> {
        vec![
            row(1, true, Some(1)),
            row(3, true, Some(1)),
            row(2, true, Some(0)),
            row(4, false, None),
            row(5, false, None),
        ]
    }

    #[test]
    fn n5_oracle_loglik_matches_hand_computation() {
        // Hand-computed, term by term, from the definition:
        // events ln(lambda_c) - lambda_total * wait; censored
        // -lambda_total * wait.
        let expected = (0.10f64.ln() - 0.15 * 1.0)
            + (0.10f64.ln() - 0.15 * 3.0)
            + (0.05f64.ln() - 0.15 * 2.0)
            + (-0.15 * 4.0)
            + (-0.15 * 5.0);
        let got = waiting_time_loglik(&oracle_rows(), &oracle_model());
        assert!(
            (got - expected).abs() < 1e-8,
            "oracle loglik {got} vs hand-computed {expected}"
        );
    }

    #[test]
    fn n5_mle_is_closed_form() {
        let model = fit_constant_hazard(&oracle_rows());
        // Total time at risk: 1 + 3 + 2 + 4 + 5 = 15 bars.
        assert_eq!(model.total_time_at_risk_bars, 15.0);
        assert_eq!(model.n_events_up, 2);
        assert_eq!(model.n_events_down, 1);
        assert_eq!(model.n_censored, 2);
        assert!((model.lambda_up_per_bar - 2.0 / 15.0).abs() < 1e-12);
        assert!((model.lambda_down_per_bar - 1.0 / 15.0).abs() < 1e-12);
        assert!((model.lambda_total_per_bar - 3.0 / 15.0).abs() < 1e-12);
    }

    #[test]
    fn n5_mle_maximizes_likelihood() {
        // On any rows, the closed-form fit must score at least as
        // well as the oracle that generated them.
        let rows = oracle_rows();
        let fitted = fit_constant_hazard(&rows);
        let ll_fit = waiting_time_loglik(&rows, &fitted);
        let ll_oracle = waiting_time_loglik(&rows, &oracle_model());
        assert!(
            ll_fit >= ll_oracle - 1e-12,
            "{ll_fit} vs oracle {ll_oracle}"
        );
    }

    /// Deterministic SplitMix64, mirroring the simulation harness
    /// in `examples/speed_power_sim.rs`.
    struct TestRng(u64);

    impl TestRng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^ (z >> 31)
        }

        fn next_f64(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    #[test]
    fn n5_constant_hazard_oracle_recovery() {
        // Synthetic constant-hazard sample: total hazard 0.12/bar
        // (up 0.08, down 0.04), cutoff-censored at 10 bars, seed
        // fixed. The MLE must recover each hazard within 10%.
        let mut rng = TestRng(2026_1117);
        let mut rows = Vec::with_capacity(4000);
        for _ in 0..4000 {
            let u = rng.next_f64();
            let w_cont = -(1.0 - u).ln() / 0.12;
            let wait = (w_cont.ceil() as usize).max(1);
            let up = rng.next_f64() < 0.08 / 0.12;
            if wait > 10 {
                rows.push(row(10, false, None));
            } else {
                rows.push(row(wait, true, Some(u8::from(up))));
            }
        }
        let model = fit_constant_hazard(&rows);
        assert!(
            (model.lambda_up_per_bar - 0.08).abs() / 0.08 < 0.10,
            "lambda_up {} vs 0.08",
            model.lambda_up_per_bar
        );
        assert!(
            (model.lambda_down_per_bar - 0.04).abs() / 0.04 < 0.10,
            "lambda_down {} vs 0.04",
            model.lambda_down_per_bar
        );
        assert!(
            (model.lambda_total_per_bar - 0.12).abs() / 0.12 < 0.10,
            "lambda_total {} vs 0.12",
            model.lambda_total_per_bar
        );
        // And the fitted likelihood beats the oracle's on its own
        // sample, as the MLE must.
        let oracle = ConstantHazard {
            lambda_up_per_bar: 0.08,
            lambda_down_per_bar: 0.04,
            lambda_total_per_bar: 0.12,
            n_events_up: 0,
            n_events_down: 0,
            n_censored: 0,
            total_time_at_risk_bars: 0.0,
            n_train: 0,
        };
        assert!(waiting_time_loglik(&rows, &model) >= waiting_time_loglik(&rows, &oracle));
    }

    #[test]
    fn n5_kaplan_meier_known_answer() {
        // Events at waits 1, 2, 2, 5; censored at 3, 4.
        // S(1) = 5/6; S(2) = 5/6 * 3/5 = 1/2; S(5) = 0.
        let rows = vec![
            row(1, true, Some(1)),
            row(2, true, Some(0)),
            row(2, true, Some(1)),
            row(5, true, Some(0)),
            row(3, false, None),
            row(4, false, None),
        ];
        assert!((kaplan_meier_survival(&rows, 0) - 1.0).abs() < 1e-12);
        assert!((kaplan_meier_survival(&rows, 1) - 5.0 / 6.0).abs() < 1e-12);
        assert!((kaplan_meier_survival(&rows, 2) - 0.5).abs() < 1e-12);
        assert!((kaplan_meier_survival(&rows, 3) - 0.5).abs() < 1e-12);
        assert!((kaplan_meier_survival(&rows, 4) - 0.5).abs() < 1e-12);
        assert!((kaplan_meier_survival(&rows, 5) - 0.0).abs() < 1e-12);
        assert!((kaplan_meier_survival(&[], 5) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn n5_calibration_table_known_answer() {
        let rows = oracle_rows();
        let lambdas = vec![0.15; rows.len()];
        let table = calibration_table(&rows, &lambdas, &[1, 2, 10]);
        assert_eq!(table.len(), 3);
        // Horizon 1: KM is 4/5 (one event at wait 1 of five rows);
        // predicted exp(-0.15).
        assert!((table[0].observed_survival - 0.8).abs() < 1e-12);
        assert!((table[0].predicted_survival - (-0.15f64).exp()).abs() < 1e-12);
        assert_eq!(table[0].n_events_to_horizon, 1);
        assert_eq!(table[0].n_censored_to_horizon, 0);
        // Horizon 2: KM is 4/5 * 3/4 = 0.6 (events at waits 1, 2).
        assert!((table[1].observed_survival - 0.6).abs() < 1e-12);
        assert!((table[1].predicted_survival - (-0.30f64).exp()).abs() < 1e-12);
        assert_eq!(table[1].n_events_to_horizon, 2);
        assert_eq!(table[1].n_censored_to_horizon, 0);
        // Horizon 10: all three events observed, but the two rows
        // censored at waits 4 and 5 stay at risk past the last event
        // time, so KM settles at 4/5 * 3/4 * 2/3 = 0.4, not 0.
        assert!((table[2].observed_survival - 0.4).abs() < 1e-12);
        assert_eq!(table[2].n_events_to_horizon, 3);
        assert_eq!(table[2].n_censored_to_horizon, 2);
    }

    #[test]
    fn n5_no_events_fit_is_zero() {
        let rows = vec![row(10, false, None), row(5, false, None)];
        let model = fit_constant_hazard(&rows);
        assert_eq!(model.lambda_total_per_bar, 0.0);
        assert_eq!(model.n_censored, 2);
        // Censored rows under a zero hazard survive with certainty.
        assert_eq!(waiting_time_loglik(&rows, &model), 0.0);
    }

    #[test]
    fn n5_impossible_cause_event_is_minus_inf() {
        // Train rows with up events only: the down hazard is zero,
        // so a down event on held-out rows scores -inf, honestly.
        let train = vec![row(2, true, Some(1)), row(3, true, Some(1))];
        let model = fit_constant_hazard(&train);
        assert_eq!(model.lambda_down_per_bar, 0.0);
        let test = vec![row(1, true, Some(0))];
        assert_eq!(waiting_time_loglik(&test, &model), f64::NEG_INFINITY);
    }

    #[test]
    fn n5_walk_forward_train_only() {
        // 200 flat-then-stepping rows: waits cycle 1..10 so every
        // fold has both causes in train and test.
        let mut rows = Vec::with_capacity(200);
        for k in 0..200 {
            let wait = (k % 10) + 1;
            let up = k % 2 == 0;
            rows.push(row(wait, true, Some(u8::from(up))));
        }
        let wf = walk_forward_hazard(&rows, 4, 10, &[1, 2, 3, 5, 10]).expect("walk-forward");
        // Geometry mirrors walk_forward_detailed: fold = 200/5 = 40.
        assert_eq!(wf.folds.len(), 4);
        for (k, f) in wf.folds.iter().enumerate() {
            let a = (k + 1) * 40;
            let b = ((k + 2) * 40).min(200);
            assert_eq!(f.test_start, a);
            assert_eq!(f.test_end, b);
            assert_eq!(f.train_end, a - 10);
            // The fold model is exactly the fit on its train slice.
            let direct = fit_constant_hazard(&rows[..f.train_end]);
            assert_eq!(f.model.lambda_up_per_bar, direct.lambda_up_per_bar);
            assert_eq!(f.model.lambda_down_per_bar, direct.lambda_down_per_bar);
            assert_eq!(
                f.test_loglik,
                waiting_time_loglik(&rows[f.test_start..f.test_end], &f.model)
            );
        }
        let sum: f64 = wf.folds.iter().map(|f| f.test_loglik).sum();
        assert!((wf.pooled_loglik - sum).abs() < 1e-9);
        assert_eq!(wf.n_test, wf.folds.iter().map(|f| f.n_test).sum::<usize>());
        assert_eq!(wf.pooled_calibration.len(), 5);
    }

    #[test]
    fn n5_walk_forward_too_short_returns_none() {
        let rows: Vec<HazardRow> = (0..50).map(|k| row((k % 10) + 1, true, Some(1))).collect();
        assert!(walk_forward_hazard(&rows, 4, 10, &[1]).is_none());
    }

    #[test]
    fn n5_grid_parse_roundtrip() {
        let grid = preregistered_hazard_grid();
        let text = serde_json::json!({
            "bar_ms": grid.bar_ms,
            "cutoff_bars": grid.cutoff_bars,
            "n_splits": grid.n_splits,
            "calibration_horizons_bars": grid.calibration_horizons_bars,
            "test_sessions": grid.test_sessions,
        })
        .to_string();
        assert_eq!(hazard_grid_from_json(&text).unwrap(), grid);
        // A mutated grid must not match the library grid.
        let mutated = serde_json::json!({
            "bar_ms": grid.bar_ms,
            "cutoff_bars": grid.cutoff_bars + 1,
            "n_splits": grid.n_splits,
            "calibration_horizons_bars": grid.calibration_horizons_bars,
            "test_sessions": grid.test_sessions,
        })
        .to_string();
        assert_ne!(hazard_grid_from_json(&mutated).unwrap(), grid);
    }
}
