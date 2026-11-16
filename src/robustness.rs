//! Robustness, negative controls and estimator sensitivity (Stage 3 E5).
//!
//! The headline estimators use one fixed parameter set (50 buckets,
//! 20-bucket VPIN window, 10 s bars, 60-bar lambda window, 0.7 VPIN
//! gate) with no documented robustness, and Kyle's lambda has an
//! extreme tail. This module adds:
//!
//! * a preregistered parameter grid (`preregistered_grid`), fixed
//!   before any grid run and checked against
//!   `experiments/e5/grid.json` by the runner and the tests;
//! * per-window lambda diagnostics: valid sample counts, tail flags,
//!   and a winsorized variant (`kyle_lambda_robust`);
//! * train-only calibration of the VPIN gate
//!   (`calibrate_gate_train_only`);
//! * lead-lag probes of the OFI price response
//!   (`lead_lag_response`) and timestamp-shift nulls
//!   (`shift_flow_bars`);
//! * OFI response ablations with volume, intensity, and volatility
//!   controls (`ofi_response_ablation`).
//!
//! Nothing here selects parameters from test outcomes: the default
//! configuration stays the reported configuration regardless of
//! results. The grid only documents how far each number moves.

use serde::{Deserialize, Serialize};

use crate::evidence::sha256_hex;
use crate::metrics::{FlowBar, Quote, Trade};
use crate::strategy::MmConfig;

/// One estimator configuration: VPIN bucket count and window, OFI
/// bar length, lambda window, and VPIN gate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RobustnessConfig {
    pub n_buckets: usize,
    pub vpin_window: usize,
    pub bar_ms: i64,
    pub lambda_window: usize,
    pub vpin_gate: f64,
}

impl RobustnessConfig {
    pub fn new(
        n_buckets: usize,
        vpin_window: usize,
        bar_ms: i64,
        lambda_window: usize,
        vpin_gate: f64,
    ) -> Self {
        Self {
            n_buckets,
            vpin_window,
            bar_ms,
            lambda_window,
            vpin_gate,
        }
    }

    /// The headline configuration: 50 buckets, 20-bucket window,
    /// 10 s bars, 60-bar lambda window, 0.7 gate.
    pub fn default_config() -> Self {
        Self::new(50, 20, 10_000, 60, 0.7)
    }

    /// Stable short id: `default`, or the parameters that differ
    /// from the default joined with `-` (`b25`, `w10`, `bar5`,
    /// `lam30`, `g060`, `b25-w10`).
    pub fn id(&self) -> String {
        let d = Self::default_config();
        if *self == d {
            return "default".to_string();
        }
        let mut parts: Vec<String> = Vec::new();
        if self.n_buckets != d.n_buckets {
            parts.push(format!("b{}", self.n_buckets));
        }
        if self.vpin_window != d.vpin_window {
            parts.push(format!("w{}", self.vpin_window));
        }
        if self.bar_ms != d.bar_ms {
            parts.push(format!("bar{}", self.bar_ms / 1000));
        }
        if self.lambda_window != d.lambda_window {
            parts.push(format!("lam{}", self.lambda_window));
        }
        if (self.vpin_gate - d.vpin_gate).abs() > 1e-12 {
            parts.push(format!("g{:03}", (self.vpin_gate * 100.0).round() as i64));
        }
        parts.join("-")
    }

    /// Content hash of the canonical parameter encoding, so a
    /// configuration in a results file can be checked exactly.
    pub fn config_hash(&self) -> String {
        sha256_hex(
            format!(
                "n_buckets={};vpin_window={};bar_ms={};lambda_window={};vpin_gate={:.3}",
                self.n_buckets, self.vpin_window, self.bar_ms, self.lambda_window, self.vpin_gate
            )
            .as_bytes(),
        )
    }

    /// Market-making configuration for this grid point: the gate,
    /// bucket count, and window follow the grid; size and
    /// calibration window stay at their defaults.
    pub fn mm_config(&self) -> MmConfig {
        MmConfig {
            vpin_threshold: self.vpin_gate,
            vpin_buckets: self.n_buckets,
            vpin_window: self.vpin_window,
            ..MmConfig::default()
        }
    }

    /// Per-configuration manifest: id, parameters, hash, session,
    /// and input hash. Every grid result carries one.
    pub fn manifest(&self, session: &str, input_hash: &str) -> serde_json::Value {
        serde_json::json!({
            "protocol": "e5-robustness/v1",
            "config_id": self.id(),
            "config": {
                "n_buckets": self.n_buckets,
                "vpin_window": self.vpin_window,
                "bar_ms": self.bar_ms,
                "lambda_window": self.lambda_window,
                "vpin_gate": self.vpin_gate,
            },
            "config_hash": self.config_hash(),
            "session": session,
            "input_hash": input_hash,
        })
    }
}

/// The preregistered grid: the default, one-at-a-time sweeps of each
/// parameter, and the buckets-by-window factorial. Fixed before any
/// grid run; the runner refuses a grid file that differs from this.
pub fn preregistered_grid() -> Vec<RobustnessConfig> {
    let d = RobustnessConfig::default_config();
    vec![
        d,
        RobustnessConfig::new(25, 20, 10_000, 60, 0.7),
        RobustnessConfig::new(100, 20, 10_000, 60, 0.7),
        RobustnessConfig::new(50, 10, 10_000, 60, 0.7),
        RobustnessConfig::new(50, 40, 10_000, 60, 0.7),
        RobustnessConfig::new(50, 20, 5_000, 60, 0.7),
        RobustnessConfig::new(50, 20, 30_000, 60, 0.7),
        RobustnessConfig::new(50, 20, 10_000, 30, 0.7),
        RobustnessConfig::new(50, 20, 10_000, 120, 0.7),
        RobustnessConfig::new(50, 20, 10_000, 60, 0.6),
        RobustnessConfig::new(50, 20, 10_000, 60, 0.8),
        RobustnessConfig::new(25, 10, 10_000, 60, 0.7),
        RobustnessConfig::new(25, 40, 10_000, 60, 0.7),
        RobustnessConfig::new(100, 10, 10_000, 60, 0.7),
        RobustnessConfig::new(100, 40, 10_000, 60, 0.7),
    ]
}

fn config_from_value(v: &serde_json::Value) -> Option<RobustnessConfig> {
    Some(RobustnessConfig::new(
        v["n_buckets"].as_u64()? as usize,
        v["vpin_window"].as_u64()? as usize,
        v["bar_ms"].as_i64()?,
        v["lambda_window"].as_u64()? as usize,
        v["vpin_gate"].as_f64()?,
    ))
}

/// Parse the `configs` array of a grid file (e.g.
/// `experiments/e5/grid.json`) into configurations, in file order.
pub fn grid_from_json(text: &str) -> Result<Vec<RobustnessConfig>, String> {
    let v: serde_json::Value =
        serde_json::from_str(text).map_err(|e| format!("grid JSON must parse: {e}"))?;
    let arr = v["configs"]
        .as_array()
        .ok_or_else(|| "grid must hold a configs array".to_string())?;
    let mut out = Vec::with_capacity(arr.len());
    for (i, c) in arr.iter().enumerate() {
        out.push(
            config_from_value(c)
                .ok_or_else(|| format!("grid config {i} is missing a parameter"))?,
        );
        let want = out.last().unwrap().id();
        let got = c["id"].as_str().unwrap_or("");
        if want != got {
            return Err(format!(
                "grid config {i} id mismatch: file says {got}, parameters say {want}"
            ));
        }
    }
    Ok(out)
}

fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = (p * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn median(xs: &[f64]) -> f64 {
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    percentile_sorted(&s, 0.5)
}

/// One rolling lambda window with its diagnostics.
#[derive(Debug, Clone, Copy)]
pub struct LambdaWindow {
    pub time_ms: i64,
    /// Rolling OLS slope, `None` where the window is too short or
    /// flow variance is zero (same rule as `kyle_lambda`).
    pub lambda: Option<f64>,
    pub r2: Option<f64>,
    /// Regression points actually used in this window. A window
    /// cut short by a feed gap has fewer valid points than the
    /// requested window; the count is reported even when the fit
    /// itself is undefined.
    pub n_valid: usize,
    /// Requested window length.
    pub n_total: usize,
    /// True when this lambda sits far outside the bulk of the
    /// series: more than ten robust standard deviations (1.4826
    /// times the median absolute deviation) from the median lambda.
    pub tail_flag: bool,
    /// Lambda clamped to the 1st/99th percentiles of the
    /// non-flagged windows, so one extreme window cannot dominate
    /// a session mean.
    pub winsorized_lambda: Option<f64>,
}

/// Rolling OLS lambda with per-window valid counts, tail flags, and
/// a winsorized variant. The fit itself is identical to
/// `kyle_lambda` (segment restarts at feed gaps, at least 10 points,
/// nonzero flow variance); only the diagnostics are new.
pub fn kyle_lambda_robust(bars: &[FlowBar], window: usize) -> Vec<LambdaWindow> {
    // Segment starts: one past the latest boundary bar, or 0.
    let mut seg_start = vec![0usize; bars.len()];
    let mut cur = 0usize;
    for (i, b) in bars.iter().enumerate() {
        if b.gap_before {
            cur = i + 1;
        }
        seg_start[i] = cur;
    }
    // Raw fits first: the tail rule needs the whole series.
    let mut raw: Vec<(Option<f64>, Option<f64>, usize)> = Vec::with_capacity(bars.len());
    for (i, _) in bars.iter().enumerate() {
        let start = (i + 1).saturating_sub(window).max(seg_start[i]);
        let xs: Vec<f64> = bars[start..=i].iter().map(|b| b.signed_vol).collect();
        let ys: Vec<f64> = bars[start..=i].iter().map(|b| b.dmid).collect();
        let n = xs.len();
        if n < 10 {
            raw.push((None, None, n));
            continue;
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
            raw.push((None, None, n));
        } else {
            raw.push((Some(sxy / sxx), Some((sxy * sxy) / (sxx * syy)), n));
        }
    }
    let lambdas: Vec<f64> = raw.iter().filter_map(|(l, _, _)| *l).collect();
    let med = median(&lambdas);
    let devs: Vec<f64> = lambdas.iter().map(|l| (l - med).abs()).collect();
    let mad = median(&devs);
    let eps = (10.0 * 1.4826 * mad).max(1e-9);
    let mut non_flagged: Vec<f64> = Vec::new();
    for (l, _, _) in &raw {
        if let Some(v) = l {
            if (v - med).abs() <= eps {
                non_flagged.push(*v);
            }
        }
    }
    non_flagged.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lo = percentile_sorted(&non_flagged, 0.01);
    let hi = percentile_sorted(&non_flagged, 0.99);
    raw.into_iter()
        .enumerate()
        .map(|(i, (lambda, r2, n_valid))| {
            let tail_flag = lambda.map(|v| (v - med).abs() > eps).unwrap_or(false);
            let winsorized_lambda = lambda.map(|v| {
                if non_flagged.is_empty() {
                    v
                } else {
                    v.clamp(lo, hi)
                }
            });
            LambdaWindow {
                time_ms: bars[i].time_ms,
                lambda,
                r2,
                n_valid,
                n_total: window,
                tail_flag,
                winsorized_lambda,
            }
        })
        .collect()
}

/// Session summary of the robust lambda windows.
#[derive(Debug, Clone, Copy)]
pub struct LambdaSummary {
    pub n_windows: usize,
    pub n_with_lambda: usize,
    pub n_tail: usize,
    pub tail_share: f64,
    pub mean_lambda: f64,
    pub winsorized_mean_lambda: f64,
    pub median_abs_lambda: f64,
    pub p99_abs_lambda: f64,
    /// Regression points used across all windows.
    pub total_n_valid: usize,
}

pub fn lambda_tail_summary(windows: &[LambdaWindow]) -> LambdaSummary {
    let n_with_lambda = windows.iter().filter(|w| w.lambda.is_some()).count();
    let n_tail = windows.iter().filter(|w| w.tail_flag).count();
    let mean = |f: fn(&LambdaWindow) -> Option<f64>| -> f64 {
        let vals: Vec<f64> = windows.iter().filter_map(f).collect();
        if vals.is_empty() {
            f64::NAN
        } else {
            vals.iter().sum::<f64>() / vals.len() as f64
        }
    };
    let abs: Vec<f64> = windows
        .iter()
        .filter_map(|w| w.lambda.map(|v| v.abs()))
        .collect();
    let mut abs_sorted = abs.clone();
    abs_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    LambdaSummary {
        n_windows: windows.len(),
        n_with_lambda,
        n_tail,
        tail_share: if n_with_lambda == 0 {
            0.0
        } else {
            n_tail as f64 / n_with_lambda as f64
        },
        mean_lambda: mean(|w| w.lambda),
        winsorized_mean_lambda: mean(|w| w.winsorized_lambda),
        median_abs_lambda: median(&abs),
        p99_abs_lambda: percentile_sorted(&abs_sorted, 0.99),
        total_n_valid: windows.iter().map(|w| w.n_valid).sum(),
    }
}

/// Calibrate the VPIN gate from the chronological train split only:
/// the `percentile` quantile of the train VPIN values. The test
/// split is never read, so the calibrated gate cannot depend on any
/// outcome it is later evaluated on. `None` on an empty train set.
pub fn calibrate_gate_train_only(train_vpin: &[f64], percentile: f64) -> Option<f64> {
    if train_vpin.is_empty() || !(0.0..1.0).contains(&percentile) {
        return None;
    }
    let mut sorted = train_vpin.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(percentile_sorted(&sorted, percentile))
}

/// One point of the lead-lag curve: the slope of forward midprice
/// returns (bps) on standardized OFI at a given lag.
#[derive(Debug, Clone, Copy)]
pub struct LeadLagPoint {
    /// Lag in bars: positive means OFI leads the return, zero is
    /// contemporaneous, negative is the negative control (OFI
    /// predicting a return that already happened).
    pub lag: i32,
    /// OLS slope in bps per one standard deviation of OFI.
    pub slope_bps_per_sd: f64,
    pub n: usize,
}

/// Lead-lag probe of the OFI price response. `ret_bps[i]` is the
/// midprice return over the bar starting at `i`, in bps, parallel to
/// `ofi[i]`. A real response shows up at lag 0 or positive lags;
/// the negative lags are controls and must stay near zero when the
/// alignment is honest.
pub fn lead_lag_response(ofi: &[f64], ret_bps: &[f64], max_lag: usize) -> Vec<LeadLagPoint> {
    let n = ofi.len().min(ret_bps.len());
    let mut out = Vec::new();
    for lag in -(max_lag as i32)..=(max_lag as i32) {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for (i, &ofi_val) in ofi.iter().enumerate().take(n) {
            let j = i as i64 + lag as i64;
            if j >= 0 && (j as usize) < n {
                xs.push(ofi_val);
                ys.push(ret_bps[j as usize]);
            }
        }
        let m = xs.len();
        if m < 10 {
            out.push(LeadLagPoint {
                lag,
                slope_bps_per_sd: f64::NAN,
                n: m,
            });
            continue;
        }
        let mx = xs.iter().sum::<f64>() / m as f64;
        let var = xs.iter().map(|x| (x - mx).powi(2)).sum::<f64>() / m as f64;
        let sd = var.sqrt();
        if sd == 0.0 {
            out.push(LeadLagPoint {
                lag,
                slope_bps_per_sd: f64::NAN,
                n: m,
            });
            continue;
        }
        let my = ys.iter().sum::<f64>() / m as f64;
        let mut num = 0.0;
        let mut den = 0.0;
        for k in 0..m {
            let z = (xs[k] - mx) / sd;
            num += z * (ys[k] - my);
            den += z * z;
        }
        out.push(LeadLagPoint {
            lag,
            slope_bps_per_sd: if den == 0.0 { f64::NAN } else { num / den },
            n: m,
        });
    }
    out
}

/// Shift signed flow `shift_bars` bars against the price changes:
/// positive shifts move flow later, breaking the contemporaneous
/// alignment. The null lambda on the shifted bars must collapse
/// toward zero when the original relation is real and aligned.
pub fn shift_flow_bars(bars: &[FlowBar], shift_bars: isize) -> Vec<FlowBar> {
    bars.iter()
        .enumerate()
        .map(|(i, b)| {
            let src = i as isize - shift_bars;
            let signed_vol = if src >= 0 && (src as usize) < bars.len() {
                bars[src as usize].signed_vol
            } else {
                0.0
            };
            FlowBar {
                time_ms: b.time_ms,
                signed_vol,
                dmid: b.dmid,
                gap_before: b.gap_before,
            }
        })
        .collect()
}

/// Per-bar signed flow and midprice change on a fixed bar grid, with
/// the same gap flags as the export path: a bar is flagged when it
/// or the preceding bar had no observed quote, so rolling consumers
/// restart at the gap instead of treating the outage as continuous
/// time. Bars with no quotes carry the last observed mid forward.
pub fn flow_bars(trades: &[Trade], quotes: &[Quote], bar_ms: i64) -> Vec<FlowBar> {
    if trades.is_empty() || quotes.is_empty() || bar_ms <= 0 {
        return Vec::new();
    }
    let t0 = quotes[0].time_ms / bar_ms * bar_ms;
    let n = ((quotes.last().unwrap().time_ms - t0) / bar_ms + 1).max(1) as usize;
    let mut signed = vec![0.0; n];
    let mut mid = vec![f64::NAN; n];
    let mut obs = vec![false; n];
    for t in trades {
        if t.time_ms < t0 {
            continue;
        }
        let i = ((t.time_ms - t0) / bar_ms) as usize;
        if i < n {
            signed[i] += t.signed_qty();
        }
    }
    for q in quotes {
        let i = ((q.time_ms - t0) / bar_ms) as usize;
        if i < n {
            mid[i] = q.mid();
            obs[i] = true;
        }
    }
    let mut last = f64::NAN;
    let mut out = Vec::new();
    for i in 0..n {
        if mid[i].is_nan() {
            mid[i] = last;
        } else {
            last = mid[i];
        }
        if i > 0 && !mid[i].is_nan() && !mid[i - 1].is_nan() {
            out.push(FlowBar {
                time_ms: t0 + i as i64 * bar_ms,
                signed_vol: signed[i],
                dmid: mid[i] - mid[i - 1],
                gap_before: !obs[i] || !obs[i - 1],
            });
        }
    }
    out
}

/// One bar for the OFI response ablation.
#[derive(Debug, Clone, Copy)]
pub struct AblationRow {
    pub ofi: f64,
    pub dmid: f64,
    /// Unsigned traded volume in the bar.
    pub volume: f64,
    /// Trade count in the bar.
    pub intensity: f64,
    /// Trailing mean absolute midprice change.
    pub volatility: f64,
}

/// OFI response with and without controls.
#[derive(Debug, Clone, Copy)]
pub struct AblationResult {
    /// Univariate OLS slope of dmid on OFI (with intercept).
    pub slope_raw: f64,
    /// Slope on OFI with volume, intensity, and volatility held
    /// fixed.
    pub slope_controlled: f64,
    pub se_controlled: f64,
    pub t_controlled: f64,
    pub r2_controlled: f64,
    pub n: usize,
}

/// Solve a small dense linear system by Gaussian elimination with
/// partial pivoting. `None` when the matrix is singular.
fn solve_linear(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    if a.len() != n || a.iter().any(|r| r.len() != n) {
        return None;
    }
    for col in 0..n {
        let mut piv = col;
        for r in col + 1..n {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for r in col + 1..n {
            let f = a[r][col] / a[col][col];
            for c in col..n {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in i + 1..n {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
    }
    Some(x)
}

/// Invert a small dense matrix by Gauss-Jordan elimination.
/// `None` when the matrix is singular.
fn invert(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    if a.iter().any(|r| r.len() != n) {
        return None;
    }
    let mut m: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = a[i].clone();
            row.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 }));
            row
        })
        .collect();
    for col in 0..n {
        let mut piv = col;
        for r in col + 1..n {
            if m[r][col].abs() > m[piv][col].abs() {
                piv = r;
            }
        }
        if m[piv][col].abs() < 1e-12 {
            return None;
        }
        m.swap(col, piv);
        let d = m[col][col];
        for v in m[col].iter_mut() {
            *v /= d;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = m[r][col];
            if f != 0.0 {
                for c in 0..2 * n {
                    m[r][c] -= f * m[col][c];
                }
            }
        }
    }
    Some(
        m.iter()
            .map(|row| row[n..2 * n].to_vec())
            .collect::<Vec<_>>(),
    )
}

fn ols(rows: &[AblationRow], use_controls: bool) -> Option<(Vec<f64>, f64, f64)> {
    let p = if use_controls { 5 } else { 2 };
    let n = rows.len();
    if n <= p {
        return None;
    }
    let design = |r: &AblationRow| -> Vec<f64> {
        if use_controls {
            vec![1.0, r.ofi, r.volume, r.intensity, r.volatility]
        } else {
            vec![1.0, r.ofi]
        }
    };
    let mut xtx = vec![vec![0.0; p]; p];
    let mut xty = vec![0.0; p];
    for r in rows {
        let x = design(r);
        for i in 0..p {
            for j in 0..p {
                xtx[i][j] += x[i] * x[j];
            }
            xty[i] += x[i] * r.dmid;
        }
    }
    let beta = solve_linear(xtx.clone(), xty)?;
    let mut rss = 0.0;
    let mean_y = rows.iter().map(|r| r.dmid).sum::<f64>() / n as f64;
    let mut tss = 0.0;
    for r in rows {
        let x = design(r);
        let fit: f64 = x.iter().zip(beta.iter()).map(|(a, b)| a * b).sum();
        rss += (r.dmid - fit).powi(2);
        tss += (r.dmid - mean_y).powi(2);
    }
    let r2 = if tss == 0.0 {
        if rss == 0.0 {
            1.0
        } else {
            0.0
        }
    } else {
        1.0 - rss / tss
    };
    Some((beta, rss, r2))
}

/// OFI price response, raw and with volume, intensity, and
/// volatility controls. `None` when the design is too small or
/// singular. The raw slope carries any omitted-variable bias from
/// the controls; the controlled slope is the OFI coefficient with
/// those held fixed.
pub fn ofi_response_ablation(rows: &[AblationRow]) -> Option<AblationResult> {
    if rows.len() <= 5 {
        return None;
    }
    let (raw_beta, _, _) = ols(rows, false)?;
    let (beta, rss, r2) = ols(rows, true)?;
    let p = 5;
    let n = rows.len();
    let dof = (n - p) as f64;
    if dof <= 0.0 {
        return None;
    }
    let sigma2 = rss / dof;
    let mut xtx = vec![vec![0.0; p]; p];
    for r in rows {
        let x = [1.0, r.ofi, r.volume, r.intensity, r.volatility];
        for i in 0..p {
            for j in 0..p {
                xtx[i][j] += x[i] * x[j];
            }
        }
    }
    let inv = invert(&xtx)?;
    let se = (sigma2 * inv[1][1]).max(0.0).sqrt();
    Some(AblationResult {
        slope_raw: raw_beta[1],
        slope_controlled: beta[1],
        se_controlled: se,
        t_controlled: if se == 0.0 {
            f64::INFINITY * beta[1].signum()
        } else {
            beta[1] / se
        },
        r2_controlled: r2,
        n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_ids_match_parameters() {
        for c in preregistered_grid() {
            let again = RobustnessConfig::new(
                c.n_buckets,
                c.vpin_window,
                c.bar_ms,
                c.lambda_window,
                c.vpin_gate,
            );
            assert_eq!(c.id(), again.id());
        }
        assert_eq!(
            preregistered_grid()
                .iter()
                .map(|c| c.id())
                .collect::<Vec<_>>(),
            vec![
                "default", "b25", "b100", "w10", "w40", "bar5", "bar30", "lam30", "lam120", "g060",
                "g080", "b25-w10", "b25-w40", "b100-w10", "b100-w40"
            ]
        );
    }

    #[test]
    fn percentile_known_values() {
        let xs = vec![1.0, 2.0, 3.0, 4.0];
        assert!((percentile_sorted(&xs, 0.5) - 3.0).abs() < 1e-12);
        assert!((percentile_sorted(&xs, 0.0) - 1.0).abs() < 1e-12);
        assert!((percentile_sorted(&xs, 1.0) - 4.0).abs() < 1e-12);
    }
}
