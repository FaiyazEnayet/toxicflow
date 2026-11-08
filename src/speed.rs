//! Observational OFI-price response: how the mid price moves in
//! the seconds and minutes after a bar of order-flow imbalance.
//!
//! The existing metrics say how much prices move with order flow
//! (Kyle's lambda, OFI, VPIN). This module adds the time dimension:
//! the shape of the cumulative mid-price response after a shock
//! bar, reported as an observational association. Order flow and
//! prices are measured from the same quote stream and the recorded
//! sessions carry no news timestamps, so nothing here identifies a
//! news event, an informed trader, or a causal effect: a "shock" is
//! one bar of order-flow imbalance (OFI, Cont-Kukanov-Stoikov, as
//! computed in `metrics`), scaled by its own session standard
//! deviation, and the response is the local-projection association
//! between that bar's OFI and later mid-price changes.
//!
//! Timing conventions (two series are reported per session):
//!
//! - Inclusive response: the baseline is the close of the bar
//!   *before* the shock bar, so the h = 0 rung is the move inside
//!   the shock bar itself and every later rung includes it.
//! - Strictly post-bar response: the baseline is the shock bar's
//!   own close, so the h = 0 rung is identically zero by
//!   construction (it is stated, not estimated) and later rungs
//!   measure only movement after the shock bar closes.
//!
//! Method: local projections. For each horizon h in a fixed ladder,
//! the cumulative mid return (simple return, basis points) from the
//! baseline bar to h bars after the shock bar is regressed on the
//! standardized shock (one OLS regression per horizon, intercept
//! included). Local projections are used instead of a VAR because
//! the object of interest is the whole response path: each horizon
//! gets its own regression, so the curve is not forced through a
//! VAR's implied dynamics. Origins whose window touches a bar with
//! no observed quote are censored, the same gap discipline as the
//! prediction dataset: a feed outage is never treated as
//! continuous time.
//!
//! Two estimands are reported side by side and never conflated:
//! the response to raw OFI (the total association, including any
//! follow-on flow the originating bar predicts) and the response to
//! OFI innovations (residuals from regressing OFI on its own past
//! 10 bars, so the regressor is the part of the shock not predicted
//! by recent flow). Shock diagnostics (OFI autocorrelation and
//! pre-shock price-movement regressions) accompany both.
//!
//! Uncertainty: a segment-respecting moving-block bootstrap. Each
//! session is split into its contiguous fully-observed segments and
//! blocks are drawn only within a segment, with splice points
//! marked so no response window in a resample ever crosses a splice,
//! an unobserved stretch, or the session boundary. The headline
//! block length is the longest horizon (300 s); sensitivity to
//! 150/600/900 s blocks is reported alongside, never cherry-picked.
//! Intervals are (2.5%, 97.5%) percentiles of the resample
//! distribution with a fixed published seed and resample count, so
//! an interval is a deterministic function of the session.
//! Simultaneous bands across horizons (sup-statistic) and
//! prespecified horizon contrasts are computed from the same draws.
//! A horizon-aware HAC (Newey-West, lag truncation at the horizon)
//! standard error is reported next to the bootstrap intervals.
//!
//! The cumulative curve is also fit with an adjustment path that
//! carries a separately identified instantaneous term,
//! R(h) = J + R_inf * (1 - exp(-h / tau)), by profiling tau over a
//! grid. The headline fit is weighted by the bootstrap covariance
//! of the response vector (the horizons are correlated estimates,
//! not independent points); the fit is also re-run unweighted
//! inside every bootstrap draw, which is what the reported fit
//! share and parameter region summarize. A tau (and the half-life
//! tau * ln 2) is reported only when the fit is actually good
//! (R^2 at the published threshold, positive long-run response, tau
//! interior to the search grid) *and* the joint parameter region
//! from the within-draw fits is bounded; otherwise the empirical
//! curve stands on its own and the report says the fit was not
//! reported, with the reason, the profile loss over tau, and the
//! outcome across a table of gate thresholds and interior margins.

use crate::metrics::{bar_status, ofi_bars, ofi_events, Quote};

/// Schema tag written into every OFI-response report. v2 adds the
/// post-bar series, the model-free profile with contrasts and
/// share suppression, segment-bootstrap sensitivity and
/// simultaneous bands, HAC intervals, innovation projections,
/// shock diagnostics, the jump-term fit with its parameter region
/// and gate-sensitivity table, the extended main-session ladder,
/// and bootstrap provenance, on top of the v1 fields (see the
/// report's `schema_note`).
pub const SPEED_SCHEMA: &str = "toxicflow.info-speed/v2";

/// Bar length for the response analysis. One second is the finest
/// bar the recorded sessions support: the main session averages
/// about nine quotes per second, so most one-second bars carry a
/// fresh quote, and the horizons of interest (seconds to minutes)
/// stay well resolved.
pub const SPEED_BAR_MS: i64 = 1_000;

/// Horizon ladder, in seconds: the contemporaneous bar, then a
/// geometric climb out to five minutes.
pub const SPEED_HORIZONS_S: [i64; 16] = [
    0, 1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 300,
];

/// Extended horizon ladder, in seconds, computed only for sessions
/// long enough to support it (the main session): the standard
/// ladder plus rungs out to thirty minutes, so a slow adjustment
/// component is tested against data that could in principle show
/// its plateau instead of being extrapolated from 300 s.
pub const SPEED_EXTENDED_HORIZONS_S: [i64; 24] = [
    0, 1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 300, 360, 450, 600, 750, 900, 1200,
    1500, 1800,
];

/// Minimum sessions length (observed bars) before the extended
/// ladder is attempted.
pub const EXTENDED_MIN_OBSERVED_BARS: usize = 5_000;

/// Block lengths, in seconds, whose bootstrap sensitivity is
/// reported for every session (feasibility permitting). The
/// headline length stays 300 s for comparability with the first
/// published report.
pub const SENSITIVITY_BLOCKS_S: [i64; 4] = [150, 300, 600, 900];

/// Minimum R^2 for the exponential fit to be reported as a fit at
/// all. Below this the empirical curve is published and tau is null.
pub const FIT_MIN_R_SQUARED: f64 = 0.85;

/// R^2 thresholds in the gate-sensitivity table.
pub const GATE_R_SQUARED_THRESHOLDS: [f64; 4] = [0.50, 0.70, 0.85, 0.95];

/// Interior-margin choices (grid steps from each edge of the tau
/// search grid) in the gate-sensitivity table.
pub const GATE_INTERIOR_MARGINS: [usize; 3] = [1, 12, 24];

/// Bootstrap resamples behind every reported headline interval.
/// The same count as the TF6 evaluation
/// (`predict::BOOTSTRAP_RESAMPLES`), so intervals across the repo
/// share one convention.
pub const SPEED_BOOTSTRAP_RESAMPLES: usize = 2000;

/// Resamples behind each block-length sensitivity entry and the
/// extended-ladder intervals: enough to compare interval widths
/// across designs, while the headline intervals keep the full
/// published count.
pub const SPEED_SENSITIVITY_RESAMPLES: usize = 500;

/// Seed for the response bootstrap. Fixed and published, like the
/// TF6 seeds, so a report is reproducible from the session alone.
pub const SPEED_BOOTSTRAP_SEED: u64 = 0x85EB_CA6B_27D4_EA27;

/// Lag set (bars) for the OFI-innovation regression: OFI is
/// regressed on its own previous 10 bars, within observed segments
/// only, and the residual is the innovation.
pub const INNOVATION_LAGS: usize = 10;

/// Knobs for one response analysis. `Default` is the published
/// configuration; tests shrink the bootstrap counts, nothing else.
#[derive(Debug, Clone)]
pub struct SpeedConfig {
    pub bar_ms: i64,
    pub horizons_s: Vec<i64>,
    pub bootstrap_resamples: usize,
    pub sensitivity_resamples: usize,
    pub extended_resamples: usize,
    pub seed: u64,
}

impl Default for SpeedConfig {
    fn default() -> Self {
        Self {
            bar_ms: SPEED_BAR_MS,
            horizons_s: SPEED_HORIZONS_S.to_vec(),
            bootstrap_resamples: SPEED_BOOTSTRAP_RESAMPLES,
            sensitivity_resamples: SPEED_SENSITIVITY_RESAMPLES,
            extended_resamples: SPEED_SENSITIVITY_RESAMPLES,
            seed: SPEED_BOOTSTRAP_SEED,
        }
    }
}

/// Per-bar inputs on the fixed bar grid: last observed mid, summed
/// OFI, and whether any quote was actually observed in the bar
/// (carried-forward bars keep the last mid and zero OFI but are
/// flagged unobserved, exactly as `metrics::bar_status` reports).
#[derive(Debug, Clone)]
pub struct BarSeries {
    pub mid: Vec<f64>,
    pub ofi: Vec<f64>,
    pub observed: Vec<bool>,
}

/// Build the bar series from recorded quotes on the `ofi_bars` grid.
pub fn build_bar_series(quotes: &[Quote], bar_ms: i64) -> BarSeries {
    let events = ofi_events(quotes);
    let bars = ofi_bars(quotes, &events, bar_ms);
    let status = bar_status(quotes, bar_ms);
    BarSeries {
        mid: bars.iter().map(|b| b.2).collect(),
        ofi: bars.iter().map(|b| b.1).collect(),
        observed: status.iter().map(|s| s.observed).collect(),
    }
}

/// Contiguous fully-observed segments of a bar series, as
/// half-open index ranges. Bootstrap blocks are drawn only inside
/// one of these, and every diagnostic that needs consecutive bars
/// works within them, so no computation ever joins bars across an
/// unobserved stretch.
fn segments_of(observed: &[bool]) -> Vec<(usize, usize)> {
    let mut segs = Vec::new();
    let mut start: Option<usize> = None;
    for (i, &o) in observed.iter().enumerate() {
        if o && start.is_none() {
            start = Some(i);
        }
        if !o {
            if let Some(s) = start.take() {
                segs.push((s, i));
            }
        }
    }
    if let Some(s) = start {
        segs.push((s, observed.len()));
    }
    segs
}

/// Prefix sums for O(1) window checks: unobserved bars, and
/// resample splice points (`break_before[i]` marks a splice
/// between bars i-1 and i of a resampled path).
struct WindowIndex {
    unobs: Vec<usize>,
    breaks: Vec<usize>,
}

impl WindowIndex {
    fn new(observed: &[bool], break_before: &[bool]) -> Self {
        let n = observed.len();
        let mut unobs = vec![0usize; n + 1];
        let mut breaks = vec![0usize; n + 1];
        for i in 0..n {
            unobs[i + 1] = unobs[i] + usize::from(!observed[i]);
            breaks[i + 1] = breaks[i] + usize::from(break_before.get(i).copied().unwrap_or(false));
        }
        Self { unobs, breaks }
    }

    /// Bars `from..=to` all observed and no splice strictly inside
    /// the window (a splice between `from` and `to` would join
    /// bars that were never adjacent in real time).
    fn clean(&self, from: usize, to: usize) -> bool {
        self.unobs[to + 1] - self.unobs[from] == 0
            && self.breaks[to + 1] - self.breaks[from + 1] == 0
    }
}

/// Local-projection curves for both timing conventions in one
/// pass. For origin bar `t` and horizon `h` bars:
///
/// - inclusive: return from `mid[t-1]` (pre-shock-bar close) to
///   `mid[t+h]`, window `t-1..=t+h` must be clean;
/// - post-bar: return from `mid[t]` (shock-bar close) to
///   `mid[t+h]`, window `t..=t+h` must be clean; at h = 0 the
///   return is identically zero, so the slope is exactly 0.
///
/// The regressor is `reg[t] / sd`; an origin also requires
/// `origin_ok[t]`. Returns (inclusive slopes, inclusive counts,
/// post slopes, post counts). A slope is NaN when fewer than 30
/// usable origins support it.
#[allow(clippy::too_many_arguments)]
fn lp_pair(
    mid: &[f64],
    reg: &[f64],
    observed: &[bool],
    break_before: &[bool],
    origin_ok: &[bool],
    horizons_bars: &[usize],
    sd: f64,
    want_post: bool,
) -> (Vec<f64>, Vec<usize>, Vec<f64>, Vec<usize>) {
    let n = mid.len();
    let idx = WindowIndex::new(observed, break_before);
    let mut incl = Vec::with_capacity(horizons_bars.len());
    let mut incl_n = Vec::with_capacity(horizons_bars.len());
    let mut post = Vec::with_capacity(horizons_bars.len());
    let mut post_n = Vec::with_capacity(horizons_bars.len());
    for &h in horizons_bars {
        let (mut acc_i, mut acc_p) = (Ols::default(), Ols::default());
        if sd > 0.0 {
            for t in 1..n {
                let end = t + h;
                if end >= n {
                    break;
                }
                if !origin_ok[t] || !reg[t].is_finite() {
                    continue;
                }
                let x = reg[t] / sd;
                if idx.clean(t - 1, end) {
                    let base = mid[t - 1];
                    if base > 0.0 && mid[end].is_finite() {
                        acc_i.add(x, (mid[end] / base - 1.0) * 1e4);
                    }
                }
                if want_post && idx.clean(t, end) {
                    let base = mid[t];
                    if base > 0.0 && mid[end].is_finite() {
                        acc_p.add(x, (mid[end] / base - 1.0) * 1e4);
                    }
                }
            }
        }
        incl.push(acc_i.slope());
        incl_n.push(acc_i.n);
        post.push(if want_post { acc_p.slope() } else { f64::NAN });
        post_n.push(acc_p.n);
    }
    (incl, incl_n, post, post_n)
}

/// Running OLS sums for one slope-with-intercept regression.
#[derive(Default)]
struct Ols {
    sx: f64,
    sr: f64,
    sxx: f64,
    sxr: f64,
    n: usize,
}

impl Ols {
    fn add(&mut self, x: f64, r: f64) {
        self.sx += x;
        self.sr += r;
        self.sxx += x * x;
        self.sxr += x * r;
        self.n += 1;
    }

    fn slope(&self) -> f64 {
        if self.n < 30 {
            return f64::NAN;
        }
        let nf = self.n as f64;
        let var = self.sxx - self.sx * self.sx / nf;
        if var > 0.0 {
            (self.sxr - self.sx * self.sr / nf) / var
        } else {
            f64::NAN
        }
    }
}

/// Inclusive-convention local-projection curve, kept as the
/// single-series entry point (the original estimator).
#[cfg(test)]
fn lp_curve(
    mid: &[f64],
    ofi: &[f64],
    observed: &[bool],
    horizons_bars: &[usize],
    ofi_sd: f64,
) -> (Vec<f64>, Vec<usize>) {
    let n = mid.len();
    let no_breaks = vec![false; n];
    let all_ok = vec![true; n];
    let (incl, counts, _, _) = lp_pair(
        mid,
        ofi,
        observed,
        &no_breaks,
        &all_ok,
        horizons_bars,
        ofi_sd,
        false,
    );
    (incl, counts)
}

/// Solve a dense linear system by Gaussian elimination with
/// partial pivoting. Returns None when the system is singular.
fn solve_linear(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let p = b.len();
    let mut m: Vec<Vec<f64>> = a
        .iter()
        .zip(b.iter())
        .map(|(row, bi)| {
            let mut r = row.clone();
            r.push(*bi);
            r
        })
        .collect();
    for col in 0..p {
        let mut piv = col;
        for r in col + 1..p {
            if m[r][col].abs() > m[piv][col].abs() {
                piv = r;
            }
        }
        if m[piv][col].abs() < 1e-12 {
            return None;
        }
        m.swap(col, piv);
        let d = m[col][col];
        let pivot_row = m[col].clone();
        for (r, row) in m.iter_mut().enumerate() {
            if r != col {
                let f = row[col] / d;
                if f != 0.0 {
                    for (c, cell) in row.iter_mut().enumerate().skip(col).take(p + 1 - col) {
                        *cell -= f * pivot_row[c];
                    }
                }
            }
        }
    }
    Some((0..p).map(|i| m[i][p] / m[i][i]).collect())
}

/// Multiple OLS through the normal equations: design rows are
/// supplied as (features, target) pairs with an intercept added.
/// Returns (coefficients with intercept first, R^2).
fn ols_multi(rows: &[(Vec<f64>, f64)]) -> Option<(Vec<f64>, f64)> {
    if rows.len() < 10 {
        return None;
    }
    let p = rows[0].0.len() + 1;
    let mut xtx = vec![vec![0.0; p]; p];
    let mut xty = vec![0.0; p];
    for (feats, y) in rows {
        let mut x = Vec::with_capacity(p);
        x.push(1.0);
        x.extend_from_slice(feats);
        for i in 0..p {
            xty[i] += x[i] * y;
            for j in 0..p {
                xtx[i][j] += x[i] * x[j];
            }
        }
    }
    let coef = solve_linear(&xtx, &xty)?;
    let mean = rows.iter().map(|r| r.1).sum::<f64>() / rows.len() as f64;
    let (mut sse, mut sst) = (0.0, 0.0);
    for (feats, y) in rows {
        let mut pred = coef[0];
        for (k, f) in feats.iter().enumerate() {
            pred += coef[k + 1] * f;
        }
        sse += (y - pred).powi(2);
        sst += (y - mean).powi(2);
    }
    let r2 = if sst > 0.0 { 1.0 - sse / sst } else { f64::NAN };
    Some((coef, r2))
}

/// OFI innovations: residuals from regressing each bar's OFI on
/// the session's OFI over the previous `INNOVATION_LAGS` bars,
/// using only origins whose whole lag window sits inside one
/// observed segment. Bars without a clean lag window carry NaN.
/// Returns (residuals, lag coefficients, intercept, residual SD).
fn ofi_innovations(series: &BarSeries) -> (Vec<f64>, Vec<f64>, f64, f64) {
    let n = series.ofi.len();
    let lags = INNOVATION_LAGS;
    let segs = segments_of(&series.observed);
    let mut seg_of = vec![usize::MAX; n];
    for (si, &(a, b)) in segs.iter().enumerate() {
        for slot in seg_of.iter_mut().take(b).skip(a) {
            *slot = si;
        }
    }
    let mut rows: Vec<(Vec<f64>, f64)> = Vec::new();
    let mut row_t: Vec<usize> = Vec::new();
    for t in 0..n {
        if seg_of[t] == usize::MAX || t < lags {
            continue;
        }
        let (a, _) = segs[seg_of[t]];
        if t - lags < a {
            continue;
        }
        let feats: Vec<f64> = (1..=lags).map(|l| series.ofi[t - l]).collect();
        if feats.iter().all(|v| v.is_finite()) && series.ofi[t].is_finite() {
            rows.push((feats, series.ofi[t]));
            row_t.push(t);
        }
    }
    let mut resid = vec![f64::NAN; n];
    let Some((coef, _)) = ols_multi(&rows) else {
        return (resid, vec![f64::NAN; lags], f64::NAN, f64::NAN);
    };
    for (k, t) in row_t.iter().enumerate() {
        let (feats, y) = &rows[k];
        let mut pred = coef[0];
        for (j, f) in feats.iter().enumerate() {
            pred += coef[j + 1] * f;
        }
        resid[*t] = y - pred;
    }
    let vals: Vec<f64> = resid.iter().copied().filter(|v| v.is_finite()).collect();
    let sd = population_sd(&vals);
    (resid, coef[1..].to_vec(), coef[0], sd)
}

/// Pearson correlation over paired samples.
fn correlation(pairs: &[(f64, f64)]) -> f64 {
    if pairs.len() < 3 {
        return f64::NAN;
    }
    let nf = pairs.len() as f64;
    let mx = pairs.iter().map(|p| p.0).sum::<f64>() / nf;
    let my = pairs.iter().map(|p| p.1).sum::<f64>() / nf;
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (x, y) in pairs {
        sxy += (x - mx) * (y - my);
        sxx += (x - mx).powi(2);
        syy += (y - my).powi(2);
    }
    if sxx > 0.0 && syy > 0.0 {
        sxy / (sxx * syy).sqrt()
    } else {
        f64::NAN
    }
}

/// OFI autocorrelation at lags 1..=30 bars, from pairs that lie
/// inside a single observed segment (a lag never reaches across
/// an unobserved stretch).
fn ofi_acf(series: &BarSeries) -> Vec<f64> {
    let segs = segments_of(&series.observed);
    (1..=30usize)
        .map(|lag| {
            let mut pairs = Vec::new();
            for &(a, b) in &segs {
                if b - a > lag {
                    for t in a..b - lag {
                        pairs.push((series.ofi[t], series.ofi[t + lag]));
                    }
                }
            }
            correlation(&pairs)
        })
        .collect()
}

/// Pre-shock diagnostics, stated observationally: does recent
/// price movement predict the shock bar's OFI, and is the price
/// move in the bar just before the shock associated with the
/// shock? Neither result is given a causal reading.
fn pre_shock_diagnostics(series: &BarSeries, ofi_sd: f64) -> PreShockDiagnostics {
    let n = series.mid.len();
    // ret[t]: return of bar t against bar t-1, inside a segment.
    let mut ret = vec![f64::NAN; n];
    for &(a, b) in &segments_of(&series.observed) {
        for (off, slot) in ret.iter_mut().enumerate().take(b).skip(a + 1) {
            if series.mid[off - 1] > 0.0 {
                *slot = (series.mid[off] / series.mid[off - 1] - 1.0) * 1e4;
            }
        }
    }
    let std_x = |t: usize| series.ofi[t] / ofi_sd;
    // Shock on its three lagged returns.
    let mut rows: Vec<(Vec<f64>, f64)> = Vec::new();
    for t in 0..n {
        if t >= 4 && series.observed[t] {
            let feats = vec![ret[t - 1], ret[t - 2], ret[t - 3]];
            if feats.iter().all(|v| v.is_finite()) {
                rows.push((feats, std_x(t)));
            }
        }
    }
    let (lag_coef, lag_r2, lag_n) = match ols_multi(&rows) {
        Some((coef, r2)) => (coef[1..].to_vec(), r2, rows.len()),
        None => (vec![f64::NAN; 3], f64::NAN, rows.len()),
    };
    // Pre-shock return (the bar before the shock bar) on the shock.
    let mut rows2: Vec<(Vec<f64>, f64)> = Vec::new();
    let mut pairs: Vec<(f64, f64)> = Vec::new();
    for t in 0..n {
        if t >= 2 && series.observed[t] && ret[t - 1].is_finite() {
            rows2.push((vec![std_x(t)], ret[t - 1]));
            pairs.push((std_x(t), ret[t - 1]));
        }
    }
    let slope = match ols_multi(&rows2) {
        Some((coef, _)) => coef[1],
        None => f64::NAN,
    };
    PreShockDiagnostics {
        lagged_return_coefficients_per_bps: lag_coef,
        lagged_return_r_squared: lag_r2,
        lagged_return_n: lag_n,
        preshock_return_slope_bps_per_sd: slope,
        preshock_return_correlation: correlation(&pairs),
        preshock_return_n: pairs.len(),
    }
}

/// Horizon-aware HAC (Newey-West) inference for the inclusive
/// curve: per horizon, the OLS residuals' score autocovariances
/// are summed with Bartlett weights out to a lag equal to the
/// horizon in bars (the span over which response windows overlap),
/// giving a standard error that does not treat overlapping origins
/// as independent. Returns (standard errors, normal 95% intervals).
fn hac_inference(
    mid: &[f64],
    reg: &[f64],
    observed: &[bool],
    horizons_bars: &[usize],
    sd: f64,
) -> (Vec<f64>, Vec<(f64, f64)>) {
    let n = mid.len();
    let no_breaks = vec![false; n];
    let idx = WindowIndex::new(observed, &no_breaks);
    let mut ses = Vec::with_capacity(horizons_bars.len());
    let mut cis = Vec::with_capacity(horizons_bars.len());
    for &h in horizons_bars {
        let mut pts: Vec<(usize, f64, f64)> = Vec::new(); // (t, x, r)
        if sd > 0.0 {
            for t in 1..n {
                let end = t + h;
                if end >= n {
                    break;
                }
                if !idx.clean(t - 1, end) || !reg[t].is_finite() {
                    continue;
                }
                let base = mid[t - 1];
                if base > 0.0 && mid[end].is_finite() {
                    pts.push((t, reg[t] / sd, (mid[end] / base - 1.0) * 1e4));
                }
            }
        }
        if pts.len() < 30 {
            ses.push(f64::NAN);
            cis.push((f64::NAN, f64::NAN));
            continue;
        }
        let nf = pts.len() as f64;
        let mx = pts.iter().map(|p| p.1).sum::<f64>() / nf;
        let mr = pts.iter().map(|p| p.2).sum::<f64>() / nf;
        let (mut sxx, mut sxr) = (0.0, 0.0);
        for (_, x, r) in &pts {
            sxx += (x - mx).powi(2);
            sxr += (x - mx) * (r - mr);
        }
        if sxx <= 0.0 {
            ses.push(f64::NAN);
            cis.push((f64::NAN, f64::NAN));
            continue;
        }
        let beta = sxr / sxx;
        let alpha = mr - beta * mx;
        let mut u = vec![f64::NAN; n];
        for (t, x, r) in &pts {
            u[*t] = (x - mx) * (r - alpha - beta * x);
        }
        let lag_max = h;
        let mut meat = 0.0;
        for lag in 0..=lag_max {
            let mut g = 0.0;
            if lag == 0 {
                for v in u.iter().filter(|v| v.is_finite()) {
                    g += v * v;
                }
                meat += g;
            } else {
                for t in lag..n {
                    if u[t].is_finite() && u[t - lag].is_finite() {
                        g += u[t] * u[t - lag];
                    }
                }
                meat += 2.0 * (1.0 - lag as f64 / (lag_max as f64 + 1.0)) * g;
            }
        }
        let var = (meat / (sxx * sxx)).max(0.0);
        let se = var.sqrt();
        ses.push(se);
        cis.push((beta - 1.96 * se, beta + 1.96 * se));
    }
    (ses, cis)
}

/// One adjustment fit, R(h) = J + R_inf * (1 - exp(-h/tau)). The
/// instantaneous term J is separately identified from the slow
/// component: the curve is not forced through zero at h = 0.
#[derive(Debug, Clone, Copy)]
pub struct ExpFit {
    pub jump_bps: f64,
    pub r_inf_bps: f64,
    pub tau_s: f64,
    /// Unweighted R^2 of the fitted curve against the response
    /// points (the scale the published gate reads), regardless of
    /// the weighting used to fit.
    pub r_squared: f64,
    /// The loss the fit minimized (weighted when a covariance
    /// weight was supplied, plain SSE otherwise).
    pub loss: f64,
}

/// Tau grid for the profile fit: log-spaced, in seconds.
fn tau_grid() -> Vec<f64> {
    let (lo, hi, steps) = (0.25f64, 1500.0f64, 240usize);
    (0..steps)
        .map(|i| lo * (hi / lo).powf(i as f64 / (steps - 1) as f64))
        .collect()
}

/// Invert a dense matrix (Gauss-Jordan with partial pivoting).
/// Returns None when a pivot is numerically zero.
fn invert_matrix(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let p = a.len();
    let mut m: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.extend((0..p).map(|j| if i == j { 1.0 } else { 0.0 }));
            r
        })
        .collect();
    for col in 0..p {
        let mut piv = col;
        for r in col + 1..p {
            if m[r][col].abs() > m[piv][col].abs() {
                piv = r;
            }
        }
        if m[piv][col].abs() < 1e-300 {
            return None;
        }
        m.swap(col, piv);
        let d = m[col][col];
        for cell in m[col].iter_mut() {
            *cell /= d;
        }
        let pivot_row = m[col].clone();
        for (r, row) in m.iter_mut().enumerate() {
            if r != col {
                let f = row[col];
                if f != 0.0 {
                    for (c, cell) in row.iter_mut().enumerate() {
                        *cell -= f * pivot_row[c];
                    }
                }
            }
        }
    }
    Some(m.iter().map(|r| r[p..].to_vec()).collect())
}

/// Covariance matrix of the response vector across bootstrap
/// draws, lightly ridge-regularized so its inverse is usable as a
/// fit weight even when horizons are near-collinear.
fn draws_covariance(draws: &[Vec<f64>], h_len: usize) -> Option<Vec<Vec<f64>>> {
    if draws.len() < 10 || h_len == 0 {
        return None;
    }
    let mut mean = vec![0.0; h_len];
    for d in draws {
        for (k, v) in d.iter().enumerate() {
            mean[k] += v;
        }
    }
    for m in mean.iter_mut() {
        *m /= draws.len() as f64;
    }
    let mut cov = vec![vec![0.0; h_len]; h_len];
    for d in draws {
        for (i, row) in cov.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell += (d[i] - mean[i]) * (d[j] - mean[j]);
            }
        }
    }
    let avg_diag = (0..h_len).map(|i| cov[i][i]).sum::<f64>() / h_len as f64 / draws.len() as f64;
    for (i, row) in cov.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell /= draws.len() as f64;
            if i != j {
                // Shrink the off-diagonal toward zero (halfway to
                // the diagonal): neighbouring horizons are so
                // strongly correlated that the raw covariance is
                // near-singular, and an unshrunk inverse makes the
                // weighted fit chase the correlation structure
                // instead of the curve.
                *cell *= 0.5;
            }
        }
        row[i] += (avg_diag * 1e-6).max(1e-12);
    }
    Some(cov)
}

/// Fit the jump-term adjustment path over the finite response
/// points, profiling tau over the grid and solving for (J, R_inf)
/// in closed form at each step. With `weight` (an inverse
/// covariance), the fit is covariance-aware; without it, plain
/// least squares. Returns the best fit found, without judging it;
/// `judge_fit` decides whether it may be reported.
fn fit_exponential_w(
    horizons_s: &[f64],
    response: &[f64],
    weight: Option<&[Vec<f64>]>,
) -> Option<ExpFit> {
    let keep: Vec<usize> = (0..response.len())
        .filter(|&i| response[i].is_finite())
        .collect();
    if keep.len() < 4 {
        return None;
    }
    let hs: Vec<f64> = keep.iter().map(|&i| horizons_s[i]).collect();
    let ys: Vec<f64> = keep.iter().map(|&i| response[i]).collect();
    let hp = hs.len();
    let w: Vec<Vec<f64>> = match weight {
        Some(full) => keep
            .iter()
            .map(|&i| keep.iter().map(|&j| full[i][j]).collect())
            .collect(),
        None => (0..hp)
            .map(|i| (0..hp).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect(),
    };
    let mean = ys.iter().sum::<f64>() / hp as f64;
    let sst: f64 = ys.iter().map(|y| (y - mean).powi(2)).sum();
    if sst <= 0.0 || sst.is_nan() {
        return None;
    }
    let mut best: Option<ExpFit> = None;
    for tau in tau_grid() {
        let f: Vec<f64> = hs.iter().map(|h| 1.0 - (-h / tau).exp()).collect();
        // Normal equations for [J, R_inf] under weight w.
        let (mut m11, mut m12, mut m22, mut v1, mut v2) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for i in 0..hp {
            let (mut wi1, mut wif, mut wiy) = (0.0, 0.0, 0.0);
            for j in 0..hp {
                wi1 += w[i][j];
                wif += w[i][j] * f[j];
                wiy += w[i][j] * ys[j];
            }
            m11 += wi1;
            m12 += wif;
            m22 += f[i] * wif;
            v1 += wiy;
            v2 += f[i] * wiy;
        }
        let det = m11 * m22 - m12 * m12;
        if det.abs() < 1e-14 * (m11 * m22).abs().max(1.0) {
            continue;
        }
        let jump = (v1 * m22 - v2 * m12) / det;
        let r_inf = (m11 * v2 - m12 * v1) / det;
        let resid: Vec<f64> = (0..hp).map(|i| ys[i] - jump - r_inf * f[i]).collect();
        let mut loss = 0.0;
        for i in 0..hp {
            let mut wr = 0.0;
            for j in 0..hp {
                wr += w[i][j] * resid[j];
            }
            loss += resid[i] * wr;
        }
        let sse: f64 = resid.iter().map(|r| r * r).sum();
        let fit = ExpFit {
            jump_bps: jump,
            r_inf_bps: r_inf,
            tau_s: tau,
            r_squared: 1.0 - sse / sst,
            loss,
        };
        if best.as_ref().is_none_or(|b| fit.loss < b.loss) {
            best = Some(fit);
        }
    }
    best
}

/// Unweighted jump-term fit (the within-bootstrap-draw fit, and
/// the fall-back when no covariance weight is available).
fn fit_exponential_raw(horizons_s: &[f64], response: &[f64]) -> Option<ExpFit> {
    fit_exponential_w(horizons_s, response, None)
}

/// Profile loss over the tau grid for a (possibly weighted) fit:
/// how flat or unbounded the loss surface is matters more than
/// the single minimizing tau. Returns (taus, losses) downsampled
/// by two, plus the tau range whose loss is within 10% of the
/// minimum (the flat region).
fn profile_loss(
    horizons_s: &[f64],
    response: &[f64],
    weight: Option<&[Vec<f64>]>,
) -> (Vec<f64>, Vec<f64>, Option<(f64, f64)>) {
    let grid = tau_grid();
    let mut losses = Vec::with_capacity(grid.len());
    for tau in &grid {
        // Refit (J, R_inf) at this fixed tau by scoring the best
        // fit restricted to it: reuse the profile routine by
        // fitting with a one-tau scan through a helper closure.
        losses.push(fixed_tau_loss(horizons_s, response, weight, *tau));
    }
    let finite: Vec<(f64, f64)> = grid
        .iter()
        .zip(losses.iter())
        .filter(|(_, l)| l.is_finite())
        .map(|(t, l)| (*t, *l))
        .collect();
    let flat = if finite.is_empty() {
        None
    } else {
        let min = finite.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
        let inside: Vec<f64> = finite
            .iter()
            .filter(|p| p.1 <= min * 1.10 + 1e-12)
            .map(|p| p.0)
            .collect();
        if inside.is_empty() {
            None
        } else {
            Some((*inside.first().unwrap(), *inside.last().unwrap()))
        }
    };
    let taus: Vec<f64> = grid.iter().step_by(2).copied().collect();
    let ls: Vec<f64> = losses.iter().step_by(2).copied().collect();
    (taus, ls, flat)
}

/// Weighted loss of the best (J, R_inf) at one fixed tau.
fn fixed_tau_loss(
    horizons_s: &[f64],
    response: &[f64],
    weight: Option<&[Vec<f64>]>,
    tau: f64,
) -> f64 {
    let keep: Vec<usize> = (0..response.len())
        .filter(|&i| response[i].is_finite())
        .collect();
    if keep.len() < 4 {
        return f64::NAN;
    }
    let hs: Vec<f64> = keep.iter().map(|&i| horizons_s[i]).collect();
    let ys: Vec<f64> = keep.iter().map(|&i| response[i]).collect();
    let hp = hs.len();
    let w: Vec<Vec<f64>> = match weight_filtered(weight, &keep) {
        Some(w) => w,
        None => (0..hp)
            .map(|i| (0..hp).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect(),
    };
    let f: Vec<f64> = hs.iter().map(|h| 1.0 - (-h / tau).exp()).collect();
    let (mut m11, mut m12, mut m22, mut v1, mut v2) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..hp {
        let (mut wi1, mut wif, mut wiy) = (0.0, 0.0, 0.0);
        for j in 0..hp {
            wi1 += w[i][j];
            wif += w[i][j] * f[j];
            wiy += w[i][j] * ys[j];
        }
        m11 += wi1;
        m12 += wif;
        m22 += f[i] * wif;
        v1 += wiy;
        v2 += f[i] * wiy;
    }
    let det = m11 * m22 - m12 * m12;
    if det.abs() < 1e-14 * (m11 * m22).abs().max(1.0) {
        return f64::NAN;
    }
    let jump = (v1 * m22 - v2 * m12) / det;
    let r_inf = (m11 * v2 - m12 * v1) / det;
    let resid: Vec<f64> = (0..hp).map(|i| ys[i] - jump - r_inf * f[i]).collect();
    let mut loss = 0.0;
    for i in 0..hp {
        let mut wr = 0.0;
        for j in 0..hp {
            wr += w[i][j] * resid[j];
        }
        loss += resid[i] * wr;
    }
    loss
}

fn weight_filtered(weight: Option<&[Vec<f64>]>, keep: &[usize]) -> Option<Vec<Vec<f64>>> {
    weight.map(|full| {
        keep.iter()
            .map(|&i| keep.iter().map(|&j| full[i][j]).collect())
            .collect()
    })
}

/// Judge a raw fit under explicit gates: it is a measurement only
/// when the long-run response is positive, the curve is actually
/// exponential-shaped at the R^2 bar, and tau is interior to the
/// search grid by at least `margin_steps` grid steps (a boundary
/// tau means the grid, not the data, chose the number).
fn judge_fit_with(
    horizons_s: &[f64],
    response: &[f64],
    fit: &ExpFit,
    r2_min: f64,
    margin_steps: usize,
) -> Result<(), String> {
    let mut last = f64::NAN;
    for (_, r) in horizons_s
        .iter()
        .zip(response.iter())
        .filter(|(_, r)| r.is_finite())
    {
        last = *r;
    }
    if last <= 0.0 || last.is_nan() {
        return Err(format!(
            "the cumulative response at the longest horizon is {last:+.3} bps, not positive, so no positive exponential adjustment path describes it"
        ));
    }
    if fit.r_inf_bps <= 0.0 || fit.r_inf_bps.is_nan() {
        return Err(format!(
            "the fitted long-run response is {:+.3} bps, not positive",
            fit.r_inf_bps
        ));
    }
    if fit.r_squared < r2_min {
        return Err(format!(
            "the exponential fit explains only R^2 = {:.3} of the response curve, below the {:.2} bar for reporting it as a fit",
            fit.r_squared, r2_min
        ));
    }
    let grid = tau_grid();
    let m = margin_steps.min(grid.len() / 2 - 1);
    let (lo, hi) = (grid[m], grid[grid.len() - 1 - m]);
    if fit.tau_s <= lo || fit.tau_s >= hi {
        return Err(format!(
            "the fitted tau of {:.2} s sits at the edge of the search grid, so the grid rather than the data picked it",
            fit.tau_s
        ));
    }
    Ok(())
}

/// Judge under the published gates (R^2 at `FIT_MIN_R_SQUARED`,
/// one-step interior margin).
fn judge_fit(horizons_s: &[f64], response: &[f64], fit: &ExpFit) -> Result<(), String> {
    judge_fit_with(horizons_s, response, fit, FIT_MIN_R_SQUARED, 1)
}

/// SplitMix64, the same deterministic PRNG the TF6 bootstrap uses
/// (kept local because `predict`'s copy is private to that module).
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

/// (2.5%, 97.5%) percentiles of a bootstrap distribution, indexed
/// the same way as `predict::block_bootstrap_mean_ci`.
fn percentile_ci(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = sorted.len();
    (
        sorted[((0.025 * m as f64) as usize).min(m - 1)],
        sorted[((0.975 * m as f64) as usize).min(m - 1)],
    )
}

fn percentile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted[((q * sorted.len() as f64) as usize).min(sorted.len() - 1)]
}

/// One segment-respecting resample, as source-bar indices plus
/// the splice positions in the resampled path. Every block of
/// consecutive source indices lies inside a single observed
/// segment and advances one bar at a time; a splice is marked
/// before the first bar of each block after the first, so window
/// checks can refuse any response window that would cross it.
/// The resampled length is the session's observed-bar count.
fn resample_indices(
    segments: &[(usize, usize)],
    block: usize,
    target: usize,
    rng: &mut SplitMix64,
) -> (Vec<usize>, Vec<bool>) {
    // All valid (segment, start) choices for a full block.
    let mut starts: Vec<usize> = Vec::new();
    for &(a, b) in segments {
        if b - a >= block {
            starts.extend(a..=b - block);
        }
    }
    if starts.is_empty() || target == 0 || block == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut out: Vec<usize> = Vec::with_capacity(target);
    let mut breaks = vec![false; target];
    while out.len() < target {
        let start = starts[rng.next_below(starts.len())];
        let pos = out.len();
        if pos > 0 {
            breaks[pos] = true;
        }
        for j in 0..block {
            if out.len() >= target {
                break;
            }
            out.push(start + j);
        }
    }
    (out, breaks)
}

/// Draws produced by one bootstrap run.
struct BootOut {
    incl: Vec<Vec<f64>>,
    post: Vec<Vec<f64>>,
    innov: Vec<Vec<f64>>,
    /// Every draw's own jump-term fit (unweighted), passing or not.
    fits_all: Vec<ExpFit>,
    /// The subset of draw fits that passed the published judging.
    fits_pass: Vec<ExpFit>,
}

/// Segment-respecting block bootstrap of the whole estimator.
/// Blocks are drawn only within fully-observed segments, splices
/// are marked so no resample window joins bars across a splice,
/// and the curve (both timing conventions), the innovation curve,
/// and the jump-term fit are all recomputed on each resample: the
/// fit is re-run inside every draw, which is what the reported
/// fit share and parameter region summarize.
#[allow(clippy::too_many_arguments)]
fn bootstrap_engine(
    series: &BarSeries,
    innov: Option<&[f64]>,
    horizons_bars: &[usize],
    horizons_s: &[f64],
    block_bars: usize,
    resamples: usize,
    seed: u64,
    want_post: bool,
    want_innov: bool,
) -> BootOut {
    let mut out = BootOut {
        incl: Vec::new(),
        post: Vec::new(),
        innov: Vec::new(),
        fits_all: Vec::new(),
        fits_pass: Vec::new(),
    };
    let segments = segments_of(&series.observed);
    let target: usize = series.observed.iter().filter(|o| **o).count();
    let longest = segments.iter().map(|(a, b)| b - a).max().unwrap_or(0);
    if target == 0 || longest == 0 || resamples == 0 {
        return out;
    }
    // Drawn blocks are extended by the longest horizon: a
    // response window at the longest horizon spans max_h + 1 bars
    // and must be able to lie wholly inside one drawn block,
    // because windows crossing a splice are censored. The nominal
    // block length is the number of bars a block contributes
    // beyond that span.
    let max_h = horizons_bars.iter().copied().max().unwrap_or(0);
    let block = (block_bars.max(1) + max_h).min(longest);
    if block <= max_h {
        return out;
    }
    let mut rng = SplitMix64(seed);
    for _ in 0..resamples {
        let (idxs, breaks) = resample_indices(&segments, block, target, &mut rng);
        if idxs.len() < 30 {
            continue;
        }
        let mid: Vec<f64> = idxs.iter().map(|&i| series.mid[i]).collect();
        let reg: Vec<f64> = idxs.iter().map(|&i| series.ofi[i]).collect();
        let obs = vec![true; idxs.len()];
        let all_ok = vec![true; idxs.len()];
        let sd = population_sd(&reg);
        if sd <= 0.0 || sd.is_nan() {
            continue;
        }
        let (incl, _, post, _) = lp_pair(
            &mid,
            &reg,
            &obs,
            &breaks,
            &all_ok,
            horizons_bars,
            sd,
            want_post,
        );
        if incl.iter().any(|v| !v.is_finite()) {
            continue;
        }
        out.incl.push(incl.clone());
        if want_post {
            out.post.push(post);
        }
        if let Some(fit) = fit_exponential_raw(horizons_s, &incl) {
            if judge_fit(horizons_s, &incl, &fit).is_ok() {
                out.fits_pass.push(fit);
            }
            out.fits_all.push(fit);
        }
        if want_innov {
            if let Some(innov_series) = innov {
                let reg2: Vec<f64> = idxs.iter().map(|&i| innov_series[i]).collect();
                let ok2: Vec<bool> = reg2.iter().map(|v| v.is_finite()).collect();
                let finite2: Vec<f64> = reg2.iter().copied().filter(|v| v.is_finite()).collect();
                let sd2 = population_sd(&finite2);
                if sd2 > 0.0 && sd2.is_finite() {
                    let (icurve, _, _, _) =
                        lp_pair(&mid, &reg2, &obs, &breaks, &ok2, horizons_bars, sd2, false);
                    if icurve.iter().all(|v| v.is_finite()) {
                        out.innov.push(icurve);
                    }
                }
            }
        }
    }
    out
}

fn population_sd(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let m = xs.iter().sum::<f64>() / xs.len() as f64;
    (xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64).sqrt()
}

/// Per-horizon percentile intervals from draws.
fn draws_ci(draws: &[Vec<f64>], h_len: usize) -> Vec<(f64, f64)> {
    (0..h_len)
        .map(|k| {
            let col: Vec<f64> = draws.iter().map(|c| c[k]).collect();
            percentile_ci(&col)
        })
        .collect()
}

/// Simultaneous (sup-statistic) confidence band across horizons:
/// each draw's largest standardized deviation from the point
/// curve sets the band's multiplier, so the band covers the whole
/// curve jointly at the stated level, unlike the pointwise
/// intervals beside it.
fn simultaneous_band(point: &[f64], draws: &[Vec<f64>]) -> Vec<(f64, f64)> {
    let h_len = point.len();
    if draws.is_empty() {
        return vec![(f64::NAN, f64::NAN); h_len];
    }
    let mut sds = vec![0.0; h_len];
    for k in 0..h_len {
        let col: Vec<f64> = draws.iter().map(|c| c[k]).collect();
        sds[k] = population_sd(&col);
    }
    let mut stats: Vec<f64> = Vec::with_capacity(draws.len());
    for d in draws {
        let mut mx = 0.0f64;
        for k in 0..h_len {
            if sds[k] > 1e-9 {
                mx = mx.max((d[k] - point[k]).abs() / sds[k]);
            }
        }
        stats.push(mx);
    }
    let q = percentile(&stats, 0.95);
    (0..h_len)
        .map(|k| (point[k] - q * sds[k], point[k] + q * sds[k]))
        .collect()
}

/// A prespecified horizon contrast with its within-draw
/// bootstrap interval: the contrast is recomputed inside every
/// resample from that resample's own curve, so its interval is a
/// joint statement about the two horizons, not a combination of
/// two pointwise intervals.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Contrast {
    pub name: String,
    pub convention: String,
    pub from_s: f64,
    pub to_s: f64,
    pub estimate_bps: f64,
    pub ci_bps: (f64, f64),
}

/// A response-share ratio, reported only when it is meaningful.
/// When the denominator's pointwise interval includes zero the
/// ratio can swing arbitrarily large; the share is then
/// suppressed and the reason is recorded instead of a number.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ShareRatio {
    pub numerator_s: f64,
    pub denominator_s: f64,
    pub convention: String,
    pub ratio: Option<f64>,
    pub ratio_ci: Option<(f64, f64)>,
    pub suppressed: bool,
    pub suppression_reason: Option<String>,
}

/// The model-free response profile: prespecified horizons and
/// contrasts with joint bootstrap inference, plus share ratios
/// under the suppression rule.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelFreeProfile {
    pub horizons_s: Vec<f64>,
    pub inclusive_bps: Vec<f64>,
    pub inclusive_ci_bps: Vec<(f64, f64)>,
    pub post_bps: Vec<f64>,
    pub post_ci_bps: Vec<(f64, f64)>,
    pub contrasts: Vec<Contrast>,
    pub shares: Vec<ShareRatio>,
}

/// Block-length sensitivity for one candidate block length.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BlockSensitivity {
    pub block_len_s: f64,
    pub feasible: bool,
    /// Why this length could not be run, when it could not.
    pub reason: Option<String>,
    pub n_resamples: usize,
    /// Pointwise interval at the 300 s horizon under this length.
    pub response_ci_300_bps: Option<(f64, f64)>,
    pub ci_width_300_bps: Option<f64>,
    /// Mean pointwise interval width across all horizons.
    pub mean_ci_width_bps: Option<f64>,
}

/// Pre-shock diagnostics (observational; see
/// `pre_shock_diagnostics`).
#[derive(Debug, Clone, serde::Serialize)]
pub struct PreShockDiagnostics {
    /// Coefficients of the shock (per-SD OFI) on the mid returns
    /// of the three bars before it, per bp of return.
    pub lagged_return_coefficients_per_bps: Vec<f64>,
    pub lagged_return_r_squared: f64,
    pub lagged_return_n: usize,
    /// Slope of the pre-shock bar's return on the shock, bps of
    /// pre-shock return per one-SD shock.
    pub preshock_return_slope_bps_per_sd: f64,
    pub preshock_return_correlation: f64,
    pub preshock_return_n: usize,
}

/// Shock diagnostics for one session.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ShockDiagnostics {
    pub ofi_acf_lags_s: Vec<i64>,
    pub ofi_acf: Vec<f64>,
    pub pre_shock: PreShockDiagnostics,
}

/// The OFI-innovation projections: the same local-projection
/// protocol with the shock replaced by the innovation (the part
/// of OFI not predicted by its own past 10 bars). This is a
/// different estimand from the raw-OFI response and is reported
/// alongside it, never merged with it.
#[derive(Debug, Clone, serde::Serialize)]
pub struct InnovationAnalysis {
    pub lag_set_s: Vec<i64>,
    /// OFI-on-its-own-lags regression: intercept, then one
    /// coefficient per lag in `lag_set_s`.
    pub regression_intercept: f64,
    pub regression_coefficients: Vec<f64>,
    pub residual_sd: f64,
    pub response_bps: Vec<f64>,
    pub response_ci_bps: Vec<(f64, f64)>,
    pub simultaneous_band_bps: Vec<(f64, f64)>,
    pub n_origins: Vec<usize>,
}

/// Joint parameter region of the jump-term fit, summarized from
/// the within-draw fits (all of them, not only the passing ones).
/// A tau is promoted only when this region is bounded: its tau
/// interval sits strictly inside the search grid and spans less
/// than a fourfold ratio.
#[derive(Debug, Clone, serde::Serialize)]
pub struct FitRegion {
    pub tau_ci_s: (f64, f64),
    pub jump_ci_bps: (f64, f64),
    pub r_inf_ci_bps: (f64, f64),
    pub bounded: bool,
    pub n_draw_fits: usize,
}

/// One cell of the gate-sensitivity table: whether the headline
/// fit would pass under this R^2 threshold and interior margin.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GateSensitivity {
    pub r_squared_threshold: f64,
    pub interior_margin_steps: usize,
    pub interior_lo_s: f64,
    pub interior_hi_s: f64,
    pub passes: bool,
}

/// The extended-ladder result for a long session (the main
/// session on current data): the standard ladder plus rungs out
/// to 1800 s, with its own segment-bootstrap intervals and fit
/// diagnostics, so slow-component claims are checked against
/// horizons that could show a plateau.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ExtendedResponse {
    pub horizons_s: Vec<f64>,
    pub response_bps: Vec<f64>,
    pub response_ci_bps: Vec<(f64, f64)>,
    pub post_response_bps: Vec<f64>,
    pub n_origins: Vec<usize>,
    pub block_len_s: f64,
    pub n_resamples: usize,
    /// Set only when the extended fit itself passes the published
    /// judging; otherwise null and `fit_verdict` says why.
    pub fit_tau_s: Option<f64>,
    pub fit_jump_bps: Option<f64>,
    pub fit_r_inf_bps: Option<f64>,
    pub fit_r_squared: Option<f64>,
    pub fit_verdict: String,
    pub profile_flat_tau_range_s: Option<(f64, f64)>,
}

/// Build the model-free profile from point curves, pointwise
/// intervals, and the bootstrap draws of both conventions.
#[allow(clippy::too_many_arguments)]
fn build_profile(
    horizons_s: &[f64],
    incl: &[f64],
    incl_ci: &[(f64, f64)],
    incl_draws: &[Vec<f64>],
    post: &[f64],
    post_ci: &[(f64, f64)],
    post_draws: &[Vec<f64>],
) -> ModelFreeProfile {
    let at = |h: f64| {
        horizons_s
            .iter()
            .position(|v| (*v - h).abs() < 1e-9)
            .expect("profile horizon is on the ladder")
    };
    let prof_h = [5.0, 30.0, 60.0, 300.0];
    let pick = |curve: &[f64]| prof_h.iter().map(|h| curve[at(*h)]).collect::<Vec<_>>();
    let pick_ci = |ci: &[(f64, f64)]| prof_h.iter().map(|h| ci[at(*h)]).collect::<Vec<_>>();
    let contrast =
        |name: &str, conv: &str, from: f64, to: f64, curve: &[f64], draws: &[Vec<f64>]| {
            let (a, b) = (at(from), at(to));
            let diffs: Vec<f64> = draws.iter().map(|d| d[b] - d[a]).collect();
            Contrast {
                name: name.to_string(),
                convention: conv.to_string(),
                from_s: from,
                to_s: to,
                estimate_bps: curve[b] - curve[a],
                ci_bps: percentile_ci(&diffs),
            }
        };
    let mut contrasts = vec![
        contrast("R(60)-R(0)", "inclusive", 0.0, 60.0, incl, incl_draws),
        contrast("R(300)-R(60)", "inclusive", 60.0, 300.0, incl, incl_draws),
        contrast("R(60)-R(0)", "post_bar", 0.0, 60.0, post, post_draws),
        contrast("R(300)-R(60)", "post_bar", 60.0, 300.0, post, post_draws),
    ];
    contrasts.sort_by(|a, b| a.name.cmp(&b.name).then(a.convention.cmp(&b.convention)));
    let share = |num: f64, conv: &str, curve: &[f64], ci: &[(f64, f64)], draws: &[Vec<f64>]| {
        let (a, b) = (at(num), at(300.0));
        let (lo, hi) = ci[b];
        if lo <= 0.0 && hi >= 0.0 {
            ShareRatio {
                numerator_s: num,
                denominator_s: 300.0,
                convention: conv.to_string(),
                ratio: None,
                ratio_ci: None,
                suppressed: true,
                suppression_reason: Some(format!(
                    "suppressed: the denominator R(300) interval [{lo:+.3}, {hi:+.3}] bps includes zero, so the share is not identified"
                )),
            }
        } else {
            let ratios: Vec<f64> = draws
                .iter()
                .filter(|d| d[b].abs() > 1e-9)
                .map(|d| d[a] / d[b])
                .collect();
            ShareRatio {
                numerator_s: num,
                denominator_s: 300.0,
                convention: conv.to_string(),
                ratio: Some(curve[a] / curve[b]),
                ratio_ci: Some(percentile_ci(&ratios)),
                suppressed: false,
                suppression_reason: None,
            }
        }
    };
    let mut shares = Vec::new();
    for num in [0.0, 5.0, 30.0, 60.0] {
        shares.push(share(num, "inclusive", incl, incl_ci, incl_draws));
    }
    shares.push(share(60.0, "post_bar", post, post_ci, post_draws));
    ModelFreeProfile {
        horizons_s: prof_h.to_vec(),
        inclusive_bps: pick(incl),
        inclusive_ci_bps: pick_ci(incl_ci),
        post_bps: pick(post),
        post_ci_bps: pick_ci(post_ci),
        contrasts,
        shares,
    }
}

/// One session's OFI-response result. `tau_s` and friends are null
/// whenever the adjustment fit was not reported; the empirical
/// curves are always reported.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionSpeed {
    pub session: String,
    pub bar_ms: i64,
    pub n_bars: usize,
    pub n_observed_bars: usize,
    pub span_s: f64,
    /// Session standard deviation of per-bar OFI: the shock size.
    pub ofi_sd: f64,
    pub horizons_s: Vec<f64>,
    /// Cumulative mid response (inclusive convention), basis
    /// points per one-SD OFI shock.
    pub response_bps: Vec<f64>,
    /// Segment-bootstrap pointwise interval per horizon.
    pub response_ci_bps: Vec<(f64, f64)>,
    /// Usable origins per horizon after gap censoring.
    pub n_origins: Vec<usize>,
    pub fit_ok: bool,
    /// Why the adjustment fit was not reported, when it was not.
    pub fit_failure: Option<String>,
    pub r_inf_bps: Option<f64>,
    pub r_inf_ci_bps: Option<(f64, f64)>,
    pub tau_s: Option<f64>,
    pub tau_ci_s: Option<(f64, f64)>,
    /// tau * ln 2: time for half the long-run adjustment to complete.
    pub half_life_s: Option<f64>,
    pub half_life_ci_s: Option<(f64, f64)>,
    pub fit_r_squared: Option<f64>,
    /// Share of bootstrap resamples whose own fit passed the same
    /// judging bar as the headline fit.
    pub bootstrap_fit_share: f64,
    // ---- v2 additions ----
    /// Strictly post-bar response (shock-bar close baseline; the
    /// 0 s rung is identically zero by construction).
    pub post_response_bps: Vec<f64>,
    pub post_response_ci_bps: Vec<(f64, f64)>,
    pub post_n_origins: Vec<usize>,
    /// Simultaneous sup-statistic bands across horizons.
    pub response_simultaneous_band_bps: Vec<(f64, f64)>,
    pub post_response_simultaneous_band_bps: Vec<(f64, f64)>,
    /// Horizon-aware HAC (Newey-West) standard errors and normal
    /// intervals for the inclusive curve, next to the bootstrap.
    pub hac_se_bps: Vec<f64>,
    pub hac_ci_bps: Vec<(f64, f64)>,
    pub profile: ModelFreeProfile,
    pub shock_diagnostics: ShockDiagnostics,
    pub innovation: InnovationAnalysis,
    pub block_sensitivity: Vec<BlockSensitivity>,
    /// Effective sample at the longest horizon: the number of
    /// non-overlapping max-horizon blocks the session's observed
    /// segments contain (summed over segments). Bar counts alone
    /// overstate the sample, because origins overlap.
    pub effective_blocks_at_max_horizon: usize,
    pub n_segments: usize,
    pub longest_segment_bars: usize,
    /// Fitted instantaneous term J of the headline fit, when the
    /// fit is reported.
    pub jump_bps: Option<f64>,
    pub jump_ci_bps: Option<(f64, f64)>,
    /// Raw (unjudged) headline fit values, reported even when the
    /// fit is not promoted, so the failure is quantified.
    pub fit_jump_raw_bps: Option<f64>,
    pub fit_r_inf_raw_bps: Option<f64>,
    pub fit_tau_raw_s: Option<f64>,
    pub fit_region: Option<FitRegion>,
    /// Profile loss over tau for the covariance-weighted headline
    /// fit: flat or unbounded regions are published, not hidden
    /// behind the minimizing tau.
    pub profile_loss_tau_s: Vec<f64>,
    pub profile_loss_values: Vec<f64>,
    pub profile_flat_tau_range_s: Option<(f64, f64)>,
    pub gate_sensitivity: Vec<GateSensitivity>,
    /// Extended ladder (main-length sessions only).
    pub extended: Option<ExtendedResponse>,
}

/// Analyze one session's bar series. Returns None when the
/// session is too short to support the longest horizon with a
/// usable number of origins, or carries no OFI variation.
fn analyze_series(
    session: &str,
    series: &BarSeries,
    span_s: f64,
    cfg: &SpeedConfig,
) -> Option<SessionSpeed> {
    let n = series.mid.len();
    if n == 0 || cfg.bar_ms <= 0 {
        return None;
    }
    let horizons_bars: Vec<usize> = cfg
        .horizons_s
        .iter()
        .map(|s| (s * 1000 / cfg.bar_ms).max(0) as usize)
        .collect();
    let max_h = *horizons_bars.iter().max().unwrap_or(&0);
    if n < max_h + 100 {
        return None;
    }
    let segments = segments_of(&series.observed);
    let n_observed = series.observed.iter().filter(|o| **o).count();
    let longest_seg = segments.iter().map(|(a, b)| b - a).max().unwrap_or(0);
    let obs_ofi: Vec<f64> = series
        .ofi
        .iter()
        .zip(series.observed.iter())
        .filter(|(_, o)| **o)
        .map(|(v, _)| *v)
        .collect();
    let ofi_sd = population_sd(&obs_ofi);
    if ofi_sd <= 0.0 || ofi_sd.is_nan() {
        return None;
    }
    let horizons_s: Vec<f64> = cfg.horizons_s.iter().map(|s| *s as f64).collect();
    let no_breaks = vec![false; n];
    let all_ok = vec![true; n];
    let (curve, counts, post_curve, post_counts) = lp_pair(
        &series.mid,
        &series.ofi,
        &series.observed,
        &no_breaks,
        &all_ok,
        &horizons_bars,
        ofi_sd,
        true,
    );
    if curve.iter().any(|v| !v.is_finite()) || *counts.last().unwrap_or(&0) < 30 {
        return None;
    }
    // Innovations, computed once on the recorded series and
    // carried through the bootstrap as a per-bar regressor.
    let (innov, innov_coef, innov_intercept, innov_sd) = ofi_innovations(series);
    let innov_ok: Vec<bool> = innov.iter().map(|v| v.is_finite()).collect();
    let (innov_curve, innov_counts, _, _) = if innov_sd > 0.0 && innov_sd.is_finite() {
        lp_pair(
            &series.mid,
            &innov,
            &series.observed,
            &no_breaks,
            &innov_ok,
            &horizons_bars,
            innov_sd,
            false,
        )
    } else {
        (
            vec![f64::NAN; horizons_s.len()],
            vec![0; horizons_s.len()],
            Vec::new(),
            Vec::new(),
        )
    };
    // Headline bootstrap: segment-respecting, block = longest
    // horizon (or the longest segment, if shorter).
    let headline_block = max_h.max(1).min(longest_seg.max(1));
    let boot = bootstrap_engine(
        series,
        Some(&innov),
        &horizons_bars,
        &horizons_s,
        headline_block,
        cfg.bootstrap_resamples,
        cfg.seed,
        true,
        true,
    );
    let h_len = horizons_s.len();
    let response_ci_bps = draws_ci(&boot.incl, h_len);
    let post_response_ci_bps = draws_ci(&boot.post, h_len);
    let response_band = simultaneous_band(&curve, &boot.incl);
    let post_band = simultaneous_band(&post_curve, &boot.post);
    let innov_ci = draws_ci(&boot.innov, h_len);
    let innov_band = simultaneous_band(&innov_curve, &boot.innov);
    let n_draws = boot.incl.len().max(1);
    let bootstrap_fit_share = boot.fits_pass.len() as f64 / n_draws as f64;
    let profile = build_profile(
        &horizons_s,
        &curve,
        &response_ci_bps,
        &boot.incl,
        &post_curve,
        &post_response_ci_bps,
        &boot.post,
    );
    let (hac_se, hac_ci) = hac_inference(
        &series.mid,
        &series.ofi,
        &series.observed,
        &horizons_bars,
        ofi_sd,
    );
    // Headline fit: covariance-aware, weighted by the bootstrap
    // covariance of the response vector (diagonal fallback through
    // the inverse when the covariance cannot be formed).
    let cov = draws_covariance(&boot.incl, h_len);
    let weight = cov.as_ref().and_then(|c| invert_matrix(c));
    let raw_fit = fit_exponential_w(&horizons_s, &curve, weight.as_deref())
        .or_else(|| fit_exponential_raw(&horizons_s, &curve));
    let (pl_taus, pl_vals, pl_flat) = profile_loss(&horizons_s, &curve, weight.as_deref());
    // Joint parameter region from the within-draw fits.
    let fit_region = if boot.fits_all.len() >= 10 {
        let tau_ci = percentile_ci(&boot.fits_all.iter().map(|f| f.tau_s).collect::<Vec<_>>());
        let grid = tau_grid();
        let bounded = tau_ci.0 > grid[1]
            && tau_ci.1 < grid[grid.len() - 2]
            && tau_ci.0 > 0.0
            && tau_ci.1 / tau_ci.0 < 4.0;
        Some(FitRegion {
            tau_ci_s: tau_ci,
            jump_ci_bps: percentile_ci(
                &boot.fits_all.iter().map(|f| f.jump_bps).collect::<Vec<_>>(),
            ),
            r_inf_ci_bps: percentile_ci(
                &boot
                    .fits_all
                    .iter()
                    .map(|f| f.r_inf_bps)
                    .collect::<Vec<_>>(),
            ),
            bounded,
            n_draw_fits: boot.fits_all.len(),
        })
    } else {
        None
    };
    let (fit, fit_failure) = match raw_fit {
        Some(f) => match judge_fit(&horizons_s, &curve, &f) {
            Ok(()) => {
                if fit_region.as_ref().map(|r| r.bounded).unwrap_or(false) {
                    (Some(f), None)
                } else {
                    (
                        None,
                        Some(
                            "the fit passes the point-estimate gates but the joint parameter region from the bootstrap fits is not bounded, so no tau is reported"
                                .to_string(),
                        ),
                    )
                }
            }
            Err(why) => (None, Some(why)),
        },
        None => (
            None,
            Some("the response curve has too few finite points to fit".to_string()),
        ),
    };
    let mut gate_sensitivity: Vec<GateSensitivity> = Vec::new();
    {
        let grid = tau_grid();
        for thr in GATE_R_SQUARED_THRESHOLDS {
            for m in GATE_INTERIOR_MARGINS {
                let mm = m.min(grid.len() / 2 - 1);
                gate_sensitivity.push(GateSensitivity {
                    r_squared_threshold: thr,
                    interior_margin_steps: m,
                    interior_lo_s: grid[mm],
                    interior_hi_s: grid[grid.len() - 1 - mm],
                    passes: raw_fit
                        .map(|f| judge_fit_with(&horizons_s, &curve, &f, thr, m).is_ok())
                        .unwrap_or(false),
                });
            }
        }
    }
    let (tau_ci, half_ci, rinf_ci, jump_ci) = if boot.fits_pass.len() >= 100 {
        (
            Some(percentile_ci(
                &boot.fits_pass.iter().map(|f| f.tau_s).collect::<Vec<_>>(),
            )),
            Some(percentile_ci(
                &boot
                    .fits_pass
                    .iter()
                    .map(|f| f.tau_s * std::f64::consts::LN_2)
                    .collect::<Vec<_>>(),
            )),
            Some(percentile_ci(
                &boot
                    .fits_pass
                    .iter()
                    .map(|f| f.r_inf_bps)
                    .collect::<Vec<_>>(),
            )),
            Some(percentile_ci(
                &boot
                    .fits_pass
                    .iter()
                    .map(|f| f.jump_bps)
                    .collect::<Vec<_>>(),
            )),
        )
    } else {
        (None, None, None, None)
    };
    // Block-length sensitivity: the 300 s entry reuses the
    // headline draws; the other lengths run their own
    // segment-respecting bootstraps and are all reported.
    let mut sensitivity = Vec::new();
    for len_s in SENSITIVITY_BLOCKS_S {
        let len_bars = (len_s * 1000 / cfg.bar_ms).max(1) as usize;
        // A drawn block is the nominal length plus the longest
        // horizon (so max-horizon windows fit inside one block);
        // feasibility needs a segment at least that long and at
        // least four nominal blocks per resample.
        let feasible = longest_seg >= len_bars + max_h && n_observed / len_bars >= 4;
        if len_s == 300 {
            let ci300 = response_ci_bps.last().copied();
            sensitivity.push(BlockSensitivity {
                block_len_s: len_s as f64,
                feasible,
                reason: if feasible {
                    None
                } else {
                    Some(
                        "headline length retained for comparability; this session's segments support it only in reduced form"
                            .to_string(),
                    )
                },
                n_resamples: boot.incl.len(),
                response_ci_300_bps: ci300,
                ci_width_300_bps: ci300.map(|(lo, hi)| hi - lo),
                mean_ci_width_bps: Some(
                    response_ci_bps.iter().map(|(lo, hi)| hi - lo).sum::<f64>() / h_len as f64,
                ),
            });
        } else if !feasible {
            sensitivity.push(BlockSensitivity {
                block_len_s: len_s as f64,
                feasible: false,
                reason: Some(format!(
                    "not feasible: the longest fully-observed segment is {longest_seg} bars and the session holds {} observed bars, so a {len_s} s block (drawn at {len_s} s plus the longest horizon) either exceeds every segment or leaves fewer than four blocks per resample",
                    n_observed
                )),
                n_resamples: 0,
                response_ci_300_bps: None,
                ci_width_300_bps: None,
                mean_ci_width_bps: None,
            });
        } else {
            let alt = bootstrap_engine(
                series,
                None,
                &horizons_bars,
                &horizons_s,
                len_bars,
                cfg.sensitivity_resamples,
                cfg.seed
                    .wrapping_add((len_s as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                false,
                false,
            );
            let alt_ci = draws_ci(&alt.incl, h_len);
            let ci300 = alt_ci.last().copied();
            sensitivity.push(BlockSensitivity {
                block_len_s: len_s as f64,
                feasible: true,
                reason: None,
                n_resamples: alt.incl.len(),
                response_ci_300_bps: ci300,
                ci_width_300_bps: ci300.map(|(lo, hi)| hi - lo),
                mean_ci_width_bps: Some(
                    alt_ci.iter().map(|(lo, hi)| hi - lo).sum::<f64>() / h_len as f64,
                ),
            });
        }
    }
    let effective_blocks = segments.iter().map(|(a, b)| (b - a) / max_h.max(1)).sum();
    let extended = if n_observed >= EXTENDED_MIN_OBSERVED_BARS {
        extended_analysis(series, cfg)
    } else {
        None
    };
    Some(SessionSpeed {
        session: session.to_string(),
        bar_ms: cfg.bar_ms,
        n_bars: n,
        n_observed_bars: n_observed,
        span_s,
        ofi_sd,
        horizons_s: horizons_s.clone(),
        response_bps: curve,
        response_ci_bps,
        n_origins: counts,
        fit_ok: fit.is_some(),
        fit_failure,
        r_inf_bps: fit.map(|f| f.r_inf_bps),
        r_inf_ci_bps: rinf_ci,
        tau_s: fit.map(|f| f.tau_s),
        tau_ci_s: tau_ci,
        half_life_s: fit.map(|f| f.tau_s * std::f64::consts::LN_2),
        half_life_ci_s: half_ci,
        fit_r_squared: raw_fit.map(|f| f.r_squared),
        bootstrap_fit_share,
        post_response_bps: post_curve,
        post_response_ci_bps,
        post_n_origins: post_counts,
        response_simultaneous_band_bps: response_band,
        post_response_simultaneous_band_bps: post_band,
        hac_se_bps: hac_se,
        hac_ci_bps: hac_ci,
        profile,
        shock_diagnostics: ShockDiagnostics {
            ofi_acf_lags_s: (1..=30).collect(),
            ofi_acf: ofi_acf(series),
            pre_shock: pre_shock_diagnostics(series, ofi_sd),
        },
        innovation: InnovationAnalysis {
            lag_set_s: (1..=(INNOVATION_LAGS as i64)).collect(),
            regression_intercept: innov_intercept,
            regression_coefficients: innov_coef,
            residual_sd: innov_sd,
            response_bps: innov_curve,
            response_ci_bps: innov_ci,
            simultaneous_band_bps: innov_band,
            n_origins: innov_counts,
        },
        block_sensitivity: sensitivity,
        effective_blocks_at_max_horizon: effective_blocks,
        n_segments: segments.len(),
        longest_segment_bars: longest_seg,
        jump_bps: fit.map(|f| f.jump_bps),
        jump_ci_bps: jump_ci,
        fit_jump_raw_bps: raw_fit.map(|f| f.jump_bps),
        fit_r_inf_raw_bps: raw_fit.map(|f| f.r_inf_bps),
        fit_tau_raw_s: raw_fit.map(|f| f.tau_s),
        fit_region,
        profile_loss_tau_s: pl_taus,
        profile_loss_values: pl_vals,
        profile_flat_tau_range_s: pl_flat,
        gate_sensitivity,
        extended,
    })
}

/// Extended ladder for a long session: point curves and
/// segment-bootstrap intervals out to 1800 s, plus the jump-term
/// fit diagnostics on that ladder.
fn extended_analysis(series: &BarSeries, cfg: &SpeedConfig) -> Option<ExtendedResponse> {
    let n = series.mid.len();
    let horizons_bars: Vec<usize> = SPEED_EXTENDED_HORIZONS_S
        .iter()
        .map(|s| (s * 1000 / cfg.bar_ms).max(0) as usize)
        .collect();
    let horizons_s: Vec<f64> = SPEED_EXTENDED_HORIZONS_S
        .iter()
        .map(|s| *s as f64)
        .collect();
    let obs_ofi: Vec<f64> = series
        .ofi
        .iter()
        .zip(series.observed.iter())
        .filter(|(_, o)| **o)
        .map(|(v, _)| *v)
        .collect();
    let ofi_sd = population_sd(&obs_ofi);
    if ofi_sd <= 0.0 || !ofi_sd.is_finite() {
        return None;
    }
    let no_breaks = vec![false; n];
    let all_ok = vec![true; n];
    let (curve, counts, post_curve, _) = lp_pair(
        &series.mid,
        &series.ofi,
        &series.observed,
        &no_breaks,
        &all_ok,
        &horizons_bars,
        ofi_sd,
        true,
    );
    // Trim trailing horizons whose usable origins fall below 30.
    let mut keep = horizons_s.len();
    while keep > 0 && (counts[keep - 1] < 30 || !curve[keep - 1].is_finite()) {
        keep -= 1;
    }
    if keep == 0 || horizons_s[keep - 1] < 900.0 {
        return None;
    }
    let boot = bootstrap_engine(
        series,
        None,
        &horizons_bars[..keep],
        &horizons_s[..keep],
        300,
        cfg.extended_resamples,
        cfg.seed ^ 0xE57E_D1A6_5EED_0001,
        true,
        false,
    );
    let ci = draws_ci(&boot.incl, keep);
    let cov = draws_covariance(&boot.incl, keep);
    let weight = cov.as_ref().and_then(|c| invert_matrix(c));
    let fit = fit_exponential_w(&horizons_s[..keep], &curve[..keep], weight.as_deref())
        .or_else(|| fit_exponential_raw(&horizons_s[..keep], &curve[..keep]));
    let (_, _, flat) = profile_loss(&horizons_s[..keep], &curve[..keep], weight.as_deref());
    let (fit_tau, fit_jump, fit_rinf, fit_r2, verdict) = match fit {
        Some(f) => match judge_fit(&horizons_s[..keep], &curve[..keep], &f) {
            Ok(()) => (
                Some(f.tau_s),
                Some(f.jump_bps),
                Some(f.r_inf_bps),
                Some(f.r_squared),
                "the extended fit passes the published judging on the long ladder".to_string(),
            ),
            Err(why) => (None, None, None, Some(f.r_squared), why),
        },
        None => (
            None,
            None,
            None,
            None,
            "the extended curve has too few finite points to fit".to_string(),
        ),
    };
    Some(ExtendedResponse {
        horizons_s: horizons_s[..keep].to_vec(),
        response_bps: curve[..keep].to_vec(),
        response_ci_bps: ci,
        post_response_bps: post_curve[..keep].to_vec(),
        n_origins: counts[..keep].to_vec(),
        block_len_s: 300.0,
        n_resamples: boot.incl.len(),
        fit_tau_s: fit_tau,
        fit_jump_bps: fit_jump,
        fit_r_inf_bps: fit_rinf,
        fit_r_squared: fit_r2,
        fit_verdict: verdict,
        profile_flat_tau_range_s: flat,
    })
}

/// Analyze one session's quotes. Returns None when the session is
/// too short to support the longest horizon with a usable number
/// of origins, or carries no OFI variation.
pub fn analyze_session_speed(
    session: &str,
    quotes: &[Quote],
    cfg: &SpeedConfig,
) -> Option<SessionSpeed> {
    if quotes.is_empty() {
        return None;
    }
    let series = build_bar_series(quotes, cfg.bar_ms);
    let span_s = (quotes.last().unwrap().time_ms - quotes.first().unwrap().time_ms) as f64 / 1000.0;
    analyze_series(session, &series, span_s, cfg)
}

/// Cross-session summary over the sessions whose fit passed. The
/// independent recording session is the sample unit, as in the TF6
/// pilot: these are plain descriptive statistics over sessions,
/// never a pooled bar-level significance claim.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SpeedSummary {
    pub n_sessions: usize,
    pub n_fit_ok: usize,
    pub tau_mean_s: f64,
    pub tau_median_s: f64,
    pub tau_min_s: f64,
    pub tau_max_s: f64,
    pub tau_std_s: f64,
    pub half_life_mean_s: f64,
    pub half_life_median_s: f64,
    pub half_life_min_s: f64,
    pub half_life_max_s: f64,
    pub half_life_std_s: f64,
}

fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = s.len();
    if m % 2 == 1 {
        s[m / 2]
    } else {
        0.5 * (s[m / 2 - 1] + s[m / 2])
    }
}

fn summarize(sessions: &[SessionSpeed]) -> SpeedSummary {
    let taus: Vec<f64> = sessions.iter().filter_map(|s| s.tau_s).collect();
    let halfs: Vec<f64> = sessions.iter().filter_map(|s| s.half_life_s).collect();
    let stats = |xs: &[f64]| -> (f64, f64, f64, f64, f64) {
        if xs.is_empty() {
            return (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN);
        }
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        (
            mean,
            median(xs),
            xs.iter().cloned().fold(f64::INFINITY, f64::min),
            xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            population_sd(xs),
        )
    };
    let (tm, tmed, tmin, tmax, tstd) = stats(&taus);
    let (hm, hmed, hmin, hmax, hstd) = stats(&halfs);
    SpeedSummary {
        n_sessions: sessions.len(),
        n_fit_ok: taus.len(),
        tau_mean_s: tm,
        tau_median_s: tmed,
        tau_min_s: tmin,
        tau_max_s: tmax,
        tau_std_s: tstd,
        half_life_mean_s: hm,
        half_life_median_s: hmed,
        half_life_min_s: hmin,
        half_life_max_s: hmax,
        half_life_std_s: hstd,
    }
}

/// Bootstrap provenance published with the report, so every
/// interval can be traced to the exact design that produced it.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BootstrapProvenance {
    pub seed: u64,
    pub resamples: usize,
    pub headline_block_len_s: f64,
    pub block_lengths_tested_s: Vec<f64>,
    pub sensitivity_resamples: usize,
    pub tau_grid_lo_s: f64,
    pub tau_grid_hi_s: f64,
    pub tau_grid_steps: usize,
    pub fit_min_r_squared: f64,
    pub min_origins_per_horizon: usize,
    pub segment_rule: String,
}

/// The multi-session OFI-response report, serialized as the
/// committed results artifact.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SpeedReport {
    pub schema: String,
    pub bar_ms: i64,
    pub horizons_s: Vec<f64>,
    /// What one unit of the shock variable means, stated in the
    /// artifact so the numbers cannot travel without it.
    pub shock_definition: String,
    pub effective_sample_unit: String,
    pub sessions: Vec<SessionSpeed>,
    pub summary: SpeedSummary,
    /// Plain-language limits of this report, written into the
    /// artifact so they travel with the numbers.
    pub notes: Vec<String>,
    // ---- v2 additions ----
    /// What changed between schema v1 and v2, field by field.
    pub schema_note: String,
    /// The two timing conventions, stated where the numbers live.
    pub timing_conventions: Vec<String>,
    pub bootstrap: BootstrapProvenance,
}

impl SpeedReport {
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("speed report serializes")
    }
}

/// Aggregate per-session results into the report.
pub fn summarize_speed(sessions: Vec<SessionSpeed>, cfg: &SpeedConfig) -> SpeedReport {
    let summary = summarize(&sessions);
    let grid = tau_grid();
    let notes = vec![
        "The estimand is observational: order-flow imbalance and mid-price changes are \
         measured from the same quote stream, and the recorded sessions carry no news \
         timestamps. Nothing here identifies a news event, an informed trader, or a \
         causal effect; the curves are local-projection associations between one bar's \
         OFI and later mid-price changes."
            .to_string(),
        "Two timing conventions are reported per session. The inclusive response uses \
         the close of the bar before the shock bar as its baseline, so its 0 s rung is \
         the move inside the shock bar itself. The strictly post-bar response uses the \
         shock bar's own close, so its 0 s rung is identically zero by construction."
            .to_string(),
        "Effective sample unit: one independent recording session. Bars inside a \
         session are serially dependent and are never counted as independent samples; \
         each session also reports its effective number of non-overlapping \
         max-horizon blocks, and the cross-session summary is descriptive statistics \
         over sessions."
            .to_string(),
        "Single venue (Binance), one symbol (BTCUSDT), short recorded sessions. The \
         response describes this sample's OFI-price association, not a market-wide \
         constant."
            .to_string(),
        "Raw-OFI and OFI-innovation projections are different estimands and are \
         reported side by side: the raw curve is the total association including \
         follow-on flow the originating bar predicts; the innovation curve conditions \
         on the previous 10 bars of OFI. Neither is a causal effect."
            .to_string(),
        "Response-share ratios are reported only when the denominator's interval \
         excludes zero; otherwise the share is suppressed in the report with the \
         reason recorded."
            .to_string(),
        "tau is reported only for sessions whose response the jump-term adjustment \
         curve actually fits (R^2 at or above the published bar, positive long-run \
         response, tau interior to the search grid) and whose joint parameter region \
         from the bootstrap fits is bounded. Sessions that fail keep their empirical \
         curves, the failure reason, the profile loss over tau, and the gate \
         sensitivity table instead of a forced number."
            .to_string(),
    ];
    SpeedReport {
        schema: SPEED_SCHEMA.to_string(),
        bar_ms: cfg.bar_ms,
        horizons_s: cfg.horizons_s.iter().map(|s| *s as f64).collect(),
        shock_definition:
            "one standard deviation of per-bar order-flow imbalance (OFI) within the session; an observational regressor measured from the same quote stream as the price, not an identified news shock"
                .to_string(),
        effective_sample_unit: "independent recording session".to_string(),
        sessions,
        summary,
        notes,
        schema_note: "v2 keeps every v1 field with its v1 meaning (schema, bar_ms, \
            horizons_s, shock_definition, effective_sample_unit, the per-session \
            response/response_ci/n_origins/fit fields, and the summary block) and \
            adds: post_response_bps / post_response_ci_bps / post_n_origins (the \
            strictly post-bar timing convention), response_simultaneous_band_bps and \
            its post-bar twin, hac_se_bps / hac_ci_bps, profile (prespecified \
            horizons, contrasts, and share ratios under the suppression rule), \
            shock_diagnostics (OFI autocorrelation and pre-shock regressions), \
            innovation (the lag-conditioned OFI-innovation projections), \
            block_sensitivity, effective_blocks_at_max_horizon, n_segments and \
            longest_segment_bars, the jump-term fit fields (jump_bps, \
            fit_jump_raw_bps, fit_r_inf_raw_bps, fit_tau_raw_s, fit_region, \
            profile_loss_* and gate_sensitivity), extended (the long-session \
            ladder to 1800 s), timing_conventions, and bootstrap provenance. \
            Descriptive strings were renamed from incorporation-speed wording to \
            observational OFI-price-response wording; no v1 numeric field changed \
            meaning, and the inclusive point estimates are the v1 estimates."
            .to_string(),
        timing_conventions: vec![
            "inclusive: baseline is the close of the bar before the shock bar; the 0 s rung is the within-shock-bar move"
                .to_string(),
            "post_bar: baseline is the shock bar's own close; the 0 s rung is identically zero by construction and is stated, not estimated"
                .to_string(),
        ],
        bootstrap: BootstrapProvenance {
            seed: cfg.seed,
            resamples: cfg.bootstrap_resamples,
            headline_block_len_s: 300.0,
            block_lengths_tested_s: SENSITIVITY_BLOCKS_S.iter().map(|v| *v as f64).collect(),
            sensitivity_resamples: cfg.sensitivity_resamples,
            tau_grid_lo_s: grid[0],
            tau_grid_hi_s: *grid.last().unwrap(),
            tau_grid_steps: grid.len(),
            fit_min_r_squared: FIT_MIN_R_SQUARED,
            min_origins_per_horizon: 30,
            segment_rule: "blocks are drawn only within contiguous fully-observed \
                segments; splice points are marked and no response window in a \
                resample crosses a splice, an unobserved stretch, or the session \
                boundary; each drawn block is extended by the longest horizon so \
                that a max-horizon window can lie wholly inside one block"
                .to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::Quote;

    fn xorshift(state: &mut u64) -> f64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state as f64) / (u64::MAX as f64)
    }

    /// Bars whose returns are a geometrically decaying lag response
    /// to white-noise flow: r_t = sum_k c_k x_{t-k} with
    /// c_k = A (1 - rho) rho^k. The local-projection cumulative
    /// response is then A (1 - rho^{h+1}), an exponential approach
    /// to A with tau = -1 / ln(rho) bars, plus an instantaneous
    /// term A (1 - rho) at h = 0.
    fn synth_series(n: usize, rho: f64, a_bps: f64) -> BarSeries {
        let mut s: u64 = 0x1234_5678_9ABC_DEF0;
        let x: Vec<f64> = (0..n).map(|_| xorshift(&mut s) * 2.0 - 1.0).collect();
        let mut mid = Vec::with_capacity(n);
        let mut price = 100_000.0f64;
        let mut lag: Vec<f64> = Vec::new();
        for &xt in &x {
            lag.push(xt);
            let mut r_bps = 0.0;
            for (k, xk) in lag.iter().rev().enumerate() {
                r_bps += a_bps * (1.0 - rho) * rho.powi(k as i32) * xk;
            }
            if lag.len() > 200 {
                lag.remove(0);
            }
            price *= 1.0 + r_bps / 1e4;
            mid.push(price);
        }
        // OFI proportional to the flow, in arbitrary units.
        BarSeries {
            mid,
            ofi: x.iter().map(|v| v * 37.0).collect(),
            observed: vec![true; n],
        }
    }

    /// Bars whose flow is AR(1) with coefficient `phi` and whose
    /// returns respond only contemporaneously to flow (plus the
    /// mechanical carry of the price level): any raw-OFI response
    /// beyond h = 0 is the spurious tail that correlated follow-on
    /// flow creates, and the innovation response must not have it.
    fn synth_ar_series(n: usize, phi: f64, impact_bps: f64) -> BarSeries {
        let mut s: u64 = 0x0BAD_5EED_0BAD_5EED;
        let mut x = Vec::with_capacity(n);
        let mut cur = 0.0;
        for _ in 0..n {
            let e = xorshift(&mut s) * 2.0 - 1.0;
            cur = phi * cur + e;
            x.push(cur);
        }
        let mut mid = Vec::with_capacity(n);
        let mut price = 100_000.0f64;
        for &xt in &x {
            price *= 1.0 + impact_bps * xt / 1e4;
            mid.push(price);
        }
        BarSeries {
            mid,
            ofi: x.iter().map(|v| v * 37.0).collect(),
            observed: vec![true; n],
        }
    }

    fn small_cfg() -> SpeedConfig {
        SpeedConfig {
            bootstrap_resamples: 40,
            sensitivity_resamples: 20,
            extended_resamples: 20,
            ..SpeedConfig::default()
        }
    }

    #[test]
    fn lp_recovers_known_exponential_tau() {
        // rho = exp(-1/20): the true tau is 20 bars = 20 s at 1 s bars.
        let rho = (-1.0f64 / 20.0).exp();
        let series = synth_series(30_000, rho, 9.0);
        let ofi_sd = population_sd(&series.ofi);
        let horizons_bars: Vec<usize> = SPEED_HORIZONS_S.iter().map(|s| *s as usize).collect();
        let horizons_s: Vec<f64> = SPEED_HORIZONS_S.iter().map(|s| *s as f64).collect();
        let (curve, counts) = lp_curve(
            &series.mid,
            &series.ofi,
            &series.observed,
            &horizons_bars,
            ofi_sd,
        );
        assert!(curve.iter().all(|v| v.is_finite()), "curve {curve:?}");
        assert_eq!(*counts.last().unwrap(), 30_000 - 300 - 1);
        let fit = fit_exponential_raw(&horizons_s, &curve).expect("fit");
        assert!(
            judge_fit(&horizons_s, &curve, &fit).is_ok(),
            "known exponential must pass judging: r2 {} tau {}",
            fit.r_squared,
            fit.tau_s
        );
        assert!(
            (fit.tau_s - 20.0).abs() < 8.0,
            "tau {} should recover the planted 20 s",
            fit.tau_s
        );
        assert!(
            fit.r_inf_bps > 3.0 && fit.r_inf_bps < 8.0,
            "r_inf {}",
            fit.r_inf_bps
        );
    }

    /// Geometric-kernel series built by a state recursion
    /// (s_t = rho s_{t-1} + (1 - rho) x_t, return = A s_t), which
    /// plants the same cumulative response as `synth_series`
    /// without truncating the lag kernel: usable for time
    /// constants far beyond the 200-lag window `synth_series`
    /// keeps.
    fn synth_geometric_state(n: usize, rho: f64, a_bps: f64) -> BarSeries {
        let mut s: u64 = 0x2468_ACE0_1357_9BDF;
        let mut mid = Vec::with_capacity(n);
        let mut ofi = Vec::with_capacity(n);
        let mut price = 100_000.0f64;
        let mut state = 0.0;
        for _ in 0..n {
            let x = xorshift(&mut s) * 2.0 - 1.0;
            state = rho * state + (1.0 - rho) * x;
            price *= 1.0 + a_bps * state / 1e4;
            mid.push(price);
            ofi.push(x * 37.0);
        }
        BarSeries {
            mid,
            ofi,
            observed: vec![true; n],
        }
    }

    #[test]
    fn jump_term_recovers_planted_instantaneous_response() {
        // An exact planted curve with a large instantaneous term:
        // R(h) = 3 + 5 (1 - exp(-h/45)). The jump-term fit must
        // recover all three parameters, and a fit forced through
        // R(0) = 0 must fail badly on the same curve (its best
        // R^2 is about 0.48), which is the specification defect
        // the jump term removes.
        let horizons_s: Vec<f64> = SPEED_HORIZONS_S.iter().map(|s| *s as f64).collect();
        let curve: Vec<f64> = horizons_s
            .iter()
            .map(|h| 3.0 + 5.0 * (1.0 - (-h / 45.0).exp()))
            .collect();
        let fit = fit_exponential_raw(&horizons_s, &curve).expect("fit");
        assert!((fit.jump_bps - 3.0).abs() < 0.3, "jump {}", fit.jump_bps);
        assert!((fit.r_inf_bps - 5.0).abs() < 0.3, "r_inf {}", fit.r_inf_bps);
        assert!((fit.tau_s - 45.0).abs() < 6.0, "tau {}", fit.tau_s);
        assert!(judge_fit(&horizons_s, &curve, &fit).is_ok());
        // Zero-jump benchmark, fit by hand over the same grid.
        let mean = curve.iter().sum::<f64>() / curve.len() as f64;
        let sst: f64 = curve.iter().map(|r| (r - mean).powi(2)).sum();
        let mut best_sse = f64::INFINITY;
        for tau in tau_grid() {
            let (mut sff, mut sfr) = (0.0, 0.0);
            for (h, r) in horizons_s.iter().zip(curve.iter()) {
                let f = 1.0 - (-h / tau).exp();
                sff += f * f;
                sfr += f * r;
            }
            let r_inf = sfr / sff;
            let sse: f64 = horizons_s
                .iter()
                .zip(curve.iter())
                .map(|(h, r)| (r - r_inf * (1.0 - (-h / tau).exp())).powi(2))
                .sum();
            best_sse = best_sse.min(sse);
        }
        let zero_jump_r2 = 1.0 - best_sse / sst;
        assert!(
            zero_jump_r2 < 0.6,
            "zero-jump R^2 {zero_jump_r2} should fail on a jump curve"
        );
        // And on a sampled series the fitted jump is positive.
        let rho = (-1.0f64 / 20.0).exp();
        let series = synth_series(30_000, rho, 9.0);
        let ofi_sd = population_sd(&series.ofi);
        let horizons_bars: Vec<usize> = SPEED_HORIZONS_S.iter().map(|s| *s as usize).collect();
        let (sampled, _) = lp_curve(
            &series.mid,
            &series.ofi,
            &series.observed,
            &horizons_bars,
            ofi_sd,
        );
        let sfit = fit_exponential_raw(&horizons_s, &sampled).expect("sampled fit");
        assert!(sfit.jump_bps > 0.1, "sampled jump {}", sfit.jump_bps);
    }

    #[test]
    fn fit_rejected_when_response_decays_away() {
        // A response that rises then falls back through zero is not
        // an exponential adjustment path; judging must refuse it.
        let horizons_s: Vec<f64> = SPEED_HORIZONS_S.iter().map(|s| *s as f64).collect();
        let response: Vec<f64> = horizons_s
            .iter()
            .map(|h| 6.0 * (-h / 30.0).exp() - 1.0)
            .collect();
        let fit = fit_exponential_raw(&horizons_s, &response).expect("raw fit exists");
        let verdict = judge_fit(&horizons_s, &response, &fit);
        assert!(
            verdict.is_err(),
            "decaying response must not be reported as a fit"
        );
    }

    #[test]
    fn reversal_curve_is_not_promoted() {
        // Fast positive response overwhelmed by a slower negative
        // one: the cumulative curve ends below zero, and no
        // positive adjustment path may be reported for it.
        let horizons_s: Vec<f64> = SPEED_HORIZONS_S.iter().map(|s| *s as f64).collect();
        let response: Vec<f64> = horizons_s
            .iter()
            .map(|h| 5.0 * (1.0 - (-h / 20.0).exp()) - 6.0 * (1.0 - (-h / 120.0).exp()))
            .collect();
        assert!(*response.last().unwrap() < 0.0);
        let fit = fit_exponential_raw(&horizons_s, &response).expect("raw fit exists");
        assert!(
            judge_fit(&horizons_s, &response, &fit).is_err(),
            "reversal must not be promoted"
        );
    }

    #[test]
    fn two_time_scales_are_not_promoted_as_one() {
        // A fast 2 s component and a slow 300 s component of equal
        // size: a single jump-term exponential explains only about
        // 0.84 of this exact curve, below the 0.85 bar, so the
        // documented behaviour is refusal, not a forced tau.
        let horizons_s: Vec<f64> = SPEED_HORIZONS_S.iter().map(|s| *s as f64).collect();
        let response: Vec<f64> = horizons_s
            .iter()
            .map(|h| 4.0 * (1.0 - (-h / 2.0).exp()) + 4.0 * (1.0 - (-h / 300.0).exp()))
            .collect();
        let fit = fit_exponential_raw(&horizons_s, &response).expect("raw fit exists");
        assert!(
            fit.r_squared < FIT_MIN_R_SQUARED,
            "two-scale R^2 {} should fall below the bar",
            fit.r_squared
        );
        assert!(judge_fit(&horizons_s, &response, &fit).is_err());
    }

    #[test]
    fn tau_beyond_the_horizon_grid_is_not_gate_blocked() {
        // A planted tau of 600 s: the tau search grid runs to
        // 1500 s, so the fit must be able to land beyond the 300 s
        // response ladder, and judging must not refuse it for its
        // size. Identification from 300 s of curve is weak (the
        // report publishes the region and profile for that), but
        // the gate is not what blocks it.
        let rho = (-1.0f64 / 600.0).exp();
        let series = synth_geometric_state(30_000, rho, 9.0);
        let ofi_sd = population_sd(&series.ofi);
        let horizons_bars: Vec<usize> = SPEED_HORIZONS_S.iter().map(|s| *s as usize).collect();
        let horizons_s: Vec<f64> = SPEED_HORIZONS_S.iter().map(|s| *s as f64).collect();
        let (curve, _) = lp_curve(
            &series.mid,
            &series.ofi,
            &series.observed,
            &horizons_bars,
            ofi_sd,
        );
        let fit = fit_exponential_raw(&horizons_s, &curve).expect("fit");
        assert!(
            fit.tau_s > 300.0,
            "tau {} should land beyond the 300 s ladder",
            fit.tau_s
        );
        assert!(
            judge_fit(&horizons_s, &curve, &fit).is_ok(),
            "a 600 s tau is interior to the search grid and must not be gate-blocked: r2 {}",
            fit.r_squared
        );
    }

    #[test]
    fn extended_ladder_recovers_a_slow_tau_on_a_long_series() {
        // A planted 600 s tau on a 6,000-bar series: the extended
        // ladder must reach 1800 s, and the extended fit (judged
        // under the same published gates) must report a tau in the
        // planted neighbourhood, because on this ladder the slow
        // component's plateau is actually in view.
        let rho = (-1.0f64 / 600.0).exp();
        let series = synth_geometric_state(6_000, rho, 9.0);
        let cfg = SpeedConfig {
            extended_resamples: 8,
            ..small_cfg()
        };
        let ext = extended_analysis(&series, &cfg).expect("extended analysis");
        assert_eq!(*ext.horizons_s.last().unwrap(), 1800.0);
        assert!(ext.response_bps.iter().all(|v| v.is_finite()));
        let tau = ext
            .fit_tau_s
            .expect("extended fit should be reported on an exact-family curve");
        assert!(
            (250.0..1200.0).contains(&tau),
            "extended tau {tau} should sit near the planted 600 s"
        );
    }

    #[test]
    fn lp_censors_origins_crossing_a_gap() {
        // 12 bars, bars 5..=7 unobserved. Hand-count usable origins.
        let n = 12;
        let mid: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();
        let ofi: Vec<f64> = (0..n)
            .map(|i| if i % 2 == 0 { 2.0 } else { -1.0 })
            .collect();
        let mut observed = vec![true; n];
        observed[5] = false;
        observed[6] = false;
        observed[7] = false;
        let (_, counts) = lp_curve(&mid, &ofi, &observed, &[0, 2], 1.0);
        // h = 0: origins t in 1..=11 with bars t-1..=t observed:
        // t = 1..=5 ok except t=5 (bar 5 unobs) and t=6,7,8 (window
        // touches 5..=7); usable t = 1,2,3,4,9,10,11 -> 7.
        assert_eq!(counts[0], 7, "h=0 origins");
        // h = 2: t + 2 <= 11 and bars t-1..=t+2 observed:
        // t = 1,2 (windows 0..=3, 1..=4) and t = 9 (8..=11) -> 3.
        assert_eq!(counts[1], 3, "h=2 origins");
    }

    #[test]
    fn post_bar_response_is_zero_at_h0_by_construction() {
        let rho = (-1.0f64 / 20.0).exp();
        let series = synth_series(3_000, rho, 9.0);
        let ofi_sd = population_sd(&series.ofi);
        let horizons_bars: Vec<usize> = SPEED_HORIZONS_S.iter().map(|s| *s as usize).collect();
        let n = series.mid.len();
        let (incl, incl_n, post, post_n) = lp_pair(
            &series.mid,
            &series.ofi,
            &series.observed,
            &vec![false; n],
            &vec![true; n],
            &horizons_bars,
            ofi_sd,
            true,
        );
        assert!(incl[0] > 0.0, "inclusive h=0 carries the shock-bar move");
        assert_eq!(post[0], 0.0, "post-bar h=0 is identically zero");
        // The post-bar window is one bar shorter, so it never has
        // fewer usable origins than the inclusive window.
        for k in 0..horizons_bars.len() {
            assert!(post_n[k] >= incl_n[k], "horizon {k}");
        }
    }

    #[test]
    fn resample_blocks_never_cross_a_gap() {
        // One outage: bars 400..600 unobserved in a 1000-bar series.
        let mut observed = vec![true; 1000];
        for slot in observed.iter_mut().take(600).skip(400) {
            *slot = false;
        }
        let segments = segments_of(&observed);
        assert_eq!(segments, vec![(0, 400), (600, 1000)]);
        let mut rng = SplitMix64(42);
        for _ in 0..50 {
            let (idxs, breaks) = resample_indices(&segments, 50, 800, &mut rng);
            assert_eq!(idxs.len(), 800);
            // Every drawn bar was observed in the original series.
            assert!(idxs.iter().all(|&i| observed[i]));
            // Inside a block, source indices advance by exactly 1
            // and never enter the gap; splices are marked where a
            // new block starts.
            for w in 1..idxs.len() {
                if breaks[w] {
                    continue;
                }
                assert_eq!(idxs[w], idxs[w - 1] + 1, "block must be contiguous");
                assert!(
                    !(idxs[w - 1] < 400 && idxs[w] >= 600),
                    "a block joined bars across the outage"
                );
            }
        }
    }

    #[test]
    fn bootstrap_is_deterministic() {
        let rho = (-1.0f64 / 20.0).exp();
        let series = synth_series(3_000, rho, 9.0);
        let cfg = small_cfg();
        let horizons_bars: Vec<usize> = SPEED_HORIZONS_S.iter().map(|s| *s as usize).collect();
        let horizons_s: Vec<f64> = SPEED_HORIZONS_S.iter().map(|s| *s as f64).collect();
        let run = |cfg: &SpeedConfig| {
            bootstrap_engine(
                &series,
                None,
                &horizons_bars,
                &horizons_s,
                300,
                cfg.bootstrap_resamples,
                cfg.seed,
                true,
                false,
            )
        };
        let (c1, c2) = (run(&cfg), run(&cfg));
        assert_eq!(c1.incl, c2.incl);
        assert_eq!(c1.fits_pass.len(), c2.fits_pass.len());
        assert!(!c1.incl.is_empty());
    }

    #[test]
    fn correlated_ofi_diagnostics_and_the_innovation_estimand() {
        // Serially correlated OFI with a purely contemporaneous
        // price response: the OFI autocorrelation diagnostic must
        // see the correlation, the innovation regression must
        // recover the persistence and leave near-white residuals,
        // and the two estimands' relationship must match the
        // construction (assertions below).
        let series = synth_ar_series(30_000, 0.85, 3.0);
        let acf = ofi_acf(&series);
        assert!(acf[0] > 0.6, "lag-1 OFI ACF {} should be large", acf[0]);
        let ofi_sd = population_sd(&series.ofi);
        let horizons_bars: Vec<usize> = SPEED_HORIZONS_S.iter().map(|s| *s as usize).collect();
        let (raw, _) = lp_curve(
            &series.mid,
            &series.ofi,
            &series.observed,
            &horizons_bars,
            ofi_sd,
        );
        let (resid, _, _, resid_sd) = ofi_innovations(&series);
        assert!(resid_sd > 0.0 && resid_sd.is_finite());
        // Residual autocorrelation is far below the raw one.
        let resid_series = BarSeries {
            mid: series.mid.clone(),
            ofi: resid
                .iter()
                .map(|v| if v.is_finite() { *v } else { 0.0 })
                .collect(),
            observed: resid.iter().map(|v| v.is_finite()).collect(),
        };
        let resid_acf = ofi_acf(&resid_series);
        assert!(
            resid_acf[0].abs() < 0.35,
            "residual lag-1 ACF {} should be small",
            resid_acf[0]
        );
        let n = series.mid.len();
        let ok: Vec<bool> = resid.iter().map(|v| v.is_finite()).collect();
        let (innov, _, _, _) = lp_pair(
            &series.mid,
            &resid,
            &series.observed,
            &vec![false; n],
            &ok,
            &horizons_bars,
            resid_sd,
            false,
        );
        let i60 = SPEED_HORIZONS_S.iter().position(|s| *s == 60).unwrap();
        // Both curves grow far beyond their h = 0 values in this
        // construction, and for a documented reason: the price
        // response is purely contemporaneous, so the entire tail
        // is follow-on flow persistence (an innovation in bar t is
        // still present, decayed by phi^k, in bar t+k's flow, and
        // that flow moves the price then). The innovation estimand
        // does not delete that propagation; what it changes is the
        // regressor (next assertions) and the per-SD scale: with
        // sd(x)/sd(e) = 1/sqrt(1-phi^2) ~ 1.9, the raw per-SD
        // responses sit about 1.9x the innovation ones.
        assert!(
            raw[i60] > 3.0 * raw[0] && innov[i60] > 3.0 * innov[0],
            "both tails are persistence-driven: raw {} -> {}, innov {} -> {}",
            raw[0],
            raw[i60],
            innov[0],
            innov[i60]
        );
        let scale = raw[0] / innov[0];
        assert!(
            (1.4..2.5).contains(&scale),
            "per-SD scale ratio {scale} should sit near 1/sqrt(1-phi^2) = 1.9"
        );
        // The innovation regression itself recovers the planted
        // persistence in its first lag coefficient.
        let (_, coefs, _, _) = ofi_innovations(&series);
        assert!(
            (coefs[0] - 0.85).abs() < 0.15,
            "lag-1 innovation coefficient {} should recover phi = 0.85",
            coefs[0]
        );
    }

    #[test]
    fn analyze_reports_both_conventions_and_suppresses_unidentified_shares() {
        // Flow with no price response at all: the price is
        // constant, every response is exactly zero, and every
        // share denominator interval is [0, 0], which includes
        // zero, so every share must be suppressed with a reason.
        let mut s: u64 = 7;
        let n = 1799;
        let series = BarSeries {
            mid: vec![100_000.0; n],
            ofi: (0..n)
                .map(|_| (xorshift(&mut s) * 2.0 - 1.0) * 37.0)
                .collect(),
            observed: vec![true; n],
        };
        let res = analyze_series("flat", &series, 1798.0, &small_cfg()).expect("analysis");
        assert_eq!(res.post_response_bps[0], 0.0);
        assert!(res.profile.shares.iter().all(|sh| sh.suppressed));
        assert!(res
            .profile
            .shares
            .iter()
            .all(|sh| sh.suppression_reason.is_some()));
        assert_eq!(res.profile.contrasts.len(), 4);
        assert_eq!(res.gate_sensitivity.len(), 12);
        assert_eq!(res.profile_loss_tau_s.len(), 120);
        // One segment of 1799 bars: 150 s and 300 s blocks are
        // feasible, 600 s and 900 s are not, and the effective
        // number of non-overlapping 300 s blocks is 5.
        let feas: Vec<bool> = res.block_sensitivity.iter().map(|b| b.feasible).collect();
        assert_eq!(feas, vec![true, true, false, false]);
        assert!(res.block_sensitivity[2].reason.is_some());
        assert_eq!(res.effective_blocks_at_max_horizon, 5);
        assert_eq!(res.n_segments, 1);
        assert!(res.extended.is_none());
        assert!(!res.fit_ok);
    }

    #[test]
    fn simultaneous_band_uses_the_joint_multiplier() {
        // Fabricated draws across all 16 horizons: the sup
        // multiplier (95th percentile of the largest standardized
        // deviation) must exceed the single-horizon 1.96, which is
        // the whole point of a simultaneous band.
        let mut s: u64 = 0xFEED;
        let h = SPEED_HORIZONS_S.len();
        let point = vec![0.0; h];
        let draws: Vec<Vec<f64>> = (0..500)
            .map(|_| {
                (0..h)
                    .map(|_| {
                        let u: f64 = (0..3).map(|_| xorshift(&mut s)).sum();
                        (u - 1.5) * 2.0
                    })
                    .collect()
            })
            .collect();
        let band = simultaneous_band(&point, &draws);
        for k in 0..h {
            let col: Vec<f64> = draws.iter().map(|d| d[k]).collect();
            let sd = population_sd(&col);
            assert!(
                band[k].1 - point[k] >= 1.96 * sd,
                "band half-width {} should exceed 1.96 sd {}",
                band[k].1,
                sd
            );
            assert!(band[k].0 <= point[k] && point[k] <= band[k].1);
        }
    }

    #[test]
    fn hac_interval_for_a_null_response_contains_zero() {
        // Returns driven by noise independent of the flow: the
        // HAC interval at 60 s must contain zero, and its standard
        // error must be positive and finite at every horizon.
        let mut s: u64 = 0xABCD;
        let n = 20_000;
        let mut price = 100_000.0f64;
        let mut mid = Vec::with_capacity(n);
        let mut ofi = Vec::with_capacity(n);
        for _ in 0..n {
            let r = (xorshift(&mut s) * 2.0 - 1.0) * 3.0;
            price *= 1.0 + r / 1e4;
            mid.push(price);
            ofi.push((xorshift(&mut s) * 2.0 - 1.0) * 37.0);
        }
        let sd = population_sd(&ofi);
        let horizons_bars: Vec<usize> = SPEED_HORIZONS_S.iter().map(|v| *v as usize).collect();
        let observed = vec![true; n];
        let (ses, cis) = hac_inference(&mid, &ofi, &observed, &horizons_bars, sd);
        assert!(ses.iter().all(|v| v.is_finite() && *v > 0.0));
        let i60 = SPEED_HORIZONS_S.iter().position(|v| *v == 60).unwrap();
        assert!(
            cis[i60].0 <= 0.0 && cis[i60].1 >= 0.0,
            "HAC CI {:?} should contain zero",
            cis[i60]
        );
    }

    #[test]
    fn report_serializes_v2_with_provenance() {
        let rho = (-1.0f64 / 20.0).exp();
        let series = synth_series(1799, rho, 9.0);
        let cfg = small_cfg();
        let res = analyze_series("synth", &series, 1798.0, &cfg).expect("analysis");
        let report = summarize_speed(vec![res], &cfg);
        let v: serde_json::Value = serde_json::from_str(&report.to_json()).unwrap();
        assert_eq!(v["schema"], "toxicflow.info-speed/v2");
        assert_eq!(v["bootstrap"]["seed"], SPEED_BOOTSTRAP_SEED);
        assert_eq!(v["bootstrap"]["resamples"], 40);
        let s0 = &v["sessions"][0];
        assert!(s0["post_response_bps"].is_array());
        assert!(s0["profile"]["shares"].is_array());
        assert!(s0["shock_diagnostics"]["ofi_acf"].as_array().unwrap().len() == 30);
        assert!(s0["innovation"]["response_bps"].is_array());
        // v1 fields keep their places.
        assert!(s0["response_bps"].is_array());
        assert!(s0["bootstrap_fit_share"].is_number());
        assert!(v["summary"]["n_sessions"] == 1);
    }

    #[test]
    fn short_or_flat_session_returns_none() {
        let q = |t: i64| Quote {
            time_ms: t,
            bid_px: 100.0,
            bid_sz: 1.0,
            ask_px: 100.1,
            ask_sz: 1.0,
        };
        // Constant quotes: no OFI variation at all.
        let quotes: Vec<Quote> = (0..2000).map(|i| q(i * 1000)).collect();
        assert!(analyze_session_speed("flat", &quotes, &SpeedConfig::default()).is_none());
        // Far too short for the 300 s horizon.
        let quotes: Vec<Quote> = (0..100).map(|i| q(i * 1000)).collect();
        assert!(analyze_session_speed("tiny", &quotes, &SpeedConfig::default()).is_none());
        assert!(analyze_session_speed("empty", &[], &SpeedConfig::default()).is_none());
    }
}
