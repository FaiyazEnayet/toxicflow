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
//! Segment choice is two-stage and segment-fair (see
//! `resample_indices`): a segment is chosen with weight
//! max(segment length, block length), then a uniform valid start
//! inside it, and a segment shorter than the block is drawn whole.
//! That weighting makes every segment's expected share of the
//! resampled bars equal to its share of the observed bars, so no
//! segment is silently excluded for being short and no segment is
//! weighted by its count of valid starts. Every ladder reports the
//! size of the block-start pool it sampled from and how many
//! distinct blocks its resamples actually drew; a ladder whose
//! pool falls below `MIN_POOL_DISTINCT_STARTS` distinct starts is
//! reported as degenerate, with its intervals suppressed and the
//! reason recorded, instead of as a narrow interval backed by a
//! full resample count.
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
//!
//! Cross-checks (reported per session in a `crosschecks` block,
//! never in the report summary):
//!
//! - A reduced-form VAR on the same bar series (price returns,
//!   OFI, spread, imbalance), estimated per session with its lag
//!   order chosen by BIC over a bounded range. Its cumulative
//!   price response to a one-SD OFI shock is computed under two
//!   Cholesky orderings and compared against the local-projection
//!   curves horizon by horizon. This is a reduced-form
//!   specification cross-check only; it is NOT a Hasbrouck
//!   information-share estimate (single venue, single price
//!   series — that estimand is unavailable); no information share
//!   is computed or implied anywhere; orthogonalisation is an
//!   ordering assumption, not identification.
//! - A two-component adjustment fit,
//!   R(h) = J + Af (1 - exp(-h / tau_f)) + As (1 - exp(-h / tau_s)),
//!   fitted inside the same segment-respecting bootstrap draws as
//!   the single-component fit. No tau_f, tau_s, or derived
//!   half-life from this model is reported as a result anywhere:
//!   the block carries joint parameter regions and a stability
//!   label only, and sessions are labelled unidentified/unstable
//!   wherever the draws say so.

use crate::metrics::{bar_status, ofi_bars, ofi_events, Quote, Trade};

/// Schema tag written into every OFI-response report. v2 adds the
/// post-bar series, the model-free profile with contrasts and
/// share suppression, segment-bootstrap sensitivity and
/// simultaneous bands, HAC intervals, innovation projections,
/// shock diagnostics, the jump-term fit with its parameter region
/// and gate-sensitivity table, the extended main-session ladder,
/// and bootstrap provenance, on top of the v1 fields (see the
/// report's `schema_note`).
pub const SPEED_SCHEMA: &str = "toxicflow.info-speed/v3";

/// Minimum number of distinct block starts in a ladder's resample
/// pool before its bootstrap intervals are reported at all. Below
/// this pool size the resamples are repeats of a handful of chunks:
/// with fewer than 20 distinct blocks, the 2.5% tail of a resample
/// distribution is set by at most one or two distinct chunks, so a
/// percentile interval computed from it states a precision the
/// pool does not contain. Such a ladder is reported as degenerate
/// (its pool size and diversity are published, its intervals are
/// suppressed, and the reason is recorded) rather than as a narrow
/// interval with a full resample count.
pub const MIN_POOL_DISTINCT_STARTS: usize = 20;

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

/// Bounded lag range searched for the reduced-form VAR
/// cross-check: orders 1..=VAR_MAX_LAG are fit on a common sample
/// and the order minimizing BIC is used. The range is stated here,
/// before any result, and lags are never hand-tuned per session.
pub const VAR_MAX_LAG: usize = 10;

/// Residual-bootstrap resamples behind the VAR cross-check bands.
/// The VAR bootstrap is a separate, smaller design from the
/// headline segment bootstrap (it resamples VAR residuals, not
/// bars), so it carries its own stated count and a seed derived
/// from the published seed.
pub const VAR_BOOTSTRAP_RESAMPLES: usize = 200;

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

// ------------------------------------------------------------------
// Cross-check 2: the two-component adjustment fit, under guardrails.
//
// Model: R(h) = J + Af (1 - exp(-h / tau_f)) + As (1 - exp(-h / tau_s))
// with tau_f < tau_s, fitted unweighted to the full response vector
// (the same timing convention as the headline single-component fit:
// the inclusive curve). The fit runs on the point curve and inside
// every headline bootstrap draw, reusing the segment-respecting
// bootstrap and its within-draw fitting machinery; no second
// bootstrap is built for it.
//
// Amplitude policy, stated before any result: amplitudes are
// fitted UNCONSTRAINED, so Af or As may come out negative. A
// negative amplitude is not an adjustment component: it describes
// reversal or overshoot (part of the response being given back),
// which is exactly what the three sessions whose responses decline
// from 60 to 300 s would need. Allowing negative amplitudes keeps
// the fit honest on those curves instead of forcing a shape they
// do not have, at the cost of the two-component reading: a fit
// with a negative amplitude cannot be read as two adjustment
// speeds, and the stability criteria below say so.
//
// Stability criteria, stated before looking at any outcome. A
// single draw's fit counts as stable / identified-looking only if
// ALL of the following hold:
//   1. its unweighted R^2 is at least FIT_MIN_R_SQUARED (the same
//      bar the headline single-component fit is judged at);
//   2. tau_f and tau_s are both strictly interior to the
//      two-component search grid by at least one grid step;
//   3. the components are separated: tau_s / tau_f >= 3;
//   4. both amplitudes are positive (Af > 0 and As > 0);
//   5. the total long-run response J + Af + As is positive.
// A session is labelled stable only if its point fit passes the
// same five criteria, at least half of its bootstrap draws are
// stable, and the bootstrap percentile regions for both taus are
// bounded (inside the grid interior, ratio of ends below 4, the
// same boundedness reading as the single-component fit region).
// Otherwise the session is labelled unidentified/unstable. The
// label is the published outcome: no tau_f, tau_s or derived
// half-life from this model is promoted anywhere.
// ------------------------------------------------------------------

/// One two-component adjustment fit.
#[derive(Debug, Clone, Copy)]
pub struct TwoCompFit {
    pub jump_bps: f64,
    pub a_fast_bps: f64,
    pub a_slow_bps: f64,
    pub tau_fast_s: f64,
    pub tau_slow_s: f64,
    pub r_squared: f64,
    pub loss: f64,
}

/// Tau grid for the two-component fit: every fifth point of the
/// single-component grid (48 log-spaced values, 0.25-1500 s). The
/// pair search is quadratic in the grid size, and this fit runs
/// inside every bootstrap draw, so it profiles a coarser grid
/// than the single-component fit; the grid is stated here and the
/// stability criteria read interiority against this grid.
fn two_tau_grid() -> Vec<f64> {
    tau_grid().iter().step_by(5).copied().collect()
}

/// Fit the two-component path over the finite response points by
/// profiling (tau_f, tau_s) over the two-component grid and solving
/// for (J, Af, As) in closed form at each pair. Unweighted, like
/// the within-draw single-component fits it sits alongside.
/// Returns the best fit found, without judging it.
fn fit_two_component_raw(horizons_s: &[f64], response: &[f64]) -> Option<TwoCompFit> {
    let keep: Vec<usize> = (0..response.len())
        .filter(|&i| response[i].is_finite())
        .collect();
    if keep.len() < 5 {
        return None;
    }
    let hs: Vec<f64> = keep.iter().map(|&i| horizons_s[i]).collect();
    let ys: Vec<f64> = keep.iter().map(|&i| response[i]).collect();
    let hp = hs.len();
    let mean = ys.iter().sum::<f64>() / hp as f64;
    let sst: f64 = ys.iter().map(|y| (y - mean).powi(2)).sum();
    if sst <= 0.0 || sst.is_nan() {
        return None;
    }
    let syy: f64 = ys.iter().map(|y| y * y).sum();
    let sy: f64 = ys.iter().sum();
    let grid = two_tau_grid();
    // Basis vectors per grid tau: f_k[h] = 1 - exp(-h / tau_k).
    let basis: Vec<Vec<f64>> = grid
        .iter()
        .map(|tau| hs.iter().map(|h| 1.0 - (-h / tau).exp()).collect())
        .collect();
    let bsum: Vec<f64> = basis.iter().map(|b| b.iter().sum()).collect();
    let by: Vec<f64> = basis
        .iter()
        .map(|b| b.iter().zip(ys.iter()).map(|(f, y)| f * y).sum())
        .collect();
    let mut best: Option<TwoCompFit> = None;
    for i in 0..grid.len() {
        for j in (i + 1)..grid.len() {
            let (u, v) = (&basis[i], &basis[j]);
            let suu: f64 = u.iter().map(|x| x * x).sum();
            let svv: f64 = v.iter().map(|x| x * x).sum();
            let suv: f64 = u.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            let m = [
                vec![hp as f64, bsum[i], bsum[j]],
                vec![bsum[i], suu, suv],
                vec![bsum[j], suv, svv],
            ];
            let rhs = vec![sy, by[i], by[j]];
            let Some(coef) = solve_linear(&m, &rhs) else {
                continue;
            };
            // SSE from the normal equations: syy - coef . (X^T y).
            let sse = syy - (coef[0] * sy + coef[1] * by[i] + coef[2] * by[j]);
            if !sse.is_finite() {
                continue;
            }
            let fit = TwoCompFit {
                jump_bps: coef[0],
                a_fast_bps: coef[1],
                a_slow_bps: coef[2],
                tau_fast_s: grid[i],
                tau_slow_s: grid[j],
                r_squared: 1.0 - sse / sst,
                loss: sse,
            };
            if best.as_ref().is_none_or(|b| fit.loss < b.loss) {
                best = Some(fit);
            }
        }
    }
    best
}

/// Whether one two-component fit passes the pre-stated stability
/// criteria (see the block comment above). This judges stability
/// of the two-component reading only; it never promotes a number.
fn two_component_stable(fit: &TwoCompFit) -> bool {
    if fit.r_squared < FIT_MIN_R_SQUARED {
        return false;
    }
    let grid = two_tau_grid();
    if grid.len() < 3 {
        return false;
    }
    let (lo, hi) = (grid[1], grid[grid.len() - 2]);
    if fit.tau_fast_s <= lo || fit.tau_slow_s >= hi {
        return false;
    }
    if fit.tau_slow_s / fit.tau_fast_s < 3.0 {
        return false;
    }
    if fit.a_fast_bps <= 0.0 || fit.a_slow_bps <= 0.0 {
        return false;
    }
    if fit.jump_bps + fit.a_fast_bps + fit.a_slow_bps <= 0.0 {
        return false;
    }
    true
}

// ------------------------------------------------------------------
// Cross-check 1: the reduced-form VAR.
//
// Specification, stated once and followed for every session:
// - Per-session estimation. The six sessions are heterogeneous
//   (their OFI standard deviations differ about fivefold, and the
//   local projections are per-session for the same reason), so no
//   pooled VAR is fit; a pooled fit with session effects would
//   average dynamics the report elsewhere refuses to average.
// - Variables, one per bar on the same 1 s grid the projections
//   use: the simple mid-price return of the bar in bps (the same
//   return definition as the projections, whose cumulative sums
//   the VAR responses are compared against), per-bar OFI divided
//   by the session OFI standard deviation (the projection shock
//   scale), the quoted spread in bps of the mid, and the book
//   imbalance, both as computed by `metrics::ofi_bars`.
// - Observations are bars whose return is defined inside one
//   fully-observed segment; lag windows never cross a segment
//   boundary, the same censoring discipline as the projections.
// - Lag order: BIC over orders 1..=VAR_MAX_LAG on a common sample
//   (every order is fit on the observations usable at the maximum
//   lag, so the criteria are comparable). BIC per observation is
//   ln det Sigma_ML + (p d^2 + d) ln(T) / T with d = 4 variables.
//   Lags are chosen by this criterion alone, never adjusted to
//   match the projection curves.
// - Estimation is equation-by-equation OLS (exact for a VAR with
//   the same regressors in every equation), carried out on
//   standardized variables for numerical conditioning (the spread
//   is near-constant at one tick); responses are converted back
//   to bps per one-SD OFI shock before they are reported.
// - Impulse responses: cumulative sums of the return responses to
//   a structural OFI shock scaled so its impact on OFI is one
//   session SD of per-bar OFI, under two Cholesky orderings (OFI
//   ordered first; returns ordered first). Summing the per-bar
//   return responses to bar H gives the response over the same
//   window as the inclusive local projection at horizon H, whose
//   baseline is the close before the shock bar.
// - Bands: a residual bootstrap (VAR_BOOTSTRAP_RESAMPLES draws,
//   SplitMix64 seeded from the published seed). Residuals are
//   centred, resampled with replacement, and new series are
//   generated recursively inside each original segment from that
//   segment's own first p observations; the VAR is re-estimated
//   at the chosen lag (the lag is not re-selected per draw) and
//   the responses recomputed. Bands are (2.5%, 97.5%) percentiles.
//
// Restraints, in substance: this is a reduced-form specification
// cross-check only; it is NOT a Hasbrouck information-share
// estimate (single venue, single price series — that estimand is
// unavailable); no information share is computed or implied
// anywhere; orthogonalisation is an ordering assumption, not
// identification.
// ------------------------------------------------------------------

/// Variable names of the VAR, in the storage order of `VarData`.
pub const VAR_VARIABLES: [&str; 4] = [
    "return_bps",
    "ofi_per_session_sd",
    "spread_bps",
    "imbalance",
];

/// Index of the return variable in the VAR storage order.
const VAR_RET: usize = 0;
/// Index of the OFI variable in the VAR storage order.
const VAR_OFI: usize = 1;

/// Per-bar VAR observations: rows of [return, OFI, spread,
/// imbalance] for bars whose return is defined inside a segment,
/// plus the half-open row ranges of each segment in row space.
struct VarData {
    rows: Vec<[f64; 4]>,
    seg_rows: Vec<(usize, usize)>,
    /// Sample mean and SD of each variable over the common
    /// estimation sample at the maximum lag (the standardization
    /// the estimation is carried out under).
    mean: [f64; 4],
    sd: [f64; 4],
}

/// Build the VAR observations from the bar series and the
/// liquidity series (`ofi_bars` spread and imbalance, aligned to
/// the same bar grid). Returns are simple returns in bps against
/// the previous bar of the same segment.
fn build_var_data(series: &BarSeries, spread: &[f64], imbalance: &[f64]) -> Option<VarData> {
    let n = series.mid.len();
    if spread.len() != n || imbalance.len() != n || n == 0 {
        return None;
    }
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
    let mut rows = Vec::new();
    let mut seg_rows = Vec::new();
    for (a, b) in segments_of(&series.observed) {
        if b - a < 2 {
            continue;
        }
        let start = rows.len();
        for t in (a + 1)..b {
            if series.mid[t - 1] <= 0.0 || series.mid[t] <= 0.0 {
                continue;
            }
            rows.push([
                (series.mid[t] / series.mid[t - 1] - 1.0) * 1e4,
                series.ofi[t] / ofi_sd,
                spread[t] / series.mid[t] * 1e4,
                imbalance[t],
            ]);
        }
        if rows.len() > start {
            seg_rows.push((start, rows.len()));
        }
    }
    if rows.len() < 200 {
        return None;
    }
    // Standardization over the common sample at the maximum lag.
    let sample: Vec<usize> = var_sample_indices(&seg_rows, VAR_MAX_LAG);
    if sample.len() < 100 {
        return None;
    }
    let mut mean = [0.0; 4];
    let mut sd = [0.0; 4];
    for (v, m) in mean.iter_mut().enumerate() {
        let vals: Vec<f64> = sample.iter().map(|&i| rows[i][v]).collect();
        *m = vals.iter().sum::<f64>() / vals.len() as f64;
        sd[v] = population_sd(&vals);
    }
    if sd.iter().any(|v| *v <= 0.0 || !v.is_finite()) {
        return None;
    }
    Some(VarData {
        rows,
        seg_rows,
        mean,
        sd,
    })
}

/// Row indices of the estimation sample at lag `p`: rows t whose
/// whole lag window (t-p..=t) lies in the same segment as t.
fn var_sample_indices(seg_rows: &[(usize, usize)], p: usize) -> Vec<usize> {
    let mut out = Vec::new();
    for &(a, b) in seg_rows {
        if b - a > p {
            out.extend((a + p)..b);
        }
    }
    out
}

/// One fitted VAR at a fixed lag, on standardized variables.
struct VarFit {
    /// Intercept per equation (standardized space).
    intercept: [f64; 4],
    /// Coefficient matrices A_1..A_p (standardized space), each
    /// stored as coef[lag][equation][variable].
    coef: Vec<[[f64; 4]; 4]>,
    /// Residual covariance (ML, divided by T) in standardized space.
    sigma_ml: [[f64; 4]; 4],
    /// Residuals in sample order (standardized space).
    residuals: Vec<[f64; 4]>,
    n_obs: usize,
    bic: f64,
}

/// Fit a VAR(p) on the common sample at the maximum lag (the
/// sample is the same for every p, by the stated BIC design).
/// Returns None when the design is singular or too short.
fn fit_var(data: &VarData, p: usize) -> Option<VarFit> {
    let sample = var_sample_indices(&data.seg_rows, VAR_MAX_LAG);
    let t_n = sample.len();
    let k = 1 + 4 * p;
    if t_n < k + 20 {
        return None;
    }
    let z = |row: &[f64; 4], v: usize| (row[v] - data.mean[v]) / data.sd[v];
    // Design: [1, z_{t-1}, ..., z_{t-p}] (only the first p lag
    // blocks are used at this order).
    let mut xtx = vec![vec![0.0; k]; k];
    let mut xty = vec![vec![0.0; 4]; k];
    for &t in &sample {
        let mut x = vec![0.0; k];
        x[0] = 1.0;
        for lag in 1..=p {
            for v in 0..4 {
                x[1 + (lag - 1) * 4 + v] = z(&data.rows[t - lag], v);
            }
        }
        let y: [f64; 4] = std::array::from_fn(|v| z(&data.rows[t], v));
        for i in 0..k {
            for j in 0..k {
                xtx[i][j] += x[i] * x[j];
            }
            for v in 0..4 {
                xty[i][v] += x[i] * y[v];
            }
        }
    }
    let mut beta = vec![vec![0.0; 4]; k]; // beta[regressor][equation]
    for v in 0..4 {
        let rhs: Vec<f64> = xty.iter().map(|r| r[v]).collect();
        let col = solve_linear(&xtx, &rhs)?;
        for i in 0..k {
            beta[i][v] = col[i];
        }
    }
    let mut intercept = [0.0; 4];
    let mut coef = vec![[[0.0; 4]; 4]; p];
    for v in 0..4 {
        intercept[v] = beta[0][v];
        for (lag, mat) in coef.iter_mut().enumerate() {
            for (u, cell) in mat[v].iter_mut().enumerate() {
                *cell = beta[1 + lag * 4 + u][v];
            }
        }
    }
    // Residuals and ML covariance.
    let mut residuals = Vec::with_capacity(t_n);
    let mut rss = [[0.0; 4]; 4];
    for &t in &sample {
        let mut e = [0.0; 4];
        for v in 0..4 {
            let mut pred = intercept[v];
            for (li, mat) in coef.iter().enumerate() {
                let lrow = &data.rows[t - 1 - li];
                pred += mat[v]
                    .iter()
                    .enumerate()
                    .map(|(u, c)| c * z(lrow, u))
                    .sum::<f64>();
            }
            e[v] = z(&data.rows[t], v) - pred;
        }
        for i in 0..4 {
            for j in 0..4 {
                rss[i][j] += e[i] * e[j];
            }
        }
        residuals.push(e);
    }
    let mut sigma_ml = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            sigma_ml[i][j] = rss[i][j] / t_n as f64;
        }
    }
    let det = det4(&sigma_ml);
    if det <= 0.0 || !det.is_finite() {
        return None;
    }
    let bic = det.ln() + ((p * 16 + 4) as f64) * (t_n as f64).ln() / t_n as f64;
    Some(VarFit {
        intercept,
        coef,
        sigma_ml,
        residuals,
        n_obs: t_n,
        bic,
    })
}

/// Determinant of a 4x4 matrix by cofactor expansion.
fn det4(m: &[[f64; 4]; 4]) -> f64 {
    let det3 = |a: [[f64; 3]; 3]| -> f64 {
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    };
    let mut total = 0.0;
    for j in 0..4 {
        let mut sub = [[0.0; 3]; 3];
        for (ri, i) in (1..4).enumerate() {
            let mut cj = 0;
            for (jj, val) in m[i].iter().enumerate() {
                if jj != j {
                    sub[ri][cj] = *val;
                    cj += 1;
                }
            }
        }
        let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
        total += sign * m[0][j] * det3(sub);
    }
    total
}

/// Lower Cholesky factor of a symmetric positive-definite 4x4.
fn cholesky4(sigma: &[[f64; 4]; 4]) -> Option<[[f64; 4]; 4]> {
    let mut l = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..=i {
            let mut s = sigma[i][j];
            s -= l[i]
                .iter()
                .zip(l[j].iter())
                .take(j)
                .map(|(a, b)| a * b)
                .sum::<f64>();
            if i == j {
                if s <= 0.0 || !s.is_finite() {
                    return None;
                }
                l[i][j] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    Some(l)
}

/// Cumulative price response (bps) to the OFI structural shock,
/// at the given horizon rungs in bars, under one Cholesky
/// ordering (`ordering` lists variable indices in Cholesky order,
/// first variable first). The shock is scaled so its impact on
/// the OFI variable is exactly one unit of that variable in
/// original scale (one session SD of per-bar OFI). Responses are
/// computed in standardized space and the return leg is converted
/// back to bps with the return variable's SD.
fn var_cumulative_irf(
    fit: &VarFit,
    sd: &[f64; 4],
    ordering: &[usize; 4],
    horizons_bars: &[usize],
) -> Option<Vec<f64>> {
    let p = fit.coef.len();
    // Permute coefficients and covariance into the ordering.
    let mut a: Vec<[[f64; 4]; 4]> = vec![[[0.0; 4]; 4]; p];
    let mut sigma = [[0.0; 4]; 4];
    for (lag, amat) in a.iter_mut().enumerate() {
        for (i, row) in amat.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = fit.coef[lag][ordering[i]][ordering[j]];
            }
        }
    }
    for (i, row) in sigma.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = fit.sigma_ml[ordering[i]][ordering[j]];
        }
    }
    let chol = cholesky4(&sigma)?;
    let ofi_pos = ordering.iter().position(|&v| v == VAR_OFI)?;
    let ret_pos = ordering.iter().position(|&v| v == VAR_RET)?;
    // Structural shock vector (standardized space), rescaled so
    // the OFI impact in original units is one SD of OFI.
    let impact_z = chol[ofi_pos][ofi_pos];
    if impact_z <= 0.0 {
        return None;
    }
    let scale = 1.0 / (impact_z * sd[VAR_OFI]);
    let shock: [f64; 4] = std::array::from_fn(|i| chol[i][ofi_pos] * scale);
    let max_h = horizons_bars.iter().copied().max().unwrap_or(0);
    // MA recursion: psi_0 = I, psi_h = sum_i A_i psi_{h-i}.
    let mut psis: Vec<[[f64; 4]; 4]> = Vec::with_capacity(max_h + 1);
    let mut id = [[0.0; 4]; 4];
    for (i, row) in id.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    psis.push(id);
    for h in 1..=max_h {
        let mut acc = [[0.0; 4]; 4];
        for (li, amat) in a.iter().take(p.min(h)).enumerate() {
            let psi = &psis[h - 1 - li];
            for i in 0..4 {
                for j in 0..4 {
                    let mut s = 0.0;
                    for q in 0..4 {
                        s += amat[i][q] * psi[q][j];
                    }
                    acc[i][j] += s;
                }
            }
        }
        psis.push(acc);
    }
    let mut cum = 0.0;
    let mut by_h: Vec<f64> = Vec::with_capacity(max_h + 1);
    for psi in &psis {
        let mut resp = 0.0;
        for q in 0..4 {
            resp += psi[ret_pos][q] * shock[q];
        }
        cum += resp * sd[VAR_RET];
        by_h.push(cum);
    }
    Some(horizons_bars.iter().map(|&h| by_h[h.min(max_h)]).collect())
}

/// The two Cholesky orderings used as the sensitivity check:
/// OFI ordered first, and returns ordered first. Everything else
/// (spread, imbalance last) is held fixed between them.
pub const VAR_ORDERING_OFI_FIRST: [usize; 4] = [1, 0, 2, 3];
/// Returns-ordered-first Cholesky ordering (see above).
pub const VAR_ORDERING_RET_FIRST: [usize; 4] = [0, 1, 2, 3];

/// Residual-bootstrap VAR responses: new series are generated
/// recursively within each original segment from that segment's
/// first p standardized observations, with centred residuals
/// resampled with replacement. The VAR is re-estimated at the
/// fixed chosen lag on the generated series (same segment layout
/// and common-sample rule) and both orderings' cumulative
/// responses recomputed. Returns draws for (OFI-first,
/// returns-first).
#[allow(clippy::too_many_arguments)]
fn var_bootstrap_irfs(
    data: &VarData,
    fit: &VarFit,
    horizons_bars: &[usize],
    resamples: usize,
    seed: u64,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let p = fit.coef.len();
    let mut draws_a = Vec::new();
    let mut draws_b = Vec::new();
    if fit.residuals.is_empty() || p == 0 {
        return (draws_a, draws_b);
    }
    // Centred residuals.
    let mut centre = [0.0; 4];
    for e in &fit.residuals {
        for v in 0..4 {
            centre[v] += e[v];
        }
    }
    for c in centre.iter_mut() {
        *c /= fit.residuals.len() as f64;
    }
    let centred: Vec<[f64; 4]> = fit
        .residuals
        .iter()
        .map(|e| std::array::from_fn(|v| e[v] - centre[v]))
        .collect();
    // Standardized observations, segment by segment.
    let zrow = |row: &[f64; 4]| -> [f64; 4] {
        std::array::from_fn(|v| (row[v] - data.mean[v]) / data.sd[v])
    };
    let mut rng = SplitMix64(seed);
    for _ in 0..resamples {
        // Generate a full standardized series with the original
        // segment layout.
        let mut gen: Vec<[f64; 4]> = vec![[0.0; 4]; data.rows.len()];
        let mut ok = true;
        for &(a, b) in &data.seg_rows {
            // This segment's generated values, built locally and
            // copied in: seeded with the segment's own first p
            // standardized observations (segments too short to
            // seed keep their own values throughout and never
            // enter the estimation sample).
            let mut seg: Vec<[f64; 4]> = data.rows[a..b].iter().map(&zrow).collect();
            if b - a > p {
                seg.truncate(p);
                while seg.len() < b - a {
                    let e = centred[rng.next_below(centred.len())];
                    let mut y = fit.intercept;
                    for (li, amat) in fit.coef.iter().enumerate() {
                        let prev = seg[seg.len() - 1 - li];
                        for (v, yv) in y.iter_mut().enumerate() {
                            *yv += amat[v]
                                .iter()
                                .zip(prev.iter())
                                .map(|(c, x)| c * x)
                                .sum::<f64>();
                        }
                    }
                    for (v, yv) in y.iter_mut().enumerate() {
                        *yv += e[v];
                    }
                    if y.iter().any(|v| !v.is_finite() || v.abs() > 1e6) {
                        ok = false;
                        break;
                    }
                    seg.push(y);
                }
                if !ok {
                    break;
                }
            }
            for (slot, val) in gen.iter_mut().take(b).skip(a).zip(seg.iter()) {
                *slot = *val;
            }
        }
        if !ok {
            continue;
        }
        // Re-estimate at the same lag on the generated series:
        // wrap it as VarData in original units so `fit_var`
        // applies the identical sample rule (its standardization
        // then re-derives mean/SD of the generated sample, which
        // is the correct bootstrap analogue).
        let gen_data = VarData {
            rows: gen
                .iter()
                .map(|z| std::array::from_fn(|v| z[v] * data.sd[v] + data.mean[v]))
                .collect(),
            seg_rows: data.seg_rows.clone(),
            mean: data.mean,
            sd: data.sd,
        };
        // Recompute the standardization from the generated sample
        // itself, exactly as build_var_data does.
        let Some(gen_data) = var_restandardize(gen_data) else {
            continue;
        };
        let Some(gfit) = fit_var(&gen_data, p) else {
            continue;
        };
        let ia = var_cumulative_irf(&gfit, &gen_data.sd, &VAR_ORDERING_OFI_FIRST, horizons_bars);
        let ib = var_cumulative_irf(&gfit, &gen_data.sd, &VAR_ORDERING_RET_FIRST, horizons_bars);
        if let (Some(a), Some(b)) = (ia, ib) {
            if a.iter()
                .chain(b.iter())
                .all(|v| v.is_finite() && v.abs() < 1e4)
            {
                draws_a.push(a);
                draws_b.push(b);
            }
        }
    }
    (draws_a, draws_b)
}

/// Recompute a VarData's standardization (mean/SD over the common
/// sample at the maximum lag), as `build_var_data` does. Returns
/// None when a variable has no variation.
fn var_restandardize(mut data: VarData) -> Option<VarData> {
    let sample = var_sample_indices(&data.seg_rows, VAR_MAX_LAG);
    if sample.len() < 100 {
        return None;
    }
    for v in 0..4 {
        let vals: Vec<f64> = sample.iter().map(|&i| data.rows[i][v]).collect();
        data.mean[v] = vals.iter().sum::<f64>() / vals.len() as f64;
        data.sd[v] = population_sd(&vals);
    }
    if data.sd.iter().any(|v| *v <= 0.0 || !v.is_finite()) {
        return None;
    }
    Some(data)
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

/// Number of distinct block starts a segment set offers for a
/// drawn block of `block` bars: a segment at least as long as the
/// block offers (length - block + 1) starts; a shorter segment
/// offers exactly one chunk, itself drawn whole. This is the size
/// of the pool `resample_indices` samples from.
fn pool_distinct_starts(segments: &[(usize, usize)], block: usize) -> usize {
    if block == 0 {
        return 0;
    }
    segments
        .iter()
        .map(|&(a, b)| {
            let len = b - a;
            if len >= block {
                len - block + 1
            } else if len > 0 {
                1
            } else {
                0
            }
        })
        .sum()
}

/// One segment-respecting resample, as source-bar indices plus
/// the splice positions in the resampled path, plus the distinct
/// chunks drawn into it as (start, chunk length) pairs. Every
/// block of consecutive source indices lies inside a single
/// observed segment and advances one bar at a time; a splice is
/// marked before the first bar of each block after the first, so
/// window checks can refuse any response window that would cross
/// it. The resampled length is the session's observed-bar count.
///
/// Segment-fair two-stage sampling. An earlier scheme pooled
/// every valid start of every segment and drew uniformly from
/// that pool, which had two defects on segmented sessions: a
/// segment shorter than the drawn block offered no valid start
/// and never appeared in any resample at all, and a longer
/// segment was weighted by (length - block + 1) starts rather
/// than by its length, tilting the resample universe toward the
/// longest segments. The scheme here instead chooses a segment
/// first, with weight max(segment length, block): a segment at
/// least as long as the block is weighted by its length, and a
/// shorter segment is weighted as one whole block, because a
/// draw from it contributes its whole length. A draw from a
/// chosen segment then takes a uniform valid start inside it
/// (a block-length chunk), or the whole segment when the
/// segment is shorter than the block. Since the chunk a draw
/// contributes is min(segment length, block) bars, the weight
/// makes each segment's expected share of the resampled bars
/// exactly its share of the observed bars: every observed bar
/// is represented in expectation, short segments included.
///
/// On a single segment the scheme reduces to the earlier one
/// exactly: the segment choice is forced, the start is uniform
/// over the same valid starts, and the RNG stream is consumed
/// identically, so a clean single-segment session's resamples
/// are unchanged draw for draw. Splice and censoring semantics
/// are unchanged too: chunks never join bars across a segment
/// boundary, and a response window still has to lie wholly
/// inside one drawn chunk to be used, so a short chunk drawn
/// whole can only support the horizons that fit inside it,
/// exactly as the corresponding segment does in the recorded
/// session.
fn resample_indices(
    segments: &[(usize, usize)],
    block: usize,
    target: usize,
    rng: &mut SplitMix64,
) -> (Vec<usize>, Vec<bool>, Vec<(usize, usize)>) {
    if segments.is_empty() || target == 0 || block == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    // Two-stage choice: segment weights max(length, block), then
    // a uniform start within the chosen segment.
    let weights: Vec<usize> = segments.iter().map(|&(a, b)| (b - a).max(block)).collect();
    let total_weight: usize = weights.iter().sum();
    if total_weight == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let mut out: Vec<usize> = Vec::with_capacity(target);
    let mut breaks = vec![false; target];
    let mut drawn: Vec<(usize, usize)> = Vec::new();
    while out.len() < target {
        let seg_idx = if segments.len() == 1 {
            0
        } else {
            let mut pick = rng.next_below(total_weight);
            let mut idx = 0;
            while pick >= weights[idx] {
                pick -= weights[idx];
                idx += 1;
            }
            idx
        };
        let (a, b) = segments[seg_idx];
        let seg_len = b - a;
        if seg_len == 0 {
            break;
        }
        let (start, chunk) = if seg_len >= block {
            let n_starts = seg_len - block + 1;
            let s = if n_starts == 1 {
                a
            } else {
                a + rng.next_below(n_starts)
            };
            (s, block)
        } else {
            (a, seg_len)
        };
        drawn.push((start, chunk));
        let pos = out.len();
        if pos > 0 {
            breaks[pos] = true;
        }
        for j in 0..chunk {
            if out.len() >= target {
                break;
            }
            out.push(start + j);
        }
    }
    (out, breaks, drawn)
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
    /// Every draw's own two-component fit (unweighted), collected
    /// only on the headline run (`want_two`): the two-component
    /// cross-check reuses these same draws and fits no second
    /// bootstrap of its own.
    two_fits_all: Vec<TwoCompFit>,
    /// Resample-pool provenance for this run (pool size, drawn
    /// diversity, degeneracy).
    pool: ResamplePool,
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
    want_two: bool,
) -> BootOut {
    let segments = segments_of(&series.observed);
    let target: usize = series.observed.iter().filter(|o| **o).count();
    let longest = segments.iter().map(|(a, b)| b - a).max().unwrap_or(0);
    let max_h_pre = horizons_bars.iter().copied().max().unwrap_or(0);
    let block_pre = (block_bars.max(1) + max_h_pre).min(longest);
    let mut out = BootOut {
        incl: Vec::new(),
        post: Vec::new(),
        innov: Vec::new(),
        fits_all: Vec::new(),
        fits_pass: Vec::new(),
        two_fits_all: Vec::new(),
        pool: ResamplePool::new(block_pre, pool_distinct_starts(&segments, block_pre)),
    };
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
    let mut distinct_drawn: std::collections::HashSet<(usize, usize)> =
        std::collections::HashSet::new();
    out.pool.resamples_attempted = resamples;
    for _ in 0..resamples {
        let (idxs, breaks, drawn) = resample_indices(&segments, block, target, &mut rng);
        for d in drawn {
            distinct_drawn.insert(d);
        }
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
        if want_two {
            if let Some(tfit) = fit_two_component_raw(horizons_s, &incl) {
                out.two_fits_all.push(tfit);
            }
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
    out.pool.distinct_blocks_drawn = distinct_drawn.len();
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

/// Resample-pool provenance for one bootstrap ladder: how large
/// the pool of distinct blocks the resamples were drawn from is,
/// and how diverse the drawn resamples actually were. A distinct
/// block start is one valid (segment, start) chunk choice: for a
/// segment at least as long as the drawn block, each valid start
/// counts once; for a shorter segment, the whole segment is the
/// single chunk choice it offers. `distinct_blocks_drawn` counts
/// the distinct chunks that appeared across all resamples of the
/// run. When `distinct_starts` falls below
/// `MIN_POOL_DISTINCT_STARTS` the ladder is `degenerate`: its
/// intervals are suppressed and `suppression_reason` says why,
/// instead of a narrow interval being published behind a full
/// resample count (`n_resamples` on the ladder itself is then 0,
/// while `resamples_attempted` here keeps the draw count that was
/// actually run for provenance).
#[derive(Debug, Clone, serde::Serialize)]
pub struct ResamplePool {
    /// Drawn block length in bars (nominal block plus the
    /// ladder's longest horizon, capped at the longest segment).
    pub block_bars: usize,
    pub distinct_starts: usize,
    pub distinct_blocks_drawn: usize,
    pub resamples_attempted: usize,
    pub degenerate: bool,
    pub suppression_reason: Option<String>,
}

impl ResamplePool {
    fn new(block_bars: usize, distinct_starts: usize) -> Self {
        let degenerate = distinct_starts < MIN_POOL_DISTINCT_STARTS;
        Self {
            block_bars,
            distinct_starts,
            distinct_blocks_drawn: 0,
            resamples_attempted: 0,
            degenerate,
            suppression_reason: degenerate.then(|| {
                format!(
                    "degenerate resample pool: only {distinct_starts} distinct block starts \
                     are available for a {block_bars}-bar drawn block, below the minimum of \
                     {MIN_POOL_DISTINCT_STARTS}; bootstrap intervals for this ladder are \
                     suppressed rather than reported as narrow intervals from repeated \
                     resamples of the same few chunks"
                )
            }),
        }
    }
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
    /// Resample pool behind this entry, when the length was run.
    pub resample_pool: Option<ResamplePool>,
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
    /// Resample pool behind the extended intervals. When it is
    /// degenerate the extended intervals are suppressed (null in
    /// the serialized report), `n_resamples` is 0, and
    /// `fit_verdict` records the suppression reason.
    pub resample_pool: ResamplePool,
}

/// The reduced-form VAR cross-check for one session. This is a
/// reduced-form specification cross-check only; it is NOT a
/// Hasbrouck information-share estimate (single venue, single
/// price series — that estimand is unavailable); no information
/// share is computed or implied anywhere; orthogonalisation is
/// an ordering assumption, not identification.
#[derive(Debug, Clone, serde::Serialize)]
pub struct VarCrosscheck {
    pub available: bool,
    pub unavailable_reason: Option<String>,
    /// Variable definitions, in the order of `VAR_VARIABLES`.
    pub variables: Vec<String>,
    pub returns_definition: String,
    pub estimation: String,
    pub lag_criterion: String,
    pub lag_range: (usize, usize),
    /// BIC per candidate lag order (null where that order's
    /// design was singular on this session).
    pub bic_by_lag: Vec<Option<f64>>,
    pub chosen_lag: Option<usize>,
    pub n_obs: usize,
    pub horizons_s: Vec<f64>,
    /// The inclusive local-projection curve the VAR is compared
    /// against (copied from the session's headline fields).
    pub lp_response_bps: Vec<f64>,
    pub ofi_first_irf_bps: Vec<f64>,
    pub ofi_first_ci_bps: Vec<(f64, f64)>,
    pub returns_first_irf_bps: Vec<f64>,
    pub returns_first_ci_bps: Vec<(f64, f64)>,
    /// VAR minus LP, per horizon and ordering, in bps.
    pub diff_ofi_first_minus_lp_bps: Vec<f64>,
    pub diff_returns_first_minus_lp_bps: Vec<f64>,
    /// Agreement criterion, per horizon, for the OFI-first
    /// ordering: the LP estimate lies inside the VAR band AND
    /// the VAR estimate lies inside the LP pointwise interval.
    pub lp_inside_var_band: Vec<bool>,
    pub var_inside_lp_band: Vec<bool>,
    pub horizons_agreeing_ofi_first: usize,
    pub horizons_agreeing_returns_first: usize,
    pub max_abs_ordering_difference_bps: f64,
    pub mean_abs_ordering_difference_bps: f64,
    pub n_bootstrap_draws: usize,
    pub bootstrap: String,
    pub restraint: String,
}

/// The two-component adjustment fit cross-check for one session.
/// This block is the only place this model's parameters appear:
/// no tau_f, tau_s or derived half-life from it is a headline, a
/// summary field, or a reported result anywhere in the report.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TwoComponentCrosscheck {
    pub timing_convention: String,
    /// Raw point-fit values on the session's inclusive curve,
    /// reported inside this cross-check block only and never
    /// promoted (null when the curve could not be fit at all).
    pub point_jump_bps: Option<f64>,
    pub point_a_fast_bps: Option<f64>,
    pub point_a_slow_bps: Option<f64>,
    pub point_tau_fast_s: Option<f64>,
    pub point_tau_slow_s: Option<f64>,
    pub point_r_squared: Option<f64>,
    /// Joint parameter region from the within-draw fits
    /// ((2.5%, 97.5%) percentiles over all draws, stable or not).
    pub tau_fast_region_s: Option<(f64, f64)>,
    pub tau_slow_region_s: Option<(f64, f64)>,
    pub a_fast_region_bps: Option<(f64, f64)>,
    pub a_slow_region_bps: Option<(f64, f64)>,
    pub jump_region_bps: Option<(f64, f64)>,
    pub n_draw_fits: usize,
    pub stable_draws: usize,
    pub stable_share: f64,
    /// "unidentified/unstable" unless the point fit and the draws
    /// jointly pass the pre-stated stability criteria.
    pub status: String,
    pub criteria: Vec<String>,
    pub amplitude_policy: String,
    pub promotion_note: String,
}

/// Both Phase-3 cross-checks for one session.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionCrosschecks {
    pub var: VarCrosscheck,
    pub two_component: TwoComponentCrosscheck,
}

/// Report-level cross-check block: the methods and restraints
/// that govern the per-session `crosschecks` blocks. It carries
/// no pooled VAR result and no two-component parameter of any
/// kind.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ReportCrosschecks {
    pub restraints: Vec<String>,
    pub var_method: String,
    pub two_component_method: String,
    pub agreement_criterion: String,
}

fn var_unavailable(reason: &str, horizons_s: &[f64], lp: &[f64]) -> VarCrosscheck {
    let h = horizons_s.len();
    VarCrosscheck {
        available: false,
        unavailable_reason: Some(reason.to_string()),
        variables: VAR_VARIABLES.iter().map(|s| s.to_string()).collect(),
        returns_definition:
            "simple one-bar mid-price return in basis points, the same return definition as the local projections"
                .to_string(),
        estimation: "per-session reduced-form VAR, equation-by-equation OLS on standardized variables".to_string(),
        lag_criterion: format!(
            "BIC over lag orders 1..={VAR_MAX_LAG} on a common sample: ln det Sigma_ML + (p d^2 + d) ln(T) / T, d = 4"
        ),
        lag_range: (1, VAR_MAX_LAG),
        bic_by_lag: vec![None; VAR_MAX_LAG],
        chosen_lag: None,
        n_obs: 0,
        horizons_s: horizons_s.to_vec(),
        lp_response_bps: lp.to_vec(),
        ofi_first_irf_bps: vec![f64::NAN; h],
        ofi_first_ci_bps: vec![(f64::NAN, f64::NAN); h],
        returns_first_irf_bps: vec![f64::NAN; h],
        returns_first_ci_bps: vec![(f64::NAN, f64::NAN); h],
        diff_ofi_first_minus_lp_bps: vec![f64::NAN; h],
        diff_returns_first_minus_lp_bps: vec![f64::NAN; h],
        lp_inside_var_band: vec![false; h],
        var_inside_lp_band: vec![false; h],
        horizons_agreeing_ofi_first: 0,
        horizons_agreeing_returns_first: 0,
        max_abs_ordering_difference_bps: f64::NAN,
        mean_abs_ordering_difference_bps: f64::NAN,
        n_bootstrap_draws: 0,
        bootstrap:
            "residual bootstrap: centred residuals resampled with replacement, series regenerated within each original segment, VAR re-estimated at the chosen lag"
                .to_string(),
        restraint:
            "reduced-form specification cross-check only; not a Hasbrouck information-share estimate (single venue, single price series, so that estimand is unavailable); no information share is computed or implied; orthogonalisation is an ordering assumption, not identification"
                .to_string(),
    }
}

/// Compute the VAR cross-check for one session.
#[allow(clippy::too_many_arguments)]
fn compute_var_crosscheck(
    series: &BarSeries,
    spread: &[f64],
    imbalance: &[f64],
    horizons_s: &[f64],
    horizons_bars: &[usize],
    lp: &[f64],
    lp_ci: &[(f64, f64)],
    cfg: &SpeedConfig,
) -> VarCrosscheck {
    let Some(data) = build_var_data(series, spread, imbalance) else {
        return var_unavailable(
            "the session does not provide enough usable bars with return and liquidity variation for a VAR",
            horizons_s,
            lp,
        );
    };
    let mut bics: Vec<Option<f64>> = Vec::new();
    let mut fits: Vec<Option<VarFit>> = Vec::new();
    for p in 1..=VAR_MAX_LAG {
        let f = fit_var(&data, p);
        bics.push(f.as_ref().map(|x| x.bic));
        fits.push(f);
    }
    let mut chosen: Option<usize> = None;
    for (i, b) in bics.iter().enumerate() {
        if let Some(b) = b {
            if chosen.is_none_or(|c| *b < bics[c].unwrap()) {
                chosen = Some(i);
            }
        }
    }
    let Some(ci) = chosen else {
        return var_unavailable(
            "every candidate VAR lag order had a singular design or a non-positive residual determinant on this session",
            horizons_s,
            lp,
        );
    };
    let fit = fits
        .into_iter()
        .nth(ci)
        .flatten()
        .expect("chosen fit exists");
    let p = ci + 1;
    let (Some(irf_a), Some(irf_b)) = (
        var_cumulative_irf(&fit, &data.sd, &VAR_ORDERING_OFI_FIRST, horizons_bars),
        var_cumulative_irf(&fit, &data.sd, &VAR_ORDERING_RET_FIRST, horizons_bars),
    ) else {
        return var_unavailable(
            "the residual covariance at the chosen lag is not positive definite, so no Cholesky responses can be computed",
            horizons_s,
            lp,
        );
    };
    let resamples = cfg.sensitivity_resamples.min(VAR_BOOTSTRAP_RESAMPLES);
    let (draws_a, draws_b) = var_bootstrap_irfs(
        &data,
        &fit,
        horizons_bars,
        resamples,
        cfg.seed ^ 0x0A12_0000_0000_0004,
    );
    let h_len = horizons_s.len();
    let ci_a = draws_ci(&draws_a, h_len);
    let ci_b = draws_ci(&draws_b, h_len);
    let agree = |irf: &[f64], ci: &[(f64, f64)]| -> (Vec<bool>, Vec<bool>, usize) {
        let mut lp_in = Vec::with_capacity(h_len);
        let mut var_in = Vec::with_capacity(h_len);
        let mut count = 0;
        for k in 0..h_len {
            let a = lp[k] >= ci[k].0 && lp[k] <= ci[k].1;
            let b = irf[k] >= lp_ci[k].0 && irf[k] <= lp_ci[k].1;
            if a && b {
                count += 1;
            }
            lp_in.push(a);
            var_in.push(b);
        }
        (lp_in, var_in, count)
    };
    let (lp_in_a, var_in_a, agree_a) = agree(&irf_a, &ci_a);
    let (_, _, agree_b) = agree(&irf_b, &ci_b);
    let ord_diffs: Vec<f64> = irf_a
        .iter()
        .zip(irf_b.iter())
        .map(|(a, b)| (a - b).abs())
        .collect();
    let mut out = var_unavailable("", horizons_s, lp);
    out.available = true;
    out.unavailable_reason = None;
    out.bic_by_lag = bics;
    out.chosen_lag = Some(p);
    out.n_obs = fit.n_obs;
    out.ofi_first_irf_bps = irf_a.clone();
    out.ofi_first_ci_bps = ci_a;
    out.returns_first_irf_bps = irf_b.clone();
    out.returns_first_ci_bps = ci_b;
    out.diff_ofi_first_minus_lp_bps = irf_a.iter().zip(lp.iter()).map(|(v, l)| v - l).collect();
    out.diff_returns_first_minus_lp_bps = irf_b.iter().zip(lp.iter()).map(|(v, l)| v - l).collect();
    out.lp_inside_var_band = lp_in_a;
    out.var_inside_lp_band = var_in_a;
    out.horizons_agreeing_ofi_first = agree_a;
    out.horizons_agreeing_returns_first = agree_b;
    out.max_abs_ordering_difference_bps = ord_diffs.iter().copied().fold(0.0, f64::max);
    out.mean_abs_ordering_difference_bps = if ord_diffs.is_empty() {
        f64::NAN
    } else {
        ord_diffs.iter().sum::<f64>() / ord_diffs.len() as f64
    };
    out.n_bootstrap_draws = draws_a.len();
    out
}

/// Summarize the two-component cross-check for one session from
/// the point curve and the within-draw fits collected by the
/// headline bootstrap (the same draws as the single-component
/// fit; no second bootstrap).
fn summarize_two_component(
    horizons_s: &[f64],
    curve: &[f64],
    draw_fits: &[TwoCompFit],
) -> TwoComponentCrosscheck {
    let point = fit_two_component_raw(horizons_s, curve);
    let region = |get: &dyn Fn(&TwoCompFit) -> f64| -> Option<(f64, f64)> {
        if draw_fits.len() >= 10 {
            Some(percentile_ci(
                &draw_fits.iter().map(get).collect::<Vec<_>>(),
            ))
        } else {
            None
        }
    };
    let stable_draws = draw_fits.iter().filter(|f| two_component_stable(f)).count();
    let stable_share = if draw_fits.is_empty() {
        0.0
    } else {
        stable_draws as f64 / draw_fits.len() as f64
    };
    let grid = two_tau_grid();
    let region_bounded = |r: Option<(f64, f64)>| -> bool {
        match r {
            Some((lo, hi)) => {
                lo > grid[1] && hi < grid[grid.len() - 2] && lo > 0.0 && hi / lo < 4.0
            }
            None => false,
        }
    };
    let tau_f_region = region(&|f| f.tau_fast_s);
    let tau_s_region = region(&|f| f.tau_slow_s);
    let stable = point.map(|f| two_component_stable(&f)).unwrap_or(false)
        && stable_share >= 0.5
        && region_bounded(tau_f_region)
        && region_bounded(tau_s_region);
    TwoComponentCrosscheck {
        timing_convention:
            "inclusive (baseline is the close of the bar before the shock bar), the same convention as the headline single-component fit"
                .to_string(),
        point_jump_bps: point.map(|f| f.jump_bps),
        point_a_fast_bps: point.map(|f| f.a_fast_bps),
        point_a_slow_bps: point.map(|f| f.a_slow_bps),
        point_tau_fast_s: point.map(|f| f.tau_fast_s),
        point_tau_slow_s: point.map(|f| f.tau_slow_s),
        point_r_squared: point.map(|f| f.r_squared),
        tau_fast_region_s: tau_f_region,
        tau_slow_region_s: tau_s_region,
        a_fast_region_bps: region(&|f| f.a_fast_bps),
        a_slow_region_bps: region(&|f| f.a_slow_bps),
        jump_region_bps: region(&|f| f.jump_bps),
        n_draw_fits: draw_fits.len(),
        stable_draws,
        stable_share,
        status: if stable {
            "stable in the bootstrap draws under the pre-stated criteria; still not promoted, because no held-out confirmation on independent sessions exists"
                .to_string()
        } else {
            "unidentified/unstable".to_string()
        },
        criteria: vec![
            "a draw fit is stable only if its R^2 reaches the headline bar (0.85), both taus are interior to the two-component search grid by at least one grid step, tau_slow / tau_fast >= 3, both amplitudes are positive, and the total long-run response J + Af + As is positive".to_string(),
            "a session is labelled stable only if its point fit passes the same criteria, at least half of its bootstrap draws are stable, and both tau percentile regions are bounded (inside the grid interior, ratio of ends below 4)".to_string(),
        ],
        amplitude_policy:
            "amplitudes are fitted unconstrained and may be negative: a negative amplitude describes reversal or overshoot, not an adjustment component, so any fit carrying one fails the stability criteria and cannot be read as two adjustment speeds; this caveat applies wherever a negative amplitude appears in this block"
                .to_string(),
        promotion_note:
            "no tau_f, tau_s or derived half-life from this model is reported as a result anywhere: these values live only in this per-session cross-check block, and the session is labelled unidentified/unstable wherever the draws say so; held-out confirmation on independent sessions does not exist yet"
                .to_string(),
    }
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
    /// Resample pool behind the headline bootstrap intervals.
    /// When it is degenerate the headline bootstrap intervals
    /// above are suppressed (null in the serialized report) and
    /// no fit is promoted from those draws.
    pub resample_pool: ResamplePool,
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
    // ---- v2 extension: robustness variants ----
    /// Robustness variants under the same frozen protocol, when
    /// the session's recorded quotes (and trades, where a variant
    /// needs them) were supplied to the analysis.
    pub variants: Option<SessionVariants>,
    // ---- v2 extension: cross-checks ----
    /// The reduced-form VAR and two-component cross-checks, when
    /// the session's recorded quotes were supplied to the
    /// analysis. The two-component parameters live only inside
    /// this block.
    pub crosschecks: Option<SessionCrosschecks>,
}

/// Analyze one session's bar series. Returns None when the
/// session is too short to support the longest horizon with a
/// usable number of origins, or carries no OFI variation.
fn analyze_series(
    session: &str,
    series: &BarSeries,
    span_s: f64,
    cfg: &SpeedConfig,
    variant_inputs: Option<&VariantInputs>,
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
        true,
    );
    let h_len = horizons_s.len();
    // Degenerate-pool suppression: when the headline pool has
    // too few distinct block starts, every interval computed from
    // those draws is suppressed (NaN, serialized as null) and no
    // fit is promoted from them below; the point curves and the
    // HAC intervals, which do not come from the resamples, stand.
    let headline_degenerate = boot.pool.degenerate;
    let nan_ci = vec![(f64::NAN, f64::NAN); h_len];
    let response_ci_bps = if headline_degenerate {
        nan_ci.clone()
    } else {
        draws_ci(&boot.incl, h_len)
    };
    let post_response_ci_bps = if headline_degenerate {
        nan_ci.clone()
    } else {
        draws_ci(&boot.post, h_len)
    };
    let response_band = if headline_degenerate {
        nan_ci.clone()
    } else {
        simultaneous_band(&curve, &boot.incl)
    };
    let post_band = if headline_degenerate {
        nan_ci.clone()
    } else {
        simultaneous_band(&post_curve, &boot.post)
    };
    let innov_ci = if headline_degenerate {
        nan_ci.clone()
    } else {
        draws_ci(&boot.innov, h_len)
    };
    let innov_band = if headline_degenerate {
        nan_ci.clone()
    } else {
        simultaneous_band(&innov_curve, &boot.innov)
    };
    let n_draws = boot.incl.len().max(1);
    let bootstrap_fit_share = if headline_degenerate {
        0.0
    } else {
        boot.fits_pass.len() as f64 / n_draws as f64
    };
    let empty_draws: &[Vec<f64>] = &[];
    let profile = build_profile(
        &horizons_s,
        &curve,
        &response_ci_bps,
        if headline_degenerate {
            empty_draws
        } else {
            &boot.incl
        },
        &post_curve,
        &post_response_ci_bps,
        if headline_degenerate {
            empty_draws
        } else {
            &boot.post
        },
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
    let cov = if headline_degenerate {
        None
    } else {
        draws_covariance(&boot.incl, h_len)
    };
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
    let (fit, fit_failure) = if headline_degenerate {
        (
            None,
            Some(format!(
                "{}; no tau is reported from a degenerate resample pool",
                boot.pool
                    .suppression_reason
                    .clone()
                    .unwrap_or_else(|| "degenerate resample pool".to_string())
            )),
        )
    } else {
        match raw_fit {
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
        }
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
    let (tau_ci, half_ci, rinf_ci, jump_ci) = if !headline_degenerate && boot.fits_pass.len() >= 100
    {
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
            let ci300 = if headline_degenerate {
                None
            } else {
                response_ci_bps.last().copied()
            };
            sensitivity.push(BlockSensitivity {
                block_len_s: len_s as f64,
                feasible: feasible && !headline_degenerate,
                reason: if headline_degenerate {
                    boot.pool.suppression_reason.clone()
                } else if feasible {
                    None
                } else {
                    Some(
                        "headline length retained for comparability; this session's segments support it only in reduced form"
                            .to_string(),
                    )
                },
                n_resamples: if headline_degenerate {
                    0
                } else {
                    boot.incl.len()
                },
                response_ci_300_bps: ci300,
                ci_width_300_bps: ci300.map(|(lo, hi)| hi - lo),
                mean_ci_width_bps: if headline_degenerate {
                    None
                } else {
                    Some(
                        response_ci_bps
                            .iter()
                            .map(|(lo, hi)| hi - lo)
                            .sum::<f64>()
                            / h_len as f64,
                    )
                },
                resample_pool: Some(boot.pool.clone()),
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
                resample_pool: None,
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
                false,
            );
            let alt_degenerate = alt.pool.degenerate;
            let alt_ci = draws_ci(&alt.incl, h_len);
            let ci300 = if alt_degenerate {
                None
            } else {
                alt_ci.last().copied()
            };
            sensitivity.push(BlockSensitivity {
                block_len_s: len_s as f64,
                feasible: !alt_degenerate,
                reason: if alt_degenerate {
                    alt.pool.suppression_reason.clone()
                } else {
                    None
                },
                n_resamples: if alt_degenerate { 0 } else { alt.incl.len() },
                response_ci_300_bps: ci300,
                ci_width_300_bps: ci300.map(|(lo, hi)| hi - lo),
                mean_ci_width_bps: if alt_degenerate {
                    None
                } else {
                    Some(alt_ci.iter().map(|(lo, hi)| hi - lo).sum::<f64>() / h_len as f64)
                },
                resample_pool: Some(alt.pool.clone()),
            });
        }
    }
    let variants = variant_inputs
        .map(|inp| compute_variants(series, inp, &horizons_bars, &horizons_s, ofi_sd, cfg));
    let crosschecks = variant_inputs.map(|inp| SessionCrosschecks {
        var: compute_var_crosscheck(
            series,
            &inp.spread,
            &inp.imbalance,
            &horizons_s,
            &horizons_bars,
            &curve,
            &response_ci_bps,
            cfg,
        ),
        two_component: summarize_two_component(&horizons_s, &curve, &boot.two_fits_all),
    });
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
        resample_pool: boot.pool.clone(),
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
        variants,
        crosschecks,
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
        false,
    );
    let ext_degenerate = boot.pool.degenerate;
    let ci = if ext_degenerate {
        vec![(f64::NAN, f64::NAN); keep]
    } else {
        draws_ci(&boot.incl, keep)
    };
    let cov = if ext_degenerate {
        None
    } else {
        draws_covariance(&boot.incl, keep)
    };
    let weight = cov.as_ref().and_then(|c| invert_matrix(c));
    let fit = fit_exponential_w(&horizons_s[..keep], &curve[..keep], weight.as_deref())
        .or_else(|| fit_exponential_raw(&horizons_s[..keep], &curve[..keep]));
    let (_, _, flat) = profile_loss(&horizons_s[..keep], &curve[..keep], weight.as_deref());
    let (fit_tau, fit_jump, fit_rinf, fit_r2, verdict) = if ext_degenerate {
        (
            None,
            None,
            None,
            fit.map(|f| f.r_squared),
            boot.pool
                .suppression_reason
                .clone()
                .unwrap_or_else(|| "degenerate resample pool".to_string()),
        )
    } else {
        match fit {
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
        }
    };
    Some(ExtendedResponse {
        horizons_s: horizons_s[..keep].to_vec(),
        response_bps: curve[..keep].to_vec(),
        response_ci_bps: ci,
        post_response_bps: post_curve[..keep].to_vec(),
        n_origins: counts[..keep].to_vec(),
        block_len_s: 300.0,
        n_resamples: if ext_degenerate { 0 } else { boot.incl.len() },
        fit_tau_s: fit_tau,
        fit_jump_bps: fit_jump,
        fit_r_inf_bps: fit_rinf,
        fit_r_squared: fit_r2,
        fit_verdict: verdict,
        profile_flat_tau_range_s: flat,
        resample_pool: boot.pool.clone(),
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
    analyze_session_speed_with_trades(session, quotes, None, cfg)
}

/// Analyze one session's quotes and (when available) its recorded
/// trades: the trades feed the trade-price, activity and
/// volume-clock variants; every other variant needs only quotes.
pub fn analyze_session_speed_with_trades(
    session: &str,
    quotes: &[Quote],
    trades: Option<&[Trade]>,
    cfg: &SpeedConfig,
) -> Option<SessionSpeed> {
    if quotes.is_empty() {
        return None;
    }
    let series = build_bar_series(quotes, cfg.bar_ms);
    let span_s = (quotes.last().unwrap().time_ms - quotes.first().unwrap().time_ms) as f64 / 1000.0;
    let inputs = build_variant_inputs(quotes, trades, cfg.bar_ms);
    analyze_series(session, &series, span_s, cfg, Some(&inputs))
}

/// Analyze one session's bar series directly, without the
/// quote-level robustness variants and cross-checks (those blocks
/// never feed the headline curves, intervals, or fit). This is the
/// entry point the power/coverage simulation harness in
/// `examples/speed_power_sim.rs` runs: it exercises the same
/// local-projection, segment-bootstrap, and jump-term fit code as
/// `analyze_session_speed`, on synthetic bar series whose true
/// response is known by construction.
pub fn analyze_bar_series(
    session: &str,
    series: &BarSeries,
    span_s: f64,
    cfg: &SpeedConfig,
) -> Option<SessionSpeed> {
    analyze_series(session, series, span_s, cfg, None)
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
    // ---- v2 extension: robustness variants ----
    /// Report-level variants block: the pooled six-session
    /// description and the prespecified rules every per-session
    /// variant follows. Per-session variant results live on each
    /// session's `variants` field.
    pub variants: ReportVariants,
    // ---- v2 extension: cross-checks ----
    /// Report-level cross-check block: methods and restraints for
    /// the per-session `crosschecks` blocks. It carries no pooled
    /// VAR result and no two-component parameter.
    pub crosschecks: ReportCrosschecks,
}

impl SpeedReport {
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("speed report serializes")
    }
}

/// Aggregate per-session results into the report.
pub fn summarize_speed(sessions: Vec<SessionSpeed>, cfg: &SpeedConfig) -> SpeedReport {
    let summary = summarize(&sessions);
    let variants = report_variants(&sessions);
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
        "Bootstrap intervals are reported only where the ladder's resample pool \
         is non-degenerate: each session and ladder reports its pool of distinct \
         block starts and the distinct blocks actually drawn, and a pool below \
         20 distinct starts has its intervals suppressed with the reason \
         recorded in its resample_pool block, instead of a narrow interval \
         being published behind a full resample count."
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
            meaning, and the inclusive point estimates are the v1 estimates. \
            Extension within v2 (robustness variants): each session gains a \
            `variants` block (gap and censoring report with quote-age classes, \
            microprice and trade-price curves, raw-OFI curves, the \
            largest-OFI-burst event study, the volume-clock curve, pre-shock \
            state strata, and frequency/event-time support), and the report \
            gains a top-level `variants` block with the pooled six-session \
            description and the prespecified variant rules. Every pre-existing \
            v2 field and value is unchanged by the extension. \
            Extension within v2 (cross-checks): each session gains a \
            `crosschecks` block (the reduced-form VAR cross-check with its \
            per-horizon comparison against the local-projection curve, and \
            the two-component adjustment fit's parameter regions and \
            stability label), and the report gains a top-level `crosschecks` \
            block with the governing methods and restraints. Every \
            pre-existing field and value is unchanged by this extension \
            either, and no two-component parameter appears in the summary. \
            v3 corrects the segment bootstrap's resampling: segment choice \
            is two-stage and segment-fair (a segment is chosen with weight \
            max(segment length, drawn block length), then a uniform valid \
            start inside it, with a segment shorter than the block drawn \
            whole), so every fully-observed segment is represented in \
            proportion to its length instead of segments shorter than the \
            block being excluded and longer segments being weighted by \
            their valid-start counts. Each session gains a `resample_pool` \
            block (headline ladder), each block-sensitivity entry gains \
            one, and the extended ladder gains one: the drawn block \
            length, the number of distinct block starts in the pool, the \
            number of distinct blocks actually drawn, and the resamples \
            attempted. A ladder whose pool has fewer than 20 distinct \
            starts is degenerate: its bootstrap intervals are suppressed \
            (null), its `n_resamples` reads 0, no fit is promoted from \
            its draws, and the pool block records the reason. Point \
            estimates, HAC intervals, and every non-bootstrap field keep \
            their v2 meanings; bootstrap intervals and fit outcomes are \
            regenerated under the corrected resampling with the same \
            published seed, so their values move where the corrected \
            resample distribution moves them."
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
                that a max-horizon window can lie wholly inside one block; segment \
                choice is two-stage and segment-fair: a segment is chosen with \
                weight max(segment length, drawn block length), then a uniform \
                valid start inside it, and a segment shorter than the block is \
                drawn whole, so each segment's expected share of the resampled \
                bars equals its share of the observed bars; a ladder whose \
                block-start pool has fewer than 20 distinct starts is degenerate \
                and its intervals are suppressed with the reason recorded"
                .to_string(),
        },
        variants,
        crosschecks: ReportCrosschecks {
            restraints: vec![
                "The VAR block is a reduced-form specification cross-check only; it is NOT a Hasbrouck information-share estimate (single venue, single price series — that estimand is unavailable); no information share is computed or implied anywhere; orthogonalisation is an ordering assumption, not identification.".to_string(),
                "No tau_f, tau_s or derived half-life from the two-component model is reported as a result anywhere: those values live only in the per-session crosschecks blocks, labelled unidentified/unstable wherever the bootstrap draws say so, and the report summary carries no two-component field.".to_string(),
            ],
            var_method: format!(
                "per-session reduced-form VAR on one-second bars over the simple mid-price return in bps, per-bar OFI per session SD, the quoted spread in bps of the mid, and book imbalance (the spread and imbalance as computed by the bar builder); lag order chosen by BIC over 1..={VAR_MAX_LAG} on a common sample, never hand-tuned; cumulative price response to a one-SD OFI shock on the local-projection horizon ladder under two Cholesky orderings (OFI ordered first; returns ordered first); bands from a {VAR_BOOTSTRAP_RESAMPLES}-draw residual bootstrap with the lag held at the chosen order, seeded from the published seed"
            ),
            two_component_method:
                "R(h) = J + Af (1 - exp(-h / tau_f)) + As (1 - exp(-h / tau_s)) on the inclusive curve, fitted unweighted on the point curve and inside every headline segment-bootstrap draw (the same draws as the single-component fit; no second bootstrap); joint parameter regions are (2.5%, 97.5%) percentiles over all draw fits, and the stability label follows criteria stated in the per-session block before any outcome"
                    .to_string(),
            agreement_criterion:
                "a horizon agrees (per ordering) when the local-projection estimate lies inside the VAR bootstrap band AND the VAR estimate lies inside the local-projection pointwise bootstrap interval; disagreements are reported per horizon, not smoothed over"
                    .to_string(),
        },
    }
}

// ------------------------------------------------------------------
// Robustness variants (schema v2 extension): every variant below
// runs under the same frozen protocol as the headline estimator —
// the same sessions, the same horizon ladder, the same inclusive
// timing convention unless stated otherwise, the same censoring
// rule (a window touching an unobserved bar is dropped), and the
// same segment-respecting bootstrap design where intervals are
// reported (variant intervals use the sensitivity resample count
// and a seed derived from the published seed, stated in the
// report). The headline curves are never recomputed differently:
// variants sit next to them, including where they come out null.
// ------------------------------------------------------------------

/// Prespecified quantile of |OFI| that defines a burst event.
/// Stated here, before any result: a burst is an origin bar whose
/// |OFI| reaches the session's 95th percentile of |OFI| over
/// observed bars.
pub const BURST_QUANTILE: f64 = 0.95;

/// Refractory window, in bars, for the burst event study: after
/// an accepted event, candidate bursts within this many bars are
/// skipped (first-event-wins), so event windows at the longest
/// horizon never overlap.
pub const BURST_REFRACTORY_BARS: usize = 300;

/// Minimum clean windows per horizon for a burst mean to be
/// reported at that horizon.
pub const BURST_MIN_WINDOWS: usize = 10;

/// Minimum usable origins at the longest horizon for a stratum
/// to be reported; strata below it are suppressed with a reason.
pub const STRATUM_MIN_ORIGINS_AT_MAX_HORIZON: usize = 100;

/// Trailing window, in bars, for the pre-shock activity
/// covariate (measured strictly before the origin bar).
pub const STRATA_ACTIVITY_WINDOW_BARS: usize = 60;

/// Trailing window, in bars, for the pre-shock spread covariate.
pub const STRATA_SPREAD_WINDOW_BARS: usize = 10;

/// Quote updates per event-time bar in the frequency variant.
pub const EVENT_TIME_QUOTES_PER_BAR: usize = 10;

/// Quote-age classification of an unobserved bar, from the
/// recorded `bar_status` quote age at the bar's end.
pub fn classify_quote_age(quote_age_ms: i64) -> &'static str {
    if quote_age_ms <= 2_000 {
        "valid_no_update"
    } else if quote_age_ms <= 30_000 {
        "stale_quote"
    } else {
        "confirmed_feed_loss"
    }
}

/// (bucket last-trade price, bucket summed quote OFI, bucket
/// volume) on the volume clock.
pub type VolumeClockSeries = (Vec<f64>, Vec<f64>, f64);

/// Per-bar inputs the headline bar series discards but the
/// variants need, aligned to the same bar grid as `BarSeries`.
#[derive(Debug, Clone)]
pub struct VariantInputs {
    pub quote_age_ms: Vec<i64>,
    pub micro: Vec<f64>,
    pub spread: Vec<f64>,
    /// Book imbalance per bar, as computed by `ofi_bars` (the
    /// liquidity series the VAR cross-check uses alongside the
    /// spread).
    pub imbalance: Vec<f64>,
    pub quotes_in_bar: Vec<usize>,
    pub trade_count: Vec<usize>,
    pub trades_supplied: bool,
    pub trade_price: Option<Vec<f64>>,
    /// (bucket last-trade price, bucket summed quote OFI,
    /// bucket volume) on the volume clock, when trades support it.
    pub volume_clock: Option<VolumeClockSeries>,
    pub volume_clock_reason: Option<String>,
    pub half_second_series: Option<BarSeries>,
    pub event_series: Option<BarSeries>,
    pub n_quotes: usize,
    pub median_interquote_ms: f64,
    pub p90_interquote_ms: f64,
    pub mean_quotes_per_second: f64,
}

/// Equal-volume bucket series from the crate's bucket machinery:
/// buckets are built with `volume_buckets_sized` at a bucket
/// volume of total traded volume / `target_buckets`, the bucket
/// price is the last trade price in the bucket, and the bucket
/// regressor is the sum of quote OFI events whose timestamps fall
/// inside the bucket's trade-time interval. Returns (prices,
/// ofis, bucket_vol).
pub fn build_volume_clock_series(
    trades: &[Trade],
    quotes: &[Quote],
    target_buckets: usize,
) -> Option<VolumeClockSeries> {
    if trades.is_empty() || quotes.is_empty() || target_buckets == 0 {
        return None;
    }
    let total: f64 = trades.iter().map(|t| t.qty).sum();
    if total <= 0.0 {
        return None;
    }
    let bucket_vol = total / target_buckets as f64;
    // Bucket boundaries (end time, last trade price) with the
    // same proportional trade splitting as volume_buckets_sized.
    let mut sorted: Vec<&Trade> = trades.iter().collect();
    sorted.sort_by_key(|t| t.time_ms);
    let mut ends: Vec<(i64, f64)> = Vec::new();
    let mut filled = 0.0f64;
    for t in &sorted {
        let mut remaining = t.qty;
        while remaining > 1e-12 {
            let take = (bucket_vol - filled).min(remaining);
            filled += take;
            remaining -= take;
            if filled >= bucket_vol - 1e-9 {
                ends.push((t.time_ms, t.price));
                filled = 0.0;
            }
        }
    }
    if ends.len() < 30 {
        return None;
    }
    // Quote OFI events into buckets by timestamp.
    let events = ofi_events(quotes);
    let mut ofi = vec![0.0f64; ends.len()];
    for (q, e) in quotes.iter().zip(events.iter()) {
        let idx = ends.partition_point(|(end, _)| *end < q.time_ms);
        if idx < ends.len() {
            ofi[idx] += e;
        }
    }
    let price: Vec<f64> = ends.iter().map(|(_, p)| *p).collect();
    Some((price, ofi, bucket_vol))
}

/// Build the variant inputs from one session's recorded quotes
/// and (when available) trades, on the `bar_ms` grid.
pub fn build_variant_inputs(
    quotes: &[Quote],
    trades: Option<&[Trade]>,
    bar_ms: i64,
) -> VariantInputs {
    let events = ofi_events(quotes);
    let bars = ofi_bars(quotes, &events, bar_ms);
    let status = bar_status(quotes, bar_ms);
    let n = bars.len();
    // Last quote per bar, carried forward, for the microprice.
    let mut micro = vec![f64::NAN; n];
    let mut quotes_in_bar = vec![0usize; n];
    if !quotes.is_empty() && bar_ms > 0 {
        let t0 = quotes[0].time_ms / bar_ms * bar_ms;
        let micro_of = |q: &Quote| -> f64 {
            let tot = q.bid_sz + q.ask_sz;
            if tot > 0.0 {
                (q.bid_px * q.ask_sz + q.ask_px * q.bid_sz) / tot
            } else {
                q.mid()
            }
        };
        let mut last_in_bar: Vec<Option<f64>> = vec![None; n];
        for q in quotes {
            let idx = ((q.time_ms - t0) / bar_ms) as usize;
            if idx < n {
                quotes_in_bar[idx] += 1;
                last_in_bar[idx] = Some(micro_of(q));
            }
        }
        let mut carry = micro_of(&quotes[0]);
        for i in 0..n {
            if let Some(m) = last_in_bar[i] {
                carry = m;
            }
            micro[i] = carry;
        }
    }
    // Trade price / counts per bar (last trade in the bar,
    // carried forward; seeded with the bar mid before the first
    // trade in the session).
    let (trade_price, trade_count, trades_supplied) = match trades {
        Some(ts) if !ts.is_empty() && !quotes.is_empty() && bar_ms > 0 => {
            let t0 = quotes[0].time_ms / bar_ms * bar_ms;
            let mut px: Vec<Option<f64>> = vec![None; n];
            let mut cnt = vec![0usize; n];
            for t in ts {
                let idx = ((t.time_ms - t0) / bar_ms) as usize;
                if idx < n {
                    px[idx] = Some(t.price);
                    cnt[idx] += 1;
                }
            }
            let mut out = Vec::with_capacity(n);
            let mut carry: Option<f64> = None;
            for i in 0..n {
                if let Some(p) = px[i] {
                    carry = Some(p);
                }
                out.push(carry.unwrap_or(bars[i].2));
            }
            (Some(out), cnt, true)
        }
        _ => (None, vec![0usize; n], false),
    };
    let (volume_clock, volume_clock_reason) = match trades {
        Some(ts) if trades_supplied => match build_volume_clock_series(ts, quotes, n.max(1)) {
            Some(v) => (Some(v), None),
            None => (
                None,
                Some(
                    "the recorded trades do not fill enough equal-volume buckets"
                        .to_string(),
                ),
            ),
        },
        _ => (
            None,
            Some(
                "no recorded trades were supplied for this session, so the volume clock cannot be built"
                    .to_string(),
            ),
        ),
    };
    let half_second_series = if bar_ms == 1_000 {
        Some(build_bar_series(quotes, 500))
    } else {
        None
    };
    let event_series = {
        let mut mid = Vec::new();
        let mut ofi = Vec::new();
        let mut i = 0usize;
        while i < quotes.len() {
            let j = (i + EVENT_TIME_QUOTES_PER_BAR).min(quotes.len());
            mid.push(quotes[j - 1].mid());
            ofi.push(events[i..j].iter().sum());
            i = j;
        }
        if mid.len() >= 30 {
            Some(BarSeries {
                observed: vec![true; mid.len()],
                mid,
                ofi,
            })
        } else {
            None
        }
    };
    let mut diffs: Vec<i64> = quotes
        .windows(2)
        .map(|w| w[1].time_ms - w[0].time_ms)
        .filter(|d| *d > 0)
        .collect();
    diffs.sort_unstable();
    let pct = |p: f64| -> f64 {
        if diffs.is_empty() {
            f64::NAN
        } else {
            diffs[((p * diffs.len() as f64) as usize).min(diffs.len() - 1)] as f64
        }
    };
    let span_s = if quotes.len() >= 2 {
        (quotes.last().unwrap().time_ms - quotes.first().unwrap().time_ms) as f64 / 1000.0
    } else {
        f64::NAN
    };
    VariantInputs {
        quote_age_ms: status.iter().map(|s| s.quote_age_ms).collect(),
        micro,
        spread: bars.iter().map(|b| b.3).collect(),
        imbalance: bars.iter().map(|b| b.4).collect(),
        quotes_in_bar,
        trade_count,
        trades_supplied,
        trade_price,
        volume_clock,
        volume_clock_reason,
        half_second_series,
        event_series,
        n_quotes: quotes.len(),
        median_interquote_ms: pct(0.5),
        p90_interquote_ms: pct(0.9),
        mean_quotes_per_second: if span_s > 0.0 {
            quotes.len() as f64 / span_s
        } else {
            f64::NAN
        },
    }
}

/// Counts of unobserved bars per quote-age class, plus the
/// distinct unobserved stretches.
pub fn gap_classification(
    observed: &[bool],
    quote_age_ms: &[i64],
) -> (GapClassCounts, Vec<GapStretch>) {
    let mut counts = GapClassCounts {
        valid_no_update: 0,
        stale_quote: 0,
        confirmed_feed_loss: 0,
    };
    let mut stretches = Vec::new();
    let mut i = 0usize;
    while i < observed.len() {
        if observed[i] {
            i += 1;
            continue;
        }
        let start = i;
        let mut max_age = 0i64;
        while i < observed.len() && !observed[i] {
            let age = quote_age_ms.get(i).copied().unwrap_or(0);
            max_age = max_age.max(age);
            match classify_quote_age(age) {
                "valid_no_update" => counts.valid_no_update += 1,
                "stale_quote" => counts.stale_quote += 1,
                _ => counts.confirmed_feed_loss += 1,
            }
            i += 1;
        }
        stretches.push(GapStretch {
            start_bar: start,
            len_bars: i - start,
            max_quote_age_ms: max_age,
            class: classify_quote_age(max_age).to_string(),
        });
    }
    (counts, stretches)
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct GapClassCounts {
    pub valid_no_update: usize,
    pub stale_quote: usize,
    pub confirmed_feed_loss: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct GapStretch {
    pub start_bar: usize,
    pub len_bars: usize,
    pub max_quote_age_ms: i64,
    pub class: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CensorBin {
    pub covariate: String,
    pub bin: usize,
    pub lo: f64,
    pub hi: f64,
    pub n_origins: usize,
    pub n_censored: usize,
    pub censored_share: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SelectionSensitivity {
    pub horizon_s: f64,
    pub observed_slope_bps: f64,
    pub n_usable: usize,
    pub n_censored: usize,
    pub required_censored_slope_bps: f64,
    pub required_multiple_of_observed: f64,
    pub statement: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct GapReport {
    pub n_unobserved_bars: usize,
    pub class_counts: GapClassCounts,
    pub classification_rule: String,
    pub stretches: Vec<GapStretch>,
    pub n_stretches: usize,
    pub longest_stretch_bars: usize,
    pub longest_stretch_share_of_unobserved: f64,
    pub bars_outside_longest_stretch: usize,
    pub longest_stretch_note: String,
    /// Censoring tabulation at the longest horizon, by quartile
    /// bins of each covariate (bins are fit on all potential
    /// origins; the covariate is measured strictly before the
    /// origin except |OFI|, which is the origin bar's own shock).
    pub censoring_bins: Vec<CensorBin>,
    pub n_potential_origins_h0: usize,
    pub n_censored_h0: usize,
    pub n_potential_origins_max_h: usize,
    pub n_censored_max_h: usize,
    pub selection_sensitivity: SelectionSensitivity,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PriceVariants {
    pub micro_response_bps: Vec<f64>,
    pub micro_response_ci_bps: Vec<(f64, f64)>,
    pub micro_n_origins: Vec<usize>,
    pub micro_minus_mid_bps: Vec<f64>,
    pub trade_response_bps: Option<Vec<f64>>,
    pub trade_response_ci_bps: Option<Vec<(f64, f64)>>,
    pub trade_n_origins: Option<Vec<usize>>,
    pub trade_minus_mid_bps: Option<Vec<f64>>,
    pub trade_unavailable_reason: Option<String>,
    pub max_abs_micro_minus_mid_bps: f64,
    pub max_abs_trade_minus_mid_bps: Option<f64>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RawOfiVariant {
    pub response_bps_per_ofi_unit: Vec<f64>,
    pub response_ci_bps_per_ofi_unit: Vec<(f64, f64)>,
    pub n_origins: Vec<usize>,
    pub post_response_bps_per_ofi_unit: Vec<f64>,
    pub conversion_note: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BurstStudy {
    pub threshold_quantile: f64,
    pub threshold_abs_ofi: f64,
    pub refractory_bars: usize,
    pub overlap_rule: String,
    pub label: String,
    pub n_candidate_events: usize,
    pub n_events_retained: usize,
    pub horizons_s: Vec<f64>,
    pub mean_signed_response_bps: Vec<f64>,
    pub n_clean_windows: Vec<usize>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct VolumeClockVariant {
    pub bucket_volume: f64,
    pub n_buckets: usize,
    pub bucket_ofi_sd: f64,
    pub horizons_buckets: Vec<f64>,
    pub response_bps: Vec<f64>,
    pub response_ci_bps: Vec<(f64, f64)>,
    pub n_origins: Vec<usize>,
    pub estimand_note: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Stratum {
    pub label: String,
    pub response_bps: Vec<f64>,
    pub n_origins: Vec<usize>,
    pub reported: bool,
    pub suppression_reason: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Stratification {
    pub dimension: String,
    pub covariate_definition: String,
    pub tercile_edges: (f64, f64),
    pub min_origins_at_max_horizon: usize,
    pub strata: Vec<Stratum>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct FinerBarsVariant {
    pub bar_ms: i64,
    pub observed_share: f64,
    pub horizons_s: Vec<f64>,
    pub response_bps: Vec<f64>,
    pub n_origins: Vec<usize>,
    pub note: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct EventTimeVariant {
    pub quotes_per_bar: usize,
    pub n_bars: usize,
    pub horizons_event_bars: Vec<f64>,
    pub response_bps: Vec<f64>,
    pub response_ci_bps: Vec<(f64, f64)>,
    pub n_origins: Vec<usize>,
    pub note: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct FrequencySupport {
    pub n_quotes: usize,
    pub median_interquote_ms: f64,
    pub p90_interquote_ms: f64,
    pub mean_quotes_per_second: f64,
    pub finest_full_ladder_bar_ms: i64,
    pub statement: String,
    pub half_second: Option<FinerBarsVariant>,
    pub event_time: Option<EventTimeVariant>,
    pub event_time_unavailable_reason: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionVariants {
    pub gap: GapReport,
    pub prices: PriceVariants,
    pub raw_ofi: RawOfiVariant,
    pub bursts: BurstStudy,
    pub volume_clock: Option<VolumeClockVariant>,
    pub volume_clock_unavailable_reason: Option<String>,
    pub strata: Vec<Stratification>,
    pub frequency: FrequencySupport,
}

/// Burst event selection under the prespecified rule: candidates
/// are observed bars with |OFI| at or above `threshold`; scanning
/// in time order, a candidate is accepted only if it is at least
/// `refractory` bars after the last accepted event
/// (first-event-wins, so accepted windows at the longest horizon
/// never overlap).
pub fn select_burst_events(
    ofi: &[f64],
    observed: &[bool],
    threshold: f64,
    refractory: usize,
) -> Vec<usize> {
    let mut events = Vec::new();
    let mut last: Option<usize> = None;
    for t in 1..ofi.len() {
        if !observed[t] || ofi[t].abs() < threshold {
            continue;
        }
        if let Some(l) = last {
            if t < l + refractory.max(1) {
                continue;
            }
        }
        events.push(t);
        last = Some(t);
    }
    events
}

/// Variant bootstrap: one segment-respecting resample set (same
/// design and headline block length as the main bootstrap, seed
/// derived from the published seed) carrying the microprice,
/// trade-price and raw-OFI curves together, so their intervals
/// come from the same draws as each other.
#[allow(clippy::too_many_arguments)]
/// Draws produced by one variant bootstrap run.
struct VariantDraws {
    micro: Vec<Vec<f64>>,
    trade: Option<Vec<Vec<f64>>>,
    raw: Vec<Vec<f64>>,
}

fn variant_bootstrap(
    series: &BarSeries,
    micro: &[f64],
    trade: Option<&[f64]>,
    horizons_bars: &[usize],
    resamples: usize,
    seed: u64,
) -> VariantDraws {
    let segments = segments_of(&series.observed);
    let target: usize = series.observed.iter().filter(|o| **o).count();
    let longest = segments.iter().map(|(a, b)| b - a).max().unwrap_or(0);
    let max_h = horizons_bars.iter().copied().max().unwrap_or(0);
    let block = (300usize + max_h).min(longest);
    let mut micro_draws = Vec::new();
    let mut trade_draws: Option<Vec<Vec<f64>>> = trade.map(|_| Vec::new());
    let mut raw_draws = Vec::new();
    if target == 0 || block <= max_h || resamples == 0 {
        return VariantDraws {
            micro: micro_draws,
            trade: trade_draws,
            raw: raw_draws,
        };
    }
    let mut rng = SplitMix64(seed);
    for _ in 0..resamples {
        let (idxs, breaks, _) = resample_indices(&segments, block, target, &mut rng);
        if idxs.len() < 30 {
            continue;
        }
        let obs = vec![true; idxs.len()];
        let all_ok = vec![true; idxs.len()];
        let reg: Vec<f64> = idxs.iter().map(|&i| series.ofi[i]).collect();
        let sd = population_sd(&reg);
        if sd <= 0.0 || !sd.is_finite() {
            continue;
        }
        let pick = |price: &[f64]| idxs.iter().map(|&i| price[i]).collect::<Vec<f64>>();
        let (m, _, _, _) = lp_pair(
            &pick(micro),
            &reg,
            &obs,
            &breaks,
            &all_ok,
            horizons_bars,
            sd,
            false,
        );
        if m.iter().any(|v| !v.is_finite()) {
            continue;
        }
        micro_draws.push(m);
        let (r, _, _, _) = lp_pair(
            &pick(&series.mid),
            &reg,
            &obs,
            &breaks,
            &all_ok,
            horizons_bars,
            1.0,
            false,
        );
        if r.iter().all(|v| v.is_finite()) {
            raw_draws.push(r);
        }
        if let (Some(tp), Some(draws)) = (trade, trade_draws.as_mut()) {
            let (c, _, _, _) = lp_pair(
                &pick(tp),
                &reg,
                &obs,
                &breaks,
                &all_ok,
                horizons_bars,
                sd,
                false,
            );
            if c.iter().all(|v| v.is_finite()) {
                draws.push(c);
            }
        }
    }
    VariantDraws {
        micro: micro_draws,
        trade: trade_draws,
        raw: raw_draws,
    }
}

fn quartile_edges(values: &[f64]) -> (f64, f64, f64, f64) {
    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if sorted.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    let q = |p: f64| sorted[((p * sorted.len() as f64) as usize).min(sorted.len() - 1)];
    (sorted[0], q(0.25), q(0.5), q(0.75))
}

#[allow(clippy::too_many_arguments)]
fn compute_variants(
    series: &BarSeries,
    inputs: &VariantInputs,
    horizons_bars: &[usize],
    horizons_s: &[f64],
    ofi_sd: f64,
    cfg: &SpeedConfig,
) -> SessionVariants {
    let n = series.mid.len();
    let max_h = *horizons_bars.last().unwrap_or(&0);
    let no_breaks = vec![false; n];
    let all_ok = vec![true; n];
    let idx = WindowIndex::new(&series.observed, &no_breaks);

    // ---- (a) gap and censoring ----
    let (class_counts, stretches) = gap_classification(&series.observed, &inputs.quote_age_ms);
    let n_unobserved = n - series.observed.iter().filter(|o| **o).count();
    let longest_stretch = stretches.iter().map(|s| s.len_bars).max().unwrap_or(0);
    // Covariates per potential origin.
    let activity_src: &[usize] = if inputs.trades_supplied {
        &inputs.trade_count
    } else {
        &inputs.quotes_in_bar
    };
    let mut trailing_activity = vec![f64::NAN; n];
    let mut trailing_spread = vec![f64::NAN; n];
    let mut prev_vol = vec![f64::NAN; n];
    for t in 1..n {
        let a0 = t.saturating_sub(STRATA_ACTIVITY_WINDOW_BARS);
        trailing_activity[t] = activity_src[a0..t].iter().sum::<usize>() as f64;
        let s0 = t.saturating_sub(STRATA_SPREAD_WINDOW_BARS);
        if t > s0 {
            trailing_spread[t] = inputs.spread[s0..t].iter().sum::<f64>() / (t - s0) as f64;
        }
        let v0 = t.saturating_sub(61);
        let mut rets = Vec::new();
        for i in (v0 + 1)..t {
            if series.observed[i] && series.observed[i - 1] && series.mid[i - 1] > 0.0 {
                rets.push((series.mid[i] / series.mid[i - 1] - 1.0) * 1e4);
            }
        }
        if rets.len() >= 10 {
            prev_vol[t] = population_sd(&rets);
        }
    }
    let cov_of = |t: usize, kind: usize| -> f64 {
        match kind {
            0 => trailing_activity[t],
            1 => trailing_spread[t],
            2 => prev_vol[t],
            _ => series.ofi[t].abs(),
        }
    };
    let cov_names = [
        "trailing_activity_60_bars",
        "trailing_spread_10_bars",
        "preceding_volatility_60_bars_bps",
        "abs_ofi_at_origin",
    ];
    let potential = |h: usize| -> Vec<usize> { (1..n).filter(|&t| t + h < n).collect() };
    let (pot0, pot_max) = (potential(0), potential(max_h));
    let censored = |t: usize, h: usize| !idx.clean(t - 1, t + h);
    let n_cens0 = pot0.iter().filter(|&&t| censored(t, 0)).count();
    let n_cens_max = pot_max.iter().filter(|&&t| censored(t, max_h)).count();
    let mut censoring_bins = Vec::new();
    for (kind, cov_name) in cov_names.iter().enumerate() {
        let vals: Vec<f64> = pot_max.iter().map(|&t| cov_of(t, kind)).collect();
        let (lo, q1, q2, q3) = quartile_edges(&vals);
        let edges = [lo, q1, q2, q3, f64::INFINITY];
        for b in 0..4 {
            let in_bin: Vec<usize> = pot_max
                .iter()
                .copied()
                .filter(|&t| {
                    let v = cov_of(t, kind);
                    v.is_finite() && v >= edges[b] && (b == 3 || v < edges[b + 1])
                })
                .collect();
            let nc = in_bin.iter().filter(|&&t| censored(t, max_h)).count();
            censoring_bins.push(CensorBin {
                covariate: cov_name.to_string(),
                bin: b,
                lo: edges[b],
                hi: if b == 3 {
                    vals.iter()
                        .copied()
                        .filter(|v| v.is_finite())
                        .fold(f64::NEG_INFINITY, f64::max)
                } else {
                    edges[b + 1]
                },
                n_origins: in_bin.len(),
                n_censored: nc,
                censored_share: if in_bin.is_empty() {
                    f64::NAN
                } else {
                    nc as f64 / in_bin.len() as f64
                },
            });
        }
    }
    let slope_max = {
        // Inclusive slope at the longest horizon (recomputed from
        // the headline curve would need it passed in; recompute
        // cheaply from the series to keep this function local).
        let (c, _, _, _) = lp_pair(
            &series.mid,
            &series.ofi,
            &series.observed,
            &no_breaks,
            &all_ok,
            &[max_h],
            ofi_sd,
            false,
        );
        c[0]
    };
    let n_usable_max = pot_max.len() - n_cens_max;
    let required = if n_cens_max > 0 && slope_max.is_finite() {
        -slope_max * n_usable_max as f64 / n_cens_max as f64
    } else {
        f64::NAN
    };
    let gap = GapReport {
        n_unobserved_bars: n_unobserved,
        class_counts,
        classification_rule: "an unobserved bar is valid_no_update when the most recent quote was at most 2 s old at the bar end (the book simply did not change inside that second), stale_quote when the newest quote was over 2 s and at most 30 s old, and confirmed_feed_loss when the newest quote was over 30 s old at the bar end; a stretch takes the class of its oldest quote age".to_string(),
        stretches,
        n_stretches: 0, // filled below from stretches length
        longest_stretch_bars: longest_stretch,
        longest_stretch_share_of_unobserved: if n_unobserved > 0 {
            longest_stretch as f64 / n_unobserved as f64
        } else {
            f64::NAN
        },
        bars_outside_longest_stretch: n_unobserved - longest_stretch,
        longest_stretch_note: String::new(),
        censoring_bins,
        n_potential_origins_h0: pot0.len(),
        n_censored_h0: n_cens0,
        n_potential_origins_max_h: pot_max.len(),
        n_censored_max_h: n_cens_max,
        selection_sensitivity: SelectionSensitivity {
            horizon_s: horizons_s.last().copied().unwrap_or(f64::NAN),
            observed_slope_bps: slope_max,
            n_usable: n_usable_max,
            n_censored: n_cens_max,
            required_censored_slope_bps: required,
            required_multiple_of_observed: if slope_max.abs() > 1e-12 {
                (required / slope_max).abs()
            } else {
                f64::NAN
            },
            statement: String::new(),
        },
    };
    let mut gap = gap;
    gap.n_stretches = gap.stretches.len();
    gap.longest_stretch_note = if gap.n_unobserved_bars == 0 {
        "this session has no unobserved bars".to_string()
    } else {
        format!(
        "the {} unobserved bars fall in {} distinct stretches; the longest is {} bars and the remaining {} bars fall in the other {} stretches",
        gap.n_unobserved_bars,
        gap.n_stretches,
        gap.longest_stretch_bars,
        gap.bars_outside_longest_stretch,
        gap.n_stretches.saturating_sub(1),
        )
    };
    gap.selection_sensitivity.statement = if gap.selection_sensitivity.n_censored == 0 {
        "no origins are censored at any horizon in this session (the session has no unobserved bars), so censoring selection cannot affect this session's estimates".to_string()
    } else {
        format!(
        "at the {} s horizon {} of {} potential origins are censored. Under an equal-regressor-variance approximation, the censored origins would need an average response of {:+.3} bps per SD ({:.2}x the usable origins' {:+.3} bps, and of the opposite sign) for the pooled response at this horizon to be zero; selection weaker than that changes the size of the estimate, not its sign. This is a bound on what censoring alone could do, not evidence about what the censored origins actually did.",
        horizons_s.last().copied().unwrap_or(f64::NAN),
        gap.selection_sensitivity.n_censored,
        gap.n_potential_origins_max_h,
        gap.selection_sensitivity.required_censored_slope_bps,
        gap.selection_sensitivity.required_multiple_of_observed,
        gap.selection_sensitivity.observed_slope_bps,
        )
    };

    // ---- (b) price variants and (c) raw OFI ----
    let (micro_curve, micro_counts, _, _) = lp_pair(
        &inputs.micro,
        &series.ofi,
        &series.observed,
        &no_breaks,
        &all_ok,
        horizons_bars,
        ofi_sd,
        false,
    );
    let (trade_curve, trade_counts) = match &inputs.trade_price {
        Some(tp) => {
            let (c, k, _, _) = lp_pair(
                tp,
                &series.ofi,
                &series.observed,
                &no_breaks,
                &all_ok,
                horizons_bars,
                ofi_sd,
                false,
            );
            (Some(c), Some(k))
        }
        None => (None, None),
    };
    let (raw_curve, raw_counts, raw_post, _) = lp_pair(
        &series.mid,
        &series.ofi,
        &series.observed,
        &no_breaks,
        &all_ok,
        horizons_bars,
        1.0,
        true,
    );
    let VariantDraws {
        micro: micro_draws,
        trade: trade_draws,
        raw: raw_draws,
    } = variant_bootstrap(
        series,
        &inputs.micro,
        inputs.trade_price.as_deref(),
        horizons_bars,
        cfg.sensitivity_resamples,
        cfg.seed ^ 0x5EED_0000_0000_0001,
    );
    let h_len = horizons_bars.len();
    let mid_curve: Vec<f64> = {
        let (c, _, _, _) = lp_pair(
            &series.mid,
            &series.ofi,
            &series.observed,
            &no_breaks,
            &all_ok,
            horizons_bars,
            ofi_sd,
            false,
        );
        c
    };
    let diff =
        |a: &[f64]| -> Vec<f64> { a.iter().zip(mid_curve.iter()).map(|(x, m)| x - m).collect() };
    let max_abs = |a: &[f64]| {
        a.iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(0.0f64, |m, v| m.max(v.abs()))
    };
    let prices = PriceVariants {
        micro_minus_mid_bps: diff(&micro_curve),
        max_abs_micro_minus_mid_bps: max_abs(&diff(&micro_curve)),
        micro_response_ci_bps: draws_ci(&micro_draws, h_len),
        micro_response_bps: micro_curve,
        micro_n_origins: micro_counts,
        trade_minus_mid_bps: trade_curve.as_ref().map(|c| diff(c)),
        max_abs_trade_minus_mid_bps: trade_curve.as_ref().map(|c| max_abs(&diff(c))),
        trade_response_ci_bps: trade_draws.as_ref().map(|d| draws_ci(d, h_len)),
        trade_response_bps: trade_curve,
        trade_n_origins: trade_counts,
        trade_unavailable_reason: if inputs.trade_price.is_some() {
            None
        } else {
            Some("no recorded trades were supplied for this session, so no trade-price series can be built".to_string())
        },
    };
    let raw_ofi = RawOfiVariant {
        response_bps_per_ofi_unit: raw_curve,
        response_ci_bps_per_ofi_unit: draws_ci(&raw_draws, h_len),
        n_origins: raw_counts,
        post_response_bps_per_ofi_unit: raw_post,
        conversion_note: format!(
            "raw-OFI slopes are in basis points per one unit of per-bar OFI; the per-SD curve is this curve multiplied by the session OFI standard deviation ({ofi_sd:.4} here), so the two are the same estimates on different shock scales"
        ),
    };

    // ---- (d) burst event study ----
    let obs_abs: Vec<f64> = series
        .ofi
        .iter()
        .zip(series.observed.iter())
        .filter(|(_, o)| **o)
        .map(|(v, _)| v.abs())
        .collect();
    let threshold = percentile(&obs_abs, BURST_QUANTILE);
    let n_candidates = (1..n)
        .filter(|&t| series.observed[t] && series.ofi[t].abs() >= threshold)
        .count();
    let events = select_burst_events(
        &series.ofi,
        &series.observed,
        threshold,
        BURST_REFRACTORY_BARS,
    );
    let mut burst_means = Vec::with_capacity(h_len);
    let mut burst_counts = Vec::with_capacity(h_len);
    for &h in horizons_bars {
        let mut vals = Vec::new();
        for &t in &events {
            let end = t + h;
            if end < n && idx.clean(t - 1, end) && series.mid[t - 1] > 0.0 {
                vals.push(
                    series.ofi[t].signum() * (series.mid[end] / series.mid[t - 1] - 1.0) * 1e4,
                );
            }
        }
        burst_counts.push(vals.len());
        burst_means.push(if vals.len() >= BURST_MIN_WINDOWS {
            vals.iter().sum::<f64>() / vals.len() as f64
        } else {
            f64::NAN
        });
    }
    let bursts = BurstStudy {
        threshold_quantile: BURST_QUANTILE,
        threshold_abs_ofi: threshold,
        refractory_bars: BURST_REFRACTORY_BARS,
        overlap_rule: "first-event-wins with a 300-bar refractory window: after an accepted event, candidate bursts within the next 300 bars are skipped, so accepted event windows at the longest horizon never overlap".to_string(),
        label: "conditional result for extreme events only: this is the mean signed response around the largest OFI bursts, not the average response to a typical shock, and it must not be read as one".to_string(),
        n_candidate_events: n_candidates,
        n_events_retained: events.len(),
        horizons_s: horizons_s.to_vec(),
        mean_signed_response_bps: burst_means,
        n_clean_windows: burst_counts,
    };

    // ---- (e) volume clock ----
    let (volume_clock, volume_clock_reason) = match &inputs.volume_clock {
        Some((price, ofi, vol)) => {
            let bs = BarSeries {
                mid: price.clone(),
                ofi: ofi.clone(),
                observed: vec![true; price.len()],
            };
            let bsd = population_sd(&bs.ofi);
            if bsd > 0.0 && bs.mid.len() > max_h + 100 {
                let nb = bs.mid.len();
                let (c, k, _, _) = lp_pair(
                    &bs.mid,
                    &bs.ofi,
                    &bs.observed,
                    &vec![false; nb],
                    &vec![true; nb],
                    horizons_bars,
                    bsd,
                    false,
                );
                let boot = bootstrap_engine(
                    &bs,
                    None,
                    horizons_bars,
                    horizons_s,
                    300,
                    cfg.sensitivity_resamples,
                    cfg.seed ^ 0xB0C1_0000_0000_0002,
                    false,
                    false,
                    false,
                );
                (
                    Some(VolumeClockVariant {
                        bucket_volume: *vol,
                        n_buckets: nb,
                        bucket_ofi_sd: bsd,
                        horizons_buckets: horizons_s.to_vec(),
                        response_ci_bps: draws_ci(&boot.incl, h_len),
                        response_bps: c,
                        n_origins: k,
                        estimand_note: "separate volume-time estimand: horizons are equal-volume buckets, not seconds, and this curve is not a seconds-based speed and is not comparable rung-by-rung with the clock-time curves".to_string(),
                    }),
                    None,
                )
            } else {
                (
                    None,
                    Some(
                        "the volume-clock series is too short for the longest horizon".to_string(),
                    ),
                )
            }
        }
        None => (None, inputs.volume_clock_reason.clone()),
    };

    // ---- (g) pre-shock state stratification ----
    let mut strata = Vec::new();
    for kind in 0..2usize {
        let (dimension, definition, cov): (&str, String, Vec<f64>) = if kind == 0 {
            (
                "pre_shock_activity",
                format!(
                    "sum of {} in the {} bars strictly before the origin (trailing window ends at the bar before the shock bar)",
                    if inputs.trades_supplied { "trades" } else { "quotes" },
                    STRATA_ACTIVITY_WINDOW_BARS
                ),
                trailing_activity.clone(),
            )
        } else {
            (
                "pre_shock_spread",
                format!(
                    "mean quoted spread over the {} bars strictly before the origin",
                    STRATA_SPREAD_WINDOW_BARS
                ),
                trailing_spread.clone(),
            )
        };
        let finite: Vec<f64> = cov.iter().copied().filter(|v| v.is_finite()).collect();
        let mut sorted = finite.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let terc = |p: f64| -> f64 {
            if sorted.is_empty() {
                f64::NAN
            } else {
                sorted[((p * sorted.len() as f64) as usize).min(sorted.len() - 1)]
            }
        };
        let (e1, e2) = (terc(1.0 / 3.0), terc(2.0 / 3.0));
        let mut stratum_list = Vec::new();
        for (si, label) in ["low", "middle", "high"].iter().enumerate() {
            let ok: Vec<bool> = (0..n)
                .map(|t| {
                    let v = cov[t];
                    v.is_finite()
                        && match si {
                            0 => v <= e1,
                            1 => v > e1 && v <= e2,
                            _ => v > e2,
                        }
                })
                .collect();
            let (c, k, _, _) = lp_pair(
                &series.mid,
                &series.ofi,
                &series.observed,
                &no_breaks,
                &ok,
                horizons_bars,
                ofi_sd,
                false,
            );
            let reported = *k.last().unwrap_or(&0) >= STRATUM_MIN_ORIGINS_AT_MAX_HORIZON
                && k.iter().all(|&x| x >= 30);
            stratum_list.push(Stratum {
                label: label.to_string(),
                response_bps: if reported { c } else { Vec::new() },
                n_origins: k.clone(),
                reported,
                suppression_reason: if reported {
                    None
                } else {
                    Some(format!(
                        "suppressed: {} usable origins at the longest horizon, below the prespecified minimum of {}",
                        k.last().copied().unwrap_or(0),
                        STRATUM_MIN_ORIGINS_AT_MAX_HORIZON
                    ))
                },
            });
        }
        strata.push(Stratification {
            dimension: dimension.to_string(),
            covariate_definition: definition,
            tercile_edges: (e1, e2),
            min_origins_at_max_horizon: STRATUM_MIN_ORIGINS_AT_MAX_HORIZON,
            strata: stratum_list,
        });
    }

    // ---- (h) frequency support ----
    let half_second = inputs.half_second_series.as_ref().map(|hs| {
        let hn = hs.mid.len();
        let hsd = population_sd(
            &hs.ofi
                .iter()
                .zip(hs.observed.iter())
                .filter(|(_, o)| **o)
                .map(|(v, _)| *v)
                .collect::<Vec<_>>(),
        );
        let hbars: Vec<usize> = horizons_s.iter().map(|s| (s * 2.0) as usize).collect();
        let (c, k, _, _) = if hsd > 0.0 && hsd.is_finite() {
            lp_pair(&hs.mid, &hs.ofi, &hs.observed, &vec![false; hn], &vec![true; hn], &hbars, hsd, false)
        } else {
            (vec![f64::NAN; h_len], vec![0; h_len], Vec::new(), Vec::new())
        };
        let obs_share = hs.observed.iter().filter(|o| **o).count() as f64 / hn.max(1) as f64;
        FinerBarsVariant {
            bar_ms: 500,
            observed_share: obs_share,
            horizons_s: horizons_s.to_vec(),
            response_bps: c,
            n_origins: k,
            note: "500 ms bars formed from the same quote stream; a clean response window at horizon h needs 2h+1 consecutive observed half-second bars, and rungs with fewer than 30 usable origins are not estimated (null). This is a resolution sensitivity on the published one-second grid, not a replacement for it.".to_string(),
        }
    });
    let (event_time, event_reason) = match &inputs.event_series {
        Some(es) if es.mid.len() > max_h + 100 => {
            let en = es.mid.len();
            let esd = population_sd(&es.ofi);
            let (c, k, _, _) = lp_pair(&es.mid, &es.ofi, &es.observed, &vec![false; en], &vec![true; en], horizons_bars, esd, false);
            let boot = bootstrap_engine(es, None, horizons_bars, horizons_s, 300, cfg.sensitivity_resamples, cfg.seed ^ 0x5EED_0000_0000_0003, false, false, false);
            (
                Some(EventTimeVariant {
                    quotes_per_bar: EVENT_TIME_QUOTES_PER_BAR,
                    n_bars: en,
                    horizons_event_bars: horizons_s.to_vec(),
                    response_ci_bps: draws_ci(&boot.incl, h_len),
                    response_bps: c,
                    n_origins: k,
                    note: "event-time sensitivity: bars are blocks of 10 consecutive quote updates, so horizons are event-bar counts, not seconds; this checks whether the response shape is an artefact of the one-second clock grid, and its rungs are not seconds-based speeds".to_string(),
                }),
                None,
            )
        }
        _ => (
            None,
            Some("the session's quote stream does not produce enough event-time bars for the longest horizon".to_string()),
        ),
    };
    let frequency = FrequencySupport {
        n_quotes: inputs.n_quotes,
        median_interquote_ms: inputs.median_interquote_ms,
        p90_interquote_ms: inputs.p90_interquote_ms,
        mean_quotes_per_second: inputs.mean_quotes_per_second,
        finest_full_ladder_bar_ms: 500,
        statement: format!(
            "the recordings are quote-update streams (median inter-quote gap {:.0} ms, 90th percentile {:.0} ms, {:.1} quotes per second on average). A 500 ms grid is {:.1}% observed in this session and supports the full 0-300 s ladder under the censoring rule, so the half-second curve is reported as a resolution sensitivity and 500 ms is the finest fully supported clock resolution on these recordings; finer clock grids were not attempted, because at a {:.0} ms median inter-quote gap a 100 ms grid would leave most bars unobserved and collapse the clean-window counts. The published grid stays at one second, and an event-time variant ({} quote updates per bar) is reported alongside",
            inputs.median_interquote_ms,
            inputs.p90_interquote_ms,
            inputs.mean_quotes_per_second,
            inputs
                .half_second_series
                .as_ref()
                .map(|hs| {
                    100.0 * hs.observed.iter().filter(|o| **o).count() as f64
                        / hs.observed.len().max(1) as f64
                })
                .unwrap_or(f64::NAN),
            inputs.median_interquote_ms,
            EVENT_TIME_QUOTES_PER_BAR
        ),
        half_second,
        event_time,
        event_time_unavailable_reason: event_reason,
    };

    SessionVariants {
        gap,
        prices,
        raw_ofi,
        bursts,
        volume_clock,
        volume_clock_unavailable_reason: volume_clock_reason,
        strata,
        frequency,
    }
}

/// Pooled description over the sessions in the report: a
/// precision-weighted and an equal-weight mean curve, plus a
/// session-demeaned shape. This is a description of these
/// sessions only (six clusters); it carries no pooled tau, no
/// half-life, and no market-wide claim.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PooledDescription {
    pub n_sessions: usize,
    pub horizons_s: Vec<f64>,
    pub precision_weighted_response_bps: Vec<f64>,
    pub equal_weight_response_bps: Vec<f64>,
    pub session_demeaned_shape_bps: Vec<f64>,
    pub response_range_bps: Vec<(f64, f64)>,
    pub pooled_tau_reported: bool,
    pub label: String,
}

pub fn pooled_description(sessions: &[SessionSpeed]) -> PooledDescription {
    let horizons_s = sessions
        .first()
        .map(|s| s.horizons_s.clone())
        .unwrap_or_default();
    let h_len = horizons_s.len();
    let mut precision_weighted = vec![f64::NAN; h_len];
    let mut equal_weight = vec![f64::NAN; h_len];
    let mut demeaned = vec![f64::NAN; h_len];
    let mut range = vec![(f64::NAN, f64::NAN); h_len];
    for k in 0..h_len {
        let mut wsum = 0.0;
        let mut acc = 0.0;
        let mut plain = Vec::new();
        for s in sessions {
            let v = s.response_bps[k];
            if !v.is_finite() {
                continue;
            }
            plain.push(v);
            let (lo, hi) = s.response_ci_bps[k];
            let width = hi - lo;
            let w = if width > 0.0 && width.is_finite() {
                let sd = width / (2.0 * 1.96);
                1.0 / (sd * sd)
            } else {
                0.0
            };
            if w > 0.0 {
                wsum += w;
                acc += w * v;
            }
        }
        if wsum > 0.0 {
            precision_weighted[k] = acc / wsum;
        }
        if !plain.is_empty() {
            equal_weight[k] = plain.iter().sum::<f64>() / plain.len() as f64;
            range[k] = (
                plain.iter().copied().fold(f64::INFINITY, f64::min),
                plain.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            );
        }
        // Session-demeaned shape: remove each session's own mean
        // response across horizons, then average what is left.
        let mut dacc = 0.0;
        let mut dn = 0usize;
        for s in sessions {
            let finite: Vec<f64> = s
                .response_bps
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .collect();
            if finite.is_empty() || !s.response_bps[k].is_finite() {
                continue;
            }
            let mean = finite.iter().sum::<f64>() / finite.len() as f64;
            dacc += s.response_bps[k] - mean;
            dn += 1;
        }
        if dn > 0 {
            demeaned[k] = dacc / dn as f64;
        }
    }
    PooledDescription {
        n_sessions: sessions.len(),
        horizons_s,
        precision_weighted_response_bps: precision_weighted,
        equal_weight_response_bps: equal_weight,
        session_demeaned_shape_bps: demeaned,
        response_range_bps: range,
        pooled_tau_reported: false,
        label: "description of these six recorded sessions only (six clusters): a pooled summary of this sample, not a population or market-wide estimate, and no pooled tau or half-life is reported".to_string(),
    }
}

/// The report-level variants block (schema v2 extension).
#[derive(Debug, Clone, serde::Serialize)]
pub struct ReportVariants {
    pub pooled: PooledDescription,
    pub prespecified_rules: Vec<String>,
    pub notes: Vec<String>,
}

pub fn report_variants(sessions: &[SessionSpeed]) -> ReportVariants {
    ReportVariants {
        pooled: pooled_description(sessions),
        prespecified_rules: vec![
            format!("burst events: |OFI| at or above the session's {:.0}th percentile of |OFI| over observed bars, first-event-wins with a {}-bar refractory window, means suppressed below {} clean windows per horizon", BURST_QUANTILE * 100.0, BURST_REFRACTORY_BARS, BURST_MIN_WINDOWS),
            format!("strata: terciles of the pre-shock covariate per session, reported only with at least {} usable origins at the longest horizon and at least 30 at every horizon", STRATUM_MIN_ORIGINS_AT_MAX_HORIZON),
            "volume clock: equal-volume buckets at a bucket volume of total traded volume divided by the session's one-second bar count; horizons are bucket counts".to_string(),
            format!("event time: bars of {} consecutive quote updates; horizons are event-bar counts", EVENT_TIME_QUOTES_PER_BAR),
            "gap classes by quote age at the unobserved bar's end: at most 2 s valid_no_update, at most 30 s stale_quote, beyond that confirmed_feed_loss".to_string(),
        ],
        notes: vec![
            "Variants are published alongside the headline curves under the same frozen protocol (same sessions, horizons, timing conventions and censoring rule); variant intervals use the same segment-respecting bootstrap design at the sensitivity resample count with seeds derived from the published seed.".to_string(),
            "The burst event study is a conditional result for extreme events only. The volume-clock and event-time curves are separate estimands whose horizons are buckets and event bars, not seconds. The pooled block describes these six sessions only and reports no pooled tau or half-life.".to_string(),
        ],
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
            let (idxs, breaks, _) = resample_indices(&segments, 50, 800, &mut rng);
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
    fn short_segments_appear_in_resamples() {
        // Segments of 300, 50 and 600 bars with a drawn block of
        // 200: the 50-bar segment is shorter than the block, so a
        // pooled-valid-start scheme could never draw it at all.
        // Under the segment-fair scheme it is drawn whole, and the
        // pool counts it as the one chunk it offers.
        let segments = vec![(0, 300), (400, 450), (500, 1100)];
        assert_eq!(pool_distinct_starts(&segments, 200), 101 + 1 + 401);
        let mut rng = SplitMix64(7);
        let mut short_draws = 0usize;
        let mut total_draws = 0usize;
        for _ in 0..200 {
            let (_, _, drawn) = resample_indices(&segments, 200, 950, &mut rng);
            total_draws += drawn.len();
            short_draws += drawn.iter().filter(|&&d| d == (400, 50)).count();
        }
        assert!(total_draws > 0);
        assert!(
            short_draws >= 20,
            "the short segment was drawn {short_draws} times in 200 resamples; \
             it must be a regular part of the resample universe, not an exception"
        );
    }

    #[test]
    fn single_segment_resampling_matches_the_pooled_scheme() {
        // On one clean segment the segment-fair scheme must
        // reduce to the earlier pooled-start scheme draw for
        // draw (same seed, same stream), so clean sessions move
        // only by Monte Carlo reshuffling, never systematically.
        let segments = vec![(0, 2000)];
        let block = 300;
        let target = 1500;
        // The earlier scheme, reimplemented here as the oracle:
        // pool every valid start, draw uniformly from the pool.
        let mut starts: Vec<usize> = Vec::new();
        for &(a, b) in &segments {
            if b - a >= block {
                starts.extend(a..=b - block);
            }
        }
        let mut rng_old = SplitMix64(99);
        let mut old: Vec<usize> = Vec::new();
        while old.len() < target {
            let start = starts[rng_old.next_below(starts.len())];
            for j in 0..block {
                if old.len() >= target {
                    break;
                }
                old.push(start + j);
            }
        }
        let mut rng_new = SplitMix64(99);
        let (new, _, _) = resample_indices(&segments, block, target, &mut rng_new);
        assert_eq!(new, old);
        assert_eq!(pool_distinct_starts(&segments, block), 2000 - block + 1);
    }

    #[test]
    fn tiny_pool_extended_ladder_is_suppressed_not_narrow() {
        // Three clean segments of 1,800 bars each (5,400 observed
        // bars, so the extended ladder runs) separated by outages.
        // The extended drawn block is 1,800 bars, so each segment
        // offers exactly one distinct start: a pool of 3, far
        // below the minimum of 20. The extended intervals must
        // come out suppressed with the reason recorded, not as
        // narrow intervals behind a full resample count.
        let rho = (-1.0f64 / 120.0).exp();
        let base = synth_series(5_700, rho, 3.0);
        let mut observed = vec![true; 5_700];
        for slot in observed.iter_mut().take(1_950).skip(1_800) {
            *slot = false;
        }
        for slot in observed.iter_mut().take(3_900).skip(3_750) {
            *slot = false;
        }
        let series = BarSeries {
            mid: base.mid,
            ofi: base.ofi,
            observed,
        };
        let cfg = small_cfg();
        let res =
            analyze_bar_series("tiny-pool", &series, 5_699.0, &cfg).expect("session must analyze");
        let ext = res.extended.expect("extended ladder must run");
        assert!(ext.resample_pool.degenerate, "pool {:?}", ext.resample_pool);
        assert_eq!(ext.resample_pool.distinct_starts, 3);
        assert_eq!(ext.n_resamples, 0);
        assert!(
            ext.fit_verdict.contains("degenerate resample pool"),
            "verdict: {}",
            ext.fit_verdict
        );
        assert!(
            ext.response_ci_bps
                .iter()
                .all(|(lo, hi)| lo.is_nan() && hi.is_nan()),
            "suppressed intervals must be null, not narrow: {:?}",
            ext.response_ci_bps
        );
        assert!(ext.fit_tau_s.is_none());
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
        let res = analyze_series("flat", &series, 1798.0, &small_cfg(), None).expect("analysis");
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
    fn report_serializes_v3_with_provenance() {
        let rho = (-1.0f64 / 20.0).exp();
        let series = synth_series(1799, rho, 9.0);
        let cfg = small_cfg();
        let res = analyze_series("synth", &series, 1798.0, &cfg, None).expect("analysis");
        let report = summarize_speed(vec![res], &cfg);
        let v: serde_json::Value = serde_json::from_str(&report.to_json()).unwrap();
        assert_eq!(v["schema"], "toxicflow.info-speed/v3");
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
        // v3: the headline resample pool is published and, on this
        // single clean segment, non-degenerate with the pooled
        // scheme's start count (1799 - 600 + 1).
        assert_eq!(s0["resample_pool"]["distinct_starts"], 1200);
        assert_eq!(s0["resample_pool"]["degenerate"], false);
        assert!(
            s0["resample_pool"]["distinct_blocks_drawn"]
                .as_u64()
                .unwrap()
                > 0
        );
        assert!(s0["block_sensitivity"][1]["resample_pool"].is_object());
    }

    #[test]
    fn gap_classification_counts_known_stretches() {
        // Observed bars with three unobserved stretches: a fresh
        // 2-bar no-update stretch (ages 800, 1800 ms), a stale
        // 1-bar stretch (age 12 s), and a feed-loss stretch whose
        // ages run 45 s and 300 s.
        let observed = vec![
            true, true, false, false, true, false, true, false, false, true,
        ];
        let ages = vec![0, 0, 800, 1800, 0, 12_000, 0, 45_000, 300_000, 0];
        let (counts, stretches) = gap_classification(&observed, &ages);
        assert_eq!(counts.valid_no_update, 2);
        assert_eq!(counts.stale_quote, 1);
        assert_eq!(counts.confirmed_feed_loss, 2);
        assert_eq!(stretches.len(), 3);
        assert_eq!(stretches[0].len_bars, 2);
        assert_eq!(stretches[0].class, "valid_no_update");
        assert_eq!(stretches[1].class, "stale_quote");
        assert_eq!(stretches[2].len_bars, 2);
        assert_eq!(stretches[2].class, "confirmed_feed_loss");
        assert_eq!(classify_quote_age(2_000), "valid_no_update");
        assert_eq!(classify_quote_age(2_001), "stale_quote");
        assert_eq!(classify_quote_age(30_001), "confirmed_feed_loss");
    }

    #[test]
    fn microprice_construction_on_a_known_book() {
        // One bar, two quotes: the bar microprice is the last
        // quote's size-weighted price (bid weighted by ask size).
        let q = |t: i64, bp: f64, bq: f64, ap: f64, aq: f64| Quote {
            time_ms: t,
            bid_px: bp,
            bid_sz: bq,
            ask_px: ap,
            ask_sz: aq,
        };
        let quotes = vec![
            q(0, 100.0, 1.0, 101.0, 3.0),   // micro (100*3 + 101*1)/4 = 100.25
            q(500, 100.0, 3.0, 101.0, 1.0), // micro (100*1 + 101*3)/4 = 100.75
            q(3_000, 102.0, 1.0, 103.0, 1.0),
        ];
        let inputs = build_variant_inputs(&quotes, None, 1_000);
        assert_eq!(inputs.micro.len(), 4);
        assert!(
            (inputs.micro[0] - 100.75).abs() < 1e-12,
            "micro {}",
            inputs.micro[0]
        );
        // Unobserved bars carry the last observed microprice.
        assert!((inputs.micro[1] - 100.75).abs() < 1e-12);
        assert!((inputs.micro[2] - 100.75).abs() < 1e-12);
        assert!((inputs.micro[3] - 102.5).abs() < 1e-12);
        assert_eq!(inputs.quotes_in_bar, vec![2, 0, 0, 1]);
    }

    #[test]
    fn burst_selection_is_first_event_wins_with_refractory() {
        // Bursts at t = 10, 20, 400 with a 300-bar refractory:
        // t = 20 falls inside t = 10's window and is skipped.
        let n = 1_000;
        let mut ofi = vec![0.0; n];
        ofi[10] = 100.0;
        ofi[20] = -100.0;
        ofi[400] = 100.0;
        let observed = vec![true; n];
        let events = select_burst_events(&ofi, &observed, 50.0, 300);
        assert_eq!(events, vec![10, 400]);
        // A burst on an unobserved bar is never an event.
        let mut obs2 = observed.clone();
        obs2[400] = false;
        assert_eq!(select_burst_events(&ofi, &obs2, 50.0, 300), vec![10]);
    }

    #[test]
    fn volume_clock_bucket_invariants() {
        // 100 trades of 0.1 base units at alternating prices and
        // a steady quote stream: target 10 buckets of 1.0 each.
        let trades: Vec<Trade> = (0..100)
            .map(|i| Trade {
                time_ms: i * 100,
                price: 100.0 + (i % 2) as f64,
                qty: 0.1,
                is_buyer_maker: i % 2 == 0,
            })
            .collect();
        let quotes: Vec<Quote> = (0..100)
            .map(|i| Quote {
                time_ms: i * 100,
                bid_px: 99.9,
                bid_sz: 1.0 + (i % 3) as f64,
                ask_px: 100.1,
                ask_sz: 1.0,
            })
            .collect();
        let (price, ofi, vol) =
            build_volume_clock_series(&trades, &quotes, 50).expect("volume clock");
        assert_eq!(price.len(), 50);
        assert_eq!(ofi.len(), 50);
        assert!((vol - 0.2).abs() < 1e-9, "bucket vol {vol}");
        // Every bucket closes on the trade that filled it: bucket
        // k ends with trade 2k+1, whose price is 101 (odd index).
        assert!(price.iter().all(|p| (*p - 101.0).abs() < 1e-12));
        // Bucket OFI sums to the total quote OFI inside the
        // covered window.
        let events = ofi_events(&quotes);
        let total: f64 = events.iter().sum();
        let bucketed: f64 = ofi.iter().sum();
        assert!((bucketed - total).abs() < 1e-9, "{bucketed} vs {total}");
    }

    #[test]
    fn raw_ofi_curve_is_the_per_sd_curve_rescaled() {
        let rho = (-1.0f64 / 20.0).exp();
        let series = synth_series(3_000, rho, 9.0);
        let ofi_sd = population_sd(&series.ofi);
        let horizons_bars: Vec<usize> = SPEED_HORIZONS_S.iter().map(|s| *s as usize).collect();
        let n = series.mid.len();
        let (per_sd, _, _, _) = lp_pair(
            &series.mid,
            &series.ofi,
            &series.observed,
            &vec![false; n],
            &vec![true; n],
            &horizons_bars,
            ofi_sd,
            false,
        );
        let (raw, _, _, _) = lp_pair(
            &series.mid,
            &series.ofi,
            &series.observed,
            &vec![false; n],
            &vec![true; n],
            &horizons_bars,
            1.0,
            false,
        );
        for k in 0..horizons_bars.len() {
            assert!(
                (raw[k] * ofi_sd - per_sd[k]).abs() < 1e-9,
                "horizon {k}: raw {} x sd != per-SD {}",
                raw[k],
                per_sd[k]
            );
        }
    }

    #[test]
    fn variants_end_to_end_on_a_synthetic_session() {
        // 1,799 one-second quotes with wandering sizes (so OFI
        // varies) and matching trades: the full variants block
        // must be produced, with the gap report finding no
        // unobserved bars and the volume clock available.
        let mut s: u64 = 0xC0FFEE;
        let n = 1799;
        let mut price = 100_000.0f64;
        let mut quotes = Vec::with_capacity(n);
        let mut trades = Vec::with_capacity(n);
        for i in 0..n {
            let u = xorshift(&mut s) * 2.0 - 1.0;
            price *= 1.0 + u * 0.0002;
            quotes.push(Quote {
                time_ms: i as i64 * 1_000,
                bid_px: price - 0.05,
                bid_sz: 1.0 + (xorshift(&mut s) + 1.0) * 2.0,
                ask_px: price + 0.05,
                ask_sz: 1.0 + (xorshift(&mut s) + 1.0) * 2.0,
            });
            trades.push(Trade {
                time_ms: i as i64 * 1_000 + 500,
                price,
                qty: 0.01,
                is_buyer_maker: i % 2 == 0,
            });
        }
        let res = analyze_session_speed_with_trades(
            "synth-variants",
            &quotes,
            Some(&trades),
            &small_cfg(),
        )
        .expect("analysis");
        let v = res.variants.as_ref().expect("variants present");
        assert_eq!(v.gap.n_unobserved_bars, 0);
        assert_eq!(v.gap.n_stretches, 0);
        assert_eq!(v.gap.n_censored_h0, 0);
        assert!(v.prices.trade_response_bps.is_some());
        assert!(v.prices.micro_response_bps.iter().all(|x| x.is_finite()));
        assert!(v
            .raw_ofi
            .response_bps_per_ofi_unit
            .iter()
            .all(|x| x.is_finite()));
        assert!(v.bursts.threshold_abs_ofi > 0.0);
        assert!(
            v.volume_clock.is_some(),
            "volume clock: {:?}",
            v.volume_clock_unavailable_reason
        );
        assert_eq!(v.strata.len(), 2);
        assert!(v.frequency.half_second.is_some());
        // The report-level pooled block describes the session set
        // and never reports a pooled tau.
        let report = summarize_speed(vec![res], &small_cfg());
        assert_eq!(report.variants.pooled.n_sessions, 1);
        assert!(!report.variants.pooled.pooled_tau_reported);
        let j: serde_json::Value = serde_json::from_str(&report.to_json()).unwrap();
        assert!(j["variants"]["pooled"].is_object());
        assert!(j["sessions"][0]["variants"]["gap"].is_object());
        assert_eq!(
            j["sessions"][0]["variants"]["bursts"]["threshold_quantile"],
            0.95
        );
    }

    /// A planted VAR(1) in bar space: OFI (session-standardized)
    /// is AR(1) with phi = 0.6, returns load on lagged OFI with
    /// beta = 2 bps and on their own lag with alpha = 0.2, and
    /// spread / imbalance are exogenous noise. Returns (bar
    /// series, spread, imbalance).
    fn synth_var_series(n: usize, contemporaneous: f64) -> (BarSeries, Vec<f64>, Vec<f64>) {
        let mut s: u64 = 0x5EED_1234_5678_9ABC;
        let (phi, alpha, beta) = (0.6f64, 0.2f64, 2.0f64);
        let mut ofi_z = 0.0f64;
        let mut ret = 0.0f64;
        let mut imb = 0.0f64;
        let mut price = 100_000.0f64;
        let mut mid = Vec::with_capacity(n);
        let mut ofi = Vec::with_capacity(n);
        let mut spread = Vec::with_capacity(n);
        let mut imbalance = Vec::with_capacity(n);
        for _ in 0..n {
            // Unit-variance innovations (the raw xorshift draw is
            // uniform on (-1, 1), variance 1/3, hence the sqrt(3)).
            let e = (xorshift(&mut s) * 2.0 - 1.0) * 3.0f64.sqrt();
            let u = (xorshift(&mut s) * 2.0 - 1.0) * 0.5;
            // Returns load on the PREVIOUS bar's OFI (and, when
            // `contemporaneous` is nonzero, on this bar's OFI
            // innovation e, which is what correlates the two
            // series within the bar). ofi_z has SD
            // 1/sqrt(1-phi^2) = 1.25, so ofi_z / 1.25 is the
            // session-standardized OFI the estimator recovers.
            ret = alpha * ret + beta * (ofi_z / 1.25) + contemporaneous * e + u;
            ofi_z = phi * ofi_z + e;
            imb = 0.3 * imb + 0.5 * (xorshift(&mut s) * 2.0 - 1.0);
            price *= 1.0 + ret / 1e4;
            mid.push(price);
            ofi.push(ofi_z * 37.0);
            spread.push(price * 1e-4 * (1.0 + 0.1 * (xorshift(&mut s) * 2.0 - 1.0)));
            imbalance.push(imb);
        }
        (
            BarSeries {
                mid,
                ofi,
                observed: vec![true; n],
            },
            spread,
            imbalance,
        )
    }

    #[test]
    fn var_recovers_planted_var1_dynamics() {
        let (series, spread, imbalance) = synth_var_series(20_000, 0.0);
        let data = build_var_data(&series, &spread, &imbalance).expect("var data");
        // BIC over the bounded range must select the planted
        // order 1 (the criterion is the stated one; nothing is
        // hand-tuned to the projections, which are not involved).
        let bics: Vec<Option<f64>> = (1..=VAR_MAX_LAG)
            .map(|p| fit_var(&data, p).map(|f| f.bic))
            .collect();
        let chosen = bics
            .iter()
            .enumerate()
            .filter_map(|(i, b)| b.map(|v| (i + 1, v)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(p, _)| p);
        assert_eq!(chosen, Some(1), "BIC by lag: {bics:?}");
        let fit = fit_var(&data, 1).expect("VAR(1) fit");
        let horizons_bars: Vec<usize> = SPEED_HORIZONS_S.iter().map(|s| *s as usize).collect();
        let irf = var_cumulative_irf(&fit, &data.sd, &VAR_ORDERING_OFI_FIRST, &horizons_bars)
            .expect("irf");
        // Analytic cumulative response of the planted system to a
        // one-unit (one session SD) OFI shock: the standardized
        // OFI is AR(1) with phi and innovation SD sqrt(1-phi^2);
        // a one-unit level shock propagates as phi^k, and returns
        // follow ret_k = alpha ret_{k-1} + beta ofi_{k-1} with a
        // zero impact return (independent innovations).
        let (phi, alpha, beta) = (0.6f64, 0.2f64, 2.0f64);
        let analytic = |h_max: usize| -> f64 {
            let (mut cum, mut r, mut o) = (0.0, 0.0, 1.0);
            for _ in 0..=h_max {
                cum += r;
                let next_o = phi * o;
                r = alpha * r + beta * o;
                o = next_o;
            }
            cum
        };
        for (idx, h) in [(10usize, 10usize), (13, 60), (15, 300)] {
            let want = analytic(h);
            assert!(
                (irf[idx] - want).abs() < 1.0,
                "h={h}: fitted IRF {} vs analytic {want}",
                irf[idx]
            );
        }
        // With independent innovations the contemporaneous
        // response is zero under either ordering.
        assert!(irf[0].abs() < 0.35, "impact response {}", irf[0]);
    }

    #[test]
    fn var_ordering_changes_only_the_contemporaneous_split() {
        // Returns load on the OFI innovation itself, so the
        // within-bar split between flow and price is real in the
        // data and the ordering decides who is credited with it:
        // OFI-first must show a positive impact response, and
        // returns-first must show exactly none at h = 0 (the OFI
        // shock cannot move returns contemporaneously when returns
        // are ordered first), while both orderings agree closely
        // at long horizons.
        let (series, spread, imbalance) = synth_var_series(20_000, 1.5);
        let data = build_var_data(&series, &spread, &imbalance).expect("var data");
        let fit = fit_var(&data, 1).expect("VAR(1) fit");
        let horizons_bars: Vec<usize> = SPEED_HORIZONS_S.iter().map(|s| *s as usize).collect();
        let a = var_cumulative_irf(&fit, &data.sd, &VAR_ORDERING_OFI_FIRST, &horizons_bars)
            .expect("ofi-first irf");
        let b = var_cumulative_irf(&fit, &data.sd, &VAR_ORDERING_RET_FIRST, &horizons_bars)
            .expect("returns-first irf");
        assert!(a[0] > 0.3, "OFI-first impact {} should be positive", a[0]);
        assert!(
            b[0].abs() < 1e-9,
            "returns-first impact {} is zero by construction of the ordering",
            b[0]
        );
        // Both orderings find the lagged response (beta on lagged
        // OFI), so both cumulative curves are large at 300 s. They
        // do not fully converge in cumulative terms: the impact-bar
        // difference is retained by the cumulation and decays only
        // through the return's own alpha, so the OFI-first curve
        // stays above the returns-first one by roughly
        // a[0] / (1 - alpha). The ordering sensitivity is the
        // contemporaneous split, and the test pins its size.
        let last = horizons_bars.len() - 1;
        assert!(
            b[last] > 1.5,
            "returns-first long-horizon response {} should find the lagged response",
            b[last]
        );
        let gap = a[last] - b[last];
        let expected_gap = a[0] / (1.0 - 0.2);
        assert!(
            (gap - expected_gap).abs() < 0.45 * expected_gap,
            "cumulative ordering gap {gap} should sit near the impact gap cumulated through alpha: {expected_gap}"
        );
    }

    #[test]
    fn two_component_fit_recovers_a_planted_two_scale_curve() {
        // Exact planted curve: J = 1, Af = 2 at tau_f = 5 s,
        // As = 4 at tau_s = 120 s. The fit must recover the
        // parameters closely and pass the pre-stated stability
        // criteria (positive amplitudes, separated interior taus).
        let horizons_s: Vec<f64> = SPEED_HORIZONS_S.iter().map(|s| *s as f64).collect();
        let curve: Vec<f64> = horizons_s
            .iter()
            .map(|h| 1.0 + 2.0 * (1.0 - (-h / 5.0).exp()) + 4.0 * (1.0 - (-h / 120.0).exp()))
            .collect();
        let fit = fit_two_component_raw(&horizons_s, &curve).expect("two-component fit");
        assert!(fit.r_squared > 0.99, "R^2 {}", fit.r_squared);
        assert!((fit.jump_bps - 1.0).abs() < 0.4, "J {}", fit.jump_bps);
        assert!((fit.a_fast_bps - 2.0).abs() < 0.6, "Af {}", fit.a_fast_bps);
        assert!((fit.a_slow_bps - 4.0).abs() < 0.6, "As {}", fit.a_slow_bps);
        assert!(
            (fit.tau_fast_s - 5.0).abs() < 3.0,
            "tau_f {}",
            fit.tau_fast_s
        );
        assert!(
            (fit.tau_slow_s - 120.0).abs() < 45.0,
            "tau_s {}",
            fit.tau_slow_s
        );
        assert!(
            two_component_stable(&fit),
            "an exact, separated, positive two-scale curve must read as stable"
        );
    }

    #[test]
    fn two_component_declining_curve_is_labelled_unstable() {
        // Adversarial case: a response that declines steadily
        // from 60 to 300 s, the shape three recorded sessions
        // actually have. Representing it needs a negative slow
        // amplitude, which the pre-stated criteria refuse to call
        // stable, and the summary must label the session
        // unidentified/unstable rather than promoting anything.
        let horizons_s: Vec<f64> = SPEED_HORIZONS_S.iter().map(|s| *s as f64).collect();
        let curve: Vec<f64> = horizons_s
            .iter()
            .map(|h| 2.6 - 1.8 * (1.0 - (-h / 100.0).exp()))
            .collect();
        assert!(curve[10] > *curve.last().unwrap(), "curve declines");
        let fit = fit_two_component_raw(&horizons_s, &curve).expect("raw fit exists");
        assert!(
            !two_component_stable(&fit),
            "declining curve must not read as a stable two-component adjustment: {fit:?}"
        );
        let summary = summarize_two_component(&horizons_s, &curve, &[]);
        assert_eq!(summary.status, "unidentified/unstable");
        assert_eq!(summary.stable_share, 0.0);
    }

    #[test]
    fn crosschecks_guardrail_no_two_component_fields_in_summary() {
        // The guardrail, asserted on a full serialized report:
        // the summary block keeps exactly its pre-existing fields
        // (no two-component tau, half-life or amplitude of any
        // kind), while the per-session cross-checks block exists
        // and the report-level restraints travel with the numbers.
        let mut s: u64 = 0xBEEF;
        let n = 1799;
        let mut price = 100_000.0f64;
        let mut quotes = Vec::with_capacity(n);
        let mut trades = Vec::with_capacity(n);
        for i in 0..n {
            let u = xorshift(&mut s) * 2.0 - 1.0;
            price *= 1.0 + u * 0.0002;
            quotes.push(Quote {
                time_ms: i as i64 * 1_000,
                bid_px: price - 0.05,
                bid_sz: 1.0 + (xorshift(&mut s) + 1.0) * 2.0,
                ask_px: price + 0.05,
                ask_sz: 1.0 + (xorshift(&mut s) + 1.0) * 2.0,
            });
            trades.push(Trade {
                time_ms: i as i64 * 1_000 + 500,
                price,
                qty: 0.01,
                is_buyer_maker: i % 2 == 0,
            });
        }
        let res = analyze_session_speed_with_trades(
            "synth-guardrail",
            &quotes,
            Some(&trades),
            &small_cfg(),
        )
        .expect("analysis");
        let cc = res.crosschecks.as_ref().expect("crosschecks present");
        assert!(cc.var.available, "VAR: {:?}", cc.var.unavailable_reason);
        assert!(cc.var.chosen_lag.is_some());
        assert_eq!(cc.two_component.status, "unidentified/unstable");
        let report = summarize_speed(vec![res], &small_cfg());
        let v: serde_json::Value = serde_json::from_str(&report.to_json()).unwrap();
        let summary = v["summary"].as_object().unwrap();
        let mut keys: Vec<&str> = summary.keys().map(|k| k.as_str()).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            vec![
                "half_life_max_s",
                "half_life_mean_s",
                "half_life_median_s",
                "half_life_min_s",
                "half_life_std_s",
                "n_fit_ok",
                "n_sessions",
                "tau_max_s",
                "tau_mean_s",
                "tau_median_s",
                "tau_min_s",
                "tau_std_s",
            ],
            "the summary must keep exactly its pre-existing fields"
        );
        let summary_json = serde_json::to_string(&v["summary"]).unwrap();
        assert!(!summary_json.contains("tau_fast") && !summary_json.contains("tau_slow"));
        assert!(v["sessions"][0]["crosschecks"]["two_component"]["status"]
            .as_str()
            .unwrap()
            .contains("unidentified"));
        assert!(v["crosschecks"]["restraints"][0]
            .as_str()
            .unwrap()
            .contains("Hasbrouck"));
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
