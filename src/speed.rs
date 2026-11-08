//! Information incorporation speed: how fast the mid price finishes
//! adjusting after order flow arrives.
//!
//! The existing metrics say how much prices move with order flow
//! (Kyle's lambda, OFI, VPIN). This module adds the time dimension:
//! the shape of the cumulative mid-price response in the seconds and
//! minutes after a shock, and a single time constant summarizing it.
//!
//! Framing: order-flow shocks are the information-arrival proxy.
//! Following Hasbrouck, information is inferred from trades
//! themselves; the recorded sessions carry no news timestamps, so
//! nothing here identifies a news event or an informed trader. A
//! "shock" is one bar of order-flow imbalance (OFI, Cont-Kukanov-
//! Stoikov, as computed in `metrics`), scaled by its own session
//! standard deviation.
//!
//! Method: local projections. For each horizon h in a fixed ladder,
//! the cumulative mid return from just before the shock bar to h
//! bars after it is regressed on the standardized shock (one OLS
//! regression per horizon, intercept included). Local projections
//! are used instead of a VAR because the object of interest is the
//! whole response path rather than a one-step forecasting system:
//! each horizon gets its own regression, so the curve is not forced
//! through a VAR's implied dynamics, and the estimator is the same
//! per-bar OLS machinery the rest of the crate already uses. Origins
//! whose window touches a bar with no observed quote are censored,
//! the same gap discipline as the prediction dataset (N2): a feed
//! outage is never treated as continuous time.
//!
//! The cumulative curve is then fit with an exponential adjustment
//! path, R(h) = R_inf * (1 - exp(-h / tau)), by profiling tau over a
//! grid and solving for R_inf in closed form at each step. The fit
//! is reported only when it is actually good (R^2 at the published
//! threshold, positive long-run response, tau interior to the grid);
//! otherwise the empirical curve stands on its own and the report
//! says the fit failed, with the reason. Uncertainty uses the same
//! moving-block bootstrap conventions as the TF6 evaluation: circular
//! blocks at least as long as the dependence the estimator spans
//! (here the longest horizon), a fixed published resample count and
//! seed, and (2.5%, 97.5%) percentiles, so an interval is a
//! deterministic function of the session it summarizes.

use crate::metrics::{bar_status, ofi_bars, ofi_events, Quote};

/// Schema tag written into every information-speed report.
pub const SPEED_SCHEMA: &str = "toxicflow.info-speed/v1";

/// Bar length for the speed analysis. One second is the finest bar
/// the recorded sessions support: the main session averages about
/// nine quotes per second, so most one-second bars carry a fresh
/// quote, and the horizons of interest (seconds to minutes) stay
/// well resolved.
pub const SPEED_BAR_MS: i64 = 1_000;

/// Horizon ladder, in seconds: the contemporaneous bar, then a
/// geometric climb out to five minutes.
pub const SPEED_HORIZONS_S: [i64; 16] = [
    0, 1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 300,
];

/// Minimum R^2 for the exponential fit to be reported as a fit at
/// all. Below this the empirical curve is published and tau is null.
pub const FIT_MIN_R_SQUARED: f64 = 0.85;

/// Bootstrap resamples behind every reported interval. The same
/// count as the TF6 evaluation (`predict::BOOTSTRAP_RESAMPLES`), so
/// intervals across the repo share one convention.
pub const SPEED_BOOTSTRAP_RESAMPLES: usize = 2000;

/// Seed for the speed bootstrap. Fixed and published, like the TF6
/// seeds, so a report is reproducible from the session alone.
pub const SPEED_BOOTSTRAP_SEED: u64 = 0x85EB_CA6B_27D4_EA27;

/// Knobs for one speed analysis. `Default` is the published
/// configuration; tests shrink the bootstrap, nothing else.
#[derive(Debug, Clone)]
pub struct SpeedConfig {
    pub bar_ms: i64,
    pub horizons_s: Vec<i64>,
    pub bootstrap_resamples: usize,
    pub seed: u64,
}

impl Default for SpeedConfig {
    fn default() -> Self {
        Self {
            bar_ms: SPEED_BAR_MS,
            horizons_s: SPEED_HORIZONS_S.to_vec(),
            bootstrap_resamples: SPEED_BOOTSTRAP_RESAMPLES,
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

/// Local-projection curve: for each horizon (in bars), the OLS slope
/// of the cumulative mid return (basis points, from the bar before
/// the shock bar to the horizon bar) on the shock standardized by
/// `ofi_sd`, plus the number of usable origins. An origin is usable
/// only when every bar from the pre-shock bar through the horizon
/// bar was observed, so no response window crosses a feed gap.
fn lp_curve(
    mid: &[f64],
    ofi: &[f64],
    observed: &[bool],
    horizons_bars: &[usize],
    ofi_sd: f64,
) -> (Vec<f64>, Vec<usize>) {
    let n = mid.len();
    // Prefix sums of unobserved bars make the window check O(1).
    let mut unobs = vec![0usize; n + 1];
    for i in 0..n {
        unobs[i + 1] = unobs[i] + usize::from(!observed[i]);
    }
    let window_clean = |from: usize, to: usize| unobs[to + 1] - unobs[from] == 0;
    let mut slopes = Vec::with_capacity(horizons_bars.len());
    let mut counts = Vec::with_capacity(horizons_bars.len());
    for &h in horizons_bars {
        let (mut sx, mut sr, mut sxx, mut sxr, mut cnt) = (0.0, 0.0, 0.0, 0.0, 0usize);
        if ofi_sd > 0.0 {
            for t in 1..n {
                let end = t + h;
                if end >= n {
                    break;
                }
                if !window_clean(t - 1, end) {
                    continue;
                }
                let base = mid[t - 1];
                if base <= 0.0 || base.is_nan() || !mid[end].is_finite() {
                    continue;
                }
                let x = ofi[t] / ofi_sd;
                let r = (mid[end] / base - 1.0) * 1e4;
                sx += x;
                sr += r;
                sxx += x * x;
                sxr += x * r;
                cnt += 1;
            }
        }
        let slope = if cnt >= 30 {
            let nf = cnt as f64;
            let var = sxx - sx * sx / nf;
            if var > 0.0 {
                (sxr - sx * sr / nf) / var
            } else {
                f64::NAN
            }
        } else {
            f64::NAN
        };
        slopes.push(slope);
        counts.push(cnt);
    }
    (slopes, counts)
}

/// One exponential adjustment fit, R(h) = R_inf * (1 - exp(-h/tau)).
#[derive(Debug, Clone, Copy)]
pub struct ExpFit {
    pub r_inf_bps: f64,
    pub tau_s: f64,
    pub r_squared: f64,
}

/// Tau grid for the profile fit: log-spaced, in seconds.
fn tau_grid() -> Vec<f64> {
    let (lo, hi, steps) = (0.25f64, 1500.0f64, 240usize);
    (0..steps)
        .map(|i| lo * (hi / lo).powf(i as f64 / (steps - 1) as f64))
        .collect()
}

/// Least-squares exponential fit over the finite response points.
/// Returns the best fit found, without judging it; `judge_fit`
/// decides whether it may be reported as a measurement.
fn fit_exponential_raw(horizons_s: &[f64], response: &[f64]) -> Option<ExpFit> {
    let pts: Vec<(f64, f64)> = horizons_s
        .iter()
        .zip(response.iter())
        .filter(|(_, r)| r.is_finite())
        .map(|(h, r)| (*h, *r))
        .collect();
    if pts.len() < 4 {
        return None;
    }
    let mean = pts.iter().map(|p| p.1).sum::<f64>() / pts.len() as f64;
    let sst: f64 = pts.iter().map(|p| (p.1 - mean).powi(2)).sum();
    if sst <= 0.0 || sst.is_nan() {
        return None;
    }
    let mut best: Option<(f64, f64, f64)> = None; // (sse, r_inf, tau)
    for tau in tau_grid() {
        let (mut sff, mut sfr) = (0.0, 0.0);
        for (h, r) in &pts {
            let f = 1.0 - (-h / tau).exp();
            sff += f * f;
            sfr += f * r;
        }
        if sff <= 0.0 || sff.is_nan() {
            continue;
        }
        let r_inf = sfr / sff;
        let mut sse = 0.0;
        for (h, r) in &pts {
            let f = 1.0 - (-h / tau).exp();
            sse += (r - r_inf * f).powi(2);
        }
        if best.as_ref().is_none_or(|b| sse < b.0) {
            best = Some((sse, r_inf, tau));
        }
    }
    let (sse, r_inf, tau) = best?;
    Some(ExpFit {
        r_inf_bps: r_inf,
        tau_s: tau,
        r_squared: 1.0 - sse / sst,
    })
}

/// Judge a raw fit: it is a measurement only when the long-run
/// response is positive, the curve is actually exponential-shaped
/// at the published R^2 bar, and tau is interior to the search grid
/// (a boundary tau means the grid, not the data, chose the number).
fn judge_fit(horizons_s: &[f64], response: &[f64], fit: &ExpFit) -> Result<(), String> {
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
    if fit.r_squared < FIT_MIN_R_SQUARED {
        return Err(format!(
            "the exponential fit explains only R^2 = {:.3} of the response curve, below the {:.2} bar for reporting it as a fit",
            fit.r_squared, FIT_MIN_R_SQUARED
        ));
    }
    let grid = tau_grid();
    let (lo, hi) = (grid[1], grid[grid.len() - 2]);
    if fit.tau_s <= lo || fit.tau_s >= hi {
        return Err(format!(
            "the fitted tau of {:.2} s sits at the edge of the search grid, so the grid rather than the data picked it",
            fit.tau_s
        ));
    }
    Ok(())
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

/// Moving-block bootstrap of the whole estimator: resample circular
/// blocks of bars (block length = the longest horizon, the span the
/// estimator's dependence reaches), rebuild the series, and re-run
/// the curve and the fit on each resample. Returns the per-horizon
/// response draws and the judged-ok fit draws.
fn bootstrap_draws(
    series: &BarSeries,
    horizons_bars: &[usize],
    horizons_s: &[f64],
    cfg: &SpeedConfig,
) -> (Vec<Vec<f64>>, Vec<ExpFit>) {
    let n = series.mid.len();
    let block = horizons_bars
        .iter()
        .copied()
        .max()
        .unwrap_or(1)
        .max(1)
        .min(n);
    let blocks_per = n.div_ceil(block);
    let mut rng = SplitMix64(cfg.seed);
    let mut curve_draws: Vec<Vec<f64>> = Vec::with_capacity(cfg.bootstrap_resamples);
    let mut fit_draws: Vec<ExpFit> = Vec::new();
    let (mut mid, mut ofi, mut obs) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for _ in 0..cfg.bootstrap_resamples {
        mid.clear();
        ofi.clear();
        obs.clear();
        for _ in 0..blocks_per {
            let start = rng.next_below(n);
            for j in 0..block {
                if mid.len() == n {
                    break;
                }
                let idx = (start + j) % n;
                mid.push(series.mid[idx]);
                ofi.push(series.ofi[idx]);
                obs.push(series.observed[idx]);
            }
        }
        // The resample's own OFI scale defines its one-SD shock.
        let obs_ofi: Vec<f64> = ofi
            .iter()
            .zip(obs.iter())
            .filter(|(_, o)| **o)
            .map(|(v, _)| *v)
            .collect();
        let sd = population_sd(&obs_ofi);
        if sd <= 0.0 || sd.is_nan() {
            continue;
        }
        let (curve, _) = lp_curve(&mid, &ofi, &obs, horizons_bars, sd);
        if curve.iter().any(|v| !v.is_finite()) {
            continue;
        }
        curve_draws.push(curve.clone());
        if let Some(fit) = fit_exponential_raw(horizons_s, &curve) {
            if judge_fit(horizons_s, &curve, &fit).is_ok() {
                fit_draws.push(fit);
            }
        }
    }
    (curve_draws, fit_draws)
}

fn population_sd(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let m = xs.iter().sum::<f64>() / xs.len() as f64;
    (xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64).sqrt()
}

/// One session's information-speed result. `tau_s` and friends are
/// null whenever the exponential fit failed; the empirical curve is
/// always reported.
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
    /// Cumulative mid response, basis points per one-SD OFI shock.
    pub response_bps: Vec<f64>,
    /// Moving-block bootstrap interval per horizon.
    pub response_ci_bps: Vec<(f64, f64)>,
    /// Usable origins per horizon after gap censoring.
    pub n_origins: Vec<usize>,
    pub fit_ok: bool,
    /// Why the exponential fit was not reported, when it was not.
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
}

/// Analyze one session's quotes. Returns None when the session is
/// too short to support the longest horizon with a usable number of
/// origins, or carries no OFI variation.
pub fn analyze_session_speed(
    session: &str,
    quotes: &[Quote],
    cfg: &SpeedConfig,
) -> Option<SessionSpeed> {
    if quotes.is_empty() || cfg.bar_ms <= 0 {
        return None;
    }
    let series = build_bar_series(quotes, cfg.bar_ms);
    let n = series.mid.len();
    let horizons_bars: Vec<usize> = cfg
        .horizons_s
        .iter()
        .map(|s| (s * 1000 / cfg.bar_ms).max(0) as usize)
        .collect();
    let max_h = *horizons_bars.iter().max().unwrap_or(&0);
    if n < max_h + 100 {
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
    if ofi_sd <= 0.0 || ofi_sd.is_nan() {
        return None;
    }
    let horizons_s: Vec<f64> = cfg.horizons_s.iter().map(|s| *s as f64).collect();
    let (curve, counts) = lp_curve(
        &series.mid,
        &series.ofi,
        &series.observed,
        &horizons_bars,
        ofi_sd,
    );
    if curve.iter().any(|v| !v.is_finite()) || *counts.last().unwrap_or(&0) < 30 {
        return None;
    }
    let raw_fit = fit_exponential_raw(&horizons_s, &curve);
    let (fit, fit_failure) = match raw_fit {
        Some(f) => match judge_fit(&horizons_s, &curve, &f) {
            Ok(()) => (Some(f), None),
            Err(why) => (None, Some(why)),
        },
        None => (
            None,
            Some("the response curve has too few finite points to fit".to_string()),
        ),
    };
    let (curve_draws, fit_draws) = bootstrap_draws(&series, &horizons_bars, &horizons_s, cfg);
    let response_ci_bps: Vec<(f64, f64)> = (0..horizons_s.len())
        .map(|k| {
            let col: Vec<f64> = curve_draws.iter().map(|c| c[k]).collect();
            percentile_ci(&col)
        })
        .collect();
    let n_draws = curve_draws.len().max(1);
    let bootstrap_fit_share = fit_draws.len() as f64 / n_draws as f64;
    let (tau_ci, half_ci, rinf_ci) = if fit_draws.len() >= 100 {
        (
            Some(percentile_ci(
                &fit_draws.iter().map(|f| f.tau_s).collect::<Vec<_>>(),
            )),
            Some(percentile_ci(
                &fit_draws
                    .iter()
                    .map(|f| f.tau_s * std::f64::consts::LN_2)
                    .collect::<Vec<_>>(),
            )),
            Some(percentile_ci(
                &fit_draws.iter().map(|f| f.r_inf_bps).collect::<Vec<_>>(),
            )),
        )
    } else {
        (None, None, None)
    };
    let span_s = if n > 0 {
        (quotes.last().unwrap().time_ms - quotes.first().unwrap().time_ms) as f64 / 1000.0
    } else {
        0.0
    };
    Some(SessionSpeed {
        session: session.to_string(),
        bar_ms: cfg.bar_ms,
        n_bars: n,
        n_observed_bars: series.observed.iter().filter(|o| **o).count(),
        span_s,
        ofi_sd,
        horizons_s,
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
        // The best raw fit's R^2 is reported even when judging
        // rejected it, so the failure is quantified in the artifact
        // and not only in the prose reason.
        fit_r_squared: raw_fit.map(|f| f.r_squared),
        bootstrap_fit_share,
    })
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

/// The multi-session information-speed report, serialized as the
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
}

impl SpeedReport {
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("speed report serializes")
    }
}

/// Aggregate per-session results into the report.
pub fn summarize_speed(sessions: Vec<SessionSpeed>, cfg: &SpeedConfig) -> SpeedReport {
    let summary = summarize(&sessions);
    let notes = vec![
        "Order-flow shocks are the information-arrival proxy: information is inferred \
         from trades themselves (Hasbrouck-style), and the recorded sessions carry no \
         news timestamps. Nothing here identifies a news event or an informed trader."
            .to_string(),
        "Effective sample unit: one independent recording session. Bars inside a \
         session are serially dependent and are never counted as independent samples; \
         the cross-session summary is descriptive statistics over sessions."
            .to_string(),
        "Single venue (Binance), one symbol (BTCUSDT), short recorded sessions. Tau \
         describes this sample's adjustment to OFI shocks, not a market-wide constant."
            .to_string(),
        "tau is reported only for sessions whose cumulative response the exponential \
         adjustment curve actually fits (R^2 at or above the published bar, positive \
         long-run response, tau interior to the search grid). Sessions that fail keep \
         their empirical curve and the failure reason instead of a forced number."
            .to_string(),
    ];
    SpeedReport {
        schema: SPEED_SCHEMA.to_string(),
        bar_ms: cfg.bar_ms,
        horizons_s: cfg.horizons_s.iter().map(|s| *s as f64).collect(),
        shock_definition:
            "one standard deviation of per-bar order-flow imbalance (OFI) within the session"
                .to_string(),
        effective_sample_unit: "independent recording session".to_string(),
        sessions,
        summary,
        notes,
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
    /// to A with tau = -1 / ln(rho) bars.
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
        // Long-run response: A scaled by sd(x)/sd(ofi-in-x-units).
        // ofi = 37x and the regression standardizes by sd(ofi), so
        // the slope is A * sd(x) / 37 * 37 = A in bps per SD of x;
        // sd of x here is ~0.577, and c was applied per unit x, so
        // the per-SD response is A * sd(x) ~ 5.2 bps.
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
    fn bootstrap_is_deterministic() {
        let rho = (-1.0f64 / 20.0).exp();
        let series = synth_series(3_000, rho, 9.0);
        let cfg = SpeedConfig {
            bootstrap_resamples: 40,
            ..SpeedConfig::default()
        };
        let horizons_bars: Vec<usize> = SPEED_HORIZONS_S.iter().map(|s| *s as usize).collect();
        let horizons_s: Vec<f64> = SPEED_HORIZONS_S.iter().map(|s| *s as f64).collect();
        let (c1, f1) = bootstrap_draws(&series, &horizons_bars, &horizons_s, &cfg);
        let (c2, f2) = bootstrap_draws(&series, &horizons_bars, &horizons_s, &cfg);
        assert_eq!(c1, c2);
        assert_eq!(f1.len(), f2.len());
        assert!(!c1.is_empty());
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
