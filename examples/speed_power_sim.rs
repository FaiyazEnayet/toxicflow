//! Power/coverage simulation harness for the OFI-price response
//! estimator in `speed`.
//!
//! The estimator's published sessions are few and short, and every
//! one of them fails the adjustment-fit gates, so the report alone
//! cannot say whether that is a property of the market or of the
//! design (session length, gap pattern, noise level). This harness
//! answers the design question: it generates synthetic one-second
//! bar series whose population local-projection response is known
//! exactly, runs the shipped estimator on them unchanged, and
//! records how often its intervals cover the truth and how often
//! its fit gates promote a tau, per design cell.
//!
//! Data-generating process (per session):
//!
//! - Standardised flow `z_t` is AR(1) with coefficient `phi`
//!   (lag-1 autocorrelation) and unit variance; the bar OFI is
//!   `z_t * ofi_sd` with `ofi_sd = 3.0636`, the main recorded
//!   session's OFI standard deviation. The estimator standardises
//!   by the sample SD exactly as shipped, so responses are per
//!   one sample SD, matching the report's units.
//! - The per-bar mid return (bps) is a convolution of flow with a
//!   kernel `g` plus iid Gaussian noise (`sigma_noise_bps`,
//!   default 0.75 bps, the per-bar return volatility implied by
//!   the main session's horizon-0 HAC standard error):
//!   `r_t = sum_k g_k z_{t-k} + sigma * eps_t`.
//!   Single-exponential cells use `g_0 = J` and
//!   `g_k = R_inf (1 - rho) rho^{k-1}` for k >= 1 with
//!   `rho = exp(-1/tau)`, so with white flow the population
//!   response is exactly `R(h) = J + R_inf (1 - exp(-h/tau))`.
//!   Two-component cells sum two such kernels; the null cell has
//!   `g_k = 0` for k >= 1 (a pure contemporaneous jump).
//! - With serially correlated flow the raw-OFI population
//!   response is NOT the kernel's partial sums: follow-on flow
//!   correlated with the shock bar contributes. The harness
//!   computes the exact population response for the `phi` in
//!   force, `R(h) = sum_{j<=h} sum_k g_k phi^|j-k|`, by forward /
//!   backward recursions over the kernel (truncated at 6,000
//!   lags, where the slowest kernel has decayed by exp(-10)),
//!   and coverage is judged against that curve. The structural
//!   `tau` of the kernel is what tau recovery is judged against.
//! - Gaps (`--gaps mainlike`): one contiguous unobserved stretch
//!   of 7.8% of the session placed 45% through it, plus scattered
//!   unobserved bars at 0.2% (the main session's pattern: a
//!   919-bar outage in 11,727 bars plus 21 scattered bars).
//!   Unobserved bars carry the last observed mid and zero OFI,
//!   exactly as the recorder's bar builder represents them.
//!
//! The estimator is run per session, the report's own unit; no
//! pooling is done anywhere in this harness. A session counts as
//! promoted exactly when the shipped analysis reports a tau for
//! it (`fit_ok`).
//!
//! Usage: speed_power_sim --out results.jsonl [--truths a,b]
//!   [--phis 0.1,0.25] [--gaps none,mainlike] [--lens 1800,3600]
//!   [--reps 100] [--rep-offset 0] [--bootstrap 2000]
//!   [--sensitivity 500] [--extended 500] [--threads 4]
//!   [--sigma-noise-bps 0.75] [--total-bps 0.30]
//!
//! Every run is deterministic: the DGP seed of a (cell,
//! replication) is a fixed hash of its indices, stated in
//! `cell_seed`, and the estimator keeps its published seed.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use toxicflow::speed::{
    analyze_bar_series, BarSeries, SpeedConfig, SPEED_BOOTSTRAP_SEED,
};

// ----------------------------------------------------------------
// Deterministic RNG (SplitMix64 + Box-Muller), independent of the
// estimator's own bootstrap RNG.
// ----------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        // Box-Muller, cosine branch only (one draw per call keeps
        // the stream simple and reproducible).
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn cell_seed(truth_idx: u64, phi_idx: u64, gap_idx: u64, len_bars: u64, rep: u64) -> u64 {
    // Fixed published mixing: SplitMix64 rounds over the indices.
    let mut s = 0x5057_4552_5349_4D01u64
        ^ truth_idx.wrapping_mul(0x9E37_79B9)
        ^ phi_idx.wrapping_mul(0x85EB_CA6B)
        ^ gap_idx.wrapping_mul(0x27D4_EA27)
        ^ len_bars.wrapping_mul(0x1656_7A93)
        ^ rep.wrapping_mul(0x9FB2_1C65);
    for _ in 0..3 {
        s = Rng(s).next_u64();
    }
    s
}

// ----------------------------------------------------------------
// Truths
// ----------------------------------------------------------------

#[derive(Clone, Debug)]
struct Truth {
    name: &'static str,
    /// (weight_bps, tau_s) adjustment components; empty = null.
    comps: Vec<(f64, f64)>,
    jump_bps: f64,
    /// Structural tau for tau-recovery judgements, when the truth
    /// is a single exponential.
    tau_s: Option<f64>,
}

fn truths(total_bps: f64) -> Vec<Truth> {
    let jump = 0.43 * total_bps;
    let rinf = total_bps - jump;
    let mut v = Vec::new();
    for tau in [30.0, 120.0, 600.0] {
        v.push(Truth {
            name: match tau {
                30.0 => "tau30_j0",
                120.0 => "tau120_j0",
                _ => "tau600_j0",
            },
            comps: vec![(total_bps, tau)],
            jump_bps: 0.0,
            tau_s: Some(tau),
        });
        v.push(Truth {
            name: match tau {
                30.0 => "tau30_jpos",
                120.0 => "tau120_jpos",
                _ => "tau600_jpos",
            },
            comps: vec![(rinf, tau)],
            jump_bps: jump,
            tau_s: Some(tau),
        });
    }
    v.push(Truth {
        name: "twocomp",
        comps: vec![(rinf / 2.0, 15.0), (rinf / 2.0, 450.0)],
        jump_bps: jump,
        tau_s: None,
    });
    v.push(Truth {
        name: "null",
        comps: vec![],
        jump_bps: total_bps,
        tau_s: None,
    });
    v
}

/// Kernel g_k (bps of return per unit standardised flow at lag k).
fn kernel(truth: &Truth, kmax: usize) -> Vec<f64> {
    let mut g = vec![0.0; kmax + 1];
    g[0] = truth.jump_bps;
    for &(w, tau) in &truth.comps {
        let rho = (-1.0f64 / tau).exp();
        for k in 1..=kmax {
            g[k] += w * (1.0 - rho) * rho.powi((k - 1) as i32);
        }
    }
    g
}

/// Exact population raw-OFI response R(h) for h = 0..=hmax under
/// AR(1) flow with coefficient phi: R(h) = sum_{j<=h} C_j with
/// C_j = sum_k g_k phi^|j-k|, computed by recursions:
/// A_j = phi A_{j-1} + g_j (k <= j part), B_j from the right.
fn population_response(g: &[f64], phi: f64, hmax: usize) -> Vec<f64> {
    let kmax = g.len() - 1;
    let mut c = vec![0.0; hmax + 1];
    // Forward part for j <= hmax, plus the backward part needs
    // B_j = sum_{k>j} g_k phi^{k-j}; fold it in by running the
    // backward recursion from kmax down to 0, storing only the
    // values at j <= hmax (the recursion must run from the top).
    let mut b = 0.0f64; // B at current j+1 boundary
    let mut back = vec![0.0; hmax + 1];
    for j in (0..=kmax).rev() {
        if j <= hmax {
            back[j] = b;
        }
        // B_{j-1} = phi * (g_j + B_j)
        b = phi * (g[j] + b);
    }
    let mut a = 0.0f64;
    for j in 0..=hmax {
        a = phi * a + g[j];
        // a currently holds sum_{k<=j} g_k phi^{j-k} + ... check:
        // A_j = phi*A_{j-1} + g_j with A_{-1}=0 is exact.
        c[j] = a + back[j];
    }
    let mut r = vec![0.0; hmax + 1];
    let mut acc = 0.0;
    for h in 0..=hmax {
        acc += c[h];
        r[h] = acc;
    }
    r
}

// ----------------------------------------------------------------
// Session generation
// ----------------------------------------------------------------

const OFI_SD: f64 = 3.0636;
const BURN_IN: usize = 5000;

fn gen_session(
    truth: &Truth,
    phi: f64,
    gaps_mainlike: bool,
    n_bars: usize,
    sigma_noise: f64,
    seed: u64,
) -> BarSeries {
    let mut rng = Rng(seed);
    // Component states: s_c,t = (1-rho_c) * sum_{m>=1} rho_c^{m-1} z_{t-m}
    let rhos: Vec<f64> = truth.comps.iter().map(|&(_, tau)| (-1.0f64 / tau).exp()).collect();
    let mut states = vec![0.0; truth.comps.len()];
    let mut z = 0.0f64;
    let inov_sd = (1.0 - phi * phi).sqrt();
    let total_bars = n_bars + BURN_IN;
    let mut mid = Vec::with_capacity(n_bars);
    let mut ofi = Vec::with_capacity(n_bars);
    let mut price = 100_000.0f64;
    for t in 0..total_bars {
        let e = rng.gauss();
        z = phi * z + inov_sd * e;
        let mut signal = truth.jump_bps * z;
        for (c, &(w, _)) in truth.comps.iter().enumerate() {
            signal += w * states[c];
        }
        let r_bps = signal + sigma_noise * rng.gauss();
        if t >= BURN_IN {
            price *= 1.0 + r_bps / 1e4;
            mid.push(price);
            ofi.push(z * OFI_SD);
        }
        for c in 0..states.len() {
            states[c] = rhos[c] * states[c] + (1.0 - rhos[c]) * z;
        }
    }
    let mut observed = vec![true; n_bars];
    if gaps_mainlike {
        let stretch = ((0.078 * n_bars as f64).round() as usize).max(1);
        let start = (0.45 * n_bars as f64) as usize;
        for slot in observed.iter_mut().take((start + stretch).min(n_bars)).skip(start) {
            *slot = false;
        }
        for i in 0..n_bars {
            if observed[i] && rng.next_f64() < 0.002 {
                observed[i] = false;
            }
        }
        // Carried-forward mid and zero OFI on unobserved bars, as
        // the recorder's bar builder represents them.
        let mut last = mid[0];
        for i in 0..n_bars {
            if observed[i] {
                last = mid[i];
            } else {
                mid[i] = last;
                ofi[i] = 0.0;
            }
        }
    }
    BarSeries { mid, ofi, observed }
}

// ----------------------------------------------------------------
// One session run -> one JSON record
// ----------------------------------------------------------------

fn run_one(
    truth: &Truth,
    phi: f64,
    gap_name: &str,
    n_bars: usize,
    rep: usize,
    cfg: &SpeedConfig,
    sigma_noise: f64,
    pop: &[f64],
) -> serde_json::Value {
    let seed = cell_seed(
        truth.name.len() as u64 * 131 + truth.tau_s.unwrap_or(0.0) as u64,
        (phi * 100.0) as u64,
        if gap_name == "mainlike" { 1 } else { 0 },
        n_bars as u64,
        rep as u64,
    );
    let series = gen_session(truth, phi, gap_name == "mainlike", n_bars, sigma_noise, seed);
    let t0 = std::time::Instant::now();
    let res = analyze_bar_series(&format!("sim-{}", rep), &series, (n_bars - 1) as f64, cfg);
    let secs = t0.elapsed().as_secs_f64();
    let key_h = [0usize, 60, 300];
    match res {
        None => serde_json::json!({
            "truth": truth.name, "phi": phi, "gap": gap_name, "len_bars": n_bars,
            "rep": rep, "seed": seed, "analyzed": false, "secs": secs,
        }),
        Some(s) => {
            let idx = |h: i64| s.horizons_s.iter().position(|v| *v == h as f64);
            let mut cov_pt = serde_json::Map::new();
            let mut cov_sim = serde_json::Map::new();
            let mut est = serde_json::Map::new();
            for h in key_h {
                if let Some(k) = idx(h as i64) {
                    let truth_h = pop[h];
                    let (lo, hi) = s.response_ci_bps[k];
                    cov_pt.insert(h.to_string(), serde_json::json!(lo <= truth_h && truth_h <= hi));
                    let (slo, shi) = s.response_simultaneous_band_bps[k];
                    cov_sim.insert(h.to_string(), serde_json::json!(slo <= truth_h && truth_h <= shi));
                    est.insert(h.to_string(), serde_json::json!(s.response_bps[k]));
                }
            }
            // Extended ladder coverage / fit, where present.
            let mut ext = serde_json::Value::Null;
            if let Some(e) = &s.extended {
                let mut ecov = serde_json::Map::new();
                let mut eci = serde_json::Map::new();
                for h in [900usize, 1500] {
                    if let Some(k) = e.horizons_s.iter().position(|v| *v == h as f64) {
                        let (lo, hi) = e.response_ci_bps[k];
                        ecov.insert(h.to_string(), serde_json::json!(lo <= pop[h] && pop[h] <= hi));
                        eci.insert(h.to_string(), serde_json::json!([lo, hi]));
                    }
                }
                ext = serde_json::json!({
                    "max_h": e.horizons_s.last().copied().unwrap_or(f64::NAN),
                    "fit_tau_s": e.fit_tau_s, "fit_r2": e.fit_r_squared,
                    "cov": ecov, "ci": eci, "n_resamples": e.n_resamples,
                });
            }
            // Diagnostic fields: interval bounds and HAC coverage
            // at the key horizons (the HAC interval is centred on
            // the point estimate, the bootstrap percentile interval
            // is not, so the pair separates centring failures from
            // width failures).
            let mut ci_map = serde_json::Map::new();
            let mut hac_cov = serde_json::Map::new();
            for h in key_h {
                if let Some(k) = idx(h as i64) {
                    ci_map.insert(h.to_string(), serde_json::json!(s.response_ci_bps[k]));
                    let (lo, hi) = s.hac_ci_bps[k];
                    hac_cov.insert(h.to_string(), serde_json::json!(lo <= pop[h] && pop[h] <= hi));
                }
            }
            serde_json::json!({
                "truth": truth.name, "tau_true": truth.tau_s, "phi": phi,
                "gap": gap_name, "len_bars": n_bars, "rep": rep, "seed": seed,
                "analyzed": true, "secs": secs,
                "n_observed": s.n_observed_bars, "ofi_sd_sample": s.ofi_sd,
                "fit_ok": s.fit_ok, "tau_s": s.tau_s, "tau_ci": s.tau_ci_s,
                "tau_raw": s.fit_tau_raw_s, "fit_r2": s.fit_r_squared,
                "fit_failure": s.fit_failure, "fit_share": s.bootstrap_fit_share,
                "region_bounded": s.fit_region.as_ref().map(|r| r.bounded),
                "region_tau_ci": s.fit_region.as_ref().map(|r| r.tau_ci_s),
                "cov_pt": cov_pt, "cov_sim": cov_sim, "est": est,
                "ci": ci_map, "hac_cov": hac_cov,
                "fit_jump_raw": s.fit_jump_raw_bps, "fit_rinf_raw": s.fit_r_inf_raw_bps,
                "truth_resp": {"0": pop[0], "60": pop[60], "300": pop[300],
                               "900": pop[900], "1500": pop[1500]},
                "extended": ext,
            })
        }
    }
}

// ----------------------------------------------------------------
// CLI + thread pool
// ----------------------------------------------------------------

fn arg_val(args: &[String], key: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == key).map(|w| w[1].clone())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = arg_val(&args, "--out").unwrap_or_else(|| "sim_results.jsonl".to_string());
    let total_bps: f64 = arg_val(&args, "--total-bps").and_then(|v| v.parse().ok()).unwrap_or(0.30);
    let sigma_noise: f64 = arg_val(&args, "--sigma-noise-bps").and_then(|v| v.parse().ok()).unwrap_or(0.75);
    let reps: usize = arg_val(&args, "--reps").and_then(|v| v.parse().ok()).unwrap_or(100);
    let rep_offset: usize = arg_val(&args, "--rep-offset").and_then(|v| v.parse().ok()).unwrap_or(0);
    let threads: usize = arg_val(&args, "--threads").and_then(|v| v.parse().ok()).unwrap_or(4);
    let cfg = SpeedConfig {
        bootstrap_resamples: arg_val(&args, "--bootstrap").and_then(|v| v.parse().ok()).unwrap_or(SpeedConfig::default().bootstrap_resamples),
        sensitivity_resamples: arg_val(&args, "--sensitivity").and_then(|v| v.parse().ok()).unwrap_or(SpeedConfig::default().sensitivity_resamples),
        extended_resamples: arg_val(&args, "--extended").and_then(|v| v.parse().ok()).unwrap_or(SpeedConfig::default().extended_resamples),
        ..SpeedConfig::default()
    };
    let all_truths = truths(total_bps);
    let truth_filter: Vec<String> = arg_val(&args, "--truths")
        .map(|v| v.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_default();
    let phis: Vec<f64> = arg_val(&args, "--phis")
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![0.1, 0.25]);
    let gaps: Vec<String> = arg_val(&args, "--gaps")
        .map(|v| v.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_else(|| vec!["none".to_string(), "mainlike".to_string()]);
    let lens: Vec<usize> = arg_val(&args, "--lens")
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1800, 3600, 7200, 14400]);

    // Work items: (truth, phi, gap, len, rep, population curve)
    let mut items: Vec<(Truth, f64, String, usize, usize, Vec<f64>)> = Vec::new();
    for t in &all_truths {
        if !truth_filter.is_empty() && !truth_filter.iter().any(|f| f == t.name) {
            continue;
        }
        let g = kernel(t, 6000);
        for &phi in &phis {
            let pop = population_response(&g, phi, 1800);
            for gap in &gaps {
                for &len in &lens {
                    for rep in rep_offset..rep_offset + reps {
                        items.push((t.clone(), phi, gap.clone(), len, rep, pop.clone()));
                    }
                }
            }
        }
    }
    eprintln!(
        "speed_power_sim: {} session runs, bootstrap {}, sensitivity {}, extended {}, estimator seed {:#x}",
        items.len(), cfg.bootstrap_resamples, cfg.sensitivity_resamples, cfg.extended_resamples, SPEED_BOOTSTRAP_SEED
    );
    let counter = AtomicUsize::new(0);
    let (tx, rx) = mpsc::channel::<String>();
    std::thread::scope(|scope| {
        for _ in 0..threads {
            let tx = tx.clone();
            let counter = &counter;
            let items = &items;
            let cfg = &cfg;
            scope.spawn(move || loop {
                let i = counter.fetch_add(1, Ordering::SeqCst);
                if i >= items.len() {
                    break;
                }
                let (t, phi, gap, len, rep, pop) = &items[i];
                let v = run_one(t, *phi, gap, *len, *rep, cfg, sigma_noise, pop);
                let _ = tx.send(serde_json::to_string(&v).unwrap());
            });
        }
        drop(tx);
        let mut f = std::fs::File::create(&out).expect("create out file");
        use std::io::Write;
        let mut n = 0;
        for line in rx {
            writeln!(f, "{line}").unwrap();
            n += 1;
            if n % 50 == 0 {
                eprintln!("  {n} runs written");
            }
        }
        eprintln!("speed_power_sim: wrote {n} records to {out}");
    });
}
