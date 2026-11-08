//! Pinning test for the power/coverage simulation harness in
//! `examples/speed_power_sim.rs`: a small, fixed-seed cell of the
//! harness's data-generating process, run through the shipped
//! estimator at a reduced bootstrap count so the test stays in
//! the seconds range. The assertions are tolerance bands around
//! the cell's outcome (the same bands the harness's full runs
//! land in), not exact values: they pin that the estimator, on a
//! known planted response of the main recorded session's scale,
//! recovers the response level at 60 s and does not promote a
//! tau at this session length, which is the behaviour the
//! recording plan is built on.

use toxicflow::speed::{analyze_bar_series, BarSeries, SpeedConfig};

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
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// One synthetic session with the harness's tau = 120 s,
/// jump-share 43% cell: standardised AR(1) flow (phi = 0.1),
/// returns driven by a geometric-lag kernel with total response
/// 0.30 bps plus 0.75 bps of iid noise, 1,800 one-second bars
/// after a 5,000-bar burn-in.
fn synth_session(seed: u64) -> BarSeries {
    let (phi, tau) = (0.1f64, 120.0f64);
    let (jump, rinf) = (0.129f64, 0.171f64);
    let rho = (-1.0f64 / tau).exp();
    let inov_sd = (1.0 - phi * phi).sqrt();
    let mut rng = Rng(seed);
    let mut z = 0.0f64;
    let mut state = 0.0f64;
    let mut mid = Vec::new();
    let mut ofi = Vec::new();
    let mut price = 100_000.0f64;
    for t in 0..1800 + 5000 {
        let e = rng.gauss();
        z = phi * z + inov_sd * e;
        let r_bps = jump * z + rinf * state + 0.75 * rng.gauss();
        if t >= 5000 {
            price *= 1.0 + r_bps / 1e4;
            mid.push(price);
            ofi.push(z * 3.0636);
        }
        state = rho * state + (1.0 - rho) * z;
    }
    BarSeries {
        mid,
        ofi,
        observed: vec![true; 1800],
    }
}

/// Population response at 60 s for this cell under AR(1) flow,
/// R(60) = sum_{j<=60} sum_k g_k phi^|j-k| with the kernel above:
/// computed once here from the kernel, not from the estimator.
fn population_r60() -> f64 {
    let (phi, tau) = (0.1f64, 120.0f64);
    let (jump, rinf) = (0.129f64, 0.171f64);
    let rho = (-1.0f64 / tau).exp();
    let mut g = vec![0.0; 6001];
    g[0] = jump;
    for (k, gk) in g.iter_mut().enumerate().skip(1) {
        *gk = rinf * (1.0 - rho) * rho.powi((k - 1) as i32);
    }
    let mut total = 0.0;
    for j in 0..=60usize {
        let mut c = 0.0;
        for (k, gk) in g.iter().enumerate() {
            c += gk * phi.powi((j as i32 - k as i32).abs());
        }
        total += c;
    }
    total
}

#[test]
fn harness_cell_recovers_level_and_promotes_nothing() {
    let cfg = SpeedConfig {
        bootstrap_resamples: 60,
        sensitivity_resamples: 20,
        extended_resamples: 20,
        ..SpeedConfig::default()
    };
    let truth60 = population_r60();
    assert!(
        (0.20..0.25).contains(&truth60),
        "population R(60) {truth60} should sit at the planted level"
    );
    let mut est60 = Vec::new();
    let mut covered60 = 0usize;
    let mut promoted = 0usize;
    for rep in 0..6u64 {
        let series = synth_session(0x5EED_0001 + rep * 7919);
        let s = analyze_bar_series("pin", &series, 1799.0, &cfg).expect("session analyzes");
        let k60 = s
            .horizons_s
            .iter()
            .position(|h| *h == 60.0)
            .expect("60 s rung exists");
        est60.push(s.response_bps[k60]);
        let (lo, hi) = s.response_ci_bps[k60];
        if lo <= truth60 && truth60 <= hi {
            covered60 += 1;
        }
        if s.fit_ok {
            promoted += 1;
        }
    }
    let mean60 = est60.iter().sum::<f64>() / est60.len() as f64;
    assert!(
        (mean60 - truth60).abs() < 0.15,
        "mean estimated R(60) {mean60:.3} should sit within 0.15 bps of the population {truth60:.3}"
    );
    assert!(
        covered60 >= 3,
        "at least half the sessions' 60 s intervals should cover the truth (got {covered60}/6)"
    );
    assert_eq!(
        promoted, 0,
        "no tau should be promoted at 1,800 bars in this cell"
    );
}
