//! Stage 3 toxicflow E5: robustness, negative controls and estimator
//! sensitivity.
//!
//! The parameter grid is preregistered in `experiments/e5/grid.json`
//! before any grid run, and the runner refuses to run a grid file
//! that does not match the library grid exactly. No test outcome is
//! used for parameter selection: the default configuration stays the
//! reported configuration regardless of results.
//!
//! These tests pin the new estimators on known-answer fixtures:
//! per-window lambda counts and tail flags, train-only gate
//! calibration, lead-lag probes, timestamp-shift nulls, and OFI
//! response ablations with volume, intensity, and volatility controls.

use std::path::{Path, PathBuf};

use serde_json::Value;
use toxicflow::metrics::{FlowBar, Quote, Trade};
use toxicflow::robustness::{
    calibrate_gate_train_only, flow_bars, grid_from_json, kyle_lambda_robust, lambda_tail_summary,
    lead_lag_response, ofi_response_ablation, preregistered_grid, shift_flow_bars, AblationRow,
    RobustnessConfig,
};
use toxicflow::strategy::{run_mm, MmConfig};

fn repo_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

/// Fixed-seed xorshift; deterministic, no randomness at test time.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 as f64) / (u64::MAX as f64)
    }
}

#[test]
fn stage3_toxicflow_5_preregistered_grid_matches_grid_file() {
    let text = std::fs::read_to_string(repo_path("experiments/e5/grid.json"))
        .expect("experiments/e5/grid.json must exist");
    let file_grid = grid_from_json(&text).expect("grid.json must parse");
    let lib_grid = preregistered_grid();
    assert_eq!(file_grid.len(), 15, "grid must hold 15 configs");
    assert_eq!(lib_grid.len(), 15, "library grid must hold 15 configs");
    for (from_file, from_lib) in file_grid.iter().zip(lib_grid.iter()) {
        assert_eq!(from_file, from_lib, "grid file must match the library grid");
    }
    assert_eq!(lib_grid[0], RobustnessConfig::default_config());
    assert_eq!(lib_grid[0].id(), "default");
    let ids: Vec<String> = lib_grid.iter().map(|c| c.id()).collect();
    let mut sorted = ids.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(ids.len(), sorted.len(), "grid ids must be unique");
    let grid: Value = serde_json::from_str(&text).expect("grid.json must parse as JSON");
    let rule = grid["selection_rule"].as_str().unwrap_or("");
    assert!(
        rule.contains("No test outcome"),
        "grid must state that no test outcome is used for selection: {rule}"
    );
    let train = grid["train_sessions"].as_array().unwrap();
    let test = grid["test_sessions"].as_array().unwrap();
    assert!(!train.is_empty() && !test.is_empty());
    for t in test {
        assert!(
            !train.contains(t),
            "train and test sessions must be disjoint"
        );
    }
    // The default stays the reported configuration regardless of
    // results; the grid only documents how far each number moves.
    assert_eq!(
        grid["default"]["vpin_gate"].as_f64(),
        Some(0.7),
        "default gate must stay 0.7"
    );
}

#[test]
fn stage3_toxicflow_5_config_hash_stable_and_distinct() {
    let a = RobustnessConfig::default_config();
    let b = RobustnessConfig::new(25, 20, 10_000, 60, 0.7);
    assert_eq!(a.config_hash(), a.config_hash());
    assert_eq!(a.config_hash().len(), 64);
    assert_ne!(a.config_hash(), b.config_hash());
    let m = a.manifest("session-20261109-083238-utc", "deadbeef");
    assert_eq!(m["config_id"].as_str(), Some("default"));
    assert_eq!(m["config_hash"].as_str(), Some(a.config_hash().as_str()));
    assert_eq!(m["session"].as_str(), Some("session-20261109-083238-utc"));
    assert_eq!(m["input_hash"].as_str(), Some("deadbeef"));
    assert_eq!(m["protocol"].as_str(), Some("e5-robustness/v1"));
}

#[test]
fn stage3_toxicflow_5_mm_config_follows_grid() {
    let cfg = RobustnessConfig::new(25, 10, 5_000, 30, 0.6);
    let mm: MmConfig = cfg.mm_config();
    assert_eq!(mm.vpin_buckets, 25);
    assert_eq!(mm.vpin_window, 10);
    assert!((mm.vpin_threshold - 0.6).abs() < 1e-12);
    let d = RobustnessConfig::default_config().mm_config();
    assert_eq!(d.vpin_buckets, 50);
    assert_eq!(d.vpin_window, 20);
    assert!((d.vpin_threshold - 0.7).abs() < 1e-12);
}

/// Exact-slope bars: dmid = 2 * signed_vol, no gaps.
fn exact_bars(n: usize) -> Vec<FlowBar> {
    (0..n)
        .map(|i| {
            let x = (i as f64 - 30.0) / 10.0;
            FlowBar {
                time_ms: i as i64 * 10_000,
                signed_vol: x,
                dmid: 2.0 * x,
                gap_before: false,
            }
        })
        .collect()
}

#[test]
fn stage3_toxicflow_5_lambda_robust_known_answer() {
    let bars = exact_bars(60);
    let w = kyle_lambda_robust(&bars, 60);
    assert_eq!(w.len(), 60);
    // Fewer than 10 regression points: no lambda, but the valid
    // count is still reported.
    assert!(w[5].lambda.is_none());
    assert_eq!(w[5].n_valid, 6);
    assert_eq!(w[5].n_total, 60);
    assert!(!w[5].tail_flag);
    assert!(w[5].winsorized_lambda.is_none());
    // The full window recovers the exact slope with no tail flag.
    let last = &w[59];
    assert_eq!(last.n_valid, 60);
    assert!((last.lambda.unwrap() - 2.0).abs() < 1e-9);
    assert!(!last.tail_flag);
    assert!((last.winsorized_lambda.unwrap() - 2.0).abs() < 1e-9);
    let s = lambda_tail_summary(&w);
    assert_eq!(s.n_windows, 60);
    assert_eq!(s.n_with_lambda, 51);
    assert_eq!(s.n_tail, 0);
    assert!((s.tail_share).abs() < 1e-12);
    assert!((s.mean_lambda - 2.0).abs() < 1e-9);
    assert!((s.winsorized_mean_lambda - 2.0).abs() < 1e-9);
}

#[test]
fn stage3_toxicflow_5_lambda_tail_flag_and_winsorize() {
    // One extreme bar (dmid 1e6) inside an otherwise exact slope-2
    // series: windows covering it get an extreme lambda and must be
    // flagged, while the winsorized mean stays near the true slope.
    let mut bars = exact_bars(120);
    bars[100].dmid = 1_000_000.0;
    let w = kyle_lambda_robust(&bars, 60);
    let s = lambda_tail_summary(&w);
    assert!(s.n_tail > 0, "extreme windows must be flagged");
    assert!(s.tail_share > 0.0 && s.tail_share < 0.5);
    assert!(
        s.mean_lambda > 100.0,
        "raw mean must show the tail: {}",
        s.mean_lambda
    );
    assert!(
        (s.winsorized_mean_lambda - 2.0).abs() < 1.0,
        "winsorized mean must stay near the true slope: {}",
        s.winsorized_mean_lambda
    );
    // A flagged window keeps its raw lambda but carries the flag and
    // a clamped winsorized value.
    let flagged = w.iter().find(|x| x.tail_flag).expect("a flagged window");
    assert!(flagged.lambda.unwrap().abs() > 100.0);
    assert!(flagged.winsorized_lambda.unwrap().abs() < 10.0);
    // Windows clear of the extreme bar are unflagged and exact.
    let clean = w
        .iter()
        .find(|x| !x.tail_flag && x.lambda.is_some())
        .expect("a clean window");
    assert!((clean.lambda.unwrap() - 2.0).abs() < 1e-9);
}

#[test]
fn stage3_toxicflow_5_lambda_valid_counts_restart_at_gap() {
    // A gap boundary at bar 30 restarts the regression window: bar
    // 35 has only 5 valid points (no lambda), bar 40 has 10.
    let mut bars = exact_bars(60);
    bars[30].gap_before = true;
    let w = kyle_lambda_robust(&bars, 60);
    assert!(w[35].lambda.is_none());
    assert_eq!(w[35].n_valid, 5);
    assert_eq!(w[40].n_valid, 10);
    assert!(w[40].lambda.is_some());
    // Pre-gap windows are untouched by the gap.
    assert_eq!(w[29].n_valid, 30);
}

#[test]
fn stage3_toxicflow_5_lambda_none_on_flat_flow() {
    let bars: Vec<FlowBar> = (0..60)
        .map(|i| FlowBar {
            time_ms: i as i64 * 10_000,
            signed_vol: 1.0,
            dmid: i as f64 * 0.01,
            gap_before: false,
        })
        .collect();
    let w = kyle_lambda_robust(&bars, 60);
    let last = &w[59];
    assert!(last.lambda.is_none());
    assert!(last.winsorized_lambda.is_none());
    assert!(!last.tail_flag);
    // The count is still reported even though the fit is undefined.
    assert_eq!(last.n_valid, 60);
}

#[test]
fn stage3_toxicflow_5_calibrate_gate_train_only() {
    let train: Vec<f64> = (1..=9).map(|i| i as f64 / 10.0).collect();
    // 90th percentile of 0.1..0.9 is the index-7 value, 0.8.
    assert_eq!(
        calibrate_gate_train_only(&train, 0.9),
        Some(0.8),
        "gate must be the train 90th percentile"
    );
    assert_eq!(calibrate_gate_train_only(&train, 0.5), Some(0.5));
    assert_eq!(calibrate_gate_train_only(&[], 0.9), None);
    // A train slice that differs from a later slice calibrates from
    // the train slice alone: the value 0.95 never appears in train.
    let test_slice = [0.95, 0.96, 0.97];
    let gate = calibrate_gate_train_only(&train, 0.9).unwrap();
    assert!(
        !test_slice.contains(&gate),
        "gate must come from train values only"
    );
}

#[test]
fn stage3_toxicflow_5_lead_lag_known_answer() {
    // OFI drives the next bar's return with slope exactly 2:
    // ret[i] = 2 * ofi[i-1]. The lag-1 probe must recover it, the
    // lead probes (OFI predicting past returns) must stay near zero.
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let n = 300;
    let ofi: Vec<f64> = (0..n).map(|_| rng.next() * 2.0 - 1.0).collect();
    let mut ret = vec![0.0; n];
    for i in 1..n {
        ret[i] = 2.0 * ofi[i - 1];
    }
    let curve = lead_lag_response(&ofi, &ret, 5);
    assert_eq!(curve.len(), 11);
    assert_eq!(curve[0].lag, -5);
    assert_eq!(curve[10].lag, 5);
    let at = |lag: i32| curve.iter().find(|p| p.lag == lag).unwrap();
    // Paired OFI sd for the lag-1 pairs (ofi[0..n-1]).
    let paired = &ofi[..n - 1];
    let mean = paired.iter().sum::<f64>() / paired.len() as f64;
    let sd = (paired.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / paired.len() as f64).sqrt();
    let lag1 = at(1);
    assert!(
        (lag1.slope_bps_per_sd - 2.0 * sd).abs() < 0.05,
        "lag 1 must recover slope 2 per unit: got {}, want {}",
        lag1.slope_bps_per_sd,
        2.0 * sd
    );
    for lag in [-5, -1, 0, 5] {
        let p = at(lag);
        assert!(
            p.slope_bps_per_sd.abs() < 0.5,
            "lag {lag} must stay near zero, got {}",
            p.slope_bps_per_sd
        );
    }
}

#[test]
fn stage3_toxicflow_5_timestamp_shift_null() {
    // Exact slope-2 bars: shifting signed flow 3 bars against the
    // price changes must collapse the lambda toward zero.
    let mut rng = Rng(12345);
    let bars: Vec<FlowBar> = (0..120)
        .map(|i| {
            let x = rng.next() * 2.0 - 1.0;
            FlowBar {
                time_ms: i as i64 * 10_000,
                signed_vol: x,
                dmid: 2.0 * x,
                gap_before: false,
            }
        })
        .collect();
    let orig = kyle_lambda_robust(&bars, 60);
    let orig_mean = lambda_tail_summary(&orig).mean_lambda;
    assert!(
        (orig_mean - 2.0).abs() < 1e-9,
        "unshifted mean must be 2, got {orig_mean}"
    );
    let shifted = shift_flow_bars(&bars, 3);
    assert_eq!(shifted.len(), bars.len());
    assert_eq!(shifted[0].signed_vol, 0.0);
    assert!((shifted[3].signed_vol - bars[0].signed_vol).abs() < 1e-12);
    assert!((shifted[3].dmid - bars[3].dmid).abs() < 1e-12);
    let null_mean = lambda_tail_summary(&kyle_lambda_robust(&shifted, 60)).mean_lambda;
    assert!(
        null_mean.abs() < 0.5,
        "shifted null mean must collapse toward zero, got {null_mean}"
    );
}

#[test]
fn stage3_toxicflow_5_flow_bars_known_answer() {
    // Quotes at 0s, 10s, 20s, 40s: the 30s bar is empty and the 40s
    // bar follows a hole, so both carry gap_before.
    let q = |t: i64, mid: f64| Quote {
        time_ms: t,
        bid_px: mid - 0.05,
        bid_sz: 5.0,
        ask_px: mid + 0.05,
        ask_sz: 5.0,
    };
    let quotes = vec![
        q(0, 100.0),
        q(10_000, 101.0),
        q(20_000, 102.0),
        q(40_000, 105.0),
    ];
    let trades = vec![
        Trade {
            time_ms: 5_000,
            price: 100.0,
            qty: 0.5,
            is_buyer_maker: false,
        },
        Trade {
            time_ms: 25_000,
            price: 102.0,
            qty: 0.25,
            is_buyer_maker: true,
        },
    ];
    let bars = flow_bars(&trades, &quotes, 10_000);
    assert_eq!(bars.len(), 4);
    assert!((bars[0].dmid - 1.0).abs() < 1e-12);
    assert!(!bars[0].gap_before);
    // The 5s trade sits in bar 0, which emits no flow bar; the 10s
    // bar itself holds no trade.
    assert!(bars[0].signed_vol.abs() < 1e-12);
    assert!((bars[1].dmid - 1.0).abs() < 1e-12);
    assert!(!bars[1].gap_before);
    // The 25s sale sits in the 20s bar.
    assert!((bars[1].signed_vol + 0.25).abs() < 1e-12);
    // Bar 2 is the empty 30s bar: carried mid, zero change, gap.
    assert!(bars[2].dmid.abs() < 1e-12);
    assert!(bars[2].gap_before);
    assert!(bars[2].signed_vol.abs() < 1e-12);
    // Bar 3 follows the hole: gap flag set, dmid spans the hole.
    assert!((bars[3].dmid - 3.0).abs() < 1e-12);
    assert!(bars[3].gap_before);
}

#[test]
fn stage3_toxicflow_5_ablation_recovers_controlled_slope() {
    // dmid = 2 * ofi + 0.5 * volume exactly, with ofi correlated
    // with volume: the raw slope is biased and the controlled slope
    // must recover exactly 2.
    let mut rows = Vec::new();
    for i in 0..60 {
        let ofi = i as f64 - 30.0;
        let volume = 10.0 + 0.3 * ofi + (i % 3) as f64;
        let intensity = 5.0 + (i % 7) as f64;
        let volatility = 0.1 + (i % 5) as f64 * 0.01;
        rows.push(AblationRow {
            ofi,
            dmid: 2.0 * ofi + 0.5 * volume,
            volume,
            intensity,
            volatility,
        });
    }
    let r = ofi_response_ablation(&rows).expect("ablation must fit");
    assert_eq!(r.n, 60);
    assert!(
        (r.slope_controlled - 2.0).abs() < 1e-9,
        "controlled slope must be exactly 2, got {}",
        r.slope_controlled
    );
    assert!(
        (r.slope_raw - 2.0).abs() > 0.05,
        "raw slope must show the omitted-variable bias, got {}",
        r.slope_raw
    );
    assert!((r.r2_controlled - 1.0).abs() < 1e-9);
    assert!(r.se_controlled.is_finite());
    assert!(r.t_controlled.abs() > 10.0);
}

#[test]
fn stage3_toxicflow_5_ablation_needs_data() {
    assert!(ofi_response_ablation(&[]).is_none());
    let one = vec![AblationRow {
        ofi: 1.0,
        dmid: 2.0,
        volume: 3.0,
        intensity: 4.0,
        volatility: 0.1,
    }];
    assert!(ofi_response_ablation(&one).is_none());
}

/// Small toxic session: flat quotes, one-sided buying in the second
/// half pushing the mid up. The gate must fire more often at a lower
/// threshold than at a higher one.
fn toxic_session() -> (Vec<Quote>, Vec<Trade>) {
    let mut quotes = Vec::new();
    let mut trades = Vec::new();
    let mut mid = 100.0;
    let mut s: u64 = 7;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s as f64) / (u64::MAX as f64)
    };
    for i in 0..600 {
        let t = i as i64 * 1000;
        let informed = i > 300;
        let dir = if informed { 1.0 } else { 0.0 };
        quotes.push(Quote {
            time_ms: t,
            bid_px: mid - 0.05,
            bid_sz: 5.0,
            ask_px: mid + 0.05,
            ask_sz: 5.0,
        });
        for k in 0..4 {
            let buy = rnd() < 0.5 + 0.45 * dir;
            trades.push(Trade {
                time_ms: t + k * 200,
                price: mid,
                qty: 0.05,
                is_buyer_maker: !buy,
            });
        }
        mid += dir * 0.05 + (rnd() - 0.5) * 0.02;
    }
    (quotes, trades)
}

#[test]
fn stage3_toxicflow_5_gate_sensitivity_monotone() {
    let (q, t) = toxic_session();
    let lo = run_mm(
        &q,
        &t,
        &RobustnessConfig::new(50, 20, 10_000, 60, 0.6).mm_config(),
    )
    .expect("sim failed");
    let mid = run_mm(&q, &t, &RobustnessConfig::default_config().mm_config()).expect("sim failed");
    let hi = run_mm(
        &q,
        &t,
        &RobustnessConfig::new(50, 20, 10_000, 60, 0.8).mm_config(),
    )
    .expect("sim failed");
    assert!(lo.seconds_gated > 0, "low gate must fire");
    assert!(
        lo.seconds_gated >= mid.seconds_gated,
        "lower gate must fire at least as often: {} vs {}",
        lo.seconds_gated,
        mid.seconds_gated
    );
    assert!(
        mid.seconds_gated >= hi.seconds_gated,
        "higher gate must fire at most as often: {} vs {}",
        mid.seconds_gated,
        hi.seconds_gated
    );
    assert_eq!(lo.seconds_scored, mid.seconds_scored);
    assert_eq!(mid.seconds_scored, hi.seconds_scored);
}
