//! E5 robustness runner: evaluates the preregistered parameter grid
//! on real recorded sessions and writes a results packet with a
//! manifest and hash per configuration.
//!
//! Usage:
//!   e5_robustness <grid.json> <out.json> <train_dir>... -- <test_dir>...
//!
//! The grid file must match the library preregistered grid exactly,
//! and every session directory must be listed in the grid file under
//! its role (train or test); otherwise the runner refuses to run.
//! Train sessions calibrate the VPIN gate (90th percentile of the
//! pooled train VPIN series under the default configuration); test
//! sessions are untouched chronologically later sessions and are
//! reported even when the result is null. Failed runs are retained
//! in the output with their reason instead of being dropped.

use std::collections::BTreeMap;

use toxicflow::evidence::{sha256_file, sha256_hex};
use toxicflow::metrics::{ofi_bars, ofi_events, volume_buckets, vpin, FlowBar};
use toxicflow::robustness::{
    calibrate_gate_train_only, flow_bars, grid_from_json, kyle_lambda_robust, lambda_tail_summary,
    lead_lag_response, ofi_response_ablation, preregistered_grid, shift_flow_bars, AblationRow,
    RobustnessConfig,
};
use toxicflow::{predict, storage, strategy};

fn session_name(dir: &str) -> String {
    dir.trim_end_matches('/')
        .rsplit('/')
        .next()
        .unwrap_or(dir)
        .to_string()
}

fn input_hash(dir: &str) -> Result<String, String> {
    let (tsha, _) = sha256_file(std::path::Path::new(&format!("{dir}/trades.parquet")))
        .map_err(|e| format!("{dir}: {e}"))?;
    let (qsha, _) = sha256_file(std::path::Path::new(&format!("{dir}/quotes.parquet")))
        .map_err(|e| format!("{dir}: {e}"))?;
    Ok(sha256_hex(
        format!("trades:{tsha}\nquotes:{qsha}\n").as_bytes(),
    ))
}

fn percentile(mut xs: Vec<f64>, p: f64) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[(p * (xs.len() - 1) as f64).round() as usize]
}

fn json_f64(v: f64) -> serde_json::Value {
    if v.is_finite() {
        serde_json::json!(v)
    } else {
        serde_json::Value::Null
    }
}

struct SessionData {
    name: String,
    input_hash: String,
    quotes: Vec<toxicflow::metrics::Quote>,
    trades: Vec<toxicflow::metrics::Trade>,
}

fn load_session(dir: &str) -> Result<SessionData, String> {
    let quotes = storage::read_quotes(&format!("{dir}/quotes.parquet"))
        .map_err(|e| format!("{dir}: {e}"))?;
    let trades = storage::read_trades(&format!("{dir}/trades.parquet"))
        .map_err(|e| format!("{dir}: {e}"))?;
    Ok(SessionData {
        name: session_name(dir),
        input_hash: input_hash(dir)?,
        quotes,
        trades,
    })
}

/// Per-bar unsigned volume and trade count on the OFI bar grid.
fn bar_volume_intensity(
    trades: &[toxicflow::metrics::Trade],
    t0: i64,
    bar_ms: i64,
    n: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut vol = vec![0.0; n];
    let mut cnt = vec![0.0; n];
    for t in trades {
        if t.time_ms < t0 {
            continue;
        }
        let i = ((t.time_ms - t0) / bar_ms) as usize;
        if i < n {
            vol[i] += t.qty;
            cnt[i] += 1.0;
        }
    }
    (vol, cnt)
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let sep = args.iter().position(|a| a == "--").ok_or_else(|| {
        "usage: e5_robustness <grid.json> <out.json> <train_dir>... -- <test_dir>...".to_string()
    })?;
    if args.len() < 5 || sep < 4 {
        return Err(
            "usage: e5_robustness <grid.json> <out.json> <train_dir>... -- <test_dir>..."
                .to_string(),
        );
    }
    let grid_path = args[1].clone();
    let out_path = args[2].clone();
    let train_dirs: Vec<String> = args[3..sep].to_vec();
    let test_dirs: Vec<String> = args[sep + 1..].to_vec();
    if train_dirs.is_empty() || test_dirs.is_empty() {
        return Err("need at least one train and one test session".to_string());
    }

    let grid_text =
        std::fs::read_to_string(&grid_path).map_err(|e| format!("cannot read grid: {e}"))?;
    let grid_hash = sha256_hex(grid_text.as_bytes());
    let file_grid = grid_from_json(&grid_text)?;
    let lib_grid = preregistered_grid();
    if file_grid != lib_grid {
        return Err(
            "grid file does not match the preregistered library grid; refusing to run".to_string(),
        );
    }
    let grid: serde_json::Value =
        serde_json::from_str(&grid_text).map_err(|e| format!("grid JSON must parse: {e}"))?;
    let grid_train: Vec<String> = grid["train_sessions"]
        .as_array()
        .ok_or("grid must list train_sessions")?
        .iter()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();
    let grid_test: Vec<String> = grid["test_sessions"]
        .as_array()
        .ok_or("grid must list test_sessions")?
        .iter()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();
    for d in &train_dirs {
        if !grid_train.contains(&session_name(d)) {
            return Err(format!(
                "train session {} is not in the grid file",
                session_name(d)
            ));
        }
    }
    for d in &test_dirs {
        if !grid_test.contains(&session_name(d)) {
            return Err(format!(
                "test session {} is not in the grid file",
                session_name(d)
            ));
        }
    }
    let shift_ms: i64 = grid["nulls"]["timestamp_shift_ms"]
        .as_i64()
        .unwrap_or(120_000);
    let shift_bars: isize = grid["nulls"]["shift_bars"].as_i64().unwrap_or(3) as isize;
    let calib_pct: f64 = grid["calibration"]["percentile"].as_f64().unwrap_or(0.9);

    // Train-only calibration: pool the default-configuration VPIN
    // series of the train sessions and take the percentile. Test
    // sessions are never read here.
    let mut train_sessions = Vec::new();
    let mut train_vpin: Vec<f64> = Vec::new();
    for d in &train_dirs {
        let s = load_session(d)?;
        let bk = volume_buckets(&s.trades, RobustnessConfig::default_config().n_buckets);
        let vp = vpin(&bk, RobustnessConfig::default_config().vpin_window);
        train_vpin.extend(vp.iter().map(|(_, v)| *v));
        train_sessions.push(s);
    }
    let calibrated_gate = calibrate_gate_train_only(&train_vpin, calib_pct)
        .ok_or("train VPIN series is empty; cannot calibrate the gate")?;
    println!(
        "calibrated gate (train-only, p{} of {} train VPIN values): {:.4}",
        calib_pct,
        train_vpin.len(),
        calibrated_gate
    );

    let mut test_sessions = Vec::new();
    for d in &test_dirs {
        test_sessions.push(load_session(d)?);
    }

    let bar_grid = [5_000i64, 10_000, 30_000];
    let mut session_results = Vec::new();
    for (role, sessions) in [("train", &train_sessions), ("test", &test_sessions)] {
        for s in sessions {
            println!("session {} ({role})", s.name);
            let span_s = if s.trades.is_empty() {
                0.0
            } else {
                (s.trades.last().unwrap().time_ms - s.trades.first().unwrap().time_ms) as f64
                    / 1000.0
            };
            // Per-bar-length diagnostics, computed once per session.
            let mut bar_diags = Vec::new();
            for bar_ms in bar_grid {
                let bars = flow_bars(&s.trades, &s.quotes, bar_ms);
                let lam = kyle_lambda_robust(&bars, 60);
                let lam_sum = lambda_tail_summary(&lam);
                let events = ofi_events(&s.quotes);
                let ob = ofi_bars(&s.quotes, &events, bar_ms);
                if ob.is_empty() {
                    bar_diags.push(serde_json::json!({
                        "bar_ms": bar_ms,
                        "status": "failed",
                        "reason": "no OFI bars",
                    }));
                    continue;
                }
                let n = ob.len();
                // ret[i] is the midprice return realized during bar
                // i, measured end to end from the last mid of bar
                // i-1 to the last mid of bar i, so lag 0 pairs OFI
                // with its own bar's return and negative lags are
                // genuine controls.
                let mut ret_bps = Vec::new();
                let mut ofi = Vec::new();
                for i in 1..n {
                    ofi.push(ob[i].1);
                    ret_bps.push((ob[i].2 / ob[i - 1].2 - 1.0) * 1e4);
                }
                let lead_lag: Vec<serde_json::Value> = lead_lag_response(&ofi, &ret_bps, 5)
                    .iter()
                    .map(|p| {
                        serde_json::json!({
                            "lag": p.lag,
                            "slope_bps_per_sd": json_f64(p.slope_bps_per_sd),
                            "n": p.n,
                        })
                    })
                    .collect();
                // Timestamp-shift nulls: shift trade times, and shift
                // signed flow bars, against the price changes.
                let shifted_trades: Vec<toxicflow::metrics::Trade> = s
                    .trades
                    .iter()
                    .map(|t| {
                        let mut t = *t;
                        t.time_ms += shift_ms;
                        t
                    })
                    .collect();
                let shifted_bars = flow_bars(&shifted_trades, &s.quotes, bar_ms);
                let shifted_mean =
                    lambda_tail_summary(&kyle_lambda_robust(&shifted_bars, 60)).mean_lambda;
                let bar_shifted = shift_flow_bars(&bars, shift_bars);
                let bar_shifted_mean =
                    lambda_tail_summary(&kyle_lambda_robust(&bar_shifted, 60)).mean_lambda;
                // Ablation rows, aligned on bar time.
                let by_time: BTreeMap<i64, FlowBar> =
                    bars.iter().map(|b| (b.time_ms, *b)).collect();
                let (vol, cnt) = bar_volume_intensity(&s.trades, ob[0].0, bar_ms, n);
                let mut rows = Vec::new();
                let mut trailing: Vec<f64> = Vec::new();
                for (i, b) in ob.iter().enumerate() {
                    let Some(f) = by_time.get(&b.0) else {
                        continue;
                    };
                    trailing.push(f.dmid.abs());
                    if trailing.len() > 10 {
                        trailing.remove(0);
                    }
                    let vola = trailing.iter().sum::<f64>() / trailing.len() as f64;
                    rows.push(AblationRow {
                        ofi: b.1,
                        dmid: f.dmid,
                        volume: vol[i],
                        intensity: cnt[i],
                        volatility: vola,
                    });
                }
                let ablation = ofi_response_ablation(&rows).map(|r| {
                    serde_json::json!({
                        "slope_raw": json_f64(r.slope_raw),
                        "slope_controlled": json_f64(r.slope_controlled),
                        "se_controlled": json_f64(r.se_controlled),
                        "t_controlled": json_f64(r.t_controlled),
                        "r2_controlled": json_f64(r.r2_controlled),
                        "n": r.n,
                    })
                });
                bar_diags.push(serde_json::json!({
                    "bar_ms": bar_ms,
                    "n_bars": bars.len(),
                    "lambda": {
                        "n_windows": lam_sum.n_windows,
                        "n_with_lambda": lam_sum.n_with_lambda,
                        "n_tail": lam_sum.n_tail,
                        "tail_share": json_f64(lam_sum.tail_share),
                        "mean_lambda": json_f64(lam_sum.mean_lambda),
                        "winsorized_mean_lambda": json_f64(lam_sum.winsorized_mean_lambda),
                        "median_abs_lambda": json_f64(lam_sum.median_abs_lambda),
                        "p99_abs_lambda": json_f64(lam_sum.p99_abs_lambda),
                        "total_n_valid": lam_sum.total_n_valid,
                    },
                    "lead_lag": lead_lag,
                    "shift_null_trade_ms": {
                        "shift_ms": shift_ms,
                        "orig_mean_lambda": json_f64(lam_sum.mean_lambda),
                        "shifted_mean_lambda": json_f64(shifted_mean),
                    },
                    "shift_null_bars": {
                        "shift_bars": shift_bars,
                        "orig_mean_lambda": json_f64(lam_sum.mean_lambda),
                        "shifted_mean_lambda": json_f64(bar_shifted_mean),
                    },
                    "ablation": ablation,
                }));
            }
            // Full grid on this session.
            let mut grid_results = Vec::new();
            for cfg in &lib_grid {
                let manifest = cfg.manifest(&s.name, &s.input_hash);
                let bk = volume_buckets(&s.trades, cfg.n_buckets);
                let vp = vpin(&bk, cfg.vpin_window);
                let vals: Vec<f64> = vp.iter().map(|(_, v)| *v).collect();
                let vpin_json = if vals.is_empty() {
                    serde_json::json!({"n_buckets": bk.len(), "mean": null, "p90": null, "max": null, "last": null})
                } else {
                    serde_json::json!({
                        "n_buckets": bk.len(),
                        "mean": vals.iter().sum::<f64>() / vals.len() as f64,
                        "p90": percentile(vals.clone(), 0.9),
                        "max": vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                        "last": vals.last().copied().unwrap_or(f64::NAN),
                    })
                };
                let bars = flow_bars(&s.trades, &s.quotes, cfg.bar_ms);
                let lam_sum = lambda_tail_summary(&kyle_lambda_robust(&bars, cfg.lambda_window));
                let lambda_json = serde_json::json!({
                    "n_windows": lam_sum.n_windows,
                    "n_with_lambda": lam_sum.n_with_lambda,
                    "n_tail": lam_sum.n_tail,
                    "tail_share": json_f64(lam_sum.tail_share),
                    "mean_lambda": json_f64(lam_sum.mean_lambda),
                    "winsorized_mean_lambda": json_f64(lam_sum.winsorized_mean_lambda),
                    "median_abs_lambda": json_f64(lam_sum.median_abs_lambda),
                    "p99_abs_lambda": json_f64(lam_sum.p99_abs_lambda),
                    "total_n_valid": lam_sum.total_n_valid,
                });
                match strategy::run_mm(&s.quotes, &s.trades, &cfg.mm_config()) {
                    Some(mm) => {
                        let mut entry = serde_json::json!({
                            "manifest": manifest,
                            "status": "ok",
                            "vpin": vpin_json,
                            "lambda": lambda_json,
                            "mm": {
                                "pnl_open": mm.pnl_open,
                                "pnl_gated": mm.pnl_gated,
                                "seconds": mm.seconds,
                                "seconds_gated": mm.seconds_gated,
                                "seconds_scored": mm.seconds_scored,
                                "turnover": mm.turnover,
                                "inventory_end": mm.inventory_end,
                            },
                        });
                        // On test sessions, the default configuration
                        // is also scored at the train-calibrated gate.
                        if role == "test" && *cfg == RobustnessConfig::default_config() {
                            let mut cal_cfg = cfg.mm_config();
                            cal_cfg.vpin_threshold = calibrated_gate;
                            if let Some(mm_cal) = strategy::run_mm(&s.quotes, &s.trades, &cal_cfg) {
                                entry["mm_calibrated_gate"] = serde_json::json!({
                                    "gate": calibrated_gate,
                                    "pnl_open": mm_cal.pnl_open,
                                    "pnl_gated": mm_cal.pnl_gated,
                                    "seconds": mm_cal.seconds,
                                    "seconds_gated": mm_cal.seconds_gated,
                                    "seconds_scored": mm_cal.seconds_scored,
                                    "turnover": mm_cal.turnover,
                                    "inventory_end": mm_cal.inventory_end,
                                });
                            }
                        }
                        grid_results.push(entry);
                    }
                    None => {
                        grid_results.push(serde_json::json!({
                            "manifest": manifest,
                            "status": "failed",
                            "reason": "run_mm returned none: session too short or empty",
                            "vpin": vpin_json,
                            "lambda": lambda_json,
                        }));
                    }
                }
            }
            session_results.push(serde_json::json!({
                "session": s.name,
                "role": role,
                "input_hash": s.input_hash,
                "n_trades": s.trades.len(),
                "n_quotes": s.quotes.len(),
                "span_s": span_s,
                "bar_diagnostics": bar_diags,
                "grid": grid_results,
            }));
        }
    }

    // Walk-forward direction check on the test sessions (unchanged
    // protocol, 10 s bars, 3-bar horizon, 4 splits): reported
    // alongside the grid, never used to pick a configuration.
    let mut walk_forward = Vec::new();
    for s in &test_sessions {
        let ds = predict::build_dataset(&s.quotes, &s.trades, 10_000, 3);
        match predict::walk_forward_detailed(&ds, 4, 3) {
            Some(detail) => walk_forward.push(serde_json::json!({
                "session": s.name,
                "status": "ok",
                "accuracy": detail.report.accuracy,
                "baseline": detail.report.baseline,
                "brier": detail.report.brier,
                "log_loss": detail.report.log_loss,
                "n_labeled": detail.report.n_labeled,
                "n_oos": detail.report.n_oos,
                "n_folds": detail.report.n_folds,
            })),
            None => walk_forward.push(serde_json::json!({
                "session": s.name,
                "status": "failed",
                "reason": "not enough usable bars",
            })),
        }
    }

    let out = serde_json::json!({
        "experiment_id": "e5-robustness",
        "schema_version": "toxicflow.e5-results/v1",
        "grid_hash": grid_hash,
        "calibration": {
            "rule": "gate is the 90th percentile of the pooled train-session VPIN series under the default configuration",
            "percentile": calib_pct,
            "calibrated_gate": calibrated_gate,
            "n_train_vpin_values": train_vpin.len(),
        },
        "train_sessions": grid_train,
        "test_sessions": grid_test,
        "sessions": session_results,
        "walk_forward_test": walk_forward,
        "selection_rule": "No test outcome is used for parameter selection. The default configuration stays the reported configuration regardless of results.",
    });
    let out_text =
        serde_json::to_string_pretty(&out).map_err(|e| format!("cannot serialize results: {e}"))?;
    std::fs::write(&out_path, out_text).map_err(|e| format!("cannot write {out_path}: {e}"))?;
    println!("e5 results -> {out_path}");
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("e5_robustness: {e}");
        std::process::exit(1);
    }
}
