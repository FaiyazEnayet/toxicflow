//! P3 conditional-toxicity pilot runner: evaluates the frozen design
//! grid on real recorded sessions and writes a results packet.
//!
//! Usage:
//!   p3_pilot <grid.json> <out.json> <session_dir>...
//!
//! The grid file must match the library preregistered grid exactly,
//! and every session directory must be listed in the grid file under
//! `test_sessions`; otherwise the runner refuses to run. Every grid
//! cell is evaluated for every outcome (direction, absolute return,
//! spread widening, adverse markout) on the same decision rows:
//! the reference ladder (majority, v3, v4 controls-only), the paired
//! VPIN cells (v3 plus VPIN, VPIN only, controls plus VPIN), and the
//! permutation and 5-bar shift controls on the VPIN block. No test
//! outcome is used for parameter selection: the default cell stays
//! the reported cell regardless of results. Failed runs are retained
//! in the output with their reason instead of being dropped.
//!
//! Regime tables compare controls+VPIN against controls-only within
//! activity (trade rate) and depth (touch depth) tertiles learned on
//! training rows only. Dependence accounting is a session-block
//! bootstrap over the six sessions on the paired log-loss
//! differential (10,000 resamples, fixed seed).

use toxicflow::evidence::{sha256_file, sha256_hex};
use toxicflow::predict::{
    build_p3_dataset, p3_grid_from_json, p3_tertile, preregistered_p3_grid, walk_forward_p3, P3Bar,
    P3Eval, P3Outcome,
};
use toxicflow::storage;

const OUTCOMES: [P3Outcome; 4] = [
    P3Outcome::Direction,
    P3Outcome::AbsRet,
    P3Outcome::SpreadWide,
    P3Outcome::AdverseMarkout,
];

/// Session-block bootstrap resamples (fixed, deterministic).
const BOOTSTRAP_RESAMPLES: usize = 10_000;
/// Seed for the session-block bootstrap.
const BOOTSTRAP_SEED: u64 = 0xB005_7B05_0000_00A3;

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

fn json_f64(v: f64) -> serde_json::Value {
    if v.is_finite() {
        serde_json::json!(v)
    } else {
        serde_json::Value::Null
    }
}

fn eval_json(ev: &P3Eval, include_models: bool) -> serde_json::Value {
    let mut j = serde_json::json!({
        "outcome": ev.outcome.id(),
        "n_labeled": ev.n_labeled,
        "n_oos": ev.n_oos,
        "n_pos_oos": ev.n_pos_oos,
        "n_neg_oos": ev.n_neg_oos,
        "n_unscored": ev.n_unscored,
        "accuracy": json_f64(ev.accuracy),
        "baseline": json_f64(ev.baseline),
        "balanced_accuracy": json_f64(ev.balanced_accuracy),
        "brier": json_f64(ev.brier),
        "log_loss": json_f64(ev.log_loss),
        "per_fold": ev.per_fold.iter().map(|f| serde_json::json!({
            "fold": f.fold,
            "n_train": f.n_train,
            "n_test": f.n_test,
            "n_scored": f.n_scored,
            "accuracy": json_f64(f.accuracy),
            "baseline_accuracy": json_f64(f.baseline_accuracy),
            "brier": json_f64(f.brier),
            "log_loss": json_f64(f.log_loss),
            "activity_cut1": json_f64(f.activity_cut1),
            "activity_cut2": json_f64(f.activity_cut2),
            "depth_cut1": json_f64(f.depth_cut1),
            "depth_cut2": json_f64(f.depth_cut2),
            "absret_median": json_f64(f.absret_median),
        })).collect::<Vec<_>>(),
    });
    if include_models {
        j["fold_models"] = serde_json::json!(ev
            .fold_models
            .iter()
            .map(|(w, b)| serde_json::json!({ "weights": w, "intercept": b }))
            .collect::<Vec<_>>());
    }
    j
}

struct SessionData {
    name: String,
    input_hash: String,
    quotes: Vec<toxicflow::metrics::Quote>,
    trades: Vec<toxicflow::metrics::Trade>,
}

/// Session-equal pooled accumulator for one (outcome, config):
/// the per-session accuracy, baseline, Brier and log-loss values
/// in run order.
struct Pooled {
    outcome: &'static str,
    id: String,
    accuracy: Vec<f64>,
    baseline: Vec<f64>,
    brier: Vec<f64>,
    log_loss: Vec<f64>,
}

/// One scored row's paired losses, for the regime tables.
#[derive(Clone)]
struct RegimeRow {
    activity: usize,
    depth: usize,
    ll_controls: f64,
    ll_vpin: f64,
    brier_controls: f64,
    brier_vpin: f64,
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

/// Deterministic xorshift stream for the session-block bootstrap.
struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        let mut s = self.0 | 1;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        self.0 = s;
        s
    }
}

/// Session-block bootstrap on paired per-session differentials:
/// returns (mean, 2.5th percentile, 97.5th percentile, fraction of
/// resampled means below zero).
fn block_bootstrap(deltas: &[f64]) -> (f64, f64, f64, f64) {
    let n = deltas.len();
    let mean = deltas.iter().sum::<f64>() / n as f64;
    let mut rng = XorShift(BOOTSTRAP_SEED);
    let mut means = Vec::with_capacity(BOOTSTRAP_RESAMPLES);
    for _ in 0..BOOTSTRAP_RESAMPLES {
        let mut s = 0.0;
        for _ in 0..n {
            s += deltas[(rng.next() % n as u64) as usize];
        }
        means.push(s / n as f64);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo = means[(0.025 * BOOTSTRAP_RESAMPLES as f64) as usize];
    let hi = means[(0.975 * BOOTSTRAP_RESAMPLES as f64) as usize];
    let frac_below = means.iter().filter(|&&m| m < 0.0).count() as f64 / BOOTSTRAP_RESAMPLES as f64;
    (mean, lo, hi, frac_below)
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        return Err("usage: p3_pilot <grid.json> <out.json> <session_dir>...".to_string());
    }
    let grid_path = args[1].clone();
    let out_path = args[2].clone();
    let session_dirs: Vec<String> = args[3..].to_vec();

    let grid_text =
        std::fs::read_to_string(&grid_path).map_err(|e| format!("cannot read grid: {e}"))?;
    let grid_hash = sha256_hex(grid_text.as_bytes());
    let file_grid = p3_grid_from_json(&grid_text)?;
    let lib_grid = preregistered_p3_grid();
    if file_grid != lib_grid {
        return Err(
            "grid file does not match the preregistered library grid; refusing to run".to_string(),
        );
    }
    let grid: serde_json::Value =
        serde_json::from_str(&grid_text).map_err(|e| format!("grid JSON must parse: {e}"))?;
    let grid_sessions: Vec<String> = grid["test_sessions"]
        .as_array()
        .ok_or("grid must list test_sessions")?
        .iter()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();
    for d in &session_dirs {
        if !grid_sessions.contains(&session_name(d)) {
            return Err(format!(
                "session {} is not in the grid file test_sessions",
                session_name(d)
            ));
        }
    }
    let bar_ms = grid["bar_ms"].as_i64().unwrap_or(10_000);
    let horizon = grid["horizon_bars"].as_u64().unwrap_or(3) as usize;
    let n_splits = grid["n_splits"].as_u64().unwrap_or(4) as usize;

    // Pooled means across sessions, session-equal weighting: one
    // accumulator per (outcome, config), in grid order.
    let mut pooled: Vec<Pooled> = Vec::new();
    for outcome in OUTCOMES {
        for cfg in &lib_grid {
            pooled.push(Pooled {
                outcome: outcome.id(),
                id: cfg.id.clone(),
                accuracy: Vec::new(),
                baseline: Vec::new(),
                brier: Vec::new(),
                log_loss: Vec::new(),
            });
        }
    }
    // Paired per-session log-loss differentials (controls+VPIN minus
    // controls-only) per outcome, for the session-block bootstrap.
    let mut paired_ll: Vec<Vec<f64>> = vec![Vec::new(); OUTCOMES.len()];
    // Regime rows per outcome, pooled across sessions.
    let mut regime_rows: Vec<Vec<RegimeRow>> = vec![Vec::new(); OUTCOMES.len()];
    // Per-session regime counts per outcome: [outcome][session][var][tertile].
    let mut regime_n: Vec<Vec<[[usize; 3]; 2]>> = vec![Vec::new(); OUTCOMES.len()];

    let mut session_results = Vec::new();
    for d in &session_dirs {
        let s = load_session(d)?;
        println!("session {}", s.name);
        let ds: Vec<P3Bar> = build_p3_dataset(&s.quotes, &s.trades, bar_ms, horizon);
        println!("  rows {}", ds.len());
        let mut outcome_results = Vec::new();
        for (oi, outcome) in OUTCOMES.iter().enumerate() {
            let mut cell_results = Vec::new();
            let mut evals: Vec<Option<P3Eval>> = Vec::new();
            for (ci, cfg) in lib_grid.iter().enumerate() {
                let cols = cfg.col_indices();
                let ev = walk_forward_p3(&ds, &cols, cfg.control(), n_splits, horizon, *outcome);
                match &ev {
                    Some(e) => {
                        pooled[oi * lib_grid.len() + ci].accuracy.push(e.accuracy);
                        pooled[oi * lib_grid.len() + ci].baseline.push(e.baseline);
                        pooled[oi * lib_grid.len() + ci].brier.push(e.brier);
                        pooled[oi * lib_grid.len() + ci].log_loss.push(e.log_loss);
                        println!(
                            "  {:<10} {:<22} acc {:.3} vs base {:.3} | brier {:.4} logloss {:.4} (n_oos {})",
                            outcome.id(),
                            cfg.id,
                            e.accuracy,
                            e.baseline,
                            e.brier,
                            e.log_loss,
                            e.n_oos
                        );
                    }
                    None => println!(
                        "  {:<10} {:<22} no evaluation (thin data)",
                        outcome.id(),
                        cfg.id
                    ),
                }
                let cell_json = match &ev {
                    Some(e) => eval_json(e, cfg.id == "controls-vpin"),
                    None => serde_json::json!({"failed": "walk-forward returned no evaluation"}),
                };
                cell_results.push(serde_json::json!({
                    "id": cfg.id,
                    "kind": cfg.kind,
                    "cols": cfg.cols,
                    "control": cfg.control,
                    "eval": cell_json,
                }));
                evals.push(ev);
            }
            // Paired differential and regime rows from the two Off
            // configs that share every scored row.
            let ci_controls = lib_grid.iter().position(|c| c.id == "b5-controls").unwrap();
            let ci_vpin = lib_grid
                .iter()
                .position(|c| c.id == "controls-vpin")
                .unwrap();
            let mut counts = [[0usize; 3]; 2];
            if let (Some(controls), Some(vpin)) =
                (evals[ci_controls].as_ref(), evals[ci_vpin].as_ref())
            {
                paired_ll[oi].push(vpin.log_loss - controls.log_loss);
                let mut rows: Vec<RegimeRow> = Vec::new();
                for (pc, pv) in controls.predictions.iter().zip(vpin.predictions.iter()) {
                    assert_eq!(pc.row, pv.row, "paired configs must score the same rows");
                    assert_eq!(pc.fold, pv.fold);
                    let bar = &ds[pc.row];
                    let fold = &controls.per_fold[pc.fold];
                    let activity =
                        p3_tertile(bar.trade_rate, fold.activity_cut1, fold.activity_cut2);
                    let depth = p3_tertile(bar.depth_mean, fold.depth_cut1, fold.depth_cut2);
                    let ll = |p: f64, y: u8| {
                        let pc = p.clamp(1e-15, 1.0 - 1e-15);
                        if y == 1 {
                            -pc.ln()
                        } else {
                            -(1.0 - pc).ln()
                        }
                    };
                    let brier = |p: f64, y: u8| (p - f64::from(y)).powi(2);
                    counts[0][activity] += 1;
                    counts[1][depth] += 1;
                    rows.push(RegimeRow {
                        activity,
                        depth,
                        ll_controls: ll(pc.proba, pc.target),
                        ll_vpin: ll(pv.proba, pv.target),
                        brier_controls: brier(pc.proba, pc.target),
                        brier_vpin: brier(pv.proba, pv.target),
                    });
                }
                regime_rows[oi].extend(rows);
            }
            regime_n[oi].push(counts);
            outcome_results.push(serde_json::json!({
                "outcome": outcome.id(),
                "n_rows": ds.len(),
                "cells": cell_results,
            }));
        }
        session_results.push(serde_json::json!({
            "session": s.name,
            "input_hash": s.input_hash,
            "n_quotes": s.quotes.len(),
            "n_trades": s.trades.len(),
            "n_rows": ds.len(),
            "outcomes": outcome_results,
        }));
    }

    let mean = |v: &[f64]| {
        if v.is_empty() {
            f64::NAN
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    };
    let pooled_json: Vec<serde_json::Value> = pooled
        .iter()
        .map(|p| {
            serde_json::json!({
                "outcome": p.outcome,
                "id": p.id,
                "n_sessions": p.accuracy.len(),
                "accuracy_mean": json_f64(mean(&p.accuracy)),
                "baseline_mean": json_f64(mean(&p.baseline)),
                "brier_mean": json_f64(mean(&p.brier)),
                "log_loss_mean": json_f64(mean(&p.log_loss)),
            })
        })
        .collect();

    // Regime tables: pooled across sessions, per outcome.
    let mut regimes_json = Vec::new();
    for (oi, outcome) in OUTCOMES.iter().enumerate() {
        let rows = &regime_rows[oi];
        let mut cells = Vec::new();
        for (vi, var) in ["activity", "depth"].iter().enumerate() {
            for t in 0..3 {
                let cell_rows: Vec<&RegimeRow> = rows
                    .iter()
                    .filter(|r| {
                        if vi == 0 {
                            r.activity == t
                        } else {
                            r.depth == t
                        }
                    })
                    .collect();
                let n = cell_rows.len();
                let ll_delta = if n == 0 {
                    f64::NAN
                } else {
                    cell_rows
                        .iter()
                        .map(|r| r.ll_vpin - r.ll_controls)
                        .sum::<f64>()
                        / n as f64
                };
                let brier_delta = if n == 0 {
                    f64::NAN
                } else {
                    cell_rows
                        .iter()
                        .map(|r| r.brier_vpin - r.brier_controls)
                        .sum::<f64>()
                        / n as f64
                };
                cells.push(serde_json::json!({
                    "regime": var,
                    "tertile": t,
                    "n": n,
                    "n_per_session": regime_n[oi].iter().map(|c| c[vi][t]).collect::<Vec<_>>(),
                    "log_loss_delta": json_f64(ll_delta),
                    "brier_delta": json_f64(brier_delta),
                }));
            }
        }
        regimes_json.push(serde_json::json!({
            "outcome": outcome.id(),
            "comparison": "controls-vpin minus b5-controls, pooled scored rows",
            "cells": cells,
        }));
    }

    // Session-block bootstrap on the paired log-loss differential.
    let mut bootstrap_json = Vec::new();
    for (oi, outcome) in OUTCOMES.iter().enumerate() {
        let (m, lo, hi, frac) = block_bootstrap(&paired_ll[oi]);
        println!(
            "bootstrap {:<15} paired log-loss delta mean {:.4} 95% CI [{:.4}, {:.4}] frac<0 {:.3}",
            outcome.id(),
            m,
            lo,
            hi,
            frac
        );
        bootstrap_json.push(serde_json::json!({
            "outcome": outcome.id(),
            "comparison": "controls-vpin minus b5-controls, per-session log-loss means",
            "n_sessions": paired_ll[oi].len(),
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": format!("0x{:X}", BOOTSTRAP_SEED),
            "per_session_deltas": paired_ll[oi],
            "mean": json_f64(m),
            "ci_lo": json_f64(lo),
            "ci_hi": json_f64(hi),
            "frac_below_zero": json_f64(frac),
        }));
    }

    let out = serde_json::json!({
        "schema_version": "toxicflow.p3-results/v1",
        "experiment_id": "p3-conditional-toxicity",
        "grid_hash": grid_hash,
        "selection_rule": "No test outcome is used for parameter selection. The default configuration stays the reported configuration regardless of results. Every grid cell is reported for every outcome.",
        "default_cell": "controls-vpin",
        "bar_ms": bar_ms,
        "horizon_bars": horizon,
        "n_splits": n_splits,
        "feature_version": "v5",
        "bucket_vol_btc": grid["bucket_vol_btc"].clone(),
        "vpin_window": grid["vpin_window"].clone(),
        "vpin_lag_bars": grid["vpin_lag_bars"].clone(),
        "permutation_seed": grid["permutation_seed"].clone(),
        "baselines": grid["baselines"].clone(),
        "decision_rule": grid["decision_rule"].as_str().unwrap_or(""),
        "censoring_rule": grid["censoring_rule"].as_str().unwrap_or(""),
        "notes": [
            "All configs score the same decision rows in each session and outcome, except the shift controls, which drop the first 5 rows of each test block (20 rows per session).",
            "The adverse-markout outcome scores only rows whose bar carries nonzero net signed flow, so its row count is smaller; every config compared on that outcome shares the same rows.",
            "The absolute-return outcome labels each fold from that fold's training median of absolute forward return; the threshold is never learned from test rows.",
            "The P3 row set is smaller than the P2 row set: a row also needs a full 20-bucket VPIN window completed before its bar close, so early bars are outside the set for every config. P3 headline numbers are therefore not directly comparable with P2 headline numbers; the paired comparisons within P3 are exact.",
            "The adverse-markout outcome excludes zero net signed-flow rows; on these six sessions that exclusion removed no rows (every scored bar carries nonzero net flow), so its row set equals the other outcomes.",
            "Regime tertiles are learned on training rows only, per fold. Regime cells report pooled scored rows with per-session counts; no per-cell inference is drawn: subsamples are small and multiple testing applies.",
            "The bootstrap resamples whole sessions, so within-session dependence stays intact; with six sessions the intervals are wide by construction.",
            "Sessions are the six frozen 2026-11-11/12 test sessions, the same set as the E5, P1, N5 and P2 runs. Results are associations on one venue and symbol, not causal effects, and no cross-session improvement claim is made: that stays blocked until P3::confirm.",
        ],
        "pooled": pooled_json,
        "regimes": regimes_json,
        "bootstrap": bootstrap_json,
        "sessions": session_results,
    });
    std::fs::write(&out_path, serde_json::to_string_pretty(&out).unwrap())
        .map_err(|e| format!("cannot write {out_path}: {e}"))?;
    println!("p3 results -> {out_path}");
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("p3_pilot: {e}");
        std::process::exit(1);
    }
}
