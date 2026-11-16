//! P1 target-semantics runner: evaluates the frozen design grid on
//! real recorded sessions and writes a results packet.
//!
//! Usage:
//!   p1_report <grid.json> <out.json> <session_dir>...
//!
//! The grid file must match the library preregistered grid exactly,
//! and every session directory must be listed in the grid file under
//! `test_sessions`; otherwise the runner refuses to run. Every grid
//! cell is evaluated and reported for every session: the preserved
//! baseline (binary sign at the cell horizon), the next-nonzero
//! track, and the three-class track. No test outcome is used for
//! parameter selection: the default cell stays the reported cell
//! regardless of results. Failed runs are retained in the output
//! with their reason instead of being dropped.

use toxicflow::evidence::{sha256_file, sha256_hex};
use toxicflow::predict::{
    build_p1_dataset, p1_grid_from_json, preregistered_p1_grid, walk_forward_detailed,
    walk_forward_three_class, P1_NONZERO_CUTOFF_BARS,
};
use toxicflow::storage;

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

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        return Err("usage: p1_report <grid.json> <out.json> <session_dir>...".to_string());
    }
    let grid_path = args[1].clone();
    let out_path = args[2].clone();
    let session_dirs: Vec<String> = args[3..].to_vec();

    let grid_text =
        std::fs::read_to_string(&grid_path).map_err(|e| format!("cannot read grid: {e}"))?;
    let grid_hash = sha256_hex(grid_text.as_bytes());
    let file_grid = p1_grid_from_json(&grid_text)?;
    let lib_grid = preregistered_p1_grid();
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
    let n_splits = grid["n_splits"].as_u64().unwrap_or(4) as usize;
    let cutoff = grid["nonzero_cutoff_bars"]
        .as_u64()
        .unwrap_or(P1_NONZERO_CUTOFF_BARS as u64) as usize;

    let mut session_results = Vec::new();
    for d in &session_dirs {
        let s = load_session(d)?;
        println!("session {}", s.name);
        let mut cell_results = Vec::new();
        for cfg in &lib_grid {
            let ds = build_p1_dataset(&s.quotes, &s.trades, bar_ms, *cfg, cutoff);
            let c = &ds.counts;
            let baseline = walk_forward_detailed(&ds.baseline_bars(), n_splits, cfg.horizon_bars);
            let nonzero = walk_forward_detailed(&ds.nonzero_bars(), n_splits, cutoff);
            let three =
                walk_forward_three_class(&ds.three_class_rows(), n_splits, cfg.horizon_bars);
            let baseline_json = match &baseline {
                Some(detail) => {
                    let r = &detail.report;
                    serde_json::json!({
                        "n_labeled": r.n_labeled,
                        "n_oos": r.n_oos,
                        "n_pos_oos": r.n_pos_oos,
                        "n_neg_oos": r.n_neg_oos,
                        "n_unscored": r.n_unscored,
                        "accuracy": json_f64(r.accuracy),
                        "baseline": json_f64(r.baseline),
                        "balanced_accuracy": json_f64(r.balanced_accuracy),
                        "brier": json_f64(r.brier),
                        "log_loss": json_f64(r.log_loss),
                    })
                }
                None => serde_json::json!({"failed": "walk-forward returned no evaluation"}),
            };
            let nonzero_json = match &nonzero {
                Some(detail) => {
                    let r = &detail.report;
                    serde_json::json!({
                        "n_labeled": r.n_labeled,
                        "n_oos": r.n_oos,
                        "n_pos_oos": r.n_pos_oos,
                        "n_neg_oos": r.n_neg_oos,
                        "n_unscored": r.n_unscored,
                        "accuracy": json_f64(r.accuracy),
                        "baseline": json_f64(r.baseline),
                        "balanced_accuracy": json_f64(r.balanced_accuracy),
                        "brier": json_f64(r.brier),
                        "log_loss": json_f64(r.log_loss),
                    })
                }
                None => serde_json::json!({"failed": "walk-forward returned no evaluation"}),
            };
            let three_json = match &three {
                Some(rep) => serde_json::json!({
                    "n_labeled": rep.n_labeled,
                    "n_oos": rep.n_oos,
                    "n_down_oos": rep.n_down_oos,
                    "n_flat_oos": rep.n_flat_oos,
                    "n_up_oos": rep.n_up_oos,
                    "n_unscored": rep.n_unscored,
                    "accuracy": json_f64(rep.accuracy),
                    "baseline": json_f64(rep.baseline),
                    "balanced_accuracy": json_f64(rep.balanced_accuracy),
                    "brier": json_f64(rep.brier),
                    "log_loss": json_f64(rep.log_loss),
                    "confusion": rep.confusion,
                    "per_fold": rep.per_fold.iter().map(|f| serde_json::json!({
                        "fold": f.fold,
                        "n_train": f.n_train,
                        "train_down": f.train_down,
                        "train_flat": f.train_flat,
                        "train_up": f.train_up,
                        "train_majority_label": f.train_majority_label,
                        "n_test": f.n_test,
                        "accuracy": json_f64(f.accuracy),
                        "baseline_accuracy": json_f64(f.baseline_accuracy),
                        "brier": json_f64(f.brier),
                        "log_loss": json_f64(f.log_loss),
                        "confusion": f.confusion,
                    })).collect::<Vec<_>>(),
                }),
                None => serde_json::json!({"failed": "walk-forward returned no evaluation"}),
            };
            println!(
                "  {} h={} eps={}: decision={} base_acc={} nz_acc={} 3acc={}",
                cfg.id(),
                cfg.horizon_bars,
                cfg.flat_eps,
                c.n_decision,
                baseline
                    .as_ref()
                    .map(|d| format!("{:.3}", d.report.accuracy))
                    .unwrap_or_else(|| "-".to_string()),
                nonzero
                    .as_ref()
                    .map(|d| format!("{:.3}", d.report.accuracy))
                    .unwrap_or_else(|| "-".to_string()),
                three
                    .as_ref()
                    .map(|r| format!("{:.3}", r.accuracy))
                    .unwrap_or_else(|| "-".to_string()),
            );
            cell_results.push(serde_json::json!({
                "id": cfg.id(),
                "horizon_bars": cfg.horizon_bars,
                "flat_eps": cfg.flat_eps,
                "is_default": cfg.is_default(),
                "counts": {
                    "n_bars": c.n_bars,
                    "n_decision": c.n_decision,
                    "baseline_labeled": c.baseline_labeled,
                    "baseline_censored": c.baseline_censored,
                    "baseline_zero_fwd": c.baseline_zero_fwd,
                    "nonzero_labeled": c.nonzero_labeled,
                    "nonzero_up": c.nonzero_up,
                    "nonzero_down": c.nonzero_down,
                    "nonzero_censored_no_move": c.nonzero_censored_no_move,
                    "nonzero_censored_incomplete": c.nonzero_censored_incomplete,
                    "nonzero_censored_gap": c.nonzero_censored_gap,
                    "three_labeled": c.three_labeled,
                    "three_down": c.three_down,
                    "three_flat": c.three_flat,
                    "three_up": c.three_up,
                    "three_censored_incomplete": c.three_censored_incomplete,
                    "three_censored_gap": c.three_censored_gap,
                },
                "baseline": baseline_json,
                "nonzero": nonzero_json,
                "three_class": three_json,
            }));
        }
        session_results.push(serde_json::json!({
            "session": s.name,
            "input_hash": s.input_hash,
            "n_quotes": s.quotes.len(),
            "n_trades": s.trades.len(),
            "cells": cell_results,
        }));
    }

    let out = serde_json::json!({
        "schema_version": "toxicflow.p1-results/v1",
        "experiment_id": "p1-target-semantics",
        "grid_hash": grid_hash,
        "selection_rule": "No test outcome is used for parameter selection. The default configuration stays the reported configuration regardless of results. Every grid cell is reported.",
        "default_cell": "h3-e1e-5",
        "bar_ms": bar_ms,
        "n_splits": n_splits,
        "nonzero_cutoff_bars": cutoff,
        "censoring_rule": grid["censoring_rule"].as_str().unwrap_or(""),
        "notes": [
            "Baseline is the preserved binary-sign task at each cell horizon; its features and labels are unchanged.",
            "The nonzero track trains and scores only on rows where a move occurs within the cutoff, so its accuracy is conditional on a move occurring; the censored share is reported next to it.",
            "Train class priors are train-only in every fold; the reported baseline in each cell is the training-majority predictor.",
        ],
        "sessions": session_results,
    });
    std::fs::write(&out_path, serde_json::to_string_pretty(&out).unwrap())
        .map_err(|e| format!("cannot write {out_path}: {e}"))?;
    println!("p1 results -> {out_path}");
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("p1_report: {e}");
        std::process::exit(1);
    }
}
