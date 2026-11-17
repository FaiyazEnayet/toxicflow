//! P2 normalized-flow ablation runner: evaluates the frozen design
//! grid on real recorded sessions and writes a results packet.
//!
//! Usage:
//!   p2_ablation <grid.json> <out.json> <session_dir>...
//!
//! The grid file must match the library preregistered grid exactly,
//! and every session directory must be listed in the grid file under
//! `test_sessions`; otherwise the runner refuses to run. Every grid
//! cell is evaluated and reported for every session on the same
//! decision rows: the B0-B5 baseline ladder, single-feature
//! additions on B3, leave-one-out from B5, and permutation / 5-bar
//! shift controls on the P2 block. No test outcome is used for
//! parameter selection: the default cell stays the reported cell
//! regardless of results. Failed runs are retained in the output
//! with their reason instead of being dropped.

use toxicflow::evidence::{sha256_file, sha256_hex};
use toxicflow::predict::{
    build_p2_dataset, p2_grid_from_json, preregistered_p2_grid, walk_forward_p2, P2Eval,
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

fn eval_json(ev: &P2Eval, include_models: bool) -> serde_json::Value {
    let mut j = serde_json::json!({
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

/// Session-equal pooled accumulator for one config: the per-session
/// accuracy, baseline, Brier and log-loss values in run order.
struct Pooled {
    id: String,
    accuracy: Vec<f64>,
    baseline: Vec<f64>,
    brier: Vec<f64>,
    log_loss: Vec<f64>,
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
        return Err("usage: p2_ablation <grid.json> <out.json> <session_dir>...".to_string());
    }
    let grid_path = args[1].clone();
    let out_path = args[2].clone();
    let session_dirs: Vec<String> = args[3..].to_vec();

    let grid_text =
        std::fs::read_to_string(&grid_path).map_err(|e| format!("cannot read grid: {e}"))?;
    let grid_hash = sha256_hex(grid_text.as_bytes());
    let file_grid = p2_grid_from_json(&grid_text)?;
    let lib_grid = preregistered_p2_grid();
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
    // accumulator per config id, in grid order.
    let mut pooled: Vec<Pooled> = lib_grid
        .iter()
        .map(|c| Pooled {
            id: c.id.clone(),
            accuracy: Vec::new(),
            baseline: Vec::new(),
            brier: Vec::new(),
            log_loss: Vec::new(),
        })
        .collect();

    let mut session_results = Vec::new();
    for d in &session_dirs {
        let s = load_session(d)?;
        println!("session {}", s.name);
        let ds = build_p2_dataset(&s.quotes, &s.trades, bar_ms, horizon);
        let mut cell_results = Vec::new();
        for (ci, cfg) in lib_grid.iter().enumerate() {
            let cols = cfg.col_indices();
            let ev = walk_forward_p2(&ds, &cols, cfg.control(), n_splits, horizon);
            match &ev {
                Some(e) => {
                    pooled[ci].accuracy.push(e.accuracy);
                    pooled[ci].baseline.push(e.baseline);
                    pooled[ci].brier.push(e.brier);
                    pooled[ci].log_loss.push(e.log_loss);
                    println!(
                        "  {:<22} acc {:.3} vs base {:.3} | brier {:.4} logloss {:.4} (n_oos {})",
                        cfg.id, e.accuracy, e.baseline, e.brier, e.log_loss, e.n_oos
                    );
                }
                None => println!("  {:<22} no evaluation (thin data)", cfg.id),
            }
            let cell_json = match &ev {
                Some(e) => eval_json(e, cfg.id == "b5-full"),
                None => serde_json::json!({"failed": "walk-forward returned no evaluation"}),
            };
            cell_results.push(serde_json::json!({
                "id": cfg.id,
                "kind": cfg.kind,
                "cols": cfg.cols,
                "control": cfg.control,
                "eval": cell_json,
            }));
        }
        session_results.push(serde_json::json!({
            "session": s.name,
            "input_hash": s.input_hash,
            "n_quotes": s.quotes.len(),
            "n_trades": s.trades.len(),
            "n_rows": ds.len(),
            "cells": cell_results,
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
                "id": p.id,
                "n_sessions": p.accuracy.len(),
                "accuracy_mean": json_f64(mean(&p.accuracy)),
                "baseline_mean": json_f64(mean(&p.baseline)),
                "brier_mean": json_f64(mean(&p.brier)),
                "log_loss_mean": json_f64(mean(&p.log_loss)),
            })
        })
        .collect();

    let out = serde_json::json!({
        "schema_version": "toxicflow.p2-results/v1",
        "experiment_id": "p2-normalize-flow",
        "grid_hash": grid_hash,
        "selection_rule": "No test outcome is used for parameter selection. The default configuration stays the reported configuration regardless of results. Every grid cell is reported.",
        "default_cell": "b5-full",
        "bar_ms": bar_ms,
        "horizon_bars": horizon,
        "n_splits": n_splits,
        "feature_version": "v4",
        "baselines": grid["baselines"].clone(),
        "censoring_rule": grid["censoring_rule"].as_str().unwrap_or(""),
        "notes": [
            "All configs score the same decision rows in each session, except the shift controls, which drop the first 5 rows of each test block (20 rows per session).",
            "The B0-B5 ladder is defined in the grid: B0 training-majority, B1 OFI only, B2 book state, B3 v3 frame, B4 B3 plus flow normalization, B5 full v4 frame.",
            "Permutation shuffles the P2 block within each fold's test block with a fixed seed; the fitted model is unchanged. Shift takes the P2 block from 5 bars earlier in the same test block.",
            "VPIN as a feature is absent (deferred to P3). Depth beyond the touch is absent (unavailable in archived data).",
            "Sessions are the six frozen 2026-11-11/12 test sessions, the same set as the E5, P1 and N5 runs. Results are associations on one venue and symbol, not causal effects.",
        ],
        "pooled": pooled_json,
        "sessions": session_results,
    });
    std::fs::write(&out_path, serde_json::to_string_pretty(&out).unwrap())
        .map_err(|e| format!("cannot write {out_path}: {e}"))?;
    println!("p2 results -> {out_path}");
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("p2_ablation: {e}");
        std::process::exit(1);
    }
}
