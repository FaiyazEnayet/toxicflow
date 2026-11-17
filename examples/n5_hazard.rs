//! N5 hazard-baseline runner: evaluates the frozen design grid on
//! real recorded sessions and writes a results packet.
//!
//! Usage:
//!   n5_hazard <grid.json> <out.json> <session_dir>...
//!
//! The grid file must match the library preregistered grid exactly,
//! and every session directory must be listed in the grid file under
//! `test_sessions`; otherwise the runner refuses to run. Every
//! session is evaluated and reported: the hazard dataset counts, the
//! per-fold train-only constant-hazard fits with held-out
//! waiting-time log-likelihood, and the per-fold and pooled
//! calibration tables. No test outcome is used for parameter
//! selection: the baseline has no tuned parameters, and the final
//! holdout is never used for tuning. Failed runs are retained in the
//! output with their reason instead of being dropped.

use toxicflow::evidence::{sha256_file, sha256_hex};
use toxicflow::hazard::{
    build_hazard_dataset, hazard_grid_from_json, preregistered_hazard_grid, walk_forward_hazard,
    HAZARD_RESULTS_SCHEMA,
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
        return Err("usage: n5_hazard <grid.json> <out.json> <session_dir>...".to_string());
    }
    let grid_path = args[1].clone();
    let out_path = args[2].clone();
    let session_dirs: Vec<String> = args[3..].to_vec();

    let grid_text =
        std::fs::read_to_string(&grid_path).map_err(|e| format!("cannot read grid: {e}"))?;
    let grid_hash = sha256_hex(grid_text.as_bytes());
    let file_grid = hazard_grid_from_json(&grid_text)?;
    let lib_grid = preregistered_hazard_grid();
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
    let cutoff = grid["cutoff_bars"].as_u64().unwrap_or(10) as usize;
    let horizons: Vec<usize> = grid["calibration_horizons_bars"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_u64().map(|h| h as usize))
                .collect()
        })
        .unwrap_or_else(|| vec![1, 2, 3, 5, 10]);

    let mut session_results = Vec::new();
    for d in &session_dirs {
        let s = load_session(d)?;
        println!("session {}", s.name);
        let ds = build_hazard_dataset(&s.quotes, bar_ms, cutoff);
        let c = &ds.counts;
        let wf = walk_forward_hazard(&ds.rows, n_splits, cutoff, &horizons);
        let wf_json = match &wf {
            Some(w) => serde_json::json!({
                "n_rows": w.n_rows,
                "n_test": w.n_test,
                "pooled_loglik": json_f64(w.pooled_loglik),
                "pooled_mean_loglik": json_f64(w.pooled_mean_loglik),
                "pooled_calibration": w.pooled_calibration.iter().map(|r| serde_json::json!({
                    "horizon_bars": r.horizon_bars,
                    "n_rows": r.n_rows,
                    "predicted_survival": json_f64(r.predicted_survival),
                    "observed_survival": json_f64(r.observed_survival),
                    "n_events_to_horizon": r.n_events_to_horizon,
                    "n_censored_to_horizon": r.n_censored_to_horizon,
                })).collect::<Vec<_>>(),
                "folds": w.folds.iter().map(|f| serde_json::json!({
                    "fold": f.fold,
                    "train_end": f.train_end,
                    "test_start": f.test_start,
                    "test_end": f.test_end,
                    "purge_bars": f.purge_bars,
                    "n_train": f.model.n_train,
                    "train_events_up": f.model.n_events_up,
                    "train_events_down": f.model.n_events_down,
                    "train_censored": f.model.n_censored,
                    "train_time_at_risk_bars": f.model.total_time_at_risk_bars,
                    "lambda_up_per_bar": json_f64(f.model.lambda_up_per_bar),
                    "lambda_down_per_bar": json_f64(f.model.lambda_down_per_bar),
                    "lambda_total_per_bar": json_f64(f.model.lambda_total_per_bar),
                    "n_test": f.n_test,
                    "test_loglik": json_f64(f.test_loglik),
                    "test_mean_loglik": json_f64(f.test_mean_loglik),
                    "calibration": f.calibration.iter().map(|r| serde_json::json!({
                        "horizon_bars": r.horizon_bars,
                        "n_rows": r.n_rows,
                        "predicted_survival": json_f64(r.predicted_survival),
                        "observed_survival": json_f64(r.observed_survival),
                        "n_events_to_horizon": r.n_events_to_horizon,
                        "n_censored_to_horizon": r.n_censored_to_horizon,
                    })).collect::<Vec<_>>(),
                })).collect::<Vec<_>>(),
            }),
            None => serde_json::json!({"failed": "walk-forward returned no evaluation"}),
        };
        println!(
            "  decision={} events={} (up {} down {}) censored={} (no_move {} incomplete {} gap {}) pooled_mean_loglik={}",
            c.n_decision,
            c.n_events,
            c.n_up,
            c.n_down,
            c.n_censored,
            c.censored_no_move,
            c.censored_incomplete,
            c.censored_gap,
            wf.as_ref()
                .map(|w| format!("{:.4}", w.pooled_mean_loglik))
                .unwrap_or_else(|| "-".to_string()),
        );
        session_results.push(serde_json::json!({
            "session": s.name,
            "input_hash": s.input_hash,
            "n_quotes": s.quotes.len(),
            "n_trades": s.trades.len(),
            "counts": {
                "n_bars": c.n_bars,
                "n_decision": c.n_decision,
                "n_events": c.n_events,
                "n_up": c.n_up,
                "n_down": c.n_down,
                "n_censored": c.n_censored,
                "censored_no_move": c.censored_no_move,
                "censored_incomplete": c.censored_incomplete,
                "censored_gap": c.censored_gap,
            },
            "walk_forward": wf_json,
        }));
    }

    let out = serde_json::json!({
        "schema_version": HAZARD_RESULTS_SCHEMA,
        "experiment_id": "n5-hazard-baseline",
        "grid_hash": grid_hash,
        "selection_rule": "No test outcome is used for parameter selection. The constant-hazard baseline has no tuned parameters and each walk-forward fold fits on its train rows only with a purge of one cutoff length. Every session is reported.",
        "bar_ms": bar_ms,
        "n_splits": n_splits,
        "cutoff_bars": cutoff,
        "calibration_horizons_bars": horizons,
        "censoring_rule": grid["censoring_rule"].as_str().unwrap_or(""),
        "notes": [
            "The hazard target is the waiting time to the first nonzero mid change, with up and down moves as competing events. It is a separate target from P1's binary/three-class direction labels; no hazard number is compared against a direction accuracy or baseline.",
            "Every row carries valid t_observe <= t_decision <= t_outcome, or explicit censoring with a recorded reason. No outcome window crosses a feed gap.",
            "Calibration compares the fitted survival exp(-lambda_total * t) against a Kaplan-Meier estimate on held-out rows at each horizon.",
        ],
        "sessions": session_results,
    });
    std::fs::write(&out_path, serde_json::to_string_pretty(&out).unwrap())
        .map_err(|e| format!("cannot write {out_path}: {e}"))?;
    println!("n5 results -> {out_path}");
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("n5_hazard: {e}");
        std::process::exit(1);
    }
}
