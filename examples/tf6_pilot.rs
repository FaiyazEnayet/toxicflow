//! TF6 session pilot runner: evaluates each recorded session
//! directory named on the command line with the same purged
//! walk-forward protocol as the single-session report (10 s bars,
//! 3-bar horizon, 4 splits), and writes the paired multi-session
//! pilot report as JSON.
//!
//! Usage: tf6_pilot <out.json> <session_dir> [<session_dir> ...]
//!
//! A session directory holds quotes.parquet and trades.parquet as
//! written by the recorder. Sessions are the caller's
//! responsibility: pass each independent recording exactly once.
//! The report counts sessions, never bars, as its sample size and
//! states plainly when fewer than the required 10 sessions exist.

use toxicflow::{predict, storage};

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        return Err(format!(
            "usage: {} <out.json> <session_dir> [<session_dir> ...]",
            args.first().map(String::as_str).unwrap_or("tf6_pilot")
        ));
    }
    let out = &args[1];
    let mut sessions = Vec::new();
    for dir in &args[2..] {
        let quotes = storage::read_quotes(&format!("{dir}/quotes.parquet"))
            .map_err(|e| format!("{dir}: {e}"))?;
        let trades = storage::read_trades(&format!("{dir}/trades.parquet"))
            .map_err(|e| format!("{dir}: {e}"))?;
        let bars = predict::build_dataset(&quotes, &trades, 10_000, 3);
        let label = dir
            .trim_end_matches('/')
            .rsplit('/')
            .next()
            .unwrap_or(dir.as_str())
            .to_string();
        match predict::evaluate_session(&label, &bars, 4, 3) {
            Some(s) => {
                println!(
                    "{label}: {} bars, {} OOS rows, accuracy {:.4} vs training-majority {:.4} (diff {:+.4})",
                    s.n_bars, s.n_oos, s.accuracy, s.train_majority_accuracy, s.paired_diff
                );
                sessions.push(s);
            }
            None => println!("{label}: not enough usable bars, skipped"),
        }
    }
    let pilot = predict::summarize_session_pilot(sessions, 10_000, 3, 4);
    std::fs::write(out, pilot.to_json()).map_err(|e| format!("{out}: {e}"))?;
    println!(
        "wrote {out}: {} sessions (requirement {} met: {})",
        pilot.n_sessions, pilot.required_sessions, pilot.session_requirement_met
    );
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("tf6_pilot: {e}");
        std::process::exit(1);
    }
}
