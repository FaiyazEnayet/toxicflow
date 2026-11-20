//! E4 reproduction runner: evaluates each recorded session directory
//! named on the command line with the same purged walk-forward
//! protocol as the single-session report (10 s bars, 3-bar horizon,
//! 4 splits), and reprints the historical and the repaired metrics
//! side by side with the protocols flagged incomparable.
//!
//! Usage: e4_reproduce <session_dir> [<session_dir> ...]
//!
//! A session directory holds quotes.parquet and trades.parquet as
//! written by the recorder. The historical block is a record of what
//! the old protocol reported from the same evaluation, not a number
//! the repaired block can be judged against.

use toxicflow::{predict, storage};

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        return Err(format!(
            "usage: {} <session_dir> [<session_dir> ...]",
            args.first().map(String::as_str).unwrap_or("e4_reproduce")
        ));
    }
    for dir in &args[1..] {
        let quotes = storage::read_quotes(&format!("{dir}/quotes.parquet"))
            .map_err(|e| format!("{dir}: {e}"))?;
        let trades = storage::read_trades(&format!("{dir}/trades.parquet"))
            .map_err(|e| format!("{dir}: {e}"))?;
        let bars =
            predict::build_dataset(&quotes, &trades, 10_000, 3).map_err(|e| e.to_string())?;
        let label = dir
            .trim_end_matches('/')
            .rsplit('/')
            .next()
            .unwrap_or(dir.as_str())
            .to_string();
        match predict::walk_forward_detailed(&bars, 4, 3) {
            Some(detail) => {
                println!("session {label}:");
                println!("{}", predict::reproduction_summary(&detail));
                for f in &detail.per_fold {
                    println!(
                        "  fold {}: accuracy {:.4} vs baseline {:.4}, brier {:.4}, log loss {:.4} (n={} pos={} neg={})",
                        f.fold, f.accuracy, f.baseline_accuracy, f.brier, f.log_loss, f.n_test, f.n_pos, f.n_neg
                    );
                }
            }
            None => println!("{label}: not enough usable bars, skipped"),
        }
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("e4_reproduce: {e}");
        std::process::exit(1);
    }
}
