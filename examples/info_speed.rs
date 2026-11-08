//! OFI-price response runner: estimates the cumulative mid-price
//! response to a one-standard-deviation OFI shock (both timing
//! conventions, with the model-free profile, innovation
//! projections, and the jump-term adjustment fit), for each
//! recorded session directory named on the command line, and
//! writes the multi-session report as JSON. Each session's
//! recorded trades are read as well when present: they feed the
//! robustness variants (trade price, activity strata, volume
//! clock) that sit alongside the headline curves in the report.
//!
//! Usage: info_speed <out.json> <session_dir> [<session_dir> ...]
//!
//! A session directory holds quotes.parquet and trades.parquet as
//! written by the recorder (only quotes are read here). The
//! estimand is observational: order flow and prices come from the
//! same quote stream and the sessions carry no news timestamps,
//! and the report says so in its own notes. Sessions are the
//! report's sample unit, never bars.

use toxicflow::{speed, storage};

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        return Err(format!(
            "usage: {} <out.json> <session_dir> [<session_dir> ...]",
            args.first().map(String::as_str).unwrap_or("info_speed")
        ));
    }
    let out = &args[1];
    let cfg = speed::SpeedConfig::default();
    let mut sessions = Vec::new();
    for dir in &args[2..] {
        let quotes = storage::read_quotes(&format!("{dir}/quotes.parquet"))
            .map_err(|e| format!("{dir}: {e}"))?;
        let trades = storage::read_trades(&format!("{dir}/trades.parquet")).ok();
        let label = dir
            .trim_end_matches('/')
            .rsplit('/')
            .next()
            .unwrap_or(dir.as_str())
            .to_string();
        match speed::analyze_session_speed_with_trades(&label, &quotes, trades.as_deref(), &cfg) {
            Some(s) => {
                match (s.tau_s, s.half_life_s) {
                    (Some(tau), Some(half)) => println!(
                        "{label}: {} bars, response {:+.2} bps at 300 s, tau {tau:.1} s, half-life {half:.1} s",
                        s.n_bars,
                        s.response_bps.last().copied().unwrap_or(f64::NAN)
                    ),
                    _ => println!(
                        "{label}: {} bars, response {:+.2} bps at 300 s, exponential adjustment fit not reported: {}",
                        s.n_bars,
                        s.response_bps.last().copied().unwrap_or(f64::NAN),
                        s.fit_failure.as_deref().unwrap_or("unknown reason")
                    ),
                }
                sessions.push(s);
            }
            None => println!("{label}: not enough usable bars, skipped"),
        }
    }
    let report = speed::summarize_speed(sessions, &cfg);
    std::fs::write(out, report.to_json()).map_err(|e| format!("{out}: {e}"))?;
    println!(
        "wrote {out}: {} sessions, {} with a reported tau",
        report.summary.n_sessions, report.summary.n_fit_ok
    );
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("info_speed: {e}");
        std::process::exit(1);
    }
}
