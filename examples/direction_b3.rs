//! B3 development runner for the registered 60-second direction
//! confirmation study (`docs/direction_confirmation_protocol.md`).
//!
//! Fits the four ridge-logistic families on the six archive sessions
//! plus campaign slots 01-06, walks the fit pool forward over
//! development slots 07-10 under the 660-second embargo, and reports
//! the eight development rows with the section 6 selection. It never
//! touches the sealed holdout: every session resolves through the
//! manifest, slots 11-20 are an error, and this command scores
//! nothing on them.
//!
//! Usage:
//!   direction_b3 <out.json> <session_dir> [<session_dir> ...]
//!
//! Each session directory holds quotes.parquet and trades.parquet
//! and is named by its session ID. Exactly the six archives and
//! campaign slots 01-10 must be supplied.

use std::collections::HashSet;
use std::process::ExitCode;
use toxicflow::direction::DirectionError;
use toxicflow::direction::{
    load_session, run_development, BootstrapIntervals, SlotInput, CAMPAIGN_SLOT_IDS,
};
use toxicflow::direction_manifest::{
    parse_session_start_unix, DirectionManifest, ARCHIVE_SESSION_IDS, PROTOCOL_DIGEST,
};

fn is_campaign_id(id: &str) -> bool {
    // The five dated archive sessions share the campaign ID shape
    // (session-YYYYMMDD-HHMMSS-utc); the archive list takes
    // precedence, exactly as the manifest classifies them.
    !ARCHIVE_SESSION_IDS.contains(&id) && parse_session_start_unix(id).is_some()
}

fn content_hash(dir: &str) -> Result<String, DirectionError> {
    let mut bytes = std::fs::read(format!("{dir}/quotes.parquet"))
        .map_err(|e| DirectionError::Storage(e.to_string()))?;
    let mut trades = std::fs::read(format!("{dir}/trades.parquet"))
        .map_err(|e| DirectionError::Storage(e.to_string()))?;
    bytes.append(&mut trades);
    Ok(toxicflow::direction_manifest::sha256_hex(&bytes))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 17 {
        eprintln!(
            "usage: direction_b3 <out.json> <session_dir> [<session_dir> ...] (need 16 sessions)"
        );
        return ExitCode::from(2);
    }
    let out = args[0].clone();
    let dirs = &args[1..];
    if let Err(e) = run(&out, dirs) {
        eprintln!("direction_b3 failed: {e}");
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

fn run(out: &str, dirs: &[String]) -> Result<(), DirectionError> {
    // Session IDs are the directory basenames.
    let mut ids: Vec<String> = Vec::new();
    for dir in dirs {
        let base = dir
            .trim_end_matches('/')
            .rsplit('/')
            .next()
            .unwrap_or(dir)
            .to_string();
        ids.push(base);
    }
    let mut seen: HashSet<&str> = HashSet::new();
    for id in &ids {
        if !seen.insert(id.as_str()) {
            return Err(DirectionError::UnexpectedSession(format!(
                "duplicate session directory: {id}"
            )));
        }
    }
    // The supplied campaign sessions must be exactly slots 01-10 of
    // the counted campaign, in chronological order.
    let mut campaign: Vec<String> = ids
        .iter()
        .filter(|id| is_campaign_id(id))
        .cloned()
        .collect();
    campaign.sort_by_key(|id| parse_session_start_unix(id).unwrap());
    let expected: Vec<String> = CAMPAIGN_SLOT_IDS[..10]
        .iter()
        .map(|s| s.to_string())
        .collect();
    if campaign != expected {
        return Err(DirectionError::UnexpectedSession(format!(
            "campaign sessions must be exactly slots 01-10; got {campaign:?}"
        )));
    }
    // The supplied archives must be exactly the six registered ones.
    let mut archives: Vec<String> = ids
        .iter()
        .filter(|id| ARCHIVE_SESSION_IDS.contains(&id.as_str()))
        .cloned()
        .collect();
    archives.sort();
    let mut expected_archives: Vec<String> =
        ARCHIVE_SESSION_IDS.iter().map(|s| s.to_string()).collect();
    expected_archives.sort();
    if archives != expected_archives {
        return Err(DirectionError::UnexpectedSession(format!(
            "archive sessions must be exactly the six registered archives; got {archives:?}"
        )));
    }
    if ids.len() != 16 {
        return Err(DirectionError::UnexpectedSession(format!(
            "expected 16 sessions, got {}",
            ids.len()
        )));
    }
    // The manifest is the only resolution path. Content hashes
    // attach to each slot identity as the protocol requires.
    let campaign_refs: Vec<&str> = campaign.iter().map(|s| s.as_str()).collect();
    let mut manifest = DirectionManifest::new(&campaign_refs)?;
    for (id, dir) in ids.iter().zip(dirs.iter()) {
        if is_campaign_id(id) {
            manifest.attach_hash(id, &content_hash(dir)?)?;
        }
    }
    // Load every session through the allowlist (slots 11-20 would
    // fail here with SealedHoldout).
    let dir_by_id: std::collections::HashMap<&str, &str> = ids
        .iter()
        .zip(dirs.iter())
        .map(|(id, dir)| (id.as_str(), dir.as_str()))
        .collect();
    let mut archive_frames = Vec::new();
    for id in &expected_archives {
        let frame = load_session(&manifest, id, dir_by_id[id.as_str()])?;
        println!(
            "archive {id}: {} bars, {} eligible origins",
            frame.n_bars,
            toxicflow::direction::eligible_origins(&frame).len()
        );
        archive_frames.push(frame);
    }
    let mut slot_inputs: Vec<SlotInput> = Vec::new();
    let mut slot_frames = Vec::new();
    for (slot, id) in expected.iter().enumerate() {
        let frame = load_session(&manifest, id, dir_by_id[id.as_str()])?;
        println!(
            "slot {:02} {id}: {} bars, {} observed seconds, {} eligible origins",
            slot + 1,
            frame.n_bars,
            frame.n_observed_secs(),
            toxicflow::direction::eligible_origins(&frame).len()
        );
        slot_frames.push(frame);
    }
    for (slot, frame) in slot_frames.iter().enumerate() {
        slot_inputs.push(SlotInput {
            slot: slot + 1,
            session_id: expected[slot].clone(),
            frame,
        });
    }
    let archive_refs: Vec<&toxicflow::direction::SessionFrame> = archive_frames.iter().collect();
    let outcome = run_development(&archive_refs, &slot_inputs)?;
    println!("\ndevelopment rows (pooled over slots 07-10):");
    println!(
        "{:>8} {:>14} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "family", "rule", "Delta", "Gamma", "BA", "acc", "S"
    );
    for row in &outcome.rows {
        println!(
            "{:>8} {:>14} {:>9.5} {:>9.5} {:>9.5} {:>9.5} {:>9.3}",
            row.family.name(),
            row.rule.name(),
            row.pooled.delta,
            row.pooled.gamma,
            row.pooled.balanced_accuracy,
            row.pooled.accuracy,
            row.pooled.s
        );
    }
    println!(
        "\nselected: {} under {}",
        outcome.selection.selected.name(),
        outcome.selection.selected_rule.name()
    );
    let report = build_report(&outcome);
    std::fs::write(out, serde_json::to_string_pretty(&report).unwrap())
        .map_err(|e| DirectionError::Storage(e.to_string()))?;
    println!("wrote {out}");
    Ok(())
}

fn build_report(outcome: &toxicflow::direction::DevelopmentOutcome) -> serde_json::Value {
    use serde_json::json;
    let rows: Vec<serde_json::Value> = outcome
        .rows
        .iter()
        .map(|row| {
            let boot = |b: &BootstrapIntervals| match b {
                Some(((dl, du), (gl, gu))) => json!({
                    "delta_ci": [dl, du],
                    "gamma_ci": [gl, gu],
                }),
                None => json!({"suppressed": true}),
            };
            json!({
                "family": row.family.name(),
                "rule": row.rule.name(),
                "n_scored": row.pooled.n_scored,
                "accuracy": row.pooled.accuracy,
                "baseline_accuracy": row.pooled.baseline_accuracy,
                "balanced_accuracy": row.pooled.balanced_accuracy,
                "delta": row.pooled.delta,
                "gamma": row.pooled.gamma,
                "s": row.pooled.s,
                "per_session": row.per_session.iter().map(|s| json!({
                    "session": s.session,
                    "n_scored": s.n_scored,
                    "accuracy": s.accuracy,
                    "baseline_accuracy": s.baseline_accuracy,
                    "balanced_accuracy": s.balanced_accuracy,
                    "delta": s.delta,
                    "gamma": s.gamma,
                    "s": s.s,
                })).collect::<Vec<_>>(),
                "bootstrap_300s": boot(&row.boot_300s),
                "bootstrap_600s": boot(&row.boot_600s),
            })
        })
        .collect();
    let sel_row = outcome
        .rows
        .iter()
        .find(|r| {
            r.family == outcome.selection.selected && r.rule == outcome.selection.selected_rule
        })
        .unwrap();
    // Validation-vs-bar comparison for the selected row. This is an
    // assessment against the preregistered B4 win bar, not a B4
    // decision: the go/no-go stays with the owner.
    let delta_ok = sel_row.pooled.delta >= 0.010;
    let ba_ok = sel_row.pooled.balanced_accuracy >= 0.520;
    let bounds_ok = [sel_row.boot_300s, sel_row.boot_600s]
        .iter()
        .all(|b| match b {
            Some(((dl, _), (gl, _))) => *dl > 0.0 && *gl > 0.0,
            None => false,
        });
    let b4_competitive = delta_ok && ba_ok && bounds_ok;
    json!({
        "schema": "toxicflow.direction-b3/v1",
        "protocol_digest": PROTOCOL_DIGEST,
        "protocol": "docs/direction_confirmation_protocol.md",
        "archives": ARCHIVE_SESSION_IDS,
        "slots_fit": CAMPAIGN_SLOT_IDS[..6],
        "slots_development": CAMPAIGN_SLOT_IDS[6..10],
        "slots_sealed": CAMPAIGN_SLOT_IDS[10..],
        "note": "Development results are descriptive and selection-biased; they are never confirmatory evidence (protocol section 6).",
        "boundaries": outcome.boundaries.iter().map(|b| json!({
            "eval_slot": b.eval_slot,
            "eval_session": b.eval_session,
            "fit_sessions": b.fit_sessions,
            "embargo_cutoff_ms": b.embargo_cutoff_ms,
            "n_fit_origins": b.n_fit_origins,
            "prevalence": b.prevalence,
            "majority_label": b.majority_label,
        })).collect::<Vec<_>>(),
        "rows": rows,
        "selection": {
            "families": outcome.selection.families.iter().map(|f| json!({
                "family": f.family.name(),
                "selected_rule": f.selected_rule.name(),
                "s": f.s,
                "delta": f.delta,
                "gamma": f.gamma,
                "clears_r": f.clears_r,
                "clears_f": f.clears_f,
                "qualifies": f.qualifies,
            })).collect::<Vec<_>>(),
            "selected_family": outcome.selection.selected.name(),
            "selected_rule": outcome.selection.selected_rule.name(),
        },
        "assessment": {
            "selected_delta_vs_bar_0.010": delta_ok,
            "selected_ba_vs_bar_0.520": ba_ok,
            "selected_bootstrap_lower_bounds_positive_both_block_lengths": bounds_ok,
            "b4_looks_competitive_on_validation": b4_competitive,
            "note": "B4 is a single locked run on slots 11-20; this flag only compares validation numbers with the preregistered win bar.",
        },
    })
}
