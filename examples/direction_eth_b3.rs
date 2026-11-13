//! ETH B3 transfer validation runner: the BTC-selected configuration
//! (family F under the p>0.50 rule) validated on the ETHUSDT campaign
//! sessions, under `docs/direction_eth_transfer_protocol.md`.
//!
//! Only family F is evaluated, with coefficients refit on ETH
//! training data at each boundary: ETH slots 01-06 predict slot 07,
//! then the fit pool grows by only the earlier campaign slots for
//! slots 08-10, under the 660-second embargo. No family is selected
//! here and no threshold is tuned; the configuration arrives frozen
//! from the BTC study. There are no archive sessions on the ETH
//! side: the six archives were BTC-only. Slots 11-20 are sealed for
//! fit use and are never scored here; supplying one is an error.
//!
//! Usage:
//!   direction_eth_b3 <out.json> <session_dir> [<session_dir> ...]
//!
//! Each session directory holds quotes.parquet and trades.parquet
//! and is named by its session ID. Exactly ETH campaign slots 01-10
//! must be supplied.

use std::collections::HashSet;
use std::process::ExitCode;
use toxicflow::direction::DirectionError;
use toxicflow::direction::{
    load_session_unsealed, run_transfer_validation, BootstrapIntervals, Family, Rule, SlotInput,
};
use toxicflow::direction_manifest::{
    parse_session_start_unix, EthDirectionManifest, ETH_FIT_DEV_SLOT_IDS, ETH_PROTOCOL_DIGEST,
    ETH_SEALED_SLOT_IDS,
};

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
    if args.len() != 11 {
        eprintln!("usage: direction_eth_b3 <out.json> <session_dir> [<session_dir> ...] (need 10 sessions)");
        return ExitCode::from(2);
    }
    let out = args[0].clone();
    let dirs = &args[1..];
    if let Err(e) = run(&out, dirs) {
        eprintln!("direction_eth_b3 failed: {e}");
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
    // The manifest is the only resolution path: it names all twenty
    // slots, so a sealed slot is an error, not a result.
    let mut all_ids: Vec<&str> = Vec::new();
    all_ids.extend_from_slice(&ETH_FIT_DEV_SLOT_IDS);
    all_ids.extend_from_slice(&ETH_SEALED_SLOT_IDS);
    let mut manifest = EthDirectionManifest::new(&all_ids).map_err(DirectionError::Manifest)?;
    for id in &ids {
        manifest
            .resolve_for_fit(id)
            .map_err(DirectionError::Manifest)?;
    }
    // The supplied sessions must be exactly ETH slots 01-10, in
    // chronological order.
    let mut sorted: Vec<String> = ids.clone();
    sorted.sort_by_key(|id| parse_session_start_unix(id).unwrap_or(0));
    let expected: Vec<String> = ETH_FIT_DEV_SLOT_IDS.iter().map(|s| s.to_string()).collect();
    if sorted != expected {
        return Err(DirectionError::UnexpectedSession(format!(
            "ETH sessions must be exactly slots 01-10; got {sorted:?}"
        )));
    }
    // Content hashes attach to each slot identity.
    let dir_by_id: std::collections::HashMap<&str, &str> = ids
        .iter()
        .zip(dirs.iter())
        .map(|(id, dir)| (id.as_str(), dir.as_str()))
        .collect();
    let mut slot_hashes: Vec<(String, String)> = Vec::new();
    for id in &expected {
        let hash = content_hash(dir_by_id[id.as_str()])?;
        manifest
            .attach_hash(id, &hash)
            .map_err(DirectionError::Manifest)?;
        slot_hashes.push((id.clone(), hash));
    }
    let mut slot_frames = Vec::new();
    for (slot, id) in expected.iter().enumerate() {
        let frame = load_session_unsealed(id, dir_by_id[id.as_str()])?;
        println!(
            "slot {:02} {id}: {} bars, {} observed seconds, {} eligible origins",
            slot + 1,
            frame.n_bars,
            frame.n_observed_secs(),
            toxicflow::direction::eligible_origins(&frame).len()
        );
        slot_frames.push(frame);
    }
    let mut slot_inputs: Vec<SlotInput> = Vec::new();
    for (slot, frame) in slot_frames.iter().enumerate() {
        slot_inputs.push(SlotInput {
            slot: slot + 1,
            session_id: expected[slot].clone(),
            frame,
        });
    }
    // No archives on the ETH side: the six archives were BTC-only.
    let empty_archives: Vec<&toxicflow::direction::SessionFrame> = Vec::new();
    let (boundaries, row) = run_transfer_validation(
        &empty_archives,
        &slot_inputs,
        Family::F,
        Rule::P50,
    )?;
    println!("\ntransfer validation row, family F under p>0.50 (pooled over slots 07-10):");
    println!(
        "  accuracy {:.5} | baseline {:.5} | Delta {:+.5} | BA {:.5} | Gamma {:+.5}",
        row.pooled.accuracy,
        row.pooled.baseline_accuracy,
        row.pooled.delta,
        row.pooled.balanced_accuracy,
        row.pooled.gamma
    );
    let delta_ok = row.pooled.delta >= 0.010;
    let ba_ok = row.pooled.balanced_accuracy >= 0.520;
    println!(
        "\nB3 gates: Delta >= +0.010: {} | BA >= 0.520: {} | passes: {}",
        delta_ok,
        ba_ok,
        delta_ok && ba_ok
    );
    let report = build_report(&boundaries, &row, &slot_hashes);
    std::fs::write(out, serde_json::to_string_pretty(&report).unwrap())
        .map_err(|e| DirectionError::Storage(e.to_string()))?;
    println!("wrote {out}");
    Ok(())
}

fn build_report(
    boundaries: &[toxicflow::direction::BoundaryDetail],
    row: &toxicflow::direction::DevRow,
    slot_hashes: &[(String, String)],
) -> serde_json::Value {
    use serde_json::json;
    let boot = |b: &BootstrapIntervals| match b {
        Some(((dl, du), (gl, gu))) => json!({
            "delta_ci": [dl, du],
            "gamma_ci": [gl, gu],
        }),
        None => json!({"suppressed": true}),
    };
    let delta_ok = row.pooled.delta >= 0.010;
    let ba_ok = row.pooled.balanced_accuracy >= 0.520;
    json!({
        "schema": "toxicflow.direction-eth-b3/v1",
        "protocol_digest": ETH_PROTOCOL_DIGEST,
        "protocol": "docs/direction_eth_transfer_protocol.md",
        "design": {
            "asset": "ETHUSDT",
            "kind": "cross-asset transfer validation (option b): the BTC-selected configuration validated on ETH, no family selection, no threshold tuning",
            "frozen_configuration": {
                "family": "F",
                "rule": "p>0.50",
                "ridge_lambda": 0.01,
                "embargo_s": 660,
                "target": "60-second post-close direction on 1-second bars",
            },
            "note": "Only family F under p>0.50 is evaluated. Coefficients, scalers, AR coefficients, thresholds and the training-majority rule are refit on ETH training data at each boundary; no BTC fitted value is carried over. Validation results are descriptive; they are never confirmatory evidence. Slots 11-20 are sealed for fit use and were not scored.",
        },
        "archives": [],
        "archives_note": "No archive sessions on the ETH side: the six archives were BTC-only. The ETH fit pool starts with six four-hour recordings.",
        "slots_fit": ETH_FIT_DEV_SLOT_IDS[..6],
        "slots_validation": ETH_FIT_DEV_SLOT_IDS[6..10],
        "slots_sealed": ETH_SEALED_SLOT_IDS,
        "slot_content_hashes": slot_hashes.iter().map(|(id, h)| json!({
            "session": id, "content_hash": h,
        })).collect::<Vec<_>>(),
        "boundaries": boundaries.iter().map(|b| json!({
            "eval_slot": b.eval_slot,
            "eval_session": b.eval_session,
            "fit_sessions": b.fit_sessions,
            "embargo_cutoff_ms": b.embargo_cutoff_ms,
            "n_fit_origins": b.n_fit_origins,
            "prevalence": b.prevalence,
            "majority_label": b.majority_label,
        })).collect::<Vec<_>>(),
        "row": {
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
        },
        "assessment": {
            "delta_vs_bar_0.010": delta_ok,
            "ba_vs_bar_0.520": ba_ok,
            "passes_b3": delta_ok && ba_ok,
            "note": "B3 has no confidence-interval gate; it is a screening step and B4 provides the confirmation. If either bar fails, the outcome is a failed cross-asset validation, published as such, and ETH slots 11-20 are not opened for this experiment. There is no fallback family or threshold.",
        },
        "observations": {
            "unused_family_r_fit": "While the earlier all-family harness was being built, family R stalled at one boundary (eval slot 08, fit pool slots 01-07): the ridge fit reached gradient inf-norm 1.615e-8 against the 1e-8 tolerance and then stopped improving. Family R is not evaluated in this study, so this is not a competitor result and nothing was dropped from a tournament; no threshold was changed. The numerical investigation is a separate workstream.",
        },
        "qualification": "BTC and ETH were recorded during the same November 9-12, 2026 campaign. An ETH success demonstrates cross-asset generalization under substantially shared market conditions, not fully independent replication across different market regimes.",
    })
}
