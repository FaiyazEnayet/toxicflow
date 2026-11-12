//! B4 confirmation runner for the registered 60-second direction
//! confirmation study (`docs/direction_confirmation_protocol.md`).
//!
//! Freezes the single B3-selected artifact (family F under rule
//! p>0.50, refit on the six archive sessions plus campaign slots
//! 01-10 under the frozen 660-second embargo) and scores the sealed
//! holdout, slots 11-20, exactly once. The holdout resolves only
//! through the confirmation path, which requires the registered
//! protocol digest; no other variant is scored on the holdout, not
//! even as a diagnostic. A deterministic replay from the serialized
//! artifact verifies the run before the report is written.
//!
//! Usage:
//!   direction_b4 <out.json> <session_dir> [<session_dir> ...]
//!
//! Each session directory holds quotes.parquet and trades.parquet
//! and is named by its session ID. Exactly the six archives and all
//! twenty campaign slots must be supplied. The report is written to
//! `docs/direction_confirmation_report.json` (the protocol section 10
//! filename) when invoked with that path.

use std::collections::HashSet;
use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};
use toxicflow::direction::DirectionError;
use toxicflow::direction::{
    classify_confirmation, freeze_artifact, load_session, load_session_for_confirmation,
    run_confirmation, score_with_artifact, ConfirmationOutcomeKind, Family, FrameCalc,
    HoldoutInput, Rule, CAMPAIGN_SLOT_IDS, CONFIRMATION_FIRST_SLOT, EMBARGO_SECS,
};
use toxicflow::direction_manifest::{
    parse_session_start_unix, sha256_hex, DirectionManifest, ARCHIVE_SESSION_IDS, PROTOCOL_DIGEST,
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
    Ok(sha256_hex(&bytes))
}

fn code_hash() -> Result<String, DirectionError> {
    // Content hash of the two source files that implement the
    // study, so the artifact records the code that actually ran.
    let root = env!("CARGO_MANIFEST_DIR");
    let mut bytes = std::fs::read(format!("{root}/src/direction.rs"))
        .map_err(|e| DirectionError::Storage(e.to_string()))?;
    let mut manifest = std::fs::read(format!("{root}/src/direction_manifest.rs"))
        .map_err(|e| DirectionError::Storage(e.to_string()))?;
    bytes.append(&mut manifest);
    Ok(sha256_hex(&bytes))
}

/// The B3 selection the freeze must use. Read from the committed
/// B3 report so a wrong configuration cannot freeze silently.
fn selected_config() -> Result<(Family, Rule), DirectionError> {
    let root = env!("CARGO_MANIFEST_DIR");
    let text = std::fs::read_to_string(format!("{root}/docs/direction_b3_report.json"))
        .map_err(|e| DirectionError::Storage(e.to_string()))?;
    let report: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| DirectionError::Storage(e.to_string()))?;
    let family = report
        .get("selection")
        .and_then(|s| s.get("selected_family"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            DirectionError::InvalidArtifact("B3 report has no selected_family".to_string())
        })?;
    let rule = report
        .get("selection")
        .and_then(|s| s.get("selected_rule"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            DirectionError::InvalidArtifact("B3 report has no selected_rule".to_string())
        })?;
    let family = match family {
        "R" => Family::R,
        "F" => Family::F,
        "V" => Family::V,
        "I" => Family::I,
        _ => {
            return Err(DirectionError::InvalidArtifact(format!(
                "B3 report selects unknown family {family}"
            )));
        }
    };
    let rule = match rule {
        "p>0.50" => Rule::P50,
        "p>prevalence" => Rule::Prevalence,
        _ => {
            return Err(DirectionError::InvalidArtifact(format!(
                "B3 report selects unknown rule {rule}"
            )));
        }
    };
    Ok((family, rule))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 27 {
        eprintln!(
            "usage: direction_b4 <out.json> <session_dir> [<session_dir> ...] (need 26 sessions)"
        );
        return ExitCode::from(2);
    }
    let out = args[0].clone();
    let dirs = &args[1..];
    if let Err(e) = run(&out, dirs) {
        eprintln!("direction_b4 failed: {e}");
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
    // The supplied campaign sessions must be exactly slots 01-20
    // of the counted campaign, in chronological order.
    let mut campaign: Vec<String> = ids
        .iter()
        .filter(|id| is_campaign_id(id))
        .cloned()
        .collect();
    campaign.sort_by_key(|id| parse_session_start_unix(id).unwrap());
    let expected: Vec<String> = CAMPAIGN_SLOT_IDS.iter().map(|s| s.to_string()).collect();
    if campaign != expected {
        return Err(DirectionError::UnexpectedSession(format!(
            "campaign sessions must be exactly slots 01-20; got {campaign:?}"
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
    if ids.len() != 26 {
        return Err(DirectionError::UnexpectedSession(format!(
            "expected 26 sessions, got {}",
            ids.len()
        )));
    }
    // The manifest is the only resolution path. Content hashes
    // attach to each campaign slot identity as the protocol requires.
    let campaign_refs: Vec<&str> = campaign.iter().map(|s| s.as_str()).collect();
    let mut manifest = DirectionManifest::new(&campaign_refs)?;
    for (id, dir) in ids.iter().zip(dirs.iter()) {
        if is_campaign_id(id) {
            manifest.attach_hash(id, &content_hash(dir)?)?;
        }
    }
    let dir_by_id: std::collections::HashMap<&str, &str> = ids
        .iter()
        .zip(dirs.iter())
        .map(|(id, dir)| (id.as_str(), dir.as_str()))
        .collect();

    // Load the fit pool through the fit allowlist.
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
    let mut slot_frames = Vec::new();
    for (slot, id) in expected.iter().enumerate().take(10) {
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
    // Load the sealed holdout only through the confirmation path,
    // presenting the registered protocol digest.
    let mut holdout_frames = Vec::new();
    for (slot, id) in expected.iter().enumerate().skip(10) {
        let frame =
            load_session_for_confirmation(&manifest, id, dir_by_id[id.as_str()], PROTOCOL_DIGEST)?;
        println!(
            "slot {:02} {id}: {} bars, {} observed seconds, {} eligible origins",
            slot + 1,
            frame.n_bars,
            frame.n_observed_secs(),
            toxicflow::direction::eligible_origins(&frame).len()
        );
        holdout_frames.push(frame);
    }

    // Section 7 freeze: refit the B3-selected family and rule on
    // the archives plus slots 01-10 under the frozen embargo
    // (660 seconds before slot 11's scheduled start).
    let (family, rule) = selected_config()?;
    println!(
        "\nselected: {} under {} (from docs/direction_b3_report.json)",
        family.name(),
        rule.name()
    );
    let slot11_start_ms = parse_session_start_unix(&expected[10]).ok_or_else(|| {
        DirectionError::MissingSession("cannot parse scheduled start of slot 11".to_string())
    })? * 1000;
    let cutoff_ms = slot11_start_ms - EMBARGO_SECS * 1000;
    let mut train: Vec<(FrameCalc, Vec<toxicflow::direction::OriginData>)> = Vec::new();
    let mut fit_ids: Vec<String> = Vec::new();
    let mut fit_hashes: Vec<(String, String)> = Vec::new();
    for frame in archive_frames.iter().chain(slot_frames.iter()) {
        let calc = FrameCalc::new(frame);
        let origins: Vec<toxicflow::direction::OriginData> = calc
            .origins()
            .into_iter()
            .filter(|o| o.decision_ms <= cutoff_ms)
            .collect();
        fit_ids.push(frame.session_id.clone());
        fit_hashes.push((
            frame.session_id.clone(),
            content_hash(dir_by_id[frame.session_id.as_str()])?,
        ));
        train.push((calc, origins));
    }
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| DirectionError::Storage(e.to_string()))?
        .as_secs() as i64;
    let artifact = freeze_artifact(
        &train,
        &fit_ids,
        &fit_hashes,
        cutoff_ms,
        family,
        rule,
        PROTOCOL_DIGEST,
        &code_hash()?,
        timestamp,
    )?;
    println!(
        "frozen: {} under {}, {} fit origins, embargo cutoff {cutoff_ms}",
        artifact.family.name(),
        artifact.rule.name(),
        artifact.n_fit_origins
    );

    // Section 8 confirmation: score slots 11-20 exactly once with
    // the frozen artifact. Nothing else is scored on the holdout.
    let holdout: Vec<HoldoutInput> = holdout_frames
        .iter()
        .enumerate()
        .map(|(k, f)| HoldoutInput {
            slot: CONFIRMATION_FIRST_SLOT + k,
            session_id: expected[10 + k].clone(),
            frame: f,
        })
        .collect();
    let outcome = run_confirmation(&artifact, &holdout)?;
    let (kind, cleared) = classify_confirmation(
        &outcome.pooled,
        outcome.boot_300s,
        outcome.boot_600s,
        outcome.per_session.len(),
    )?;
    assert_eq!(kind, outcome.outcome);
    assert_eq!(cleared, outcome.cleared_endpoint);

    // Deterministic replay: deserialize the artifact and re-score
    // every holdout session. Decisions must match exactly and
    // probabilities to within 1e-12, or the run is not verified.
    let replay = toxicflow::direction::FrozenArtifact::from_json(&artifact.to_json())?;
    for h in holdout.iter() {
        let calc = FrameCalc::new(h.frame);
        let origins = calc.origins();
        let (rows_a, probs_a) = score_with_artifact(&artifact, &calc, &origins);
        let (rows_b, probs_b) = score_with_artifact(&replay, &calc, &origins);
        if rows_a.len() != rows_b.len() {
            return Err(DirectionError::InvalidArtifact(format!(
                "replay row count mismatch on slot {}",
                h.slot
            )));
        }
        for (ra, rb) in rows_a.iter().zip(rows_b.iter()) {
            if ra.time_ms != rb.time_ms
                || ra.target != rb.target
                || ra.model_pred != rb.model_pred
                || ra.baseline_pred != rb.baseline_pred
            {
                return Err(DirectionError::InvalidArtifact(format!(
                    "replay decision mismatch on slot {}",
                    h.slot
                )));
            }
        }
        for (pa, pb) in probs_a.iter().zip(probs_b.iter()) {
            if (pa - pb).abs() > 1e-12 {
                return Err(DirectionError::InvalidArtifact(format!(
                    "replay probability mismatch on slot {}",
                    h.slot
                )));
            }
        }
        println!("replay verified for slot {:02}", h.slot);
    }

    println!("\nconfirmation outcome: {}", outcome.outcome.name());
    println!(
        "pooled: accuracy {:.5} vs baseline {:.5}, Delta {:+.5}, BA {:.5}, Gamma {:+.5}",
        outcome.pooled.accuracy,
        outcome.pooled.baseline_accuracy,
        outcome.pooled.delta,
        outcome.pooled.balanced_accuracy,
        outcome.pooled.gamma
    );
    for (label, boot) in [("300s", outcome.boot_300s), ("600s", outcome.boot_600s)] {
        match boot {
            Some(((dl, du), (gl, gu))) => {
                println!("bootstrap {label}: Delta [{dl:.5}, {du:.5}], Gamma [{gl:.5}, {gu:.5}]")
            }
            None => println!("bootstrap {label}: suppressed"),
        }
    }
    let report = build_report(&artifact, &outcome, kind, cleared);
    std::fs::write(out, serde_json::to_string_pretty(&report).unwrap())
        .map_err(|e| DirectionError::Storage(e.to_string()))?;
    println!("wrote {out}");
    Ok(())
}

fn boot_json(b: &toxicflow::direction::BootstrapIntervals) -> serde_json::Value {
    use serde_json::json;
    match b {
        Some(((dl, du), (gl, gu))) => json!({
            "delta_ci": [dl, du],
            "gamma_ci": [gl, gu],
        }),
        None => json!({"suppressed": true}),
    }
}

fn build_report(
    artifact: &toxicflow::direction::FrozenArtifact,
    outcome: &toxicflow::direction::ConfirmationOutcome,
    kind: ConfirmationOutcomeKind,
    cleared: Option<&'static str>,
) -> serde_json::Value {
    use serde_json::json;
    let pooled = &outcome.pooled;
    let boot = |b: &toxicflow::direction::BootstrapIntervals| match b {
        Some(((dl, du), (gl, gu))) => (*dl, *du, *gl, *gu),
        None => (f64::NAN, f64::NAN, f64::NAN, f64::NAN),
    };
    let (dl300, du300, gl300, gu300) = boot(&outcome.boot_300s);
    let (dl600, du600, gl600, gu600) = boot(&outcome.boot_600s);
    // Registered wording (protocol section 10), with both block
    // lengths stated for each interval.
    let metric_sentence = format!(
        "On ten prespecified held-out BTCUSDT recordings, the 60-second model scored {:.6} accuracy versus {:.6} for training majority, with paired difference {:+.6} (95% block intervals [{:.6}, {:.6}] at 300 s and [{:.6}, {:.6}] at 600 s). Balanced accuracy was {:.6} versus 0.50 (95% block intervals [{:.6}, {:.6}] at 300 s and [{:.6}, {:.6}] at 600 s).",
        pooled.accuracy,
        pooled.baseline_accuracy,
        pooled.delta,
        dl300,
        du300,
        dl600,
        du600,
        pooled.balanced_accuracy,
        gl300,
        gu300,
        gl600,
        gu600
    );
    let wording = match kind {
        ConfirmationOutcomeKind::Win => format!(
            "{metric_sentence} Both preregistered gates cleared at both block lengths."
        ),
        ConfirmationOutcomeKind::Partial => format!(
            "{metric_sentence} Only the {} gate cleared. The study did not establish a joint win.",
            cleared.unwrap_or("unknown")
        ),
        ConfirmationOutcomeKind::Null => format!(
            "{metric_sentence} Neither preregistered gate cleared. The campaign produced no confirmed improvement."
        ),
    };
    json!({
        "schema": "toxicflow.direction-confirmation/v1",
        "protocol_digest": PROTOCOL_DIGEST,
        "protocol": "docs/direction_confirmation_protocol.md",
        "note": "Five adjacent pairs give limited support for generalization (protocol section 8).",
        "selection": {
            "selected_family": artifact.family.name(),
            "selected_rule": artifact.rule.name(),
            "development_report": "docs/direction_b3_report.json",
            "note": "The eight development selection rows with their parent comparisons are in docs/direction_b3_report.json (protocol section 10).",
        },
        "frozen_artifact": artifact.to_json(),
        "outcome": {
            "kind": kind.name(),
            "cleared_endpoint": cleared,
            "wording": wording,
        },
        "pooled": {
            "n_scored": pooled.n_scored,
            "accuracy": pooled.accuracy,
            "baseline_accuracy": pooled.baseline_accuracy,
            "balanced_accuracy": pooled.balanced_accuracy,
            "delta": pooled.delta,
            "gamma": pooled.gamma,
            "s": pooled.s,
        },
        "per_session": outcome.per_session.iter().map(|s| json!({
            "slot": s.slot,
            "session": s.session,
            "n_scored": s.n_scored,
            "n_eligible": s.n_eligible,
            "coverage": s.coverage,
            "accuracy": s.accuracy,
            "baseline_accuracy": s.baseline_accuracy,
            "balanced_accuracy": s.balanced_accuracy,
            "delta": s.delta,
            "gamma": s.gamma,
            "s": s.s,
            "recall_pos": s.recall_pos,
            "recall_neg": s.recall_neg,
            "prevalence": s.prevalence,
            "flat_count": s.flat_count,
            "brier": s.brier,
            "calibration": s.calibration.iter().map(|b| json!({
                "lo": b.lo, "hi": b.hi, "n": b.n, "mean_p": b.mean_p, "mean_y": b.mean_y,
            })).collect::<Vec<_>>(),
            "bootstrap_300s": boot_json(&s.boot_300s),
        })).collect::<Vec<_>>(),
        "bootstrap_300s": boot_json(&outcome.boot_300s),
        "bootstrap_600s": boot_json(&outcome.boot_600s),
        "fixed_cohort_300s": boot_json(&outcome.fixed_cohort_300s),
        "leave_one_pair_out": outcome.leave_one_pair_out.iter().map(|l| json!({
            "dropped_pair": [l.dropped_pair.0, l.dropped_pair.1],
            "delta": l.delta,
            "gamma": l.gamma,
        })).collect::<Vec<_>>(),
        "lopo_delta_range": [outcome.lopo_delta_range.0, outcome.lopo_delta_range.1],
        "lopo_gamma_range": [outcome.lopo_gamma_range.0, outcome.lopo_gamma_range.1],
        "audit_300s": outcome.audit_300s.iter().map(|a| json!({
            "session": a.session, "block_ms": a.block_ms, "bin": a.bin,
            "n": a.n, "n1": a.n1, "tp": a.tp, "tn": a.tn, "fp": a.fp, "fn": a.fn_,
            "base_correct": a.base_correct,
        })).collect::<Vec<_>>(),
        "audit_600s": outcome.audit_600s.iter().map(|a| json!({
            "session": a.session, "block_ms": a.block_ms, "bin": a.bin,
            "n": a.n, "n1": a.n1, "tp": a.tp, "tn": a.tn, "fp": a.fp, "fn": a.fn_,
            "base_correct": a.base_correct,
        })).collect::<Vec<_>>(),
        "replay_verified": true,
    })
}
