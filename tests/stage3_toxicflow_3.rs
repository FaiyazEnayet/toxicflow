//! Stage 3 toxicflow E3: reproducible archived experiments.
//!
//! The TF7 fixture is the one experiment reproducible from the repo
//! alone. These tests pin its evidence manifest (schema
//! `toxicflow.evidence/v1`), prove a clean checkout regenerates and
//! verifies it, prove tampering is rejected, and prove every older
//! published result is marked archived-not-reproducible with its
//! reason recorded and its retained reports pinned by SHA256.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;
use toxicflow::evidence::{
    build_evidence_manifest, verify_manifest, EvidenceBuild, EVIDENCE_SCHEMA,
};
use toxicflow::storage::journal_digest;

/// The evidence manifest for the reproducible fixture experiment.
const EVIDENCE: &str = include_str!("../experiments/evidence/tf7-fixture.json");
/// The archive registry for the older published results.
const ARCHIVE: &str = include_str!("../experiments/evidence/archive.json");
/// The experiment index mapping claims to experiment IDs.
const INDEX: &str = include_str!("../experiments/evidence/index.json");
/// The TF7 raw-to-report manifest, the source of the fixture hashes.
const TF7_MANIFEST: &str = include_str!("../experiments/manifest.json");
/// The README, whose claims must all trace to an experiment ID.
const README: &str = include_str!("../README.md");

fn repo_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn evidence() -> Value {
    serde_json::from_str(EVIDENCE).expect("evidence manifest must parse as JSON")
}

fn archive() -> Value {
    serde_json::from_str(ARCHIVE).expect("archive registry must parse as JSON")
}

fn index() -> Value {
    serde_json::from_str(INDEX).expect("experiment index must parse as JSON")
}

fn tf7_manifest() -> Value {
    serde_json::from_str(TF7_MANIFEST).expect("TF7 manifest must parse as JSON")
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

fn sha256_hex(data: &[u8]) -> String {
    // Same minimal SHA-256 as the TF7 test suite: no extra dependency.
    toxicflow::evidence::sha256_hex(data)
}

fn is_hex(s: &str, n: usize) -> bool {
    s.len() == n && s.chars().all(|c| c.is_ascii_hexdigit())
}

/// Rebuild the fixture session and export in a fresh temp dir, then
/// build a fresh evidence manifest for them. Returns
/// (session_dir, export_dir, fresh_manifest).
fn regenerate() -> (PathBuf, PathBuf, Value) {
    // Each call gets its own directory: several tests regenerate in
    // parallel inside one test process, so the process id alone is
    // not unique.
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("toxicflow_e3_chain_{}_{}", std::process::id(), n));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let bin = env!("CARGO_BIN_EXE_toxicflow");
    let session = dir.join("session");
    let out = dir.join("dashboard_data");
    // Pass the journal as a repo-relative path from the repo root, the
    // same spelling the TF7 manifest commands use: journal-to-session
    // stores the path string in the rebuilt session meta.json, so the
    // spelling has to match the generation run byte for byte.
    let status = Command::new(bin)
        .args([
            "journal-to-session",
            "experiments/tf7_fixture/journal.jsonl",
            session.to_str().unwrap(),
        ])
        .current_dir(repo_path("."))
        .status()
        .expect("journal-to-session must run");
    assert!(
        status.success(),
        "journal-to-session must exit successfully"
    );
    let status = Command::new(bin)
        .args([
            "export",
            session.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ])
        .status()
        .expect("export must run");
    assert!(status.success(), "export must exit successfully");
    let git_sha = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(repo_path("."))
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    let fresh = build_evidence_manifest(&EvidenceBuild {
        experiment_id: "tf7-fixture".to_string(),
        title: evidence()["title"].as_str().unwrap().to_string(),
        journal: repo_path("experiments/tf7_fixture/journal.jsonl"),
        session_dir: session.clone(),
        export_dir: out.clone(),
        repo_root: repo_path("."),
        source_session: "session-20261108-160454-utc".to_string(),
        source_session_id: "btcusdt-1794153894158".to_string(),
        git_sha,
        rustc_version: String::new(),
    })
    .expect("evidence manifest must build from the regenerated session");
    (session, out, fresh)
}

#[test]
fn stage3_toxicflow_3_evidence_schema() {
    let m = evidence();
    assert_eq!(
        m["schema_version"].as_str(),
        Some(EVIDENCE_SCHEMA),
        "evidence schema_version must be toxicflow.evidence/v1"
    );
    assert_eq!(m["kind"].as_str(), Some("evidence-manifest"));
    assert_eq!(m["experiment_id"].as_str(), Some("tf7-fixture"));
    assert_eq!(m["status"].as_str(), Some("reproducible"));
    assert!(m["title"].as_str().is_some_and(|t| !t.is_empty()));

    let capture = &m["capture"];
    assert_eq!(capture["symbol"].as_str(), Some("BTCUSDT"));
    assert_eq!(
        capture["source_session"].as_str(),
        Some("session-20261108-160454-utc")
    );
    assert_eq!(
        capture["source_session_id"].as_str(),
        Some("btcusdt-1794153894158")
    );
    assert!(
        capture["ws_endpoint"]
            .as_str()
            .is_some_and(|s| s.contains("data-stream.binance.vision")),
        "capture must name the market-data websocket host"
    );
    assert!(
        capture["rest_endpoint"]
            .as_str()
            .is_some_and(|s| s.contains("data-api.binance.vision")),
        "capture must name the market-data REST host"
    );
    let utc_start = capture["utc_start"].as_str().unwrap_or("");
    let utc_end = capture["utc_end"].as_str().unwrap_or("");
    assert!(
        utc_start.contains('T') && utc_start.ends_with('Z'),
        "utc_start must be an ISO UTC timestamp, got {utc_start}"
    );
    assert!(
        utc_end.contains('T') && utc_end.ends_with('Z'),
        "utc_end must be an ISO UTC timestamp, got {utc_end}"
    );
    assert!(utc_start < utc_end, "capture interval must be non-empty");
    let edges = capture["chunk_edges"]
        .as_array()
        .expect("capture must list chunk edges");
    assert!(
        edges.len() >= 4,
        "capture must list first/last trade and quote edges"
    );
    for edge in edges {
        assert!(edge["label"].as_str().is_some_and(|l| !l.is_empty()));
        assert!(edge["time_ms"].as_i64().is_some());
        assert!(edge["utc"].as_str().is_some_and(|u| u.ends_with('Z')));
    }
    assert_eq!(capture["n_records"].as_u64(), Some(10_240));
    assert_eq!(capture["record_kinds"]["trade"].as_u64(), Some(2_975));
    assert_eq!(capture["record_kinds"]["depth"].as_u64(), Some(7_263));

    let software = &m["software"];
    assert!(
        is_hex(software["git_sha"].as_str().unwrap_or(""), 40),
        "software git_sha must be 40 hex chars"
    );
    assert!(
        is_hex(software["cargo_lock_sha256"].as_str().unwrap_or(""), 64),
        "software cargo_lock_sha256 must be 64 hex chars"
    );

    let inputs = m["inputs"].as_array().expect("inputs must be an array");
    assert!(!inputs.is_empty(), "inputs must not be empty");
    for input in inputs {
        assert!(is_hex(input["sha256"].as_str().unwrap_or(""), 64));
        assert!(input["bytes"].as_u64().is_some_and(|b| b > 0));
        assert!(input["path"].as_str().is_some_and(|p| !p.is_empty()));
    }
    let journal = inputs
        .iter()
        .find(|i| i["path"].as_str() == Some("experiments/tf7_fixture/journal.jsonl"))
        .expect("inputs must include the fixture journal");
    assert_eq!(journal["bytes"].as_u64(), Some(4_126_120));

    let session = &m["session"];
    for name in ["trades.parquet", "quotes.parquet", "meta.json"] {
        assert!(
            session["files"]
                .as_array()
                .unwrap()
                .iter()
                .any(|f| f["name"].as_str() == Some(name)
                    && is_hex(f["sha256"].as_str().unwrap_or(""), 64)),
            "session must hash {name}"
        );
    }
    let outputs = m["outputs"]["files"]
        .as_array()
        .expect("outputs must be an array");
    for name in [
        "vpin.csv",
        "kyle.csv",
        "ofi.csv",
        "speed.csv",
        "calibration.csv",
        "summary.json",
    ] {
        assert!(
            outputs.iter().any(|f| f["name"].as_str() == Some(name)
                && is_hex(f["sha256"].as_str().unwrap_or(""), 64)),
            "outputs must hash {name}"
        );
    }
    let gap = &m["resync_gap"];
    assert_eq!(gap["snapshots"].as_u64(), Some(1));
    assert!(gap["resyncs"].as_u64().is_some());
    assert_eq!(gap["n_bars"].as_u64(), Some(74));
    assert_eq!(gap["n_dataset_rows"].as_u64(), Some(67));
    assert!(gap["n_unobserved_bars"].as_u64().is_some());
    assert_eq!(gap["bar_ms"].as_u64(), Some(10_000));

    let params = &m["parameters"];
    assert_eq!(params["vpin_buckets"].as_u64(), Some(50));
    assert_eq!(params["vpin_window"].as_u64(), Some(20));
    assert_eq!(params["ofi_bar_ms"].as_u64(), Some(10_000));
    assert_eq!(params["kyle_window_bars"].as_u64(), Some(60));
    assert!(m["labels"].as_object().is_some_and(|l| l.len() >= 4));

    let expected = &m["expected"];
    assert_eq!(expected["n_trades"].as_u64(), Some(2_975));
    assert_eq!(expected["n_quotes"].as_u64(), Some(7_263));
    assert!(expected["tolerance"].as_f64().is_some_and(|t| t > 0.0));
    assert!(m["fetch"]["method"].as_str().is_some_and(|s| !s.is_empty()));
}

#[test]
fn stage3_toxicflow_3_fixture_input_hash_matches_tf7_manifest() {
    let m = evidence();
    let tf7 = tf7_manifest();
    let bytes = std::fs::read(repo_path("experiments/tf7_fixture/journal.jsonl"))
        .expect("fixture journal must exist");
    let sha = sha256_hex(&bytes);
    assert_eq!(
        sha,
        tf7["fixture"]["sha256"].as_str().unwrap(),
        "fixture sha256 must match the TF7 manifest"
    );
    let journal = m["inputs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|i| i["path"].as_str() == Some("experiments/tf7_fixture/journal.jsonl"))
        .expect("evidence inputs must include the fixture journal");
    assert_eq!(
        journal["sha256"].as_str().unwrap(),
        sha,
        "evidence input hash must match the fixture file"
    );
    let records =
        toxicflow::storage::load_journal_jsonl(&repo_path("experiments/tf7_fixture/journal.jsonl"))
            .expect("fixture journal must load");
    assert_eq!(
        journal_digest(&records),
        tf7["fixture"]["journal_digest"].as_str().unwrap(),
        "replay digest must match the TF7 manifest"
    );
}

#[test]
fn stage3_toxicflow_3_clean_checkout_regenerates_and_verifies() {
    let (session, out, fresh) = regenerate();
    let committed = evidence();
    let tol = committed["expected"]["tolerance"].as_f64().unwrap();
    // The committed manifest must verify against the fresh rebuild:
    // inputs, session files, output hashes, and expected numbers.
    let failures = verify_manifest(&committed, &repo_path("."), &session, &out);
    assert!(
        failures.is_empty(),
        "committed evidence manifest must verify against a fresh rebuild: {}",
        failures.join("; ")
    );
    // The fresh manifest must agree with the committed one on every
    // expected number and every output hash.
    for (key, expected) in committed["expected"].as_object().unwrap() {
        if key == "tolerance" || key == "speed_fit_ok" {
            continue;
        }
        // First/last trade and quote entries are objects, not
        // numbers; they are pinned by the session file hashes above.
        let Some(a) = expected.as_f64() else {
            continue;
        };
        let b = fresh["expected"][key].as_f64().unwrap();
        assert!(
            close(a, b, tol),
            "expected {key} must match: committed {a}, fresh {b}"
        );
    }
    assert_eq!(
        committed["expected"]["speed_fit_ok"].as_bool(),
        fresh["expected"]["speed_fit_ok"].as_bool()
    );
    for file in committed["outputs"]["files"].as_array().unwrap() {
        let name = file["name"].as_str().unwrap();
        let fresh_file = fresh["outputs"]["files"]
            .as_array()
            .unwrap()
            .iter()
            .find(|f| f["name"].as_str() == Some(name))
            .unwrap_or_else(|| panic!("fresh manifest must hash {name}"));
        assert_eq!(
            file["sha256"].as_str().unwrap(),
            fresh_file["sha256"].as_str().unwrap(),
            "output {name} must be byte-identical across rebuilds"
        );
    }
    let _ = std::fs::remove_dir_all(session.parent().unwrap());
}

#[test]
fn stage3_toxicflow_3_tampered_output_rejected() {
    let (session, out, fresh) = regenerate();
    let tampered_csv = out.join("vpin.csv");
    let mut text = std::fs::read_to_string(&tampered_csv).expect("vpin.csv must read");
    text.push_str("0,9.999999\n");
    std::fs::write(&tampered_csv, text).unwrap();
    let failures = verify_manifest(&fresh, &repo_path("."), &session, &out);
    assert!(
        !failures.is_empty(),
        "tampered output must fail verification"
    );
    assert!(
        failures.iter().any(|f| f.contains("vpin.csv")),
        "tamper failure must name vpin.csv, got: {}",
        failures.join("; ")
    );
    let _ = std::fs::remove_dir_all(session.parent().unwrap());
}

#[test]
fn stage3_toxicflow_3_tampered_input_hash_rejected() {
    let (session, out, fresh) = regenerate();
    let mut tampered = fresh.clone();
    let inputs = tampered["inputs"].as_array_mut().unwrap();
    let first = &mut inputs[0];
    let sha = first["sha256"].as_str().unwrap().to_string();
    let flipped = format!("0{}", &sha[1..]);
    assert_ne!(flipped, sha);
    first["sha256"] = Value::String(flipped);
    let failures = verify_manifest(&tampered, &repo_path("."), &session, &out);
    assert!(
        !failures.is_empty(),
        "tampered input hash must fail verification"
    );
    assert!(
        failures.iter().any(|f| f.contains("journal.jsonl")),
        "tamper failure must name the journal input, got: {}",
        failures.join("; ")
    );
    let _ = std::fs::remove_dir_all(session.parent().unwrap());
}

#[test]
fn stage3_toxicflow_3_archive_marks_old_results_not_reproducible() {
    let a = archive();
    assert_eq!(
        a["schema_version"].as_str(),
        Some(EVIDENCE_SCHEMA),
        "archive registry must use the evidence schema"
    );
    assert_eq!(a["kind"].as_str(), Some("archive-registry"));
    let entries = a["entries"].as_array().expect("entries must be an array");
    let ids: Vec<&str> = entries
        .iter()
        .map(|e| e["experiment_id"].as_str().unwrap())
        .collect();
    for required in [
        "main-session-325h",
        "info-speed-historical-captures",
        "closing-btc-80h",
        "closing-eth-80h",
        "direction-confirmation",
        "direction-b3-btc",
        "direction-b3-eth",
    ] {
        assert!(
            ids.contains(&required),
            "archive must list {required}, got {ids:?}"
        );
    }
    for entry in entries {
        let id = entry["experiment_id"].as_str().unwrap();
        assert_eq!(
            entry["status"].as_str(),
            Some("archived-not-reproducible"),
            "archive entry {id} must be archived-not-reproducible"
        );
        assert!(
            entry["reason"].as_str().is_some_and(|r| r.len() >= 40),
            "archive entry {id} must record a reason"
        );
        assert!(
            entry["reason"]
                .as_str()
                .unwrap()
                .to_lowercase()
                .contains("not in the repo")
                || entry["reason"]
                    .as_str()
                    .unwrap()
                    .to_lowercase()
                    .contains("not in this repo"),
            "archive entry {id} must say the raw input is not in the repo"
        );
        let reports = entry["reports"]
            .as_array()
            .unwrap_or_else(|| panic!("archive entry {id} must list retained reports"));
        assert!(
            !reports.is_empty(),
            "archive entry {id} must retain reports"
        );
        for report in reports {
            let path = report["path"].as_str().unwrap();
            let bytes = std::fs::read(repo_path(path))
                .unwrap_or_else(|_| panic!("retained report {path} must exist"));
            assert_eq!(
                sha256_hex(&bytes),
                report["sha256"].as_str().unwrap(),
                "retained report {path} sha256 must match the archive"
            );
            assert_eq!(report["bytes"].as_u64(), Some(bytes.len() as u64));
        }
        assert!(
            entry["sessions"].as_array().is_some_and(|s| !s.is_empty()),
            "archive entry {id} must name its sessions"
        );
    }
}

#[test]
fn stage3_toxicflow_3_index_maps_every_claim_to_an_experiment() {
    let idx = index();
    assert_eq!(
        idx["schema_version"].as_str(),
        Some(EVIDENCE_SCHEMA),
        "index must use the evidence schema"
    );
    assert_eq!(idx["kind"].as_str(), Some("experiment-index"));
    let tf7 = tf7_manifest();
    let artifact_names: Vec<&str> = tf7["artifacts"]
        .as_object()
        .unwrap()
        .keys()
        .map(|k| k.as_str())
        .collect();
    let experiments = idx["experiments"]
        .as_array()
        .expect("index experiments must be an array");
    let mut indexed_ids: Vec<&str> = Vec::new();
    for exp in experiments {
        let id = exp["experiment_id"].as_str().unwrap();
        indexed_ids.push(id);
        let status = exp["status"].as_str().unwrap();
        match status {
            "reproducible" => {
                let manifest_path = exp["manifest"].as_str().unwrap();
                let m: Value = serde_json::from_str(
                    &std::fs::read_to_string(repo_path(manifest_path))
                        .unwrap_or_else(|_| panic!("index manifest {manifest_path} must exist")),
                )
                .unwrap();
                assert_eq!(m["experiment_id"].as_str(), Some(id));
                assert_eq!(m["status"].as_str(), Some("reproducible"));
            }
            "archived-not-reproducible" => {
                let a = archive();
                assert!(
                    a["entries"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|e| e["experiment_id"].as_str() == Some(id)),
                    "indexed archived experiment {id} must exist in the archive"
                );
            }
            other => panic!("experiment {id} has unknown status {other}"),
        }
        for claim in exp["claims"].as_array().unwrap() {
            let kind = claim["kind"].as_str().unwrap();
            let target = claim["target"].as_str().unwrap();
            match kind {
                "readme-section" => assert!(
                    README.contains(target),
                    "README must contain section {target} for {id}"
                ),
                "report" | "dashboard" | "animation" => assert!(
                    repo_path(target).exists(),
                    "claim target {target} for {id} must exist"
                ),
                "dashboard-figure" => assert!(
                    artifact_names.contains(&target),
                    "dashboard figure {target} for {id} must be in the TF7 artifact map"
                ),
                other => panic!("experiment {id} has unknown claim kind {other}"),
            }
        }
    }
    // Every evidence manifest and every archive entry must be indexed.
    for entry in std::fs::read_dir(repo_path("experiments/evidence")).unwrap() {
        let name = entry.unwrap().file_name().to_string_lossy().into_owned();
        if name.ends_with(".json") && name != "archive.json" && name != "index.json" {
            let m: Value = serde_json::from_str(
                &std::fs::read_to_string(repo_path(&format!("experiments/evidence/{name}")))
                    .unwrap(),
            )
            .unwrap();
            let id = m["experiment_id"].as_str().unwrap();
            assert!(
                indexed_ids.contains(&id),
                "evidence manifest {name} ({id}) must be in the index"
            );
        }
    }
    for entry in archive()["entries"].as_array().unwrap() {
        let id = entry["experiment_id"].as_str().unwrap();
        assert!(
            indexed_ids.contains(&id),
            "archive entry {id} must be in the index"
        );
    }
    // The TF7 fixture figures must all trace to the reproducible experiment.
    let fixture = experiments
        .iter()
        .find(|e| e["experiment_id"].as_str() == Some("tf7-fixture"))
        .expect("index must list tf7-fixture");
    for figure in &artifact_names {
        assert!(
            fixture["claims"]
                .as_array()
                .unwrap()
                .iter()
                .any(|c| c["kind"].as_str() == Some("dashboard-figure")
                    && c["target"].as_str() == Some(figure)),
            "figure {figure} must map to tf7-fixture"
        );
    }
}

#[test]
fn stage3_toxicflow_3_readme_names_experiments_and_statuses() {
    assert!(
        README.contains("Experiments and reproducibility"),
        "README must carry an Experiments and reproducibility section"
    );
    for (id, status) in [
        ("tf7-fixture", "reproducible"),
        ("main-session-325h", "archived-not-reproducible"),
        ("closing-btc-80h", "archived-not-reproducible"),
        ("closing-eth-80h", "archived-not-reproducible"),
    ] {
        assert!(README.contains(id), "README must name experiment {id}");
        assert!(README.contains(status), "README must state status {status}");
    }
}
