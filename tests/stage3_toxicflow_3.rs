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
use toxicflow::metrics::{Quote, Trade};
use toxicflow::predict::{
    build_dataset, build_p3_dataset, p3_grid_from_json, preregistered_p3_grid, walk_forward_p3,
    P3Bar, P3Control, P3Outcome, P3_B3_COLS, P3_B5_COLS, P3_BUCKET_VOL_BTC, P3_CONTROLS_VPIN_COLS,
    P3_N_FEATURES, P3_VPIN_COLS, P3_VPIN_WINDOW,
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

// ---- P3: conditional toxicity pilot (fixtures written first) ----

/// Hand-built session with exactly known VPIN alignment.
///
/// 400 s of quotes at 1 s spacing, mid 100.00 until bar 32, then
/// 101.00. 660 trades at 2/s, 7.5 BTC each, so one bucket is 15 BTC
/// (two trades). Buckets 270..279 are balanced (one buy plus one
/// sell, imbalance 0.0); every other bucket is one-sided (imbalance
/// 1.0). Bucket k completes at 1000k + 500 ms, so bar i (closing at
/// (10i + 10) s) has completed buckets 0..=10i + 9, and the VPIN at
/// the bar is the mean imbalance of the last 20 completed buckets:
/// bar 29 vpin 1.0, lag3 (bar 26) 1.0; bar 30 vpin 1.0, lag3 (bar 27)
/// 0.5; bar 31 vpin 1.0, lag3 (bar 28) 0.5; bar 32 vpin 1.0, lag3
/// (bar 29) 1.0. Trades in bars 30..32 are all sells while the mid
/// rises, so those rows carry adverse-markout label 1.
fn p3_hand_session() -> (Vec<Quote>, Vec<Trade>) {
    let mut quotes = Vec::new();
    for s in 0..400 {
        let mid = if s < 320 { 100.0 } else { 101.0 };
        quotes.push(Quote {
            time_ms: s * 1000,
            bid_px: mid - 0.05,
            bid_sz: 5.0,
            ask_px: mid + 0.05,
            ask_sz: 5.0,
        });
    }
    let mut trades = Vec::new();
    for j in 0..660 {
        trades.push(Trade {
            time_ms: j as i64 * 500,
            price: 100.0,
            qty: 7.5,
            // Buckets 270..279 (trades 540..559) are balanced;
            // trades in bars 30..32 (j >= 600) are all sells.
            is_buyer_maker: ((540..560).contains(&j) && j % 2 == 1) || j >= 600,
        });
    }
    (quotes, trades)
}

#[test]
fn stage3_toxicflow_3_p3_vpin_alignment_known_answer() {
    assert_eq!(P3_BUCKET_VOL_BTC, 15.0);
    assert_eq!(P3_VPIN_WINDOW, 20);
    let (quotes, trades) = p3_hand_session();
    let ds = build_p3_dataset(&quotes, &trades, 10_000, 3);
    assert_eq!(ds.len(), 8, "hand session must yield bars 29..=36");
    let row = |i: usize| &ds[i];
    // (bar, vpin, vpin_lag3, direction, spread_wide, adverse)
    let expected: [(i64, f64, f64, u8, u8, Option<u8>); 8] = [
        (290_000, 1.0, 1.0, 1, 0, Some(0)),
        (300_000, 1.0, 0.5, 1, 0, Some(1)),
        (310_000, 1.0, 0.5, 1, 0, Some(1)),
        (320_000, 1.0, 1.0, 0, 0, Some(0)),
        (330_000, 1.0, 1.0, 0, 0, None),
        (340_000, 1.0, 1.0, 0, 0, None),
        (350_000, 1.0, 1.0, 0, 0, None),
        (360_000, 1.0, 1.0, 0, 0, None),
    ];
    for (k, (t, vpin, lag3, dir, wide, adv)) in expected.iter().enumerate() {
        let r = row(k);
        assert_eq!(r.time_ms, *t, "row {k} time");
        assert_eq!(r.features[15], *vpin, "row {k} vpin");
        assert_eq!(r.features[16], *lag3, "row {k} vpin_lag3");
        assert_eq!(r.target_direction, *dir, "row {k} direction");
        assert_eq!(r.target_spread_wide, *wide, "row {k} spread_wide");
        assert_eq!(r.target_adverse, *adv, "row {k} adverse_markout");
    }
    assert!((row(0).fwd_abs - 0.01).abs() < 1e-12);
    assert_eq!(row(3).fwd_abs, 0.0);
}

#[test]
fn stage3_toxicflow_3_p3_vpin_causal_under_future_extension() {
    // Appending trades after the last bar must not change any
    // already-built row's VPIN: buckets complete in time order and
    // a bar only ever sees buckets completed before its close.
    let (quotes, trades) = p3_hand_session();
    let ds = build_p3_dataset(&quotes, &trades, 10_000, 3);
    let mut extended = trades.clone();
    for j in 0..200 {
        extended.push(Trade {
            time_ms: 500_000 + j as i64 * 500,
            price: 101.0,
            qty: 7.5,
            is_buyer_maker: false,
        });
    }
    let ds2 = build_p3_dataset(&quotes, &extended, 10_000, 3);
    assert_eq!(ds2.len(), ds.len(), "future trades add no rows");
    for (a, b) in ds.iter().zip(ds2.iter()) {
        assert_eq!(a.time_ms, b.time_ms);
        assert_eq!(a.features[15], b.features[15], "vpin must be prefix-stable");
        assert_eq!(
            a.features[16], b.features[16],
            "vpin_lag3 must be prefix-stable"
        );
    }
}

#[test]
fn stage3_toxicflow_3_p3_v3_columns_bit_identical() {
    // The first five P3 columns reproduce build_dataset
    // bit-identically on shared rows, so the v3 baseline is
    // preserved exactly inside the v5 frame.
    let (quotes, trades) = p3_hand_session();
    let v3 = build_dataset(&quotes, &trades, 10_000, 3);
    let p3 = build_p3_dataset(&quotes, &trades, 10_000, 3);
    assert!(!p3.is_empty());
    for row in &p3 {
        let m = v3
            .iter()
            .find(|b| b.time_ms == row.time_ms)
            .expect("every p3 row must exist in the v3 frame");
        assert_eq!(row.features[0], m.ofi);
        assert_eq!(row.features[1], m.ofi_z);
        assert_eq!(row.features[2], m.imbalance);
        assert_eq!(row.features[3], m.spread_z);
        assert_eq!(row.features[4], m.signed_vol);
        assert_eq!(row.target_direction, m.target);
    }
}

/// Synthetic P3 rows where the VPIN block equals the direction
/// label and every control column is exactly zero: the clean model
/// must separate, the controls-only model must sit at the majority
/// baseline, and the permutation control must collapse.
fn p3_synth_bars() -> Vec<P3Bar> {
    (0..400)
        .map(|i| {
            let target = if (i / 50) % 2 == 0 { 1u8 } else { 0u8 };
            let mut features = [0.0; P3_N_FEATURES];
            features[15] = target as f64;
            features[16] = target as f64;
            P3Bar {
                time_ms: i as i64 * 10_000,
                features,
                target_direction: target,
                fwd_abs: 0.01,
                target_spread_wide: 0,
                target_adverse: Some(0),
                spread_degenerate: false,
                trade_rate: 1.0,
                depth_mean: 5.0,
            }
        })
        .collect()
}

#[test]
fn stage3_toxicflow_3_p3_known_answer_separation_and_collapse() {
    let bars = p3_synth_bars();
    let clean = walk_forward_p3(
        &bars,
        P3_CONTROLS_VPIN_COLS,
        P3Control::Off,
        4,
        3,
        P3Outcome::Direction,
    )
    .expect("clean eval failed");
    let controls = walk_forward_p3(
        &bars,
        P3_B5_COLS,
        P3Control::Off,
        4,
        3,
        P3Outcome::Direction,
    )
    .expect("controls eval failed");
    let perm = walk_forward_p3(
        &bars,
        P3_CONTROLS_VPIN_COLS,
        P3Control::Permute,
        4,
        3,
        P3Outcome::Direction,
    )
    .expect("perm eval failed");
    let shift = walk_forward_p3(
        &bars,
        P3_CONTROLS_VPIN_COLS,
        P3Control::Shift,
        4,
        3,
        P3Outcome::Direction,
    )
    .expect("shift eval failed");
    assert_eq!(clean.n_oos, controls.n_oos);
    assert_eq!(clean.n_oos, perm.n_oos);
    assert_eq!(shift.n_oos, clean.n_oos - 20);
    assert!(
        clean.log_loss < 0.55,
        "clean model must separate: log loss {}",
        clean.log_loss
    );
    assert!(clean.accuracy > 0.85, "clean accuracy {}", clean.accuracy);
    assert!(
        controls.log_loss > 0.65,
        "zero controls must sit near the majority baseline: log loss {}",
        controls.log_loss
    );
    assert!(
        perm.accuracy < clean.accuracy - 0.15,
        "permuted VPIN must collapse accuracy: {} vs {}",
        perm.accuracy,
        clean.accuracy
    );
    assert!(
        perm.log_loss > clean.log_loss + 0.15,
        "permuted VPIN must collapse log loss: {} vs {}",
        perm.log_loss,
        clean.log_loss
    );
    // Every outcome runs on the same rows.
    for outcome in [
        P3Outcome::AbsRet,
        P3Outcome::SpreadWide,
        P3Outcome::AdverseMarkout,
    ] {
        let ev = walk_forward_p3(&bars, P3_B5_COLS, P3Control::Off, 4, 3, outcome)
            .unwrap_or_else(|| panic!("{outcome:?} eval failed"));
        assert_eq!(
            ev.n_oos, clean.n_oos,
            "{outcome:?} must score the same rows"
        );
        assert!(
            ev.log_loss.is_finite(),
            "{outcome:?} log loss must be finite"
        );
    }
}

/// Absolute-return labels use the training-fold median only:
/// perturbing rows that no fold ever trains on must leave every
/// fitted fold model bit-identical.
fn p3_synth_bars_absret() -> Vec<P3Bar> {
    (0..400)
        .map(|i| {
            let fwd_abs = if i < 320 {
                if i % 2 == 0 {
                    0.001
                } else {
                    0.003
                }
            } else {
                0.0025
            };
            let mut features = [0.0; P3_N_FEATURES];
            features[0] = (i % 7) as f64;
            P3Bar {
                time_ms: i as i64 * 10_000,
                features,
                target_direction: 0,
                fwd_abs,
                target_spread_wide: 0,
                target_adverse: Some(0),
                spread_degenerate: false,
                trade_rate: 1.0,
                depth_mean: 5.0,
            }
        })
        .collect()
}

#[test]
fn stage3_toxicflow_3_p3_absret_threshold_is_train_only() {
    let bars = p3_synth_bars_absret();
    let ev1 = walk_forward_p3(&bars, P3_B5_COLS, P3Control::Off, 4, 3, P3Outcome::AbsRet)
        .expect("absret eval failed");
    let mut perturbed = bars.clone();
    for row in perturbed.iter_mut().skip(320) {
        row.fwd_abs = 0.0005;
    }
    let ev2 = walk_forward_p3(
        &perturbed,
        P3_B5_COLS,
        P3Control::Off,
        4,
        3,
        P3Outcome::AbsRet,
    )
    .expect("perturbed absret eval failed");
    assert_eq!(
        ev1.fold_models, ev2.fold_models,
        "perturbing never-trained rows must not change any fitted model"
    );
    assert!(
        ev1.n_pos_oos != ev2.n_pos_oos,
        "the perturbation must actually move test labels ({} vs {})",
        ev1.n_pos_oos,
        ev2.n_pos_oos
    );
    for (a, b) in ev1.per_fold.iter().zip(ev2.per_fold.iter()) {
        assert_eq!(
            a.absret_median, b.absret_median,
            "fold {} absret median must be train-only",
            a.fold
        );
    }
}

#[test]
fn stage3_toxicflow_3_p3_regime_thresholds_are_train_only() {
    let mut bars = p3_synth_bars();
    for (i, row) in bars.iter_mut().enumerate() {
        row.trade_rate = (i % 9) as f64;
        row.depth_mean = (i % 5) as f64;
    }
    let ev1 = walk_forward_p3(
        &bars,
        P3_B5_COLS,
        P3Control::Off,
        4,
        3,
        P3Outcome::Direction,
    )
    .expect("regime eval failed");
    let mut perturbed = bars.clone();
    for row in perturbed.iter_mut().skip(320) {
        row.trade_rate = 1000.0;
        row.depth_mean = 1000.0;
    }
    let ev2 = walk_forward_p3(
        &perturbed,
        P3_B5_COLS,
        P3Control::Off,
        4,
        3,
        P3Outcome::Direction,
    )
    .expect("perturbed regime eval failed");
    for (a, b) in ev1.per_fold.iter().zip(ev2.per_fold.iter()) {
        assert_eq!(a.activity_cut1, b.activity_cut1);
        assert_eq!(a.activity_cut2, b.activity_cut2);
        assert_eq!(a.depth_cut1, b.depth_cut1);
        assert_eq!(a.depth_cut2, b.depth_cut2);
    }
}

#[test]
fn stage3_toxicflow_3_p3_grid_file_matches_library() {
    // The committed grid file must match the preregistered library
    // grid exactly; the runner refuses to run otherwise.
    let text = std::fs::read_to_string("experiments/p3/grid.json").expect("grid.json missing");
    let file_grid = p3_grid_from_json(&text).expect("grid.json must parse");
    assert_eq!(file_grid, preregistered_p3_grid());
}

#[test]
fn stage3_toxicflow_3_p3_vpin_only_and_b3_vpin_cells_run() {
    let bars = p3_synth_bars();
    let vpin_only = walk_forward_p3(
        &bars,
        P3_VPIN_COLS,
        P3Control::Off,
        4,
        3,
        P3Outcome::Direction,
    )
    .expect("vpin-only eval failed");
    let b3 = walk_forward_p3(
        &bars,
        P3_B3_COLS,
        P3Control::Off,
        4,
        3,
        P3Outcome::Direction,
    )
    .expect("b3 eval failed");
    assert_eq!(vpin_only.n_oos, b3.n_oos);
    assert!(
        vpin_only.log_loss < b3.log_loss,
        "VPIN-only must beat the zero-control frame on separating data"
    );
}
