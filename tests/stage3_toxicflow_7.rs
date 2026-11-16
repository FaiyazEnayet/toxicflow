//! Stage 3 toxicflow TF7: a reproducible small raw-to-report fixture.
//!
//! The fixture is a slice of a real recorded session journal, stored as
//! text under `experiments/tf7_fixture/`. These tests pin the manifest
//! schema, the fixture checksums, the replay digest, and the numbers a
//! clean checkout regenerates from the fixture, so a future change that
//! moves any of them fails the suite instead of failing silently.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;
use toxicflow::storage::{journal_digest, load_journal_jsonl, reconstruct_session};

/// The experiment manifest, embedded at compile time so the tests work
/// no matter which directory the test binary runs from.
const MANIFEST: &str = include_str!("../experiments/manifest.json");

fn manifest() -> Value {
    serde_json::from_str(MANIFEST).expect("experiments/manifest.json must parse as JSON")
}

fn repo_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9
}

/// Minimal SHA-256, so the checksum check needs no extra dependency.
fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in msg.as_chunks::<64>().0 {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[4 * i],
                chunk[4 * i + 1],
                chunk[4 * i + 2],
                chunk[4 * i + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (i, v) in [a, b, c, d, e, f, g, hh].iter().enumerate() {
            h[i] = h[i].wrapping_add(*v);
        }
    }
    h.iter().map(|v| format!("{v:08x}")).collect()
}

#[test]
fn stage3_toxicflow_7_manifest_schema() {
    let m = manifest();
    assert_eq!(
        m["schema_version"].as_str(),
        Some("toxicflow.raw-to-report/v1"),
        "manifest schema_version must be toxicflow.raw-to-report/v1"
    );
    let fixture = &m["fixture"];
    assert_eq!(
        fixture["path"].as_str(),
        Some("experiments/tf7_fixture/journal.jsonl")
    );
    let sha = fixture["sha256"]
        .as_str()
        .expect("fixture sha256 must be a string");
    assert_eq!(sha.len(), 64, "fixture sha256 must be 64 hex chars");
    assert!(
        sha.chars().all(|c| c.is_ascii_hexdigit()),
        "fixture sha256 must be hex"
    );
    assert_eq!(fixture["bytes"].as_u64(), Some(4_126_120));
    assert_eq!(fixture["n_records"].as_u64(), Some(10_240));
    assert_eq!(
        fixture["source_session"].as_str(),
        Some("session-20261108-160454-utc")
    );
    assert_eq!(
        fixture["source_session_id"].as_str(),
        Some("btcusdt-1794153894158")
    );
    assert_eq!(fixture["symbol"].as_str(), Some("BTCUSDT"));
    let kinds = &fixture["record_kinds"];
    assert_eq!(kinds["session_start"].as_u64(), Some(1));
    assert_eq!(kinds["snapshot"].as_u64(), Some(1));
    assert_eq!(kinds["depth"].as_u64(), Some(7_263));
    assert_eq!(kinds["trade"].as_u64(), Some(2_975));
    let commands = m["commands"]
        .as_array()
        .expect("manifest commands must be an array");
    assert_eq!(
        commands.len(),
        3,
        "manifest must list exactly the three offline commands"
    );
    for (i, c) in commands.iter().enumerate() {
        assert!(
            c.as_str().is_some_and(|s| !s.is_empty()),
            "command {i} must be a string"
        );
    }
    assert!(
        commands[0].as_str().unwrap().contains("journal-to-session"),
        "first command must rebuild the session from the fixture journal"
    );
    assert!(
        commands[1].as_str().unwrap().contains("export"),
        "second command must export the rebuilt session"
    );
    assert!(
        commands[2]
            .as_str()
            .unwrap()
            .contains("scripts/dashboard.py"),
        "third command must build the dashboard from the export"
    );
    let artifacts = m["artifacts"]
        .as_object()
        .expect("manifest artifacts must be an object");
    assert!(
        artifacts.len() >= 9,
        "artifact manifest must map every dashboard figure"
    );
    for (figure, entry) in artifacts {
        assert!(
            entry["file"].as_str().is_some_and(|f| !f.is_empty()),
            "figure {figure} must name its export file"
        );
        assert!(
            entry.get("columns").is_some() || entry.get("fields").is_some(),
            "figure {figure} must list its columns or fields"
        );
    }
    let note = m["large_sessions"]["note"].as_str().unwrap_or("");
    assert!(
        note.contains("SHA256") && note.contains("not in this repo"),
        "manifest must state that large sessions keep separate checksums and are not in the repo: {note}"
    );
}

#[test]
fn stage3_toxicflow_7_fixture_hash_matches_manifest() {
    let m = manifest();
    let bytes = std::fs::read(repo_path("experiments/tf7_fixture/journal.jsonl"))
        .expect("fixture journal must exist");
    assert_eq!(
        bytes.len(),
        4_126_120,
        "fixture size must match the manifest"
    );
    assert_eq!(
        sha256_hex(&bytes),
        m["fixture"]["sha256"].as_str().unwrap(),
        "fixture sha256 must match the manifest"
    );
}

#[test]
fn stage3_toxicflow_7_fixture_replays_to_recorded_digest() {
    let m = manifest();
    let records = load_journal_jsonl(&repo_path("experiments/tf7_fixture/journal.jsonl"))
        .expect("fixture journal must load");
    assert_eq!(
        records.len(),
        10_240,
        "fixture must hold 10,240 journal records"
    );
    assert_eq!(
        journal_digest(&records),
        m["fixture"]["journal_digest"].as_str().unwrap(),
        "replay digest must match the recorded digest"
    );
}

#[test]
fn stage3_toxicflow_7_reconstruction_matches_recorded_session() {
    let m = manifest();
    let records = load_journal_jsonl(&repo_path("experiments/tf7_fixture/journal.jsonl"))
        .expect("fixture journal must load");
    let (trades, quotes) =
        reconstruct_session(&records).expect("fixture journal must reconstruct a session");
    let recon = &m["fixture"]["reconstruction"];
    assert_eq!(
        trades.len(),
        2_975,
        "reconstruction must yield 2,975 trades"
    );
    assert_eq!(
        quotes.len(),
        7_263,
        "reconstruction must yield 7,263 quotes"
    );
    assert_eq!(trades.len() as u64, recon["n_trades"].as_u64().unwrap());
    assert_eq!(quotes.len() as u64, recon["n_quotes"].as_u64().unwrap());
    let volume: f64 = trades.iter().map(|t| t.qty).sum();
    assert!(
        close(volume, recon["volume_btc"].as_f64().unwrap()),
        "volume must match: {volume}"
    );
    let first_trade = &recon["first_trade"];
    let first = &trades[0];
    assert_eq!(first.time_ms, first_trade["time_ms"].as_i64().unwrap());
    assert!(close(first.price, first_trade["price"].as_f64().unwrap()));
    assert!(close(first.qty, first_trade["qty"].as_f64().unwrap()));
    assert_eq!(
        first.is_buyer_maker,
        first_trade["is_buyer_maker"].as_bool().unwrap()
    );
    let last_trade = &recon["last_trade"];
    let last = &trades[trades.len() - 1];
    assert_eq!(last.time_ms, last_trade["time_ms"].as_i64().unwrap());
    assert!(close(last.price, last_trade["price"].as_f64().unwrap()));
    assert!(close(last.qty, last_trade["qty"].as_f64().unwrap()));
    assert_eq!(
        last.is_buyer_maker,
        last_trade["is_buyer_maker"].as_bool().unwrap()
    );
    let first_quote = &recon["first_quote"];
    let first_q = &quotes[0];
    assert_eq!(first_q.time_ms, first_quote["time_ms"].as_i64().unwrap());
    assert!(close(
        first_q.bid_px,
        first_quote["bid_px"].as_f64().unwrap()
    ));
    assert!(close(
        first_q.bid_sz,
        first_quote["bid_sz"].as_f64().unwrap()
    ));
    assert!(close(
        first_q.ask_px,
        first_quote["ask_px"].as_f64().unwrap()
    ));
    assert!(close(
        first_q.ask_sz,
        first_quote["ask_sz"].as_f64().unwrap()
    ));
    assert!(close(first_q.mid(), recon["first_mid"].as_f64().unwrap()));
    let last_quote = &recon["last_quote"];
    let last_q = &quotes[quotes.len() - 1];
    assert_eq!(last_q.time_ms, last_quote["time_ms"].as_i64().unwrap());
    assert!(close(last_q.bid_px, last_quote["bid_px"].as_f64().unwrap()));
    assert!(close(last_q.bid_sz, last_quote["bid_sz"].as_f64().unwrap()));
    assert!(close(last_q.ask_px, last_quote["ask_px"].as_f64().unwrap()));
    assert!(close(last_q.ask_sz, last_quote["ask_sz"].as_f64().unwrap()));
    assert!(close(last_q.mid(), recon["last_mid"].as_f64().unwrap()));
}

#[test]
fn stage3_toxicflow_7_tampered_fixture_rejected() {
    let dir = std::env::temp_dir().join(format!("toxicflow_tf7_tamper_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let src = repo_path("experiments/tf7_fixture/journal.jsonl");
    let tampered = dir.join("journal.jsonl");
    let mut text = std::fs::read_to_string(&src).expect("fixture must read");
    // Change one price digit in the first trade payload; the stored
    // checksum must no longer match. The payload is a JSON string
    // inside the JSONL row, so its quotes appear escaped in the file.
    let from = "\\\"price\\\":76093.75";
    assert!(
        text.contains(from),
        "fixture must contain the first trade payload"
    );
    text = text.replacen(from, "\\\"price\\\":76093.76", 1);
    std::fs::write(&tampered, text).unwrap();
    let err = load_journal_jsonl(&tampered).expect_err("tampered fixture must be rejected");
    assert!(
        err.to_string().contains("checksum"),
        "tamper must fail on the checksum, got: {err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn stage3_toxicflow_7_clean_checkout_regenerates_report() {
    let m = manifest();
    let dir = std::env::temp_dir().join(format!("toxicflow_tf7_chain_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let bin = env!("CARGO_BIN_EXE_toxicflow");
    let session = dir.join("session");
    let out = dir.join("dashboard_data");
    let status = Command::new(bin)
        .args([
            "journal-to-session",
            repo_path("experiments/tf7_fixture/journal.jsonl")
                .to_str()
                .unwrap(),
            session.to_str().unwrap(),
        ])
        .status()
        .expect("journal-to-session must run");
    assert!(
        status.success(),
        "journal-to-session must exit successfully"
    );
    assert!(
        session.join("trades.parquet").exists(),
        "session must hold trades.parquet"
    );
    assert!(
        session.join("quotes.parquet").exists(),
        "session must hold quotes.parquet"
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
    let artifacts = m["artifacts"].as_object().unwrap();
    for (figure, entry) in artifacts {
        let file = entry["file"].as_str().unwrap();
        let path = out.join(file);
        assert!(path.exists(), "figure {figure} needs export file {file}");
        if file.ends_with(".csv") {
            let header = std::fs::read_to_string(&path)
                .unwrap_or_else(|_| panic!("figure {figure} file {file} must read"))
                .lines()
                .next()
                .unwrap_or_default()
                .to_string();
            let cols: Vec<&str> = header.split(',').collect();
            for col in entry["columns"].as_array().unwrap() {
                let col = col.as_str().unwrap();
                assert!(
                    cols.contains(&col),
                    "figure {figure} file {file} must have column {col}"
                );
            }
        }
    }
    let summary: Value = serde_json::from_str(
        &std::fs::read_to_string(out.join("summary.json")).expect("summary.json must exist"),
    )
    .expect("summary.json must parse");
    let expected = &m["expected"]["export_summary"];
    assert_eq!(summary["n_trades"].as_u64(), expected["n_trades"].as_u64());
    assert_eq!(summary["n_quotes"].as_u64(), expected["n_quotes"].as_u64());
    assert!(close(
        summary["span_s"].as_f64().unwrap(),
        expected["span_s"].as_f64().unwrap()
    ));
    assert!(close(
        summary["volume_btc"].as_f64().unwrap(),
        expected["volume_btc"].as_f64().unwrap()
    ));
    assert!(close(
        summary["price_lo"].as_f64().unwrap(),
        expected["price_lo"].as_f64().unwrap()
    ));
    assert!(close(
        summary["price_hi"].as_f64().unwrap(),
        expected["price_hi"].as_f64().unwrap()
    ));
    assert!(close(
        summary["buy_share"].as_f64().unwrap(),
        expected["buy_share"].as_f64().unwrap()
    ));
    assert_eq!(
        summary["gaps"]["n_bars"].as_u64(),
        expected["gaps_n_bars"].as_u64()
    );
    assert_eq!(
        summary["gaps"]["n_dataset_rows"].as_u64(),
        expected["gaps_n_dataset_rows"].as_u64()
    );
    assert!(close(
        summary["vpin"]["mean"].as_f64().unwrap(),
        expected["vpin_mean"].as_f64().unwrap()
    ));
    assert!(close(
        summary["vpin"]["max"].as_f64().unwrap(),
        expected["vpin_max"].as_f64().unwrap()
    ));
    assert!(close(
        summary["kyle_lambda"]["mean"].as_f64().unwrap(),
        expected["kyle_lambda_mean"].as_f64().unwrap()
    ));
    assert!(close(
        summary["ofi"]["mean"].as_f64().unwrap(),
        expected["ofi_mean"].as_f64().unwrap()
    ));
    assert!(close(
        summary["ofi"]["std"].as_f64().unwrap(),
        expected["ofi_std"].as_f64().unwrap()
    ));
    assert!(close(
        summary["predict"]["accuracy"].as_f64().unwrap(),
        expected["predict_accuracy"].as_f64().unwrap()
    ));
    assert!(close(
        summary["predict"]["baseline"].as_f64().unwrap(),
        expected["predict_baseline"].as_f64().unwrap()
    ));
    assert!(close(
        summary["mm"]["pnl_open"].as_f64().unwrap(),
        expected["mm_pnl_open"].as_f64().unwrap()
    ));
    assert!(close(
        summary["mm"]["pnl_gated"].as_f64().unwrap(),
        expected["mm_pnl_gated"].as_f64().unwrap()
    ));
    assert_eq!(
        summary["speed"]["fit_ok"].as_bool(),
        expected["speed_fit_ok"].as_bool()
    );
    assert!(close(
        summary["speed"]["fit_r_squared"].as_f64().unwrap(),
        expected["speed_fit_r_squared"].as_f64().unwrap()
    ));
    let horizons = summary["speed"]["horizons_s"].as_array().unwrap();
    let response = summary["speed"]["response_bps"].as_array().unwrap();
    let at_60 = horizons
        .iter()
        .position(|h| h.as_f64() == Some(60.0))
        .expect("speed horizons must include 60 s");
    assert!(close(
        response[at_60].as_f64().unwrap(),
        expected["speed_response_bps_60s"].as_f64().unwrap()
    ));
    let _ = std::fs::remove_dir_all(&dir);
}
