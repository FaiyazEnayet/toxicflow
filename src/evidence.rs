//! Versioned evidence manifests for recorded experiments.
//!
//! A manifest pins everything an independent reader needs to re-run
//! one experiment from the repo alone: the capture interval and chunk
//! edges, the market-data endpoints, the symbol, the software commit
//! and dependency lock hash, the SHA-256 of every input, the resync
//! and gap accounting, the parameter and label definitions, the
//! SHA-256 of every output, and the numbers the run must reproduce
//! within a stated tolerance.
//!
//! Schema `toxicflow.evidence/v1`. Only experiments whose raw input
//! is in the repo get a manifest with status `reproducible`. Older
//! results whose raw input is not in the repo are listed in the
//! archive registry (`experiments/evidence/archive.json`) with status
//! `archived-not-reproducible` and the reason recorded; nothing here
//! claims a recovery that is not shown.

use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::storage::{load_journal_jsonl, JournalRecord};

/// The evidence manifest schema version.
pub const EVIDENCE_SCHEMA: &str = "toxicflow.evidence/v1";

/// The market-data websocket host every capture uses.
pub const WS_HOST: &str = "wss://data-stream.binance.vision";
/// The market-data REST host every capture uses for snapshots.
pub const REST_HOST: &str = "https://data-api.binance.vision";

/// Minimal SHA-256, so manifest hashing needs no extra dependency.
pub fn sha256_hex(data: &[u8]) -> String {
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

/// SHA-256 hex plus byte length of a file.
pub fn sha256_file(path: &Path) -> Result<(String, u64), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let len = bytes.len() as u64;
    Ok((sha256_hex(&bytes), len))
}

/// Format epoch milliseconds as an ISO 8601 UTC timestamp.
pub fn ms_to_utc_iso(ms: i64) -> String {
    let secs = ms.div_euclid(1000);
    let millis = ms.rem_euclid(1000);
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (hour, minute, second) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    // Civil date from days since the Unix epoch (Howard Hinnant's
    // algorithm, no calendar dependency).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{millis:03}Z")
}

/// Inputs for building an evidence manifest.
pub struct EvidenceBuild {
    pub experiment_id: String,
    pub title: String,
    /// Fixture journal the session was rebuilt from (absolute path;
    /// stored repo-relative when it sits under `repo_root`).
    pub journal: PathBuf,
    /// Rebuilt session directory (`trades.parquet`, `quotes.parquet`,
    /// `meta.json`).
    pub session_dir: PathBuf,
    /// Export directory (`vpin.csv`, `kyle.csv`, `ofi.csv`,
    /// `speed.csv`, `calibration.csv`, `summary.json`).
    pub export_dir: PathBuf,
    /// Repo checkout the manifest was generated from.
    pub repo_root: PathBuf,
    pub source_session: String,
    pub source_session_id: String,
    /// Commit the manifest was generated from (recorded provenance;
    /// verification pins the lock hash and the file hashes, not this).
    pub git_sha: String,
    /// `rustc --version` output at generation time; empty when unknown.
    pub rustc_version: String,
}

/// The export files an evidence manifest hashes.
const EXPORT_FILES: [&str; 6] = [
    "vpin.csv",
    "kyle.csv",
    "ofi.csv",
    "speed.csv",
    "calibration.csv",
    "summary.json",
];

/// The rebuilt session files an evidence manifest hashes.
const SESSION_FILES: [&str; 3] = ["trades.parquet", "quotes.parquet", "meta.json"];

fn repo_rel(repo_root: &Path, path: &Path) -> String {
    match path.strip_prefix(repo_root) {
        Ok(rel) if !rel.as_os_str().is_empty() => rel.to_string_lossy().into_owned(),
        _ => path.to_string_lossy().into_owned(),
    }
}

fn file_entry(path: &Path, label: &str, kind: &str) -> Result<Value, String> {
    let (sha, bytes) = sha256_file(path)?;
    Ok(json!({"path": label, "sha256": sha, "bytes": bytes, "kind": kind}))
}

fn named_entry(dir: &Path, name: &str) -> Result<Value, String> {
    let (sha, bytes) = sha256_file(&dir.join(name))?;
    Ok(json!({"name": name, "sha256": sha, "bytes": bytes}))
}

/// Build an evidence manifest for a rebuilt session and its export.
///
/// The journal is read for the capture interval, chunk edges, record
/// kinds, and the symbol; the export `summary.json` supplies the
/// expected numbers and the gap accounting. Every hash is computed
/// from the files on disk at build time.
pub fn build_evidence_manifest(cfg: &EvidenceBuild) -> Result<Value, String> {
    let records = load_journal_jsonl(&cfg.journal)
        .map_err(|e| format!("cannot load journal {}: {e}", cfg.journal.display()))?;
    let symbol = records
        .iter()
        .find_map(|r| match r {
            JournalRecord::SessionStart { symbol } => Some(symbol.clone()),
            _ => None,
        })
        .unwrap_or_else(|| "BTCUSDT".to_string());
    let mut kinds = std::collections::BTreeMap::<&str, u64>::new();
    let mut first_trade: Option<(i64, f64, f64, bool)> = None;
    let mut last_trade: Option<(i64, f64, f64, bool)> = None;
    let mut t_min = i64::MAX;
    let mut t_max = i64::MIN;
    for r in &records {
        match r {
            JournalRecord::SessionStart { .. } => *kinds.entry("session_start").or_insert(0) += 1,
            JournalRecord::Snapshot { .. } => *kinds.entry("snapshot").or_insert(0) += 1,
            JournalRecord::Depth { time_ms, .. } => {
                *kinds.entry("depth").or_insert(0) += 1;
                t_min = t_min.min(*time_ms);
                t_max = t_max.max(*time_ms);
            }
            JournalRecord::Trade {
                time_ms,
                price,
                qty,
                is_buyer_maker,
            } => {
                *kinds.entry("trade").or_insert(0) += 1;
                t_min = t_min.min(*time_ms);
                t_max = t_max.max(*time_ms);
                let row = (*time_ms, *price, *qty, *is_buyer_maker);
                if first_trade.is_none() {
                    first_trade = Some(row);
                }
                last_trade = Some(row);
            }
            JournalRecord::Overflow { .. } => *kinds.entry("overflow").or_insert(0) += 1,
            JournalRecord::SessionEnd { .. } => *kinds.entry("session_end").or_insert(0) += 1,
        }
    }
    if t_min == i64::MAX {
        return Err("journal holds no depth or trade records".to_string());
    }
    // Touch quotes come from the rebuilt session, which is the same
    // data the export consumed.
    let quotes = crate::storage::read_quotes(
        cfg.session_dir
            .join("quotes.parquet")
            .to_str()
            .ok_or("session dir is not UTF-8")?,
    )
    .map_err(|e| format!("cannot read rebuilt quotes: {e}"))?;
    let trades = crate::storage::read_trades(
        cfg.session_dir
            .join("trades.parquet")
            .to_str()
            .ok_or("session dir is not UTF-8")?,
    )
    .map_err(|e| format!("cannot read rebuilt trades: {e}"))?;
    if quotes.is_empty() || trades.is_empty() {
        return Err("rebuilt session is empty".to_string());
    }
    let fq = &quotes[0];
    let lq = &quotes[quotes.len() - 1];
    let first_quote = (fq.time_ms, fq.bid_px, fq.bid_sz, fq.ask_px, fq.ask_sz);
    let last_quote = (lq.time_ms, lq.bid_px, lq.bid_sz, lq.ask_px, lq.ask_sz);
    let first_mid = (fq.bid_px + fq.ask_px) / 2.0;
    let last_mid = (lq.bid_px + lq.ask_px) / 2.0;
    let volume: f64 = trades.iter().map(|t| t.qty).sum();

    let summary_text = std::fs::read_to_string(cfg.export_dir.join("summary.json"))
        .map_err(|e| format!("cannot read export summary.json: {e}"))?;
    let summary: Value = serde_json::from_str(&summary_text)
        .map_err(|e| format!("export summary.json must parse: {e}"))?;
    let speed = &summary["speed"];
    let at_60 = speed["horizons_s"]
        .as_array()
        .and_then(|h| h.iter().position(|v| v.as_f64() == Some(60.0)))
        .ok_or("export speed horizons must include 60 s")?;
    let response_60 = speed["response_bps"]
        .as_array()
        .and_then(|r| r[at_60].as_f64())
        .ok_or("export speed response must hold 60 s")?;

    let mut inputs = Vec::new();
    inputs.push(file_entry(
        &cfg.journal,
        &repo_rel(&cfg.repo_root, &cfg.journal),
        "journal",
    )?);
    let lock = cfg.repo_root.join("Cargo.lock");
    inputs.push(file_entry(&lock, "Cargo.lock", "dependency-lock")?);
    let (lock_sha, _) = sha256_file(&lock)?;

    let mut session_files = Vec::new();
    for name in SESSION_FILES {
        session_files.push(named_entry(&cfg.session_dir, name)?);
    }
    let mut output_files = Vec::new();
    for name in EXPORT_FILES {
        output_files.push(named_entry(&cfg.export_dir, name)?);
    }

    let snapshots = kinds.get("snapshot").copied().unwrap_or(0);
    let resyncs = snapshots.saturating_sub(1);
    let (ft_ms, ft_price, ft_qty, ft_maker) = first_trade.unwrap();
    let (lt_ms, lt_price, lt_qty, lt_maker) = last_trade.unwrap();
    let (fq_ms, fq_bid, fq_bid_sz, fq_ask, fq_ask_sz) = first_quote;
    let (lq_ms, lq_bid, lq_bid_sz, lq_ask, lq_ask_sz) = last_quote;

    let mut root = serde_json::Map::new();
    root.insert(
        "schema_version".to_string(),
        Value::String(EVIDENCE_SCHEMA.to_string()),
    );
    root.insert(
        "kind".to_string(),
        Value::String("evidence-manifest".to_string()),
    );
    root.insert(
        "experiment_id".to_string(),
        Value::String(cfg.experiment_id.clone()),
    );
    root.insert("title".to_string(), Value::String(cfg.title.clone()));
    root.insert(
        "status".to_string(),
        Value::String("reproducible".to_string()),
    );
    root.insert(
        "capture".to_string(),
        json!({
            "symbol": symbol,
            "source_session": cfg.source_session,
            "source_session_id": cfg.source_session_id,
            "ws_endpoint": format!("{WS_HOST}/stream?streams={}@aggTrade/{}@depth@100ms", symbol.to_lowercase(), symbol.to_lowercase()),
            "rest_endpoint": format!("{REST_HOST}/api/v3/depth"),
            "utc_start": ms_to_utc_iso(t_min),
            "utc_end": ms_to_utc_iso(t_max),
            "chunk_edges": [
                {"label": "first_trade", "time_ms": ft_ms, "utc": ms_to_utc_iso(ft_ms)},
                {"label": "last_trade", "time_ms": lt_ms, "utc": ms_to_utc_iso(lt_ms)},
                {"label": "first_quote", "time_ms": fq_ms, "utc": ms_to_utc_iso(fq_ms)},
                {"label": "last_quote", "time_ms": lq_ms, "utc": ms_to_utc_iso(lq_ms)},
            ],
            "n_records": records.len(),
            "record_kinds": kinds,
        }),
    );
    root.insert(
        "software".to_string(),
        json!({
            "git_sha": cfg.git_sha,
            "cargo_lock_sha256": lock_sha,
            "rustc_version": cfg.rustc_version,
        }),
    );
    root.insert("inputs".to_string(), Value::Array(inputs));
    root.insert(
        "session".to_string(),
        json!({
            "dir": "runs/tf7_session",
            "note": "Rebuilt from the fixture journal by journal-to-session; the rebuilt directory is not committed, so verification rebuilds it.",
            "files": session_files,
        }),
    );
    root.insert(
        "resync_gap".to_string(),
        json!({
            "snapshots": snapshots,
            "resyncs": resyncs,
            "note": "One initial snapshot loads the book; a resync is a snapshot after the first, and this slice has none.",
            "n_bars": summary["gaps"]["n_bars"],
            "n_dataset_rows": summary["gaps"]["n_dataset_rows"],
            "n_unobserved_bars": summary["gaps"]["n_unobserved_bars"],
            "bar_ms": summary["gaps"]["bar_ms"],
        }),
    );
    root.insert(
        "parameters".to_string(),
        json!({
            "vpin_buckets": 50,
            "vpin_window": 20,
            "vpin_bucket_volume": "50 equal-volume buckets (default export)",
            "ofi_bar_ms": 10_000,
            "kyle_window_bars": 60,
            "speed_bar_ms": 10_000,
            "speed_shock": "one standard deviation of per-bar OFI; an observational regressor measured from the same quote stream as the price, no news timestamps",
            "predict_horizon_bars": 3,
            "predict_folds": 4,
            "mm": "toy market-making simulation with the default MmConfig",
        }),
    );
    root.insert(
        "labels".to_string(),
        json!({
            "vpin": "VPIN over time, equal-volume buckets",
            "kyle_lambda": "Kyle's lambda, rolling 60-bar OLS of midprice change on signed volume",
            "ofi": "OFI per 10 s bar with midprice, spread, imbalance, signed volume, and 30 s forward mid return",
            "speed": "Observational OFI-price response (local projections, inclusive baseline)",
            "calibration": "Direction predictor calibration by predicted-probability decile",
            "summary": "Session summary: counts, span, volume, price range, VPIN, Kyle lambda, OFI, speed fit, prediction, and market-making PnL",
        }),
    );
    root.insert(
        "outputs".to_string(),
        json!({
            "dir": "runs/tf7_dashboard_data",
            "note": "The prediction ledger under experiments/<run>/ is excluded from the output hashes: it carries a wall-clock creation timestamp, so its bytes differ across runs while its numbers do not.",
            "files": output_files,
        }),
    );
    root.insert(
        "expected".to_string(),
        json!({
            "tolerance": 1e-9,
            "n_trades": summary["n_trades"],
            "n_quotes": summary["n_quotes"],
            "span_s": summary["span_s"],
            "volume_btc": volume,
            "buy_share": summary["buy_share"],
            "price_lo": summary["price_lo"],
            "price_hi": summary["price_hi"],
            "vpin_mean": summary["vpin"]["mean"],
            "vpin_max": summary["vpin"]["max"],
            "kyle_lambda_mean": summary["kyle_lambda"]["mean"],
            "ofi_mean": summary["ofi"]["mean"],
            "ofi_std": summary["ofi"]["std"],
            "predict_accuracy": summary["predict"]["accuracy"],
            "predict_baseline": summary["predict"]["baseline"],
            "mm_pnl_open": summary["mm"]["pnl_open"],
            "mm_pnl_gated": summary["mm"]["pnl_gated"],
            "speed_fit_ok": summary["speed"]["fit_ok"],
            "speed_fit_r_squared": summary["speed"]["fit_r_squared"],
            "speed_response_bps_60s": response_60,
            "gaps_n_bars": summary["gaps"]["n_bars"],
            "gaps_n_dataset_rows": summary["gaps"]["n_dataset_rows"],
            "first_trade": {"time_ms": ft_ms, "price": ft_price, "qty": ft_qty, "is_buyer_maker": ft_maker},
            "last_trade": {"time_ms": lt_ms, "price": lt_price, "qty": lt_qty, "is_buyer_maker": lt_maker},
            "first_quote": {"time_ms": fq_ms, "bid_px": fq_bid, "bid_sz": fq_bid_sz, "ask_px": fq_ask, "ask_sz": fq_ask_sz},
            "last_quote": {"time_ms": lq_ms, "bid_px": lq_bid, "bid_sz": lq_bid_sz, "ask_px": lq_ask, "ask_sz": lq_ask_sz},
            "first_mid": first_mid,
            "last_mid": last_mid,
        }),
    );
    root.insert(
        "fetch".to_string(),
        json!({
            "method": "Clone the repo and run the three commands in experiments/manifest.json; no download is needed.",
            "note": "The fixture is the only session data in the repo. Full recorded sessions stay out of the repo and ship with their own checksums.",
        }),
    );
    Ok(Value::Object(root))
}

/// Check one expected number against the fresh export summary.
fn check_number(
    failures: &mut Vec<String>,
    expected: &Value,
    summary: &Value,
    key: &str,
    path: &str,
    tol: f64,
) {
    let want = expected[key].as_f64();
    let mut got = Some(summary);
    for p in path.split('.') {
        got = got.and_then(|v| v.get(p));
    }
    let got = got.and_then(|v| v.as_f64());
    match (want, got) {
        (Some(w), Some(g)) if (w - g).abs() <= tol => {}
        (Some(w), Some(g)) => failures.push(format!(
            "expected {key} {w} but the fresh export has {g} (tolerance {tol})"
        )),
        _ => failures.push(format!("expected number {key} is missing or not a number")),
    }
}

/// Verify an evidence manifest against the repo and a rebuilt
/// session plus export. Returns one failure string per mismatch;
/// empty means the manifest verifies.
pub fn verify_manifest(
    manifest: &Value,
    repo_root: &Path,
    session_dir: &Path,
    export_dir: &Path,
) -> Vec<String> {
    let mut failures: Vec<String> = Vec::new();
    if manifest["schema_version"].as_str() != Some(EVIDENCE_SCHEMA) {
        failures.push(format!("schema_version must be {EVIDENCE_SCHEMA}"));
    }
    if manifest["kind"].as_str() != Some("evidence-manifest") {
        failures.push("kind must be evidence-manifest".to_string());
    }
    if manifest["experiment_id"]
        .as_str()
        .is_none_or(|s| s.is_empty())
    {
        failures.push("experiment_id must be a non-empty string".to_string());
    }
    if manifest["status"].as_str() != Some("reproducible") {
        failures.push("verify only covers manifests with status reproducible".to_string());
    }
    let capture = &manifest["capture"];
    if capture["symbol"].as_str().is_none_or(|s| s.is_empty()) {
        failures.push("capture.symbol must be a non-empty string".to_string());
    }
    if !capture["ws_endpoint"]
        .as_str()
        .is_some_and(|s| s.contains("data-stream.binance.vision"))
    {
        failures.push("capture.ws_endpoint must name data-stream.binance.vision".to_string());
    }
    if !capture["rest_endpoint"]
        .as_str()
        .is_some_and(|s| s.contains("data-api.binance.vision"))
    {
        failures.push("capture.rest_endpoint must name data-api.binance.vision".to_string());
    }
    let git_sha = manifest["software"]["git_sha"].as_str().unwrap_or("");
    if git_sha.len() != 40 || !git_sha.chars().all(|c| c.is_ascii_hexdigit()) {
        failures.push("software.git_sha must be 40 hex chars".to_string());
    }
    // Dependency lock state: the lock file in this checkout must hash
    // to the recorded value.
    match sha256_file(&repo_root.join("Cargo.lock")) {
        Ok((sha, _)) => {
            if manifest["software"]["cargo_lock_sha256"].as_str() != Some(sha.as_str()) {
                failures.push(
                    "software.cargo_lock_sha256 does not match Cargo.lock in this checkout"
                        .to_string(),
                );
            }
        }
        Err(e) => failures.push(e),
    }
    // Inputs: repo-relative files, re-hashed from this checkout.
    let empty_inputs: Vec<Value> = Vec::new();
    let inputs = manifest["inputs"].as_array().unwrap_or(&empty_inputs);
    if inputs.is_empty() {
        failures.push("inputs must not be empty".to_string());
    }
    let mut journal_path: Option<PathBuf> = None;
    for input in inputs {
        let rel = input["path"].as_str().unwrap_or("");
        let path = repo_root.join(rel);
        match sha256_file(&path) {
            Ok((sha, bytes)) => {
                if input["sha256"].as_str() != Some(sha.as_str()) {
                    failures.push(format!("input {rel} sha256 mismatch"));
                }
                if input["bytes"].as_u64() != Some(bytes) {
                    failures.push(format!("input {rel} byte length mismatch"));
                }
            }
            Err(e) => failures.push(e),
        }
        if input["kind"].as_str() == Some("journal") {
            journal_path = Some(path);
        }
    }
    // The journal recount must match the recorded capture accounting.
    if let Some(journal) = journal_path {
        match load_journal_jsonl(&journal) {
            Ok(records) => {
                if manifest["capture"]["n_records"].as_u64() != Some(records.len() as u64) {
                    failures.push(format!(
                        "capture.n_records {} does not match the journal record count {}",
                        manifest["capture"]["n_records"],
                        records.len()
                    ));
                }
                let mut kinds = std::collections::BTreeMap::<&str, u64>::new();
                for r in &records {
                    let kind = match r {
                        JournalRecord::SessionStart { .. } => "session_start",
                        JournalRecord::Snapshot { .. } => "snapshot",
                        JournalRecord::Depth { .. } => "depth",
                        JournalRecord::Trade { .. } => "trade",
                        JournalRecord::Overflow { .. } => "overflow",
                        JournalRecord::SessionEnd { .. } => "session_end",
                    };
                    *kinds.entry(kind).or_insert(0) += 1;
                }
                for (kind, count) in kinds {
                    if manifest["capture"]["record_kinds"][kind].as_u64() != Some(count) {
                        failures.push(format!(
                            "capture.record_kinds.{kind} does not match the journal recount"
                        ));
                    }
                }
            }
            Err(e) => failures.push(format!("cannot recount journal: {e}")),
        }
    } else {
        failures.push("inputs must include one journal entry".to_string());
    }
    // Session files: re-hashed from the rebuilt session directory.
    for file in manifest["session"]["files"]
        .as_array()
        .unwrap_or(&empty_inputs)
    {
        let name = file["name"].as_str().unwrap_or("");
        match sha256_file(&session_dir.join(name)) {
            Ok((sha, bytes)) => {
                if file["sha256"].as_str() != Some(sha.as_str()) {
                    failures.push(format!("session file {name} sha256 mismatch"));
                }
                if file["bytes"].as_u64() != Some(bytes) {
                    failures.push(format!("session file {name} byte length mismatch"));
                }
            }
            Err(e) => failures.push(e),
        }
    }
    // Output files: re-hashed from the fresh export directory.
    for file in manifest["outputs"]["files"]
        .as_array()
        .unwrap_or(&empty_inputs)
    {
        let name = file["name"].as_str().unwrap_or("");
        match sha256_file(&export_dir.join(name)) {
            Ok((sha, bytes)) => {
                if file["sha256"].as_str() != Some(sha.as_str()) {
                    failures.push(format!("output {name} sha256 mismatch"));
                }
                if file["bytes"].as_u64() != Some(bytes) {
                    failures.push(format!("output {name} byte length mismatch"));
                }
            }
            Err(e) => failures.push(e),
        }
    }
    // Expected numbers: the fresh export summary within tolerance.
    let summary_text = std::fs::read_to_string(export_dir.join("summary.json"));
    match summary_text {
        Ok(text) => match serde_json::from_str::<Value>(&text) {
            Ok(summary) => {
                let expected = &manifest["expected"];
                let tol = expected["tolerance"].as_f64().unwrap_or(0.0);
                if tol <= 0.0 {
                    failures.push("expected.tolerance must be positive".to_string());
                }
                for (key, path) in [
                    ("n_trades", "n_trades"),
                    ("n_quotes", "n_quotes"),
                    ("span_s", "span_s"),
                    ("volume_btc", "volume_btc"),
                    ("buy_share", "buy_share"),
                    ("price_lo", "price_lo"),
                    ("price_hi", "price_hi"),
                    ("vpin_mean", "vpin.mean"),
                    ("vpin_max", "vpin.max"),
                    ("kyle_lambda_mean", "kyle_lambda.mean"),
                    ("ofi_mean", "ofi.mean"),
                    ("ofi_std", "ofi.std"),
                    ("predict_accuracy", "predict.accuracy"),
                    ("predict_baseline", "predict.baseline"),
                    ("mm_pnl_open", "mm.pnl_open"),
                    ("mm_pnl_gated", "mm.pnl_gated"),
                    ("speed_fit_r_squared", "speed.fit_r_squared"),
                ] {
                    check_number(&mut failures, expected, &summary, key, path, tol);
                }
                if expected["speed_fit_ok"].as_bool() != summary["speed"]["fit_ok"].as_bool() {
                    failures.push("expected speed_fit_ok does not match the fresh export".into());
                }
                let horizons = summary["speed"]["horizons_s"].as_array();
                let response = summary["speed"]["response_bps"].as_array();
                match (horizons, response) {
                    (Some(h), Some(r)) => match h.iter().position(|v| v.as_f64() == Some(60.0)) {
                        Some(at_60) => {
                            let want = expected["speed_response_bps_60s"].as_f64();
                            let got = r[at_60].as_f64();
                            match (want, got) {
                                    (Some(w), Some(g)) if (w - g).abs() <= tol => {}
                                    (Some(w), Some(g)) => failures.push(format!(
                                        "expected speed_response_bps_60s {w} but the fresh export has {g}"
                                    )),
                                    _ => failures.push(
                                        "speed_response_bps_60s is missing or not a number"
                                            .to_string(),
                                    ),
                                }
                        }
                        None => {
                            failures.push("fresh export speed horizons must include 60 s".into())
                        }
                    },
                    _ => failures.push("fresh export speed block is missing".to_string()),
                }
                if expected["gaps_n_bars"].as_u64() != summary["gaps"]["n_bars"].as_u64() {
                    failures.push("expected gaps_n_bars does not match the fresh export".into());
                }
                if expected["gaps_n_dataset_rows"].as_u64()
                    != summary["gaps"]["n_dataset_rows"].as_u64()
                {
                    failures.push(
                        "expected gaps_n_dataset_rows does not match the fresh export".into(),
                    );
                }
            }
            Err(e) => failures.push(format!("fresh export summary.json must parse: {e}")),
        },
        Err(e) => failures.push(format!("cannot read fresh export summary.json: {e}")),
    }
    failures
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ms_to_utc_iso_matches_known_timestamps() {
        assert_eq!(ms_to_utc_iso(0), "1970-01-01T00:00:00.000Z");
        assert_eq!(ms_to_utc_iso(1_794_153_897_314), "2026-11-08T16:04:57.314Z");
        assert_eq!(ms_to_utc_iso(1_794_154_628_414), "2026-11-08T16:17:08.414Z");
        assert_eq!(ms_to_utc_iso(1_700_000_000_000), "2023-11-14T22:13:20.000Z");
    }

    #[test]
    fn sha256_hex_matches_known_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
