//! Live recording from Binance market-data streams (no API key).
//!
//! Combined stream over the market-data-only host:
//!   wss://data-stream.binance.vision/stream?streams=<s>@aggTrade/<s>@depth@100ms
//! Snapshots come from https://data-api.binance.vision/api/v3/depth.
//!
//! Book synchronization lives in `booksync`: one shared state machine
//! with transactional snapshot merges. This module owns the network
//! loop and the recorded trades/quotes; it never applies a depth event
//! that failed validation, and a failed merge emits no quotes.

use std::error::Error;
use std::time::{Duration, Instant, SystemTime};

use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;
use tokio_tungstenite::tungstenite::Message;

use crate::booksync::{
    parse_level_pair, BookSync, DepthEvent, DepthOutcome, MergeOutcome, SnapshotData,
};
use crate::metrics::{Quote, Trade};
use crate::orderbook::OrderBook;
use crate::storage;

const WS_HOST: &str = "wss://data-stream.binance.vision";
const REST_HOST: &str = "https://data-api.binance.vision";

#[derive(Debug, Serialize, Deserialize)]
pub struct SessionMeta {
    pub symbol: String,
    pub n_trades: usize,
    pub n_quotes: usize,
    pub resyncs: u32,
    pub t0_ms: i64,
    pub t1_ms: i64,
    pub saved_at: String,
}

#[derive(Debug, Deserialize)]
struct Snapshot {
    #[serde(rename = "lastUpdateId")]
    last_update_id: i64,
    bids: Vec<(String, String)>,
    asks: Vec<(String, String)>,
}

impl Snapshot {
    /// Convert the raw REST payload into a validated snapshot. A
    /// malformed level is an error here, not a zero.
    fn validated(&self) -> Result<SnapshotData, String> {
        if self.last_update_id < 0 {
            return Err(format!("lastUpdateId {} is negative", self.last_update_id));
        }
        let conv = |lv: &[(String, String)]| -> Result<Vec<(f64, f64)>, String> {
            lv.iter().map(parse_pair).collect()
        };
        Ok(SnapshotData {
            last_update_id: self.last_update_id,
            bids: conv(&self.bids)?,
            asks: conv(&self.asks)?,
        })
    }
}

/// Strict pair parsing: any malformed or non-finite value is an
/// error. (The previous implementation silently substituted 0.0.)
fn parse_pair(v: &(String, String)) -> Result<(f64, f64), String> {
    parse_level_pair(&v.0, &v.1)
}

/// Strict level-array parsing: any malformed level fails the whole
/// array instead of being recorded as a zero level.
#[cfg(test)]
fn parse_levels(v: &serde_json::Value) -> Result<Vec<(f64, f64)>, String> {
    crate::booksync::parse_levels_json(v)
}

async fn fetch_snapshot(symbol: &str) -> Result<Snapshot, String> {
    let url = format!(
        "{REST_HOST}/api/v3/depth?symbol={}&limit=1000",
        symbol.to_uppercase()
    );
    let snap: Snapshot = reqwest::get(&url)
        .await
        .map_err(|e| e.to_string())?
        .json()
        .await
        .map_err(|e| e.to_string())?;
    Ok(snap)
}

/// One recording session: the shared sync machine plus the recorded
/// trades and quotes. Quotes enter `quotes` only when the state
/// machine commits the event (or the snapshot merge) that produced
/// them, so a failed merge emits nothing.
pub struct Session {
    sync: BookSync,
    syncing: bool,
    pending: Option<oneshot::Receiver<Result<Snapshot, String>>>,
    trades: Vec<Trade>,
    quotes: Vec<Quote>,
    resyncs: u32,
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

impl Session {
    pub fn new() -> Self {
        Self {
            sync: BookSync::new(),
            syncing: false,
            pending: None,
            trades: Vec::new(),
            quotes: Vec::new(),
            resyncs: 0,
        }
    }

    pub fn is_synced(&self) -> bool {
        self.sync.is_synced()
    }

    pub fn book(&self) -> &OrderBook {
        self.sync.book()
    }

    pub fn buffered_len(&self) -> usize {
        self.sync.buffered_len()
    }

    pub fn quotes(&self) -> &[Quote] {
        &self.quotes
    }

    fn record_touch(&mut self, event_ms: i64) {
        if let Some((bp, bq, ap, aq)) = self.sync.book().touch() {
            self.quotes.push(Quote {
                time_ms: event_ms,
                bid_px: bp,
                bid_sz: bq,
                ask_px: ap,
                ask_sz: aq,
            });
        }
    }

    /// A fresh connection always resyncs the book.
    fn reset_sync(&mut self) {
        self.sync = BookSync::new();
        self.syncing = false;
        self.pending = None;
    }

    /// Start a background snapshot fetch; the read loop keeps buffering.
    fn kick_sync(&mut self, symbol: &str) {
        if self.syncing {
            return;
        }
        self.syncing = true;
        let sym = symbol.to_string();
        let (tx, rx) = oneshot::channel();
        tokio::spawn(async move {
            let res = fetch_snapshot(&sym).await.map_err(|e| e.to_string());
            let _ = tx.send(res);
        });
        self.pending = Some(rx);
    }

    /// Merge a fetched snapshot with the buffered diffs. Transactional:
    /// returns true only when the candidate book and its quotes were
    /// committed together; on failure the book, the quotes, and the
    /// buffered events are exactly as before the attempt.
    fn merge_snapshot(&mut self, snap: &Snapshot) -> bool {
        match snap.validated() {
            Ok(data) => self.merge_snapshot_data(&data),
            Err(e) => {
                eprintln!("record: rejecting invalid snapshot: {e}");
                false
            }
        }
    }

    /// Transactional merge of an already-validated snapshot.
    pub fn merge_snapshot_data(&mut self, snap: &SnapshotData) -> bool {
        let mut pending_quotes = Vec::new();
        match self.sync.merge_snapshot(snap, &mut pending_quotes) {
            MergeOutcome::Synced { .. } => {
                for (event_ms, (bp, bq, ap, aq)) in pending_quotes {
                    self.quotes.push(Quote {
                        time_ms: event_ms,
                        bid_px: bp,
                        bid_sz: bq,
                        ask_px: ap,
                        ask_sz: aq,
                    });
                }
                true
            }
            MergeOutcome::Refetch => false,
        }
    }

    /// Poll for a finished snapshot fetch and merge it.
    fn poll_sync(&mut self) {
        if !self.syncing {
            return;
        }
        let done = match self.pending.as_mut() {
            Some(rx) => match rx.try_recv() {
                Ok(Ok(snap)) => Some(self.merge_snapshot(&snap)),
                Ok(Err(_)) | Err(oneshot::error::TryRecvError::Closed) => Some(false),
                Err(oneshot::error::TryRecvError::Empty) => None,
            },
            None => Some(false),
        };
        if let Some(ok) = done {
            self.pending = None;
            self.syncing = false;
            if ok {
                self.resyncs += 1;
            }
            // on failure the next loop iteration kicks a fresh fetch
        }
    }

    pub fn on_depth(&mut self, data: &serde_json::Value, symbol: &str) {
        self.poll_sync();
        let ev = match DepthEvent::from_json(data) {
            Ok(ev) => ev,
            Err(e) => {
                // fail closed: an invalid message is dropped with a
                // diagnostic, never applied as zeros
                eprintln!("record: dropping invalid depthUpdate: {e}");
                return;
            }
        };
        if !self.sync.is_synced() {
            self.sync.on_depth(ev);
            self.kick_sync(symbol);
            return;
        }
        let event_ms = ev.event_ms;
        if let DepthOutcome::Applied = self.sync.on_depth(ev) {
            self.record_touch(event_ms);
        }
    }

    fn on_trade(&mut self, data: &serde_json::Value) {
        let t = Trade {
            time_ms: data["T"].as_i64().unwrap_or(0),
            price: data["p"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
            qty: data["q"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
            is_buyer_maker: data["m"].as_bool().unwrap_or(false),
        };
        if t.qty > 0.0 {
            self.trades.push(t);
        }
    }
}

/// Record `minutes` minutes of market data into `out_dir`.
/// Writes trades.parquet, quotes.parquet, meta.json.
pub async fn record_session(
    symbol: &str,
    minutes: f64,
    out_dir: &str,
) -> Result<SessionMeta, Box<dyn Error>> {
    // Validate the connection policy before the retry loop: a bad proxy
    // URL will never succeed, so retrying it until the deadline would
    // just burn the whole session. Fail fast with an actionable error.
    let proxy = crate::net::ProxyConfig::from_env().map_err(|e| Box::new(e) as Box<dyn Error>)?;
    eprintln!("record: connecting via {}", proxy.describe());
    let sym = symbol.to_lowercase();
    let url = format!("{WS_HOST}/stream?streams={sym}@aggTrade/{sym}@depth@100ms");
    let deadline = Instant::now() + Duration::from_secs_f64(minutes * 60.0);
    let mut session = Session::new();
    let mut backoff = Duration::from_secs(1);

    'outer: while Instant::now() < deadline {
        let ws = async {
            let stream = crate::net::tls_stream("data-stream.binance.vision").await?;
            let (ws, _) = tokio_tungstenite::client_async(&url, stream)
                .await
                .map_err(|e| Box::new(e) as Box<dyn Error>)?;
            Ok::<_, Box<dyn Error>>(ws)
        }
        .await;
        let ws = match ws {
            Ok(x) => x,
            Err(_) => {
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(Duration::from_secs(30));
                continue;
            }
        };
        backoff = Duration::from_secs(1);
        let (mut write, mut read) = ws.split();
        // a fresh connection always resyncs the book
        session.reset_sync();

        loop {
            if Instant::now() >= deadline {
                break 'outer;
            }
            session.poll_sync();
            let msg = tokio::time::timeout(Duration::from_secs(60), read.next()).await;
            let msg = match msg {
                Ok(Some(Ok(m))) => m,
                _ => break, // timeout, error, or closed: reconnect
            };
            match msg {
                Message::Text(txt) => {
                    let v: serde_json::Value = match serde_json::from_str(&txt) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    let data = &v["data"];
                    match data["e"].as_str() {
                        Some("aggTrade") => session.on_trade(data),
                        Some("depthUpdate") => session.on_depth(data, symbol),
                        _ => {}
                    }
                }
                Message::Ping(p) => {
                    let _ = write.send(Message::Pong(p)).await;
                }
                Message::Close(_) => break,
                _ => {}
            }
        }
    }

    std::fs::create_dir_all(out_dir)?;
    storage::write_trades(&format!("{out_dir}/trades.parquet"), &session.trades)?;
    storage::write_quotes(&format!("{out_dir}/quotes.parquet"), &session.quotes)?;
    let meta = SessionMeta {
        symbol: symbol.to_uppercase(),
        n_trades: session.trades.len(),
        n_quotes: session.quotes.len(),
        resyncs: session.resyncs,
        t0_ms: session.trades.first().map(|t| t.time_ms).unwrap_or(0),
        t1_ms: session.trades.last().map(|t| t.time_ms).unwrap_or(0),
        saved_at: format!("{:?}", SystemTime::now()),
    };
    std::fs::write(
        format!("{out_dir}/meta.json"),
        serde_json::to_string_pretty(&meta)?,
    )?;
    Ok(meta)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A depthUpdate-shaped event, as received on the websocket stream.
    fn depth_event(
        u_first: i64,
        u_last: i64,
        bids: &[(f64, f64)],
        asks: &[(f64, f64)],
    ) -> serde_json::Value {
        let pairs = |lv: &[(f64, f64)]| {
            lv.iter()
                .map(|(p, q)| vec![p.to_string(), q.to_string()])
                .collect::<Vec<_>>()
        };
        serde_json::json!({
            "e": "depthUpdate",
            "E": 1_700_000_000_000i64,
            "U": u_first,
            "u": u_last,
            "b": pairs(bids),
            "a": pairs(asks),
        })
    }

    fn snapshot(last_update_id: i64) -> Snapshot {
        Snapshot {
            last_update_id,
            bids: vec![
                ("100.0".to_string(), "5.0".to_string()),
                ("99.9".to_string(), "3.0".to_string()),
            ],
            asks: vec![
                ("100.1".to_string(), "4.0".to_string()),
                ("100.2".to_string(), "2.0".to_string()),
            ],
        }
    }

    /// Buffer an event without triggering a snapshot fetch (the unit
    /// tests drive merges explicitly).
    fn buffer(s: &mut Session, e: &serde_json::Value) {
        let ev = DepthEvent::from_json(e).expect("test event is valid");
        s.sync.on_depth(ev);
    }

    #[test]
    fn merge_applies_buffered_diffs_after_snapshot() {
        let mut s = Session::new();
        buffer(&mut s, &depth_event(101, 102, &[(100.0, 6.0)], &[]));
        buffer(
            &mut s,
            &depth_event(103, 103, &[(99.9, 0.0)], &[(100.2, 9.0)]),
        );
        assert!(s.merge_snapshot(&snapshot(100)));
        assert_eq!(s.buffered_len(), 0);
        assert_eq!(s.book().best_bid(), Some((100.0, 6.0)));
        // 99.9 removed, only 100.0 remains on the bid side
        assert_eq!(s.book().bid_levels(), vec![(100.0, 6.0)]);
        assert_eq!(s.book().ask_levels(), vec![(100.1, 4.0), (100.2, 9.0)]);
        assert_eq!(s.book().last_id, 103);
        // each applied event recorded a quote
        assert_eq!(s.quotes.len(), 2);
    }

    #[test]
    fn merge_skips_buffered_events_older_than_snapshot() {
        let mut s = Session::new();
        // fully stale: u <= snapshot id
        buffer(&mut s, &depth_event(95, 98, &[(100.0, 99.0)], &[]));
        // overlapping the snapshot edge
        buffer(&mut s, &depth_event(99, 101, &[(100.0, 7.0)], &[]));
        assert!(s.merge_snapshot(&snapshot(100)));
        assert_eq!(s.book().best_bid(), Some((100.0, 7.0)));
        assert_eq!(s.book().last_id, 101);
    }

    #[test]
    fn merge_refetches_when_snapshot_predates_buffer() {
        let mut s = Session::new();
        buffer(&mut s, &depth_event(201, 202, &[(100.0, 6.0)], &[]));
        // snapshot is older than the first buffered event: cannot merge
        assert!(!s.merge_snapshot(&snapshot(100)));
        // book untouched, buffer retained for the next attempt
        assert_eq!(s.book().last_id, -1);
        assert_eq!(s.buffered_len(), 1);
    }

    #[test]
    fn merge_refetches_on_gap_inside_buffer() {
        let mut s = Session::new();
        buffer(&mut s, &depth_event(101, 102, &[(100.0, 6.0)], &[]));
        // update 103 never arrived
        buffer(&mut s, &depth_event(104, 104, &[(100.0, 8.0)], &[]));
        assert!(!s.merge_snapshot(&snapshot(100)));
        // transactional: the first event was NOT half-applied and no
        // quote was emitted for the failed attempt
        assert_eq!(s.book().last_id, -1);
        assert!(s.book().is_empty());
        assert!(s.quotes.is_empty());
        assert_eq!(s.buffered_len(), 2);
    }

    #[test]
    fn merge_with_empty_buffer_syncs_to_snapshot() {
        let mut s = Session::new();
        assert!(s.merge_snapshot(&snapshot(100)));
        assert_eq!(s.book().best_bid(), Some((100.0, 5.0)));
        assert_eq!(s.book().last_id, 100);
        assert!(s.quotes.is_empty());
    }

    #[test]
    fn on_depth_gap_keeps_triggering_event_for_resync() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let mut s = Session::new();
            buffer(&mut s, &depth_event(101, 101, &[], &[]));
            assert!(s.merge_snapshot(&snapshot(100)));

            // a gap: the triggering event must be retained as the new
            // buffer head (kick_sync spawns a fetch; it cannot complete
            // without network and is dropped with the session)
            let gap = depth_event(110, 111, &[(100.0, 6.0)], &[]);
            s.on_depth(&gap, "BTCUSDT");
            assert!(!s.is_synced());
            assert_eq!(s.buffered_len(), 1);

            // a snapshot that cannot continue the stream forces another
            // refetch: first live event has U=110, so lastUpdateId must
            // be >= 109
            let mut too_old = snapshot(100);
            too_old.last_update_id = 108;
            assert!(!s.merge_snapshot(&too_old));
            // lastUpdateId = 109 satisfies U <= lastUpdateId + 1: merges
            // cleanly
            let mut snap2 = snapshot(100);
            snap2.last_update_id = 109;
            assert!(s.merge_snapshot(&snap2));
            assert_eq!(s.book().best_bid(), Some((100.0, 6.0)));
            assert_eq!(s.book().last_id, 111);
        });
    }

    #[test]
    fn merge_accepts_snapshot_covering_part_of_buffer() {
        // The snapshot may be newer than some buffered events; those are
        // simply skipped as stale and the stream continues.
        let mut s = Session::new();
        buffer(&mut s, &depth_event(101, 102, &[(100.0, 6.0)], &[]));
        buffer(&mut s, &depth_event(103, 104, &[(100.0, 8.0)], &[]));
        let mut snap = snapshot(100);
        snap.last_update_id = 103;
        assert!(s.merge_snapshot(&snap));
        assert_eq!(s.book().best_bid(), Some((100.0, 8.0)));
        assert_eq!(s.book().last_id, 104);
    }

    #[test]
    fn on_depth_stale_event_leaves_book_alone() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let mut s = Session::new();
            buffer(&mut s, &depth_event(101, 101, &[], &[]));
            assert!(s.merge_snapshot(&snapshot(100)));
            let before = s.book().bid_levels();
            let n_quotes = s.quotes.len();
            s.on_depth(&depth_event(90, 95, &[(100.0, 1.0)], &[]), "BTCUSDT");
            assert!(s.is_synced());
            assert_eq!(s.book().bid_levels(), before);
            assert_eq!(s.quotes.len(), n_quotes); // stale event records nothing
        });
    }

    #[test]
    fn strict_parsing_rejects_malformed_values() {
        assert!(parse_pair(&("abc".to_string(), "1.0".to_string())).is_err());
        assert!(parse_pair(&("100.0".to_string(), "-1.0".to_string())).is_err());
        assert!(parse_pair(&("0".to_string(), "1.0".to_string())).is_err());
        assert_eq!(
            parse_pair(&("100.0".to_string(), "0".to_string())),
            Ok((100.0, 0.0))
        );
        let bad = serde_json::json!([["100.0", "1.0"], ["oops", "2.0"]]);
        assert!(parse_levels(&bad).is_err());
        let not_pairs = serde_json::json!([["100.0"]]);
        assert!(parse_levels(&not_pairs).is_err());
    }

    /// Serializes tests that mutate the proxy env vars (process-global).
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[tokio::test]
    async fn record_session_fails_fast_on_bad_proxy_config() {
        // An invalid proxy URL is a configuration error, not a transient
        // network failure: record_session must return it immediately
        // instead of retrying until the session deadline.
        let _guard = ENV_LOCK.lock().unwrap();
        let prev_lower = std::env::var("https_proxy").ok();
        let prev_upper = std::env::var("HTTPS_PROXY").ok();
        std::env::set_var("https_proxy", "socks5://proxy:8080");
        std::env::remove_var("HTTPS_PROXY");
        let t0 = std::time::Instant::now();
        // 3-second deadline: without fail-fast this would take ~3s of
        // backoff retries; with fail-fast it returns in milliseconds.
        let err = record_session("BTCUSDT", 0.05, "/tmp/tf8-should-not-exist")
            .await
            .expect_err("bad proxy config must fail");
        let elapsed = t0.elapsed();
        match prev_lower {
            Some(v) => std::env::set_var("https_proxy", v),
            None => std::env::remove_var("https_proxy"),
        }
        match prev_upper {
            Some(v) => std::env::set_var("HTTPS_PROXY", v),
            None => std::env::remove_var("HTTPS_PROXY"),
        }
        let msg = err.to_string();
        assert!(
            msg.starts_with("invalid proxy configuration: "),
            "actionable error, got: {msg}"
        );
        assert!(
            elapsed < std::time::Duration::from_secs(2),
            "failed fast in {elapsed:?}, not after the deadline"
        );
    }
}
