//! Live recording from Binance market-data streams (no API key).
//!
//! Combined stream over the market-data-only host:
//!   wss://data-stream.binance.vision/stream?streams=<s>@aggTrade/<s>@depth@100ms
//! Snapshots come from https://data-api.binance.vision/api/v3/depth.
//!
//! The local book follows the documented sync procedure: buffer diffs while
//! a snapshot is fetched in the background, discard stale diffs, verify the
//! [U, u] continuity, then apply. A sequence gap triggers a full resync,
//! and a dropped connection triggers reconnect with backoff (which always
//! resyncs the book).

use std::error::Error;
use std::time::{Duration, Instant, SystemTime};

use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;
use tokio_tungstenite::tungstenite::Message;

use crate::metrics::{Quote, Trade};
use crate::orderbook::{OrderBook, SyncResult};
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

fn parse_pair(v: &(String, String)) -> (f64, f64) {
    (v.0.parse().unwrap_or(0.0), v.1.parse().unwrap_or(0.0))
}

fn parse_levels(v: &serde_json::Value) -> Vec<(f64, f64)> {
    v.as_array()
        .map(|a| {
            a.iter()
                .map(|x| {
                    let p = x[0].as_str().unwrap_or("0").to_string();
                    let q = x[1].as_str().unwrap_or("0").to_string();
                    parse_pair(&(p, q))
                })
                .collect()
        })
        .unwrap_or_default()
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

struct Session {
    book: OrderBook,
    synced: bool,
    syncing: bool,
    buffer: Vec<serde_json::Value>,
    pending: Option<oneshot::Receiver<Result<Snapshot, String>>>,
    trades: Vec<Trade>,
    quotes: Vec<Quote>,
    resyncs: u32,
}

impl Session {
    fn new() -> Self {
        Self {
            book: OrderBook::new(),
            synced: false,
            syncing: false,
            buffer: Vec::new(),
            pending: None,
            trades: Vec::new(),
            quotes: Vec::new(),
            resyncs: 0,
        }
    }

    fn record_touch(&mut self, event_ms: i64) {
        if let Some((bp, bq, ap, aq)) = self.book.touch() {
            self.quotes.push(Quote {
                time_ms: event_ms,
                bid_px: bp,
                bid_sz: bq,
                ask_px: ap,
                ask_sz: aq,
            });
        }
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

    /// Merge a fetched snapshot with the buffered diffs. Returns true when
    /// the book is in sync afterwards.
    ///
    /// Follows the exchange procedure: events fully covered by the snapshot
    /// (u <= lastUpdateId) are dropped, and the first remaining event must
    /// satisfy U <= lastUpdateId + 1; otherwise the snapshot is too old and
    /// we refetch.
    fn merge_snapshot(&mut self, snap: &Snapshot) -> bool {
        let first_live_u = self.buffer.iter().find_map(|e| {
            let u = e["u"].as_i64().unwrap_or(0);
            if u > snap.last_update_id {
                e["U"].as_i64()
            } else {
                None
            }
        });
        if let Some(fu) = first_live_u {
            if snap.last_update_id + 1 < fu {
                return false; // snapshot too old to continue the stream: refetch
            }
        }
        let bids: Vec<(f64, f64)> = snap.bids.iter().map(parse_pair).collect();
        let asks: Vec<(f64, f64)> = snap.asks.iter().map(parse_pair).collect();
        self.book.from_snapshot(&bids, &asks, snap.last_update_id);
        let buffered = std::mem::take(&mut self.buffer);
        for e in buffered {
            let u = e["u"].as_i64().unwrap_or(0);
            if u <= snap.last_update_id {
                continue; // stale
            }
            let uu = e["U"].as_i64().unwrap_or(0);
            let b = parse_levels(&e["b"]);
            let a = parse_levels(&e["a"]);
            if self.book.apply_diff(uu, u, &b, &a) == SyncResult::Gap {
                return false;
            }
            self.record_touch(e["E"].as_i64().unwrap_or(0));
        }
        true
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
                self.synced = true;
                self.resyncs += 1;
            }
            // on failure the next loop iteration kicks a fresh fetch
        }
    }

    fn on_depth(&mut self, data: &serde_json::Value, symbol: &str) {
        self.poll_sync();
        if !self.synced {
            self.buffer.push(data.clone());
            self.kick_sync(symbol);
            return;
        }
        let u_first = data["U"].as_i64().unwrap_or(0);
        let u_last = data["u"].as_i64().unwrap_or(0);
        let b = parse_levels(&data["b"]);
        let a = parse_levels(&data["a"]);
        match self.book.apply_diff(u_first, u_last, &b, &a) {
            SyncResult::Applied => {
                self.record_touch(data["E"].as_i64().unwrap_or(0));
            }
            SyncResult::Stale => {}
            SyncResult::Gap => {
                self.synced = false;
                self.buffer = vec![data.clone()];
            }
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
        session.synced = false;
        session.syncing = false;
        session.pending = None;
        session.buffer.clear();

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

    #[test]
    fn merge_applies_buffered_diffs_after_snapshot() {
        let mut s = Session::new();
        s.buffer.push(depth_event(101, 102, &[(100.0, 6.0)], &[]));
        s.buffer
            .push(depth_event(103, 103, &[(99.9, 0.0)], &[(100.2, 9.0)]));
        assert!(s.merge_snapshot(&snapshot(100)));
        assert!(s.buffer.is_empty());
        assert_eq!(s.book.best_bid(), Some((100.0, 6.0)));
        // 99.9 removed, only 100.0 remains on the bid side
        assert_eq!(s.book.bid_levels(), vec![(100.0, 6.0)]);
        assert_eq!(s.book.ask_levels(), vec![(100.1, 4.0), (100.2, 9.0)]);
        assert_eq!(s.book.last_id, 103);
        // each applied event recorded a quote
        assert_eq!(s.quotes.len(), 2);
    }

    #[test]
    fn merge_skips_buffered_events_older_than_snapshot() {
        let mut s = Session::new();
        // fully stale: u <= snapshot id
        s.buffer.push(depth_event(95, 98, &[(100.0, 99.0)], &[]));
        // overlapping the snapshot edge
        s.buffer.push(depth_event(99, 101, &[(100.0, 7.0)], &[]));
        assert!(s.merge_snapshot(&snapshot(100)));
        assert_eq!(s.book.best_bid(), Some((100.0, 7.0)));
        assert_eq!(s.book.last_id, 101);
    }

    #[test]
    fn merge_refetches_when_snapshot_predates_buffer() {
        let mut s = Session::new();
        s.buffer.push(depth_event(201, 202, &[(100.0, 6.0)], &[]));
        // snapshot is older than the first buffered event: cannot merge
        assert!(!s.merge_snapshot(&snapshot(100)));
        // book untouched, buffer retained for the next attempt
        assert_eq!(s.book.last_id, -1);
        assert_eq!(s.buffer.len(), 1);
    }

    #[test]
    fn merge_refetches_on_gap_inside_buffer() {
        let mut s = Session::new();
        s.buffer.push(depth_event(101, 102, &[(100.0, 6.0)], &[]));
        // update 103 never arrived
        s.buffer.push(depth_event(104, 104, &[(100.0, 8.0)], &[]));
        assert!(!s.merge_snapshot(&snapshot(100)));
    }

    #[test]
    fn merge_with_empty_buffer_syncs_to_snapshot() {
        let mut s = Session::new();
        assert!(s.merge_snapshot(&snapshot(100)));
        assert_eq!(s.book.best_bid(), Some((100.0, 5.0)));
        assert_eq!(s.book.last_id, 100);
        assert!(s.quotes.is_empty());
    }

    #[test]
    fn on_depth_gap_keeps_triggering_event_for_resync() {
        let mut s = Session::new();
        // drive the session to synced state first
        s.buffer.push(depth_event(101, 101, &[], &[]));
        assert!(s.merge_snapshot(&snapshot(100)));
        s.synced = true;

        // a gap: the triggering event must be retained as the new buffer head
        let gap = depth_event(110, 111, &[(100.0, 6.0)], &[]);
        s.on_depth(&gap, "BTCUSDT");
        assert!(!s.synced);
        assert_eq!(s.buffer.len(), 1);
        assert_eq!(s.buffer[0]["U"].as_i64(), Some(110));

        // a snapshot that cannot continue the stream forces another refetch:
        // first live event has U=110, so lastUpdateId must be >= 109
        let mut too_old = snapshot(100);
        too_old.last_update_id = 108;
        assert!(!s.merge_snapshot(&too_old));
        // lastUpdateId = 109 satisfies U <= lastUpdateId + 1: merges cleanly
        let mut snap2 = snapshot(100);
        snap2.last_update_id = 109;
        assert!(s.merge_snapshot(&snap2));
        assert_eq!(s.book.best_bid(), Some((100.0, 6.0)));
        assert_eq!(s.book.last_id, 111);
    }

    #[test]
    fn merge_accepts_snapshot_covering_part_of_buffer() {
        // The snapshot may be newer than some buffered events; those are
        // simply skipped as stale and the stream continues.
        let mut s = Session::new();
        s.buffer.push(depth_event(101, 102, &[(100.0, 6.0)], &[]));
        s.buffer.push(depth_event(103, 104, &[(100.0, 8.0)], &[]));
        let mut snap = snapshot(100);
        snap.last_update_id = 103;
        assert!(s.merge_snapshot(&snap));
        assert_eq!(s.book.best_bid(), Some((100.0, 8.0)));
        assert_eq!(s.book.last_id, 104);
    }

    #[test]
    fn on_depth_stale_event_leaves_book_alone() {
        let mut s = Session::new();
        s.buffer.push(depth_event(101, 101, &[], &[]));
        assert!(s.merge_snapshot(&snapshot(100)));
        s.synced = true;
        let before = s.book.bid_levels();
        let n_quotes = s.quotes.len();
        s.on_depth(&depth_event(90, 95, &[(100.0, 1.0)], &[]), "BTCUSDT");
        assert!(s.synced);
        assert_eq!(s.book.bid_levels(), before);
        assert_eq!(s.quotes.len(), n_quotes); // stale event records nothing
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
