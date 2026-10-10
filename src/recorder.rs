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
    fn merge_snapshot(&mut self, snap: &Snapshot) -> bool {
        let first_u = self.buffer.first().and_then(|e| e["U"].as_i64());
        if let Some(fu) = first_u {
            if snap.last_update_id < fu {
                return false; // snapshot older than our buffer: refetch
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
    let sym = symbol.to_lowercase();
    let url = format!("{WS_HOST}/stream?streams={sym}@aggTrade/{sym}@depth@100ms");
    let deadline = Instant::now() + Duration::from_secs_f64(minutes * 60.0);
    let mut session = Session::new();
    let mut backoff = Duration::from_secs(1);

    'outer: while Instant::now() < deadline {
        let ws = async {
            let stream = crate::net::tls_stream("data-stream.binance.vision").await?;
            let (ws, _) =
                tokio_tungstenite::client_async(&url, stream).await.map_err(|e| {
                    Box::new(e) as Box<dyn Error>
                })?;
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
