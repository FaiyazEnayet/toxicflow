use std::error::Error;

use clap::{Parser, Subcommand};
use futures_util::StreamExt;
use tokio_tungstenite::tungstenite::Message;

use toxicflow::metrics::{
    kyle_lambda, ofi_bars, ofi_events, vpin, volume_buckets, FlowBar, Quote, Trade,
};
use toxicflow::orderbook::{OrderBook, SyncResult};
use toxicflow::{predict, recorder, storage, strategy};

#[derive(Parser)]
#[command(name = "toxicflow", about = "Order-flow toxicity on free crypto data")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Record a live session from Binance market data to parquet.
    Record {
        #[arg(long, default_value = "BTCUSDT")]
        symbol: String,
        #[arg(long, default_value_t = 15.0)]
        minutes: f64,
        #[arg(long, default_value = "data/session")]
        out: String,
    },
    /// Analyze a recorded session: toxicity metrics, prediction, MM demo.
    Analyze {
        dir: String,
        #[arg(long, default_value_t = 50)]
        buckets: usize,
        #[arg(long, default_value_t = 20)]
        window: usize,
        /// Absolute bucket volume in base units (overrides --buckets).
        /// The literature calibrates V to ~1/50 of daily volume; on short
        /// sessions the default 50-bucket split makes small, noisy buckets.
        #[arg(long)]
        bucket_btc: Option<f64>,
    },
    /// Terminal dashboard: rolling VPIN and book imbalance, live.
    Live {
        #[arg(long, default_value = "BTCUSDT")]
        symbol: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Record { symbol, minutes, out } => {
            println!("recording {symbol} for {minutes} min -> {out}");
            let meta = recorder::record_session(&symbol, minutes, &out).await?;
            println!(
                "done: {} trades, {} quotes, {} resyncs",
                meta.n_trades, meta.n_quotes, meta.resyncs
            );
        }
        Cmd::Analyze {
            dir,
            buckets,
            window,
            bucket_btc,
        } => analyze(&dir, buckets, window, bucket_btc)?,
        Cmd::Live { symbol } => live_dashboard(&symbol).await?,
    }
    Ok(())
}

fn percentile(mut xs: Vec<f64>, p: f64) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[(p * (xs.len() - 1) as f64).round() as usize]
}

fn analyze(
    dir: &str,
    n_buckets: usize,
    window: usize,
    bucket_btc: Option<f64>,
) -> Result<(), Box<dyn Error>> {
    let trades = storage::read_trades(&format!("{dir}/trades.parquet"))?;
    let quotes = storage::read_quotes(&format!("{dir}/quotes.parquet"))?;
    println!("session: {} trades, {} quotes", trades.len(), quotes.len());
    if trades.is_empty() || quotes.is_empty() {
        return Ok(());
    }
    let secs =
        (trades.last().unwrap().time_ms - trades.first().unwrap().time_ms) as f64 / 1000.0;
    let vol: f64 = trades.iter().map(|t| t.qty).sum();
    println!("span: {secs:.1}s, volume: {vol:.3} base units");

    // VPIN
    let bk = match bucket_btc {
        Some(v) => {
            println!("\nVPIN bucket volume: {v} base units (explicit)");
            toxicflow::metrics::volume_buckets_sized(&trades, v)
        }
        None => {
            println!("\nVPIN: {n_buckets} equal-volume buckets");
            volume_buckets(&trades, n_buckets)
        }
    };
    let vp = vpin(&bk, window);
    let vals: Vec<f64> = vp.iter().map(|(_, v)| *v).collect();
    println!(
        "VPIN ({} buckets, window {window}): mean {:.3}, p90 {:.3}, max {:.3}, last {:.3}",
        bk.len(),
        vals.iter().sum::<f64>() / vals.len() as f64,
        percentile(vals.clone(), 0.9),
        vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        vals.last().copied().unwrap_or(f64::NAN),
    );

    // Kyle's lambda on 10s bars
    let bars = flow_bars(&trades, &quotes, 10_000);
    let lam = kyle_lambda(&bars, 60);
    let lams: Vec<f64> = lam.iter().filter_map(|x| x.map(|(l, _)| l)).collect();
    if !lams.is_empty() {
        println!(
            "Kyle lambda (10s bars, 60-bar window): mean {:.6} price/vol, last {:.6}",
            lams.iter().sum::<f64>() / lams.len() as f64,
            lams.last().copied().unwrap_or(f64::NAN),
        );
    }

    // OFI
    let ev = ofi_events(&quotes);
    let ob = ofi_bars(&quotes, &ev, 10_000);
    let ofis: Vec<f64> = ob.iter().map(|b| b.1).collect();
    let m = ofis.iter().sum::<f64>() / ofis.len() as f64;
    let sd = (ofis.iter().map(|x| (x - m).powi(2)).sum::<f64>() / ofis.len() as f64).sqrt();
    println!("OFI per 10s bar: mean {m:+.4}, std {sd:.4}");

    // prediction
    println!("\nshort-horizon prediction (10s bars, 30s horizon):");
    let ds = predict::build_dataset(&quotes, &trades, 10_000, 3);
    match predict::walk_forward(&ds, 4, 3) {
        Some(rep) => {
            println!(
                "  accuracy {:.3} +- {:.3} vs baseline {:.3} ({} bars, {} folds)",
                rep.accuracy, rep.accuracy_std, rep.baseline, rep.n_bars, rep.n_folds
            );
            println!("  brier {:.4}", rep.brier);
            println!("  calibration (pred -> realized):");
            for (mp, rl, n) in &rep.calibration {
                println!("    {mp:.3} -> {rl:.3}  (n={n})");
            }
        }
        None => println!("  not enough bars for evaluation"),
    }

    // toxicity-gated MM demo
    println!("\nmarket-making demo (toy fill model, NOT a strategy):");
    let cfg = strategy::MmConfig::default();
    match strategy::run_mm(&quotes, &trades, &cfg) {
        Some(r) => println!(
            "  pnl open {:+.4}, pnl gated {:+.4} | gated {}/{}s ({:.0}%) | turnover {:.1}",
            r.pnl_open,
            r.pnl_gated,
            r.seconds_gated,
            r.seconds,
            100.0 * r.seconds_gated as f64 / r.seconds as f64,
            r.turnover,
        ),
        None => println!("  insufficient data"),
    }
    Ok(())
}

/// Per-bar signed flow and midprice change.
fn flow_bars(trades: &[Trade], quotes: &[Quote], bar_ms: i64) -> Vec<FlowBar> {
    if trades.is_empty() || quotes.is_empty() {
        return Vec::new();
    }
    let t0 = quotes[0].time_ms / bar_ms * bar_ms;
    let n = ((quotes.last().unwrap().time_ms - t0) / bar_ms + 1).max(1) as usize;
    let mut signed = vec![0.0; n];
    let mut mid = vec![f64::NAN; n];
    for t in trades {
        let i = ((t.time_ms - t0) / bar_ms) as usize;
        if i < n {
            signed[i] += t.signed_qty();
        }
    }
    for q in quotes {
        let i = ((q.time_ms - t0) / bar_ms) as usize;
        if i < n {
            mid[i] = q.mid();
        }
    }
    let mut last = f64::NAN;
    let mut out = Vec::new();
    for i in 0..n {
        if mid[i].is_nan() {
            mid[i] = last;
        } else {
            last = mid[i];
        }
        if i > 0 && !mid[i].is_nan() && !mid[i - 1].is_nan() {
            out.push(FlowBar {
                time_ms: t0 + i as i64 * bar_ms,
                signed_vol: signed[i],
                dmid: mid[i] - mid[i - 1],
            });
        }
    }
    out
}

fn parse_levels(v: &serde_json::Value) -> Vec<(f64, f64)> {
    v.as_array()
        .map(|a| {
            a.iter()
                .map(|x| {
                    (
                        x[0].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                        x[1].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Live terminal dashboard: rolling VPIN, touch imbalance, midprice.
/// Ctrl-C to stop.
async fn live_dashboard(symbol: &str) -> Result<(), Box<dyn Error>> {
    let sym = symbol.to_lowercase();
    let url = format!(
        "wss://data-stream.binance.vision/stream?streams={sym}@aggTrade/{sym}@depth@100ms"
    );
    // prime the book with a snapshot first
    let mut book = OrderBook::new();
    let mut synced = false;
    let mut buffer: Vec<serde_json::Value> = Vec::new();

    async fn snapshot(symbol: &str) -> Result<(Vec<(f64, f64)>, Vec<(f64, f64)>, i64), Box<dyn Error>> {
        let url = format!(
            "https://data-api.binance.vision/api/v3/depth?symbol={}&limit=1000",
            symbol.to_uppercase()
        );
        let v: serde_json::Value = reqwest::get(&url).await?.json().await?;
        let pl = |x: &serde_json::Value| -> Vec<(f64, f64)> {
            x.as_array()
                .map(|a| {
                    a.iter()
                        .map(|y| {
                            (
                                y[0].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                                y[1].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                            )
                        })
                        .collect()
                })
                .unwrap_or_default()
        };
        Ok((pl(&v["bids"]), pl(&v["asks"]), v["lastUpdateId"].as_i64().unwrap_or(0)))
    }

    let stream = toxicflow::net::tls_stream("data-stream.binance.vision").await?;
    let (ws, _) = tokio_tungstenite::client_async(&url, stream).await?;
    let (_, mut read) = ws.split();
    let mut trades: Vec<Trade> = Vec::new();
    let mut tick = tokio::time::interval(std::time::Duration::from_secs(2));
    println!("live {symbol}: rolling VPIN (50 buckets / 20 window) + touch imbalance");
    loop {
        tokio::select! {
            _ = tick.tick() => {
                let bk = volume_buckets(&trades, 50);
                let vp = vpin(&bk, 20);
                let v = vp.last().map(|(_, x)| *x).unwrap_or(f64::NAN);
                let (imb, mid) = match book.touch() {
                    Some((bp, bq, ap, aq)) => {
                        let t = bq + aq;
                        ((bq - aq) / t.max(1e-12), (bp + ap) / 2.0)
                    }
                    None => (f64::NAN, f64::NAN),
                };
                let spread = book.spread().unwrap_or(f64::NAN);
                println!(
                    "mid {mid:>12.2}  spread {spread:>6.2}  vpin {v:.3}  imb {imb:+.3}  trades {}",
                    trades.len()
                );
                trades.clear();
            }
            msg = read.next() => {
                let msg = match msg {
                    Some(Ok(m)) => m,
                    _ => break,
                };
                let Message::Text(txt) = msg else { continue };
                let v: serde_json::Value = match serde_json::from_str(&txt) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let d = &v["data"];
                match d["e"].as_str() {
                    Some("aggTrade") => trades.push(Trade {
                        time_ms: d["T"].as_i64().unwrap_or(0),
                        price: d["p"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                        qty: d["q"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                        is_buyer_maker: d["m"].as_bool().unwrap_or(false),
                    }),
                    Some("depthUpdate") => {
                        if !synced {
                            buffer.push(d.clone());
                            if buffer.len() == 1 {
                                // first event: fetch snapshot and merge
                                match snapshot(symbol).await {
                                    Ok((b, a, id)) => {
                                        let first_u = buffer.first().and_then(|e| e["U"].as_i64());
                                        if first_u.map(|fu| id >= fu).unwrap_or(true) {
                                            book.from_snapshot(&b, &a, id);
                                            let mut ok = true;
                                            for e in buffer.drain(..) {
                                                let u = e["u"].as_i64().unwrap_or(0);
                                                if u <= id { continue; }
                                                let uu = e["U"].as_i64().unwrap_or(0);
                                                if book.apply_diff(uu, u, &parse_levels(&e["b"]), &parse_levels(&e["a"])) == SyncResult::Gap {
                                                    ok = false; break;
                                                }
                                            }
                                            synced = ok;
                                        }
                                    }
                                    Err(_) => { buffer.clear(); }
                                }
                            }
                        } else {
                            let uu = d["U"].as_i64().unwrap_or(0);
                            let u = d["u"].as_i64().unwrap_or(0);
                            match book.apply_diff(uu, u, &parse_levels(&d["b"]), &parse_levels(&d["a"])) {
                                SyncResult::Gap => { synced = false; buffer = vec![d.clone()]; }
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(())
}
