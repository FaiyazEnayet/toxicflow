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
    /// Export analyzer intermediates (VPIN, Kyle lambda, OFI, calibration,
    /// summary) as CSV + JSON for the scripts/dashboard.py dashboard.
    Export {
        dir: String,
        #[arg(long, default_value = "runs/dashboard_data")]
        out: String,
        #[arg(long, default_value_t = 50)]
        buckets: usize,
        #[arg(long, default_value_t = 20)]
        window: usize,
        /// Absolute bucket volume in base units (overrides --buckets).
        #[arg(long)]
        bucket_btc: Option<f64>,
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
        Cmd::Export {
            dir,
            out,
            buckets,
            window,
            bucket_btc,
        } => export_session(&dir, &out, buckets, window, bucket_btc)?,
        Cmd::Live { symbol } => live_dashboard(&symbol).await?,
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use toxicflow::metrics::{Quote, Trade};

    /// 15 minutes of synthetic quotes/trades: enough for all export series.
    fn synth_session() -> (Vec<Quote>, Vec<Trade>) {
        let mut quotes = Vec::new();
        let mut trades = Vec::new();
        let mut mid = 100.0;
        let mut s: u64 = 99;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s as f64) / (u64::MAX as f64) - 0.5
        };
        for i in 0..900 {
            let t = i as i64 * 1000;
            quotes.push(Quote {
                time_ms: t,
                bid_px: mid - 0.05,
                bid_sz: 5.0 + rnd().abs(),
                ask_px: mid + 0.05,
                ask_sz: 5.0 + rnd().abs(),
            });
            for k in 0..4 {
                trades.push(Trade {
                    time_ms: t + k * 200,
                    price: mid,
                    qty: 0.05,
                    is_buyer_maker: rnd() > 0.0,
                });
            }
            mid += rnd() * 0.02;
        }
        (quotes, trades)
    }

    #[test]
    fn export_writes_all_files() {
        let (quotes, trades) = synth_session();
        let tmp = std::env::temp_dir().join(format!("toxicflow_export_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        storage::write_trades(tmp.join("trades.parquet").to_str().unwrap(), &trades).unwrap();
        storage::write_quotes(tmp.join("quotes.parquet").to_str().unwrap(), &quotes).unwrap();
        let out = tmp.join("out");
        export_session(
            tmp.to_str().unwrap(),
            out.to_str().unwrap(),
            50,
            20,
            None,
        )
        .expect("export failed");

        let vpin = std::fs::read_to_string(out.join("vpin.csv")).unwrap();
        assert!(vpin.starts_with("time_ms,vpin\n"));
        assert!(vpin.lines().count() > 2, "vpin.csv has no data rows");

        let kyle = std::fs::read_to_string(out.join("kyle.csv")).unwrap();
        assert!(kyle.starts_with("time_ms,lambda\n"));

        let ofi = std::fs::read_to_string(out.join("ofi.csv")).unwrap();
        assert!(ofi.starts_with("time_ms,ofi,mid,spread,imbalance,signed_vol,fwd_ret_bps\n"));
        assert!(ofi.lines().count() > 40, "ofi.csv too short");

        let cal = std::fs::read_to_string(out.join("calibration.csv")).unwrap();
        assert!(cal.starts_with("mean_pred,realized,n\n"));

        let summary: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(out.join("summary.json")).unwrap())
                .unwrap();
        assert_eq!(summary["n_trades"], trades.len() as u64);
        assert_eq!(summary["n_quotes"], quotes.len() as u64);
        assert!(summary["vpin"]["mean"].as_f64().unwrap() > 0.0);
        assert!(summary["predict"]["accuracy"].as_f64().is_some());
        assert!(summary["mm"]["pnl_open"].as_f64().is_some());
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn export_rejects_empty_session() {
        let tmp = std::env::temp_dir().join(format!("toxicflow_export_empty_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        storage::write_trades(tmp.join("trades.parquet").to_str().unwrap(), &[]).unwrap();
        storage::write_quotes(tmp.join("quotes.parquet").to_str().unwrap(), &[]).unwrap();
        let r = export_session(tmp.to_str().unwrap(), tmp.join("out").to_str().unwrap(), 50, 20, None);
        assert!(r.is_err(), "empty session should fail");
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn read_trades_rejects_wrong_schema_without_panicking() {
        use arrow::array::{Float64Array, Int64Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;

        // Parquet file missing the "price" column: must be an Err, not a panic.
        let schema = Arc::new(Schema::new(vec![
            Field::new("time", DataType::Int64, false),
            Field::new("qty", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![1i64, 2])),
                Arc::new(Float64Array::from(vec![0.5f64, 0.5])),
            ],
        )
        .unwrap();
        let tmp =
            std::env::temp_dir().join(format!("toxicflow_bad_schema_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let path = tmp.join("trades.parquet");
        let file = std::fs::File::create(&path).unwrap();
        let mut w = ArrowWriter::try_new(file, schema, None).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();

        let r = storage::read_trades(path.to_str().unwrap());
        assert!(r.is_err(), "wrong-schema parquet should be an Err");
        std::fs::remove_dir_all(&tmp).unwrap();
    }
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

fn write_csv(path: &str, header: &str, rows: &[String]) -> Result<(), Box<dyn Error>> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "{header}")?;
    for r in rows {
        writeln!(f, "{r}")?;
    }
    Ok(())
}

/// Export analyzer intermediates as CSV + JSON for the dashboard script.
/// Reuses the exact same metric functions as `analyze`; it only changes
/// the output format, not the computation.
fn export_session(
    dir: &str,
    out: &str,
    n_buckets: usize,
    window: usize,
    bucket_btc: Option<f64>,
) -> Result<(), Box<dyn Error>> {
    let trades = storage::read_trades(&format!("{dir}/trades.parquet"))?;
    let quotes = storage::read_quotes(&format!("{dir}/quotes.parquet"))?;
    if trades.is_empty() || quotes.is_empty() {
        return Err("empty session".into());
    }
    std::fs::create_dir_all(out)?;
    let secs =
        (trades.last().unwrap().time_ms - trades.first().unwrap().time_ms) as f64 / 1000.0;
    let vol: f64 = trades.iter().map(|t| t.qty).sum();
    let buy_vol: f64 = trades.iter().filter(|t| !t.is_buyer_maker).map(|t| t.qty).sum();
    let lo = trades.iter().map(|t| t.price).fold(f64::INFINITY, f64::min);
    let hi = trades.iter().map(|t| t.price).fold(f64::NEG_INFINITY, f64::max);

    // VPIN series
    let bk = match bucket_btc {
        Some(v) => toxicflow::metrics::volume_buckets_sized(&trades, v),
        None => volume_buckets(&trades, n_buckets),
    };
    let vp = vpin(&bk, window);
    write_csv(
        &format!("{out}/vpin.csv"),
        "time_ms,vpin",
        &vp.iter().map(|(t, v)| format!("{t},{v:.6}")).collect::<Vec<_>>(),
    )?;

    // Kyle's lambda series on 10s bars
    let bars = flow_bars(&trades, &quotes, 10_000);
    let lam = kyle_lambda(&bars, 60);
    write_csv(
        &format!("{out}/kyle.csv"),
        "time_ms,lambda",
        &bars
            .iter()
            .zip(lam.iter())
            .filter_map(|(b, l)| l.map(|(v, _)| format!("{},{}", b.time_ms, v)))
            .collect::<Vec<_>>(),
    )?;

    // OFI bars with forward 30s mid returns (for the OFI-vs-return plot)
    let ev = ofi_events(&quotes);
    let ob = ofi_bars(&quotes, &ev, 10_000);
    let t0 = ob[0].0;
    let mut signed = vec![0.0; ob.len()];
    for t in &trades {
        let idx = ((t.time_ms - t0) / 10_000) as usize;
        if idx < signed.len() {
            signed[idx] += t.signed_qty();
        }
    }
    write_csv(
        &format!("{out}/ofi.csv"),
        "time_ms,ofi,mid,spread,imbalance,signed_vol,fwd_ret_bps",
        &ob.iter()
            .enumerate()
            .map(|(i, b)| {
                let fwd = if i + 3 < ob.len() {
                    (ob[i + 3].2 / b.2 - 1.0) * 1e4
                } else {
                    f64::NAN
                };
                format!(
                    "{},{:.4},{:.2},{:.4},{:+.4},{:.4},{}",
                    b.0, b.1, b.2, b.3, b.4, signed[i], fwd
                )
            })
            .collect::<Vec<_>>(),
    )?;

    // prediction calibration
    let ds = predict::build_dataset(&quotes, &trades, 10_000, 3);
    let rep = predict::walk_forward(&ds, 4, 3);
    write_csv(
        &format!("{out}/calibration.csv"),
        "mean_pred,realized,n",
        &rep
            .as_ref()
            .map(|r| {
                r.calibration
                    .iter()
                    .map(|(mp, rl, n)| format!("{mp:.4},{rl:.4},{n}"))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default(),
    )?;

    // MM demo summary
    let cfg = strategy::MmConfig::default();
    let mm = strategy::run_mm(&quotes, &trades, &cfg);

    let vp_vals: Vec<f64> = vp.iter().map(|(_, v)| *v).collect();
    let vp_mean = vp_vals.iter().sum::<f64>() / vp_vals.len() as f64;
    let lams: Vec<f64> = lam.iter().filter_map(|x| x.map(|(l, _)| l)).collect();
    let ofis: Vec<f64> = ob.iter().map(|b| b.1).collect();
    let ofi_m = ofis.iter().sum::<f64>() / ofis.len() as f64;
    let ofi_sd = (ofis.iter().map(|x| (x - ofi_m).powi(2)).sum::<f64>() / ofis.len() as f64).sqrt();
    let summary = serde_json::json!({
        "n_trades": trades.len(),
        "n_quotes": quotes.len(),
        "span_s": secs,
        "volume_btc": vol,
        "buy_share": buy_vol / vol,
        "price_lo": lo,
        "price_hi": hi,
        "vpin": {
            "n_buckets": bk.len(), "window": window,
            "mean": vp_mean,
            "p90": percentile(vp_vals.clone(), 0.9),
            "max": vp_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            "last": vp_vals.last().copied().unwrap_or(f64::NAN),
            "gate_threshold": cfg.vpin_threshold,
        },
        "kyle_lambda": {
            "mean": lams.iter().sum::<f64>() / lams.len().max(1) as f64,
            "last": lams.last().copied().unwrap_or(f64::NAN),
        },
        "ofi": { "mean": ofi_m, "std": ofi_sd },
        "predict": rep.as_ref().map(|r| serde_json::json!({
            "feature_version": r.feature_version,
            "accuracy": r.accuracy, "accuracy_std": r.accuracy_std,
            "baseline": r.baseline, "n_bars": r.n_bars,
            "n_folds": r.n_folds, "brier": r.brier,
        })).unwrap_or(serde_json::Value::Null),
        "mm": mm.as_ref().map(|r| serde_json::json!({
            "pnl_open": r.pnl_open, "pnl_gated": r.pnl_gated,
            "seconds": r.seconds, "seconds_gated": r.seconds_gated,
            "turnover": r.turnover, "inventory_end": r.inventory_end,
        })).unwrap_or(serde_json::Value::Null),
    });
    std::fs::write(
        format!("{out}/summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    println!("exported session -> {out} (vpin.csv, kyle.csv, ofi.csv, calibration.csv, summary.json)");
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
