# toxicflow

**Order-flow toxicity on free crypto data.** I wanted to know whether the microstructure ideas from the quant literature actually show up in markets I can touch for $0. So I built the full loop in Rust: stream Binance market data, maintain a local limit order book, compute toxicity metrics, and test whether order flow predicts short horizon returns.

Toxicity is the market maker's problem. When informed traders dominate the flow, anyone quoting passively gets picked off: they buy right before the price falls and sell right before it rises. The metrics here all try to measure that adverse selection risk in real time.

## The three metrics

**VPIN.** Easley, Lopez de Prado, O'Hara. Group trades into equal volume buckets (not time bars, so busy and quiet periods are comparable), measure |buy volume minus sell volume| / volume in each bucket, average over a rolling window. High VPIN means one sided flow: someone knows something. Trade direction comes from the exchange's own aggressor flag, so no tick rule needed.

**Kyle's lambda.** The price impact coefficient: rolling OLS slope of midprice changes on signed order flow. High lambda means each unit of flow moves the price a lot. That is a thin book, and thin books are where market makers bleed.

**OFI.** Cont, Kukanov, Stoikov (2014). Order flow imbalance built from every quote change at the touch: bid size added when the bid ticks up, removed when it ticks down, mirrored on the ask side. Their result: this quantity predicts the next midprice move. I test that claim directly.

## Quickstart

```bash
cargo build --release
./target/release/toxicflow record --symbol BTCUSDT --minutes 15 --out data/btc
./target/release/toxicflow analyze data/btc
./target/release/toxicflow live --symbol BTCUSDT   # terminal dashboard
```

Data comes from Binance's market data only hosts (`data-stream.binance.vision`), which need no API key. Sessions save to parquet: `trades.parquet`, `quotes.parquet`, `meta.json`.

## What a real session showed

I recorded 2.5 hours of BTCUSDT: 24,785 trades, 88,891 quotes, 697.4 BTC of volume, price flat between 82932 and 83107. Real analyzer output:

```
session: 24785 trades, 88891 quotes
span: 9922.3s, volume: 697.369 base units

VPIN: 50 equal-volume buckets
VPIN (50 buckets, window 20): mean 0.486, p90 0.640, max 0.681, last 0.439
Kyle lambda (10s bars, 60-bar window): mean 3.679942 price/vol, last 2.512595
OFI per 10s bar: mean +0.0848, std 11.0367

short-horizon prediction (10s bars, 30s horizon):
  accuracy 0.783 +- 0.412 vs baseline 0.826 (987 bars, 4 folds)
  brier 0.1630
  calibration (pred -> realized):
    0.012 -> 0.103  (n=78)
    0.060 -> 0.038  (n=78)
    0.094 -> 0.038  (n=78)
    0.113 -> 0.038  (n=78)
    0.156 -> 0.103  (n=78)
    0.199 -> 0.077  (n=78)
    0.243 -> 0.192  (n=78)
    0.277 -> 0.346  (n=78)
    0.337 -> 0.487  (n=78)
    0.705 -> 0.333  (n=78)
    1.000 -> 0.000  (n=8)

market-making demo (toy fill model, NOT a strategy):
  pnl open +13.1094, pnl gated +13.1094 | gated 0/9925s (0%) | turnover 854638.2
```

Read it as: the bucket lesson from the earlier 12 minute session is confirmed. With 2.5 hours of data, 50 buckets gives about 14 BTC per bucket, which is well sized, and VPIN behaves: mean 0.486, max 0.681, against a buy share of 0.479 and a price that went nowhere. The metric now agrees with reality. Calibrate the bucket volume to the session and never compare raw VPIN levels to literature values computed on daily buckets. The `--bucket-btc` flag exists for exactly this.

Kyle's lambda of about 3.7 means each BTC of signed flow moved the mid by roughly $3.70 on 10 second bars. Plausible for BTC.

The predictor found no edge: 0.783 accuracy against a 0.826 majority class baseline, on 987 bars. That is 15 times the data of the earlier session and the honest result is the same. The evaluation framework is the deliverable here: purged walk forward splits, no shuffling, calibration tables. It will tell you just as clearly when there is no edge as when there is one.

The market making demo is a toy with the lesson inverted this time. The gate never triggered because VPIN stayed below the 0.7 threshold for the whole session, so the gated leg equals the always quoting leg. That is correct behavior: a calm market with balanced flow should not trip a toxicity gate. Compare with the 12 minute session where an over sensitive gate stood down 70% of the time. The demo makes the mechanism visible in both directions.

## What is in here

- `src/orderbook.rs`: local L2 book with the exchange's snapshot plus diff sync procedure, gap detection, resync.
- `src/recorder.rs`: tokio websocket client, reconnect with backoff, parquet sessions.
- `src/metrics.rs`: VPIN, Kyle's lambda, OFI. Pure functions, unit tested on synthetic flow.
- `src/predict.rs`: hand rolled logistic regression, walk forward evaluation with a purge gap. Time series are never shuffled.
- `src/strategy.rs`: toxicity gated market making demo. Toy fill model, clearly labeled. It exists to show the direction of the effect, not to trade.
- `src/net.rs`: the sandbox I built this in has no direct internet, so websocket connections tunnel through the egress proxy via CONNECT. You will not need this on a normal machine, but it is harmless there.

## Honest limitations

- 2.5 hours of BTC is a session, not a study. Microstructure results need days of data across regimes.
- Crypto is not equities. No open or close auction, funding rates, 24/7 retail flow. Take any comparison to the equity literature with salt.
- VPIN itself is debated. It flagged the 2010 flash crash in hindsight and plenty of papers argue it is mostly repackaged volume. My session above is a live demonstration of why calibration matters.
- The market making demo has a toy fill model: no queue position, no latency, no fees. It demonstrates adverse selection, not a strategy.
- This is research code, not trading advice. Nothing here should touch real money.

## References

Easley, Lopez de Prado, O'Hara, "Flow Toxicity and Liquidity in a High Frequency World" (2012). Cont, Kukanov, Stoikov, "The Price Impact of Order Book Events" (2014). Kyle, "Continuous Auctions and Insider Trading" (1985).
