# toxicflow

**Order-flow toxicity on free crypto data.** This informs us whether the microstructure ideas from the quant literature actually show up in markets we can touch. This full loop in Rust is built for that reason: stream Binance market data, maintain a local limit order book, compute toxicity metrics, and test whether order flow predicts short horizon returns.

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

I recorded 12 minutes of BTCUSDT: 2132 trades, 7121 quotes, 77.9 BTC of volume, price flat at 83036. Real analyzer output:

```
session: 2132 trades, 7121 quotes
span: 718.1s, volume: 77.867 base units

VPIN: 50 equal-volume buckets
VPIN (50 buckets, window 20): mean 0.812, p90 0.986, max 0.986, last 0.900
Kyle lambda (10s bars, 60-bar window): mean 1.901672 price/vol, last 1.291213
OFI per 10s bar: mean -0.1622, std 10.9482

short-horizon prediction (10s bars, 30s horizon):
  accuracy 0.692 +- 0.462 vs baseline 0.731 (66 bars, 4 folds)
  brier 0.2287
  calibration (pred -> realized):
    0.020 -> 0.000  (n=2)
    0.084 -> 0.000  (n=2)
    0.210 -> 0.000  (n=2)
    0.229 -> 0.500  (n=2)
    0.234 -> 1.000  (n=2)
    0.242 -> 1.000  (n=2)
    0.248 -> 0.000  (n=2)
    0.263 -> 0.500  (n=2)
    0.299 -> 0.000  (n=2)
    0.315 -> 0.000  (n=2)
    0.334 -> 0.000  (n=2)
    0.516 -> 0.500  (n=2)
    0.789 -> 0.000  (n=2)

market-making demo (toy fill model, NOT a strategy):
  pnl open +0.2277, pnl gated +0.1038 | gated 502/719s (70%) | turnover 24737.5
```

Read it as: the VPIN level is misleading here, and that is the interesting part. Overall buy share was 0.534 and the price went nowhere, yet VPIN averaged 0.81. The reason is bucket sizing: 50 buckets over 12 minutes gives about 1.6 BTC per bucket, roughly 14 seconds of flow, and order flow is autocorrelated at that scale because parent orders get split into child orders. Rerunning with explicit 5 BTC buckets gives mean VPIN 0.563. Lesson: calibrate the bucket volume to the session and never compare raw VPIN levels to literature values computed on daily buckets. The `--bucket-btc` flag exists for exactly this.

Kyle's lambda of about 1.9 means each BTC of signed flow moved the mid by roughly $1.90 on 10 second bars. Plausible for BTC.

The predictor found no edge: 0.692 accuracy against a 0.731 majority class baseline, on 66 bars. Twelve minutes is not enough data to find anything, and the honest result is the valuable one. The evaluation framework is the deliverable here: purged walk forward splits, no shuffling, calibration tables. It will tell you just as clearly when there is no edge as when there is one.

The market making demo is a toy with a clear lesson. The gate stood down 70% of the time because VPIN sat above the 0.7 threshold almost everywhere (same bucket artifact as above), so the gated leg barely traded and finished behind the always quoting leg. That is not a failure of the idea. It is what an over conservative gate looks like, and the demo makes the mechanism visible: adverse selection avoidance has a cost in missed spread.

## What is in here

- `src/orderbook.rs`: local L2 book with the exchange's snapshot plus diff sync procedure, gap detection, resync.
- `src/recorder.rs`: tokio websocket client, reconnect with backoff, parquet sessions.
- `src/metrics.rs`: VPIN, Kyle's lambda, OFI. Pure functions, unit tested on synthetic flow.
- `src/predict.rs`: hand rolled logistic regression, walk forward evaluation with a purge gap. Time series are never shuffled.
- `src/strategy.rs`: toxicity gated market making demo. Toy fill model, clearly labeled. It exists to show the direction of the effect, not to trade.
- `src/net.rs`: the sandbox I built this in has no direct internet, so websocket connections tunnel through the egress proxy via CONNECT. You will not need this on a normal machine, but it is harmless there.

## Honest limitations

- 12 minutes of BTC is a snapshot, not a study. Microstructure results need days of data across regimes.
- Crypto is not equities. No open or close auction, funding rates, 24/7 retail flow. Take any comparison to the equity literature with salt.
- VPIN itself is debated. It flagged the 2010 flash crash in hindsight and plenty of papers argue it is mostly repackaged volume. My session above is a live demonstration of why calibration matters.
- The market making demo has a toy fill model: no queue position, no latency, no fees. It demonstrates adverse selection, not a strategy.
- This is research code, not trading advice. Nothing here should touch real money.

## References

Easley, Lopez de Prado, O'Hara, "Flow Toxicity and Liquidity in a High Frequency World" (2012). Cont, Kukanov, Stoikov, "The Price Impact of Order Book Events" (2014). Kyle, "Continuous Auctions and Insider Trading" (1985).
