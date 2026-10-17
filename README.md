<p align="center"><img src="docs/readme-banner.svg" alt="Toxicflow: Order-flow measurement and execution-risk research" width="1200"></p>

[![Tests](https://github.com/FaiyazEnayet/toxicflow/actions/workflows/ci.yml/badge.svg)](https://github.com/FaiyazEnayet/toxicflow/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-007EC6)](LICENSE) ![Rust 2021 edition](https://img.shields.io/badge/Rust-2021%20edition-A65E2E) ![Built with Tokio + Parquet](https://img.shields.io/badge/built%20with-Tokio%20%2B%20Parquet-555555) [![Interactive demo](https://img.shields.io/badge/demo-interactive-4C9D39)](https://faiyazenayet.github.io/toxicflow/docs/dashboard.html)

<p align="center"><a href="https://faiyazenayet.github.io/toxicflow/docs/dashboard.html"><b>Open interactive demo</b></a> · <a href="docs/dashboard.html">HTML file</a> · <a href="#method">Method</a> · <a href="#run-locally">Run locally</a> · <a href="#limitations">Limitations</a></p>

# Toxicflow

Toxicflow records exchange order flow, reconstructs market state and tests whether liquidity signals carry useful short-horizon information. The dashboard keeps weak results visible rather than turning diagnostics into trading claims.

**Rust · Tokio · Parquet · Python visualization**

[![Toxicflow demo](docs/demo.gif)](https://faiyazenayet.github.io/toxicflow/docs/dashboard.html)


## Explore

- Inspect market price and VPIN with aligned timestamps and cross-chart readouts.
- Compare flow, impact and calibration without hiding the majority-class baseline.
- Review market-making outcomes under the same recorded session.

## Method

A synchronized local order book feeds VPIN, rolling Kyle lambda and order-flow imbalance. Return-direction models use purged walk-forward evaluation. A deliberately simplified market-making simulation illustrates the limits of toxicity gating.

## Evidence at a glance

| Scope | Recorded result or contract |
| :--- | :--- |
| Recorded session | 28,224 trades; 106,735 quotes; about 3.25 hours |
| Direction-model accuracy | 0.776 versus a 0.809 majority-class baseline |
| Toy market-making PnL | Open -9.87; toxicity-gated -10.15 |
| Conclusion | No demonstrated predictive or trading edge in this session |

## Run locally


```sh
cargo build --release
cargo test
./target/release/toxicflow export data/new --out runs/dashboard_data
python scripts/dashboard.py --data runs/dashboard_data --out docs/dashboard.html
```

## Technical design

The Rust recorder maintains exchange sequence discipline and writes quote/trade sessions to Parquet. VPIN uses equal-volume buckets; Kyle lambda estimates price impact from signed flow; OFI aggregates changes at the best bid and ask. The analysis and export commands share the metric functions used by the dashboard.

```mermaid
flowchart LR
  N0["Exchange stream"] --> N1["Sequenced local order book"]
  N1["Sequenced local order book"] --> N2["Quote and trade bars"]
  N2["Quote and trade bars"] --> N3["VPIN lambda and OFI"]
  N3["VPIN lambda and OFI"] --> N4["Purged evaluation"]
  N3["VPIN lambda and OFI"] --> N5["Execution simulation"]
```

## Evaluation protocol

The predictor uses a purged walk-forward protocol, so future labels do not overlap the training window. Accuracy is compared with the majority-class rule and supplemented with calibration. The market-making example is intentionally simplified and cannot be interpreted as a deployable execution strategy.

![The predictor does not beat the majority baseline](docs/results.png)

The recorded session contains no demonstrated predictive edge over the baseline. Toxicity gating also does not improve the toy PnL. Those results make the evaluation and diagnostic tooling the contribution, rather than an unsupported trading-profit claim.

## Code and data

`src/orderbook.rs` handles book synchronization. `metrics.rs` contains the estimators. `predict.rs` handles walk-forward evaluation. `strategy.rs` is the toy simulation. Dashboard inputs are exported session files, not an invented dataset.


## Limitations

- The session is too short to support broad microstructure conclusions.
- VPIN depends on bucket-volume calibration and does not identify informed traders by itself.
- The fill model excludes queue position, latency and fees. Missing-quote gap handling also needs a regression test.

## License

[MIT](LICENSE). Third-party datasets, libraries and fonts retain their own terms.
