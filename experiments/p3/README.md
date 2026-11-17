# P3: Test conditional toxicity and incremental information (pilot)

## Protocol

The VPIN gate in the market-making demo assumes one static threshold
captures useful risk. P3 tests whether VPIN adds incremental
information once basic volatility, spread and trade intensity are
controlled, and whether it helps only in certain activity or depth
regimes.

The v5 frame adds a two-column VPIN block to the v4 frame (17
features): VPIN at the bar close and its 3-bar lag. Bucket volume
is a fixed 15.0 BTC, never derived from a session total, with a
20-bucket window; the VPIN at a bar close is the last completed
bucket value at or before the bar close, and a row is scored only
once a full 20-bucket window has completed and the lagged value 3
bars earlier is defined. Rows start at bar 29 so every trailing
30-bar window is full, and share the P2 gap censoring. The first
five v5 columns reproduce `build_dataset` bit-identically on shared
rows, so the v3 baseline is preserved exactly inside the new frame.

Paired models on the same folds and observations, on the six frozen
2026-11-11/12 test sessions (the same set as E5, P1, N5 and P2):

* controls-only: the full v4 frame (book state, signed flow,
  OFI/depth, event rates, realized volatility, microprice
  displacement).
* controls+VPIN: controls plus the VPIN block (the default cell).
* VPIN-only: the VPIN block alone.
* Reference rungs: training-majority predictor, the v3 frame, and
  the v3 frame plus VPIN.
* Negative controls: permutation and 5-bar shift of the VPIN block,
  with the fitted model unchanged.

Four binary outcomes, all on a 3-bar horizon with 10 s bars and 4
walk-forward splits, scored by log loss and Brier score: direction
of the forward mid return (comparable with P1/P2); future absolute
return above the training-fold median (train-only threshold);
spread widening at the horizon; conditional adverse markout, defined
against the bar's net aggressive flow side, with zero-flow rows
excluded from that outcome only. On these six sessions the
exclusion removed no rows: every scored bar carries nonzero net
flow, so the adverse-markout row set equals the other outcomes.

Regimes are activity (trade rate) and depth (touch depth) tertiles
learned on training rows only, per fold. Every regime cell reports
its sample size with the paired log-loss and Brier differential
for controls+VPIN against controls-only. Dependence accounting is
a session-block bootstrap over the six sessions (10,000 resamples,
fixed seed) on the paired log-loss differential.

The design grid is frozen in `grid.json` before any grid run (8
cells; 10 s bars; 3-bar horizon; 4 walk-forward splits). The runner
(`examples/p3_pilot.rs`) refuses a grid file that does not match
the library grid exactly, and refuses sessions not listed in the
grid file under `test_sessions`. No test outcome is used for
parameter selection: the default cell (controls-vpin) stays the
reported cell regardless of results. Every grid cell is reported
for every outcome in `results.json`.

The P3 row set is smaller than the P2 row set: a row also needs a
full 20-bucket VPIN window completed before its bar close, so
early bars are outside the set for every config. P3 headline
numbers are therefore not directly comparable with P2 headline
numbers; the paired comparisons within P3 are exact.

Absent by design, recorded here rather than silently omitted:
cross-session confirmation (blocked; P3::confirm is a separate
blocked-input unit, so this pilot reports exploratory observations
on the archived sessions only), depth beyond the touch
(unavailable in archived data), and deeper-book or queue imbalance
(not recorded in the archive).

## Results (mean over 6 sessions, session-equal weighting)

7,118 decision rows (1,185 / 1,213 / 1,257 / 1,079 / 1,139 / 1,245
per session); 5,684 held-out predictions per outcome.

### Direction

| config | accuracy | train-majority baseline | Brier | log loss |
|---|---|---|---|---|
| B0 majority | 0.516 | 0.516 | 0.251 | 0.695 |
| B3 v3 frame | 0.551 | 0.516 | 0.248 | 0.690 |
| B5 controls-only | 0.543 | 0.516 | 0.259 | 0.721 |
| B3 plus VPIN | 0.535 | 0.516 | 0.261 | 0.723 |
| VPIN-only | 0.516 | 0.516 | 0.264 | 0.727 |
| Controls plus VPIN | 0.529 | 0.516 | 0.271 | 0.755 |
| Controls plus VPIN, permuted | 0.528 | 0.516 | 0.268 | 0.748 |
| Controls plus VPIN, shifted 5 bars | 0.527 | 0.515 | 0.270 | 0.752 |

### Absolute return

| config | accuracy | train-majority baseline | Brier | log loss |
|---|---|---|---|---|
| B0 majority | 0.527 | 0.470 | 0.250 | 0.693 |
| B3 v3 frame | 0.520 | 0.470 | 0.254 | 0.704 |
| B5 controls-only | 0.574 | 0.470 | 0.246 | 0.691 |
| B3 plus VPIN | 0.530 | 0.470 | 0.257 | 0.712 |
| VPIN-only | 0.540 | 0.470 | 0.253 | 0.702 |
| Controls plus VPIN | 0.568 | 0.470 | 0.252 | 0.709 |
| Controls plus VPIN, permuted | 0.564 | 0.470 | 0.253 | 0.711 |
| Controls plus VPIN, shifted 5 bars | 0.567 | 0.469 | 0.253 | 0.712 |

### Spread widening

| config | accuracy | train-majority baseline | Brier | log loss |
|---|---|---|---|---|
| B0 majority | 0.823 | 0.823 | 0.146 | 0.468 |
| B3 v3 frame | 0.821 | 0.823 | 0.137 | 0.418 |
| B5 controls-only | 0.806 | 0.823 | 0.143 | 0.438 |
| B3 plus VPIN | 0.815 | 0.823 | 0.142 | 0.432 |
| VPIN-only | 0.823 | 0.823 | 0.150 | 0.480 |
| Controls plus VPIN | 0.793 | 0.823 | 0.151 | 0.457 |
| Controls plus VPIN, permuted | 0.786 | 0.823 | 0.152 | 0.460 |
| Controls plus VPIN, shifted 5 bars | 0.790 | 0.822 | 0.151 | 0.457 |

### Adverse markout

| config | accuracy | train-majority baseline | Brier | log loss |
|---|---|---|---|---|
| B0 majority | 0.548 | 0.548 | 0.247 | 0.687 |
| B3 v3 frame | 0.540 | 0.548 | 0.249 | 0.693 |
| B5 controls-only | 0.538 | 0.548 | 0.255 | 0.706 |
| B3 plus VPIN | 0.542 | 0.548 | 0.251 | 0.698 |
| VPIN-only | 0.544 | 0.548 | 0.249 | 0.691 |
| Controls plus VPIN | 0.542 | 0.548 | 0.256 | 0.714 |
| Controls plus VPIN, permuted | 0.538 | 0.548 | 0.257 | 0.715 |
| Controls plus VPIN, shifted 5 bars | 0.539 | 0.548 | 0.257 | 0.715 |

VPIN adds no stable contribution on any of the four outcomes once
the controls are in. Adding the VPIN block to the controls makes
log loss worse on every outcome (direction +0.034, absolute return
+0.019, spread widening +0.020, adverse markout +0.008), and the
session-block bootstrap intervals exclude zero on all four
(direction [0.008, 0.060], absolute return [0.005, 0.033], spread
widening [0.011, 0.029], adverse markout [0.002, 0.014]). The
per-session direction differentials are positive in 5 of 6
sessions. No regime cell shows a VPIN gain either: in all 24
activity and depth tertile cells the paired log-loss differential
is positive (from +0.002 to +0.052), with the smallest cells
holding as few as 26 rows in a session.

The negative controls confirm the reading: the clean
controls+VPIN model performs no better than its own permutation
or shift controls on any outcome (direction log loss 0.755 clean
against 0.748 permuted and 0.752 shifted; the same pattern holds
on the other three outcomes), so the fitted weight on VPIN carries
no incremental alignment with the targets. VPIN alone sits at the
majority baseline on direction accuracy (0.516 against 0.516)
and below it on adverse markout, and only the absolute-return
outcome shows VPIN alone above baseline (0.540 against 0.470),
without surviving as an increment once the controls are in. The
v3 frame stays the model in use, and the static VPIN gate keeps
no empirical support from this pilot.

Per-session numbers for every cell and outcome, with per-fold
scores, the default cell's fold weights, regime tables, and the
bootstrap intervals, are in `results.json`.

## Reproduce

```
cargo run --locked --release --example p3_pilot -- experiments/p3/grid.json experiments/p3/results.json <session_dir>...
```

Session directories are the closing-archive BTC sessions under
`toxicflow-recordings/sessions/`, each holding `quotes.parquet` and
`trades.parquet`.
