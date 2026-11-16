# E5: Robustness, negative controls and estimator sensitivity

## Protocol

The parameter grid is preregistered in `grid.json` before any grid run
(15 configurations: the default, one-at-a-time sweeps of bucket count,
VPIN window, bar length, lambda window, and VPIN gate, plus the
buckets-by-window factorial). The runner
(`examples/e5_robustness.rs`) refuses a grid file that does not match
the library grid exactly, and refuses sessions not listed in the grid
file under their role. No test outcome is used for parameter selection:
the default configuration (50 buckets, 20-bucket window, 10 s bars,
60-bar lambda window, 0.7 gate) stays the reported configuration
regardless of results.

Sessions are real recorded BTCUSDT sessions from the November 2026
campaign. Train (calibration only): the four 2026-11-09 sessions.
Test (untouched, chronologically later): the six 2026-11-11 and
2026-11-12 sessions. The VPIN gate is calibrated train-only as the
90th percentile of the pooled train-session VPIN series under the
default configuration. Every grid result carries a manifest with its
configuration id, parameters, configuration hash, session name, and
input hash (SHA256 of the session parquet files). Failed runs are
retained with their reason; this run had none.

## Results (test sessions unless noted)

### VPIN sensitivity

Mean VPIN moves with bucket count (25 buckets 0.214, 50 buckets 0.276,
100 buckets 0.348) and barely with window (10 buckets 0.284, 20
buckets 0.276, 40 buckets 0.270). Smaller buckets hold fewer trades
each, so their imbalances run more extreme.

### Lambda sensitivity and tail

Mean lambda falls as bars lengthen (5 s bars 5.37, 10 s bars 4.63,
30 s bars 3.59) and as the window lengthens (30 bars 5.60, 60 bars
4.63, 120 bars 3.89). The tail is heavy but gradual: the 99th
percentile of absolute lambda runs 2.2x to 4.5x the median (mean
3.1x) across the ten sessions, yet no window in any session exceeds
the preregistered tail rule (ten robust standard deviations from the
median lambda), so the tail share is 0.0% everywhere and the
winsorized mean sits within 0.04 of the raw mean in every session.
The per-window valid counts and tail flags are in the export
(`kyle.csv` columns `n_valid` and `tail_flag`, `summary.json` block
`kyle_lambda`).

### VPIN gate and market-making demo

The 0.7 gate behaves differently in the demo than the session VPIN
mean suggests: the demo freezes bucket volume from a 300 s
calibration window, so its buckets are far smaller than the
whole-session split and its VPIN runs higher. The default gate stands
down for 65.4% of scored test seconds on average (26% to 93% per
session). Gated minus open PnL at the default is +5.2, +159.1, +39.3,
-31.9, +123.5, -737.2 across the six test sessions: the gate helps in
four sessions and hurts in two, and the mean (-73.7) is dominated by
the last session, where standing down misses a large open profit.
The full per-configuration table is in `results.json`; no
configuration is selected from it.

The train-calibrated gate is 0.3841. Applied to the test sessions it
stands down for about 99% of scored seconds and earns near zero in
every session (gated PnL -1.29, 0.00, -0.01, 0.00, 0.00, +2.74),
while the open strategy earns +58.5, -201.7, -70.1, +20.5, -115.4,
+1098.4. A percentile calibration on short-session VPIN does not
transfer to the demo's calibration-frozen buckets.

### Lead-lag probes (OFI price response, bps per SD of OFI)

At 10 s bars the test-session means are: leads -5 to -1 between
-0.10 and +0.05, lag 0 +1.71 (the contemporaneous association, since
OFI and the mid move come from the same quote updates), lag +1
+0.19, lags +2 to +5 between -0.10 and +0.04. The negative controls
stay near zero while the contemporaneous and next-bar responses
stand out, at 5 s bars (lag 0 +1.14, lag +1 +0.19, leads within
+0.06) and 30 s bars (lag 0 +3.03, lag +1 -0.04, leads within 0.21)
as well.

### Timestamp-shift nulls (lambda mean, 10 s bars)

Unshifted lambda means run +3.93 to +5.58 across the test sessions.
Shifting trade times by 120 s collapses them to -0.69 to +0.16, and
shifting signed flow by 3 bars collapses them to -0.52 to +0.33.
The alignment carries the signal: break it and the estimate goes
away.

### Ablations (OFI response with volume, intensity, volatility controls)

Raw and controlled slopes agree within 0.01 in every test session at
every bar length (10 s bars: raw +0.32 to +0.50, controlled +0.32 to
+0.51, t statistics +42 to +51, R squared 0.57 to 0.66, n 1440 bars
per session). The controls do not move the response.

### Walk-forward direction check (test sessions, unchanged protocol)

Accuracies 0.502 to 0.588 against training-majority baselines 0.509
to 0.567, deltas +0.010, -0.011, +0.026, +0.044, +0.056, -0.007.
Reported alongside the grid, never used to pick a configuration.

## Files

- `grid.json`: preregistered grid, sessions, calibration rule, nulls.
- `results.json`: full results packet (per-session bar diagnostics and
  per-configuration manifests, metrics, and MM reports).
- `examples/e5_robustness.rs`: the runner that produced it.
- `src/robustness.rs`: the grid, robust lambda, train-only
  calibration, lead-lag probes, shift nulls, and ablations.
