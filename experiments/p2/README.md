# P2: Normalize flow by liquidity and activity

## Protocol

Raw OFI and signed BTC volume have regime-dependent scales, and the
v3 model has no explicit depth or intensity control. P2 adds ten
normalized features to the frame (v4, 15 features), using
touch-accessible fields only so archived data stays usable:

* signed flow over a trailing fixed window (6 bars, 60 s);
* OFI normalized by visible touch depth (trailing 30-bar mean of
  (bid size + ask size) / 2);
* quote and trade event rates (trailing 30-bar mean counts per bar);
* trailing realized volatility (trailing 30-bar log-return
  variance, 29 clean within-window returns);
* microprice displacement from the bar's last quote
  ((microprice - mid) / spread, 0 when the spread or the touch
  depth is zero);
* lagged signed flow and OFI at prespecified stationary scales
  (3 and 10 bars).

P2 rows start at bar 29 so every trailing 30-bar window is full,
and share the v3 gap censoring: a row is supervised only if every
bar feeding its features and every bar in its forward target window
was actually observed. The first five v4 columns reproduce
`build_dataset` bit-identically on shared rows, so the v3 baseline
is preserved exactly inside the new frame.

The design grid is frozen in `grid.json` before any grid run (30
cells; 10 s bars; 3-bar horizon; 4 walk-forward splits). The runner
(`examples/p2_ablation.rs`) refuses a grid file that does not match
the library grid exactly, and refuses sessions not listed in the
grid file under `test_sessions`. No test outcome is used for
parameter selection: the default cell (b5-full) stays the reported
cell regardless of results. Every grid cell is reported in
`results.json`.

The B0-B5 ladder is defined in the grid, because no B0-B5
definition exists in the workbook or roadmap text: B0 is the
training-majority predictor, B1 is OFI only, B2 is book state
(OFI, OFI z-score, imbalance, spread z-score), B3 is the v3 frame,
B4 adds the flow-normalization group, and B5 is the full v4 frame.
The grid also runs single-feature additions on B3, leave-one-out
from B5, and permutation and 5-bar shift controls on the P2 block.
The permutation shuffles the P2 block within each fold's test block
with a fixed seed while the fitted model stays unchanged; the shift
takes the P2 block from 5 bars earlier in the same test block and
drops the first 5 rows of each block (20 rows per session).

Sessions are real recorded BTCUSDT sessions from the November 2026
campaign: the six 2026-11-11 and 2026-11-12 test sessions, the same
set as the E5, P1 and N5 runs. Failed runs are retained with their
reason; this run had none.

Absent by design, recorded here rather than silently omitted: VPIN
as a feature (deferred to P3, which tests whether VPIN adds
independent information once confounds are controlled), depth
beyond the touch (unavailable in archived data), and deeper-book or
queue imbalance (not recorded in the archive).

## Results (mean over 6 sessions, session-equal weighting)

8,454 decision rows (1,409 per session); 6,744 held-out
predictions (1,124 per session). B3 is measured on the P2 row set
(1,409 per session), which starts at bar 29 rather than P1's
1,434 labeled rows per session.

| config | accuracy | train-majority baseline | Brier | log loss |
|---|---|---|---|---|
| B0 majority | 0.536 | 0.533 | 0.251 | 0.694 |
| B1 OFI only | 0.541 | 0.533 | 0.250 | 0.693 |
| B2 book state | 0.556 | 0.533 | 0.247 | 0.687 |
| B3 v3 frame | 0.553 | 0.533 | 0.247 | 0.688 |
| B4 flow group | 0.558 | 0.533 | 0.251 | 0.698 |
| B5 full v4 frame | 0.554 | 0.533 | 0.255 | 0.708 |
| B5 permuted | 0.527 | 0.533 | 0.269 | 0.752 |
| B5 shifted 5 bars | 0.529 | 0.533 | 0.267 | 0.745 |
| B4 permuted | 0.538 | 0.533 | 0.266 | 0.745 |
| B4 shifted 5 bars | 0.543 | 0.533 | 0.265 | 0.741 |

The normalized-flow features do not improve the model. The full v4
frame matches the v3 frame on accuracy (0.554 against 0.553, a
+0.0004 difference) while calibrating worse (Brier 0.255 against
0.247, log loss 0.708 against 0.688), and it beats B3 on accuracy
in only 3 of 6 sessions. The flow group alone is the same story:
B4 gains +0.0042 accuracy over B3 with a worse log loss (+0.010).
No single addition on B3 improves both accuracy and log loss; the
largest accuracy gain is the signed-flow window (+0.0052, log loss
+0.004 worse), and the only addition that improves log loss at all
is OFI/depth (-0.0004, accuracy +0.002). Leave-one-out from B5
moves accuracy by at most 0.005 in either direction.

The controls show the P2 block is not inert: permuting it drops
the full model to 0.527 accuracy (below the 0.533 baseline) and
shifting it 5 bars drops it to 0.529, with log loss rising to 0.752
and 0.745. The fitted model does rely on the normalized features,
and their alignment with the target carries signal, but the
logistic model cannot convert that signal into held-out accuracy or
calibration gains over the v3 frame on these sessions. That is the
measured outcome: the features are built, tested and reported, and
the v3 frame stays the model in use.

Per-session numbers for every cell, with per-fold scores and the
default cell's fold weights, are in `results.json`.

## Reproduce

```
cargo run --locked --release --example p2_ablation -- experiments/p2/grid.json experiments/p2/results.json <session_dir>...
```

Session directories are the closing-archive BTC sessions under
`toxicflow-recordings/sessions/`, each holding `quotes.parquet` and
`trades.parquet`.
