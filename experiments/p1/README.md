# P1: Repair target semantics and timestamps

## Protocol

The original task labels a row by the sign of the forward return over
the next three bars, so a zero return and a negative return share
class 0. P1 keeps that task exactly as the baseline
(`P1Dataset::baseline_bars` reproduces `build_dataset` row for row,
features bit-identical) and adds two separate labeled tracks over the
same decision rows:

* next-nonzero: the sign of the first nonzero mid change within a
  frozen cutoff of 10 bars, censored when no move is observed inside
  the cutoff;
* three-class: down, flat, or up over the horizon, with flat an
  explicit band around zero (|forward return| <= flat band) instead
  of a side effect of the sign.

Each decision row is fixed by information available at its bar close
(the trailing 30-bar feature window must be fully observed). Every
label traces to a concrete future mid observation: each row records
the target mid and the target bar time, and the nonzero track records
the bar of the first move. Censored rows are counted per track and
per reason, never dropped silently. Baseline and three-class rows are
censored when the target window runs past the last bar or touches an
unobserved bar. Next-nonzero rows are censored when no nonzero mid
change is observed inside the cutoff window: the window runs past the
last bar before any move, the window up to the first move touches an
unobserved bar, or a complete observed window holds no move. A move
observed before the last bar labels the row even when the cutoff
window itself is incomplete.

The design grid is frozen in `grid.json` before any grid run (3
horizons x 3 flat bands, 9 cells; horizons 1, 3, 6 bars; flat bands 0,
1e-5, 3e-5; nonzero cutoff 10 bars; 10 s bars; 4 walk-forward
splits). The runner (`examples/p1_report.rs`) refuses a grid file
that does not match the library grid exactly, and refuses sessions
not listed in the grid file under `test_sessions`. No test outcome is
used for parameter selection: the default cell (horizon 3, flat band
1e-5) stays the reported cell regardless of results. Every grid cell
is reported in `results.json`.

Sessions are real recorded BTCUSDT sessions from the November 2026
campaign: the six 2026-11-11 and 2026-11-12 test sessions, the same
set as the E5 test sessions. Failed runs are retained with their
reason; this run had none.

The nonzero track trains and scores only on rows where a move occurs
within the cutoff, so its accuracy is conditional on a move
occurring; the censored share is reported next to it. Train class
priors are train-only in every fold.

## Results (default cell: horizon 3, flat band 1e-5; mean over 6 sessions)

8,622 decision rows. Baseline: 8,604 labeled, 18 censored; 397
labeled rows (4.6%) have a forward return of exactly zero, the rows
the old binary rule lumps with negatives. Next-nonzero: 8,594
labeled (4,252 up, 4,342 down), 28 censored (0.3%: 16 with no move
in a complete window, 12 past the end, 0 gap). Three-class: 8,604
labeled (4,142 down, 618 flat, 3,844 up), 18 censored.

| track | accuracy | train-majority baseline | log loss | Brier |
|---|---|---|---|---|
| baseline (binary sign) | 0.552 | 0.533 | 0.690 | 0.248 |
| next-nonzero | 0.606 | 0.508 | 0.667 | 0.237 |
| three-class | 0.511 | 0.486 | 0.896 | 0.560 |

The next-nonzero target separates better than the baseline on every
session (accuracy 0.554 to 0.632 against baselines 0.490 to 0.522,
log loss 0.655 to 0.691), so the shorter-lived structure the
hypothesis pointed at is visible in the first-move target. The
three-class target adds little at the 3-bar horizon: accuracy 0.477
to 0.543 against baselines 0.420 to 0.528, and its flat class is
effectively unpredicted there. Pooled held-out confusion
([true][predicted], down/flat/up) is [[2009, 8, 1288], [232, 1, 248],
[1573, 11, 1494]]: the model predicts flat on only 20 of 6,864
held-out rows, so flat recall is near zero at this horizon. The flat
class is larger and predicted at the 1-bar horizon (23.2% of labeled
rows), where the three-class accuracy is 0.476 against a 0.388
baseline.

Baseline accuracy falls with horizon (1 bar 0.630, 3 bars 0.552, 6
bars 0.531), while the nonzero track is horizon-free by construction
at 0.606. Per-session numbers for every cell, with per-fold train
priors, confusion tables, log loss, and Brier scores, are in
`results.json`.

## Reproduce

```
cargo run --locked --release --example p1_report -- experiments/p1/grid.json experiments/p1/results.json <session_dir>...
```

Session directories are the closing-archive BTC sessions under
`toxicflow-recordings/sessions/`, each holding `quotes.parquet` and
`trades.parquet`.
