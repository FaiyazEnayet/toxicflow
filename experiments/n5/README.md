# N5: Censored next-nonzero price-move hazard baseline

## Protocol

The target is the waiting time from a decision bar to the first
nonzero mid change, with up and down moves as competing events. It
is a separate target from P1's binary/three-class direction labels:
P1 stays the historical direction task, and no hazard number here
is compared against a direction accuracy or baseline.

Each decision row is fixed by information available at its bar
close (the trailing 30-bar feature window must be fully observed,
the same decision rows P1 labels). Every row carries valid
`t_observe <= t_decision <= t_outcome`, or explicit censoring with
a recorded reason:

* `t_observe` is the start of the decision bar; `t_decision` is
  the end of the decision bar, the decision instant.
* For an event at move bar `j`, `t_outcome` is the start of bar `j`
  (the same timestamp P1 records as the nonzero target time) and
  the waiting time is `wait_bars = j - i`. The move is first
  visible at the start of the move bar, so
  `t_outcome - t_decision = (wait_bars - 1) * bar_ms`.
* For a censored row, `t_censor` is the end of the last fully
  observed at-risk bar, so
  `t_censor - t_decision = wait_bars * bar_ms`.

Censoring reasons: `no_move` (complete observed cutoff window
with no nonzero move), `incomplete` (window runs past the last bar
before any move), `gap` (window up to the first move, or the full
window, touches an unobserved bar). A move whose window touches an
unobserved bar is censored, never labeled, so no outcome window
crosses a feed gap. A move observed before the last bar labels the
row even when the cutoff window itself is incomplete.

The baseline is a constant-hazard (exponential) competing-risk
model: one hazard per cause, fit by maximum likelihood on train
rows only, in closed form (`lambda_c = n_c / total time at
risk`). The waiting-time log-likelihood of a row is
`ln(lambda_c) - lambda_total * wait` for an event of cause `c`,
and `-lambda_total * wait` for a censored row. Calibration
compares the fitted survival `exp(-lambda_total * t)` against a
Kaplan-Meier estimate on held-out rows at horizons 1, 2, 3, 5, and
10 bars.

The design grid is frozen in `grid.json` before any run (10 s
bars, 10-bar cutoff, 4 walk-forward splits, calibration horizons
1, 2, 3, 5, 10 bars). The runner (`examples/n5_hazard.rs`)
refuses a grid file that does not match the library grid exactly,
and refuses sessions not listed in the grid file under
`test_sessions`. Each walk-forward fold fits on its train rows
only with a purge of one cutoff length, so no train row's outcome
window reaches a test row's decision bar. The baseline has no
tuned parameters and no test outcome is used for parameter
selection. Every session is reported.

Sessions are real recorded BTCUSDT sessions from the November 2026
campaign: the six 2026-11-11 and 2026-11-12 test sessions, the same
set as the P1 and E5 test sessions. Failed runs are retained with
their reason; this run had none.

## Results

8,622 decision rows across the six sessions. 8,594 events (4,252
up, 4,342 down), 28 censored (16 no move, 12 incomplete, 0 gap).
The row populations reproduce P1's next-nonzero track exactly
(same decision rows, same move bars, same directions, same
censoring counts), which the `n5_p1_nonzero_agreement` fixture
pins; that is a dataset-consistency check, not an outcome
comparison. No session has unobserved bars, so the gap-censoring
machinery is exercised by fixtures only.

| Session | Decision | Events (up/down) | Censored (no move/incomplete/gap) | Pooled mean loglik | Fold lambda_total range |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 20261111-163501 | 1,437 | 1,435 (682/753) | 2 (0/2/0) | -1.9300 | 0.8361-0.9295 |
| 20261111-203511 | 1,437 | 1,435 (692/743) | 2 (0/2/0) | -1.9246 | 0.7249-0.7638 |
| 20261112-003521 | 1,437 | 1,435 (681/754) | 2 (0/2/0) | -1.9805 | 0.8258-0.8938 |
| 20261112-043531 | 1,437 | 1,428 (766/662) | 9 (6/3/0) | -2.3442 | 0.6207-0.7030 |
| 20261112-083541 | 1,437 | 1,425 (718/707) | 12 (10/2/0) | -2.0667 | 0.3683-0.5764 |
| 20261112-123552 | 1,437 | 1,436 (713/723) | 1 (0/1/0) | -1.7114 | 0.8602-0.9515 |

Pooled held-out calibration, fitted survival vs Kaplan-Meier
observed survival:

| Session | h=1 | h=2 | h=3 | h=5 | h=10 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 20261111-163501 | 0.415 / 0.183 | 0.172 / 0.052 | 0.072 / 0.017 | 0.012 / 0.003 | 0.000 / 0.000 |
| 20261111-203511 | 0.473 / 0.167 | 0.224 / 0.053 | 0.106 / 0.019 | 0.024 / 0.004 | 0.001 / 0.000 |
| 20261112-003521 | 0.423 / 0.208 | 0.179 / 0.062 | 0.076 / 0.023 | 0.014 / 0.009 | 0.000 / 0.000 |
| 20261112-043531 | 0.517 / 0.373 | 0.267 / 0.206 | 0.138 / 0.123 | 0.037 / 0.044 | 0.001 / 0.005 |
| 20261112-083541 | 0.611 / 0.207 | 0.376 / 0.073 | 0.233 / 0.027 | 0.091 / 0.007 | 0.010 / 0.000 |
| 20261112-123552 | 0.401 / 0.009 | 0.161 / 0.002 | 0.065 / 0.001 | 0.010 / 0.000 | 0.000 / 0.000 |

The constant-hazard baseline does not calibrate: in all six
sessions the observed survival falls faster than the fitted
exponential at short horizons. In session 20261111-163501, 1,204
of 1,435 events (84%) land on the very next bar, so the
waiting-time distribution is dominated by immediate moves and a
single constant hazard cannot describe its shape. The baseline
stands as the transparent reference it was built to be; a
decreasing-hazard or discrete-time model is the follow-up, not a
claim made here.

## Verification

* Synthetic constant-hazard oracle: the library log-likelihood
  evaluated at known hazards (up 0.10, down 0.05 per bar) matches
  a hand-computed value within 1e-8
  (`n5_oracle_loglik_matches_hand_computation`).
* MLE recovery: on a 4,000-row synthetic constant-hazard sample
  (total hazard 0.12 per bar, up 0.08, down 0.04, cutoff-censored
  at 10 bars, seed 20261117), the fit recovers lambda_up 0.0752
  (5.9% error), lambda_down 0.0369 (7.8%), lambda_total 0.1121
  (6.6%), all inside the 10% bar.
* Kaplan-Meier and calibration tables match hand-computed values
  on small known datasets; the walk-forward fits each fold on its
  train slice only and the pooled log-likelihood is the exact sum
  of the fold log-likelihoods.
* Independent cross-check: the wait distribution for session
  20261111-163501 recomputed from the session parquet in Python
  (1,435 events, waits {1: 1204, 2: 171, 3: 40, 4: 14, 5: 3, 6: 2,
  7: 1}, 2 incomplete censored) matches the library dataset row
  for row.
