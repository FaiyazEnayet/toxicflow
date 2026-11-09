# Direction confirmation protocol

Registered 2026-11-08, before the first slot of the 2026-11 recording
campaign completed and before anyone selecting the predictor had access
to any campaign outcome. This document freezes the design of a single
confirmatory test of the 60-second direction predictor. The SHA-256
digest of this document is pinned in `src/direction_manifest.rs`
(`PROTOCOL_DIGEST`); a test fails if the document changes without the
digest being updated deliberately. Substantive amendments require a
new dated protocol document, not an edit to this one.

## Why this study exists

The TF6 pilot repaired the predictor's evaluation (training-majority
baseline, block-bootstrap intervals, a prediction ledger) and left the
question open. On the main archived session the predictor scores
0.790865 accuracy against 0.786058 for the training-majority rule, with
balanced accuracy 0.535770 over 832 scored rows. Across the five pilot
sessions the paired accuracy difference is +0.018609 with a
session-resampling 95% interval of [-0.025509, +0.055882]: a small
positive point estimate that the data cannot distinguish from zero.
This study is the prespecified attempt to settle it on fresh data.

## Exposure rule

If anyone selecting, fitting or tuning the predictor saw campaign
outcomes before this registration, or sees holdout (slots 11-20)
outcomes before the model freeze in section 7, the affected study is
exploratory: the confirmation in section 8 does not run, the exposure
is recorded, and any results are reported as exploratory. There is no
partial credit version of this rule.

## 1. Eligible population

Two cohorts, kept separate in every report:

- **Archives.** The six recorded sessions named in
  `docs/info_speed_report.json`, in order: `session_325h`,
  `session-20261030-143624-utc`, `session-20261101-203631-utc`,
  `session-20261102-013620-utc`, `session-20261104-073619-utc`,
  `session-20261106-143623-utc`. Note: the TF6 pilot report covers only
  the first five; the 2026-11-06 archive has no TF6 row, and no TF6
  number may be quoted for it.
- **Campaign.** Twenty scheduled 4-hour BTCUSDT recording slots from
  the November 2026 rotation, identified by their session IDs
  (`session-YYYYMMDD-HHMMSS-utc`) in chronological scheduled-start
  order. Interrupted attempts that were never counted are not slots
  and are excluded. Slot roles are fixed by that order alone:
  slots 01-06 fit, slots 07-10 development, slots 11-20 sealed
  holdout. Content hashes attach to each slot identity in the manifest
  when its capture lands; a slot whose files change after its hash is
  attached no longer matches the manifest.

A campaign slot is admitted only if its Parquet files pass the repo's
strict storage validation (including positive quote sizes and
bid < ask), it has at least 13,680 quote-observed seconds and at least
12,960 origins with complete common support out of 14,400 scheduled
seconds, and these counts are computed without using any price-change
outcome. A slot that fails admission irreparably, a missing archive,
an invalid seal, or an incompatible acquisition makes the study
incomplete (section 8). Captures are never substituted.

## 2. Frame and target

Bars are UTC epoch-aligned one-second half-open bars. The decision
time is the right boundary (close) of the origin bar; the origin bar's
last quote supplies the mid and the depth. The target is strictly
post-close:

    y[t] = 1 if mid[t + 60] > mid[t], else 0

Down and flat outcomes are both 0. The common support mask requires
actual observations on every bar from t - 60 through t + 60 inside one
contiguous observed segment of one session, and excludes the first and
last quote-containing bars of a session as potentially partial. Every
feature family and the baseline use this same mask. Nothing conditions
on returns, burst flags, or indicator availability.

At each training boundary, fit-side origins newer than 660 seconds
before the first scheduled evaluation-slot start are removed, and the
gap between the last fit origin and the first evaluation origin is at
least 660 seconds. This covers the maximum 600-second feature lookback
plus the 60-second label window. Scalers, AR coefficients, thresholds
and the training-majority rule are fit only on eligible training-role
data.

## 3. Feature families

Four families, with fixed definitions, compared on the same prediction
ledger. All flow quantities are per one-second bar; F[t] is the bar's
order-flow imbalance (OFI, Cont-Kukanov-Stoikov) and D[t] is the mean
of the best bid and ask sizes at the bar's last quote.

- **R, reference.** Nine columns, in order: F[t]; trailing sums of F
  over 5, 30 and 60 bars; the training-fitted AR(10) residual of F;
  F[t] signed by the training-fitted burst flag (burst: |F| at or
  above the training nearest-rank 95th percentile, rank
  ceil(0.95 * n), and nonzero F); the current touch imbalance; signed
  trade quantity in the origin bar; the current log mid return in
  basis points.
- **F, flow and liquidity controls.** R plus: N[t] = asinh(F[t]/D[t]);
  the mean of N over 5 and 30 bars; the mean touch imbalance over 5
  bars; five-second signed trade quantity divided by total trade
  quantity (0 when volume is zero); log1p of total trade quantity over
  30 bars; log1p of quote count over 30 bars; the square root of the
  mean squared one-second mid return in basis points over 30 bars.
- **V, incremental VPIN.** F plus a causal VPIN value and an
  availability bit. At each decision, take trades in the last 600
  seconds strictly before decision time, clipped to the start of the
  current contiguous observed segment and to usable file support;
  split them into fixed 1 BTC buckets with the existing sized-bucket
  routine, drop the final partial bucket, and average the absolute
  buy-sell imbalance over the last 20 completed buckets. With fewer
  than 20 completed buckets the value is (0, 0); otherwise (VPIN, 1).
  No bucket-size search. A missing VPIN never drops an origin.
- **I, bounded nonlinear.** F plus three interaction terms: N[t]
  times the current touch imbalance; N[t] times log1p of quote count
  over 30 bars; N[t] times the current log mid return in basis points.
  No tree model, no temporal network, no V + I combination.

Preprocessing is identical for every family: two-pass training mean
and population standard deviation per final column; a column with
standard deviation at most 1e-12 is zeroed; otherwise z-scores are
clipped to +/-8. No whole-test-session standardization. All features
must be finite.

## 4. Fitting rule

One deterministic rule for all families: unweighted ridge logistic
regression minimizing mean binary log loss plus 0.01 times the squared
slope norm divided by 2, with a free intercept. Slopes initialize at 0
and the intercept at the logit of training prevalence. Newton steps
are solved by Cholesky factorization, with step size starting at 1 and
halved up to 40 times under an Armijo coefficient of 0.0001; the fit
must reach gradient infinity norm at most 1e-8 within 100 accepted
updates. A nonfinite fit or a single-class training set is a failure,
not a result. The baseline is the unweighted training majority, with
ties going to class 1.

## 5. Operating rules

Each family is scored under exactly two operating rules: predict 1
when p > 0.50, and predict 1 when p > training prevalence. No other
cutoff is searched, on any data, at any stage.

## 6. Development and selection

Development uses the archives plus campaign slots 01-06 to predict
slot 07; for each of slots 08-10 the fit pool grows by only the
earlier campaign slots, under the section 2 embargo. Each evaluation
session is scored once per registered variant (four families times two
operating rules: eight development rows).

For each row, per-session confusion tables are normalized by scored
counts and averaged with equal session weights. From the averaged
tables: Delta = model accuracy - training-majority accuracy, and
Gamma = balanced accuracy - 0.50. The selection score is
S = min(Delta / 0.010, Gamma / 0.020).

A family's operating rule is chosen by larger S, then larger Delta,
then the 0.50 rule. A child family clears its parent only if its
selected-rule S improves by at least 0.10, its Delta falls by at most
0.002, its Gamma falls by at most 0.005, and its within-session S
strictly improves on at least three of the four development slots (a
slot without both classes casts no positive vote). F must clear R. V
and I must each clear both F and R. Among qualifying families the
selected one is chosen by larger S, then larger Delta, then the fixed
simplicity order R, F, V, I. R qualifies as the fallback whenever its
pooled development score is defined.

No cutoffs, feature additions, hyperparameters or horizons are
searched beyond this paragraph. A failed fit or an undefined metric is
fixed only by repairing a demonstrable implementation defect and
re-running this same specification; low scores alone never cancel
the test. Development results are descriptive and selection-biased;
they are never reported as confirmatory evidence.

## 7. Freeze

The single selected family and rule are refit on the six archives plus
slots 01-10 under the frozen embargo, recomputing its transforms,
majority rule and threshold as registered. The frozen artifact records
the code, data and feature hashes, the variant and rule, the fitted
parameters, the split memberships, the exact version and the
timestamp. Decisions must replay exactly and probabilities to within
1e-12. After the freeze, nothing about the artifact may change in
response to any holdout information; only the selected configuration
is released to the confirmation step.

## 8. Confirmation

Slots 11-20 are scored exactly once, with the single frozen artifact
and the frozen majority rule. A deterministic replay of the same run
verifies it. No other variant is scored on the holdout, not even as a
diagnostic.

Endpoints come from the equal-session averaged confusion tables, as in
section 6, with per-session scores, recalls, prevalence, flat counts,
coverage, and Brier and calibration diagnostics reported alongside.
Intervals are paired bootstrap intervals on aligned nonempty time
bins, computed jointly for model and baseline: draw five consecutive
session pairs with replacement from (11,12), (13,14), (15,16),
(17,18), (19,20), and within each drawn session resample its original
number of bins with replacement. This is done at both 300-second and
600-second block lengths, with 20,000 valid draws per block length, a
xorshift generator seeded 0xD1B54A32D192ED03, chronological source
arrays, and percentile ranks ceil(0.025 * B) and ceil(0.975 * B).
Replicates missing a pooled class are redrawn; if 20,000 valid draws
cannot be reached in 20,200 attempts, interval claims are suppressed
and the study is incomplete. Fixed-cohort and per-session diagnostics
at 300 seconds use seed 0x9E3779B97F4A7C15. Leave-one-pair-out ranges
are reported, and the report states plainly that five adjacent pairs
give limited support for generalization.

Outcome gates, applied exactly as written:

- **Win:** all ten assigned sessions are present, Delta >= +0.010,
  balanced accuracy >= 0.520, and the 95% lower bounds for both Delta
  and Gamma are strictly positive at both block lengths.
- **Partial:** exactly one endpoint clears its point margin and both
  of its interval checks, and the win gate does not.
- **Null:** neither endpoint clears, on a valid complete run.
- **Incomplete:** a gate invalidates the confirmation: exposure
  (section "Exposure rule"), missing or inadmissible inputs, invalid
  intervals, or a failed replay.

After the readout the study stops. There are no rescue variants and no
outcome-conditioned exclusions. If a defect discovered after unsealing
invalidates the result, the invalid result is what gets published, and
any amended confirmation requires a fresh holdout and a new
registration.

## 9. Access boundary

Fit and development code resolves campaign and archive sessions only
through `src/direction_manifest.rs`. The manifest classifies every
session ID it knows: archives and slots 01-10 resolve for fit use;
slots 11-20 do not resolve, and any attempt to resolve one, directly
or through a derived or cache key the manifest controls, is an error.
Holdout files stay outside the fit allowlist's view; a data custodian
may run source QC on them and return only hashes and coverage counts
before the freeze. The confirmation step runs as a distinct command
and must present this protocol's registered digest; scoring any other
configuration, or scoring with a mismatched digest, is rejected.

The speed closing analysis of the same campaign runs under its own
separately frozen protocol, with per-session estimands. Its outputs
are outcome-bearing for the holdout slots and sit behind the same
boundary for anyone selecting the predictor: if that separation fails,
the exposure rule above applies.

## 10. Publication

Every valid outcome is published on the same evidence standard: the
report JSON (`docs/direction_confirmation_report.json`), all eight
development selection rows with their parent comparisons, the one
holdout result or the named incomplete gate, both metrics at both
block lengths against both baselines, and the audit ledger or its
hash with per-block aggregate counts. The registered wordings are:

- **Win.** "On ten prespecified held-out BTCUSDT recordings, the
  60-second model scored [A] accuracy versus [B] for training
  majority, with paired difference [Delta] (95% block interval
  [L, U]). Balanced accuracy was [BA] versus 0.50 (95% block interval
  [L, U]). Both preregistered gates cleared at both block lengths."
- **Partial.** The same metric sentence, plus: "Only the [endpoint]
  gate cleared. The study did not establish a joint win."
- **Null.** The same metric sentence, plus: "Neither preregistered
  gate cleared. The campaign produced no confirmed improvement."
- **Incomplete.** "The confirmation stopped at [gate] because
  [reason]. The protocol and available coverage checks are published."

No synthetic or substituted evidence is presented as a headline
result, and no outcome is retested after readout.
