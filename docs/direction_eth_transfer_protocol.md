# ETH direction transfer protocol

Registered 2026-11-12, after the BTC direction confirmation closed
and before any ETH family selection or ETH holdout scoring. The
BTC-selected configuration (family F under the p>0.50 rule) is
transferred to ETHUSDT as a cross-asset test: the same model
specification and decision rule, with coefficients refit on ETH
training data only. The SHA-256 digest of this document is pinned in
`src/direction_manifest.rs` (`ETH_PROTOCOL_DIGEST`); a test fails if
the document changes without the digest being updated deliberately.
Substantive amendments require a new dated protocol document, not an
edit to this one.

## Why this study exists

The BTC confirmation (B4) showed the frozen F rule beating the
training-majority baseline on ten held-out BTCUSDT sessions
(accuracy 0.5428 against 0.5245, paired difference +0.0184,
balanced accuracy 0.5378), but its paired-difference interval spans
zero and one session went negative, so no joint win was established.
The open question is whether that configuration, selected on BTC,
retains predictive usefulness on another asset without
asset-specific validation tuning. That is a stronger robustness
test than selecting a new ETH winner from the same four candidate
families, which would answer only whether a model can be optimized
specifically for ETH, and would add another selection step on only
four validation slots.

## Exposure rule

If anyone selecting, fitting or tuning the predictor saw ETH holdout
(slots 11-20) outcomes before the model freeze in section 7, the
affected study is exploratory: the confirmation in section 8 does
not run, the exposure is recorded, and any results are reported as
exploratory. There is no partial credit version of this rule. The
ETH information-speed results (R(60) +0.352, D +0.070, four promoted
sessions) concern a different research question and must not
influence anything in this study.

## 1. Eligible population

One cohort, kept separate from the BTC study in every report:

- **Campaign.** Twenty scheduled 4-hour ETHUSDT recording slots
  from the November 2026 rotation (November 9-12, 2026),
  identified by their session IDs
  (`session-YYYYMMDD-HHMMSS-utc`) in chronological scheduled-start
  order. Interrupted attempts that were never counted are not slots
  and are excluded. Slot roles are fixed by that order alone:
  slots 01-06 fit, slots 07-10 validation, slots 11-20 sealed
  holdout. Content hashes attach to each slot identity in the
  manifest when its capture lands; a slot whose files change after
  its hash is attached no longer matches the manifest.
- **Archives: none.** The six archive sessions of the BTC study
  were BTC-only and must not enter this study. The ETH fit pool
  starts with six four-hour recordings. That is a real limitation
  in training exposure, not a reason to change any fitting rule,
  feature definition or qualification margin; the report states the
  number of eligible training observations at every boundary.

A campaign slot is admitted only if its Parquet files pass the
repo's strict storage validation (including positive quote sizes
and bid < ask), it has at least 13,680 quote-observed seconds and
at least 12,960 origins with complete common support out of
14,400 scheduled seconds, and these counts are computed without
using any price-change outcome. A slot that fails admission
irreparably, an invalid seal, or an incompatible acquisition
makes the study incomplete (section 8). Captures are never
substituted.

## 2. Frame and target

Bars are UTC epoch-aligned one-second half-open bars. The decision
time is the right boundary (close) of the origin bar; the origin
bar's last quote supplies the mid and the depth. The target is
strictly post-close:

    y[t] = 1 if mid[t + 60] > mid[t], else 0

Down and flat outcomes are both 0. The common support mask requires
actual observations on every bar from t - 60 through t + 60 inside
one contiguous observed segment of one session, and excludes the
first and last quote-containing bars of a session as potentially
partial. Every feature family and the baseline use this same mask.
Nothing conditions on returns, burst flags, or indicator
availability.

At each training boundary, fit-side origins newer than 660 seconds
before the first scheduled evaluation-slot start are removed, and
the gap between the last fit origin and the first evaluation origin
is at least 660 seconds. This covers the maximum 600-second feature
lookback plus the 60-second label window. Scalers, AR coefficients,
thresholds and the training-majority rule are fit only on eligible
training-role data, and only on ETH data: no BTC fitted value of
any kind (means, standard deviations, regression coefficients,
thresholds, majority labels) is carried into this study.

## 3. Feature family

Exactly one family, frozen from the BTC study: **F, flow and
liquidity controls**, 17 columns, as defined in section 3 of the
BTC protocol (`docs/direction_confirmation_protocol.md`): the R
reference columns plus N[t] = asinh(F[t]/D[t]); the mean of N over
5 and 30 bars; the mean touch imbalance over 5 bars; five-second
signed trade quantity divided by total trade quantity (0 when
volume is zero); log1p of total trade quantity over 30 bars; log1p
of quote count over 30 bars; the square root of the mean squared
one-second mid return in basis points over 30 bars. F[t] is the
bar's order-flow imbalance and D[t] is the mean of the best bid and
ask sizes at the bar's last quote.

Preprocessing is identical to the BTC protocol: two-pass training
mean and population standard deviation per final column; a column
with standard deviation at most 1e-12 is zeroed; otherwise z-scores
are clipped to +/-8. No whole-test-session standardization. All
features must be finite.

## 4. Fitting rule

One deterministic rule: unweighted ridge logistic regression
minimizing mean binary log loss plus 0.01 times the squared slope
norm divided by 2, with a free intercept. Slopes initialize at 0
and the intercept at the logit of training prevalence. Newton steps
are solved by Cholesky factorization, with step size starting at 1
and halved up to 40 times under an Armijo coefficient of 0.0001;
the fit must reach gradient infinity norm at most 1e-8 within 100
accepted updates. A nonfinite fit or a single-class training set is
a failure, not a result. The baseline is the unweighted training
majority, with ties going to class 1.

## 5. Operating rule

Exactly one operating rule: predict 1 when p > 0.50. No other
cutoff is searched, on any data, at any stage.

## 6. Validation (B3)

Validation uses campaign slots 01-06 to predict slot 07; for each
of slots 08-10 the fit pool grows by only the earlier campaign
slots, under the section 2 embargo. Each evaluation session is
scored once under the frozen configuration (family F, rule
p>0.50).

For the validation row, per-session confusion tables are
normalized by scored counts and averaged with equal session
weights, exactly as in section 6 of the BTC protocol. From the
averaged tables: Delta = model accuracy - training-majority
accuracy, and Gamma = balanced accuracy - 0.50. No
confidence-interval gate is added at this stage; B3 is a screening
step, and B4 provides the confirmation.

B3 passes only if both bars clear on validation:

- Delta >= +0.010, and
- balanced accuracy >= 0.520.

If either fails, the outcome is a failed cross-asset validation:
it is published as such, ETH slots 11-20 are not opened for this
experiment, and there is no fallback family or threshold. No
cutoffs, feature additions, hyperparameters or horizons are
searched beyond this document. A failed fit or an undefined metric
is fixed only by repairing a demonstrable implementation defect
and re-running this same specification; low scores alone never
cancel the test. Validation results are descriptive; they are never
reported as confirmatory evidence.

## 7. Freeze

If B3 passes, the frozen configuration (family F, rule p>0.50)
is refit on slots 01-10 under the frozen embargo, recomputing its
transforms, majority rule and threshold as registered. The frozen
artifact records the code, data and feature hashes, the variant
and rule, the fitted parameters, the split memberships, the exact
version and the timestamp. Decisions must replay exactly and
probabilities to within 1e-12. After the freeze, nothing about the
artifact may change in response to any holdout information; only
the frozen configuration is released to the confirmation step.

## 8. Confirmation

Slots 11-20 are scored exactly once, with the single frozen
artifact and the frozen majority rule. A deterministic replay of
the same run verifies it. No other variant is scored on the
holdout, not even as a diagnostic.

Endpoints come from the equal-session averaged confusion tables,
as in section 6, with per-session scores, recalls, prevalence,
flat counts, coverage, and Brier and calibration diagnostics
reported alongside. Intervals are paired bootstrap intervals on
aligned nonempty time bins, computed jointly for model and
baseline: draw five consecutive session pairs with replacement
from (11,12), (13,14), (15,16), (17,18), (19,20), and within each
drawn session resample its original number of bins with
replacement. This is done at both 300-second and 600-second block
lengths, with 20,000 valid draws per block length, a xorshift
generator seeded 0xD1B54A32D192ED03, chronological source arrays,
and percentile ranks ceil(0.025 * B) and ceil(0.975 * B).
Replicates missing a pooled class are redrawn; if 20,000 valid
draws cannot be reached in 20,200 attempts, interval claims are
suppressed and the study is incomplete. Fixed-cohort and
per-session diagnostics at 300 seconds use seed
0x9E3779B97F4A7C15. Leave-one-pair-out ranges are reported, and
the report states plainly that five adjacent pairs give limited
support for generalization.

Outcome gates, applied exactly as written:

- **Win:** all ten assigned sessions are present, Delta >= +0.010,
  balanced accuracy >= 0.520, and the 95% lower bounds for both
  Delta and Gamma are strictly positive at both block lengths.
- **Partial:** exactly one endpoint clears its point margin and
  both of its interval checks, and the win gate does not.
- **Null:** neither endpoint clears, on a valid complete run.
- **Incomplete:** a gate invalidates the confirmation: exposure
  (section "Exposure rule"), missing or inadmissible inputs,
  invalid intervals, or a failed replay.

After the readout the study stops. There are no rescue variants
and no outcome-conditioned exclusions. If a defect discovered
after unsealing invalidates the result, the invalid result is
what gets published, and any amended confirmation requires a
fresh holdout and a new registration.

## 9. Access boundary

Fit and validation code resolves campaign sessions only through
`src/direction_manifest.rs` (the ETH transfer manifest). The
manifest classifies every session ID it knows: slots 01-10
resolve for fit use; slots 11-20 do not resolve, and any attempt
to resolve one, directly or through a derived or cache key the
manifest controls, is an error. Holdout files stay outside the
fit allowlist's view; a data custodian may run source QC on them
and return only hashes and coverage counts before the freeze. The
confirmation step runs as a distinct command and must present
this protocol's registered digest; scoring any other
configuration, or scoring with a mismatched digest, is rejected.

## 10. Publication

Every valid outcome is published on the same evidence standard:
the report JSON (`docs/direction_eth_b3_report.json` for
validation, `docs/direction_eth_confirmation_report.json` for the
confirmation), the validation row with its per-session scores,
the one holdout result or the named incomplete gate, both metrics
at both block lengths against both baselines, and the audit ledger
or its hash with per-block aggregate counts. The registered
wordings are:

- **Win.** "On ten prespecified held-out ETHUSDT recordings, the
  60-second model scored [A] accuracy versus [B] for training
  majority, with paired difference [Delta] (95% block interval
  [L, U]). Balanced accuracy was [BA] versus 0.50 (95% block
  interval [L, U]). Both preregistered gates cleared at both block
  lengths."
- **Partial.** The same metric sentence, plus: "Only the [endpoint]
  gate cleared. The study did not establish a joint win."
- **Null.** The same metric sentence, plus: "Neither preregistered
  gate cleared. The campaign produced no confirmed improvement."
- **Incomplete.** "The confirmation stopped at [gate] because
  [reason]. The protocol and available coverage checks are
  published."
- **Failed validation.** "On four prespecified ETHUSDT validation
  sessions, the frozen BTC configuration scored [A] accuracy
  versus [B] for training majority, with paired difference
  [Delta] and balanced accuracy [BA]. It did not clear the
  preregistered validation bars, so the holdout was not opened."

No synthetic or substituted evidence is presented as a headline
result, and no outcome is retested after readout.

## Qualification

BTC and ETH were recorded during the same November 9-12, 2026
campaign. An ETH success therefore demonstrates cross-asset
generalization under substantially shared market conditions, not
fully independent replication across different market regimes.
That limitation is stated with every ETH result.
