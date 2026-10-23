# Changelog

## [Unreleased]
### Fixed
- Prediction feature numerics (Stage 3 N1): `rolling_mean_sd` no longer uses sliding sums of squares (E[x^2] - E[x]^2). Each trailing window is computed independently with a two-pass mean-subtracted sum of squares, and a bit-identical window yields sd exactly 0. On the archived 3.25h BTCUSDT session the bar spread is an economically constant one tick (two f64 values near 0.01 differing at the 1e-12 level), and the old formula turned that into spread z-scores in [-14.07, +9.70] with 293 of 1,167 rows beyond |z| = 5 (legacy sd sat in (0, 1e-9) on 671 bars: pure cancellation residue). After the fix the same rows give z in [-3.89, +3.89] with no row beyond |z| = 5. Zero-variance spread windows are flagged explicitly (`Bar.spread_degenerate`, z pinned to exactly 0; 413 of 1,167 archived rows). The feature frame is frozen as `FEATURE_VERSION = "v2"` (v1 = pre-N1), carried on `EvalReport` and in the export summary JSON. Regression fixtures: analytic-variance series under a 1e6 offset (relative error 1.4e-5 before, 2.2e-16 after, against a 1e-10 bar), constant one-tick spread, prefix invariance, finite logistic probabilities. Test count: 69 → 74. No version bump.

## [v1.2.3] - 2026-10-18
### Fixed
- Connection policy (`src/net.rs`): the recorder no longer assumes an authenticated egress proxy. With no proxy configured it connects directly via TLS; with a proxy URL it tunnels through HTTP CONNECT, accepting both `http://host:port` and `http://user:pass@host:port` forms. A malformed proxy URL is validated once up front and fails fast with an actionable error instead of being retried until the session deadline (`record_session` checks the policy before its retry loop). Credentials are redacted in all log and error output. Regression tests: direct / unauthenticated-proxy / authenticated-proxy fixtures against local endpoints, proxy refusal surfacing, and a fail-fast test proving a bad proxy URL returns immediately rather than burning the deadline.

## [v1.2.2] - 2026-10-18
### Fixed
- `run_mm` no longer peeks at the future: the volume-bucket size is frozen from a leading 300-second calibration window instead of whole-session volume; each second is quoted at the last quote observed strictly before that second starts (second 0 is never quoted); the VPIN gate reads only VPIN completed strictly before the second starts. Appending future trades now leaves earlier bucket thresholds, posted quotes, gate decisions and PnL unchanged. Regression tests: a hand-computed two-trade fixture matching inventory/cash with the decision delay, and a future-trades invariance check (both fail on the old simulation). README toy PnL refreshed on the recorded session: open -11.34, gated -0.84. Test count: 56 → 58.

## [v1.2.1] - 2026-10-18
### Fixed
- `ofi_bars` no longer backfills empty bars with the incoming future quote. Empty bars now carry forward the last observed quote's mid, spread and imbalance with zero OFI, so the feature frame is causal: appending a later quote leaves already-emitted bars unchanged. Regression tests with sparse-timestamp fixtures (quotes at 0s and 30s with different mids; prefix-invariance at both the bar and dataset level). Test count: 52 → 56.

## [v1.2.0] - 2026-10-17
### Added
- 28 market-data replay tests proving the order book stays correct under out-of-order messages, duplicates, dropped messages, disconnects/reconnects, bad snapshots, and extreme activity bursts. Test count: 23 → 51.
- `OrderBook::is_crossed()` detects crossed/locked books (bid >= ask).

### Fixed
- `merge_snapshot` was rejecting valid snapshots, forcing unnecessary re-downloads. Now follows the documented exchange procedure.
- `read_trades` and `read_quotes` no longer panic on parquet files with a missing column or wrong column type. The column helpers return a descriptive error instead, so `analyze` and `export` exit cleanly on corrupt input.
