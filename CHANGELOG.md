# Changelog

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
