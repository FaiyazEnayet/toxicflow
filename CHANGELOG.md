# Changelog

## [v1.2.0] - 2026-10-17
### Added
- 28 market-data replay tests proving the order book stays correct under out-of-order messages, duplicates, dropped messages, disconnects/reconnects, bad snapshots, and extreme activity bursts. Test count: 23 → 51.
- `OrderBook::is_crossed()` detects crossed/locked books (bid >= ask).

### Fixed
- `merge_snapshot` was rejecting valid snapshots, forcing unnecessary re-downloads. Now follows the documented exchange procedure.
- `read_trades` and `read_quotes` no longer panic on parquet files with a missing column or wrong column type. The column helpers return a descriptive error instead, so `analyze` and `export` exit cleanly on corrupt input.
