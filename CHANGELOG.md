# Changelog

## [Unreleased]
### Docs
- Centered the README header: uniform dark badge labels and three action buttons in the dashboard palette (Open Demo, Explore Method, Run Locally).

### Added
- CI fmt and clippy gates (Stage 3 H6): `.github/workflows/ci.yml` now runs `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings` alongside `cargo test --locked` (the test step now pins the lockfile, and the toolchain step requests the rustfmt and clippy components explicitly). New fixtures in `tests/stage3_toxicflow_6.rs` assert the gates are present in the workflow file and that it triggers on pull requests. Adding the gates required clearing four pre-existing clippy errors in test code, all found on pristine main: two `needless_range_loop` sites in the N1 variance fixture (`src/predict.rs`, rewritten as iterator loops with identical assertions), `items_after_test_module` in `src/main.rs` (the `mod tests` block moved unchanged to the end of the file), and `await_holding_lock` in `src/recorder.rs` (a scoped `#[allow]` on the proxy env var test, which must hold the lock across the await to keep process-global env vars stable). Formatting was already clean on main. No production behavior changes. Test count: 119 -> 123. No version bump.
