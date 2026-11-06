//! Stage 3 toxicflow H6: the CI workflow must run the fmt and clippy gates
//! alongside the test suite.
//!
//! These tests read the workflow file itself, so a future edit that drops a
//! gate fails the suite instead of failing silently in CI.

/// The workflow file, embedded at compile time so the tests work no matter
/// which directory the test binary runs from.
const CI_YML: &str = include_str!("../.github/workflows/ci.yml");

#[test]
fn stage3_toxicflow_6_ci_fmt_gate_present() {
    assert!(
        CI_YML.contains("cargo fmt --check"),
        "ci.yml must run `cargo fmt --check`:\n{CI_YML}"
    );
}

#[test]
fn stage3_toxicflow_6_ci_clippy_gate_present() {
    assert!(
        CI_YML.contains("cargo clippy --all-targets -- -D warnings"),
        "ci.yml must run `cargo clippy --all-targets -- -D warnings`:\n{CI_YML}"
    );
}

#[test]
fn stage3_toxicflow_6_ci_test_gate_present() {
    assert!(
        CI_YML.contains("cargo test"),
        "ci.yml must keep running `cargo test`:\n{CI_YML}"
    );
}

#[test]
fn stage3_toxicflow_6_ci_gates_run_on_pull_request() {
    assert!(
        CI_YML.contains("pull_request"),
        "ci.yml must trigger on pull_request so the gates run on PRs:\n{CI_YML}"
    );
}
