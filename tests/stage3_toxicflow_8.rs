//! Stage 3 toxicflow N5: censored next-nonzero price-move hazard baseline.
//!
//! The hazard baseline freezes a competing-event / next-move-time target
//! over the same decision rows P1 labels: an event is the first nonzero
//! mid change within a frozen 10-bar cutoff, up and down are competing
//! events, and rows are censored with a recorded reason when no move is
//! observed (complete window, window past the last bar, or window
//! touching an unobserved bar). Every row carries valid
//! t_observe <= t_decision <= t_outcome, or explicit censoring.
//!
//! The fixtures below are hand-labeled before the implementation
//! exists: flats, a planted gap, and a move visible only across a gap
//! (which must be censored, never labeled).

use std::path::{Path, PathBuf};

use toxicflow::hazard::{
    build_hazard_dataset, hazard_grid_from_json, preregistered_hazard_grid, CensorReason,
};
use toxicflow::metrics::Quote;
use toxicflow::predict::{build_p1_dataset, P1_DEFAULT_CONFIG, P1_NONZERO_CUTOFF_BARS};

fn repo_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

/// One quote per 10s bar at a scripted mid, constant one-tick spread.
fn quote(time_ms: i64, mid: f64) -> Quote {
    Quote {
        time_ms,
        bid_px: mid - 0.05,
        bid_sz: 5.0,
        ask_px: mid + 0.05,
        ask_sz: 5.0,
    }
}

fn bar_quotes(mids: &[f64]) -> Vec<Quote> {
    mids.iter()
        .enumerate()
        .map(|(i, &m)| quote(i as i64 * 10_000 + 1_000, m))
        .collect()
}

/// Fixture A: the P1 fixture A mids. 60 bars: 100.0 for bars 0..39,
/// 100.10 for bars 40..49, 99.90 for bars 50..59. Hand-counted at
/// cutoff 10 (checked against the P1 counts for the same mids before
/// the test was written): 56 decision rows; 20 events (10 up, 10
/// down); 36 censored (26 no move, 10 past the end, 0 gap).
fn fixture_a_mids() -> Vec<f64> {
    let mut mids = vec![100.0; 40];
    mids.extend(vec![100.10; 10]);
    mids.extend(vec![99.90; 10]);
    mids
}

/// Fixture B: a planted gap. 60 bars with no quotes on bars 20..24;
/// mids 100.0 through bar 39, 100.10 from bar 40. Hand-counted at
/// cutoff 10: decision rows are bars 4..19 and 54..59 only (22 rows);
/// 0 events; 22 censored (6 no move, 6 past the end, 10 gap).
fn fixture_b_quotes() -> Vec<Quote> {
    let mut qs = Vec::new();
    for i in 0..60 {
        if (20..25).contains(&i) {
            continue;
        }
        let mid = if i < 40 { 100.0 } else { 100.10 };
        qs.push(quote(i as i64 * 10_000 + 1_000, mid));
    }
    qs
}

/// Fixture C: a move visible only across a gap. 60 bars with no
/// quotes on bars 30..32; mids 100.0 through bar 29, 100.10 from bar
/// 33. Hand-counted at cutoff 10: decision rows are bars 4..29 only
/// (26 rows); 0 events, even though the move at bar 33 is visible;
/// 26 censored (16 no move, 0 past the end, 10 gap).
fn fixture_c_quotes() -> Vec<Quote> {
    let mut qs = Vec::new();
    for i in 0..60 {
        if (30..33).contains(&i) {
            continue;
        }
        let mid = if i < 33 { 100.0 } else { 100.10 };
        qs.push(quote(i as i64 * 10_000 + 1_000, mid));
    }
    qs
}

#[test]
fn n5_fixture_a_counts() {
    let ds = build_hazard_dataset(&bar_quotes(&fixture_a_mids()), 10_000, 10);
    let c = &ds.counts;
    assert_eq!(c.n_bars, 60);
    assert_eq!(c.n_decision, 56);
    assert_eq!(c.n_events, 20);
    assert_eq!(c.n_up, 10);
    assert_eq!(c.n_down, 10);
    assert_eq!(c.n_censored, 36);
    assert_eq!(c.censored_no_move, 26);
    assert_eq!(c.censored_incomplete, 10);
    assert_eq!(c.censored_gap, 0);
    assert_eq!(c.n_events + c.n_censored, c.n_decision);
}

#[test]
fn n5_fixture_b_gap_counts() {
    let ds = build_hazard_dataset(&fixture_b_quotes(), 10_000, 10);
    let c = &ds.counts;
    assert_eq!(c.n_bars, 60);
    assert_eq!(c.n_decision, 22);
    assert_eq!(c.n_events, 0);
    assert_eq!(c.n_censored, 22);
    assert_eq!(c.censored_no_move, 6);
    assert_eq!(c.censored_incomplete, 6);
    assert_eq!(c.censored_gap, 10);
}

#[test]
fn n5_fixture_c_move_across_gap_censored() {
    let ds = build_hazard_dataset(&fixture_c_quotes(), 10_000, 10);
    let c = &ds.counts;
    assert_eq!(c.n_decision, 26);
    // The move at bar 33 is visible, but its timing crosses the
    // unobserved bars 30..32, so it must never label a row.
    assert_eq!(c.n_events, 0, "0 gap-crossing labels");
    assert_eq!(c.n_up, 0);
    assert_eq!(c.n_down, 0);
    assert_eq!(c.censored_no_move, 16);
    assert_eq!(c.censored_incomplete, 0);
    assert_eq!(c.censored_gap, 10);
}

#[test]
fn n5_time_ordering_invariants() {
    for qs in [
        bar_quotes(&fixture_a_mids()),
        fixture_b_quotes(),
        fixture_c_quotes(),
    ] {
        let ds = build_hazard_dataset(&qs, 10_000, 10);
        for r in &ds.rows {
            assert!(
                r.t_observe_ms <= r.t_decision_ms,
                "t_observe <= t_decision: {r:?}"
            );
            assert_eq!(r.t_decision_ms - r.t_observe_ms, 10_000);
            if r.event {
                assert!(r.t_decision_ms <= r.t_outcome_ms.unwrap());
                assert!(r.wait_bars >= 1);
                assert!(r.direction.is_some());
                assert_eq!(r.censor, None);
                assert_eq!(r.move_bar.unwrap() - r.decision_bar, r.wait_bars);
                assert_eq!(
                    r.t_outcome_ms.unwrap() - r.t_decision_ms,
                    (r.wait_bars as i64 - 1) * 10_000,
                    "move first visible at the start of the move bar"
                );
            } else {
                assert_eq!(r.t_outcome_ms, None);
                assert_eq!(r.direction, None);
                assert!(r.censor.is_some());
                assert!(r.t_decision_ms <= r.t_censor_ms);
                assert_eq!(
                    r.t_censor_ms - r.t_decision_ms,
                    r.wait_bars as i64 * 10_000,
                    "censoring time is the end of the last observed at-risk bar"
                );
                match r.censor.unwrap() {
                    CensorReason::NoMove => assert_eq!(r.wait_bars, 10),
                    CensorReason::Incomplete => assert!(r.wait_bars < 10),
                    CensorReason::Gap => assert!(r.wait_bars < 10),
                }
            }
        }
    }
}

#[test]
fn n5_no_event_window_crosses_gap() {
    // On every fixture, no event row's outcome window touches an
    // unobserved bar: re-derive observation status and check the
    // bars strictly between the decision bar and the move bar.
    for qs in [
        bar_quotes(&fixture_a_mids()),
        fixture_b_quotes(),
        fixture_c_quotes(),
    ] {
        let ds = build_hazard_dataset(&qs, 10_000, 10);
        let status = toxicflow::metrics::bar_status(&qs, 10_000);
        for r in ds.rows.iter().filter(|r| r.event) {
            let j = r.move_bar.unwrap();
            assert!(
                status[r.decision_bar + 1..=j].iter().all(|s| s.observed),
                "event window crosses a gap: {r:?}"
            );
        }
    }
}

#[test]
fn n5_p1_nonzero_agreement() {
    // The hazard rows are the same decision rows P1 labels, and the
    // event rows are exactly P1's next-nonzero labeled rows.
    let qs = bar_quotes(&fixture_a_mids());
    let p1 = build_p1_dataset(&qs, &[], 10_000, P1_DEFAULT_CONFIG, P1_NONZERO_CUTOFF_BARS);
    let hz = build_hazard_dataset(&qs, 10_000, P1_NONZERO_CUTOFF_BARS);
    let p1_bars: Vec<usize> = p1.rows.iter().map(|r| r.decision_bar).collect();
    let hz_bars: Vec<usize> = hz.rows.iter().map(|r| r.decision_bar).collect();
    assert_eq!(hz_bars, p1_bars, "same decision rows as P1");
    let p1_events: Vec<(usize, usize, u8)> = p1
        .rows
        .iter()
        .filter_map(|r| {
            r.nonzero
                .map(|d| (r.decision_bar, r.nonzero_move_bar.unwrap(), d))
        })
        .collect();
    let hz_events: Vec<(usize, usize, u8)> = hz
        .rows
        .iter()
        .filter(|r| r.event)
        .map(|r| (r.decision_bar, r.move_bar.unwrap(), r.direction.unwrap()))
        .collect();
    assert_eq!(hz_events, p1_events, "same events as P1 nonzero track");
    assert_eq!(hz.counts.n_events, p1.counts.nonzero_labeled);
    assert_eq!(hz.counts.n_up, p1.counts.nonzero_up);
    assert_eq!(hz.counts.n_down, p1.counts.nonzero_down);
    assert_eq!(
        hz.counts.censored_no_move,
        p1.counts.nonzero_censored_no_move
    );
    assert_eq!(
        hz.counts.censored_incomplete,
        p1.counts.nonzero_censored_incomplete
    );
    assert_eq!(hz.counts.censored_gap, p1.counts.nonzero_censored_gap);
    // The move timestamps agree with P1's target timestamps.
    for r in hz.rows.iter().filter(|r| r.event) {
        let p = p1
            .rows
            .iter()
            .find(|p| p.decision_bar == r.decision_bar)
            .unwrap();
        assert_eq!(r.t_outcome_ms, p.nonzero_target_time_ms);
    }
}

#[test]
fn n5_grid_file_matches_library() {
    let text = std::fs::read_to_string(repo_path("experiments/n5/grid.json"))
        .expect("experiments/n5/grid.json must exist");
    let file_grid = hazard_grid_from_json(&text).expect("grid.json must parse");
    let lib_grid = preregistered_hazard_grid();
    assert_eq!(file_grid, lib_grid, "grid file must match the library grid");
}

#[test]
fn n5_thin_data_yields_no_rows() {
    let qs = bar_quotes(&[100.0; 20]);
    let ds = build_hazard_dataset(&qs, 10_000, 10);
    assert!(ds.rows.is_empty());
    assert_eq!(ds.counts.n_decision, 0);
}
