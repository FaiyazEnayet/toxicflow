//! Stage 3 toxicflow P1: repair target semantics and timestamps.
//!
//! The original task labels a row by the sign of the forward return
//! over the next three bars, so a zero return and a negative return
//! share class 0 and an unchanged market can inflate headline
//! accuracy. P1 keeps that task as an immutable baseline and adds two
//! separate labeled tracks over the same decision rows:
//!
//! * next-nonzero: the sign of the first nonzero mid change within a
//!   fixed cutoff, censored when no move occurs inside the cutoff;
//! * three-class: down, flat, or up over the horizon, with flat an
//!   explicit band around zero instead of a side effect of the sign.
//!
//! Every row carries its decision timestamp and every label traces
//! to a concrete future mid observation. The fixtures below are
//! hand-labeled: flats, crossed quotes, delayed messages, gap
//! censoring, and boundary events (last labelable row, cutoff edge,
//! zero flat band).

use std::path::{Path, PathBuf};

use toxicflow::metrics::{bar_status, ofi_bars, ofi_events, Quote};
use toxicflow::predict::{
    build_dataset, build_p1_dataset, p1_grid_from_json, preregistered_p1_grid,
    walk_forward_detailed, walk_forward_three_class, P1Config, P1_DEFAULT_FLAT_EPS,
    P1_DEFAULT_HORIZON, P1_HORIZONS_BARS, P1_NONZERO_CUTOFF_BARS,
};

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

/// Fixture A: flat, then up, then down. 60 bars: 100.0 for bars 0..39,
/// 100.10 for bars 40..49, 99.90 for bars 50..59. Hand-counted at
/// horizon 3, flat band 1e-5, nonzero cutoff 10 (checked against an
/// independent model of the bar and label rules before the test was
/// written): 56 decision rows; baseline 53 labeled, 3 censored,
/// 47 zero forward returns; nonzero 20 labeled (10 up, 10 down),
/// 26 censored with no move, 10 censored past the end; three-class
/// 53 labeled (3 up, 3 down, 47 flat).
fn fixture_a_mids() -> Vec<f64> {
    let mut mids = vec![100.0; 40];
    mids.extend(vec![100.10; 10]);
    mids.extend(vec![99.90; 10]);
    mids
}

fn config_3() -> P1Config {
    P1Config {
        horizon_bars: 3,
        flat_eps: 1e-5,
    }
}

#[test]
fn stage3_toxicflow_1_grid_file_matches_library() {
    let text = std::fs::read_to_string(repo_path("experiments/p1/grid.json"))
        .expect("experiments/p1/grid.json must exist");
    let file_grid = p1_grid_from_json(&text).expect("grid.json must parse");
    let lib_grid = preregistered_p1_grid();
    assert_eq!(file_grid.len(), 9, "grid must hold 9 configs");
    assert_eq!(lib_grid.len(), 9, "library grid must hold 9 configs");
    for (from_file, from_lib) in file_grid.iter().zip(lib_grid.iter()) {
        assert_eq!(from_file, from_lib, "grid file must match the library grid");
    }
    assert_eq!(P1_HORIZONS_BARS, [1, 3, 6]);
    assert_eq!(P1_NONZERO_CUTOFF_BARS, 10);
    assert_eq!(P1_DEFAULT_HORIZON, 3);
    assert_eq!(P1_DEFAULT_FLAT_EPS, 1e-5);
    assert_eq!(
        lib_grid[4],
        config_3(),
        "default config is horizon 3, eps 1e-5"
    );
    assert_eq!(lib_grid[4].id(), "h3-e1e-5");
    let ids: Vec<String> = lib_grid.iter().map(|c| c.id()).collect();
    let mut sorted = ids.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(ids.len(), sorted.len(), "grid ids must be unique");
    let grid: serde_json::Value =
        serde_json::from_str(&text).expect("grid.json must parse as JSON");
    let rule = grid["selection_rule"].as_str().unwrap_or("");
    assert!(
        rule.contains("No test outcome"),
        "grid must state that no test outcome is used for selection: {rule}"
    );
    let censoring = grid["censoring_rule"].as_str().unwrap_or("");
    assert!(
        censoring.contains("censored"),
        "grid must state its censoring rule: {censoring}"
    );
    let sessions = grid["test_sessions"].as_array().expect("test_sessions");
    assert_eq!(sessions.len(), 6, "six frozen test sessions");
    assert_eq!(grid["nonzero_cutoff_bars"], 10);
    assert_eq!(grid["bar_ms"], 10_000);
    assert_eq!(grid["n_splits"], 4);
}

#[test]
fn stage3_toxicflow_1_flat_fixture_labels() {
    let quotes = bar_quotes(&fixture_a_mids());
    let ds = build_p1_dataset(&quotes, &[], 10_000, config_3(), 10);
    let c = &ds.counts;
    assert_eq!(c.n_bars, 60);
    assert_eq!(c.n_decision, 56);
    assert_eq!(c.baseline_labeled, 53);
    assert_eq!(c.baseline_censored, 3);
    assert_eq!(
        c.baseline_zero_fwd, 47,
        "zero returns are counted, not hidden"
    );
    assert_eq!(c.nonzero_labeled, 20);
    assert_eq!(c.nonzero_up, 10);
    assert_eq!(c.nonzero_down, 10);
    assert_eq!(c.nonzero_censored_no_move, 26);
    assert_eq!(c.nonzero_censored_incomplete, 10);
    assert_eq!(c.nonzero_censored_gap, 0);
    assert_eq!(c.three_labeled, 53);
    assert_eq!(c.three_censored_incomplete, 3);
    assert_eq!(c.three_censored_gap, 0);
    assert_eq!(c.three_up, 3);
    assert_eq!(c.three_down, 3);
    assert_eq!(c.three_flat, 47);
    // Reconciliation: every decision row is labeled or censored.
    assert_eq!(c.baseline_labeled + c.baseline_censored, c.n_decision);
    assert_eq!(
        c.nonzero_labeled
            + c.nonzero_censored_no_move
            + c.nonzero_censored_incomplete
            + c.nonzero_censored_gap,
        c.n_decision
    );
    assert_eq!(
        c.three_labeled + c.three_censored_incomplete + c.three_censored_gap,
        c.n_decision
    );

    let row = |bar: usize| ds.rows.iter().find(|r| r.decision_bar == bar).unwrap();
    // Decision row 35 sits in the flat stretch: the old binary label
    // is 0, the next-nonzero label is up at bar 40, three-class is flat.
    let r35 = row(35);
    assert_eq!(r35.decision_time_ms, 350_000);
    assert_eq!(r35.baseline, Some(0));
    assert_eq!(r35.nonzero, Some(1));
    assert_eq!(r35.nonzero_move_bar, Some(40));
    assert_eq!(r35.nonzero_target_mid, Some(100.10));
    assert_eq!(r35.nonzero_target_time_ms, Some(400_000));
    assert_eq!(r35.three, Some(1));
    assert_eq!(r35.three_target_mid, Some(100.0));
    // Decision row 47 sits at 100.10 with the drop at bar 50.
    let r47 = row(47);
    assert_eq!(r47.baseline, Some(0));
    assert_eq!(r47.nonzero, Some(0));
    assert_eq!(r47.nonzero_move_bar, Some(50));
    assert_eq!(r47.nonzero_target_mid, Some(99.90));
    assert_eq!(r47.three, Some(0));
    assert_eq!(r47.three_target_mid, Some(99.90));
    // Boundary: row 49 is the last row with a complete nonzero window
    // and still catches the bar 50 move; row 50 has no complete
    // nonzero window, but still has a 3-bar target (50 + 3 < 60).
    assert_eq!(row(49).nonzero, Some(0));
    assert_eq!(row(49).nonzero_move_bar, Some(50));
    assert!(row(50).nonzero.is_none());
    assert_eq!(row(50).baseline, Some(0));
    assert_eq!(row(50).three, Some(1));
    // Boundary: row 56 is the last baseline-labeled row, row 57 is not.
    assert!(row(56).baseline.is_some());
    assert!(row(57).baseline.is_none());
    // Three-class up and down rows are exactly the hand-counted sets.
    let ups: Vec<usize> = ds
        .rows
        .iter()
        .filter(|r| r.three == Some(2))
        .map(|r| r.decision_bar)
        .collect();
    assert_eq!(ups, vec![37, 38, 39]);
    let downs: Vec<usize> = ds
        .rows
        .iter()
        .filter(|r| r.three == Some(0))
        .map(|r| r.decision_bar)
        .collect();
    assert_eq!(downs, vec![47, 48, 49]);
}

#[test]
fn stage3_toxicflow_1_all_flat_market() {
    // A market that never moves: the old binary task scores every
    // labeled row 0, the nonzero track labels nothing, and the
    // three-class track calls every row flat. The counts say so.
    let quotes = bar_quotes(&vec![100.0; 60]);
    let ds = build_p1_dataset(&quotes, &[], 10_000, config_3(), 10);
    let c = &ds.counts;
    assert_eq!(c.n_decision, 56);
    assert_eq!(c.baseline_labeled, 53);
    assert_eq!(c.baseline_zero_fwd, 53);
    assert_eq!(c.nonzero_labeled, 0);
    assert_eq!(c.nonzero_censored_no_move, 46);
    assert_eq!(c.nonzero_censored_incomplete, 10);
    assert_eq!(c.three_labeled, 53);
    assert_eq!(c.three_flat, 53);
    assert_eq!(c.three_up, 0);
    assert_eq!(c.three_down, 0);
}

#[test]
fn stage3_toxicflow_1_crossed_quotes() {
    // Bars 40..44 are crossed (bid above ask). The builder must
    // finish without panicking, and labels still trace to the
    // observed mid of the crossed bars.
    let mut quotes = bar_quotes(&vec![100.0; 40]);
    for i in 40..45 {
        quotes.push(Quote {
            time_ms: i as i64 * 10_000 + 1_000,
            bid_px: 100.10,
            bid_sz: 5.0,
            ask_px: 100.00,
            ask_sz: 5.0,
        });
    }
    let ds = build_p1_dataset(&quotes, &[], 10_000, config_3(), 10);
    let c = &ds.counts;
    assert_eq!(c.n_bars, 45);
    assert_eq!(c.n_decision, 41);
    assert_eq!(c.baseline_labeled, 38);
    assert_eq!(c.baseline_censored, 3);
    let r37 = ds.rows.iter().find(|r| r.decision_bar == 37).unwrap();
    assert_eq!(r37.baseline, Some(1));
    assert_eq!(r37.baseline_target_mid, Some(100.05));
    assert_eq!(r37.baseline_target_time_ms, Some(400_000));
    assert_eq!(r37.nonzero, Some(1));
    assert_eq!(r37.nonzero_move_bar, Some(40));
    assert_eq!(r37.nonzero_target_mid, Some(100.05));
    assert!(r37.features.iter().all(|f| f.is_finite()));
    let ups: Vec<usize> = ds
        .rows
        .iter()
        .filter(|r| r.three == Some(2))
        .map(|r| r.decision_bar)
        .collect();
    assert_eq!(ups, vec![37, 38, 39]);
}

#[test]
fn stage3_toxicflow_1_delayed_message_prefix_invariant() {
    // A quote for bar 41 arrives after the quotes for bars 42 and 43.
    // Bar 41 also carries an on-time quote, so the late quote is the
    // last arrival in its bar and its mid is the bar mid. Decision
    // rows at or before bar 38 must be bit-identical with and without
    // the late quote: a delayed message never rewrites history.
    // Bars 0..40 flat at 100.0, bars 42..59 at 100.20; bar 41 carries
    // an on-time quote at 100.15 and a late quote at 100.10.
    let mut mids = vec![100.0; 41];
    mids.extend(vec![100.20; 19]);
    let mut quotes = bar_quotes(&mids);
    // On-time bar 41 quote at mid 100.15, then the late bar 41 quote.
    let on_time_41 = quote(411_000, 100.15);
    let late_41 = quote(415_000, 100.10);
    // Rebuild in arrival order: bars 0..40, on-time 41, bars 42, 43,
    // late 41, bars 44..59.
    let mut arrival: Vec<Quote> = quotes.drain(..41).collect();
    arrival.push(on_time_41);
    quotes.drain(..1); // drop the original bar 41 quote
    arrival.extend(quotes.drain(..2)); // bars 42, 43
    arrival.push(late_41);
    arrival.extend(quotes); // bars 44..59
    let ds = build_p1_dataset(&arrival, &[], 10_000, config_3(), 10);
    let r38 = ds.rows.iter().find(|r| r.decision_bar == 38).unwrap();
    assert_eq!(r38.baseline, Some(1));
    assert_eq!(r38.baseline_target_mid, Some(100.10));
    assert_eq!(r38.baseline_target_time_ms, Some(410_000));
    assert_eq!(r38.nonzero, Some(1));
    assert_eq!(r38.nonzero_move_bar, Some(41));
    assert_eq!(r38.nonzero_target_mid, Some(100.10));
    assert_eq!(r38.three, Some(2));
    let r39 = ds.rows.iter().find(|r| r.decision_bar == 39).unwrap();
    assert_eq!(r39.baseline_target_mid, Some(100.20));

    // The same quotes without the late message: rows up to bar 38
    // must carry bit-identical features.
    let mut without_late: Vec<Quote> = arrival.clone();
    without_late.retain(|q| q.time_ms != 415_000);
    let ds_prefix = build_p1_dataset(&without_late, &[], 10_000, config_3(), 10);
    for bar in 4..=38 {
        let a = ds.rows.iter().find(|r| r.decision_bar == bar).unwrap();
        let b = ds_prefix
            .rows
            .iter()
            .find(|r| r.decision_bar == bar)
            .unwrap();
        assert_eq!(
            a.features, b.features,
            "delayed message rewrote decision row {bar}"
        );
    }
}

#[test]
fn stage3_toxicflow_1_flat_band_boundary() {
    // Bars 40..59 sit at 100.001, a 1.0000000000065512e-05 relative
    // move from 100.0. At a zero flat band that is up; at a 3e-5 band
    // the same row is flat. A zero move at a zero band is flat.
    let mut mids = vec![100.0; 40];
    mids.extend(vec![100.001; 20]);
    let quotes = bar_quotes(&mids);
    let eps0 = build_p1_dataset(
        &quotes,
        &[],
        10_000,
        P1Config {
            horizon_bars: 3,
            flat_eps: 0.0,
        },
        10,
    );
    let r37 = eps0.rows.iter().find(|r| r.decision_bar == 37).unwrap();
    assert_eq!(r37.three, Some(2), "nonzero move is not flat at eps 0");
    let r4 = eps0.rows.iter().find(|r| r.decision_bar == 4).unwrap();
    assert_eq!(r4.three, Some(1), "exact zero is flat even at eps 0");
    let eps3 = build_p1_dataset(
        &quotes,
        &[],
        10_000,
        P1Config {
            horizon_bars: 3,
            flat_eps: 3e-5,
        },
        10,
    );
    let r37b = eps3.rows.iter().find(|r| r.decision_bar == 37).unwrap();
    assert_eq!(r37b.three, Some(1), "the same move is flat at eps 3e-5");
}

#[test]
fn stage3_toxicflow_1_gap_censors_target_windows() {
    // Bars 30..34 have no quotes. Bars 35..59 resume at 100.20.
    // No decision row after the gap can form: every trailing 30-bar
    // feature window still touches the gap. Rows before the gap whose
    // target window touches it are censored as gaps, not as moves.
    let mut mids = vec![100.0; 20];
    mids.extend(vec![100.10; 10]); // bars 20..29
    mids.extend(vec![100.20; 25]); // bars 35..59
    let mut quotes = Vec::new();
    for (i, &m) in mids.iter().enumerate() {
        let bar = if i < 30 { i } else { i + 5 };
        quotes.push(quote(bar as i64 * 10_000 + 1_000, m));
    }
    let ds = build_p1_dataset(&quotes, &[], 10_000, config_3(), 10);
    let c = &ds.counts;
    assert_eq!(c.n_bars, 60);
    assert_eq!(c.n_decision, 26, "only rows 4..29 form");
    assert_eq!(c.baseline_labeled, 23);
    assert_eq!(c.baseline_censored, 3);
    assert_eq!(c.baseline_zero_fwd, 20);
    assert_eq!(c.nonzero_labeled, 10);
    assert_eq!(c.nonzero_up, 10);
    assert_eq!(c.nonzero_censored_no_move, 6);
    assert_eq!(c.nonzero_censored_gap, 10, "gap rows are gaps, not no-move");
    assert_eq!(c.nonzero_censored_incomplete, 0);
    assert_eq!(c.three_labeled, 23);
    assert_eq!(c.three_flat, 20);
    assert_eq!(c.three_up, 3);
    assert_eq!(c.three_down, 0);
    assert_eq!(c.three_censored_gap, 3);
    assert_eq!(c.three_censored_incomplete, 0);
    assert!(!ds.rows.iter().any(|r| r.decision_bar >= 30));
}

/// Deterministic predictable stream: 700 bars, drift regimes of 60
/// bars (up, down, flat), OFI-implied bid/ask stacks. Mirrors the
/// generator in the library tests so the integration test stays
/// independent of it. The flat regime holds the mid exactly constant,
/// so the three-class track has genuine flat rows.
fn synth_quotes(n: usize) -> Vec<Quote> {
    let mut quotes = Vec::new();
    let mut mid = 100.0;
    let mut s: u64 = 42;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s as f64) / (u64::MAX as f64) - 0.5
    };
    for i in 0..n {
        let t = i as i64 * 10_000 + 1_000;
        let drift = match i % 180 {
            0..60 => 0.02,
            60..120 => -0.02,
            _ => 0.0,
        };
        let bq = 5.0 + if drift > 0.0 { 4.0 } else { rnd() };
        let aq = 5.0 + if drift < 0.0 { 4.0 } else { rnd() };
        quotes.push(Quote {
            time_ms: t,
            bid_px: mid - 0.05,
            bid_sz: bq.max(0.5),
            ask_px: mid + 0.05,
            ask_sz: aq.max(0.5),
        });
        mid += drift + if drift == 0.0 { 0.0 } else { rnd() * 0.01 };
    }
    quotes
}

#[test]
fn stage3_toxicflow_1_baseline_matches_build_dataset() {
    // The preserved baseline: P1 decision rows relabeled by the old
    // rule must equal build_dataset output row for row, features
    // bit-identical and targets equal, and the walk-forward headline
    // must be unchanged.
    let quotes = synth_quotes(700);
    let p1 = build_p1_dataset(&quotes, &[], 10_000, config_3(), 10);
    let base = build_dataset(&quotes, &[], 10_000, 3);
    let p1_base = p1.baseline_bars();
    assert_eq!(p1_base.len(), base.len());
    for (a, b) in p1_base.iter().zip(base.iter()) {
        assert_eq!(a.time_ms, b.time_ms);
        assert_eq!(a.target, b.target);
        assert_eq!(a.ofi.to_bits(), b.ofi.to_bits());
        assert_eq!(a.ofi_z.to_bits(), b.ofi_z.to_bits());
        assert_eq!(a.imbalance.to_bits(), b.imbalance.to_bits());
        assert_eq!(a.spread_z.to_bits(), b.spread_z.to_bits());
        assert_eq!(a.signed_vol.to_bits(), b.signed_vol.to_bits());
        assert_eq!(a.spread_degenerate, b.spread_degenerate);
    }
    let rep_a = walk_forward_detailed(&p1_base, 4, 3).expect("p1 baseline eval");
    let rep_b = walk_forward_detailed(&base, 4, 3).expect("legacy eval");
    assert_eq!(rep_a.report.accuracy, rep_b.report.accuracy);
    assert_eq!(rep_a.report.brier, rep_b.report.brier);
    assert_eq!(rep_a.report.log_loss, rep_b.report.log_loss);
    assert_eq!(rep_a.dataset_sha256, rep_b.dataset_sha256);
}

#[test]
fn stage3_toxicflow_1_every_target_traces_to_observed_mid() {
    let quotes = synth_quotes(700);
    let ds = build_p1_dataset(&quotes, &[], 10_000, config_3(), 10);
    let events = ofi_events(&quotes);
    let bars = ofi_bars(&quotes, &events, 10_000);
    let status = bar_status(&quotes, 10_000);
    assert_eq!(bars.len(), 700);
    let mid_at = |time_ms: i64| bars[(time_ms / 10_000) as usize].2;
    let observed_at = |time_ms: i64| status[(time_ms / 10_000) as usize].observed;
    for r in &ds.rows {
        assert!(r.decision_time_ms < r.baseline_target_time_ms.unwrap_or(i64::MAX));
        if let (Some(mid), Some(t)) = (r.baseline_target_mid, r.baseline_target_time_ms) {
            assert_eq!(mid, mid_at(t), "baseline target must be a bar mid");
            assert!(observed_at(t), "baseline target bar must be observed");
            assert_eq!(t, bars[r.decision_bar + 3].0);
        }
        if let (Some(mid), Some(t), Some(mb)) = (
            r.nonzero_target_mid,
            r.nonzero_target_time_ms,
            r.nonzero_move_bar,
        ) {
            assert_eq!(mid, mid_at(t), "nonzero target must be a bar mid");
            assert!(observed_at(t), "nonzero target bar must be observed");
            assert_eq!(t, bars[mb].0);
            assert_ne!(mid, bars[r.decision_bar].2, "the move bar moved");
            for k in r.decision_bar + 1..mb {
                assert_eq!(
                    bars[k].2, bars[r.decision_bar].2,
                    "no earlier nonzero move was skipped"
                );
            }
        }
        if let (Some(mid), Some(t)) = (r.three_target_mid, r.three_target_time_ms) {
            assert_eq!(mid, mid_at(t), "three-class target must be a bar mid");
            assert!(observed_at(t), "three-class target bar must be observed");
        }
    }
}

#[test]
fn stage3_toxicflow_1_counts_reconcile_and_priors_train_only() {
    let quotes = synth_quotes(700);
    let ds = build_p1_dataset(&quotes, &[], 10_000, config_3(), 10);
    let c = &ds.counts;
    assert_eq!(c.baseline_labeled + c.baseline_censored, c.n_decision);
    assert_eq!(
        c.nonzero_labeled
            + c.nonzero_censored_no_move
            + c.nonzero_censored_incomplete
            + c.nonzero_censored_gap,
        c.n_decision
    );
    assert_eq!(
        c.three_labeled + c.three_censored_incomplete + c.three_censored_gap,
        c.n_decision
    );
    // Binary nonzero track: OOS plus unscored reconciles to labeled.
    let nz = ds.nonzero_bars();
    assert_eq!(nz.len(), c.nonzero_labeled);
    let nz_detail = walk_forward_detailed(&nz, 4, 10).expect("nonzero eval");
    assert_eq!(
        nz_detail.report.n_oos + nz_detail.report.n_unscored,
        c.nonzero_labeled
    );
    assert!(nz_detail.report.log_loss.is_finite());
    assert!((0.0..=1.0).contains(&nz_detail.report.brier));
    // Three-class track: confusion tables and train priors reconcile.
    let rows = ds.three_class_rows();
    assert_eq!(rows.len(), c.three_labeled);
    let rep = walk_forward_three_class(&rows, 4, 3).expect("three-class eval");
    assert_eq!(rep.n_oos + rep.n_unscored, c.three_labeled);
    let conf_sum: usize = rep.confusion.iter().flat_map(|r| r.iter()).sum();
    assert_eq!(conf_sum, rep.n_oos);
    assert_eq!(rep.n_down_oos + rep.n_flat_oos + rep.n_up_oos, rep.n_oos);
    assert!(rep.log_loss.is_finite());
    assert!((0.0..=2.0).contains(&rep.brier));
    for f in &rep.per_fold {
        assert_eq!(f.train_down + f.train_flat + f.train_up, f.n_train);
        assert_eq!(f.n_test, f.n_down + f.n_flat + f.n_up);
        let conf_fold: usize = f.confusion.iter().flat_map(|r| r.iter()).sum();
        assert_eq!(conf_fold, f.n_test);
        // The train majority label is the argmax of the train-only
        // counts, lowest index on ties: no test outcome feeds it.
        let counts = [f.train_down, f.train_flat, f.train_up];
        let mut best = 0u8;
        for k in 1..3 {
            if counts[k] > counts[best as usize] {
                best = k as u8;
            }
        }
        assert_eq!(f.train_majority_label, best);
    }
    // On this predictable stream both new tracks beat their
    // train-majority baselines, and the three-class model separates.
    assert!(nz_detail.report.accuracy > nz_detail.report.baseline);
    assert!(rep.accuracy > rep.baseline);
    assert!(rep.balanced_accuracy > 0.5);
}
