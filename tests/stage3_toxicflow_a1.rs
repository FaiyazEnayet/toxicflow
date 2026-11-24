//! A1: typed L1/L2 as-of event-sourced microstructure replay engine.
//!
//! Known-answer fixtures for the new `toxicflow::replay` module. The
//! engine replays a journal in sequence order, emits L1 touch and L2
//! top-of-book features only while its typed validity is Valid, cites
//! only earlier observation IDs in every feature lineage, records
//! every resync interval, and logs L2 unavailability explicitly.
//! The reference reducer is `storage::reconstruct_session`: on any
//! journal the reference accepts, the engine must reproduce its
//! trades and quotes exactly.

use std::process::Command;

use toxicflow::replay::{replay, InvalidReason, ReplayEngine, Validity};
use toxicflow::storage::{self, JournalRecord};

fn snap(id: i64) -> JournalRecord {
    JournalRecord::Snapshot {
        last_update_id: id,
        bids: vec![(100.0, 5.0), (99.9, 3.0)],
        asks: vec![(100.1, 4.0), (100.2, 2.0)],
    }
}

fn depth(
    seq_time: i64,
    u_first: i64,
    u_last: i64,
    bids: &[(f64, f64)],
    asks: &[(f64, f64)],
) -> JournalRecord {
    JournalRecord::Depth {
        time_ms: seq_time,
        u_first,
        u_last,
        bids: bids.to_vec(),
        asks: asks.to_vec(),
    }
}

fn trade(time_ms: i64, price: f64, qty: f64) -> JournalRecord {
    JournalRecord::Trade {
        time_ms,
        price,
        qty,
        is_buyer_maker: false,
    }
}

/// A journal with a gap and a resync: depth before the first
/// snapshot, a clean stretch, a skipped update, then a fresh
/// snapshot that restores validity.
fn gapped_journal() -> Vec<JournalRecord> {
    vec![
        JournalRecord::SessionStart {
            symbol: "BTCUSDT".to_string(),
        },
        // seq 1: depth before any snapshot: invalid, no signal.
        depth(1_700_000_000_000, 101, 101, &[(100.0, 6.0)], &[]),
        // seq 2: first snapshot restores validity.
        snap(100),
        // seq 3: clean depth, emits features.
        depth(1_700_000_000_100, 101, 101, &[(100.0, 6.0)], &[]),
        // seq 4: trade while valid.
        trade(1_700_000_000_150, 100.05, 0.5),
        // seq 5: update 102 never arrives, so 103 opens a gap.
        depth(1_700_000_000_200, 103, 103, &[(100.0, 7.0)], &[]),
        // seq 6: fresh snapshot resyncs; the gap event is superseded.
        snap(102),
        // seq 7: clean depth after the resync, emits features.
        depth(1_700_000_000_300, 103, 103, &[(100.0, 7.0)], &[]),
        JournalRecord::SessionEnd { records: 7 },
    ]
}

fn assert_lineage_earlier(outcome: &toxicflow::replay::ReplayOutcome) {
    for f in &outcome.touches {
        for s in &f.lineage.sources {
            assert!(
                *s < f.as_of.0,
                "touch lineage cites seq {s}, not earlier than as-of {}",
                f.as_of.0
            );
        }
    }
    for f in &outcome.imbalances {
        for s in &f.lineage.sources {
            assert!(
                *s < f.as_of.0,
                "imbalance lineage cites seq {s}, not earlier than as-of {}",
                f.as_of.0
            );
        }
    }
}

#[test]
fn a1_zero_signal_while_invalid_and_l2_logged() {
    let outcome = replay(&gapped_journal());
    // No feature was emitted while the reducer was invalid.
    assert_eq!(outcome.features_while_invalid, 0);
    // Exactly the two clean depths emitted features: seq 3 and seq 7.
    assert_eq!(
        outcome
            .touches
            .iter()
            .map(|f| f.as_of.0)
            .collect::<Vec<_>>(),
        vec![3, 7]
    );
    assert_eq!(
        outcome
            .imbalances
            .iter()
            .map(|f| f.as_of.0)
            .collect::<Vec<_>>(),
        vec![3, 7]
    );
    // The trade is an observation, not a derived signal: it is kept.
    assert_eq!(outcome.trades.len(), 1);
    // Two resync intervals: awaiting the first snapshot, then the gap.
    assert_eq!(outcome.resyncs.len(), 2);
    assert_eq!(outcome.resyncs[0].start_seq, 0);
    assert_eq!(outcome.resyncs[0].end_seq, Some(2));
    assert_eq!(outcome.resyncs[0].reason, InvalidReason::AwaitingSnapshot);
    assert_eq!(outcome.resyncs[1].start_seq, 5);
    assert_eq!(outcome.resyncs[1].end_seq, Some(6));
    assert!(matches!(
        outcome.resyncs[1].reason,
        InvalidReason::SequenceGap { .. }
    ));
    // L2 unavailability is logged explicitly, with the observation id.
    assert!(
        outcome.log.iter().any(|l| l.contains("L2 unavailable")),
        "log must name L2 unavailability, got: {:?}",
        outcome.log
    );
    assert!(
        outcome.log.iter().any(|l| l.contains("seq 1")),
        "log must name the observation id, got: {:?}",
        outcome.log
    );
    assert!(matches!(outcome.validity, Validity::Valid));
    assert_lineage_earlier(&outcome);
}

#[test]
fn a1_lineage_cites_only_earlier_ids_and_values_are_known() {
    let outcome = replay(&gapped_journal());
    assert_lineage_earlier(&outcome);
    // seq 3 feature: snapshot seq 2 plus nothing else before it.
    assert_eq!(outcome.touches[0].lineage.sources, vec![2]);
    assert_eq!(outcome.touches[0].bid_px, 100.0);
    assert_eq!(outcome.touches[0].ask_px, 100.1);
    assert!((outcome.touches[0].mid - 100.05).abs() < 1e-9);
    assert!((outcome.touches[0].spread - 0.1).abs() < 1e-9);
    // seq 7 feature: built on the resync snapshot at seq 6 only.
    assert_eq!(outcome.touches[1].lineage.sources, vec![6]);
    assert_eq!(outcome.touches[1].bid_px, 100.0);
    // L2 imbalance at seq 3: bids 6+3=9, asks 4+2=6 over the top 5.
    let imb = &outcome.imbalances[0];
    assert_eq!(imb.levels, 5);
    assert!((imb.imbalance - 3.0 / 15.0).abs() < 1e-9);
    assert_eq!(imb.lineage.sources, vec![2]);
}

#[test]
fn a1_as_of_queries_return_latest_feature_at_or_before_seq() {
    let mut engine = ReplayEngine::new();
    for (seq, record) in gapped_journal().iter().enumerate() {
        engine.apply(seq as u64, record);
    }
    // Before the first feature there is nothing as of that point.
    assert!(engine.touch_as_of(2).is_none());
    // As of seq 4 the latest feature is the seq 3 one.
    assert_eq!(engine.touch_as_of(4).unwrap().as_of.0, 3);
    // The gap stretch still answers with the last valid feature.
    assert_eq!(engine.touch_as_of(6).unwrap().as_of.0, 3);
    // As of the end, the resync feature.
    assert_eq!(engine.touch_as_of(7).unwrap().as_of.0, 7);
    assert_eq!(engine.imbalance_as_of(7).unwrap().as_of.0, 7);
    // An L2 query while invalid returns nothing: no L2 while invalid.
    assert!(engine.imbalance_as_of(1).is_none());
}

#[test]
fn a1_crossed_book_yields_no_signal_and_logs_l2_unavailable() {
    let records = vec![
        JournalRecord::SessionStart {
            symbol: "BTCUSDT".to_string(),
        },
        JournalRecord::Snapshot {
            last_update_id: 100,
            bids: vec![(100.2, 5.0)],
            asks: vec![(100.1, 4.0)],
        },
        depth(1_700_000_000_000, 101, 101, &[(100.2, 6.0)], &[]),
        JournalRecord::SessionEnd { records: 3 },
    ];
    let outcome = replay(&records);
    assert_eq!(outcome.touches.len(), 0);
    assert_eq!(outcome.imbalances.len(), 0);
    assert_eq!(outcome.features_while_invalid, 0);
    assert!(matches!(
        outcome.validity,
        Validity::Invalid(InvalidReason::CrossedBook)
    ));
    assert!(
        outcome.log.iter().any(|l| l.contains("L2 unavailable")),
        "got: {:?}",
        outcome.log
    );
    // The resync interval never closes: the journal ends still invalid.
    assert_eq!(outcome.resyncs.len(), 1);
    assert_eq!(outcome.resyncs[0].end_seq, None);
}

#[test]
fn a1_engine_matches_reference_reducer_on_fixture() {
    let path = std::path::Path::new("experiments/tf7_fixture/journal.jsonl");
    let records = storage::load_journal_jsonl(path).expect("fixture journal loads");
    let outcome = replay(&records);
    let (ref_trades, ref_quotes) =
        storage::reconstruct_session(&records).expect("reference reducer accepts the fixture");
    assert_eq!(outcome.trades, ref_trades);
    assert_eq!(outcome.quotes, ref_quotes);
    // The fixture journal is one clean session: valid at the end, no
    // resync after the opening snapshot, no signal while invalid.
    assert!(matches!(outcome.validity, Validity::Valid));
    assert_eq!(outcome.features_while_invalid, 0);
    assert_eq!(outcome.trades.len(), 2_975);
    assert_eq!(outcome.quotes.len(), 7_263);
    assert_eq!(outcome.touches.len(), 7_263);
    assert_eq!(outcome.resyncs.len(), 1);
    assert_eq!(outcome.resyncs[0].end_seq, Some(1));
    assert_lineage_earlier(&outcome);
    // Replaying the same records twice gives the same digest.
    let again = replay(&records);
    assert_eq!(outcome.digest, again.digest);
}

#[test]
fn a1_two_clean_processes_agree_on_fixture_digest() {
    let exe = env!("CARGO_BIN_EXE_toxicflow");
    let journal = "experiments/tf7_fixture/journal.jsonl";
    let mut digests = Vec::new();
    for _ in 0..2 {
        let out = Command::new(exe)
            .args(["replay", "--journal", journal])
            .output()
            .expect("replay command runs");
        assert!(
            out.status.success(),
            "replay failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let stdout = String::from_utf8_lossy(&out.stdout);
        let line = stdout
            .lines()
            .find(|l| l.starts_with("digest: "))
            .expect("replay prints a digest line");
        digests.push(line.trim().to_string());
    }
    assert_eq!(digests[0], digests[1], "two clean processes must agree");
}
