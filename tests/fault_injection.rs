//! N6 fault-injection acceptance tests.
//!
//! The harness replays a deterministic depth stream through the production
//! sync machine under four injected communication faults (dropped,
//! duplicate, delayed, and book-crossing events) with frozen seeds. The
//! acceptance bar is the contract: across 20 deterministic mutation
//! traces no signal is ever emitted from an invalid book, the valid
//! reference digest is unchanged, and every fault carries a traceable
//! reason. Faults are never relabeled as clean data.

use std::collections::HashMap;

use toxicflow::booksync::{BookSync, MergeOutcome, SnapshotData};
use toxicflow::direction_manifest::sha256_hex;
use toxicflow::fault_injection::{
    clean_reference_digest, run_trace, run_twenty_traces, signal_for, FaultKind, IntegrityState,
    BASE_SEED, N_EVENTS, TRACE_COUNT,
};

/// Independent ground truth: absolute level updates keyed on raw f64
/// bits. No tick conversion, no BTreeMap, a different code path from the
/// production book, fed the clean in-order stream.
#[derive(Default)]
struct ReferenceBook {
    bids: HashMap<u64, f64>,
    asks: HashMap<u64, f64>,
}

impl ReferenceBook {
    fn apply(&mut self, bids: &[(f64, f64)], asks: &[(f64, f64)]) {
        for (p, q) in bids {
            if *q == 0.0 {
                self.bids.remove(&p.to_bits());
            } else {
                self.bids.insert(p.to_bits(), *q);
            }
        }
        for (p, q) in asks {
            if *q == 0.0 {
                self.asks.remove(&p.to_bits());
            } else {
                self.asks.insert(p.to_bits(), *q);
            }
        }
    }

    fn touch(&self) -> Option<(f64, f64, f64, f64)> {
        let (bp, bq) = self
            .bids
            .iter()
            .max_by(|a, b| a.0.cmp(b.0))
            .map(|(p, q)| (f64::from_bits(*p), *q))?;
        let (ap, aq) = self
            .asks
            .iter()
            .min_by(|a, b| a.0.cmp(b.0))
            .map(|(p, q)| (f64::from_bits(*p), *q))?;
        Some((bp, bq, ap, aq))
    }
}

/// One clean touch from the independent reference book: event time,
/// update id, and the touch quote.
type ReferenceTouch = (i64, i64, (f64, f64, f64, f64));

/// The clean touch sequence from the independent reference book:
/// (event time, update id, touch) per event.
fn reference_touches(seed: u64) -> Vec<ReferenceTouch> {
    let stream = toxicflow::fault_injection::build_clean_stream(N_EVENTS, seed);
    let mut book = ReferenceBook::default();
    book.apply(&stream.snapshot.bids, &stream.snapshot.asks);
    let mut out = Vec::new();
    for ev in &stream.events {
        book.apply(&ev.bids, &ev.asks);
        if let Some(t) = book.touch() {
            out.push((ev.event_ms, ev.u_last, t));
        }
    }
    out
}

/// The reference digest of the clean stream, computed through the
/// independent book instead of the production sync machine.
fn reference_digest() -> String {
    let mut buf = Vec::new();
    for (ms, _, (bp, bq, ap, aq)) in reference_touches(BASE_SEED) {
        buf.extend_from_slice(&ms.to_le_bytes());
        for v in [bp, bq, ap, aq] {
            buf.extend_from_slice(&v.to_bits().to_le_bytes());
        }
    }
    sha256_hex(&buf)
}

#[test]
fn n6_twenty_deterministic_traces_abstain_on_invalid_book() {
    assert_eq!(TRACE_COUNT, 20);
    let (first, first_ledger) = run_twenty_traces();
    let (second, second_ledger) = run_twenty_traces();
    assert_eq!(first.len(), 20);
    assert_eq!(second.len(), 20);

    // Same frozen seeds, same traces, twice.
    assert_eq!(first_ledger.digest(), second_ledger.digest());
    for (a, b) in first.iter().zip(second.iter()) {
        assert_eq!(a.valid_touch_digest(), b.valid_touch_digest());
        assert_eq!(a.invalid_signal_count, b.invalid_signal_count);
    }

    // Every trace: zero signals from an invalid book, and every fault
    // has a traceable reason.
    for trace in &first {
        assert_eq!(
            trace.invalid_signal_count, 0,
            "trace {} ({:?}) emitted a signal from an invalid book",
            trace.trace_id, trace.kind
        );
        assert!(
            !trace.reasons.is_empty(),
            "trace {} ({:?}) has no traceable reason",
            trace.trace_id,
            trace.kind
        );
        for reason in &trace.reasons {
            assert!(!reason.is_empty());
        }
    }

    // Ledger coverage: 5 traces per fault kind, 20 faults total, and
    // every kind abstained at least once except the idempotent duplicate.
    assert_eq!(first_ledger.total_faults(), 20);
    for kind in FaultKind::all() {
        assert_eq!(first_ledger.coverage(kind), 5, "kind {kind:?}");
    }
    assert_eq!(first_ledger.abstentions(), 15);
}

#[test]
fn n6_valid_reference_digest_unchanged() {
    let digest = clean_reference_digest(N_EVENTS, BASE_SEED);
    assert_eq!(digest, clean_reference_digest(N_EVENTS, BASE_SEED));
    assert_eq!(
        digest,
        reference_digest(),
        "production replay and the independent reference book disagree on the clean stream"
    );
}

#[test]
fn n6_no_optimistic_relabeling_of_faults_as_clean() {
    let (_, ledger) = run_twenty_traces();
    assert_eq!(ledger.total_faults(), 20);
    for entry in ledger.entries() {
        // Every injected fault stays recorded as a fault of its own kind,
        // with a non-empty reason. Nothing is recorded as clean.
        assert!(!entry.reason.is_empty());
        assert!(
            !entry.reason.to_lowercase().contains("clean"),
            "fault recorded as clean data: {}",
            entry.reason
        );
    }
    // The idempotent duplicate is still a recorded fault even though the
    // book stayed valid; it is not silently absorbed.
    assert_eq!(ledger.coverage(FaultKind::Duplicate), 5);
}

#[test]
fn n6_dropped_event_forces_abstention() {
    let (trace, _) = run_trace(0, FaultKind::Dropped, BASE_SEED, N_EVENTS);
    assert_eq!(trace.invalid_signal_count, 0);
    assert!(
        trace.abstentions >= 1,
        "a dropped event must open at least one invalid observation"
    );
    assert!(
        trace.reasons.iter().any(|r| r.contains("gap")),
        "dropped event reason must name the sequence gap: {:?}",
        trace.reasons
    );
    // Recovery: observations after the gap are valid again and signals resume.
    let first_invalid = trace
        .observations
        .iter()
        .position(|o| !o.integrity.is_valid())
        .expect("dropped trace has an invalid observation");
    assert!(
        trace.observations[first_invalid + 1..]
            .iter()
            .any(|o| o.integrity.is_valid() && o.signal.is_some()),
        "the book must recover to valid signaled observations after resync"
    );
}

#[test]
fn n6_delayed_event_forces_abstention() {
    let (trace, _) = run_trace(1, FaultKind::Delayed, BASE_SEED + 1, N_EVENTS);
    assert_eq!(trace.invalid_signal_count, 0);
    assert!(
        trace.abstentions >= 1,
        "a delayed event must open at least one invalid observation"
    );
    assert!(
        trace.reasons.iter().any(|r| r.contains("gap")),
        "delayed event reason must name the sequence gap: {:?}",
        trace.reasons
    );
}

#[test]
fn n6_snapshot_crossing_event_forces_abstention() {
    let (trace, _) = run_trace(2, FaultKind::SnapshotCrossing, BASE_SEED + 2, N_EVENTS);
    assert_eq!(trace.invalid_signal_count, 0);
    assert!(
        trace.abstentions >= 1,
        "a book-crossing event must open at least one invalid observation"
    );
    assert!(
        trace.reasons.iter().any(|r| r.contains("cross")),
        "crossing event reason must name the crossed book: {:?}",
        trace.reasons
    );
}

#[test]
fn n6_duplicate_redelivery_is_recorded_not_silent() {
    let seed = BASE_SEED + 3;
    let (trace, ledger) = run_trace(3, FaultKind::Duplicate, seed, N_EVENTS);
    assert_eq!(trace.invalid_signal_count, 0);
    assert_eq!(ledger.total_faults(), 1);
    assert_eq!(ledger.coverage(FaultKind::Duplicate), 1);
    assert!(
        ledger.entries()[0].reason.contains("duplicate"),
        "duplicate reason must name the redelivery: {}",
        ledger.entries()[0].reason
    );
    // An exact redelivery changes nothing: every valid observation still
    // carries the clean touch for its update id, including the repeated
    // delivery itself.
    let expected = reference_touches(seed);
    for obs in &trace.observations {
        if let Some(touch) = obs.touch {
            let want = expected
                .iter()
                .find(|(_, u, _)| *u == obs.update_id)
                .map(|(_, _, t)| *t);
            assert_eq!(Some(touch), want, "update id {}", obs.update_id);
        }
    }
}

#[test]
fn n6_crossed_snapshot_never_syncs() {
    let mut sync = BookSync::new();
    let crossed = SnapshotData::from_pairs(
        100,
        &[
            ("100.2".to_string(), "5.0".to_string()),
            ("99.9".to_string(), "3.0".to_string()),
        ],
        &[
            ("100.1".to_string(), "4.0".to_string()),
            ("100.2".to_string(), "2.0".to_string()),
        ],
    )
    .unwrap();
    let mut quotes = Vec::new();
    assert_eq!(
        sync.merge_snapshot(&crossed, &mut quotes),
        MergeOutcome::Refetch
    );
    assert!(quotes.is_empty());
    assert!(!sync.is_synced());
    // No signal from a book that never synced.
    let invalid = IntegrityState::Invalid {
        reason: "crossed snapshot refused".to_string(),
    };
    assert_eq!(signal_for(sync.book(), &invalid), None);
    // A well-formed snapshot syncs the same machine.
    let clean = SnapshotData::from_pairs(
        100,
        &[
            ("100.0".to_string(), "5.0".to_string()),
            ("99.9".to_string(), "3.0".to_string()),
        ],
        &[
            ("100.1".to_string(), "4.0".to_string()),
            ("100.2".to_string(), "2.0".to_string()),
        ],
    )
    .unwrap();
    assert_eq!(
        sync.merge_snapshot(&clean, &mut quotes),
        MergeOutcome::Synced { applied: 0 }
    );
    assert!(sync.is_synced());
}

#[test]
fn n6_seed_changes_the_fault_placement() {
    let (a, _) = run_trace(0, FaultKind::Dropped, BASE_SEED, N_EVENTS);
    let (b, _) = run_trace(0, FaultKind::Dropped, BASE_SEED + 7, N_EVENTS);
    assert_ne!(
        a.event_index, b.event_index,
        "different frozen seeds must place the fault at different events"
    );
    // ... while the same seed always places it at the same event.
    let (c, _) = run_trace(0, FaultKind::Dropped, BASE_SEED, N_EVENTS);
    assert_eq!(a.event_index, c.event_index);
}

/// The duplicate fault must be visible to the replay even though the
/// production sync machine treats an exact redelivery as idempotent:
/// the delivered stream really contains the copy.
#[test]
fn n6_duplicate_delivery_contains_the_copy() {
    let stream = toxicflow::fault_injection::build_clean_stream(N_EVENTS, BASE_SEED);
    let spec = toxicflow::fault_injection::FaultSpec {
        seed: BASE_SEED,
        kind: FaultKind::Duplicate,
        event_index: 5,
    };
    let delivered = toxicflow::fault_injection::inject_fault(&stream, spec);
    assert_eq!(delivered.len(), stream.events.len() + 1);
    assert_eq!(delivered[5], delivered[6]);
    assert_eq!(delivered[5], stream.events[5]);
}
