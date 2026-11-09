//! Tests for the registered direction confirmation manifest
//! (`src/direction_manifest.rs`): slot roles, identity validation, the
//! fit-side allowlist, the sealed-holdout negative-access cases, and the
//! protocol digest pin.

use toxicflow::direction_manifest::{
    parse_session_start_unix, role_for_slot, sha256_hex, DirectionManifest, ManifestError,
    SessionClass, SlotRole, ARCHIVE_SESSION_IDS, PROTOCOL_DIGEST, PROTOCOL_PATH,
};

/// Twenty counted campaign slots in chronological scheduled-start order,
/// at the rotation's cadence (4 h sessions, 10 s apart), beginning with
/// the campaign's actual first counted slot start.
const SLOT_IDS: [&str; 20] = [
    "session-20261109-021753-utc",
    "session-20261109-061803-utc",
    "session-20261109-101813-utc",
    "session-20261109-141823-utc",
    "session-20261109-181833-utc",
    "session-20261109-221843-utc",
    "session-20261110-021853-utc",
    "session-20261110-061903-utc",
    "session-20261110-101913-utc",
    "session-20261110-141923-utc",
    "session-20261110-181933-utc",
    "session-20261110-221943-utc",
    "session-20261111-021953-utc",
    "session-20261111-062003-utc",
    "session-20261111-102013-utc",
    "session-20261111-142023-utc",
    "session-20261111-182033-utc",
    "session-20261111-222043-utc",
    "session-20261112-022053-utc",
    "session-20261112-062103-utc",
];

fn full_manifest() -> DirectionManifest {
    DirectionManifest::new(&SLOT_IDS).expect("the registered slot schedule is valid")
}

#[test]
fn archive_ids_are_the_six_registered_sessions() {
    assert_eq!(
        ARCHIVE_SESSION_IDS,
        [
            "session_325h",
            "session-20261030-143624-utc",
            "session-20261101-203631-utc",
            "session-20261102-013620-utc",
            "session-20261104-073619-utc",
            "session-20261106-143623-utc",
        ]
    );
}

#[test]
fn slot_roles_follow_the_registered_split() {
    assert_eq!(role_for_slot(1), Some(SlotRole::Fit));
    assert_eq!(role_for_slot(6), Some(SlotRole::Fit));
    assert_eq!(role_for_slot(7), Some(SlotRole::Development));
    assert_eq!(role_for_slot(10), Some(SlotRole::Development));
    assert_eq!(role_for_slot(11), Some(SlotRole::SealedHoldout));
    assert_eq!(role_for_slot(20), Some(SlotRole::SealedHoldout));
    assert_eq!(role_for_slot(0), None);
    assert_eq!(role_for_slot(21), None);

    let m = full_manifest();
    assert_eq!(
        m.classify(SLOT_IDS[0]),
        Some(SessionClass::Campaign {
            slot: 1,
            role: SlotRole::Fit
        })
    );
    assert_eq!(
        m.classify(SLOT_IDS[9]),
        Some(SessionClass::Campaign {
            slot: 10,
            role: SlotRole::Development
        })
    );
    assert_eq!(
        m.classify(SLOT_IDS[19]),
        Some(SessionClass::Campaign {
            slot: 20,
            role: SlotRole::SealedHoldout
        })
    );
    assert_eq!(m.classify("session_325h"), Some(SessionClass::Archive));
}

#[test]
fn session_start_parsing_uses_real_calendar_time() {
    assert_eq!(
        parse_session_start_unix("session-19700101-000000-utc"),
        Some(0)
    );
    let first = parse_session_start_unix(SLOT_IDS[0]).unwrap();
    let second = parse_session_start_unix(SLOT_IDS[1]).unwrap();
    assert_eq!(second - first, 4 * 3600 + 10);
    // Not real dates or times.
    assert_eq!(
        parse_session_start_unix("session-20261301-000000-utc"),
        None
    );
    assert_eq!(
        parse_session_start_unix("session-20260230-000000-utc"),
        None
    );
    assert_eq!(
        parse_session_start_unix("session-20261109-241753-utc"),
        None
    );
    assert_eq!(parse_session_start_unix("session-20261109-021753"), None);
    assert_eq!(parse_session_start_unix("not-a-session"), None);
}

#[test]
fn slots_must_be_chronological_and_non_overlapping() {
    let mut swapped = SLOT_IDS;
    swapped.swap(3, 4);
    assert_eq!(
        DirectionManifest::new(&swapped).unwrap_err(),
        ManifestError::ChronologyViolation
    );

    // Two starts inside one 4-hour scheduled window overlap.
    let overlapping = ["session-20261109-021753-utc", "session-20261109-041753-utc"];
    assert_eq!(
        DirectionManifest::new(&overlapping).unwrap_err(),
        ManifestError::OverlappingSlots
    );
}

#[test]
fn duplicate_and_malformed_identities_are_rejected() {
    let duplicated = [SLOT_IDS[0], SLOT_IDS[1], SLOT_IDS[1]];
    assert_eq!(
        DirectionManifest::new(&duplicated).unwrap_err(),
        ManifestError::DuplicateSession(SLOT_IDS[1].to_string())
    );

    assert_eq!(
        DirectionManifest::new(&["session_325h"]).unwrap_err(),
        ManifestError::DuplicateSession("session_325h".to_string())
    );
    assert!(matches!(
        DirectionManifest::new(&["session-20261340-000000-utc"]),
        Err(ManifestError::MalformedSessionId(_))
    ));

    let mut too_many = SLOT_IDS.to_vec();
    too_many.push("session-20261112-102113-utc");
    assert_eq!(
        DirectionManifest::new(&too_many).unwrap_err(),
        ManifestError::TooManySlots(21)
    );
}

#[test]
fn partial_campaign_registers_the_slots_that_exist() {
    let m = DirectionManifest::new(&SLOT_IDS[..3]).unwrap();
    assert_eq!(m.slot_count(), 3);
    assert_eq!(
        m.classify(SLOT_IDS[2]),
        Some(SessionClass::Campaign {
            slot: 3,
            role: SlotRole::Fit
        })
    );
    // Slot 15 does not exist yet, so it is unknown, not sealed: the seal
    // applies by position once the slot is registered.
    assert_eq!(
        m.resolve_for_fit(SLOT_IDS[14]).unwrap_err(),
        ManifestError::UnknownSession(SLOT_IDS[14].to_string())
    );
}

#[test]
fn content_hashes_attach_only_in_the_registered_format() {
    let mut m = full_manifest();
    let good = "a".repeat(64);
    m.attach_hash(SLOT_IDS[0], &good).unwrap();
    assert_eq!(m.content_hash(SLOT_IDS[0]), Some(good.as_str()));
    assert_eq!(m.content_hash(SLOT_IDS[1]), None);

    for bad in [
        "a".repeat(63),
        "A".repeat(64),
        "g".repeat(64),
        String::new(),
    ] {
        assert_eq!(
            m.attach_hash(SLOT_IDS[0], &bad).unwrap_err(),
            ManifestError::InvalidHash(bad)
        );
    }
    assert_eq!(
        m.attach_hash("session-20270101-000000-utc", &good)
            .unwrap_err(),
        ManifestError::UnknownSession("session-20270101-000000-utc".to_string())
    );
}

#[test]
fn fit_allowlist_opens_archives_and_slots_01_to_10_only() {
    let m = full_manifest();
    assert_eq!(
        m.resolve_for_fit("session-20261104-073619-utc"),
        Ok(SessionClass::Archive)
    );
    assert!(m.resolve_for_fit(SLOT_IDS[0]).is_ok());
    assert!(m.resolve_for_fit(SLOT_IDS[9]).is_ok());
    for sealed in &SLOT_IDS[10..] {
        assert_eq!(
            m.resolve_for_fit(sealed).unwrap_err(),
            ManifestError::SealedHoldout(sealed.to_string())
        );
    }
    assert_eq!(
        m.resolve_for_fit("session-20270101-000000-utc")
            .unwrap_err(),
        ManifestError::UnknownSession("session-20270101-000000-utc".to_string())
    );
}

#[test]
fn derived_keys_cannot_reach_the_sealed_holdout() {
    let m = full_manifest();
    let sealed = SLOT_IDS[14];
    for key in [
        format!("{sealed}::frame-cache"),
        format!("cache/{sealed}/features.bin"),
        format!("derived/{sealed}::ofi-sums"),
    ] {
        assert_eq!(
            m.resolve_derived_for_fit(&key).unwrap_err(),
            ManifestError::SealedHoldout(sealed.to_string()),
            "derived key must stay sealed: {key}"
        );
    }
    // The same shapes resolve fine for fit-pool sources.
    assert!(m
        .resolve_derived_for_fit(&format!("{}::frame-cache", SLOT_IDS[2]))
        .is_ok());
    assert!(m
        .resolve_derived_for_fit("cache/session_325h/features.bin")
        .is_ok());
}

#[test]
fn confirmation_requires_the_registered_digest_and_a_sealed_slot() {
    let m = full_manifest();
    assert_eq!(
        m.resolve_for_confirmation(SLOT_IDS[14], PROTOCOL_DIGEST),
        Ok(SessionClass::Campaign {
            slot: 15,
            role: SlotRole::SealedHoldout
        })
    );
    assert_eq!(
        m.resolve_for_confirmation(SLOT_IDS[14], &"0".repeat(64))
            .unwrap_err(),
        ManifestError::DigestMismatch
    );
    assert_eq!(
        m.resolve_for_confirmation(SLOT_IDS[2], PROTOCOL_DIGEST)
            .unwrap_err(),
        ManifestError::NotSealed(SLOT_IDS[2].to_string())
    );
    assert_eq!(
        m.resolve_for_confirmation("session_325h", PROTOCOL_DIGEST)
            .unwrap_err(),
        ManifestError::NotSealed("session_325h".to_string())
    );
}

#[test]
fn sha256_matches_published_vectors() {
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}

#[test]
fn protocol_document_matches_the_registered_digest() {
    let path = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), PROTOCOL_PATH);
    let bytes = std::fs::read(&path).expect("the registered protocol document must exist");
    assert_eq!(
        sha256_hex(&bytes),
        PROTOCOL_DIGEST,
        "the protocol document changed; update PROTOCOL_DIGEST deliberately in the same commit"
    );
}
