//! Registered manifest for the 60-second direction confirmation study.
//!
//! The study design is frozen in `docs/direction_confirmation_protocol.md`;
//! this module is its enforcement surface in code. It names the six archive
//! sessions, fixes the campaign slot roles by chronological scheduled-start
//! order (slots 01-06 fit, 07-10 development, 11-20 sealed holdout), and
//! provides the only resolution path fit-side code may use: archives and
//! slots 01-10 resolve, slots 11-20 are an error, directly or through a
//! derived key. The confirmation step must additionally present the
//! registered protocol digest pinned below, so a changed protocol document
//! cannot silently change the study: the digest test fails first.

use std::collections::HashSet;
use std::fmt;

/// Repository-relative path of the registered protocol document.
pub const PROTOCOL_PATH: &str = "docs/direction_confirmation_protocol.md";

/// SHA-256 of the protocol document as registered on 2026-11-08. If the
/// document changes deliberately, update this constant in the same commit;
/// the digest test in `tests/direction_confirmation.rs` fails otherwise.
pub const PROTOCOL_DIGEST: &str =
    "f9644bab289981b2f70529ee877ef600d29ea1b6b9f9118088fbfafde53411fa";

/// The six archive sessions, in the order they appear in
/// `docs/info_speed_report.json`. The TF6 pilot covers only the first
/// five; the 2026-11-06 archive has no TF6 row.
pub const ARCHIVE_SESSION_IDS: [&str; 6] = [
    "session_325h",
    "session-20261030-143624-utc",
    "session-20261101-203631-utc",
    "session-20261102-013620-utc",
    "session-20261104-073619-utc",
    "session-20261106-143623-utc",
];

/// Number of slots in the recording campaign.
pub const CAMPAIGN_SLOT_COUNT: usize = 20;

/// Scheduled length of one campaign slot, in seconds (4 hours).
pub const SLOT_DURATION_SECS: i64 = 4 * 60 * 60;

/// Last slot in the fit pool.
pub const FIT_LAST_SLOT: usize = 6;

/// Last slot in the development pool; slots after this are sealed.
pub const DEVELOPMENT_LAST_SLOT: usize = 10;

/// Role of a campaign slot in the confirmation study.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotRole {
    /// Slots 01-06: may be fit on during development.
    Fit,
    /// Slots 07-10: scored during development, never fit on out of turn.
    Development,
    /// Slots 11-20: sealed holdout, scored once after the model freeze.
    SealedHoldout,
}

/// The role of the campaign slot with the given 1-based index, or `None`
/// when the index is outside the campaign.
pub fn role_for_slot(slot: usize) -> Option<SlotRole> {
    match slot {
        1..=FIT_LAST_SLOT => Some(SlotRole::Fit),
        s if s > FIT_LAST_SLOT && s <= DEVELOPMENT_LAST_SLOT => Some(SlotRole::Development),
        s if s > DEVELOPMENT_LAST_SLOT && s <= CAMPAIGN_SLOT_COUNT => Some(SlotRole::SealedHoldout),
        _ => None,
    }
}

/// How the manifest classifies a session ID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionClass {
    /// One of the six pre-campaign archive sessions.
    Archive,
    /// A counted campaign slot, with its 1-based index and role.
    Campaign { slot: usize, role: SlotRole },
}

/// Why a manifest operation failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManifestError {
    /// The session ID is not a well-formed archive or campaign ID.
    MalformedSessionId(String),
    /// The same session identity was registered twice.
    DuplicateSession(String),
    /// Slot IDs were not in chronological scheduled-start order.
    ChronologyViolation,
    /// Two counted slots' scheduled windows overlap.
    OverlappingSlots,
    /// More session IDs were supplied than the campaign has slots.
    TooManySlots(usize),
    /// The session is well-formed but not part of this study.
    UnknownSession(String),
    /// Fit-side resolution was attempted for a sealed holdout slot.
    SealedHoldout(String),
    /// A content hash was not 64 lowercase hex characters.
    InvalidHash(String),
    /// The presented protocol digest is not the registered one.
    DigestMismatch,
    /// Confirmation was attempted for a session that is not sealed.
    NotSealed(String),
}

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ManifestError::MalformedSessionId(id) => {
                write!(f, "malformed session id: {id}")
            }
            ManifestError::DuplicateSession(id) => {
                write!(f, "duplicate session identity: {id}")
            }
            ManifestError::ChronologyViolation => {
                write!(f, "campaign slots are not in chronological order")
            }
            ManifestError::OverlappingSlots => {
                write!(f, "campaign slot windows overlap")
            }
            ManifestError::TooManySlots(n) => {
                write!(
                    f,
                    "{n} session ids exceed the {CAMPAIGN_SLOT_COUNT} campaign slots"
                )
            }
            ManifestError::UnknownSession(id) => {
                write!(f, "session is not part of the registered study: {id}")
            }
            ManifestError::SealedHoldout(id) => {
                write!(f, "session is in the sealed holdout (slots 11-20): {id}")
            }
            ManifestError::InvalidHash(h) => {
                write!(f, "content hash is not 64 lowercase hex characters: {h}")
            }
            ManifestError::DigestMismatch => {
                write!(
                    f,
                    "presented digest does not match the registered protocol digest"
                )
            }
            ManifestError::NotSealed(id) => {
                write!(f, "confirmation requires a sealed holdout session: {id}")
            }
        }
    }
}

impl std::error::Error for ManifestError {}

/// Parses a campaign session ID (`session-YYYYMMDD-HHMMSS-utc`) into its
/// scheduled start as Unix seconds. Returns `None` for anything that is
/// not a well-formed ID naming a real calendar date and time.
pub fn parse_session_start_unix(session_id: &str) -> Option<i64> {
    let rest = session_id.strip_prefix("session-")?;
    let rest = rest.strip_suffix("-utc")?;
    let (date, time) = rest.split_once('-')?;
    if date.len() != 8
        || time.len() != 6
        || !date.bytes().chain(time.bytes()).all(|b| b.is_ascii_digit())
    {
        return None;
    }
    let year: i64 = date[0..4].parse().ok()?;
    let month: u32 = date[4..6].parse().ok()?;
    let day: u32 = date[6..8].parse().ok()?;
    let hour: i64 = time[0..2].parse().ok()?;
    let minute: i64 = time[2..4].parse().ok()?;
    let second: i64 = time[4..6].parse().ok()?;
    if !(1970..=9999).contains(&year)
        || !(1..=12).contains(&month)
        || day < 1
        || day > days_in_month(year, month)
        || hour > 23
        || minute > 59
        || second > 59
    {
        return None;
    }
    let days = days_from_civil(year, month, day);
    Some(days * 86_400 + hour * 3_600 + minute * 60 + second)
}

fn is_leap_year(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn days_in_month(year: i64, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

/// Days from 1970-01-01 to the given civil date (Howard Hinnant's
/// algorithm), so no date library is needed for slot ordering.
fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = i64::from((month + 9) % 12);
    let doy = (153 * mp + 2) / 5 + i64::from(day) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn is_valid_content_hash(hash: &str) -> bool {
    hash.len() == 64
        && hash
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// One counted campaign slot: its identity, scheduled start and, once the
/// capture lands and is hashed, its content digest.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SlotRecord {
    session_id: String,
    start_unix: i64,
    content_sha256: Option<String>,
}

/// The registered study manifest: the archive identities plus the counted
/// campaign slots in chronological scheduled-start order.
#[derive(Debug, Clone)]
pub struct DirectionManifest {
    slots: Vec<SlotRecord>,
}

impl DirectionManifest {
    /// Builds a manifest from counted campaign session IDs in chronological
    /// scheduled-start order. The campaign may still be recording, so fewer
    /// than 20 IDs are accepted; roles follow position in the order, never
    /// the arrival order of files. Rejects duplicates, malformed IDs,
    /// out-of-order starts and overlapping scheduled windows.
    pub fn new(campaign_session_ids: &[&str]) -> Result<Self, ManifestError> {
        if campaign_session_ids.len() > CAMPAIGN_SLOT_COUNT {
            return Err(ManifestError::TooManySlots(campaign_session_ids.len()));
        }
        let mut seen: HashSet<&str> = HashSet::new();
        let mut slots: Vec<SlotRecord> = Vec::with_capacity(campaign_session_ids.len());
        for id in campaign_session_ids {
            if ARCHIVE_SESSION_IDS.contains(id) {
                return Err(ManifestError::DuplicateSession((*id).to_string()));
            }
            if !seen.insert(id) {
                return Err(ManifestError::DuplicateSession((*id).to_string()));
            }
            let start_unix = parse_session_start_unix(id)
                .ok_or_else(|| ManifestError::MalformedSessionId((*id).to_string()))?;
            if let Some(prev) = slots.last() {
                let prev: &SlotRecord = prev;
                if start_unix <= prev.start_unix {
                    return Err(ManifestError::ChronologyViolation);
                }
                if start_unix < prev.start_unix + SLOT_DURATION_SECS {
                    return Err(ManifestError::OverlappingSlots);
                }
            }
            slots.push(SlotRecord {
                session_id: (*id).to_string(),
                start_unix,
                content_sha256: None,
            });
        }
        Ok(Self { slots })
    }

    /// Number of counted campaign slots registered so far.
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Classifies a session ID without granting any access: an archive, a
    /// campaign slot with its role, or `None` when the study does not know
    /// the session.
    pub fn classify(&self, session_id: &str) -> Option<SessionClass> {
        if ARCHIVE_SESSION_IDS.contains(&session_id) {
            return Some(SessionClass::Archive);
        }
        self.slots
            .iter()
            .position(|s| s.session_id == session_id)
            .map(|idx| SessionClass::Campaign {
                slot: idx + 1,
                role: role_for_slot(idx + 1).expect("a registered slot always has a role"),
            })
    }

    /// Attaches a capture's content hash to its slot identity, as the
    /// protocol requires when a capture lands. The hash must be 64
    /// lowercase hex characters; the session must belong to the study.
    pub fn attach_hash(
        &mut self,
        session_id: &str,
        content_sha256: &str,
    ) -> Result<(), ManifestError> {
        if !is_valid_content_hash(content_sha256) {
            return Err(ManifestError::InvalidHash(content_sha256.to_string()));
        }
        match self.classify(session_id) {
            Some(SessionClass::Campaign { slot, .. }) => {
                self.slots[slot - 1].content_sha256 = Some(content_sha256.to_string());
                Ok(())
            }
            _ => Err(ManifestError::UnknownSession(session_id.to_string())),
        }
    }

    /// The content hash attached to a campaign slot, if one has landed.
    pub fn content_hash(&self, session_id: &str) -> Option<&str> {
        self.slots
            .iter()
            .find(|s| s.session_id == session_id)
            .and_then(|s| s.content_sha256.as_deref())
    }

    /// The fit-side allowlist. Archives and campaign slots 01-10 resolve;
    /// sealed holdout slots and unknown sessions are errors. Fit and
    /// development code must resolve every session through here, so the
    /// holdout cannot be opened by a path the manifest controls.
    pub fn resolve_for_fit(&self, session_id: &str) -> Result<SessionClass, ManifestError> {
        match self.classify(session_id) {
            Some(class @ SessionClass::Archive) => Ok(class),
            Some(class @ SessionClass::Campaign { role, .. }) => match role {
                SlotRole::Fit | SlotRole::Development => Ok(class),
                SlotRole::SealedHoldout => {
                    Err(ManifestError::SealedHoldout(session_id.to_string()))
                }
            },
            None => Err(ManifestError::UnknownSession(session_id.to_string())),
        }
    }

    /// Resolves a derived or cache key for fit use. A derived key names its
    /// source session either before a `::` separator or as one of its path
    /// components; the source session faces the same allowlist as a direct
    /// resolution, so a cached frame or feature file derived from a sealed
    /// slot is rejected exactly like the slot itself.
    pub fn resolve_derived_for_fit(
        &self,
        derived_key: &str,
    ) -> Result<SessionClass, ManifestError> {
        let source = extract_source_session(derived_key);
        self.resolve_for_fit(source)
    }

    /// The confirmation gate. The caller must present the registered
    /// protocol digest, and the session must be a sealed holdout slot.
    /// Anything else, including a fit or development session scored under
    /// the correct digest, is rejected.
    pub fn resolve_for_confirmation(
        &self,
        session_id: &str,
        presented_digest: &str,
    ) -> Result<SessionClass, ManifestError> {
        if presented_digest != PROTOCOL_DIGEST {
            return Err(ManifestError::DigestMismatch);
        }
        match self.classify(session_id) {
            Some(
                class @ SessionClass::Campaign {
                    role: SlotRole::SealedHoldout,
                    ..
                },
            ) => Ok(class),
            Some(_) => Err(ManifestError::NotSealed(session_id.to_string())),
            None => Err(ManifestError::UnknownSession(session_id.to_string())),
        }
    }
}

/// Extracts the source session ID named by a derived key. The session may
/// be named directly, before a `::` separator (with or without a directory
/// prefix), or as one path component of a longer key; in every shape the
/// bare session ID is what faces the allowlist.
fn extract_source_session(derived_key: &str) -> &str {
    fn names_a_session(component: &str) -> bool {
        ARCHIVE_SESSION_IDS.contains(&component) || parse_session_start_unix(component).is_some()
    }
    let head = derived_key.split("::").next().unwrap_or(derived_key);
    let base = head.rsplit('/').next().unwrap_or(head);
    if names_a_session(base) {
        return base;
    }
    for component in derived_key.split('/') {
        if names_a_session(component) {
            return component;
        }
    }
    base
}

/// SHA-256 of `data`, as 64 lowercase hex characters. Implemented here so
/// the protocol digest needs no new dependency.
pub fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    let (chunks, _remainder) = msg.as_chunks::<64>();
    for chunk in chunks {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[4 * i],
                chunk[4 * i + 1],
                chunk[4 * i + 2],
                chunk[4 * i + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    let mut out = String::with_capacity(64);
    for word in h {
        out.push_str(&format!("{word:08x}"));
    }
    out
}
