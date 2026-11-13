//! Direction confirmation for the registered 60-second direction
//! confirmation study (`docs/direction_confirmation_protocol.md`):
//! B3 development fits, the section 7 freeze, and the section 8
//! confirmation.
//!
//! This module implements the protocol's sections 2-8 exactly as written:
//! the one-second epoch-aligned frame and strictly post-close target, the
//! four fixed feature families (R reference, F flow and liquidity
//! controls, V incremental VPIN, I bounded interactions), the single
//! unweighted ridge logistic fitting rule, the two operating rules, the
//! development selection ladder over the archives plus campaign
//! slots 01-06 predicting slot 07, with the fit pool growing by only
//! the earlier campaign slots for slots 08-10, under the 660-second
//! embargo, the section 7 freeze of the single selected family and rule
//! on the archives plus slots 01-10, and the section 8 confirmation:
//! one scoring of the frozen artifact on slots 11-20 with the
//! prespecified win / partial / null / incomplete gates, paired grouped
//! bootstrap intervals at 300 s and 600 s, fixed-cohort and per-session
//! diagnostics at 300 s, and leave-one-pair-out ranges. It never touches
//! the sealed holdout for fit use: every session resolves through
//! `direction_manifest`, and slots 11-20 resolve only through the
//! confirmation path, which requires the protocol's registered digest.
//
//! Two readings of the frozen text are fixed here rather than searched,
//! and both are stated so a reader can check them against the protocol:
//!
//! * "F[t] signed by the training-fitted burst flag" is the sign of
//!   F[t] (+1/-1) when the origin is burst (|F| at or above the training
//!   nearest-rank 95th percentile and F nonzero), else 0.
//! * Trailing sums and means are inclusive of the origin bar (bars
//!   t-k+1..=t). The five-second signed trade quantity and the total
//!   trade quantity it is divided by cover the same five bars.
//!
//! Readings fixed for the confirmation (section 8) are likewise
//! stated here:
//!
//! * "Fixed-cohort and per-session diagnostics at 300 seconds" are
//!   bootstrap intervals that keep the cohort fixed: every session is
//!   present in every replicate and bins are resampled within each
//!   session, using the section 8 diagnostic seed. The per-session
//!   diagnostic is the same procedure on one session alone.
//! * "Flat counts" count origins whose 60-second mid is exactly
//!   unchanged (mid[t+60] == mid[t]); those origins carry target 0
//!   under the strict post-close rule.
//! * Calibration diagnostics use ten fixed predicted-probability
//!   decile bins, [0, 0.1) through [0.9, 1.0], each reporting its row
//!   count, mean predicted probability and mean realized target.
//! * The section 10 headline sentence carries one interval per
//!   metric; with two prespecified block lengths the sentence names
//!   both, 300 s and 600 s, and the report JSON carries each
//!   separately.
//!
//! The AR(10) residual is an OLS fit with intercept on the training
//! origins, as the protocol's "training-fitted" requires; the fit
//! method itself is not pinned by the protocol, so the intercept is
//! documented here as the one open choice.
//!
//! Development bootstrap intervals are a stated analog of the
//! confirmation bootstrap in protocol section 8, adapted to the four
//! development sessions: two consecutive session pairs (07,08) and
//! (09,10) are drawn with replacement, and within each drawn session
//! its original number of time bins is resampled with replacement.
//! They use the section 8 diagnostic seed and 20,000 valid draws, and
//! are descriptive only, never confirmatory evidence.

use crate::direction_manifest::{
    parse_session_start_unix, sha256_hex, DirectionManifest, ManifestError, SessionClass,
    CAMPAIGN_SLOT_COUNT, DEVELOPMENT_LAST_SLOT, FIT_LAST_SLOT, PROTOCOL_DIGEST,
};
use crate::metrics::{ofi_events, Quote, Trade};
use crate::storage::{read_quotes, read_trades, StorageError};
use std::collections::HashMap;

/// One-second bars, UTC epoch-aligned, half-open.
pub const BAR_MS: i64 = 1_000;

/// Target horizon: the 60-second post-close mid move.
pub const TARGET_HORIZON_BARS: usize = 60;

/// The common support mask spans t-60 through t+60.
pub const MASK_HALF_WINDOW: usize = 60;

/// Embargo between the last fit origin and the first evaluation origin
/// (protocol section 2): 600 s maximum feature lookback + 60 s label.
pub const EMBARGO_SECS: i64 = 660;

/// VPIN trade window (protocol section 3): trades in the last 600 s
/// strictly before decision time.
pub const VPIN_LOOKBACK_MS: i64 = 600_000;

/// VPIN bucket size in base units (protocol section 3).
pub const VPIN_BUCKET_BTC: f64 = 1.0;

/// Completed buckets averaged for a VPIN value (protocol section 3).
pub const VPIN_MIN_BUCKETS: usize = 20;

/// Ridge penalty on the squared slope norm, halved (protocol
/// section 4): the objective adds 0.01 * ||slope||^2 / 2.
pub const RIDGE_LAMBDA: f64 = 0.01;

/// Maximum accepted Newton updates before a fit is a failure
/// (protocol section 4).
pub const NEWTON_MAX_UPDATES: usize = 100;

/// Required gradient infinity norm at convergence (protocol
/// section 4).
pub const NEWTON_GRAD_TOL: f64 = 1e-8;

/// Armijo sufficient-decrease coefficient (protocol section 4).
pub const ARMIJO_C: f64 = 0.0001;

/// Armijo step halvings before a fit is a failure (protocol
/// section 4).
pub const ARMIJO_MAX_HALVINGS: usize = 40;

/// AR order for the reference family's residual column.
pub const AR_ORDER: usize = 10;

/// Development bootstrap draws per block length (section 8 analog).
pub const DEV_BOOTSTRAP_DRAWS: usize = 20_000;

/// Development bootstrap attempts before interval claims are
/// suppressed (section 8 analog).
pub const DEV_BOOTSTRAP_MAX_ATTEMPTS: usize = 20_200;

/// Seed for the development bootstrap intervals: the section 8
/// diagnostic seed, used at both block lengths.
pub const DEV_BOOTSTRAP_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// Confirmation bootstrap draws per block length (protocol section 8).
pub const CONF_BOOTSTRAP_DRAWS: usize = 20_000;

/// Confirmation bootstrap attempts before interval claims are
/// suppressed and the study is incomplete (protocol section 8).
pub const CONF_BOOTSTRAP_MAX_ATTEMPTS: usize = 20_200;

/// Seed for the confirmation bootstrap intervals (protocol section 8).
pub const CONF_BOOTSTRAP_SEED: u64 = 0xD1B5_4A32_D192_ED03;

/// First sealed holdout slot (protocol section 1).
pub const CONFIRMATION_FIRST_SLOT: usize = DEVELOPMENT_LAST_SLOT + 1;

/// Last sealed holdout slot (protocol section 1).
pub const CONFIRMATION_LAST_SLOT: usize = CAMPAIGN_SLOT_COUNT;

/// Campaign admission: at least 13,680 quote-observed seconds
/// (protocol section 1).
pub const ADMISSION_MIN_OBSERVED_SECS: usize = 13_680;

/// Campaign admission: at least 12,960 origins with complete common
/// support out of 14,400 scheduled seconds (protocol section 1).
pub const ADMISSION_MIN_ORIGINS: usize = 12_960;

/// The twenty counted campaign slots in chronological scheduled-start
/// order, as landed in the November 2026 recording campaign (B1 QC
/// manifest). Roles follow position: 01-06 fit, 07-10 development,
/// 11-20 sealed holdout. The pre-campaign planned schedule named
/// different starts; these are the counted slots.
pub const CAMPAIGN_SLOT_IDS: [&str; CAMPAIGN_SLOT_COUNT] = [
    "session-20261109-083238-utc",
    "session-20261109-123248-utc",
    "session-20261109-163259-utc",
    "session-20261109-203309-utc",
    "session-20261110-003319-utc",
    "session-20261110-043329-utc",
    "session-20261110-083339-utc",
    "session-20261110-123349-utc",
    "session-20261110-163359-utc",
    "session-20261110-203410-utc",
    "session-20261111-003420-utc",
    "session-20261111-043430-utc",
    "session-20261111-083440-utc",
    "session-20261111-123450-utc",
    "session-20261111-163501-utc",
    "session-20261111-203511-utc",
    "session-20261112-003521-utc",
    "session-20261112-043531-utc",
    "session-20261112-083541-utc",
    "session-20261112-123552-utc",
];

/// A registered feature family (protocol section 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Family {
    /// Reference: nine columns.
    R,
    /// Flow and liquidity controls: R plus eight columns.
    F,
    /// Incremental VPIN: F plus a VPIN value and an availability bit.
    V,
    /// Bounded nonlinear: F plus three interaction terms.
    I,
}

impl Family {
    /// Final column count of the family.
    pub fn n_columns(self) -> usize {
        match self {
            Family::R => 9,
            Family::F => 17,
            Family::V => 19,
            Family::I => 20,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Family::R => "R",
            Family::F => "F",
            Family::V => "V",
            Family::I => "I",
        }
    }

    /// All four families in the protocol's simplicity order.
    pub fn all() -> [Family; 4] {
        [Family::R, Family::F, Family::V, Family::I]
    }
}

/// One of the two operating rules (protocol section 5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rule {
    /// Predict 1 when p > 0.50.
    P50,
    /// Predict 1 when p > training prevalence.
    Prevalence,
}

impl Rule {
    pub fn name(self) -> &'static str {
        match self {
            Rule::P50 => "p>0.50",
            Rule::Prevalence => "p>prevalence",
        }
    }

    pub fn all() -> [Rule; 2] {
        [Rule::P50, Rule::Prevalence]
    }
}

/// What went wrong in a B3 step. A failed fit or an undefined metric
/// is a failure, not a result (protocol section 6).
#[derive(Debug, Clone, PartialEq)]
pub enum DirectionError {
    /// The manifest refused a session (sealed holdout, unknown, ...).
    Manifest(ManifestError),
    /// A session file could not be read or failed strict validation.
    Storage(String),
    /// A campaign slot failed the section 1 admission counts.
    AdmissionFailed {
        session: String,
        observed_secs: usize,
        eligible_origins: usize,
    },
    /// A required study session was not supplied.
    MissingSession(String),
    /// A session was supplied that the study does not know.
    UnexpectedSession(String),
    /// The ridge fit did not converge or was nonfinite.
    FitFailed(String),
    /// A training set held a single class.
    SingleClassTraining,
    /// A metric is undefined (e.g. a class absent from scored rows).
    UndefinedMetric(String),
    /// A linear solve failed (singular system).
    SingularSystem(String),
    /// A frozen artifact record is malformed or inconsistent.
    InvalidArtifact(String),
}

impl std::fmt::Display for DirectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DirectionError::Manifest(e) => write!(f, "manifest: {e}"),
            DirectionError::Storage(e) => write!(f, "storage: {e}"),
            DirectionError::AdmissionFailed {
                session,
                observed_secs,
                eligible_origins,
            } => write!(
                f,
                "slot {session} failed admission: {observed_secs} observed seconds, {eligible_origins} eligible origins"
            ),
            DirectionError::MissingSession(s) => write!(f, "missing session: {s}"),
            DirectionError::UnexpectedSession(s) => write!(f, "unexpected session: {s}"),
            DirectionError::FitFailed(e) => write!(f, "fit failed: {e}"),
            DirectionError::SingleClassTraining => {
                write!(f, "training set holds a single class")
            }
            DirectionError::UndefinedMetric(e) => write!(f, "undefined metric: {e}"),
            DirectionError::SingularSystem(e) => write!(f, "singular system: {e}"),
            DirectionError::InvalidArtifact(e) => write!(f, "invalid artifact: {e}"),
        }
    }
}

impl std::error::Error for DirectionError {}

impl From<ManifestError> for DirectionError {
    fn from(e: ManifestError) -> Self {
        DirectionError::Manifest(e)
    }
}

impl From<StorageError> for DirectionError {
    fn from(e: StorageError) -> Self {
        DirectionError::Storage(e.to_string())
    }
}

/// One session's one-second frame: everything the feature families
/// need, built once and reused across development boundaries.
#[derive(Debug, Clone)]
pub struct SessionFrame {
    pub session_id: String,
    /// Epoch-aligned start of bar 0 (ms).
    pub t0_ms: i64,
    pub n_bars: usize,
    /// True when at least one quote was observed inside the bar.
    pub observed: Vec<bool>,
    /// Last-quote mid per bar (carried forward where unobserved;
    /// never read there: the mask admits only observed bars).
    pub mid: Vec<f64>,
    /// Mean of best bid and ask sizes at the bar's last quote.
    pub depth: Vec<f64>,
    /// Touch imbalance at the bar's last quote.
    pub imbalance: Vec<f64>,
    /// Cont-Kukanov-Stoikov OFI summed over the bar.
    pub ofi: Vec<f64>,
    /// Quotes observed inside the bar.
    pub quote_count: Vec<u32>,
    /// Signed trade quantity inside the bar (+qty buyer-initiated).
    pub signed_qty: Vec<f64>,
    /// Total trade quantity inside the bar.
    pub total_qty: Vec<f64>,
    /// All trades, time-sorted (the storage reader guarantees this).
    pub trades: Vec<Trade>,
    /// First bar index of the contiguous observed run containing i.
    pub seg_start: Vec<usize>,
    /// Index of the first quote-containing bar.
    pub first_observed: usize,
    /// Index of the last quote-containing bar.
    pub last_observed: usize,
}

impl SessionFrame {
    /// Quote-observed seconds in the session.
    pub fn n_observed_secs(&self) -> usize {
        self.observed.iter().filter(|&&o| o).count()
    }

    /// Decision time (right boundary / close) of origin bar `i`, ms.
    pub fn decision_time_ms(&self, i: usize) -> i64 {
        self.t0_ms + (i as i64 + 1) * BAR_MS
    }
}

/// Builds a session frame from loaded quotes and trades. Bars are UTC
/// epoch-aligned one-second half-open bars; a bar is observed when it
/// holds at least one quote.
pub fn build_frame(session_id: &str, quotes: &[Quote], trades: &[Trade]) -> SessionFrame {
    assert!(!quotes.is_empty(), "a session frame needs quotes");
    let t0_ms = quotes[0].time_ms / BAR_MS * BAR_MS;
    let n_bars = ((quotes.last().unwrap().time_ms - t0_ms) / BAR_MS + 1).max(1) as usize;
    let mut observed = vec![false; n_bars];
    let mut mid = vec![f64::NAN; n_bars];
    let mut depth = vec![f64::NAN; n_bars];
    let mut imbalance = vec![f64::NAN; n_bars];
    let mut quote_count = vec![0u32; n_bars];
    let mut signed_qty = vec![0.0; n_bars];
    let mut total_qty = vec![0.0; n_bars];

    let events = ofi_events(quotes);
    let mut ofi = vec![0.0; n_bars];
    for (q, e) in quotes.iter().zip(events.iter()) {
        let idx = ((q.time_ms - t0_ms) / BAR_MS) as usize;
        if idx >= n_bars {
            continue;
        }
        observed[idx] = true;
        quote_count[idx] += 1;
        ofi[idx] += e;
        // The last quote in the bar wins: quotes arrive in order.
        mid[idx] = q.mid();
        depth[idx] = (q.bid_sz + q.ask_sz) / 2.0;
        imbalance[idx] = q.imbalance();
    }
    // Carry the most recently observed quote forward, exactly as the
    // repo's ofi_bars does; the mask never reads unobserved bars.
    let mut carry_mid = quotes[0].mid();
    let mut carry_depth = (quotes[0].bid_sz + quotes[0].ask_sz) / 2.0;
    let mut carry_imb = quotes[0].imbalance();
    for i in 0..n_bars {
        if observed[i] {
            carry_mid = mid[i];
            carry_depth = depth[i];
            carry_imb = imbalance[i];
        } else {
            mid[i] = carry_mid;
            depth[i] = carry_depth;
            imbalance[i] = carry_imb;
        }
    }
    for t in trades {
        let idx = ((t.time_ms - t0_ms) / BAR_MS) as usize;
        if idx < n_bars {
            signed_qty[idx] += t.signed_qty();
            total_qty[idx] += t.qty;
        }
    }
    // First bar index of the contiguous observed run containing i
    // (for observed bars; unobserved bars keep the next run's start,
    // which the mask never reads).
    let mut seg_start = vec![0usize; n_bars];
    let mut run_start = 0usize;
    for i in 0..n_bars {
        if i == 0 || !observed[i - 1] {
            run_start = i;
        }
        if observed[i] {
            seg_start[i] = run_start;
        }
    }
    let first_observed = observed.iter().position(|&o| o).unwrap_or(0);
    let last_observed = observed
        .iter()
        .rposition(|&o| o)
        .unwrap_or(n_bars.saturating_sub(1));
    SessionFrame {
        session_id: session_id.to_string(),
        t0_ms,
        n_bars,
        observed,
        mid,
        depth,
        imbalance,
        ofi,
        quote_count,
        signed_qty,
        total_qty,
        trades: trades.to_vec(),
        seg_start,
        first_observed,
        last_observed,
    }
}

/// Eligible origins of a frame under the protocol section 2 common
/// support mask: actual observations on every bar from t-60 through
/// t+60 inside one contiguous observed segment, excluding the first
/// and last quote-containing bars of the session.
pub fn eligible_origins(frame: &SessionFrame) -> Vec<usize> {
    let n = frame.n_bars;
    let mut out = Vec::new();
    if n <= 2 * MASK_HALF_WINDOW {
        return out;
    }
    for i in MASK_HALF_WINDOW..n - MASK_HALF_WINDOW {
        if i == frame.first_observed || i == frame.last_observed {
            continue;
        }
        if frame.observed[i - MASK_HALF_WINDOW..=i + MASK_HALF_WINDOW]
            .iter()
            .all(|&o| o)
        {
            out.push(i);
        }
    }
    out
}

/// Loads one session directory (quotes.parquet + trades.parquet)
/// through the manifest allowlist, with the repo's strict storage
/// validation. Campaign slots additionally face the section 1
/// admission counts; archives do not.
pub fn load_session(
    manifest: &DirectionManifest,
    session_id: &str,
    dir: &str,
) -> Result<SessionFrame, DirectionError> {
    let class = manifest.resolve_for_fit(session_id)?;
    let quotes = read_quotes(&format!("{dir}/quotes.parquet"))?;
    let trades = read_trades(&format!("{dir}/trades.parquet"))?;
    let frame = build_frame(session_id, &quotes, &trades);
    if let SessionClass::Campaign { .. } = class {
        admission_check(&frame)?;
    }
    Ok(frame)
}

/// Loads one sealed holdout session directory through the
/// confirmation path (protocol sections 8-9): the caller must present
/// this protocol's registered digest, and the slot must be a sealed
/// holdout slot. Campaign slots face the section 1 admission counts,
/// so a missing or inadmissible holdout input is an error here, not a
/// result.
pub fn load_session_for_confirmation(
    manifest: &DirectionManifest,
    session_id: &str,
    dir: &str,
    presented_digest: &str,
) -> Result<SessionFrame, DirectionError> {
    let class = manifest.resolve_for_confirmation(session_id, presented_digest)?;
    let quotes = read_quotes(&format!("{dir}/quotes.parquet"))?;
    let trades = read_trades(&format!("{dir}/trades.parquet"))?;
    let frame = build_frame(session_id, &quotes, &trades);
    if let SessionClass::Campaign { .. } = class {
        admission_check(&frame)?;
    }
    Ok(frame)
}

/// Loads one session directory (quotes.parquet + trades.parquet)
/// without the manifest allowlist, for the exploratory ETH work
/// only: the ETH campaign has no registered manifest, so every
/// session faces the section 1 admission counts. This path never
/// resolves a sealed BTC holdout slot and must not be used for the
/// registered confirmation.
pub fn load_session_unsealed(session_id: &str, dir: &str) -> Result<SessionFrame, DirectionError> {
    let quotes = read_quotes(&format!("{dir}/quotes.parquet"))?;
    let trades = read_trades(&format!("{dir}/trades.parquet"))?;
    let frame = build_frame(session_id, &quotes, &trades);
    admission_check(&frame)?;
    Ok(frame)
}

/// Section 1 admission for a campaign slot: at least 13,680
/// quote-observed seconds and at least 12,960 eligible origins.
fn admission_check(frame: &SessionFrame) -> Result<(), DirectionError> {
    let origins = eligible_origins(frame).len();
    let observed = frame.n_observed_secs();
    if observed < ADMISSION_MIN_OBSERVED_SECS || origins < ADMISSION_MIN_ORIGINS {
        return Err(DirectionError::AdmissionFailed {
            session: frame.session_id.clone(),
            observed_secs: observed,
            eligible_origins: origins,
        });
    }
    Ok(())
}

/// Dense Cholesky solve for a symmetric positive-definite system
/// `a x = b`. Returns `None` when a pivot is not positive (the
/// system is not positive-definite) or a value is nonfinite.
pub fn cholesky_solve(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    if a.len() != n || a.iter().any(|row| row.len() != n) {
        return None;
    }
    let mut l = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i][j];
            for (x, y) in l[i].iter().zip(l[j].iter()).take(j) {
                s -= x * y;
            }
            if i == j {
                if s <= 0.0 || !s.is_finite() {
                    return None;
                }
                l[i][j] = s.sqrt();
            } else {
                if !l[j][j].is_finite() || l[j][j] == 0.0 {
                    return None;
                }
                l[i][j] = s / l[j][j];
            }
        }
    }
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i][k] * y[k];
        }
        y[i] = s / l[i][i];
    }
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in i + 1..n {
            s -= l[k][i] * x[k];
        }
        x[i] = s / l[i][i];
    }
    if x.iter().any(|v| !v.is_finite()) {
        return None;
    }
    Some(x)
}

/// Split already time-sorted trades into fixed-size equal-volume
/// buckets, mirroring `metrics::volume_buckets_sized` exactly (same
/// proportional split of straddling trades, same dropped trailing
/// partial bucket) but without re-sorting. A unit test pins equality
/// against the reference routine on random trade slices.
pub fn buckets_sorted(trades: &[Trade], bucket_vol: f64) -> Vec<crate::metrics::Bucket> {
    use crate::metrics::Bucket;
    if trades.is_empty() || bucket_vol <= 0.0 {
        return Vec::new();
    }
    let v = bucket_vol;
    let mut out = Vec::new();
    let (mut buy, mut sell, mut filled) = (0.0, 0.0, 0.0);
    let mut end_time = 0i64;
    for t in trades {
        let (mut rb, mut rs) = if t.is_buyer_maker {
            (0.0, t.qty)
        } else {
            (t.qty, 0.0)
        };
        while rb + rs > 1e-12 {
            let take = (v - filled).min(rb + rs);
            let frac = take / (rb + rs);
            buy += rb * frac;
            sell += rs * frac;
            filled += take;
            rb *= 1.0 - frac;
            rs *= 1.0 - frac;
            end_time = end_time.max(t.time_ms);
            if filled >= v - 1e-9 {
                out.push(Bucket {
                    time_ms: end_time,
                    buy_vol: buy,
                    sell_vol: sell,
                    imbalance: (buy - sell).abs() / v,
                });
                buy = 0.0;
                sell = 0.0;
                filled = 0.0;
            }
        }
    }
    out
}

/// VPIN value and availability bit for one origin (protocol
/// section 3, family V): trades in the last 600 s strictly before
/// decision time, clipped to the start of the current contiguous
/// observed segment, split into fixed 1 BTC buckets, final partial
/// bucket dropped, mean absolute buy-sell imbalance over the last 20
/// completed buckets. Fewer than 20 completed buckets gives (0, 0).
pub fn vpin_for_origin(frame: &SessionFrame, origin_bar: usize) -> (f64, f64) {
    let decision_ms = frame.decision_time_ms(origin_bar);
    let seg_start_ms = frame.t0_ms + frame.seg_start[origin_bar] as i64 * BAR_MS;
    let lo_ms = (decision_ms - VPIN_LOOKBACK_MS).max(seg_start_ms);
    let trades = &frame.trades;
    let start = trades.partition_point(|t| t.time_ms < lo_ms);
    let end = trades.partition_point(|t| t.time_ms < decision_ms);
    let buckets = buckets_sorted(&trades[start..end], VPIN_BUCKET_BTC);
    if buckets.len() < VPIN_MIN_BUCKETS {
        return (0.0, 0.0);
    }
    let last = &buckets[buckets.len() - VPIN_MIN_BUCKETS..];
    let vpin = last.iter().map(|b| b.imbalance).sum::<f64>() / VPIN_MIN_BUCKETS as f64;
    (vpin, 1.0)
}

/// One eligible origin's boundary-independent data. Raw feature
/// columns other than the AR residual and the burst-signed column are
/// computed from the frame on demand; the VPIN pair is computed once
/// here because it is the expensive column.
#[derive(Debug, Clone)]
pub struct OriginData {
    /// Origin bar index in its session frame.
    pub bar: usize,
    /// Decision time (right boundary of the origin bar), ms.
    pub decision_ms: i64,
    /// Target: 1 when mid[t+60] > mid[t], else 0.
    pub target: u8,
    /// VPIN value for the origin.
    pub vpin: f64,
    /// VPIN availability bit.
    pub vpin_avail: f64,
}

/// Per-session precomputed window helpers: per-bar N and one-second
/// log mid returns in basis points, plus prefix sums for O(1)
/// trailing windows.
pub struct FrameCalc<'a> {
    pub frame: &'a SessionFrame,
    /// N[t] = asinh(F[t] / D[t]) per bar.
    pub n_val: Vec<f64>,
    /// One-second log mid return in basis points per bar (0 at bar 0).
    pub ret_bps: Vec<f64>,
    pub pref_ofi: Vec<f64>,
    pub pref_n: Vec<f64>,
    pub pref_imb: Vec<f64>,
    pub pref_signed: Vec<f64>,
    pub pref_total: Vec<f64>,
    pub pref_qcount: Vec<f64>,
    pub pref_sqret: Vec<f64>,
}

fn prefix(xs: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; xs.len() + 1];
    for (i, &x) in xs.iter().enumerate() {
        out[i + 1] = out[i] + x;
    }
    out
}

/// Inclusive window sum over bars lo..=hi from a prefix-sum array.
fn wsum(pref: &[f64], lo: usize, hi: usize) -> f64 {
    pref[hi + 1] - pref[lo]
}

/// Inclusive window mean over bars lo..=hi from a prefix-sum array.
fn wmean(pref: &[f64], lo: usize, hi: usize) -> f64 {
    wsum(pref, lo, hi) / (hi - lo + 1) as f64
}

impl<'a> FrameCalc<'a> {
    pub fn new(frame: &'a SessionFrame) -> Self {
        let n = frame.n_bars;
        let n_val: Vec<f64> = (0..n)
            .map(|i| (frame.ofi[i] / frame.depth[i]).asinh())
            .collect();
        let mut ret_bps = vec![0.0; n];
        for (pair, r) in frame.mid.windows(2).zip(ret_bps.iter_mut().skip(1)) {
            *r = 10_000.0 * (pair[1] / pair[0]).ln();
        }
        let qcount: Vec<f64> = frame.quote_count.iter().map(|&c| c as f64).collect();
        let sqret: Vec<f64> = ret_bps.iter().map(|r| r * r).collect();
        FrameCalc {
            frame,
            pref_ofi: prefix(&frame.ofi),
            pref_n: prefix(&n_val),
            pref_imb: prefix(&frame.imbalance),
            pref_signed: prefix(&frame.signed_qty),
            pref_total: prefix(&frame.total_qty),
            pref_qcount: prefix(&qcount),
            pref_sqret: prefix(&sqret),
            n_val,
            ret_bps,
        }
    }

    /// All eligible origins of the frame with their targets and VPIN.
    pub fn origins(&self) -> Vec<OriginData> {
        let frame = self.frame;
        eligible_origins(frame)
            .into_iter()
            .map(|i| {
                let (vpin, avail) = vpin_for_origin(frame, i);
                OriginData {
                    bar: i,
                    decision_ms: frame.decision_time_ms(i),
                    target: u8::from(frame.mid[i + TARGET_HORIZON_BARS] > frame.mid[i]),
                    vpin,
                    vpin_avail: avail,
                }
            })
            .collect()
    }
}

/// Training-fitted transforms for one development boundary: the
/// AR(10) fit, the burst threshold, the training prevalence and
/// majority label, and per-family column standardization.
#[derive(Debug, Clone)]
pub struct FitContext {
    pub ar_intercept: f64,
    pub ar_coefs: [f64; AR_ORDER],
    pub burst_threshold: f64,
    pub prevalence: f64,
    pub majority_label: u8,
    /// (mean, population sd) per final column, per family.
    pub standard: HashMap<Family, Vec<(f64, f64)>>,
}

/// Fits the AR(10) residual model on training origins: OLS with
/// intercept of F[t] on F[t-1..t-10], solved by Cholesky on the
/// normal equations.
pub fn fit_ar10<F>(train: &[(F, &[OriginData])]) -> Result<(f64, [f64; AR_ORDER]), DirectionError>
where
    F: std::ops::Deref<Target = SessionFrame>,
{
    let dim = AR_ORDER + 1;
    let mut ata = vec![vec![0.0f64; dim]; dim];
    let mut atb = vec![0.0f64; dim];
    let mut n = 0usize;
    for (frame, origins) in train {
        for o in origins.iter() {
            let t = o.bar;
            let mut x = [0.0f64; AR_ORDER + 1];
            x[0] = 1.0;
            for (k, slot) in x.iter_mut().enumerate().skip(1) {
                *slot = frame.ofi[t - k];
            }
            let y = frame.ofi[t];
            for i in 0..dim {
                for j in 0..dim {
                    ata[i][j] += x[i] * x[j];
                }
                atb[i] += x[i] * y;
            }
            n += 1;
        }
    }
    if n == 0 {
        return Err(DirectionError::SingularSystem(
            "no training origins for the AR(10) fit".to_string(),
        ));
    }
    let w = cholesky_solve(&ata, &atb).ok_or_else(|| {
        DirectionError::SingularSystem(
            "AR(10) normal equations are not positive-definite".to_string(),
        )
    })?;
    let mut coefs = [0.0f64; AR_ORDER];
    coefs.copy_from_slice(&w[1..]);
    Ok((w[0], coefs))
}

/// Nearest-rank 95th percentile of |F| over training origins:
/// rank ceil(0.95 * n), 1-indexed into the sorted values.
pub fn burst_threshold(train_f: &mut [f64]) -> f64 {
    train_f.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let rank = ((0.95 * train_f.len() as f64).ceil() as usize)
        .max(1)
        .min(train_f.len());
    train_f[rank - 1]
}

/// Raw (pre-standardization) columns of one origin for one family,
/// in the protocol section 3 order. Columns 4 (AR residual) and 5
/// (burst-signed F) use the training-fitted context.
pub fn raw_columns(
    calc: &FrameCalc,
    origin: &OriginData,
    ctx: &FitContext,
    family: Family,
) -> Vec<f64> {
    let frame = calc.frame;
    let t = origin.bar;
    let f = frame.ofi[t];
    // R columns.
    let mut cols = Vec::with_capacity(family.n_columns());
    cols.push(f);
    cols.push(wsum(&calc.pref_ofi, t - 4, t));
    cols.push(wsum(&calc.pref_ofi, t - 29, t));
    cols.push(wsum(&calc.pref_ofi, t - 59, t));
    let mut ar_pred = ctx.ar_intercept;
    for k in 1..=AR_ORDER {
        ar_pred += ctx.ar_coefs[k - 1] * frame.ofi[t - k];
    }
    cols.push(f - ar_pred);
    cols.push(if f != 0.0 && f.abs() >= ctx.burst_threshold {
        f.signum()
    } else {
        0.0
    });
    cols.push(frame.imbalance[t]);
    cols.push(frame.signed_qty[t]);
    cols.push(calc.ret_bps[t]);
    if family == Family::R {
        return cols;
    }
    // F columns.
    let n_t = calc.n_val[t];
    cols.push(n_t);
    cols.push(wmean(&calc.pref_n, t - 4, t));
    cols.push(wmean(&calc.pref_n, t - 29, t));
    cols.push(wmean(&calc.pref_imb, t - 4, t));
    let signed5 = wsum(&calc.pref_signed, t - 4, t);
    let total5 = wsum(&calc.pref_total, t - 4, t);
    cols.push(if total5 == 0.0 { 0.0 } else { signed5 / total5 });
    cols.push((1.0 + wsum(&calc.pref_total, t - 29, t)).ln());
    let log_qcount30 = (1.0 + wsum(&calc.pref_qcount, t - 29, t)).ln();
    cols.push(log_qcount30);
    cols.push(wmean(&calc.pref_sqret, t - 29, t).sqrt());
    match family {
        Family::F => {}
        Family::V => {
            cols.push(origin.vpin);
            cols.push(origin.vpin_avail);
        }
        Family::I => {
            cols.push(n_t * frame.imbalance[t]);
            cols.push(n_t * log_qcount30);
            cols.push(n_t * calc.ret_bps[t]);
        }
        Family::R => unreachable!(),
    }
    cols
}

/// Builds the training-fitted context for one boundary from the fit
/// pool's origins: AR(10), burst threshold, prevalence, majority
/// label, and per-family column standardization (two-pass training
/// mean and population standard deviation).
pub fn build_fit_context(
    train: &[(FrameCalc<'_>, Vec<OriginData>)],
) -> Result<FitContext, DirectionError> {
    // AR(10) on training origins.
    let frames: Vec<(&SessionFrame, &[OriginData])> =
        train.iter().map(|(c, o)| (c.frame, o.as_slice())).collect();
    let (ar_intercept, ar_coefs) = fit_ar10(&frames)?;
    // Burst threshold from |F| over training origins.
    let mut abs_f: Vec<f64> = train
        .iter()
        .flat_map(|(c, o)| o.iter().map(move |og| c.frame.ofi[og.bar].abs()))
        .collect();
    if abs_f.is_empty() {
        return Err(DirectionError::SingularSystem(
            "no training origins for the burst threshold".to_string(),
        ));
    }
    let burst_threshold = burst_threshold(&mut abs_f);
    // Prevalence and majority label over training targets.
    let mut n = 0usize;
    let mut n1 = 0usize;
    for (_, origins) in train {
        for o in origins {
            n += 1;
            n1 += usize::from(o.target);
        }
    }
    if n == 0 {
        return Err(DirectionError::SingularSystem(
            "no training origins".to_string(),
        ));
    }
    let prevalence = n1 as f64 / n as f64;
    if n1 == 0 || n1 == n {
        return Err(DirectionError::SingleClassTraining);
    }
    // Ties go to class 1 (protocol section 4).
    let majority_label = if n1 * 2 >= n { 1 } else { 0 };
    let mut ctx = FitContext {
        ar_intercept,
        ar_coefs,
        burst_threshold,
        prevalence,
        majority_label,
        standard: HashMap::new(),
    };
    for family in Family::all() {
        let mut rows: Vec<Vec<f64>> = Vec::new();
        for (calc, origins) in train {
            for o in origins {
                rows.push(raw_columns(calc, o, &ctx, family));
            }
        }
        let d = family.n_columns();
        let mut standard = Vec::with_capacity(d);
        for j in 0..d {
            let mean = rows.iter().map(|r| r[j]).sum::<f64>() / rows.len() as f64;
            let var = rows
                .iter()
                .map(|r| {
                    let z = r[j] - mean;
                    z * z
                })
                .sum::<f64>()
                / rows.len() as f64;
            standard.push((mean, var.sqrt()));
        }
        ctx.standard.insert(family, standard);
    }
    Ok(ctx)
}

/// Standardizes one raw row with the training-fitted column stats:
/// a column with sd at most 1e-12 is zeroed, otherwise z-scores are
/// clipped to +/-8 (protocol section 3).
pub fn standardize(raw: &[f64], standard: &[(f64, f64)]) -> Vec<f64> {
    raw.iter()
        .zip(standard.iter())
        .map(|(x, (mean, sd))| {
            if *sd <= 1e-12 {
                0.0
            } else {
                ((x - mean) / sd).clamp(-8.0, 8.0)
            }
        })
        .collect()
}

/// Deterministic xorshift64* generator (protocol section 8 uses a
/// xorshift generator; the development analog uses the same family).
pub struct XorShift64Star(pub u64);

impl XorShift64Star {
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn next_below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z.clamp(-30.0, 30.0)).exp())
}

/// Numerically stable softplus: ln(1 + exp(a)).
fn softplus(a: f64) -> f64 {
    if a > 30.0 {
        a
    } else if a < -30.0 {
        a.exp()
    } else {
        (1.0 + a.exp()).ln()
    }
}

/// Fitted ridge logistic model (protocol section 4).
#[derive(Debug, Clone)]
pub struct RidgeFit {
    pub slopes: Vec<f64>,
    pub intercept: f64,
    pub n_accepted: usize,
    pub grad_inf_norm: f64,
}

/// Mean binary log loss plus 0.01 * ||slope||^2 / 2, with its
/// gradient (slopes, then intercept) and Hessian.
fn ridge_objective(xs: &[Vec<f64>], y: &[u8], w: &[f64], b: f64) -> (f64, Vec<f64>, Vec<Vec<f64>>) {
    let n = xs.len() as f64;
    let d = w.len();
    let mut loss = 0.0;
    let mut grad = vec![0.0f64; d + 1];
    let mut hess = vec![vec![0.0f64; d + 1]; d + 1];
    for (row, &yi) in xs.iter().zip(y.iter()) {
        let mut z = b;
        for j in 0..d {
            z += w[j] * row[j];
        }
        let p = sigmoid(z);
        let yf = yi as f64;
        loss += if yi == 1 { softplus(-z) } else { softplus(z) };
        let err = p - yf;
        let v = p * (1.0 - p);
        for j in 0..d {
            grad[j] += err * row[j];
            for (hk, &rk) in hess[j].iter_mut().zip(row.iter()) {
                *hk += v * row[j] * rk;
            }
            hess[j][d] += v * row[j];
            hess[d][j] += v * row[j];
        }
        grad[d] += err;
        hess[d][d] += v;
    }
    loss /= n;
    for j in 0..d {
        loss += 0.5 * RIDGE_LAMBDA * w[j] * w[j];
        grad[j] = grad[j] / n + RIDGE_LAMBDA * w[j];
        for (k, hk) in hess[j].iter_mut().enumerate() {
            *hk = *hk / n + if j == k { RIDGE_LAMBDA } else { 0.0 };
        }
        hess[j][d] /= n;
        hess[d][j] /= n;
    }
    grad[d] /= n;
    hess[d][d] /= n;
    (loss, grad, hess)
}

/// One deterministic ridge logistic fit (protocol section 4):
/// unweighted ridge logistic regression minimizing mean binary log
/// loss plus 0.01 * ||slope||^2 / 2, free intercept. Slopes start
/// at 0, the intercept at the logit of training prevalence. Newton
/// steps are solved by Cholesky factorization, step size starts at 1
/// and halves up to 40 times under Armijo coefficient 0.0001; the
/// fit must reach gradient infinity norm at most 1e-8 within 100
/// accepted updates. A nonfinite fit or a single-class training set
/// is a failure, not a result.
pub fn fit_ridge_logistic(
    xs: &[Vec<f64>],
    y: &[u8],
    prevalence: f64,
) -> Result<RidgeFit, DirectionError> {
    let d = xs.first().map(|r| r.len()).unwrap_or(0);
    if xs.is_empty() || d == 0 || xs.len() != y.len() {
        return Err(DirectionError::FitFailed(
            "empty or ragged training rows".to_string(),
        ));
    }
    if prevalence <= 0.0 || prevalence >= 1.0 || !prevalence.is_finite() {
        return Err(DirectionError::SingleClassTraining);
    }
    let mut w = vec![0.0f64; d];
    let mut b = (prevalence / (1.0 - prevalence)).ln();
    let (mut loss, mut grad, _) = ridge_objective(xs, y, &w, b);
    let mut accepted = 0usize;
    loop {
        let ginf = grad.iter().fold(0.0f64, |m, &g| m.max(g.abs()));
        if ginf <= NEWTON_GRAD_TOL {
            return Ok(RidgeFit {
                slopes: w,
                intercept: b,
                n_accepted: accepted,
                grad_inf_norm: ginf,
            });
        }
        if accepted >= NEWTON_MAX_UPDATES {
            return Err(DirectionError::FitFailed(format!(
                "gradient infinity norm {ginf:.3e} after {accepted} accepted updates"
            )));
        }
        let (_, _, hess) = ridge_objective(xs, y, &w, b);
        let rhs: Vec<f64> = grad.iter().map(|g| -g).collect();
        // Full Newton step: d slope steps, then the intercept step.
        let step = cholesky_solve(&hess, &rhs).ok_or_else(|| {
            DirectionError::FitFailed("Newton Hessian is not positive-definite".to_string())
        })?;
        debug_assert_eq!(step.len(), d + 1);
        let dir_deriv: f64 = grad.iter().zip(step.iter()).map(|(g, s)| g * s).sum();
        let mut alpha = 1.0f64;
        let mut ok = false;
        for _ in 0..=ARMIJO_MAX_HALVINGS {
            let w_try: Vec<f64> = (0..d).map(|j| w[j] + alpha * step[j]).collect();
            let b_try = b + alpha * step[d];
            if w_try
                .iter()
                .chain(std::iter::once(&b_try))
                .all(|v| v.is_finite())
            {
                let (loss_try, _, _) = ridge_objective(xs, y, &w_try, b_try);
                if loss_try.is_finite() && loss_try <= loss + ARMIJO_C * alpha * dir_deriv {
                    w = w_try;
                    b = b_try;
                    loss = loss_try;
                    ok = true;
                    break;
                }
            }
            alpha *= 0.5;
        }
        if !ok {
            return Err(DirectionError::FitFailed(
                "Armijo line search failed after 40 halvings".to_string(),
            ));
        }
        accepted += 1;
        let (_, g2, _) = ridge_objective(xs, y, &w, b);
        grad = g2;
    }
}

/// Predicted probability for one standardized row.
pub fn predict_proba(fit: &RidgeFit, x: &[f64]) -> f64 {
    let mut z = fit.intercept;
    for (j, &v) in x.iter().enumerate() {
        z += fit.slopes[j] * v;
    }
    sigmoid(z)
}

/// One scored evaluation row: the model and baseline predictions on
/// the same origin, paired.
#[derive(Debug, Clone, Copy)]
pub struct ScoredRow {
    pub time_ms: i64,
    pub target: u8,
    pub model_pred: u8,
    pub baseline_pred: u8,
}

/// Confusion counts (unnormalized unless stated).
#[derive(Debug, Clone, Copy, Default)]
pub struct Confusion {
    pub tp: f64,
    pub tn: f64,
    pub fp: f64,
    pub fn_: f64,
}

impl Confusion {
    pub fn from_rows(rows: &[ScoredRow]) -> Self {
        let mut c = Confusion::default();
        for r in rows {
            match (r.target, r.model_pred) {
                (1, 1) => c.tp += 1.0,
                (0, 0) => c.tn += 1.0,
                (0, 1) => c.fp += 1.0,
                (1, 0) => c.fn_ += 1.0,
                _ => unreachable!(),
            }
        }
        c
    }

    pub fn n(&self) -> f64 {
        self.tp + self.tn + self.fp + self.fn_
    }

    /// Divide every entry by `n` (per-session normalization,
    /// protocol section 6).
    pub fn normalized(&self, n: f64) -> Self {
        Confusion {
            tp: self.tp / n,
            tn: self.tn / n,
            fp: self.fp / n,
            fn_: self.fn_ / n,
        }
    }

    /// Accuracy of a normalized table.
    pub fn accuracy(&self) -> f64 {
        self.tp + self.tn
    }

    /// Balanced accuracy of a normalized table, or `None` when a
    /// class is absent.
    pub fn balanced_accuracy(&self) -> Option<f64> {
        let r1 = self.tp + self.fn_;
        let r0 = self.tn + self.fp;
        if r1 == 0.0 || r0 == 0.0 {
            None
        } else {
            Some(0.5 * (self.tp / r1 + self.tn / r0))
        }
    }
}

/// Pooled endpoints from equal-session averaged confusion tables
/// (protocol section 6).
#[derive(Debug, Clone, Copy)]
pub struct PooledMetrics {
    pub n_scored: usize,
    pub accuracy: f64,
    pub baseline_accuracy: f64,
    pub balanced_accuracy: f64,
    pub delta: f64,
    pub gamma: f64,
    /// Selection score S = min(Delta / 0.010, Gamma / 0.020).
    pub s: f64,
}

/// Pools per-session model/baseline row sets into the section 6
/// endpoints. Each session's table is normalized by its scored
/// count and averaged with equal session weights. The baseline
/// prediction is stored per row (each boundary's training majority),
/// so no majority label is needed here.
pub fn pool_metrics(sessions: &[Vec<ScoredRow>]) -> Result<PooledMetrics, DirectionError> {
    let mut n_scored = 0usize;
    let mut pooled = Confusion::default();
    let mut pooled_base_acc = 0.0f64;
    let mut used = 0usize;
    for rows in sessions {
        if rows.is_empty() {
            continue;
        }
        let n = rows.len() as f64;
        let table = Confusion::from_rows(rows).normalized(n);
        pooled.tp += table.tp;
        pooled.tn += table.tn;
        pooled.fp += table.fp;
        pooled.fn_ += table.fn_;
        let base_correct = rows.iter().filter(|r| r.baseline_pred == r.target).count() as f64;
        pooled_base_acc += base_correct / n;
        n_scored += rows.len();
        used += 1;
    }
    if used == 0 {
        return Err(DirectionError::UndefinedMetric(
            "no scored rows to pool".to_string(),
        ));
    }
    pooled.tp /= used as f64;
    pooled.tn /= used as f64;
    pooled.fp /= used as f64;
    pooled.fn_ /= used as f64;
    pooled_base_acc /= used as f64;
    let ba = pooled.balanced_accuracy().ok_or_else(|| {
        DirectionError::UndefinedMetric("a class is absent from pooled rows".to_string())
    })?;
    let accuracy = pooled.accuracy();
    let delta = accuracy - pooled_base_acc;
    let gamma = ba - 0.5;
    Ok(PooledMetrics {
        n_scored,
        accuracy,
        baseline_accuracy: pooled_base_acc,
        balanced_accuracy: ba,
        delta,
        gamma,
        s: (delta / 0.010).min(gamma / 0.020),
    })
}

/// Per-session endpoints for one development row.
#[derive(Debug, Clone)]
pub struct SessionMetrics {
    pub session: String,
    pub n_scored: usize,
    pub accuracy: f64,
    pub baseline_accuracy: f64,
    pub balanced_accuracy: Option<f64>,
    pub delta: f64,
    pub gamma: Option<f64>,
    pub s: Option<f64>,
}

fn session_metrics(session: &str, rows: &[ScoredRow]) -> SessionMetrics {
    let n = rows.len() as f64;
    let table = Confusion::from_rows(rows).normalized(n.max(1.0));
    let base_correct =
        rows.iter().filter(|r| r.baseline_pred == r.target).count() as f64 / n.max(1.0);
    let accuracy = table.accuracy();
    let ba = table.balanced_accuracy();
    let delta = accuracy - base_correct;
    let (gamma, s) = match ba {
        Some(b) => {
            let g = b - 0.5;
            (Some(g), Some((delta / 0.010).min(g / 0.020)))
        }
        None => (None, None),
    };
    SessionMetrics {
        session: session.to_string(),
        n_scored: rows.len(),
        accuracy,
        baseline_accuracy: base_correct,
        balanced_accuracy: ba,
        delta,
        gamma,
        s,
    }
}

/// Bootstrap intervals for pooled Delta and Gamma:
/// ((delta_lo, delta_hi), (gamma_lo, gamma_hi)), or `None` when the
/// draw budget was not reached (suppressed).
pub type BootstrapIntervals = Option<((f64, f64), (f64, f64))>;

/// One of the eight development rows (protocol section 6).
#[derive(Debug, Clone)]
pub struct DevRow {
    pub family: Family,
    pub rule: Rule,
    pub pooled: PooledMetrics,
    pub per_session: Vec<SessionMetrics>,
    /// Bootstrap intervals for the pooled Delta and Gamma, or
    /// `None` when the draw budget was not reached (suppressed).
    pub boot_300s: BootstrapIntervals,
    pub boot_600s: BootstrapIntervals,
}

/// Per-family selection detail (protocol section 6).
#[derive(Debug, Clone)]
pub struct FamilySelection {
    pub family: Family,
    pub selected_rule: Rule,
    pub s: f64,
    pub delta: f64,
    pub gamma: f64,
    /// Whether the family clears R (None for R itself).
    pub clears_r: Option<bool>,
    /// Whether the family clears F (None for R and F).
    pub clears_f: Option<bool>,
    pub qualifies: bool,
}

/// The section 6 selection outcome.
#[derive(Debug, Clone)]
pub struct SelectionOutcome {
    pub families: Vec<FamilySelection>,
    pub selected: Family,
    pub selected_rule: Rule,
}

/// The parent-clearing rule (protocol section 6): the child's
/// selected-rule S improves by at least 0.10, its Delta falls by at
/// most 0.002, its Gamma falls by at most 0.005, and its
/// within-session S strictly improves on at least three of the four
/// development slots (a slot without both classes casts no vote).
pub fn clears_parent(child: &DevRow, parent: &DevRow) -> bool {
    if child.pooled.s < parent.pooled.s + 0.10 {
        return false;
    }
    if child.pooled.delta < parent.pooled.delta - 0.002 {
        return false;
    }
    if child.pooled.gamma < parent.pooled.gamma - 0.005 {
        return false;
    }
    let mut votes = 0usize;
    for (c, p) in child.per_session.iter().zip(parent.per_session.iter()) {
        if let (Some(cs), Some(ps)) = (c.s, p.s) {
            if cs > ps {
                votes += 1;
            }
        }
    }
    votes >= 3
}

/// Runs the section 6 selection over the eight development rows.
/// Each family's operating rule is chosen by larger S, then larger
/// Delta, then the 0.50 rule. F must clear R; V and I must each clear
/// both F and R. Among qualifying families the selected one is chosen
/// by larger S, then larger Delta, then the simplicity order R, F, V,
/// I. R qualifies as the fallback whenever its pooled score is
/// defined.
pub fn select_family(rows: &[DevRow]) -> Result<SelectionOutcome, DirectionError> {
    // Selected rule per family.
    let mut by_family: HashMap<Family, &DevRow> = HashMap::new();
    for family in Family::all() {
        let cands: Vec<&DevRow> = rows.iter().filter(|r| r.family == family).collect();
        if cands.len() != 2 {
            return Err(DirectionError::UndefinedMetric(format!(
                "family {} has {} development rows, expected 2",
                family.name(),
                cands.len()
            )));
        }
        let mut best = cands[0];
        for &c in &cands[1..] {
            if c.pooled.s > best.pooled.s
                || (c.pooled.s == best.pooled.s && c.pooled.delta > best.pooled.delta)
                || (c.pooled.s == best.pooled.s
                    && c.pooled.delta == best.pooled.delta
                    && c.rule == Rule::P50
                    && best.rule != Rule::P50)
            {
                best = c;
            }
        }
        by_family.insert(family, best);
    }
    let r = by_family[&Family::R];
    let f = by_family[&Family::F];
    let v = by_family[&Family::V];
    let i = by_family[&Family::I];
    let f_clears_r = clears_parent(f, r);
    let v_clears_r = clears_parent(v, r);
    let v_clears_f = clears_parent(v, f);
    let i_clears_r = clears_parent(i, r);
    let i_clears_f = clears_parent(i, f);
    let fams = vec![
        FamilySelection {
            family: Family::R,
            selected_rule: r.rule,
            s: r.pooled.s,
            delta: r.pooled.delta,
            gamma: r.pooled.gamma,
            clears_r: None,
            clears_f: None,
            qualifies: r.pooled.s.is_finite(),
        },
        FamilySelection {
            family: Family::F,
            selected_rule: f.rule,
            s: f.pooled.s,
            delta: f.pooled.delta,
            gamma: f.pooled.gamma,
            clears_r: Some(f_clears_r),
            clears_f: None,
            qualifies: f_clears_r,
        },
        FamilySelection {
            family: Family::V,
            selected_rule: v.rule,
            s: v.pooled.s,
            delta: v.pooled.delta,
            gamma: v.pooled.gamma,
            clears_r: Some(v_clears_r),
            clears_f: Some(v_clears_f),
            qualifies: v_clears_r && v_clears_f,
        },
        FamilySelection {
            family: Family::I,
            selected_rule: i.rule,
            s: i.pooled.s,
            delta: i.pooled.delta,
            gamma: i.pooled.gamma,
            clears_r: Some(i_clears_r),
            clears_f: Some(i_clears_f),
            qualifies: i_clears_r && i_clears_f,
        },
    ];
    // Selected among qualifying: larger S, then larger Delta, then
    // the simplicity order R, F, V, I (the families vec is already
    // in that order, and a strict improvement is required to move).
    let mut selected: Option<&FamilySelection> = None;
    for fs in &fams {
        if !fs.qualifies {
            continue;
        }
        let better = match selected {
            None => true,
            Some(cur) => fs.s > cur.s || (fs.s == cur.s && fs.delta > cur.delta),
        };
        if better {
            selected = Some(fs);
        }
    }
    let sel = selected.ok_or_else(|| {
        DirectionError::UndefinedMetric("no family qualifies, not even the R fallback".to_string())
    })?;
    // Copy the selection out before moving `fams` into the outcome.
    let (sel_family, sel_rule) = (sel.family, sel.selected_rule);
    Ok(SelectionOutcome {
        families: fams,
        selected: sel_family,
        selected_rule: sel_rule,
    })
}

/// One time bin's aggregate counts for a family/rule row. The
/// bootstrap resamples bins, not rows, so a bin's contribution is
/// this small tuple.
#[derive(Debug, Clone, Copy, Default)]
struct BinAgg {
    n: f64,
    n1: f64,
    tp: f64,
    tn: f64,
    fp: f64,
    fn_: f64,
    base_correct: f64,
}

impl BinAgg {
    fn add(&mut self, o: &BinAgg) {
        self.n += o.n;
        self.n1 += o.n1;
        self.tp += o.tp;
        self.tn += o.tn;
        self.fp += o.fp;
        self.fn_ += o.fn_;
        self.base_correct += o.base_correct;
    }
}

fn bin_agg(rows: &[ScoredRow]) -> BinAgg {
    let mut a = BinAgg::default();
    for r in rows {
        a.n += 1.0;
        a.n1 += f64::from(r.target);
        match (r.target, r.model_pred) {
            (1, 1) => a.tp += 1.0,
            (0, 0) => a.tn += 1.0,
            (0, 1) => a.fp += 1.0,
            (1, 0) => a.fn_ += 1.0,
            _ => unreachable!(),
        }
        if r.baseline_pred == r.target {
            a.base_correct += 1.0;
        }
    }
    a
}

/// Endpoints from equal-session averaged normalized tables, given
/// one aggregate per drawn session instance. Returns `None` when a
/// class is absent from the pooled rows.
fn endpoints_from_instances(instances: &[BinAgg]) -> Option<(f64, f64)> {
    let mut tp = 0.0;
    let mut tn = 0.0;
    let mut fp = 0.0;
    let mut fn_ = 0.0;
    let mut base_acc = 0.0;
    let mut n1 = 0.0;
    let mut n = 0.0;
    let mut used = 0usize;
    for a in instances {
        if a.n == 0.0 {
            continue;
        }
        tp += a.tp / a.n;
        tn += a.tn / a.n;
        fp += a.fp / a.n;
        fn_ += a.fn_ / a.n;
        base_acc += a.base_correct / a.n;
        n1 += a.n1;
        n += a.n;
        used += 1;
    }
    if used == 0 || n1 == 0.0 || n1 == n {
        return None;
    }
    let k = used as f64;
    tp /= k;
    tn /= k;
    fp /= k;
    fn_ /= k;
    base_acc /= k;
    let r1 = tp + fn_;
    let r0 = tn + fp;
    if r1 == 0.0 || r0 == 0.0 {
        return None;
    }
    let ba = 0.5 * (tp / r1 + tn / r0);
    let delta = (tp + tn) - base_acc;
    Some((delta, ba - 0.5))
}

/// Epoch-aligned bins per session, in chronological order, for the
/// bootstrap: each bin holds the aggregate counts of the scored rows
/// whose decision time falls in it.
fn bins_for_sessions(sessions: &[Vec<ScoredRow>], block_ms: i64) -> Vec<Vec<BinAgg>> {
    let mut bins: Vec<Vec<BinAgg>> = Vec::new();
    for rows in sessions {
        let mut by_bin: HashMap<i64, Vec<ScoredRow>> = HashMap::new();
        let mut order: Vec<i64> = Vec::new();
        for r in rows {
            let b = r.time_ms / block_ms;
            if !by_bin.contains_key(&b) {
                order.push(b);
            }
            by_bin.entry(b).or_default().push(*r);
        }
        order.sort_unstable();
        bins.push(order.iter().map(|b| bin_agg(&by_bin[b])).collect());
    }
    bins
}

/// Percentile interval from sorted bootstrap draws, ranks
/// ceil(0.025 * B) and ceil(0.975 * B) (protocol section 8), or
/// `None` when the draw budget was not reached (suppressed).
fn intervals_from_draws(
    mut deltas: Vec<f64>,
    mut gammas: Vec<f64>,
    draws: usize,
) -> BootstrapIntervals {
    if deltas.len() < draws {
        return None;
    }
    deltas.sort_by(|a, b| a.partial_cmp(b).unwrap());
    gammas.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lo = (0.025 * draws as f64).ceil() as usize - 1;
    let hi = (0.975 * draws as f64).ceil() as usize - 1;
    Some(((deltas[lo], deltas[hi]), (gammas[lo], gammas[hi])))
}

/// Paired grouped bootstrap over precomputed bins: consecutive
/// session pairs are drawn with replacement, and within each drawn
/// session its original number of time bins is resampled with
/// replacement. Model and baseline are scored on the same resampled
/// rows (paired). Replicates missing a pooled class are redrawn;
/// when the draw budget cannot be reached in the attempt budget the
/// intervals are suppressed (`None`).
fn bootstrap_paired(
    bins: &[Vec<BinAgg>],
    seed: u64,
    draws: usize,
    max_attempts: usize,
) -> BootstrapIntervals {
    if bins.len() < 2 || !bins.len().is_multiple_of(2) {
        return None;
    }
    let n_pairs = bins.len() / 2;
    let mut rng = XorShift64Star(seed);
    let mut deltas: Vec<f64> = Vec::with_capacity(draws);
    let mut gammas: Vec<f64> = Vec::with_capacity(draws);
    let mut attempts = 0usize;
    while deltas.len() < draws && attempts < max_attempts {
        attempts += 1;
        let mut instances: Vec<BinAgg> = Vec::with_capacity(bins.len());
        for _ in 0..n_pairs {
            let pair = rng.next_below(n_pairs);
            for s in [2 * pair, 2 * pair + 1] {
                let session_bins = &bins[s];
                let mut agg = BinAgg::default();
                for _ in 0..session_bins.len() {
                    let b = rng.next_below(session_bins.len());
                    agg.add(&session_bins[b]);
                }
                instances.push(agg);
            }
        }
        if let Some((d, g)) = endpoints_from_instances(&instances) {
            deltas.push(d);
            gammas.push(g);
        }
    }
    intervals_from_draws(deltas, gammas, draws)
}

/// Fixed-cohort bootstrap over precomputed bins: every session is
/// present in every replicate, and within each session its original
/// number of time bins is resampled with replacement. Used for the
/// section 8 fixed-cohort and per-session diagnostics at 300 s.
fn bootstrap_fixed(
    bins: &[Vec<BinAgg>],
    seed: u64,
    draws: usize,
    max_attempts: usize,
) -> BootstrapIntervals {
    if bins.is_empty() {
        return None;
    }
    let mut rng = XorShift64Star(seed);
    let mut deltas: Vec<f64> = Vec::with_capacity(draws);
    let mut gammas: Vec<f64> = Vec::with_capacity(draws);
    let mut attempts = 0usize;
    while deltas.len() < draws && attempts < max_attempts {
        attempts += 1;
        let mut instances: Vec<BinAgg> = Vec::with_capacity(bins.len());
        for session_bins in bins {
            let mut agg = BinAgg::default();
            for _ in 0..session_bins.len() {
                let b = rng.next_below(session_bins.len());
                agg.add(&session_bins[b]);
            }
            instances.push(agg);
        }
        if let Some((d, g)) = endpoints_from_instances(&instances) {
            deltas.push(d);
            gammas.push(g);
        }
    }
    intervals_from_draws(deltas, gammas, draws)
}

/// Paired grouped bootstrap intervals for one development row, the
/// section 8 analog for four development sessions: consecutive
/// session pairs are drawn with replacement, and within each drawn
/// session its original number of time bins is resampled with
/// replacement. Model and baseline are scored on the same resampled
/// rows (paired). Replicates missing a pooled class are redrawn;
/// when 20,000 valid draws cannot be reached in 20,200 attempts the
/// intervals are suppressed (`None`).
pub fn paired_grouped_bootstrap(sessions: &[Vec<ScoredRow>], block_ms: i64) -> BootstrapIntervals {
    let bins = bins_for_sessions(sessions, block_ms);
    bootstrap_paired(
        &bins,
        DEV_BOOTSTRAP_SEED,
        DEV_BOOTSTRAP_DRAWS,
        DEV_BOOTSTRAP_MAX_ATTEMPTS,
    )
}

/// Paired grouped bootstrap intervals for the section 8
/// confirmation: five consecutive session pairs drawn with
/// replacement from (11,12), (13,14), (15,16), (17,18), (19,20),
/// within each drawn session its original number of time bins
/// resampled with replacement, 20,000 valid draws per block length
/// from the xorshift generator seeded 0xD1B54A32D192ED03. Replicates
/// missing a pooled class are redrawn; when 20,000 valid draws
/// cannot be reached in 20,200 attempts the intervals are suppressed
/// (`None`) and the study is incomplete.
pub fn paired_grouped_bootstrap_confirmation(
    sessions: &[Vec<ScoredRow>],
    block_ms: i64,
) -> BootstrapIntervals {
    let bins = bins_for_sessions(sessions, block_ms);
    bootstrap_paired(
        &bins,
        CONF_BOOTSTRAP_SEED,
        CONF_BOOTSTRAP_DRAWS,
        CONF_BOOTSTRAP_MAX_ATTEMPTS,
    )
}

/// Fixed-cohort and per-session diagnostic intervals at 300 s
/// (protocol section 8): the cohort is fixed, bins are resampled
/// within each session, and the section 8 diagnostic seed is used.
/// Descriptive only, never confirmatory evidence.
pub fn diagnostic_bootstrap(sessions: &[Vec<ScoredRow>], block_ms: i64) -> BootstrapIntervals {
    let bins = bins_for_sessions(sessions, block_ms);
    bootstrap_fixed(
        &bins,
        DEV_BOOTSTRAP_SEED,
        DEV_BOOTSTRAP_DRAWS,
        DEV_BOOTSTRAP_MAX_ATTEMPTS,
    )
}

/// One development boundary's detail for the report.
#[derive(Debug, Clone)]
pub struct BoundaryDetail {
    pub eval_slot: usize,
    pub eval_session: String,
    pub fit_sessions: Vec<String>,
    pub embargo_cutoff_ms: i64,
    pub n_fit_origins: usize,
    pub prevalence: f64,
    pub majority_label: u8,
}

/// A slot's frame together with its identity, for the run.
pub struct SlotInput<'a> {
    pub slot: usize,
    pub session_id: String,
    pub frame: &'a SessionFrame,
}

/// The full B3 development outcome.
#[derive(Debug)]
pub struct DevelopmentOutcome {
    pub boundaries: Vec<BoundaryDetail>,
    pub rows: Vec<DevRow>,
    pub selection: SelectionOutcome,
}

/// Scores one evaluation session's origins under one family and one
/// operating rule, returning the scored rows and the model's predicted
/// probabilities (kept alongside the rows so a deterministic replay
/// can compare probabilities, not just decisions).
fn score_session_rule(
    calc: &FrameCalc,
    origins: &[OriginData],
    ctx: &FitContext,
    fit: &RidgeFit,
    family: Family,
    rule: Rule,
) -> (Vec<ScoredRow>, Vec<f64>) {
    let standard = &ctx.standard[&family];
    let mut rows = Vec::with_capacity(origins.len());
    let mut probs = Vec::with_capacity(origins.len());
    for o in origins {
        let raw = raw_columns(calc, o, ctx, family);
        let x = standardize(&raw, standard);
        let p = predict_proba(fit, &x);
        let model_pred = match rule {
            Rule::P50 => u8::from(p > 0.50),
            Rule::Prevalence => u8::from(p > ctx.prevalence),
        };
        probs.push(p);
        rows.push(ScoredRow {
            time_ms: o.decision_ms,
            target: o.target,
            model_pred,
            baseline_pred: ctx.majority_label,
        });
    }
    (rows, probs)
}

/// The rows of a development run without the selection step: the
/// per-boundary details and one pooled, bootstrapped row per
/// requested (family, rule) pair.
#[derive(Debug)]
pub struct DevelopmentRows {
    pub boundaries: Vec<BoundaryDetail>,
    pub rows: Vec<DevRow>,
}

/// Runs the section 6 development ladder for exactly the requested
/// (family, rule) pairs: archives plus slots 01-06 predict slot 07,
/// then the fit pool grows by only the earlier campaign slots for
/// slots 08-10, under the 660-second embargo at each boundary.
/// Returns one pooled, bootstrapped row per requested pair and the
/// per-boundary details. No selection is made here.
pub fn run_development_rows(
    archives: &[&SessionFrame],
    slots: &[SlotInput],
    pairs: &[(Family, Rule)],
) -> Result<DevelopmentRows, DirectionError> {
    // Dev slots 07-10, in order, with their frames.
    let mut dev: Vec<&SlotInput> = slots
        .iter()
        .filter(|s| s.slot > FIT_LAST_SLOT && s.slot <= DEVELOPMENT_LAST_SLOT)
        .collect();
    dev.sort_by_key(|s| s.slot);
    if dev.len() != DEVELOPMENT_LAST_SLOT - FIT_LAST_SLOT {
        return Err(DirectionError::MissingSession(format!(
            "expected {} development slots, found {}",
            DEVELOPMENT_LAST_SLOT - FIT_LAST_SLOT,
            dev.len()
        )));
    }
    let fit_slots: Vec<&SlotInput> = slots.iter().filter(|s| s.slot <= FIT_LAST_SLOT).collect();
    if fit_slots.len() != FIT_LAST_SLOT {
        return Err(DirectionError::MissingSession(format!(
            "expected {FIT_LAST_SLOT} fit slots, found {}",
            fit_slots.len()
        )));
    }
    // Precompute origins (including VPIN) once per session.
    let mut arch_calcs: Vec<(FrameCalc, Vec<OriginData>)> = Vec::new();
    for f in archives {
        let calc = FrameCalc::new(f);
        let origins = calc.origins();
        arch_calcs.push((calc, origins));
    }
    let mut slot_calcs: HashMap<usize, (FrameCalc, Vec<OriginData>)> = HashMap::new();
    for s in slots {
        let calc = FrameCalc::new(s.frame);
        let origins = calc.origins();
        slot_calcs.insert(s.slot, (calc, origins));
    }
    // Per requested (family, rule) pair: scored rows per dev session,
    // in slot order.
    let mut scored: HashMap<(Family, Rule), Vec<Vec<ScoredRow>>> = HashMap::new();
    for pair in pairs {
        scored.insert(*pair, Vec::new());
    }
    // Each requested family is fit once per boundary.
    let mut families: Vec<Family> = Vec::new();
    for (family, _) in pairs {
        if !families.contains(family) {
            families.push(*family);
        }
    }
    let mut boundaries: Vec<BoundaryDetail> = Vec::new();
    for (k, eval) in dev.iter().enumerate() {
        let eval_start_ms = parse_session_start_unix(&eval.session_id).ok_or_else(|| {
            DirectionError::MissingSession(format!(
                "cannot parse scheduled start of {}",
                eval.session_id
            ))
        })? * 1000;
        let cutoff_ms = eval_start_ms - EMBARGO_SECS * 1000;
        // Fit pool: archives plus campaign slots 01..(06+k),
        // embargoed at the boundary cutoff.
        let mut fit_pool: Vec<(FrameCalc, Vec<OriginData>)> = Vec::new();
        let mut fit_ids: Vec<String> = Vec::new();
        for (calc, origins) in &arch_calcs {
            let kept: Vec<OriginData> = origins
                .iter()
                .filter(|o| o.decision_ms <= cutoff_ms)
                .cloned()
                .collect();
            fit_pool.push((FrameCalc::new(calc.frame), kept));
            fit_ids.push(calc.frame.session_id.clone());
        }
        for s in slots.iter().filter(|s| s.slot <= FIT_LAST_SLOT + k) {
            let (_, origins) = &slot_calcs[&s.slot];
            let kept: Vec<OriginData> = origins
                .iter()
                .filter(|o| o.decision_ms <= cutoff_ms)
                .cloned()
                .collect();
            fit_pool.push((FrameCalc::new(s.frame), kept));
            fit_ids.push(s.session_id.clone());
        }
        let n_fit: usize = fit_pool.iter().map(|(_, o)| o.len()).sum();
        let ctx = build_fit_context(&fit_pool)?;
        boundaries.push(BoundaryDetail {
            eval_slot: eval.slot,
            eval_session: eval.session_id.clone(),
            fit_sessions: fit_ids,
            embargo_cutoff_ms: cutoff_ms,
            n_fit_origins: n_fit,
            prevalence: ctx.prevalence,
            majority_label: ctx.majority_label,
        });
        // Fit each requested family once on the standardized fit rows,
        // then score the evaluation session under each requested rule.
        let (eval_calc, eval_origins) = &slot_calcs[&eval.slot];
        let mut fits: HashMap<Family, RidgeFit> = HashMap::new();
        for family in &families {
            let standard = &ctx.standard[family];
            let mut xs: Vec<Vec<f64>> = Vec::with_capacity(n_fit);
            let mut y: Vec<u8> = Vec::with_capacity(n_fit);
            for (calc, origins) in &fit_pool {
                for o in origins {
                    let raw = raw_columns(calc, o, &ctx, *family);
                    xs.push(standardize(&raw, standard));
                    y.push(o.target);
                }
            }
            let fit = fit_ridge_logistic(&xs, &y, ctx.prevalence)?;
            fits.insert(*family, fit);
        }
        for (family, rule) in pairs {
            let (rows, _) =
                score_session_rule(eval_calc, eval_origins, &ctx, &fits[family], *family, *rule);
            scored.get_mut(&(*family, *rule)).unwrap().push(rows);
        }
    }
    // Pool the requested rows and attach bootstrap intervals.
    let mut rows: Vec<DevRow> = Vec::new();
    for (family, rule) in pairs {
        let (family, rule) = (*family, *rule);
        let sessions = scored.remove(&(family, rule)).unwrap();
        let pooled = pool_metrics(&sessions)?;
        let per_session: Vec<SessionMetrics> = sessions
            .iter()
            .zip(dev.iter())
            .map(|(rows, d)| session_metrics(&d.session_id, rows))
            .collect();
        let boot_300s = paired_grouped_bootstrap(&sessions, 300_000);
        let boot_600s = paired_grouped_bootstrap(&sessions, 600_000);
        rows.push(DevRow {
            family,
            rule,
            pooled,
            per_session,
            boot_300s,
            boot_600s,
        });
    }
    Ok(DevelopmentRows { boundaries, rows })
}

/// Runs the section 6 development ladder over all eight registered
/// variants (four families times two operating rules) and applies
/// the section 6 selection. See `run_development_rows` for the
/// ladder itself.
pub fn run_development(
    archives: &[&SessionFrame],
    slots: &[SlotInput],
) -> Result<DevelopmentOutcome, DirectionError> {
    let mut pairs: Vec<(Family, Rule)> = Vec::new();
    for family in Family::all() {
        for rule in Rule::all() {
            pairs.push((family, rule));
        }
    }
    let outcome = run_development_rows(archives, slots, &pairs)?;
    let selection = select_family(&outcome.rows)?;
    Ok(DevelopmentOutcome {
        boundaries: outcome.boundaries,
        rows: outcome.rows,
        selection,
    })
}

/// Scores one frozen (family, rule) configuration on the
/// development ladder, without selecting among families or rules.
/// This is the cross-asset transfer path: the configuration arrives
/// frozen (for the ETH transfer, family F under p>0.50, selected on
/// BTC) and only its coefficients are refit, on the supplied
/// training data, at each boundary. The arithmetic is identical to
/// `run_development` — same boundaries, embargo, pooling and
/// bootstrap — because it runs through the same function; only the
/// pair list holds a single frozen pair.
pub fn run_transfer_validation(
    archives: &[&SessionFrame],
    slots: &[SlotInput],
    family: Family,
    rule: Rule,
) -> Result<(Vec<BoundaryDetail>, DevRow), DirectionError> {
    let outcome = run_development_rows(archives, slots, &[(family, rule)])?;
    let row = outcome
        .rows
        .into_iter()
        .next()
        .expect("one requested pair yields one row");
    Ok((outcome.boundaries, row))
}

/// The section 7 frozen artifact: the single selected family and
/// rule refit on the six archives plus slots 01-10 under the frozen
/// embargo, with everything the confirmation step needs to score
/// without refitting: the fitted transforms, the fitted ridge
/// parameters, the split memberships, the code, data and feature
/// hashes, the exact version and the timestamp.
#[derive(Debug, Clone)]
pub struct FrozenArtifact {
    pub family: Family,
    pub rule: Rule,
    pub protocol_digest: String,
    pub code_hash: String,
    pub data_hash: String,
    pub feature_hash: String,
    pub version: String,
    pub timestamp_unix: i64,
    pub fit_sessions: Vec<String>,
    pub fit_session_hashes: Vec<(String, String)>,
    pub embargo_cutoff_ms: i64,
    pub n_fit_origins: usize,
    pub prevalence: f64,
    pub majority_label: u8,
    pub ar_intercept: f64,
    pub ar_coefs: [f64; AR_ORDER],
    pub burst_threshold: f64,
    /// (mean, population sd) per final column of the frozen family.
    pub standard: Vec<(f64, f64)>,
    pub slopes: Vec<f64>,
    pub intercept: f64,
    pub n_accepted: usize,
    pub grad_inf_norm: f64,
}

fn json_f64(value: &serde_json::Value, field: &str) -> Result<f64, DirectionError> {
    value
        .get(field)
        .and_then(|v| v.as_f64())
        .ok_or_else(|| DirectionError::InvalidArtifact(format!("missing or non-numeric {field}")))
}

fn json_f64_array(
    value: &serde_json::Value,
    field: &str,
    len: usize,
) -> Result<Vec<f64>, DirectionError> {
    let arr = value
        .get(field)
        .and_then(|v| v.as_array())
        .ok_or_else(|| DirectionError::InvalidArtifact(format!("missing or non-array {field}")))?;
    if arr.len() != len {
        return Err(DirectionError::InvalidArtifact(format!(
            "{field} has {} entries, expected {len}",
            arr.len()
        )));
    }
    let mut out = Vec::with_capacity(len);
    for v in arr {
        out.push(
            v.as_f64()
                .ok_or_else(|| DirectionError::InvalidArtifact(format!("non-numeric {field}")))?,
        );
    }
    Ok(out)
}

impl FrozenArtifact {
    /// Serializes the artifact for the confirmation report. Floats
    /// serialize with serde_json's exact round-trip representation,
    /// so a deserialized artifact replays the original decisions
    /// exactly and probabilities to within 1e-12.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "schema": "toxicflow.direction-frozen-artifact/v1",
            "family": self.family.name(),
            "rule": self.rule.name(),
            "protocol_digest": self.protocol_digest,
            "code_hash": self.code_hash,
            "data_hash": self.data_hash,
            "feature_hash": self.feature_hash,
            "version": self.version,
            "timestamp_unix": self.timestamp_unix,
            "fit_sessions": self.fit_sessions,
            "fit_session_hashes": self.fit_session_hashes.iter().map(|(id, h)| {
                serde_json::json!({"session": id, "content_hash": h})
            }).collect::<Vec<_>>(),
            "embargo_cutoff_ms": self.embargo_cutoff_ms,
            "n_fit_origins": self.n_fit_origins,
            "prevalence": self.prevalence,
            "majority_label": self.majority_label,
            "ar_intercept": self.ar_intercept,
            "ar_coefs": self.ar_coefs,
            "burst_threshold": self.burst_threshold,
            "standard": self.standard.iter().map(|(m, s)| serde_json::json!([m, s])).collect::<Vec<_>>(),
            "slopes": self.slopes,
            "intercept": self.intercept,
            "n_accepted": self.n_accepted,
            "grad_inf_norm": self.grad_inf_norm,
        })
    }

    /// Rebuilds an artifact from its serialized form, validating the
    /// family, rule, digest and array lengths. A malformed record is
    /// an error, not a result.
    pub fn from_json(value: &serde_json::Value) -> Result<FrozenArtifact, DirectionError> {
        let bad =
            |field: &str| DirectionError::InvalidArtifact(format!("missing or invalid {field}"));
        let family = match value.get("family").and_then(|v| v.as_str()) {
            Some("R") => Family::R,
            Some("F") => Family::F,
            Some("V") => Family::V,
            Some("I") => Family::I,
            _ => return Err(bad("family")),
        };
        let rule = match value.get("rule").and_then(|v| v.as_str()) {
            Some("p>0.50") => Rule::P50,
            Some("p>prevalence") => Rule::Prevalence,
            _ => return Err(bad("rule")),
        };
        let protocol_digest = value
            .get("protocol_digest")
            .and_then(|v| v.as_str())
            .ok_or_else(|| bad("protocol_digest"))?;
        if protocol_digest != PROTOCOL_DIGEST {
            return Err(DirectionError::InvalidArtifact(
                "protocol digest does not match the registered protocol".to_string(),
            ));
        }
        let ar_coefs = json_f64_array(value, "ar_coefs", AR_ORDER)?;
        let slopes = json_f64_array(value, "slopes", family.n_columns())?;
        let standard_raw = value
            .get("standard")
            .and_then(|v| v.as_array())
            .ok_or_else(|| bad("standard"))?;
        if standard_raw.len() != family.n_columns() {
            return Err(DirectionError::InvalidArtifact(format!(
                "standard has {} entries, expected {}",
                standard_raw.len(),
                family.n_columns()
            )));
        }
        let mut standard = Vec::with_capacity(family.n_columns());
        for pair in standard_raw {
            let pair = pair.as_array().ok_or_else(|| bad("standard"))?;
            if pair.len() != 2 {
                return Err(bad("standard"));
            }
            let mean = pair[0].as_f64().ok_or_else(|| bad("standard"))?;
            let sd = pair[1].as_f64().ok_or_else(|| bad("standard"))?;
            standard.push((mean, sd));
        }
        let fit_sessions = value
            .get("fit_sessions")
            .and_then(|v| v.as_array())
            .ok_or_else(|| bad("fit_sessions"))?
            .iter()
            .map(|v| {
                v.as_str()
                    .map(|s| s.to_string())
                    .ok_or_else(|| bad("fit_sessions"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let hashes_raw = value
            .get("fit_session_hashes")
            .and_then(|v| v.as_array())
            .ok_or_else(|| bad("fit_session_hashes"))?;
        let mut fit_session_hashes = Vec::with_capacity(hashes_raw.len());
        for entry in hashes_raw {
            let id = entry
                .get("session")
                .and_then(|v| v.as_str())
                .ok_or_else(|| bad("fit_session_hashes"))?;
            let hash = entry
                .get("content_hash")
                .and_then(|v| v.as_str())
                .ok_or_else(|| bad("fit_session_hashes"))?;
            fit_session_hashes.push((id.to_string(), hash.to_string()));
        }
        let mut ar = [0.0f64; AR_ORDER];
        ar.copy_from_slice(&ar_coefs);
        Ok(FrozenArtifact {
            family,
            rule,
            protocol_digest: protocol_digest.to_string(),
            code_hash: value
                .get("code_hash")
                .and_then(|v| v.as_str())
                .ok_or_else(|| bad("code_hash"))?
                .to_string(),
            data_hash: value
                .get("data_hash")
                .and_then(|v| v.as_str())
                .ok_or_else(|| bad("data_hash"))?
                .to_string(),
            feature_hash: value
                .get("feature_hash")
                .and_then(|v| v.as_str())
                .ok_or_else(|| bad("feature_hash"))?
                .to_string(),
            version: value
                .get("version")
                .and_then(|v| v.as_str())
                .ok_or_else(|| bad("version"))?
                .to_string(),
            timestamp_unix: value
                .get("timestamp_unix")
                .and_then(|v| v.as_i64())
                .ok_or_else(|| bad("timestamp_unix"))?,
            fit_sessions,
            fit_session_hashes,
            embargo_cutoff_ms: value
                .get("embargo_cutoff_ms")
                .and_then(|v| v.as_i64())
                .ok_or_else(|| bad("embargo_cutoff_ms"))?,
            n_fit_origins: value
                .get("n_fit_origins")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| bad("n_fit_origins"))? as usize,
            prevalence: json_f64(value, "prevalence")?,
            majority_label: value
                .get("majority_label")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| bad("majority_label"))? as u8,
            ar_intercept: json_f64(value, "ar_intercept")?,
            ar_coefs: ar,
            burst_threshold: json_f64(value, "burst_threshold")?,
            standard,
            slopes,
            intercept: json_f64(value, "intercept")?,
            n_accepted: value
                .get("n_accepted")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| bad("n_accepted"))? as usize,
            grad_inf_norm: json_f64(value, "grad_inf_norm")?,
        })
    }
}

/// Canonical serialization of the fitted feature transforms, for the
/// artifact's feature hash. Floats use a fixed 17-significant-digit
/// exponent form, so the hash is stable across runs.
fn canonical_feature_string(family: Family, rule: Rule, ctx: &FitContext) -> String {
    let mut s = String::new();
    s.push_str(&format!("family={}\nrule={}\n", family.name(), rule.name()));
    s.push_str(&format!("ar_intercept={:.17e}\n", ctx.ar_intercept));
    for (k, c) in ctx.ar_coefs.iter().enumerate() {
        s.push_str(&format!("ar_coef[{k}]={c:.17e}\n"));
    }
    s.push_str(&format!("burst_threshold={:.17e}\n", ctx.burst_threshold));
    s.push_str(&format!("prevalence={:.17e}\n", ctx.prevalence));
    s.push_str(&format!("majority_label={}\n", ctx.majority_label));
    for (j, (mean, sd)) in ctx.standard[&family].iter().enumerate() {
        s.push_str(&format!("standard[{j}]={mean:.17e},{sd:.17e}\n"));
    }
    s
}

/// The section 7 freeze: refit the single selected family and rule
/// on the archives plus slots 01-10 under the frozen embargo,
/// recomputing its transforms, majority rule and threshold as
/// registered. The fit pool must already be embargoed: every origin
/// must decide at or before `embargo_cutoff_ms`, and an origin past
/// the cutoff is an error, not a result. The artifact records the
/// code hash supplied by the caller, the data hash over the fit
/// sessions' content hashes, the feature hash over the fitted
/// transforms, the exact crate version and the timestamp.
#[allow(clippy::too_many_arguments)]
pub fn freeze_artifact(
    train: &[(FrameCalc<'_>, Vec<OriginData>)],
    fit_ids: &[String],
    fit_hashes: &[(String, String)],
    embargo_cutoff_ms: i64,
    family: Family,
    rule: Rule,
    protocol_digest: &str,
    code_hash: &str,
    timestamp_unix: i64,
) -> Result<FrozenArtifact, DirectionError> {
    if protocol_digest != PROTOCOL_DIGEST {
        return Err(DirectionError::InvalidArtifact(
            "protocol digest does not match the registered protocol".to_string(),
        ));
    }
    if train.len() != fit_ids.len() || train.len() != fit_hashes.len() {
        return Err(DirectionError::UnexpectedSession(format!(
            "freeze fit pool has {} sessions, {} ids and {} hashes",
            train.len(),
            fit_ids.len(),
            fit_hashes.len()
        )));
    }
    for (i, (id, _)) in fit_hashes.iter().enumerate() {
        if *id != fit_ids[i] {
            return Err(DirectionError::UnexpectedSession(
                "fit session hash order does not match the fit session list".to_string(),
            ));
        }
    }
    for (_, origins) in train {
        for o in origins {
            if o.decision_ms > embargo_cutoff_ms {
                return Err(DirectionError::InvalidArtifact(format!(
                    "fit pool holds an origin at {} past the embargo cutoff {embargo_cutoff_ms}",
                    o.decision_ms
                )));
            }
        }
    }
    let ctx = build_fit_context(train)?;
    let n_fit: usize = train.iter().map(|(_, o)| o.len()).sum();
    let standard = ctx.standard[&family].clone();
    let mut xs: Vec<Vec<f64>> = Vec::with_capacity(n_fit);
    let mut y: Vec<u8> = Vec::with_capacity(n_fit);
    for (calc, origins) in train {
        for o in origins {
            let raw = raw_columns(calc, o, &ctx, family);
            xs.push(standardize(&raw, &standard));
            y.push(o.target);
        }
    }
    let fit = fit_ridge_logistic(&xs, &y, ctx.prevalence)?;
    let mut data_lines = String::new();
    for (id, hash) in fit_hashes {
        data_lines.push_str(&format!("{id}:{hash}\n"));
    }
    let data_hash = sha256_hex(data_lines.as_bytes());
    let feature_hash = sha256_hex(canonical_feature_string(family, rule, &ctx).as_bytes());
    Ok(FrozenArtifact {
        family,
        rule,
        protocol_digest: protocol_digest.to_string(),
        code_hash: code_hash.to_string(),
        data_hash,
        feature_hash,
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp_unix,
        fit_sessions: fit_ids.to_vec(),
        fit_session_hashes: fit_hashes.to_vec(),
        embargo_cutoff_ms,
        n_fit_origins: n_fit,
        prevalence: ctx.prevalence,
        majority_label: ctx.majority_label,
        ar_intercept: ctx.ar_intercept,
        ar_coefs: ctx.ar_coefs,
        burst_threshold: ctx.burst_threshold,
        standard,
        slopes: fit.slopes,
        intercept: fit.intercept,
        n_accepted: fit.n_accepted,
        grad_inf_norm: fit.grad_inf_norm,
    })
}

/// Scores origins with a frozen artifact only: no refitting, no
/// tuning, no change to the model. The context and ridge fit are
/// rebuilt from the artifact's recorded values, so a deserialized
/// artifact scores identically to the frozen one.
pub fn score_with_artifact(
    artifact: &FrozenArtifact,
    calc: &FrameCalc,
    origins: &[OriginData],
) -> (Vec<ScoredRow>, Vec<f64>) {
    let mut standard = HashMap::new();
    standard.insert(artifact.family, artifact.standard.clone());
    let ctx = FitContext {
        ar_intercept: artifact.ar_intercept,
        ar_coefs: artifact.ar_coefs,
        burst_threshold: artifact.burst_threshold,
        prevalence: artifact.prevalence,
        majority_label: artifact.majority_label,
        standard,
    };
    let fit = RidgeFit {
        slopes: artifact.slopes.clone(),
        intercept: artifact.intercept,
        n_accepted: artifact.n_accepted,
        grad_inf_norm: artifact.grad_inf_norm,
    };
    score_session_rule(calc, origins, &ctx, &fit, artifact.family, artifact.rule)
}

/// One calibration bin: a fixed predicted-probability decile with its
/// row count, mean predicted probability and mean realized target.
#[derive(Debug, Clone, Copy)]
pub struct CalibrationBin {
    pub lo: f64,
    pub hi: f64,
    pub n: usize,
    pub mean_p: Option<f64>,
    pub mean_y: Option<f64>,
}

/// Per-session confirmation diagnostics (protocol section 8):
/// scores, recalls, prevalence, flat counts, coverage, Brier and
/// calibration diagnostics, plus the per-session 300 s diagnostic
/// bootstrap interval.
#[derive(Debug, Clone)]
pub struct ConfirmationSession {
    pub slot: usize,
    pub session: String,
    pub n_scored: usize,
    pub n_eligible: usize,
    pub coverage: f64,
    pub accuracy: f64,
    pub baseline_accuracy: f64,
    pub balanced_accuracy: Option<f64>,
    pub delta: f64,
    pub gamma: Option<f64>,
    pub s: Option<f64>,
    pub recall_pos: Option<f64>,
    pub recall_neg: Option<f64>,
    pub prevalence: f64,
    pub flat_count: usize,
    pub brier: f64,
    pub calibration: Vec<CalibrationBin>,
    pub boot_300s: BootstrapIntervals,
}

/// One leave-one-pair-out recomputation: the pooled endpoints over
/// the eight sessions that remain after dropping one adjacent pair.
#[derive(Debug, Clone, Copy)]
pub struct LeaveOnePairOut {
    pub dropped_pair: (usize, usize),
    pub delta: f64,
    pub gamma: f64,
}

/// One audit-ledger bin: the aggregate counts of the scored rows in
/// one epoch-aligned time bin of one session, at one block length
/// (protocol section 10).
#[derive(Debug, Clone)]
pub struct AuditBin {
    pub session: String,
    pub block_ms: i64,
    pub bin: i64,
    pub n: usize,
    pub n1: usize,
    pub tp: usize,
    pub tn: usize,
    pub fp: usize,
    pub fn_: usize,
    pub base_correct: usize,
}

/// The section 8 confirmation outcome kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfirmationOutcomeKind {
    Win,
    Partial,
    Null,
}

impl ConfirmationOutcomeKind {
    pub fn name(self) -> &'static str {
        match self {
            ConfirmationOutcomeKind::Win => "win",
            ConfirmationOutcomeKind::Partial => "partial",
            ConfirmationOutcomeKind::Null => "null",
        }
    }
}

/// The full section 8 confirmation outcome.
#[derive(Debug)]
pub struct ConfirmationOutcome {
    pub pooled: PooledMetrics,
    pub per_session: Vec<ConfirmationSession>,
    /// Predicted probabilities per session, parallel to the scored
    /// rows, for the deterministic replay.
    pub probabilities: Vec<Vec<f64>>,
    /// Paired confirmation intervals (section 8 seed).
    pub boot_300s: BootstrapIntervals,
    /// Paired confirmation intervals (section 8 seed).
    pub boot_600s: BootstrapIntervals,
    /// Fixed-cohort diagnostic intervals at 300 s (diagnostic seed).
    pub fixed_cohort_300s: BootstrapIntervals,
    pub leave_one_pair_out: Vec<LeaveOnePairOut>,
    pub lopo_delta_range: (f64, f64),
    pub lopo_gamma_range: (f64, f64),
    pub audit_300s: Vec<AuditBin>,
    pub audit_600s: Vec<AuditBin>,
    pub outcome: ConfirmationOutcomeKind,
    /// Which endpoint cleared, for the partial wording ("Delta" or
    /// "Balanced accuracy"); `None` for a win or a null.
    pub cleared_endpoint: Option<&'static str>,
}

/// A holdout slot's frame together with its identity, for the run.
pub struct HoldoutInput<'a> {
    pub slot: usize,
    pub session_id: String,
    pub frame: &'a SessionFrame,
}

/// Fixed decile calibration table over predicted probabilities:
/// ten bins [0, 0.1) through [0.9, 1.0], each with its row count,
/// mean predicted probability and mean realized target.
fn calibration_table(probs: &[f64], rows: &[ScoredRow]) -> Vec<CalibrationBin> {
    let mut bins: Vec<CalibrationBin> = (0..10)
        .map(|j| CalibrationBin {
            lo: j as f64 / 10.0,
            hi: (j + 1) as f64 / 10.0,
            n: 0,
            mean_p: None,
            mean_y: None,
        })
        .collect();
    let mut sums: Vec<(f64, f64)> = vec![(0.0, 0.0); 10];
    for (p, r) in probs.iter().zip(rows.iter()) {
        let j = (*p * 10.0).floor() as usize;
        let j = j.min(9);
        bins[j].n += 1;
        sums[j].0 += *p;
        sums[j].1 += f64::from(r.target);
    }
    for (j, b) in bins.iter_mut().enumerate() {
        if b.n > 0 {
            b.mean_p = Some(sums[j].0 / b.n as f64);
            b.mean_y = Some(sums[j].1 / b.n as f64);
        }
    }
    bins
}

/// Audit ledger for one block length: per session, per
/// epoch-aligned bin, the aggregate counts of the scored rows in it,
/// in chronological bin order (protocol section 10).
fn audit_ledger(sessions: &[(String, Vec<ScoredRow>)], block_ms: i64) -> Vec<AuditBin> {
    let mut out = Vec::new();
    for (session, rows) in sessions {
        let mut by_bin: HashMap<i64, Vec<&ScoredRow>> = HashMap::new();
        let mut order: Vec<i64> = Vec::new();
        for r in rows {
            let b = r.time_ms / block_ms;
            if !by_bin.contains_key(&b) {
                order.push(b);
            }
            by_bin.entry(b).or_default().push(r);
        }
        order.sort_unstable();
        for b in order {
            let mut a = AuditBin {
                session: session.clone(),
                block_ms,
                bin: b,
                n: 0,
                n1: 0,
                tp: 0,
                tn: 0,
                fp: 0,
                fn_: 0,
                base_correct: 0,
            };
            for r in &by_bin[&b] {
                a.n += 1;
                a.n1 += usize::from(r.target);
                match (r.target, r.model_pred) {
                    (1, 1) => a.tp += 1,
                    (0, 0) => a.tn += 1,
                    (0, 1) => a.fp += 1,
                    (1, 0) => a.fn_ += 1,
                    _ => unreachable!(),
                }
                if r.baseline_pred == r.target {
                    a.base_correct += 1;
                }
            }
            out.push(a);
        }
    }
    out
}

/// Applies the section 8 outcome gates exactly as written. The win
/// gate needs all ten assigned sessions present, Delta at least
/// +0.010, balanced accuracy at least 0.520, and the 95% lower bounds
/// for both Delta and Gamma strictly positive at both block lengths.
/// A partial outcome is exactly one endpoint clearing its point
/// margin and both of its interval checks while the win gate does
/// not hold. Neither endpoint clearing, on a valid complete run, is
/// a null. Suppressed intervals or a missing session are not a
/// result: they are an error, and the study is incomplete.
pub fn classify_confirmation(
    pooled: &PooledMetrics,
    boot_300s: BootstrapIntervals,
    boot_600s: BootstrapIntervals,
    n_sessions: usize,
) -> Result<(ConfirmationOutcomeKind, Option<&'static str>), DirectionError> {
    let expected = CONFIRMATION_LAST_SLOT - CONFIRMATION_FIRST_SLOT + 1;
    if n_sessions != expected {
        return Err(DirectionError::MissingSession(format!(
            "expected {expected} holdout sessions, found {n_sessions}"
        )));
    }
    let (d300, g300) = boot_300s.ok_or_else(|| {
        DirectionError::UndefinedMetric("300 s confirmation intervals suppressed".to_string())
    })?;
    let (d600, g600) = boot_600s.ok_or_else(|| {
        DirectionError::UndefinedMetric("600 s confirmation intervals suppressed".to_string())
    })?;
    let delta_clears = pooled.delta >= 0.010 && d300.0 > 0.0 && d600.0 > 0.0;
    let gamma_clears = pooled.balanced_accuracy >= 0.520 && g300.0 > 0.0 && g600.0 > 0.0;
    match (delta_clears, gamma_clears) {
        (true, true) => Ok((ConfirmationOutcomeKind::Win, None)),
        (true, false) => Ok((ConfirmationOutcomeKind::Partial, Some("Delta"))),
        (false, true) => Ok((ConfirmationOutcomeKind::Partial, Some("Balanced accuracy"))),
        (false, false) => Ok((ConfirmationOutcomeKind::Null, None)),
    }
}

/// Runs the section 8 confirmation: scores slots 11-20 exactly once
/// with the single frozen artifact and the frozen majority rule. No
/// other variant is scored on the holdout, not even as a diagnostic.
/// Endpoints come from the equal-session averaged confusion tables,
/// as in section 6, with per-session scores, recalls, prevalence,
/// flat counts, coverage, and Brier and calibration diagnostics
/// reported alongside. Intervals are the paired bootstrap intervals
/// at 300 s and 600 s under the section 8 seed; the fixed-cohort
/// diagnostic at 300 s uses the diagnostic seed, as do the
/// per-session diagnostic intervals. Leave-one-pair-out ranges are
/// reported over the five adjacent pairs. A suppressed confirmatory
/// interval, a missing or misidentified holdout session, or an
/// undefined pooled metric is an error, and the study is incomplete.
pub fn run_confirmation(
    artifact: &FrozenArtifact,
    holdout: &[HoldoutInput],
) -> Result<ConfirmationOutcome, DirectionError> {
    let expected = CONFIRMATION_LAST_SLOT - CONFIRMATION_FIRST_SLOT + 1;
    if holdout.len() != expected {
        return Err(DirectionError::MissingSession(format!(
            "expected {expected} holdout sessions, found {}",
            holdout.len()
        )));
    }
    let mut sorted: Vec<&HoldoutInput> = holdout.iter().collect();
    sorted.sort_by_key(|h| h.slot);
    for (k, h) in sorted.iter().enumerate() {
        let slot = CONFIRMATION_FIRST_SLOT + k;
        if h.slot != slot {
            return Err(DirectionError::MissingSession(format!(
                "expected holdout slot {slot}, found slot {}",
                h.slot
            )));
        }
        if h.session_id != CAMPAIGN_SLOT_IDS[slot - 1] {
            return Err(DirectionError::UnexpectedSession(format!(
                "holdout slot {slot} is {}, expected {}",
                h.session_id,
                CAMPAIGN_SLOT_IDS[slot - 1]
            )));
        }
    }
    let mut per_session: Vec<ConfirmationSession> = Vec::new();
    let mut all_rows: Vec<Vec<ScoredRow>> = Vec::new();
    let mut probabilities: Vec<Vec<f64>> = Vec::new();
    for h in &sorted {
        let calc = FrameCalc::new(h.frame);
        let origins = calc.origins();
        let n_eligible = origins.len();
        let (rows, probs) = score_with_artifact(artifact, &calc, &origins);
        let sm = session_metrics(&h.session_id, &rows);
        let table = Confusion::from_rows(&rows);
        let recall_pos = if table.tp + table.fn_ > 0.0 {
            Some(table.tp / (table.tp + table.fn_))
        } else {
            None
        };
        let recall_neg = if table.tn + table.fp > 0.0 {
            Some(table.tn / (table.tn + table.fp))
        } else {
            None
        };
        let prevalence =
            rows.iter().map(|r| f64::from(r.target)).sum::<f64>() / rows.len().max(1) as f64;
        let flat_count = origins
            .iter()
            .filter(|o| h.frame.mid[o.bar + TARGET_HORIZON_BARS] == h.frame.mid[o.bar])
            .count();
        let brier = probs
            .iter()
            .zip(rows.iter())
            .map(|(p, r)| (p - f64::from(r.target)).powi(2))
            .sum::<f64>()
            / rows.len().max(1) as f64;
        let calibration = calibration_table(&probs, &rows);
        let boot_300s = diagnostic_bootstrap(std::slice::from_ref(&rows), 300_000);
        per_session.push(ConfirmationSession {
            slot: h.slot,
            session: h.session_id.clone(),
            n_scored: rows.len(),
            n_eligible,
            coverage: rows.len() as f64 / n_eligible.max(1) as f64,
            accuracy: sm.accuracy,
            baseline_accuracy: sm.baseline_accuracy,
            balanced_accuracy: sm.balanced_accuracy,
            delta: sm.delta,
            gamma: sm.gamma,
            s: sm.s,
            recall_pos,
            recall_neg,
            prevalence,
            flat_count,
            brier,
            calibration,
            boot_300s,
        });
        all_rows.push(rows);
        probabilities.push(probs);
    }
    let pooled = pool_metrics(&all_rows)?;
    let boot_300s = paired_grouped_bootstrap_confirmation(&all_rows, 300_000);
    let boot_600s = paired_grouped_bootstrap_confirmation(&all_rows, 600_000);
    let fixed_cohort_300s = diagnostic_bootstrap(&all_rows, 300_000);
    let mut leave_one_pair_out: Vec<LeaveOnePairOut> = Vec::new();
    for p in 0..5 {
        let keep: Vec<Vec<ScoredRow>> = all_rows
            .iter()
            .enumerate()
            .filter(|(i, _)| *i / 2 != p)
            .map(|(_, r)| r.clone())
            .collect();
        let m = pool_metrics(&keep)?;
        leave_one_pair_out.push(LeaveOnePairOut {
            dropped_pair: (
                CONFIRMATION_FIRST_SLOT + 2 * p,
                CONFIRMATION_FIRST_SLOT + 2 * p + 1,
            ),
            delta: m.delta,
            gamma: m.gamma,
        });
    }
    let lopo_delta_range = (
        leave_one_pair_out
            .iter()
            .map(|l| l.delta)
            .fold(f64::INFINITY, f64::min),
        leave_one_pair_out
            .iter()
            .map(|l| l.delta)
            .fold(f64::NEG_INFINITY, f64::max),
    );
    let lopo_gamma_range = (
        leave_one_pair_out
            .iter()
            .map(|l| l.gamma)
            .fold(f64::INFINITY, f64::min),
        leave_one_pair_out
            .iter()
            .map(|l| l.gamma)
            .fold(f64::NEG_INFINITY, f64::max),
    );
    let named: Vec<(String, Vec<ScoredRow>)> = sorted
        .iter()
        .zip(all_rows.iter())
        .map(|(h, r)| (h.session_id.clone(), r.clone()))
        .collect();
    let audit_300s = audit_ledger(&named, 300_000);
    let audit_600s = audit_ledger(&named, 600_000);
    let (outcome, cleared_endpoint) =
        classify_confirmation(&pooled, boot_300s, boot_600s, sorted.len())?;
    Ok(ConfirmationOutcome {
        pooled,
        per_session,
        probabilities,
        boot_300s,
        boot_600s,
        fixed_cohort_300s,
        leave_one_pair_out,
        lopo_delta_range,
        lopo_gamma_range,
        audit_300s,
        audit_600s,
        outcome,
        cleared_endpoint,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::volume_buckets_sized;

    /// Tiny deterministic PRNG for test fixtures (LCG, fixed seed).
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }

        fn uniform(&mut self) -> f64 {
            (self.next() % 1_000_000) as f64 / 1_000_000.0
        }
    }

    #[test]
    fn cholesky_solve_known_answer() {
        // [[4, 2], [2, 3]] x = [6, 5]  =>  x = [1, 1].
        let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
        let x = cholesky_solve(&a, &[6.0, 5.0]).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12);
        assert!((x[1] - 1.0).abs() < 1e-12);
        // Not positive-definite: no solution.
        let bad = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
        assert!(cholesky_solve(&bad, &[1.0, 1.0]).is_none());
    }

    #[test]
    fn ridge_logistic_converges_to_the_optimum() {
        // Two well-separated clusters: the optimum is interior and
        // the gradient must vanish there.
        let mut rng = Lcg(42);
        let mut xs: Vec<Vec<f64>> = Vec::new();
        let mut y: Vec<u8> = Vec::new();
        for _ in 0..400 {
            let v = rng.uniform() * 2.0 - 1.0;
            xs.push(vec![v, rng.uniform()]);
            y.push(u8::from(v > 0.0));
        }
        // A few label flips so the fit is not degenerate.
        y[0] = 1 - y[0];
        y[1] = 1 - y[1];
        let fit = fit_ridge_logistic(&xs, &y, 0.5).unwrap();
        assert!(fit.grad_inf_norm <= NEWTON_GRAD_TOL);
        assert!(fit.n_accepted <= NEWTON_MAX_UPDATES);
        // Cross-check against long gradient descent on the same
        // objective: probabilities must agree closely.
        let d = 2;
        let mut w = vec![0.0f64; d];
        let mut b = 0.0f64;
        let n = xs.len() as f64;
        for _ in 0..200_000 {
            let mut grad = vec![0.0f64; d + 1];
            for (row, &yi) in xs.iter().zip(y.iter()) {
                let mut z = b;
                for j in 0..d {
                    z += w[j] * row[j];
                }
                let err = sigmoid(z) - yi as f64;
                for j in 0..d {
                    grad[j] += err * row[j];
                }
                grad[d] += err;
            }
            for j in 0..d {
                w[j] -= 0.05 * (grad[j] / n + RIDGE_LAMBDA * w[j]);
            }
            b -= 0.05 * grad[d] / n;
        }
        for (row, &yi) in xs.iter().zip(y.iter()).take(50) {
            let _ = yi;
            let p_newton = predict_proba(&fit, row);
            let mut z = b;
            for j in 0..d {
                z += w[j] * row[j];
            }
            let p_gd = sigmoid(z);
            assert!((p_newton - p_gd).abs() < 1e-3, "{p_newton} vs {p_gd}");
        }
    }

    #[test]
    fn ridge_logistic_single_class_is_a_failure_not_a_result() {
        let xs = vec![vec![0.1, -0.2], vec![0.3, 0.4]];
        let y = vec![1, 1];
        assert_eq!(
            fit_ridge_logistic(&xs, &y, 1.0).unwrap_err(),
            DirectionError::SingleClassTraining
        );
    }

    #[test]
    fn buckets_sorted_matches_the_reference_routine() {
        let mut rng = Lcg(7);
        let mut trades: Vec<Trade> = Vec::new();
        let mut t = 1_700_000_000_000i64;
        for _ in 0..5000 {
            t += (rng.next() % 500) as i64;
            trades.push(Trade {
                time_ms: t,
                price: 100_000.0,
                qty: 0.01 + rng.uniform() * 0.5,
                is_buyer_maker: rng.next().is_multiple_of(2),
            });
        }
        for (lo, hi) in [(0, 1000), (1000, 3000), (2500, 5000), (0, 5000)] {
            let a = volume_buckets_sized(&trades[lo..hi], 1.0);
            let b = buckets_sorted(&trades[lo..hi], 1.0);
            assert_eq!(a.len(), b.len(), "bucket count for {lo}..{hi}");
            for (x, y) in a.iter().zip(b.iter()) {
                assert!((x.imbalance - y.imbalance).abs() < 1e-12);
                assert_eq!(x.time_ms, y.time_ms);
            }
        }
    }

    /// A synthetic frame with `n` bars, all observed, mid drifting
    /// upward slowly, OFI alternating, no trades.
    fn synthetic_frame(n: usize) -> SessionFrame {
        let mut observed = vec![true; n];
        let _ = &mut observed;
        SessionFrame {
            session_id: "synthetic".to_string(),
            t0_ms: 1_700_000_000_000,
            n_bars: n,
            observed: vec![true; n],
            mid: (0..n).map(|i| 100_000.0 + i as f64 * 0.01).collect(),
            depth: vec![2.0; n],
            imbalance: vec![0.1; n],
            ofi: (0..n)
                .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect(),
            quote_count: vec![10; n],
            signed_qty: vec![0.5; n],
            total_qty: vec![1.0; n],
            trades: Vec::new(),
            seg_start: vec![0; n],
            first_observed: 0,
            last_observed: n - 1,
        }
    }

    #[test]
    fn mask_excludes_gap_neighbourhoods_and_session_edges() {
        let mut frame = synthetic_frame(300);
        // A three-bar outage in the middle.
        frame.observed[140..143].fill(false);
        // Recompute segment starts the way build_frame does.
        let mut run_start = 0usize;
        for i in 0..frame.n_bars {
            if i == 0 || !frame.observed[i - 1] {
                run_start = i;
            }
            if frame.observed[i] {
                frame.seg_start[i] = run_start;
            }
        }
        frame.first_observed = 0;
        frame.last_observed = 299;
        let origins = eligible_origins(&frame);
        // No origin may sit within 60 bars of the outage.
        for &i in &origins {
            assert!(i + 60 < 140 || i > 142 + 60, "origin {i} touches the gap");
        }
        // Edges: first eligible is 60, last is n-1-60.
        assert_eq!(*origins.first().unwrap(), 60);
        assert_eq!(*origins.last().unwrap(), 239);
        // Fully observed frame: every interior bar is eligible.
        let full = synthetic_frame(300);
        assert_eq!(eligible_origins(&full).len(), 300 - 2 * 60);
    }

    #[test]
    fn standardize_zeroes_constant_columns_and_clips() {
        let raw = vec![5.0, 100.0, -100.0];
        let standard = vec![(5.0, 0.0), (0.0, 1.0), (0.0, 1.0)];
        let z = standardize(&raw, &standard);
        assert_eq!(z[0], 0.0);
        assert_eq!(z[1], 8.0);
        assert_eq!(z[2], -8.0);
    }

    #[test]
    fn burst_threshold_is_the_nearest_rank_95th_percentile() {
        let mut vals: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        // rank ceil(0.95 * 100) = 95 -> value 95.
        assert_eq!(burst_threshold(&mut vals), 95.0);
        let mut small = vec![3.0, 1.0, 2.0];
        // rank ceil(0.95 * 3) = 3 -> max.
        assert_eq!(burst_threshold(&mut small), 3.0);
    }

    #[test]
    fn ar10_fit_recovers_a_synthetic_ar1() {
        // F[t] = 0.5 * F[t-1] + noise, 5,000 points.
        let mut rng = Lcg(123);
        let n = 5000;
        let mut f = vec![0.0f64; n];
        for i in 1..n {
            f[i] = 0.5 * f[i - 1] + (rng.uniform() - 0.5);
        }
        let frame = SessionFrame {
            session_id: "ar".to_string(),
            t0_ms: 1_700_000_000_000,
            n_bars: n,
            observed: vec![true; n],
            mid: vec![100_000.0; n],
            depth: vec![2.0; n],
            imbalance: vec![0.0; n],
            ofi: f,
            quote_count: vec![1; n],
            signed_qty: vec![0.0; n],
            total_qty: vec![0.0; n],
            trades: Vec::new(),
            seg_start: vec![0; n],
            first_observed: 0,
            last_observed: n - 1,
        };
        let origins: Vec<OriginData> = (60..n - 60)
            .map(|i| OriginData {
                bar: i,
                decision_ms: frame.decision_time_ms(i),
                target: 0,
                vpin: 0.0,
                vpin_avail: 0.0,
            })
            .collect();
        let (intercept, coefs) = fit_ar10(&[(&frame, origins.as_slice())]).unwrap();
        assert!(intercept.abs() < 0.05, "intercept {intercept}");
        assert!((coefs[0] - 0.5).abs() < 0.05, "phi1 {}", coefs[0]);
        for (k, &c) in coefs.iter().enumerate().skip(1) {
            assert!(c.abs() < 0.05, "phi{} {}", k + 1, c);
        }
    }

    fn test_ctx() -> FitContext {
        let mut standard = HashMap::new();
        for family in Family::all() {
            standard.insert(family, vec![(0.0, 1.0); family.n_columns()]);
        }
        FitContext {
            ar_intercept: 0.0,
            ar_coefs: [0.0; AR_ORDER],
            burst_threshold: 1e9,
            prevalence: 0.5,
            majority_label: 1,
            standard,
        }
    }

    #[test]
    fn raw_columns_have_the_registered_widths() {
        let frame = synthetic_frame(300);
        let calc = FrameCalc::new(&frame);
        let ctx = test_ctx();
        let origin = OriginData {
            bar: 100,
            decision_ms: frame.decision_time_ms(100),
            target: 1,
            vpin: 0.3,
            vpin_avail: 1.0,
        };
        for family in Family::all() {
            let cols = raw_columns(&calc, &origin, &ctx, family);
            assert_eq!(cols.len(), family.n_columns(), "{:?}", family);
            assert!(cols.iter().all(|c| c.is_finite()), "{:?}", family);
        }
        // V carries the VPIN pair in its last two columns.
        let v = raw_columns(&calc, &origin, &ctx, Family::V);
        assert_eq!(v[17], 0.3);
        assert_eq!(v[18], 1.0);
        // The burst-signed column is 0 when nothing is burst.
        let r = raw_columns(&calc, &origin, &ctx, Family::R);
        assert_eq!(r[5], 0.0);
    }

    #[test]
    fn pool_metrics_weights_sessions_equally_not_rows() {
        // Session A: 100 rows (50 ones, 50 zeros), model 90% right
        // in each class. Session B: 10 rows, model 50% right.
        // Equal-session pooled accuracy is 0.70, not the
        // row-weighted 0.864.
        let mut a = Vec::new();
        for i in 0..100 {
            let target = u8::from(i % 2 == 0);
            // Wrong on 5 ones and 5 zeros.
            let wrong = (target == 1 && i < 10) || (target == 0 && i < 11 && i % 2 == 1);
            a.push(ScoredRow {
                time_ms: i,
                target,
                model_pred: if wrong { 1 - target } else { target },
                baseline_pred: 1,
            });
        }
        let mut b = Vec::new();
        for i in 0..10 {
            let target = u8::from(i % 2 == 0);
            // Right on 2 ones and 3 zeros.
            let right = (target == 1 && i < 4) || (target == 0 && i >= 5);
            b.push(ScoredRow {
                time_ms: i,
                target,
                model_pred: if right { target } else { 1 - target },
                baseline_pred: 1,
            });
        }
        let pooled = pool_metrics(&[a, b]).unwrap();
        assert!(
            (pooled.accuracy - 0.70).abs() < 1e-12,
            "{}",
            pooled.accuracy
        );
        // Baseline predicts 1 everywhere; half the targets are 1.
        assert!((pooled.baseline_accuracy - 0.50).abs() < 1e-12);
        assert!((pooled.delta - 0.20).abs() < 1e-12);
        // Balanced accuracy: A is 0.90, B is 0.50, pooled 0.70.
        assert!((pooled.balanced_accuracy - 0.70).abs() < 1e-12);
    }

    fn dev_row(
        family: Family,
        rule: Rule,
        s: f64,
        delta: f64,
        gamma: f64,
        per_s: [f64; 4],
    ) -> DevRow {
        DevRow {
            family,
            rule,
            pooled: PooledMetrics {
                n_scored: 100,
                accuracy: 0.5 + delta,
                baseline_accuracy: 0.5,
                balanced_accuracy: 0.5 + gamma,
                delta,
                gamma,
                s,
            },
            per_session: per_s
                .iter()
                .enumerate()
                .map(|(i, &v)| SessionMetrics {
                    session: format!("s{i}"),
                    n_scored: 25,
                    accuracy: 0.5,
                    baseline_accuracy: 0.5,
                    balanced_accuracy: Some(0.5),
                    delta: 0.0,
                    gamma: Some(0.0),
                    s: Some(v),
                })
                .collect(),
            boot_300s: None,
            boot_600s: None,
        }
    }

    #[test]
    fn selection_prefers_the_simpler_family_on_ties_and_applies_the_ladder() {
        // R and F tie on S and Delta: R wins by simplicity.
        let rows = vec![
            dev_row(Family::R, Rule::P50, 1.0, 0.010, 0.020, [1.0; 4]),
            dev_row(Family::R, Rule::Prevalence, 0.5, 0.005, 0.010, [0.5; 4]),
            dev_row(Family::F, Rule::P50, 1.0, 0.010, 0.020, [1.1; 4]),
            dev_row(Family::F, Rule::Prevalence, 0.5, 0.005, 0.010, [0.5; 4]),
            dev_row(Family::V, Rule::P50, 0.0, 0.0, 0.0, [0.0; 4]),
            dev_row(Family::V, Rule::Prevalence, 0.0, 0.0, 0.0, [0.0; 4]),
            dev_row(Family::I, Rule::P50, 0.0, 0.0, 0.0, [0.0; 4]),
            dev_row(Family::I, Rule::Prevalence, 0.0, 0.0, 0.0, [0.0; 4]),
        ];
        let sel = select_family(&rows).unwrap();
        assert_eq!(sel.selected, Family::R);
        // F does NOT clear R here: S improves by 0.0, below 0.10.
        assert!(!sel.families[1].qualifies);

        // F clears R: S +0.5, Delta/Gamma unchanged, 4/4 votes.
        let rows = vec![
            dev_row(Family::R, Rule::P50, 1.0, 0.010, 0.020, [1.0; 4]),
            dev_row(Family::R, Rule::Prevalence, 0.5, 0.005, 0.010, [0.5; 4]),
            dev_row(Family::F, Rule::P50, 1.5, 0.010, 0.020, [1.5; 4]),
            dev_row(Family::F, Rule::Prevalence, 0.5, 0.005, 0.010, [0.5; 4]),
            dev_row(Family::V, Rule::P50, 0.0, 0.0, 0.0, [0.0; 4]),
            dev_row(Family::V, Rule::Prevalence, 0.0, 0.0, 0.0, [0.0; 4]),
            dev_row(Family::I, Rule::P50, 0.0, 0.0, 0.0, [0.0; 4]),
            dev_row(Family::I, Rule::Prevalence, 0.0, 0.0, 0.0, [0.0; 4]),
        ];
        let sel = select_family(&rows).unwrap();
        assert_eq!(sel.selected, Family::F);
        assert!(sel.families[1].qualifies);

        // F clears on S but fails the 3-of-4 vote: only 2 sessions improve.
        let rows = vec![
            dev_row(Family::R, Rule::P50, 1.0, 0.010, 0.020, [1.0; 4]),
            dev_row(Family::R, Rule::Prevalence, 0.5, 0.005, 0.010, [0.5; 4]),
            dev_row(
                Family::F,
                Rule::P50,
                1.5,
                0.010,
                0.020,
                [1.5, 1.5, 0.5, 0.5],
            ),
            dev_row(Family::F, Rule::Prevalence, 0.5, 0.005, 0.010, [0.5; 4]),
            dev_row(Family::V, Rule::P50, 0.0, 0.0, 0.0, [0.0; 4]),
            dev_row(Family::V, Rule::Prevalence, 0.0, 0.0, 0.0, [0.0; 4]),
            dev_row(Family::I, Rule::P50, 0.0, 0.0, 0.0, [0.0; 4]),
            dev_row(Family::I, Rule::Prevalence, 0.0, 0.0, 0.0, [0.0; 4]),
        ];
        let sel = select_family(&rows).unwrap();
        assert_eq!(sel.selected, Family::R);
        assert!(!sel.families[1].qualifies);
    }

    #[test]
    fn bootstrap_is_a_point_when_every_row_agrees() {
        // Every row: model right, baseline wrong, half the targets
        // 1 and half 0. Delta is exactly 1.0 and Gamma exactly 0.5
        // in every draw.
        let mut sessions = Vec::new();
        for s in 0..4 {
            let rows: Vec<ScoredRow> = (0..1000)
                .map(|i| {
                    let target = u8::from(i % 2 == 0);
                    ScoredRow {
                        time_ms: (s * 20_000 + i) as i64 * 1000,
                        target,
                        model_pred: target,
                        baseline_pred: 1 - target,
                    }
                })
                .collect();
            sessions.push(rows);
        }
        let ci = paired_grouped_bootstrap(&sessions, 300_000).unwrap();
        assert_eq!(ci.0, (1.0, 1.0));
        assert_eq!(ci.1, (0.5, 0.5));
    }

    /// A synthetic frame whose mid oscillates in 100-bar phases, so
    /// both target classes occur, and whose order-flow series is
    /// pseudo-random, so the AR(10) design is full rank.
    fn oscillating_frame(id: &str, n: usize, seed: u64) -> SessionFrame {
        let mut lcg = Lcg(seed);
        let mut mid = Vec::with_capacity(n);
        let mut v = 100_000.0;
        for i in 0..n {
            let phase = (i / 100) % 2;
            v += if phase == 0 { 0.05 } else { -0.05 };
            mid.push(v);
        }
        let ofi: Vec<f64> = (0..n).map(|_| lcg.uniform() * 2.0 - 1.0).collect();
        SessionFrame {
            session_id: id.to_string(),
            t0_ms: 1_700_000_000_000,
            n_bars: n,
            observed: vec![true; n],
            mid,
            depth: vec![2.0; n],
            imbalance: vec![0.1; n],
            ofi,
            quote_count: vec![10; n],
            signed_qty: vec![0.5; n],
            total_qty: vec![1.0; n],
            trades: Vec::new(),
            seg_start: vec![0; n],
            first_observed: 0,
            last_observed: n - 1,
        }
    }

    fn pooled(delta: f64, ba: f64) -> PooledMetrics {
        PooledMetrics {
            n_scored: 1000,
            accuracy: 0.5 + delta,
            baseline_accuracy: 0.5,
            balanced_accuracy: ba,
            delta,
            gamma: ba - 0.5,
            s: 0.0,
        }
    }

    #[test]
    fn confirmation_gates_apply_exactly_as_written() {
        let both = Some(((0.005, 0.030), (0.010, 0.050)));
        // Both endpoints clear at both block lengths: win.
        let (kind, cleared) = classify_confirmation(&pooled(0.020, 0.550), both, both, 10).unwrap();
        assert_eq!(kind, ConfirmationOutcomeKind::Win);
        assert_eq!(cleared, None);
        // Delta clears, Gamma does not (BA below 0.520): partial.
        let (kind, cleared) = classify_confirmation(&pooled(0.020, 0.510), both, both, 10).unwrap();
        assert_eq!(kind, ConfirmationOutcomeKind::Partial);
        assert_eq!(cleared, Some("Delta"));
        // Gamma clears, Delta does not: partial.
        let (kind, cleared) = classify_confirmation(&pooled(0.005, 0.550), both, both, 10).unwrap();
        assert_eq!(kind, ConfirmationOutcomeKind::Partial);
        assert_eq!(cleared, Some("Balanced accuracy"));
        // Neither clears: null.
        let (kind, cleared) = classify_confirmation(&pooled(0.005, 0.510), both, both, 10).unwrap();
        assert_eq!(kind, ConfirmationOutcomeKind::Null);
        assert_eq!(cleared, None);
        // A zero lower bound is not strictly positive: no win.
        let edge = Some(((0.0, 0.030), (0.010, 0.050)));
        let (kind, _) = classify_confirmation(&pooled(0.020, 0.550), edge, both, 10).unwrap();
        assert_eq!(kind, ConfirmationOutcomeKind::Partial);
        // Suppressed intervals and a missing session are errors,
        // not results: the study is incomplete.
        assert!(classify_confirmation(&pooled(0.020, 0.550), None, both, 10).is_err());
        assert!(classify_confirmation(&pooled(0.020, 0.550), both, both, 9).is_err());
    }

    fn frozen_test_artifact() -> FrozenArtifact {
        let fa = oscillating_frame("fit-a", 600, 7);
        let fb = oscillating_frame("fit-b", 600, 8);
        let ca = FrameCalc::new(&fa);
        let cb = FrameCalc::new(&fb);
        let oa = ca.origins();
        let ob = cb.origins();
        assert!(!oa.is_empty() && !ob.is_empty());
        let train = vec![(ca, oa), (cb, ob)];
        let fit_ids = vec!["fit-a".to_string(), "fit-b".to_string()];
        let fit_hashes = vec![
            ("fit-a".to_string(), "hash-a".to_string()),
            ("fit-b".to_string(), "hash-b".to_string()),
        ];
        freeze_artifact(
            &train,
            &fit_ids,
            &fit_hashes,
            i64::MAX,
            Family::F,
            Rule::P50,
            PROTOCOL_DIGEST,
            "test-code-hash",
            1_700_000_000,
        )
        .unwrap()
    }

    #[test]
    fn freeze_round_trip_replays_decisions_and_probabilities() {
        let artifact = frozen_test_artifact();
        assert_eq!(artifact.family, Family::F);
        assert_eq!(artifact.rule, Rule::P50);
        assert_eq!(artifact.protocol_digest, PROTOCOL_DIGEST);
        assert_eq!(artifact.code_hash, "test-code-hash");
        assert_eq!(artifact.fit_sessions, vec!["fit-a", "fit-b"]);
        let json = artifact.to_json();
        let replay = FrozenArtifact::from_json(&json).unwrap();
        assert_eq!(replay.slopes, artifact.slopes);
        assert_eq!(replay.intercept, artifact.intercept);
        // Score one fresh frame with both artifacts: identical
        // decisions and probabilities to within 1e-12.
        let frame = oscillating_frame("score-c", 600, 9);
        let calc = FrameCalc::new(&frame);
        let origins = calc.origins();
        let (rows_a, probs_a) = score_with_artifact(&artifact, &calc, &origins);
        let (rows_b, probs_b) = score_with_artifact(&replay, &calc, &origins);
        assert_eq!(rows_a.len(), rows_b.len());
        assert!(!rows_a.is_empty());
        for (ra, rb) in rows_a.iter().zip(rows_b.iter()) {
            assert_eq!(ra.time_ms, rb.time_ms);
            assert_eq!(ra.target, rb.target);
            assert_eq!(ra.model_pred, rb.model_pred);
            assert_eq!(ra.baseline_pred, rb.baseline_pred);
        }
        for (pa, pb) in probs_a.iter().zip(probs_b.iter()) {
            assert!((pa - pb).abs() <= 1e-12, "{pa} vs {pb}");
        }
        // A tampered digest does not deserialize.
        let mut bad = json.clone();
        bad["protocol_digest"] = serde_json::json!("0".repeat(64));
        assert!(FrozenArtifact::from_json(&bad).is_err());
    }

    #[test]
    fn run_confirmation_scores_ten_synthetic_holdout_sessions() {
        let artifact = frozen_test_artifact();
        let mut frames: Vec<SessionFrame> = Vec::new();
        for (k, id) in CAMPAIGN_SLOT_IDS[10..].iter().enumerate() {
            frames.push(oscillating_frame(id, 600, 100 + k as u64));
        }
        let inputs: Vec<HoldoutInput> = frames
            .iter()
            .enumerate()
            .map(|(k, f)| HoldoutInput {
                slot: CONFIRMATION_FIRST_SLOT + k,
                session_id: CAMPAIGN_SLOT_IDS[10 + k].to_string(),
                frame: f,
            })
            .collect();
        let outcome = run_confirmation(&artifact, &inputs).unwrap();
        assert_eq!(outcome.per_session.len(), 10);
        assert_eq!(outcome.probabilities.len(), 10);
        assert!(outcome.boot_300s.is_some());
        assert!(outcome.boot_600s.is_some());
        assert!(outcome.fixed_cohort_300s.is_some());
        assert_eq!(outcome.leave_one_pair_out.len(), 5);
        assert!(!outcome.audit_300s.is_empty());
        assert!(!outcome.audit_600s.is_empty());
        assert!(matches!(
            outcome.outcome,
            ConfirmationOutcomeKind::Win
                | ConfirmationOutcomeKind::Partial
                | ConfirmationOutcomeKind::Null
        ));
        // Nine sessions is an error, not a result.
        assert!(run_confirmation(&artifact, &inputs[..9]).is_err());
    }

    #[test]
    fn run_development_with_no_archives_selects_a_family() {
        // The ETH B3 shape: no archive sessions, ten synthetic
        // slots, fit pool growing from 01-06 through 09.
        let ids: [&str; 10] = [
            "session-20261109-084401-utc",
            "session-20261109-124411-utc",
            "session-20261109-164421-utc",
            "session-20261109-204431-utc",
            "session-20261110-004441-utc",
            "session-20261110-044452-utc",
            "session-20261110-084502-utc",
            "session-20261110-124512-utc",
            "session-20261110-164522-utc",
            "session-20261110-204532-utc",
        ];
        let mut frames: Vec<SessionFrame> = Vec::new();
        for (k, id) in ids.iter().enumerate() {
            frames.push(oscillating_frame(id, 600, 200 + k as u64));
        }
        let inputs: Vec<SlotInput> = frames
            .iter()
            .enumerate()
            .map(|(k, f)| SlotInput {
                slot: k + 1,
                session_id: ids[k].to_string(),
                frame: f,
            })
            .collect();
        let outcome = run_development(&[], &inputs).unwrap();
        assert_eq!(outcome.rows.len(), 8);
        assert_eq!(outcome.boundaries.len(), 4);
        assert!(outcome.rows.iter().any(|r| r.family == Family::F));
        // Nine slots is an error, not a result.
        assert!(run_development(&[], &inputs[..9]).is_err());
    }

    #[test]
    fn transfer_validation_matches_full_development_row() {
        // The frozen-configuration path must reproduce the
        // corresponding full-development row exactly: same
        // boundaries, embargo, pooling and bootstrap, because it
        // runs through the same function with a single pair.
        let ids: [&str; 10] = [
            "session-20261109-084401-utc",
            "session-20261109-124411-utc",
            "session-20261109-164421-utc",
            "session-20261109-204431-utc",
            "session-20261110-004441-utc",
            "session-20261110-044452-utc",
            "session-20261110-084502-utc",
            "session-20261110-124512-utc",
            "session-20261110-164522-utc",
            "session-20261110-204532-utc",
        ];
        let mut frames: Vec<SessionFrame> = Vec::new();
        for (k, id) in ids.iter().enumerate() {
            frames.push(oscillating_frame(id, 600, 200 + k as u64));
        }
        let inputs: Vec<SlotInput> = frames
            .iter()
            .enumerate()
            .map(|(k, f)| SlotInput {
                slot: k + 1,
                session_id: ids[k].to_string(),
                frame: f,
            })
            .collect();
        let (boundaries, row) =
            run_transfer_validation(&[], &inputs, Family::F, Rule::P50).unwrap();
        assert_eq!(boundaries.len(), 4);
        assert_eq!(row.family, Family::F);
        assert_eq!(row.rule, Rule::P50);
        assert_eq!(row.per_session.len(), 4);
        let full = run_development(&[], &inputs).unwrap();
        let full_row = full
            .rows
            .iter()
            .find(|r| r.family == Family::F && r.rule == Rule::P50)
            .unwrap();
        assert_eq!(row.pooled.n_scored, full_row.pooled.n_scored);
        assert_eq!(row.pooled.accuracy, full_row.pooled.accuracy);
        assert_eq!(
            row.pooled.baseline_accuracy,
            full_row.pooled.baseline_accuracy
        );
        assert_eq!(
            row.pooled.balanced_accuracy,
            full_row.pooled.balanced_accuracy
        );
        assert_eq!(row.pooled.delta, full_row.pooled.delta);
        assert_eq!(row.pooled.gamma, full_row.pooled.gamma);
        assert_eq!(row.pooled.s, full_row.pooled.s);
        assert_eq!(row.boot_300s, full_row.boot_300s);
        assert_eq!(row.boot_600s, full_row.boot_600s);
        assert_eq!(
            boundaries.iter().map(|b| b.eval_slot).collect::<Vec<_>>(),
            full.boundaries
                .iter()
                .map(|b| b.eval_slot)
                .collect::<Vec<_>>()
        );
        // Nine slots is an error, not a result.
        assert!(run_transfer_validation(&[], &inputs[..9], Family::F, Rule::P50).is_err());
    }

    #[test]
    fn load_session_unsealed_rejects_missing_directory() {
        assert!(load_session_unsealed("session-20261111-004543-utc", "/nonexistent-dir").is_err());
    }
}
