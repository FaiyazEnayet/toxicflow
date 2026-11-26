//! Local limit order book with Binance diff-depth sync.
//!
//! Follows the exchange's documented procedure: buffer diff events, take a
//! REST snapshot, discard stale diffs, verify the [U, u] continuity, then
//! apply. A gap (U > last_id + 1) forces a full resync.
//!
//! Prices are stored as integer ticks (price * 1e8) so levels can live in a
//! `BTreeMap` and best bid/ask are O(log n).
//!
//! # The 1e-8 scale is not the tick size
//!
//! The internal 1e-8 scale is representation precision only: it is the
//! finest price the book can store, chosen so that every price with up to
//! eight decimal places round-trips exactly. It says nothing about which
//! prices the exchange will actually send for a symbol.
//!
//! The exchange tick-size filter is a separate, per-symbol rule: the
//! minimum price increment the venue accepts (for example 0.01 on
//! BTCUSDT). A price can be representable at 1e-8 yet invalid for the
//! symbol: 100.005 stores exactly at the 1e-8 scale but is off-tick for
//! a 0.01 filter, and the exchange would never send it. `TickFilter`
//! carries that rule, and `OrderBook::with_tick_filter` applies it.
//!
//! Without a filter the book still validates what needs no symbol:
//! prices must be finite, positive and representable at 1e-8, and
//! quantities must be finite and nonnegative, before any conversion.
//! A level that fails validation is never inserted; rejections are
//! counted and the last one is visible through `OrderBook::validity()`.

use std::collections::BTreeMap;
use std::fmt;

const TICK_SCALE: f64 = 1e8;

/// Largest finite `price * 1e8` that still converts to `i64` exactly.
/// Beyond 2^53 an `f64` can no longer hold every integer, so a price
/// whose scaled value reaches it is rejected as overflow rather than
/// rounded into a neighbouring tick.
const MAX_SCALED: f64 = 9_007_199_254_740_992.0; // 2^53

fn to_ticks(price: f64) -> i64 {
    (price * TICK_SCALE).round() as i64
}

fn from_ticks(ticks: i64) -> f64 {
    ticks as f64 / TICK_SCALE
}

/// True when `scaled` (a price or quantity times 1e8) is within float
/// noise of an integer tick: the value is representable at the 1e-8
/// scale. The tolerance is a few ulps of the scaled value (with a
/// floor), so a price built by float arithmetic a hair off a decimal,
/// for example best ask + 0.05, still converts, while a value with a
/// genuine ninth decimal place stays a clear miss: it sits 0.1 tick
/// units from the nearest integer, far above the tolerance at any
/// realistic price.
fn near_integer_tick(scaled: f64) -> bool {
    let tol = (scaled.abs() * 1.8e-15).max(1e-6);
    (scaled - scaled.round()).abs() <= tol
}

/// Convert a filter field (tick size, lot step, minimum quantity) to
/// internal 1e-8 units. Returns `None` when the value is nonfinite,
/// nonpositive, too large to convert exactly, or has more than eight
/// decimal places.
fn scale_to_ticks(value: f64) -> Option<i64> {
    if !value.is_finite() || value <= 0.0 {
        return None;
    }
    let scaled = value * TICK_SCALE;
    if !scaled.is_finite() || scaled.abs() >= MAX_SCALED {
        return None;
    }
    if !near_integer_tick(scaled) {
        return None;
    }
    Some(to_ticks(value))
}

/// Why a price or quantity was rejected before conversion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TickError {
    /// Price is NaN or infinite.
    NonFinitePrice,
    /// Price is zero or negative.
    NonPositivePrice,
    /// Price is so large its 1e-8 scaled value no longer fits in an
    /// exact integer (at or beyond 2^53).
    PriceOverflow,
    /// Price has more than eight decimal places and cannot be stored
    /// exactly at the internal 1e-8 scale.
    UnrepresentablePrice,
    /// Price is representable at 1e-8 but is not an exact multiple of
    /// the symbol tick size.
    OffTick { price: f64, tick_size: f64 },
    /// Quantity is NaN or infinite.
    NonFiniteQty,
    /// Quantity is negative. Zero is allowed: in a diff it removes
    /// the level.
    NegativeQty,
    /// Quantity has more than eight decimal places and cannot be
    /// stored exactly at the internal 1e-8 scale.
    UnrepresentableQty,
    /// Quantity is not an exact multiple of the lot step.
    OffLotStep { qty: f64, step_size: f64 },
    /// Quantity is positive but below the minimum quantity.
    BelowMinQty { qty: f64, min_qty: f64 },
    /// The filter itself is invalid: a non-positive, nonfinite, or
    /// unrepresentable tick size, lot step, or minimum quantity.
    InvalidFilter,
}

impl fmt::Display for TickError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TickError::NonFinitePrice => write!(f, "price is not finite"),
            TickError::NonPositivePrice => write!(f, "price is not positive"),
            TickError::PriceOverflow => {
                write!(f, "price is too large to convert to ticks exactly")
            }
            TickError::UnrepresentablePrice => {
                write!(f, "price has more than eight decimal places")
            }
            TickError::OffTick { price, tick_size } => write!(
                f,
                "price {price} is not a multiple of tick size {tick_size}"
            ),
            TickError::NonFiniteQty => write!(f, "quantity is not finite"),
            TickError::NegativeQty => write!(f, "quantity is negative"),
            TickError::UnrepresentableQty => {
                write!(f, "quantity has more than eight decimal places")
            }
            TickError::OffLotStep { qty, step_size } => write!(
                f,
                "quantity {qty} is not a multiple of lot step {step_size}"
            ),
            TickError::BelowMinQty { qty, min_qty } => {
                write!(f, "quantity {qty} is below minimum quantity {min_qty}")
            }
            TickError::InvalidFilter => write!(f, "tick filter is invalid"),
        }
    }
}

impl std::error::Error for TickError {}

/// Validate a price with no symbol filter: finite, positive,
/// representable at 1e-8, and small enough to convert exactly.
/// Representability means the scaled value sits within float noise of
/// an integer tick (see `near_integer_tick`); an exact round-trip is
/// too strict, because a price built by float arithmetic can sit one
/// ulp off its decimal and still name the same tick.
fn checked_ticks(price: f64) -> Result<i64, TickError> {
    if !price.is_finite() {
        return Err(TickError::NonFinitePrice);
    }
    if price <= 0.0 {
        return Err(TickError::NonPositivePrice);
    }
    let scaled = price * TICK_SCALE;
    if !scaled.is_finite() || scaled.abs() >= MAX_SCALED {
        return Err(TickError::PriceOverflow);
    }
    if !near_integer_tick(scaled) {
        return Err(TickError::UnrepresentablePrice);
    }
    Ok(to_ticks(price))
}

/// Validate a quantity with no symbol filter: finite and nonnegative.
fn check_qty_unfiltered(qty: f64) -> Result<(), TickError> {
    if !qty.is_finite() {
        return Err(TickError::NonFiniteQty);
    }
    if qty < 0.0 {
        return Err(TickError::NegativeQty);
    }
    Ok(())
}

/// Validate one (price, qty) level and convert its price to ticks,
/// applying the tick filter when one is attached. A free function so
/// `apply_diff` can validate while holding a mutable borrow of a side.
fn convert_with(filter: Option<TickFilter>, price: f64, qty: f64) -> Result<i64, TickError> {
    match filter {
        Some(f) => {
            let ticks = f.to_ticks_checked(price)?;
            f.check_qty(qty)?;
            Ok(ticks)
        }
        None => {
            let ticks = checked_ticks(price)?;
            check_qty_unfiltered(qty)?;
            Ok(ticks)
        }
    }
}

/// Validate a price only and convert it to ticks, applying the tick
/// filter when one is attached. Removals carry no quantity, so no lot
/// rule applies to them.
fn convert_price_with(filter: Option<TickFilter>, price: f64) -> Result<i64, TickError> {
    match filter {
        Some(f) => f.to_ticks_checked(price),
        None => checked_ticks(price),
    }
}

/// Exchange tick-size / lot-size filter for one symbol.
///
/// The filter is the venue rule, not the internal representation: the
/// book stores prices at a 1e-8 scale (see the module docs) while the
/// filter says which of those prices the symbol may actually trade.
/// `tick_size` is the minimum price increment, `step_size` the minimum
/// quantity increment, `min_qty` the smallest positive quantity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TickFilter {
    pub tick_size: f64,
    pub step_size: f64,
    pub min_qty: f64,
}

impl TickFilter {
    /// Build a filter. Every field must be finite, positive, and
    /// representable at the internal 1e-8 scale.
    pub fn new(tick_size: f64, step_size: f64, min_qty: f64) -> Result<Self, TickError> {
        for v in [tick_size, step_size, min_qty] {
            if scale_to_ticks(v).is_none() {
                return Err(TickError::InvalidFilter);
            }
        }
        Ok(Self {
            tick_size,
            step_size,
            min_qty,
        })
    }

    /// Representative Binance-style spot filter for BTCUSDT: 0.01 tick,
    /// 0.001 lot step, 0.001 minimum quantity. A fixture for tests and
    /// examples, not live exchange data.
    pub fn btc_usdt() -> Self {
        Self {
            tick_size: 0.01,
            step_size: 0.001,
            min_qty: 0.001,
        }
    }

    /// Representative Binance-style spot filter for ETHUSDT: 0.01 tick,
    /// 0.001 lot step, 0.001 minimum quantity. A fixture for tests and
    /// examples, not live exchange data.
    pub fn eth_usdt() -> Self {
        Self {
            tick_size: 0.01,
            step_size: 0.001,
            min_qty: 0.001,
        }
    }

    /// Finest filter the 1e-8 scale can express: every representable
    /// price is on-tick. Pins the boundary between the internal scale
    /// and the exchange rule.
    pub fn finest() -> Self {
        Self {
            tick_size: 0.00000001,
            step_size: 0.00000001,
            min_qty: 0.00000001,
        }
    }

    /// Convert a price to internal ticks, enforcing the filter:
    /// finite, positive, representable at 1e-8, and an exact multiple
    /// of the tick size, in that order.
    pub fn to_ticks_checked(&self, price: f64) -> Result<i64, TickError> {
        let ticks = checked_ticks(price)?;
        let tick_ticks = scale_to_ticks(self.tick_size).expect("filter validated at construction");
        if ticks % tick_ticks != 0 {
            return Err(TickError::OffTick {
                price,
                tick_size: self.tick_size,
            });
        }
        Ok(ticks)
    }

    /// Validate a quantity against the lot rule. Zero is allowed (in a
    /// diff it removes the level); a positive quantity must be
    /// representable, an exact multiple of the step, and at least the
    /// minimum quantity.
    pub fn check_qty(&self, qty: f64) -> Result<(), TickError> {
        check_qty_unfiltered(qty)?;
        if qty == 0.0 {
            return Ok(());
        }
        let step_ticks = scale_to_ticks(self.step_size).expect("filter validated at construction");
        let scaled = qty * TICK_SCALE;
        if !scaled.is_finite() || scaled.abs() >= MAX_SCALED || !near_integer_tick(scaled) {
            return Err(TickError::UnrepresentableQty);
        }
        let qty_ticks = scaled.round() as i64;
        if qty_ticks % step_ticks != 0 {
            return Err(TickError::OffLotStep {
                qty,
                step_size: self.step_size,
            });
        }
        if qty < self.min_qty {
            return Err(TickError::BelowMinQty {
                qty,
                min_qty: self.min_qty,
            });
        }
        Ok(())
    }

    /// Convert internal ticks back to a decimal price.
    pub fn from_ticks(ticks: i64) -> f64 {
        from_ticks(ticks)
    }
}

/// Book validity and gap coverage metrics.
///
/// `is_crossed()` answers one question; this answers the rest, so
/// callers do not have to infer book health from a single flag.
/// Counters are cumulative since the book was constructed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BookValidity {
    /// Best bid >= best ask with both sides present.
    pub crossed: bool,
    /// Best bid == best ask with both sides present.
    pub locked: bool,
    /// Exactly one side has levels.
    pub one_sided: bool,
    /// Neither side has levels.
    pub empty: bool,
    /// Number of bid levels.
    pub n_bid_levels: usize,
    /// Number of ask levels.
    pub n_ask_levels: usize,
    /// Best ask minus best bid in internal 1e-8 ticks, when both
    /// sides are present.
    pub spread_ticks: Option<i64>,
    /// Levels rejected by validation since construction.
    pub invalid_rejected: u64,
    /// Sequence gaps seen since construction.
    pub gaps_seen: u64,
    /// Stale events seen since construction.
    pub stale_seen: u64,
    /// Events applied since construction.
    pub applied_events: u64,
    /// Fraction of update ids actually applied since construction:
    /// applied / (applied + missed in gaps). 1.0 when nothing was
    /// missed. The snapshot baseline itself is not counted: coverage
    /// measures the diff stream after it.
    pub id_coverage: f64,
    /// The most recent rejection, if any.
    pub last_rejection: Option<TickError>,
}

/// Outcome of applying one depthUpdate event.
#[derive(Debug, PartialEq, Eq)]
pub enum SyncResult {
    /// Book updated normally.
    Applied,
    /// Event was older than the book; ignored.
    Stale,
    /// Events were missed; caller must resync from a fresh snapshot.
    Gap,
}

#[derive(Debug, Default, Clone)]
pub struct OrderBook {
    bids: BTreeMap<i64, f64>,
    asks: BTreeMap<i64, f64>,
    pub last_id: i64,
    filter: Option<TickFilter>,
    invalid_rejected: u64,
    gaps_seen: u64,
    stale_seen: u64,
    applied_events: u64,
    ids_applied: u64,
    ids_missed: u64,
    last_rejection: Option<TickError>,
}

impl OrderBook {
    pub fn new() -> Self {
        Self {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            last_id: -1,
            filter: None,
            invalid_rejected: 0,
            gaps_seen: 0,
            stale_seen: 0,
            applied_events: 0,
            ids_applied: 0,
            ids_missed: 0,
            last_rejection: None,
        }
    }

    /// Attach an exchange tick-size / lot-size filter. Levels that
    /// fail the filter are rejected (counted, never inserted). Without
    /// a filter the book still validates finiteness, positivity, and
    /// 1e-8 representability, which need no symbol.
    pub fn with_tick_filter(mut self, filter: TickFilter) -> Self {
        self.filter = Some(filter);
        self
    }

    /// The attached tick filter, if any.
    pub fn tick_filter(&self) -> Option<TickFilter> {
        self.filter
    }

    /// Levels rejected by validation since construction.
    pub fn invalid_rejected(&self) -> u64 {
        self.invalid_rejected
    }

    /// The most recent rejection, if any.
    pub fn last_rejection(&self) -> Option<TickError> {
        self.last_rejection
    }

    /// Book validity and gap coverage metrics. See `BookValidity`.
    pub fn validity(&self) -> BookValidity {
        let bid_t = self.bids.keys().next_back().copied();
        let ask_t = self.asks.keys().next().copied();
        let crossed = matches!((bid_t, ask_t), (Some(b), Some(a)) if b >= a);
        let locked = matches!((bid_t, ask_t), (Some(b), Some(a)) if b == a);
        let total = self.ids_applied + self.ids_missed;
        BookValidity {
            crossed,
            locked,
            one_sided: self.bids.is_empty() != self.asks.is_empty(),
            empty: self.is_empty(),
            n_bid_levels: self.bids.len(),
            n_ask_levels: self.asks.len(),
            spread_ticks: match (bid_t, ask_t) {
                (Some(b), Some(a)) => Some(a - b),
                _ => None,
            },
            invalid_rejected: self.invalid_rejected,
            gaps_seen: self.gaps_seen,
            stale_seen: self.stale_seen,
            applied_events: self.applied_events,
            id_coverage: if total == 0 {
                1.0
            } else {
                self.ids_applied as f64 / total as f64
            },
            last_rejection: self.last_rejection,
        }
    }

    fn reject(&mut self, e: TickError) {
        self.invalid_rejected += 1;
        self.last_rejection = Some(e);
    }

    /// Validate one (price, qty) level and convert its price to ticks,
    /// applying the tick filter when one is attached.
    fn convert(&self, price: f64, qty: f64) -> Result<i64, TickError> {
        convert_with(self.filter, price, qty)
    }

    /// Validate one snapshot level and convert its price. Zero
    /// quantity drops the level silently, as before; any other failure
    /// is counted and the level is skipped.
    fn checked_level(&mut self, price: f64, qty: f64) -> Option<(i64, f64)> {
        if qty == 0.0 {
            return None;
        }
        match self.convert(price, qty) {
            Ok(ticks) => Some((ticks, qty)),
            Err(e) => {
                self.reject(e);
                None
            }
        }
    }

    /// Load a REST snapshot. `bids`/`asks` are (price, qty) pairs.
    /// Levels that fail validation are never inserted and are counted;
    /// see `validity()`.
    pub fn from_snapshot(&mut self, bids: &[(f64, f64)], asks: &[(f64, f64)], last_id: i64) {
        self.bids = bids
            .iter()
            .filter_map(|(p, q)| self.checked_level(*p, *q))
            .collect();
        self.asks = asks
            .iter()
            .filter_map(|(p, q)| self.checked_level(*p, *q))
            .collect();
        self.last_id = last_id;
    }

    /// Apply one depthUpdate. `bids`/`asks` are (price, qty) level updates;
    /// a zero quantity removes the level. Levels that fail validation
    /// are skipped and counted; the valid remainder still applies.
    pub fn apply_diff(
        &mut self,
        u_first: i64,
        u_last: i64,
        bids: &[(f64, f64)],
        asks: &[(f64, f64)],
    ) -> SyncResult {
        if u_last < self.last_id {
            self.stale_seen += 1;
            return SyncResult::Stale;
        }
        if u_first > self.last_id + 1 {
            self.gaps_seen += 1;
            self.ids_missed += (u_first - self.last_id - 1) as u64;
            return SyncResult::Gap;
        }
        // Ids this event covers for the first time: its own span past
        // the previous baseline, so a redelivery counts nothing twice
        // and the first event after construction counts only its span.
        let new_ids = (u_last - self.last_id.max(u_first - 1)).max(0) as u64;
        // Validate through the free functions (the loop holds a
        // mutable borrow of each side map) and count rejections after.
        let filter = self.filter;
        let mut rejections: Vec<TickError> = Vec::new();
        for (book, levels) in [(&mut self.bids, bids), (&mut self.asks, asks)] {
            for (price, qty) in levels {
                if *qty == 0.0 {
                    // Removal: the price still has to name a level.
                    // An unconvertible price names none, so it is a
                    // counted rejection rather than a removal.
                    match convert_price_with(filter, *price) {
                        Ok(t) => {
                            book.remove(&t);
                        }
                        Err(e) => rejections.push(e),
                    }
                } else {
                    match convert_with(filter, *price, *qty) {
                        Ok(t) => {
                            book.insert(t, *qty);
                        }
                        Err(e) => rejections.push(e),
                    }
                }
            }
        }
        for e in rejections {
            self.reject(e);
        }
        self.ids_applied += new_ids;
        self.applied_events += 1;
        self.last_id = u_last;
        SyncResult::Applied
    }

    /// Apply a depth update that starts past the last applied id, taking
    /// it as the new baseline. A journal replay uses this once after a
    /// snapshot: the updates buffered across the snapshot merge are
    /// recorded only as the snapshot boundary, so the first update after
    /// it legitimately starts past the snapshot id.
    pub fn apply_diff_baseline(
        &mut self,
        u_first: i64,
        u_last: i64,
        bids: &[(f64, f64)],
        asks: &[(f64, f64)],
    ) -> SyncResult {
        self.last_id = u_first - 1;
        self.apply_diff(u_first, u_last, bids, asks)
    }

    pub fn best_bid(&self) -> Option<(f64, f64)> {
        self.bids
            .iter()
            .next_back()
            .map(|(t, q)| (from_ticks(*t), *q))
    }

    pub fn best_ask(&self) -> Option<(f64, f64)> {
        self.asks.iter().next().map(|(t, q)| (from_ticks(*t), *q))
    }

    pub fn midprice(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some((b, _)), Some((a, _))) => Some((b + a) / 2.0),
            _ => None,
        }
    }

    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some((b, _)), Some((a, _))) => Some(a - b),
            _ => None,
        }
    }

    /// (bid_px, bid_sz, ask_px, ask_sz) at the touch.
    pub fn touch(&self) -> Option<(f64, f64, f64, f64)> {
        match (self.best_bid(), self.best_ask()) {
            (Some((bp, bq)), Some((ap, aq))) => Some((bp, bq, ap, aq)),
            _ => None,
        }
    }

    /// (bid volume - ask volume) / (bid volume + ask volume), top `n` levels.
    pub fn imbalance(&self, n: usize) -> Option<f64> {
        let bv: f64 = self.bids.iter().rev().take(n).map(|(_, q)| q).sum();
        let av: f64 = self.asks.iter().take(n).map(|(_, q)| q).sum();
        if bv + av == 0.0 {
            None
        } else {
            Some((bv - av) / (bv + av))
        }
    }

    pub fn len(&self) -> usize {
        self.bids.len() + self.asks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bids.is_empty() && self.asks.is_empty()
    }

    /// True when the book holds both sides and best bid >= best ask.
    ///
    /// A crossed book cannot arise from a correctly synchronized exchange
    /// stream; it signals stale data, a missed update, or a bad snapshot.
    /// This is one flag only; `validity()` reports the full picture
    /// (locked, one-sided, empty, spread, rejections, gap coverage).
    pub fn is_crossed(&self) -> bool {
        match (self.best_bid(), self.best_ask()) {
            (Some((b, _)), Some((a, _))) => b >= a,
            _ => false,
        }
    }

    /// Bid levels as (price, qty), ascending by price.
    pub fn bid_levels(&self) -> Vec<(f64, f64)> {
        self.bids
            .iter()
            .map(|(t, q)| (from_ticks(*t), *q))
            .collect()
    }

    /// Ask levels as (price, qty), ascending by price.
    pub fn ask_levels(&self) -> Vec<(f64, f64)> {
        self.asks
            .iter()
            .map(|(t, q)| (from_ticks(*t), *q))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn book() -> OrderBook {
        let mut b = OrderBook::new();
        b.from_snapshot(
            &[(100.0, 5.0), (99.9, 3.0)],
            &[(100.1, 4.0), (100.2, 2.0)],
            10,
        );
        b
    }

    #[test]
    fn best_quotes_and_mid() {
        let b = book();
        assert_eq!(b.best_bid(), Some((100.0, 5.0)));
        assert_eq!(b.best_ask(), Some((100.1, 4.0)));
        assert!((b.midprice().unwrap() - 100.05).abs() < 1e-9);
        assert!((b.spread().unwrap() - 0.1).abs() < 1e-9);
    }

    #[test]
    fn apply_diff_updates_and_removes() {
        let mut b = book();
        // update 100.0 qty, remove 99.9, add 99.8
        let r = b.apply_diff(11, 11, &[(100.0, 7.0), (99.9, 0.0), (99.8, 1.0)], &[]);
        assert_eq!(r, SyncResult::Applied);
        assert_eq!(b.best_bid(), Some((100.0, 7.0)));
        assert_eq!(b.bids.len(), 2);
        assert_eq!(b.last_id, 11);
    }

    #[test]
    fn stale_event_ignored() {
        let mut b = book();
        let r = b.apply_diff(5, 9, &[(100.0, 99.0)], &[]);
        assert_eq!(r, SyncResult::Stale);
        assert_eq!(b.best_bid(), Some((100.0, 5.0)));
    }

    #[test]
    fn gap_detected() {
        let mut b = book();
        let r = b.apply_diff(12, 13, &[(100.0, 99.0)], &[]);
        assert_eq!(r, SyncResult::Gap);
        // book untouched
        assert_eq!(b.best_bid(), Some((100.0, 5.0)));
    }

    #[test]
    fn imbalance_sign() {
        let b = book();
        // bids 5+3=8, asks 4+2=6 -> (8-6)/14
        let imb = b.imbalance(5).unwrap();
        assert!((imb - 2.0 / 14.0).abs() < 1e-9);
    }

    #[test]
    fn empty_book_quotes_none() {
        let b = OrderBook::new();
        assert_eq!(b.midprice(), None);
        assert_eq!(b.touch(), None);
        assert_eq!(b.imbalance(5), None);
    }

    #[test]
    fn exact_next_update_applies() {
        let mut b = book();
        let r = b.apply_diff(11, 11, &[(100.0, 8.0)], &[]);
        assert_eq!(r, SyncResult::Applied);
        assert_eq!(b.last_id, 11);
        assert_eq!(b.best_bid(), Some((100.0, 8.0)));
    }

    #[test]
    fn overlapping_redelivery_is_idempotent() {
        let mut b = book();
        // same range re-applied: state must not change
        let r = b.apply_diff(9, 10, &[(100.0, 5.0)], &[(100.1, 4.0)]);
        assert_eq!(r, SyncResult::Applied);
        assert_eq!(b.last_id, 10);
        assert_eq!(b.best_bid(), Some((100.0, 5.0)));
        assert_eq!(b.best_ask(), Some((100.1, 4.0)));
    }

    #[test]
    fn partially_overlapping_event_applies() {
        let mut b = book();
        // U <= last_id < u: newer tail must be honored
        let r = b.apply_diff(10, 12, &[(100.0, 9.0)], &[]);
        assert_eq!(r, SyncResult::Applied);
        assert_eq!(b.last_id, 12);
        assert_eq!(b.best_bid(), Some((100.0, 9.0)));
    }

    #[test]
    fn duplicate_update_id_applies_idempotently() {
        let mut b = book();
        let r = b.apply_diff(10, 10, &[(100.0, 5.0)], &[]);
        assert_eq!(r, SyncResult::Applied);
        assert_eq!(b.last_id, 10);
        assert_eq!(b.best_bid(), Some((100.0, 5.0)));
    }

    #[test]
    fn one_past_boundary_is_gap() {
        let mut b = book();
        // U == last_id + 2 misses exactly one update
        let r = b.apply_diff(12, 12, &[], &[]);
        assert_eq!(r, SyncResult::Gap);
        assert_eq!(b.last_id, 10);
    }

    #[test]
    fn fully_stale_range_ignored() {
        let mut b = book();
        // u < last_id: entirely in the past
        let r = b.apply_diff(7, 9, &[(100.0, 1.0)], &[(100.1, 1.0)]);
        assert_eq!(r, SyncResult::Stale);
        assert_eq!(b.best_bid(), Some((100.0, 5.0)));
        assert_eq!(b.best_ask(), Some((100.1, 4.0)));
    }

    #[test]
    fn snapshot_replaces_book_entirely() {
        let mut b = book();
        b.from_snapshot(&[(200.0, 1.0)], &[(200.5, 1.0)], 500);
        assert_eq!(b.best_bid(), Some((200.0, 1.0)));
        assert_eq!(b.best_ask(), Some((200.5, 1.0)));
        assert_eq!(b.last_id, 500);
        assert_eq!(b.len(), 2);
        // old levels are gone
        assert_eq!(b.bid_levels(), vec![(200.0, 1.0)]);
    }

    #[test]
    fn snapshot_drops_zero_and_negative_qty() {
        let mut b = OrderBook::new();
        b.from_snapshot(
            &[(100.0, 5.0), (99.9, 0.0), (99.8, -2.0)],
            &[(100.1, 0.0)],
            1,
        );
        assert_eq!(b.bid_levels(), vec![(100.0, 5.0)]);
        assert_eq!(b.ask_levels(), Vec::<(f64, f64)>::new());
    }

    #[test]
    fn crossed_book_detected() {
        let mut b = OrderBook::new();
        // bid above ask: crossed
        b.from_snapshot(&[(100.2, 5.0)], &[(100.1, 4.0)], 1);
        assert!(b.is_crossed());
        // a diff that uncrosses clears the flag
        b.apply_diff(2, 2, &[(100.2, 0.0), (99.9, 5.0)], &[]);
        assert!(!b.is_crossed());
    }

    #[test]
    fn crossed_book_edge_cases() {
        let b = OrderBook::new();
        assert!(!b.is_crossed()); // empty: not crossed, just empty
        let mut one_sided = OrderBook::new();
        one_sided.from_snapshot(&[(100.0, 5.0)], &[], 1);
        assert!(!one_sided.is_crossed());
        // locked market (bid == ask) counts as crossed
        let mut locked = OrderBook::new();
        locked.from_snapshot(&[(100.0, 5.0)], &[(100.0, 4.0)], 1);
        assert!(locked.is_crossed());
    }

    #[test]
    fn level_accessors_cover_whole_book() {
        let b = book();
        assert_eq!(b.bid_levels(), vec![(99.9, 3.0), (100.0, 5.0)]);
        assert_eq!(b.ask_levels(), vec![(100.1, 4.0), (100.2, 2.0)]);
    }

    #[test]
    fn removing_absent_level_is_noop() {
        let mut b = book();
        let before = b.bid_levels();
        let r = b.apply_diff(11, 11, &[(77.7, 0.0)], &[(88.8, 0.0)]);
        assert_eq!(r, SyncResult::Applied);
        assert_eq!(b.bid_levels(), before);
    }

    #[test]
    fn book_emptied_by_diffs_stays_usable() {
        let mut b = book();
        b.apply_diff(
            11,
            11,
            &[(100.0, 0.0), (99.9, 0.0)],
            &[(100.1, 0.0), (100.2, 0.0)],
        );
        assert!(b.is_empty());
        assert_eq!(b.best_bid(), None);
        assert_eq!(b.midprice(), None);
        assert!(!b.is_crossed());
        // and can be repopulated
        let r = b.apply_diff(12, 12, &[(100.0, 5.0)], &[(100.1, 4.0)]);
        assert_eq!(r, SyncResult::Applied);
        assert_eq!(b.best_bid(), Some((100.0, 5.0)));
    }
}
