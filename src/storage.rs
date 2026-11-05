//! Parquet read/write for recorded sessions, via arrow record batches.

use std::collections::VecDeque;
use std::error::Error;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{Array, BooleanArray, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use serde::{Deserialize, Serialize};

use crate::metrics::{Quote, Trade};
/// Typed error for Parquet session reads. Every variant carries the
/// column and row needed to locate the bad record; no invalid record
/// is silently coerced into a default value.
#[derive(Debug)]
pub enum StorageError {
    Io(String),
    Parquet(String),
    Arrow(String),
    MissingColumn(String),
    WrongType {
        column: String,
        expected: String,
    },
    InconsistentLength {
        column: String,
        expected: usize,
        actual: usize,
    },
    NullValue {
        column: String,
        row: usize,
    },
    NonFinite {
        column: String,
        row: usize,
    },
    InvalidValue {
        column: String,
        row: usize,
        reason: String,
    },
    NonMonotonicTimestamp {
        row: usize,
        previous: i64,
        current: i64,
    },
}

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StorageError::Io(m) => write!(f, "io error: {m}"),
            StorageError::Parquet(m) => write!(f, "parquet error: {m}"),
            StorageError::Arrow(m) => write!(f, "arrow error: {m}"),
            StorageError::MissingColumn(c) => {
                write!(f, "parquet batch is missing column '{c}'")
            }
            StorageError::WrongType { column, expected } => {
                write!(f, "parquet column '{column}' is not {expected}")
            }
            StorageError::InconsistentLength {
                column,
                expected,
                actual,
            } => write!(
                f,
                "parquet column '{column}' has {actual} values for {expected} rows"
            ),
            StorageError::NullValue { column, row } => {
                write!(f, "parquet column '{column}' has a null value at row {row}")
            }
            StorageError::NonFinite { column, row } => {
                write!(
                    f,
                    "parquet column '{column}' has a nonfinite value at row {row}"
                )
            }
            StorageError::InvalidValue {
                column,
                row,
                reason,
            } => {
                write!(
                    f,
                    "parquet column '{column}' is invalid at row {row}: {reason}"
                )
            }
            StorageError::NonMonotonicTimestamp {
                row,
                previous,
                current,
            } => write!(
                f,
                "timestamps are not monotonic at row {row}: {current} follows {previous}"
            ),
        }
    }
}

impl Error for StorageError {}

fn trades_schema() -> Schema {
    Schema::new(vec![
        Field::new("time", DataType::Int64, false),
        Field::new("price", DataType::Float64, false),
        Field::new("qty", DataType::Float64, false),
        Field::new("is_buyer_maker", DataType::Boolean, false),
    ])
}

fn quotes_schema() -> Schema {
    Schema::new(vec![
        Field::new("time", DataType::Int64, false),
        Field::new("bid_px", DataType::Float64, false),
        Field::new("bid_sz", DataType::Float64, false),
        Field::new("ask_px", DataType::Float64, false),
        Field::new("ask_sz", DataType::Float64, false),
    ])
}

pub fn write_trades(path: &str, trades: &[Trade]) -> Result<(), Box<dyn Error>> {
    let schema = Arc::new(trades_schema());
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from_iter_values(
                trades.iter().map(|t| t.time_ms),
            )),
            Arc::new(Float64Array::from_iter_values(
                trades.iter().map(|t| t.price()),
            )),
            Arc::new(Float64Array::from_iter_values(trades.iter().map(|t| t.qty))),
            Arc::new(BooleanArray::from_iter(
                trades.iter().map(|t| Some(t.is_buyer_maker)),
            )),
        ],
    )?;
    let file = File::create(path)?;
    let mut w = ArrowWriter::try_new(file, schema, None)?;
    w.write(&batch)?;
    w.close()?;
    Ok(())
}

pub fn write_quotes(path: &str, quotes: &[Quote]) -> Result<(), Box<dyn Error>> {
    let schema = Arc::new(quotes_schema());
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from_iter_values(
                quotes.iter().map(|q| q.time_ms),
            )),
            Arc::new(Float64Array::from_iter_values(
                quotes.iter().map(|q| q.bid_px),
            )),
            Arc::new(Float64Array::from_iter_values(
                quotes.iter().map(|q| q.bid_sz),
            )),
            Arc::new(Float64Array::from_iter_values(
                quotes.iter().map(|q| q.ask_px),
            )),
            Arc::new(Float64Array::from_iter_values(
                quotes.iter().map(|q| q.ask_sz),
            )),
        ],
    )?;
    let file = File::create(path)?;
    let mut w = ArrowWriter::try_new(file, schema, None)?;
    w.write(&batch)?;
    w.close()?;
    Ok(())
}

fn col_i64(batch: &RecordBatch, name: &str, row_offset: usize) -> Result<Vec<i64>, StorageError> {
    let a = batch
        .column_by_name(name)
        .ok_or_else(|| StorageError::MissingColumn(name.to_string()))?
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| StorageError::WrongType {
            column: name.to_string(),
            expected: "int64".to_string(),
        })?;
    if a.len() != batch.num_rows() {
        return Err(StorageError::InconsistentLength {
            column: name.to_string(),
            expected: batch.num_rows(),
            actual: a.len(),
        });
    }
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        if a.is_null(i) {
            return Err(StorageError::NullValue {
                column: name.to_string(),
                row: row_offset + i,
            });
        }
        out.push(a.value(i));
    }
    Ok(out)
}

fn col_f64(batch: &RecordBatch, name: &str, row_offset: usize) -> Result<Vec<f64>, StorageError> {
    let a = batch
        .column_by_name(name)
        .ok_or_else(|| StorageError::MissingColumn(name.to_string()))?
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| StorageError::WrongType {
            column: name.to_string(),
            expected: "float64".to_string(),
        })?;
    if a.len() != batch.num_rows() {
        return Err(StorageError::InconsistentLength {
            column: name.to_string(),
            expected: batch.num_rows(),
            actual: a.len(),
        });
    }
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        if a.is_null(i) {
            return Err(StorageError::NullValue {
                column: name.to_string(),
                row: row_offset + i,
            });
        }
        out.push(a.value(i));
    }
    Ok(out)
}

fn col_bool(batch: &RecordBatch, name: &str, row_offset: usize) -> Result<Vec<bool>, StorageError> {
    let a = batch
        .column_by_name(name)
        .ok_or_else(|| StorageError::MissingColumn(name.to_string()))?
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| StorageError::WrongType {
            column: name.to_string(),
            expected: "boolean".to_string(),
        })?;
    if a.len() != batch.num_rows() {
        return Err(StorageError::InconsistentLength {
            column: name.to_string(),
            expected: batch.num_rows(),
            actual: a.len(),
        });
    }
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        if a.is_null(i) {
            return Err(StorageError::NullValue {
                column: name.to_string(),
                row: row_offset + i,
            });
        }
        out.push(a.value(i));
    }
    Ok(out)
}

fn check_finite(column: &str, row: usize, v: f64) -> Result<(), StorageError> {
    if !v.is_finite() {
        return Err(StorageError::NonFinite {
            column: column.to_string(),
            row,
        });
    }
    Ok(())
}

fn check_positive(column: &str, row: usize, v: f64) -> Result<(), StorageError> {
    check_finite(column, row, v)?;
    if v <= 0.0 {
        return Err(StorageError::InvalidValue {
            column: column.to_string(),
            row,
            reason: "value must be positive".to_string(),
        });
    }
    Ok(())
}

fn check_timestamp(row: usize, t: i64, prev: &mut Option<i64>) -> Result<(), StorageError> {
    if t < 0 {
        return Err(StorageError::InvalidValue {
            column: "time".to_string(),
            row,
            reason: "timestamp must not be negative".to_string(),
        });
    }
    if let Some(p) = *prev {
        if t < p {
            return Err(StorageError::NonMonotonicTimestamp {
                row,
                previous: p,
                current: t,
            });
        }
    }
    *prev = Some(t);
    Ok(())
}

pub fn read_trades(path: &str) -> Result<Vec<Trade>, StorageError> {
    let file = File::open(path).map_err(|e| StorageError::Io(e.to_string()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| StorageError::Parquet(e.to_string()))?
        .build()
        .map_err(|e| StorageError::Parquet(e.to_string()))?;
    let mut out = Vec::new();
    let mut prev_time: Option<i64> = None;
    for batch in reader {
        let b = batch.map_err(|e| StorageError::Parquet(e.to_string()))?;
        let row_offset = out.len();
        let time = col_i64(&b, "time", row_offset)?;
        let price = col_f64(&b, "price", row_offset)?;
        let qty = col_f64(&b, "qty", row_offset)?;
        let mbm = col_bool(&b, "is_buyer_maker", row_offset)?;
        for i in 0..b.num_rows() {
            let row = row_offset + i;
            check_timestamp(row, time[i], &mut prev_time)?;
            check_positive("price", row, price[i])?;
            check_positive("qty", row, qty[i])?;
            out.push(Trade {
                time_ms: time[i],
                price: price[i],
                qty: qty[i],
                is_buyer_maker: mbm[i],
            });
        }
    }
    Ok(out)
}

pub fn read_quotes(path: &str) -> Result<Vec<Quote>, StorageError> {
    let file = File::open(path).map_err(|e| StorageError::Io(e.to_string()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| StorageError::Parquet(e.to_string()))?
        .build()
        .map_err(|e| StorageError::Parquet(e.to_string()))?;
    let mut out = Vec::new();
    let mut prev_time: Option<i64> = None;
    for batch in reader {
        let b = batch.map_err(|e| StorageError::Parquet(e.to_string()))?;
        let row_offset = out.len();
        let time = col_i64(&b, "time", row_offset)?;
        let bp = col_f64(&b, "bid_px", row_offset)?;
        let bq = col_f64(&b, "bid_sz", row_offset)?;
        let ap = col_f64(&b, "ask_px", row_offset)?;
        let aq = col_f64(&b, "ask_sz", row_offset)?;
        for i in 0..b.num_rows() {
            let row = row_offset + i;
            check_timestamp(row, time[i], &mut prev_time)?;
            check_positive("bid_px", row, bp[i])?;
            check_positive("bid_sz", row, bq[i])?;
            check_positive("ask_px", row, ap[i])?;
            check_positive("ask_sz", row, aq[i])?;
            if bp[i] >= ap[i] {
                return Err(StorageError::InvalidValue {
                    column: "bid_px".to_string(),
                    row,
                    reason: "bid price must be below ask price (crossed book)".to_string(),
                });
            }
            out.push(Quote {
                time_ms: time[i],
                bid_px: bp[i],
                bid_sz: bq[i],
                ask_px: ap[i],
                ask_sz: aq[i],
            });
        }
    }
    Ok(out)
}

/// One typed record in the append-only session journal. The journal
/// is the raw event record of a recording session: unlike the
/// end-of-session trades/quotes files it keeps every depth update and
/// every snapshot, so a session can be replayed exactly and a resync
/// boundary is visible as the snapshot record that marks it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum JournalRecord {
    SessionStart {
        symbol: String,
    },
    Snapshot {
        last_update_id: i64,
        bids: Vec<(f64, f64)>,
        asks: Vec<(f64, f64)>,
    },
    Depth {
        time_ms: i64,
        u_first: i64,
        u_last: i64,
        bids: Vec<(f64, f64)>,
        asks: Vec<(f64, f64)>,
    },
    Trade {
        time_ms: i64,
        price: f64,
        qty: f64,
        is_buyer_maker: bool,
    },
    /// Records dropped because the sink could not keep up. Written
    /// into the journal itself, so a loss is never silent.
    Overflow {
        dropped: u64,
    },
    SessionEnd {
        records: u64,
    },
}

impl JournalRecord {
    pub fn kind_str(&self) -> &'static str {
        match self {
            JournalRecord::SessionStart { .. } => "session_start",
            JournalRecord::Snapshot { .. } => "snapshot",
            JournalRecord::Depth { .. } => "depth",
            JournalRecord::Trade { .. } => "trade",
            JournalRecord::Overflow { .. } => "overflow",
            JournalRecord::SessionEnd { .. } => "session_end",
        }
    }

    fn time_ms(&self) -> i64 {
        match self {
            JournalRecord::Depth { time_ms, .. } | JournalRecord::Trade { time_ms, .. } => *time_ms,
            _ => 0,
        }
    }

    /// Canonical payload bytes for this record. serde_json emits
    /// struct fields in declaration order, so the encoding is stable
    /// for a given record value.
    fn payload(&self) -> String {
        serde_json::to_string(self).expect("journal record serializes")
    }
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

fn record_checksum(payload: &str) -> u64 {
    fnv1a(FNV_OFFSET, payload.as_bytes())
}

/// Fold a record sequence into one digest. Replaying a journal and
/// folding the records it yields must reproduce the digest recorded
/// from the original sequence.
pub fn journal_digest(records: &[JournalRecord]) -> String {
    let mut h = FNV_OFFSET;
    for (i, r) in records.iter().enumerate() {
        h = fnv1a(h, &(i as u64).to_le_bytes());
        h = fnv1a(h, r.payload().as_bytes());
    }
    format!("{h:016x}")
}

/// A journal record with its assigned sequence number.
#[derive(Debug, Clone, PartialEq)]
pub struct JournalEntry {
    pub seq: u64,
    pub record: JournalRecord,
}

/// A completed batch of journal entries, ready for a sink.
#[derive(Debug, Clone, PartialEq)]
pub struct JournalBatch {
    pub session_id: String,
    pub entries: Vec<JournalEntry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JournalAppend {
    Stored,
    Dropped,
}

/// The in-memory front of the journal. Records accumulate into a
/// filling batch; a full batch moves to a bounded pending queue for
/// the sink. When the queue and the filling batch are both at
/// capacity, appends drop the incoming record and count it; the next
/// append that finds room first writes an explicit `Overflow` record
/// carrying the count, and `finish` does the same at the end. Memory
/// is therefore bounded by `batch_size * (max_pending_batches + 1)`
/// records no matter how slow the sink is.
pub struct JournalCore {
    session_id: String,
    batch_size: usize,
    max_pending_batches: usize,
    filling: Vec<JournalEntry>,
    pending: VecDeque<JournalBatch>,
    next_seq: u64,
    dropped_pending: u64,
    stored: u64,
}

impl JournalCore {
    pub fn new(session_id: &str, batch_size: usize, max_pending_batches: usize) -> Self {
        assert!(batch_size >= 1, "journal batch size must be positive");
        Self {
            session_id: session_id.to_string(),
            batch_size,
            max_pending_batches,
            filling: Vec::new(),
            pending: VecDeque::new(),
            next_seq: 0,
            dropped_pending: 0,
            stored: 0,
        }
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Hard bound on records held in memory.
    pub fn max_buffered_records(&self) -> usize {
        self.batch_size * (self.max_pending_batches + 1)
    }

    pub fn buffered_records(&self) -> usize {
        self.filling.len() + self.pending.iter().map(|b| b.entries.len()).sum::<usize>()
    }

    /// Records dropped so far that no overflow record has covered yet.
    pub fn dropped_pending(&self) -> u64 {
        self.dropped_pending
    }

    /// Total records stored, including overflow records.
    pub fn stored_records(&self) -> u64 {
        self.stored
    }

    fn store(&mut self, record: JournalRecord) {
        let entry = JournalEntry {
            seq: self.next_seq,
            record,
        };
        self.next_seq += 1;
        self.stored += 1;
        self.filling.push(entry);
        if self.filling.len() == self.batch_size {
            let entries = std::mem::take(&mut self.filling);
            self.pending.push_back(JournalBatch {
                session_id: self.session_id.clone(),
                entries,
            });
        }
    }

    pub fn append(&mut self, record: JournalRecord) -> JournalAppend {
        if self.dropped_pending > 0 && self.buffered_records() < self.max_buffered_records() {
            let dropped = self.dropped_pending;
            self.dropped_pending = 0;
            self.store(JournalRecord::Overflow { dropped });
        }
        if self.buffered_records() >= self.max_buffered_records() {
            self.dropped_pending += 1;
            return JournalAppend::Dropped;
        }
        self.store(record);
        JournalAppend::Stored
    }

    /// Take the oldest completed batch for the sink.
    pub fn take_batch(&mut self) -> Option<JournalBatch> {
        self.pending.pop_front()
    }

    /// Put a batch back after a failed sink write, so a transient
    /// sink error does not lose journaled records.
    pub fn restore_front(&mut self, batch: JournalBatch) {
        self.pending.push_front(batch);
    }

    /// Flush everything left: pending batches, the partial filling
    /// batch, and a trailing overflow record if drops are still
    /// uncovered. The returned batch may exceed the nominal batch
    /// size; it is the final write of the journal.
    pub fn finish(mut self) -> Option<JournalBatch> {
        let mut entries: Vec<JournalEntry> = Vec::new();
        for batch in self.pending.drain(..) {
            entries.extend(batch.entries);
        }
        entries.append(&mut self.filling);
        if self.dropped_pending > 0 {
            entries.push(JournalEntry {
                seq: self.next_seq,
                record: JournalRecord::Overflow {
                    dropped: self.dropped_pending,
                },
            });
        }
        if entries.is_empty() {
            None
        } else {
            Some(JournalBatch {
                session_id: self.session_id.clone(),
                entries,
            })
        }
    }
}

/// Destination for completed journal batches.
pub trait JournalSink {
    fn write_batch(&mut self, batch: &JournalBatch) -> Result<(), StorageError>;
}

/// Move every completed batch from the core to the sink. Returns the
/// number of batches written. On a sink error the batch is restored
/// to the front of the queue and the error is returned.
pub fn pump_journal(
    core: &mut JournalCore,
    sink: &mut dyn JournalSink,
) -> Result<usize, StorageError> {
    let mut written = 0;
    while let Some(batch) = core.take_batch() {
        match sink.write_batch(&batch) {
            Ok(()) => written += 1,
            Err(e) => {
                core.restore_front(batch);
                return Err(e);
            }
        }
    }
    Ok(written)
}

fn journal_schema() -> Schema {
    Schema::new(vec![
        Field::new("seq", DataType::Int64, false),
        Field::new("session_id", DataType::Utf8, false),
        Field::new("kind", DataType::Utf8, false),
        Field::new("time_ms", DataType::Int64, false),
        Field::new("checksum", DataType::Int64, false),
        Field::new("payload", DataType::Utf8, false),
    ])
}

fn journal_batch(batch: &JournalBatch) -> Result<RecordBatch, StorageError> {
    let n = batch.entries.len();
    let mut seqs = Vec::with_capacity(n);
    let mut kinds = Vec::with_capacity(n);
    let mut times = Vec::with_capacity(n);
    let mut checksums = Vec::with_capacity(n);
    let mut payloads = Vec::with_capacity(n);
    for e in &batch.entries {
        let payload = e.record.payload();
        seqs.push(e.seq as i64);
        kinds.push(e.record.kind_str());
        times.push(e.record.time_ms());
        checksums.push(record_checksum(&payload) as i64);
        payloads.push(payload);
    }
    RecordBatch::try_new(
        Arc::new(journal_schema()),
        vec![
            Arc::new(Int64Array::from_iter_values(seqs)),
            Arc::new(StringArray::from(vec![batch.session_id.as_str(); n])),
            Arc::new(StringArray::from(kinds)),
            Arc::new(Int64Array::from_iter_values(times)),
            Arc::new(Int64Array::from_iter_values(checksums)),
            Arc::new(StringArray::from(payloads)),
        ],
    )
    .map_err(|e| StorageError::Arrow(e.to_string()))
}

fn segment_index(name: &str) -> Option<u64> {
    name.strip_prefix("seg-")?
        .strip_suffix(".parquet")?
        .parse()
        .ok()
}

/// Journal sink that writes each batch as one segment file,
/// `seg-NNNNNN.parquet`, inside a journal directory. A batch is
/// written to a `.tmp` file, synced, and renamed into place, so a
/// segment only becomes visible to replay once it is complete: a
/// process killed mid-write leaves a `.tmp` file that replay
/// ignores, and every completed batch survives. One journal
/// directory holds one session; segment numbering continues after
/// the highest existing segment rather than overwriting it.
pub struct JournalFileSink {
    dir: PathBuf,
    next_segment: u64,
}

impl JournalFileSink {
    pub fn create(dir: &Path) -> Result<Self, StorageError> {
        std::fs::create_dir_all(dir).map_err(|e| StorageError::Io(e.to_string()))?;
        let mut next = 0u64;
        let rd = std::fs::read_dir(dir).map_err(|e| StorageError::Io(e.to_string()))?;
        for entry in rd {
            let entry = entry.map_err(|e| StorageError::Io(e.to_string()))?;
            if let Some(name) = entry.file_name().to_str() {
                if let Some(idx) = segment_index(name) {
                    next = next.max(idx + 1);
                }
            }
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            next_segment: next,
        })
    }
}

impl JournalSink for JournalFileSink {
    fn write_batch(&mut self, batch: &JournalBatch) -> Result<(), StorageError> {
        if batch.entries.is_empty() {
            return Ok(());
        }
        let rb = journal_batch(batch)?;
        let tmp = self
            .dir
            .join(format!("seg-{:06}.parquet.tmp", self.next_segment));
        let fin = self
            .dir
            .join(format!("seg-{:06}.parquet", self.next_segment));
        {
            let f = File::create(&tmp).map_err(|e| StorageError::Io(e.to_string()))?;
            let mut w = ArrowWriter::try_new(f, Arc::new(journal_schema()), None)
                .map_err(|e| StorageError::Parquet(e.to_string()))?;
            w.write(&rb)
                .map_err(|e| StorageError::Parquet(e.to_string()))?;
            w.close()
                .map_err(|e| StorageError::Parquet(e.to_string()))?;
        }
        File::open(&tmp)
            .and_then(|f| f.sync_all())
            .map_err(|e| StorageError::Io(e.to_string()))?;
        std::fs::rename(&tmp, &fin).map_err(|e| StorageError::Io(e.to_string()))?;
        self.next_segment += 1;
        Ok(())
    }
}

/// The result of replaying a journal directory.
#[derive(Debug)]
pub struct JournalReplay {
    pub session_id: String,
    pub records: Vec<JournalRecord>,
    pub digest: String,
}

fn col_str(
    batch: &RecordBatch,
    name: &str,
    row_offset: usize,
) -> Result<Vec<String>, StorageError> {
    let a = batch
        .column_by_name(name)
        .ok_or_else(|| StorageError::MissingColumn(name.to_string()))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| StorageError::WrongType {
            column: name.to_string(),
            expected: "utf8".to_string(),
        })?;
    if a.len() != batch.num_rows() {
        return Err(StorageError::InconsistentLength {
            column: name.to_string(),
            expected: batch.num_rows(),
            actual: a.len(),
        });
    }
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        if a.is_null(i) {
            return Err(StorageError::NullValue {
                column: name.to_string(),
                row: row_offset + i,
            });
        }
        out.push(a.value(i).to_string());
    }
    Ok(out)
}

/// Read a journal directory back, in segment order. Every record is
/// validated before it is returned: sequence numbers must be
/// contiguous from zero, the session id must not change, the kind
/// column must match the payload, and each payload must match its
/// stored checksum. Incomplete `.tmp` segments are ignored.
pub fn replay_journal(dir: &Path) -> Result<JournalReplay, StorageError> {
    let rd = std::fs::read_dir(dir).map_err(|e| StorageError::Io(e.to_string()))?;
    let mut segments: Vec<(u64, PathBuf)> = Vec::new();
    for entry in rd {
        let entry = entry.map_err(|e| StorageError::Io(e.to_string()))?;
        if let Some(name) = entry.file_name().to_str() {
            if let Some(idx) = segment_index(name) {
                segments.push((idx, entry.path()));
            }
        }
    }
    segments.sort();
    if segments.is_empty() {
        return Err(StorageError::Io(format!(
            "no journal segments in {}",
            dir.display()
        )));
    }
    let mut records: Vec<JournalRecord> = Vec::new();
    let mut session_id: Option<String> = None;
    let mut expected_seq: u64 = 0;
    let mut row_base = 0usize;
    for (_, path) in &segments {
        let file = File::open(path).map_err(|e| StorageError::Io(e.to_string()))?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| StorageError::Parquet(e.to_string()))?
            .build()
            .map_err(|e| StorageError::Parquet(e.to_string()))?;
        for batch in reader {
            let b = batch.map_err(|e| StorageError::Parquet(e.to_string()))?;
            let seqs = col_i64(&b, "seq", row_base)?;
            let sessions = col_str(&b, "session_id", row_base)?;
            let kinds = col_str(&b, "kind", row_base)?;
            let _times = col_i64(&b, "time_ms", row_base)?;
            let checksums = col_i64(&b, "checksum", row_base)?;
            let payloads = col_str(&b, "payload", row_base)?;
            for i in 0..b.num_rows() {
                let row = row_base + i;
                match &session_id {
                    None => session_id = Some(sessions[i].clone()),
                    Some(s) if *s != sessions[i] => {
                        return Err(StorageError::InvalidValue {
                            column: "session_id".to_string(),
                            row,
                            reason: "session id changes mid-journal".to_string(),
                        });
                    }
                    _ => {}
                }
                if seqs[i] < 0 || seqs[i] as u64 != expected_seq {
                    return Err(StorageError::InvalidValue {
                        column: "seq".to_string(),
                        row,
                        reason: format!("expected sequence {expected_seq}, found {}", seqs[i]),
                    });
                }
                expected_seq += 1;
                if record_checksum(&payloads[i]) != checksums[i] as u64 {
                    return Err(StorageError::InvalidValue {
                        column: "checksum".to_string(),
                        row,
                        reason: "payload checksum mismatch".to_string(),
                    });
                }
                let record: JournalRecord =
                    serde_json::from_str(&payloads[i]).map_err(|e| StorageError::InvalidValue {
                        column: "payload".to_string(),
                        row,
                        reason: format!("payload does not decode: {e}"),
                    })?;
                if record.kind_str() != kinds[i] {
                    return Err(StorageError::InvalidValue {
                        column: "kind".to_string(),
                        row,
                        reason: "kind column does not match payload".to_string(),
                    });
                }
                records.push(record);
            }
            row_base += b.num_rows();
        }
    }
    let digest = journal_digest(&records);
    Ok(JournalReplay {
        session_id: session_id.unwrap_or_default(),
        records,
        digest,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn write_batch(dir: &str, file: &str, schema: Schema, cols: Vec<ArrayRef>) {
        let schema = Arc::new(schema);
        let batch = RecordBatch::try_new(schema.clone(), cols).unwrap();
        let path = std::path::Path::new(dir).join(file);
        let f = File::create(&path).unwrap();
        let mut w = ArrowWriter::try_new(f, schema, None).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
    }

    fn tmpdir(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("toxicflow_h3_{}_{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn trades_schema_opt() -> Schema {
        Schema::new(vec![
            Field::new("time", DataType::Int64, true),
            Field::new("price", DataType::Float64, true),
            Field::new("qty", DataType::Float64, true),
            Field::new("is_buyer_maker", DataType::Boolean, true),
        ])
    }

    fn quotes_schema_opt() -> Schema {
        Schema::new(vec![
            Field::new("time", DataType::Int64, true),
            Field::new("bid_px", DataType::Float64, true),
            Field::new("bid_sz", DataType::Float64, true),
            Field::new("ask_px", DataType::Float64, true),
            Field::new("ask_sz", DataType::Float64, true),
        ])
    }

    fn valid_trades_file(dir: &std::path::Path) -> String {
        write_batch(
            dir.to_str().unwrap(),
            "trades.parquet",
            trades_schema_opt(),
            vec![
                Arc::new(Int64Array::from(vec![1000i64, 2000, 2000, 3000])),
                Arc::new(Float64Array::from(vec![100.0f64, 100.5, 101.0, 99.5])),
                Arc::new(Float64Array::from(vec![0.5f64, 1.0, 0.25, 2.0])),
                Arc::new(BooleanArray::from(vec![true, false, true, false])),
            ],
        );
        dir.join("trades.parquet").to_str().unwrap().to_string()
    }

    fn valid_quotes_file(dir: &std::path::Path) -> String {
        write_batch(
            dir.to_str().unwrap(),
            "quotes.parquet",
            quotes_schema_opt(),
            vec![
                Arc::new(Int64Array::from(vec![1000i64, 2000, 3000])),
                Arc::new(Float64Array::from(vec![99.9f64, 100.4, 99.4])),
                Arc::new(Float64Array::from(vec![5.0f64, 4.0, 6.0])),
                Arc::new(Float64Array::from(vec![100.1f64, 100.6, 99.6])),
                Arc::new(Float64Array::from(vec![5.0f64, 3.0, 6.0])),
            ],
        );
        dir.join("quotes.parquet").to_str().unwrap().to_string()
    }

    #[test]
    fn stage3_toxicflow_3_valid_fixture_unchanged() {
        let dir = tmpdir("valid");
        let trades = read_trades(&valid_trades_file(&dir)).unwrap();
        let quotes = read_quotes(&valid_quotes_file(&dir)).unwrap();
        assert_eq!(trades.len(), 4);
        assert_eq!(trades[0].time_ms, 1000);
        assert_eq!(trades[0].price, 100.0);
        assert_eq!(trades[0].qty, 0.5);
        assert!(trades[0].is_buyer_maker);
        assert_eq!(trades[2].time_ms, 2000);
        assert_eq!(quotes.len(), 3);
        assert_eq!(quotes[1].bid_px, 100.4);
        assert_eq!(quotes[1].ask_px, 100.6);
        assert_eq!(quotes[1].bid_sz, 4.0);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn stage3_toxicflow_3_null_price_rejected() {
        let dir = tmpdir("null");
        write_batch(
            dir.to_str().unwrap(),
            "trades.parquet",
            trades_schema_opt(),
            vec![
                Arc::new(Int64Array::from(vec![1000i64, 2000])),
                Arc::new(Float64Array::from(vec![Some(100.0f64), None])),
                Arc::new(Float64Array::from(vec![0.5f64, 1.0])),
                Arc::new(BooleanArray::from(vec![true, false])),
            ],
        );
        let err = read_trades(dir.join("trades.parquet").to_str().unwrap()).unwrap_err();
        assert!(matches!(err, StorageError::NullValue { .. }), "got {err}");
        assert!(err.to_string().contains("price"), "got {err}");
        assert!(err.to_string().contains('1'), "got {err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn stage3_toxicflow_3_nan_price_rejected() {
        let dir = tmpdir("nan");
        write_batch(
            dir.to_str().unwrap(),
            "trades.parquet",
            trades_schema_opt(),
            vec![
                Arc::new(Int64Array::from(vec![1000i64, 2000])),
                Arc::new(Float64Array::from(vec![100.0f64, f64::NAN])),
                Arc::new(Float64Array::from(vec![0.5f64, 1.0])),
                Arc::new(BooleanArray::from(vec![true, false])),
            ],
        );
        let err = read_trades(dir.join("trades.parquet").to_str().unwrap()).unwrap_err();
        assert!(matches!(err, StorageError::NonFinite { .. }), "got {err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn stage3_toxicflow_3_negative_qty_rejected() {
        let dir = tmpdir("negqty");
        write_batch(
            dir.to_str().unwrap(),
            "trades.parquet",
            trades_schema_opt(),
            vec![
                Arc::new(Int64Array::from(vec![1000i64, 2000])),
                Arc::new(Float64Array::from(vec![100.0f64, 100.5])),
                Arc::new(Float64Array::from(vec![0.5f64, -1.0])),
                Arc::new(BooleanArray::from(vec![true, false])),
            ],
        );
        let err = read_trades(dir.join("trades.parquet").to_str().unwrap()).unwrap_err();
        assert!(
            matches!(err, StorageError::InvalidValue { .. }),
            "got {err}"
        );
        assert!(err.to_string().contains("qty"), "got {err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn stage3_toxicflow_3_nonmonotonic_timestamps_rejected() {
        let dir = tmpdir("nonmono");
        write_batch(
            dir.to_str().unwrap(),
            "trades.parquet",
            trades_schema_opt(),
            vec![
                Arc::new(Int64Array::from(vec![3000i64, 2000])),
                Arc::new(Float64Array::from(vec![100.0f64, 100.5])),
                Arc::new(Float64Array::from(vec![0.5f64, 1.0])),
                Arc::new(BooleanArray::from(vec![true, false])),
            ],
        );
        let err = read_trades(dir.join("trades.parquet").to_str().unwrap()).unwrap_err();
        assert!(
            matches!(err, StorageError::NonMonotonicTimestamp { .. }),
            "got {err}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn stage3_toxicflow_3_missing_column_typed_error() {
        let dir = tmpdir("schema");
        let schema = Schema::new(vec![
            Field::new("time", DataType::Int64, true),
            Field::new("qty", DataType::Float64, true),
        ]);
        write_batch(
            dir.to_str().unwrap(),
            "trades.parquet",
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1000i64])),
                Arc::new(Float64Array::from(vec![0.5f64])),
            ],
        );
        let err = read_trades(dir.join("trades.parquet").to_str().unwrap()).unwrap_err();
        assert!(matches!(err, StorageError::MissingColumn(_)), "got {err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn stage3_toxicflow_3_crossed_book_rejected() {
        let dir = tmpdir("crossed");
        write_batch(
            dir.to_str().unwrap(),
            "quotes.parquet",
            quotes_schema_opt(),
            vec![
                Arc::new(Int64Array::from(vec![1000i64])),
                Arc::new(Float64Array::from(vec![100.1f64])),
                Arc::new(Float64Array::from(vec![5.0f64])),
                Arc::new(Float64Array::from(vec![100.1f64])),
                Arc::new(Float64Array::from(vec![5.0f64])),
            ],
        );
        let err = read_quotes(dir.join("quotes.parquet").to_str().unwrap()).unwrap_err();
        assert!(
            matches!(err, StorageError::InvalidValue { .. }),
            "got {err}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn stage3_toxicflow_3_zero_quote_size_rejected() {
        let dir = tmpdir("zerosize");
        write_batch(
            dir.to_str().unwrap(),
            "quotes.parquet",
            quotes_schema_opt(),
            vec![
                Arc::new(Int64Array::from(vec![1000i64])),
                Arc::new(Float64Array::from(vec![99.9f64])),
                Arc::new(Float64Array::from(vec![0.0f64])),
                Arc::new(Float64Array::from(vec![100.1f64])),
                Arc::new(Float64Array::from(vec![5.0f64])),
            ],
        );
        let err = read_quotes(dir.join("quotes.parquet").to_str().unwrap()).unwrap_err();
        assert!(
            matches!(err, StorageError::InvalidValue { .. }),
            "got {err}"
        );
        assert!(err.to_string().contains("bid_sz"), "got {err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn stage3_toxicflow_3_inconsistent_length_rejected() {
        // RecordBatch construction enforces equal lengths, so the guard
        // is exercised directly with an unchecked batch: a price column
        // one element short of the batch row count must fail, not panic.
        let schema = Arc::new(trades_schema_opt());
        let batch = unsafe {
            RecordBatch::new_unchecked(
                schema,
                vec![
                    Arc::new(Int64Array::from(vec![1000i64, 2000])),
                    Arc::new(Float64Array::from(vec![100.0f64])),
                    Arc::new(Float64Array::from(vec![0.5f64, 1.0])),
                    Arc::new(BooleanArray::from(vec![true, false])),
                ],
                2,
            )
        };
        let err = col_f64(&batch, "price", 0).unwrap_err();
        assert!(
            matches!(err, StorageError::InconsistentLength { .. }),
            "got {err}"
        );
    }

    // ---- TF3 journal fixtures ----
    // The acceptance clauses, as independent known-answer fixtures:
    // replay determinism, termination preserving completed batches,
    // and bounded memory with an explicit overflow record under a
    // slow sink.

    fn journal_records() -> Vec<JournalRecord> {
        vec![
            JournalRecord::SessionStart {
                symbol: "BTCUSDT".to_string(),
            },
            JournalRecord::Snapshot {
                last_update_id: 100,
                bids: vec![(100.0, 5.0), (99.9, 3.0)],
                asks: vec![(100.1, 4.0)],
            },
            JournalRecord::Depth {
                time_ms: 1_700_000_000_000,
                u_first: 101,
                u_last: 102,
                bids: vec![(100.0, 6.0)],
                asks: vec![],
            },
            JournalRecord::Trade {
                time_ms: 1_700_000_000_100,
                price: 100.05,
                qty: 0.25,
                is_buyer_maker: true,
            },
            JournalRecord::Depth {
                time_ms: 1_700_000_000_200,
                u_first: 103,
                u_last: 103,
                bids: vec![],
                asks: vec![(100.2, 9.0)],
            },
            JournalRecord::Trade {
                time_ms: 1_700_000_000_300,
                price: 100.1,
                qty: 1.5,
                is_buyer_maker: false,
            },
            JournalRecord::SessionEnd { records: 5 },
        ]
    }

    /// Drain every completed batch plus the final partial batch into
    /// the sink, in order.
    fn drain_journal(core: JournalCore, sink: &mut dyn JournalSink) {
        let mut core = core;
        while let Some(batch) = core.take_batch() {
            sink.write_batch(&batch).unwrap();
        }
        if let Some(batch) = core.finish() {
            sink.write_batch(&batch).unwrap();
        }
    }

    #[test]
    fn stage3_toxicflow_3_journal_replay_digest_deterministic() {
        let dir = tmpdir("journal_digest");
        let jdir = dir.join("journal");
        let records = journal_records();
        let mut core = JournalCore::new("sess-digest", 3, 4);
        for r in &records {
            assert_eq!(core.append(r.clone()), JournalAppend::Stored);
        }
        let mut sink = JournalFileSink::create(&jdir).unwrap();
        drain_journal(core, &mut sink);

        let first = replay_journal(&jdir).unwrap();
        let second = replay_journal(&jdir).unwrap();
        assert_eq!(first.records, records, "journal must round-trip exactly");
        assert_eq!(first.digest, second.digest, "replay digest must be stable");
        assert_eq!(first.digest, journal_digest(&records));
        assert_eq!(first.session_id, "sess-digest");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn stage3_toxicflow_3_journal_termination_preserves_completed_batches() {
        let dir = tmpdir("journal_crash");
        let jdir = dir.join("journal");
        let records = journal_records();
        let mut core = JournalCore::new("sess-crash", 3, 4);
        // 6 records = exactly 2 completed batches of 3.
        for r in &records[..6] {
            core.append(r.clone());
        }
        let mut sink = JournalFileSink::create(&jdir).unwrap();
        let mut written = 0;
        while let Some(batch) = core.take_batch() {
            sink.write_batch(&batch).unwrap();
            written += 1;
        }
        assert_eq!(written, 2);
        // More records arrive and sit in the filling batch; then the
        // process is killed: no finish(), and the segment being
        // written is left behind as a partial .tmp file.
        core.append(records[6].clone());
        std::fs::write(jdir.join("seg-000002.parquet.tmp"), b"PAR1partial").unwrap();
        drop(core);
        drop(sink);

        let replay = replay_journal(&jdir).unwrap();
        assert_eq!(
            replay.records,
            records[..6].to_vec(),
            "exactly the completed batches survive a kill"
        );
        assert_eq!(replay.digest, journal_digest(&records[..6]));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn stage3_toxicflow_3_journal_slow_sink_bounded_with_overflow_record() {
        let dir = tmpdir("journal_slow");
        let jdir = dir.join("journal");
        // batch of 4, at most 2 pending batches: buffer bound is 12.
        let mut core = JournalCore::new("sess-slow", 4, 2);
        assert_eq!(core.max_buffered_records(), 12);
        let trade = JournalRecord::Trade {
            time_ms: 1_700_000_000_000,
            price: 100.0,
            qty: 0.5,
            is_buyer_maker: false,
        };
        // The sink never drains during the feed: memory must stay
        // bounded and the excess must be counted, not held.
        for _ in 0..100 {
            core.append(trade.clone());
            assert!(
                core.buffered_records() <= 12,
                "buffer grew to {}",
                core.buffered_records()
            );
        }
        assert_eq!(core.buffered_records(), 12);
        assert_eq!(core.dropped_pending(), 88);

        let mut sink = JournalFileSink::create(&jdir).unwrap();
        drain_journal(core, &mut sink);
        let replay = replay_journal(&jdir).unwrap();
        assert_eq!(replay.records.len(), 13, "12 stored + 1 overflow record");
        assert!(
            matches!(replay.records[12], JournalRecord::Overflow { dropped: 88 }),
            "last record must be the explicit overflow record, got {:?}",
            replay.records[12]
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn stage3_toxicflow_3_journal_checksum_mismatch_rejected() {
        let dir = tmpdir("journal_tamper");
        let jdir = dir.join("journal");
        let mut core = JournalCore::new("sess-tamper", 8, 2);
        core.append(JournalRecord::Trade {
            time_ms: 1_700_000_000_000,
            price: 100.0,
            qty: 0.5,
            is_buyer_maker: false,
        });
        let mut sink = JournalFileSink::create(&jdir).unwrap();
        drain_journal(core, &mut sink);

        // Tamper with the stored payload (price 100.0 -> 900.0) while
        // leaving the stored checksum untouched: replay must refuse.
        let seg = jdir.join("seg-000000.parquet");
        let file = File::open(&seg).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        let batch = reader.into_iter().next().unwrap().unwrap();
        let payloads = batch
            .column_by_name("payload")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap();
        let tampered: Vec<String> = (0..payloads.len())
            .map(|i| payloads.value(i).replace("100.0", "900.0"))
            .collect();
        let mut cols: Vec<ArrayRef> = Vec::new();
        for (i, name) in [
            "seq",
            "session_id",
            "kind",
            "time_ms",
            "checksum",
            "payload",
        ]
        .iter()
        .enumerate()
        {
            if *name == "payload" {
                cols.push(Arc::new(arrow::array::StringArray::from(
                    tampered.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                )));
            } else {
                cols.push(batch.column(i).clone());
            }
        }
        let schema = Arc::new(journal_schema());
        let new_batch = RecordBatch::try_new(schema.clone(), cols).unwrap();
        let f = File::create(&seg).unwrap();
        let mut w = ArrowWriter::try_new(f, schema, None).unwrap();
        w.write(&new_batch).unwrap();
        w.close().unwrap();

        let err = replay_journal(&jdir).unwrap_err();
        assert!(
            matches!(err, StorageError::InvalidValue { .. }),
            "tampered payload must be rejected, got {err}"
        );
        assert!(err.to_string().contains("checksum"), "got {err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
