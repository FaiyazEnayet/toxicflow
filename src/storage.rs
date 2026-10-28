//! Parquet read/write for recorded sessions, via arrow record batches.

use std::error::Error;
use std::fs::File;
use std::sync::Arc;

use arrow::array::{Array, BooleanArray, Float64Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;

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
            Arc::new(Int64Array::from_iter_values(trades.iter().map(|t| t.time_ms))),
            Arc::new(Float64Array::from_iter_values(trades.iter().map(|t| t.price()))),
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
            Arc::new(Int64Array::from_iter_values(quotes.iter().map(|q| q.time_ms))),
            Arc::new(Float64Array::from_iter_values(quotes.iter().map(|q| q.bid_px))),
            Arc::new(Float64Array::from_iter_values(quotes.iter().map(|q| q.bid_sz))),
            Arc::new(Float64Array::from_iter_values(quotes.iter().map(|q| q.ask_px))),
            Arc::new(Float64Array::from_iter_values(quotes.iter().map(|q| q.ask_sz))),
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
}
