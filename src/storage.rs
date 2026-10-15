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

fn col_i64(batch: &RecordBatch, name: &str) -> Result<Vec<i64>, Box<dyn Error>> {
    let a = batch
        .column_by_name(name)
        .ok_or_else(|| format!("parquet batch is missing column '{name}'"))?
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| format!("parquet column '{name}' is not int64"))?;
    Ok((0..a.len()).map(|i| a.value(i)).collect())
}

fn col_f64(batch: &RecordBatch, name: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    let a = batch
        .column_by_name(name)
        .ok_or_else(|| format!("parquet batch is missing column '{name}'"))?
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| format!("parquet column '{name}' is not float64"))?;
    Ok((0..a.len()).map(|i| a.value(i)).collect())
}

fn col_bool(batch: &RecordBatch, name: &str) -> Result<Vec<bool>, Box<dyn Error>> {
    let a = batch
        .column_by_name(name)
        .ok_or_else(|| format!("parquet batch is missing column '{name}'"))?
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| format!("parquet column '{name}' is not boolean"))?;
    Ok((0..a.len()).map(|i| a.value(i)).collect())
}

pub fn read_trades(path: &str) -> Result<Vec<Trade>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    let mut out = Vec::new();
    for batch in reader {
        let b = batch?;
        let time = col_i64(&b, "time")?;
        let price = col_f64(&b, "price")?;
        let qty = col_f64(&b, "qty")?;
        let mbm = col_bool(&b, "is_buyer_maker")?;
        for i in 0..b.num_rows() {
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

pub fn read_quotes(path: &str) -> Result<Vec<Quote>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    let mut out = Vec::new();
    for batch in reader {
        let b = batch?;
        let time = col_i64(&b, "time")?;
        let bp = col_f64(&b, "bid_px")?;
        let bq = col_f64(&b, "bid_sz")?;
        let ap = col_f64(&b, "ask_px")?;
        let aq = col_f64(&b, "ask_sz")?;
        for i in 0..b.num_rows() {
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
