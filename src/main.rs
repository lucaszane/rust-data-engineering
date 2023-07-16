extern crate serde;

mod records;
use polars::export::num::NumCast;
use records::StrokeRecord;

use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::linear_regression::LinearRegression;
use smartcore::metrics::mean_squared_error;
use smartcore::model_selection::train_test_split;
use std::collections::HashMap;
use std::ops::{DivAssign, MulAssign, SubAssign};
// use arrow::error::Result as ArrowResult;
use polars::frame::DataFrame;
use polars::prelude::PolarsResult;
use polars::prelude::*;
use polars_io::parquet::ParquetWriter;
use std::fs::File;
use std::path::Path;
use std::time::Instant;
use std::vec::Vec;
// use the TryFrom to convert exactly the nubmer
use smartcore::linalg::basic::arrays::{MutArray};
use std::convert::TryFrom;
use ndarray::{Array2, Array1};

// use ndarray::Array2;
// use rulinalg::matrix::Matrix;

use num::Num;
use polars::prelude::SerReader;

fn monitor_memory() -> u64 {
    // Implement memory monitoring logic
    // This might involve reading from /proc/self/status on Linux
    // or using a crate like `sys-info` if cross-platform support is needed
    return 1;
}

pub async fn read_parquet<P: AsRef<Path>>(path: P) -> PolarsResult<DataFrame> {
    /* Example function to create a dataframe from an input csv file*/
    let file = File::open(path).expect("Cannot open file.");

    ParquetReader::new(file).finish()
}

pub async fn read_csv<P: AsRef<Path>>(path: P) -> PolarsResult<DataFrame> {
    /* Example function to create a dataframe from an input csv file*/
    let file = File::open(path).expect("Cannot open file.");

    CsvReader::new(file)
        .has_header(true)
        .with_dtypes(Option::from(Arc::new(StrokeRecord::raw_schema())))
        .finish()
}

pub async fn write_csv(
    file_name: &str,
    df: &mut DataFrame,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(file_name).expect("could not create file");

    CsvWriter::new(&mut file).finish(df)?;

    Ok(())
}

pub async fn write_parquet(
    file_name: &str,
    df: &mut DataFrame,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(file_name).expect("could not create file");

    ParquetWriter::new(&mut file).finish(df)?;

    Ok(())
}

static RAW_PATH: &str = "data/output/raw/";
static SILVER_PATH: &str = "data/output/silver/";
static GOLD_PATH: &str = "data/output/gold/";
static STROKE_FILE_NAME: &str = "stroke.parquet";

async fn process_raw() -> Result<(), Box<dyn std::error::Error>> {
    let file_path = "data/healthcare-dataset-stroke-data.csv";
    let mut df = read_csv(&file_path).await?;
    write_parquet(
        format!("{}{}", RAW_PATH, STROKE_FILE_NAME).as_str(),
        &mut df,
    )
    .await?;

    Ok(())
}

async fn process_silver() -> Result<(), Box<dyn std::error::Error>> {
    let mut df = read_parquet(format!("{}{}", RAW_PATH, STROKE_FILE_NAME).as_str()).await?;

    // let series = df.column("bmi")?.utf8()?.apply(|e| e.replace("N/A", "").into());
    // df.with_column(series)?;
    // let replaced_df = replace_df_text(&mut df, "bmi", "N/A", "").await?;

    // Casting string to float
    df.with_column(df.column("bmi")?.cast(&DataType::Float32)?)?;

    // Replacing null by mean
    df = df.fill_null(FillNullStrategy::Mean)?;

    write_parquet(
        format!("{}{}", SILVER_PATH, STROKE_FILE_NAME).as_str(),
        &mut df,
    )
    .await?;

    // println!("{}", df.head(Some(5)));

    Ok(())
}

async fn process_gold() -> Result<(), Box<dyn std::error::Error>> {
    let mut df = read_parquet(format!("{}{}", SILVER_PATH, STROKE_FILE_NAME).as_str())
        .await?
        .lazy();

    df = df.with_column(
        col("gender")
            .alias("gender")
            .apply(encode_lazy, GetOutput::from_type(DataType::UInt32)),
    );
    df = df.with_column(
        col("ever_married")
            .alias("ever_married")
            .apply(encode_lazy, GetOutput::from_type(DataType::UInt32)),
    );
    df = df.with_column(
        col("Residence_type")
            .alias("Residence_type")
            .apply(encode_lazy, GetOutput::from_type(DataType::UInt32)),
    );
    df = df.with_column(
        col("smoking_status")
            .alias("smoking_status")
            .apply(encode_lazy, GetOutput::from_type(DataType::UInt32)),
    );
    df = df.with_column(
        col("work_type")
            .alias("work_type")
            .apply(encode_lazy, GetOutput::from_type(DataType::UInt32)),
    );

    df = df.with_column(col("age").alias("age").apply(
        |s| min_max_scale::<f32>(s),
        GetOutput::from_type(DataType::UInt32),
    ));
    df = df.with_column(col("avg_glucose_level").alias("avg_glucose_level").apply(
        |s| min_max_scale::<f32>(s),
        GetOutput::from_type(DataType::UInt32),
    ));
    df = df.with_column(col("bmi").alias("bmi").apply(
        |s| min_max_scale::<f32>(s),
        GetOutput::from_type(DataType::UInt32),
    ));

    // encode(&mut df, "gender")?;
    // encode(&mut df, "ever_married")?;
    // encode(&mut df, "Residence_type")?;
    // encode(&mut df, "smoking_status")?;
    // encode(&mut df, "work_type")?;

    let mut final_df = df.collect()?;
    println!("{}", final_df.head(Some(5)));

    write_csv(
        format!("{}{}", GOLD_PATH, "debug.csv").as_str(),
        &mut final_df,
    )
    .await?;
    write_parquet(
        format!("{}{}", GOLD_PATH, STROKE_FILE_NAME).as_str(),
        &mut final_df,
    )
    .await?;
    Ok(())
}

pub fn feature_and_target(in_df: &DataFrame) -> PolarsResult<(DataFrame, DataFrame)> {
    /* Read a dataframe, select the columns we need for feature training and target and return
    the two new dataframes*/
    let features = in_df.select(vec![
        "id",
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "Residence_type",
        "avg_glucose_level",
        "bmi",
        "smoking_status",
    ])?;

    let target = in_df.select(["stroke"])?;

    Ok((features, target))
}

fn to_ndarray(in_df: &DataFrame) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    // Initialize an empty Vec to hold the 1D arrays
    let mut arrays = Vec::new();

    for name in in_df.get_column_names() {
        let series = in_df.column(name)?;
        // Convert the Series to an ndarray Array1 and push it to the Vec
        let array: Array1<f32> = Array1::from(series.f32()?.cont_slice()?.to_owned());
        arrays.push(array);

    }

    // Stack the arrays along the second axis
    let array2 = ndarray::stack(ndarray::Axis(1), arrays)?;

    Ok(array2)
}

pub fn convert_features_to_matrix(
    in_df: &DataFrame,
) -> Result<DenseMatrix<f64>, Box<dyn std::error::Error>> {
    /* function to convert feature dataframe to a DenseMatrix, readable by smartcore*/

    let nrows = in_df.height();
    let ncols = in_df.width();
    // convert to array
    let features_res = to_ndarray(&in_df)?;
    // create a zero matrix and populate with features
    let mut xmatrix: DenseMatrix<f64> = DenseMatrix::zeros(nrows, ncols);
    // populate the matrix
    // initialize row and column counters
    let mut col: u32 = 0;
    let mut row: u32 = 0;

    for val in features_res.iter() {
        // Debug
        //println!("{},{}", usize::try_from(row).unwrap(), usize::try_from(col).unwrap());
        // define the row and col in the final matrix as usize
        let m_row = usize::try_from(row).unwrap();
        let m_col = usize::try_from(col).unwrap();
        // NB we are dereferencing the borrow with *val otherwise we would have a &val type, which is
        // not what set wants
        xmatrix.set((m_row, m_col), *val);
        // check what we have to update
        if m_col == ncols - 1 {
            row += 1;
            col = 0;
        } else {
            col += 1;
        }
    }

    // Ok so we can return DenseMatrix, otherwise we'll have std::result::Result<Densematrix, PolarsError>
    Ok(xmatrix)
}

async fn train_dataset() -> Result<(), Box<dyn std::error::Error>> {
    let mut df = read_parquet(format!("{}{}", GOLD_PATH, STROKE_FILE_NAME).as_str()).await?;

    let (features, target) = feature_and_target(&df)?;
    let xmatrix = convert_features_to_matrix(&features)?;
    let target_array = target.to_ndarray::<Float64Type>()?;
    // create a vec type and populate with y values
    let mut y: Vec<f64> = Vec::new();
    for val in target_array.iter() {
        y.push(*val);
    }

    // train split
    let (x_train, x_test, y_train, y_test) = train_test_split(&xmatrix, &y, 0.3, true, Some(2));

    // model
    let linear_regression = LinearRegression::fit(&x_train, &y_train, Default::default()).unwrap();
    // predictions
    let preds = linear_regression.predict(&x_test).unwrap();
    // metrics
    let mse = mean_squared_error(&y_test, &preds);
    println!("MSE: {:?}", mse);

    Ok(())
}

fn encode_lazy(column: Series) -> Result<Option<Series>, PolarsError> {
    let utf8 = column.utf8()?;
    let mut encoded = Vec::new();
    let mut map = HashMap::new();
    for val in utf8 {
        let encoded_val = match map.get(val.unwrap_or_default()) {
            Some(value) => *value,
            None => {
                let new_val = (map.len() + 1) as u32;
                map.insert(val.unwrap_or_default().to_string(), new_val);
                new_val
            }
        };
        encoded.push(Some(encoded_val));
    }
    Ok(Option::from(Series::new(column.name(), encoded)))
}

fn min_max_scale<T: Num + Copy + MulAssign + SubAssign + DivAssign + PartialOrd>(
    column: Series,
) -> Result<Option<Series>, PolarsError>
where
    T: NumCast,
{
    let min = column.min::<T>().unwrap();
    let max = column.max::<T>().unwrap();
    let range = max - min;
    Ok(Option::from((column - min) / range))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    let start_memory = monitor_memory();

    process_raw().await?;
    process_silver().await?;
    process_gold().await?;
    train_dataset().await?;

    let end_memory = monitor_memory();
    let duration = start_time.elapsed();

    println!("Time elapsed in expensive_function is: {:?}", duration);
    println!("Memory used: {:?}", end_memory - start_memory);

    Ok(())
}
