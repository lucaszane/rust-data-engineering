extern crate serde;

mod records;
use polars::export::num::NumCast;
use records::StrokeRecord;

use polars::frame::DataFrame;
use polars::prelude::PolarsResult;
use polars::prelude::*;
use polars_io::parquet::ParquetWriter;
use smartcore::linalg::basic::arrays::MutArray;
use smartcore::linalg::basic::matrix::DenseMatrix;

use smartcore::metrics::accuracy;
use smartcore::model_selection::{train_test_split, KFold, cross_validate};

use smartcore::tree::decision_tree_classifier::{DecisionTreeClassifier, DecisionTreeClassifierParameters};
use std::collections::HashMap;
use std::convert::TryFrom;
use std::fs::File;
use std::ops::{DivAssign, MulAssign, SubAssign};
use std::path::Path;
use std::time::Instant;
use std::vec::Vec;
use sysinfo::{System, SystemExt};

use num::Num;
use polars::prelude::SerReader;


fn monitor_memory() -> u64 {
    let sys = System::new_all();
    // Implement memory monitoring logic
    // This might involve reading from /proc/self/status on Linux
    // or using a crate like `sys-info` if cross-platform support is needed
    return sys.used_memory();
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

    
    df = df.with_column(col("gender").cast(DataType::Float64));
    df = df.with_column(col("hypertension").cast(DataType::Float64));
    df = df.with_column(col("heart_disease").cast(DataType::Float64));
    df = df.with_column(col("ever_married").cast(DataType::Float64));
    df = df.with_column(col("work_type").cast(DataType::Float64));
    df = df.with_column(col("Residence_type").cast(DataType::Float64));
    df = df.with_column(col("smoking_status").cast(DataType::Float64));
    df = df.with_column(col("bmi").cast(DataType::Float64));
    df = df.with_column(col("stroke").cast(DataType::UInt32));

    df = df.with_column(col("age").alias("age").apply(
        |s| min_max_scale::<f64>(s),
        GetOutput::from_type(DataType::UInt32),
    ));
    df = df.with_column(col("avg_glucose_level").alias("avg_glucose_level").apply(
        |s| min_max_scale::<f64>(s),
        GetOutput::from_type(DataType::UInt32),
    ));
    df = df.with_column(col("bmi").alias("bmi").apply(
        |s| min_max_scale::<f64>(s),
        GetOutput::from_type(DataType::UInt32),
    ));
    df = df.with_column(col("gender").alias("gender").apply(
        |s| min_max_scale::<f64>(s),
        GetOutput::from_type(DataType::UInt32),
    ));
    df = df.with_column(col("ever_married").alias("ever_married").apply(
        |s| min_max_scale::<f64>(s),
        GetOutput::from_type(DataType::UInt32),
    ));
    df = df.with_column(col("Residence_type").alias("Residence_type").apply(
        |s| min_max_scale::<f64>(s),
        GetOutput::from_type(DataType::UInt32),
    ));
    df = df.with_column(col("smoking_status").alias("smoking_status").apply(
        |s| min_max_scale::<f64>(s),
        GetOutput::from_type(DataType::UInt32),
    ));
    df = df.with_column(col("work_type").alias("work_type").apply(
        |s| min_max_scale::<f64>(s),
        GetOutput::from_type(DataType::UInt32),
    ));



    // encode(&mut df, "gender")?;
    // encode(&mut df, "ever_married")?;
    // encode(&mut df, "Residence_type")?;
    // encode(&mut df, "smoking_status")?;
    // encode(&mut df, "work_type")?;

    let mut final_df = df.collect()?;

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

pub fn convert_features_to_matrix(
    in_df: &DataFrame,
) -> Result<DenseMatrix<f64>, Box<dyn std::error::Error>> {
    /* function to convert feature dataframe to a DenseMatrix, readable by smartcore*/

    let mut xs: Vec<f64> = Vec::new();
    let nrows = in_df.height();
    let ncols = in_df.width();
    // in_df.drop_in_place("id");

    let rows: Vec<Series> = in_df.drop("id")?.iter().map(|series| series.clone()).collect();

    // Iterate over the rows
    for row in rows {
        let inputs: Vec<f64> = row
            .f64()?
            .to_vec()
            .into_iter()
            .filter_map(|x| x)
            .collect();
        xs.extend_from_slice(&inputs);
    }
    
    let mut xmatrix: DenseMatrix<f64> =
        DenseMatrix::new(nrows, ncols, vec![0.0; nrows * ncols], true);
    // populate the matrix
    // initialize row and column counters
    let mut col: u32 = 0;
    let mut row: u32 = 0;

    for val in xs.iter() {
        // Debug
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
    // println!("{}", xmatrix);

    // Ok so we can return DenseMatrix, otherwise we'll have std::result::Result<Densematrix, PolarsError>
    Ok(xmatrix)
}


pub fn convert_features_to_matrix2(
    in_df: &DataFrame,
) -> Result<DenseMatrix<f64>, Box<dyn std::error::Error>> {
    /* function to convert feature dataframe to a DenseMatrix, readable by smartcore*/

    let n = [concat_list([
    col("gender"),
    col("age"),
    col("hypertension"),
    col("heart_disease"),
    col("ever_married"),
    col("work_type"),
    col("Residence_type"),
    col("avg_glucose_level"),
    col("bmi"),
    col("smoking_status")
    ])?.alias("data")];
    let df = in_df.clone().lazy().select(n).collect()?;
    
    // iterate through the column and convert each number in the nested list into an f64
    let data = df
        .column("data")?
        .list()?
        .into_no_null_iter()
        .map(|n| {
            n.f64().unwrap()
            .into_no_null_iter()
            .map(|n| n as f64)
            .collect::<Vec<f64>>()
        })
        .collect::<Vec<Vec<f64>>>();
    
    let matrix = DenseMatrix::from_2d_vec(&data);

    // println!("{}", matrix);

    Ok(matrix)
}



async fn train_dataset() -> Result<(), Box<dyn std::error::Error>> {
    let df = read_parquet(format!("{}{}", GOLD_PATH, STROKE_FILE_NAME).as_str()).await?;

    let (features, target) = feature_and_target(&df)?;
    
    let xmatrix = convert_features_to_matrix(&features)?;
    let target_array: Vec<u32> = target["stroke"].u32()?.into_no_null_iter().collect();
    // create a vec type and populate with y values
    let mut y: Vec<u32> = Vec::new();
    for val in target_array.iter() {
        y.push(*val);
    }

    //train split
    let (x_train, x_test, y_train, y_test) = train_test_split(&xmatrix, &y, 0.3, true, Some(2));

    // // model
    // let linear_regression = LinearRegression::fit(&x_train, &y_train, Default::default()).unwrap();
    // // predictions
    // let preds = linear_regression.predict(&x_test).unwrap();

    let classifier = DecisionTreeClassifier::fit(
        &x_train,
        &y_train,        
        DecisionTreeClassifierParameters::default(),
    )?;

    let y_predict = classifier.predict(&x_test)?;

    // println!("{:?}", y_predict);
    // println!("{:?}", y_test);

    let mut equal = 0;
    let mut different = 0;
    let mut tp = 0;
    let mut fp = 0;
    let mut tn = 0;
    let mut fneg = 0;
    for (i, val) in y_test.iter().enumerate() {
        if *val == y_predict[i] {
            equal = equal + 1;
            if *val == 1 {
                tp = tp + 1;
            } else {
                tn = tn + 1;
            }
        } else {
            different = different + 1;
            if *val == 0 {
                fp = fp + 1;
            } else {
                fneg = fneg + 1;
            }
        }

    }

    println!("equals: {}", equal);
    println!("different: {}", different);
    println!("accuracy: {}", (equal) as f32 / (equal + different)  as f32);
    println!("tp: {}", tp);
    println!("fp: {}", fp);
    println!("tn: {}", tn);
    println!("fneg: {}", fneg);


  
    let cv = KFold::default().with_n_splits(3);
    
    let results = cross_validate(
        classifier,   //estimator
        &xmatrix, &y,                 //data
        Default::default(),     //hyperparameters
        &cv,                     //cross validation split
        &accuracy).unwrap();    //metric
    
    println!("Training accuracy: {}, test accuracy: {}",
        results.mean_train_score(), results.mean_test_score());
    


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
    let start_memory = monitor_memory();
    let start_time = Instant::now();

    process_raw().await?;

    println!("process_raw time is: {:?}", start_time.elapsed());
    let process_silver_start = Instant::now();

    process_silver().await?;

    println!("process_silver_start time is: {:?}", process_silver_start.elapsed());
    let process_gold_start = Instant::now();

    process_gold().await?;

    println!("process_gold_start time is: {:?}", process_gold_start.elapsed());
    let train_dataset_start = Instant::now();

    train_dataset().await?;

    println!("train_dataset_start time is: {:?}", train_dataset_start.elapsed());

    let duration = start_time.elapsed();
    println!("Time elapsed in all functions is: {:?}", duration);
    
    let end_memory = monitor_memory();
    let base: i32 = 1024;
    println!("Memory used: {:?} MB", ((end_memory - start_memory) as f32 / (base.pow(2) as f32)));

    Ok(())
}
