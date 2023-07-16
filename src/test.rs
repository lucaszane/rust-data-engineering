use std::fs;
use std::num::ParseFloatError;
use std::path::{Path, PathBuf};
use std::str::ParseBoolError;
 
use clap::{ArgEnum, Parser, Subcommand};
use datafusion::arrow::datatypes::DataType;
use datafusion::error::DataFusionError;
use datafusion::prelude::*;
use env_logger::{Builder, Env};
use log::{debug, info, LevelFilter, trace};
use thiserror::Error;

#[tokio::main]
async fn main() -> Result<(), MDataAppError> {
    let cli = MDataAppArgs::parse();
 
    let log_level = match cli.verbose {
        1 => LevelFilter::Debug,
        2 => LevelFilter::Trace,
        _ => LevelFilter::Info,
    };
 
    let env = Env::new().filter("MLOG");
    let _builder = Builder::new().filter(Some("mdata_app"), log_level).parse_env(env).init();
 
    debug!("Arguments {:#?}", cli);
 
    mdata_app(cli).await?;
    Ok(())
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
#[clap(propagate_version = true)]
pub struct MDataAppArgs {
    #[clap(short, long, parse(from_os_str), help = "Input path")]
    input: PathBuf,
    #[clap(short, long, parse(from_os_str), help = "Output path")]
    output: PathBuf,
    #[clap(short, long, parse(from_occurrences),
    help = "Verbose level")]
    verbose: usize,
    #[clap(short, long, arg_enum, default_value_t = WriteFormat::Undefined,
    help = "Output format")]
    format: WriteFormat,
    #[clap(short, long, default_value_t = 0,
    help = "Limit the result to the first <limit> rows")]
    limit: usize,
    #[clap(short, long, parse(from_flag),
    help = "Display the inferred schema")]
    schema: bool,
}
 
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, ArgEnum)]
enum WriteFormat {
    Undefined = 0,
    Csv = 1,
    Parquet = 2,
}
 
#[derive(Error, Debug)]
enum MDataAppError {
    #[error("invalid path encoding {path:?}")]
    PathEncoding { path: PathBuf },
    #[error("invalid input format {path:?}")]
    InputFormat { path: PathBuf },
    #[error("invalid output format {format:?}")]
    OutputFormat { format: WriteFormat },
    #[error("invalid filter value {error_message:?}")]
    FilterValue { error_message: String },
    #[error("transparent")]
    DataFusionOp(#[from] DataFusionError),
}
 
impl From<ParseBoolError> for MDataAppError {
    fn from(e: ParseBoolError) -> Self {
        MDataAppError::FilterValue { error_message: e.to_string() }
    }
}
 
impl From<ParseFloatError> for MDataAppError {
    fn from(e: ParseFloatError) -> Self {
        MDataAppError::FilterValue { error_message: e.to_string() }
    }
}

// impl From<DataFusionError> for MDataAppError {
//     fn from(e: DataFusionError) -> MDataAppError {
//         MDataAppError::DataFusionOp { error: e }
//     }
// }

async fn mdata_app(opts: MDataAppArgs) -> Result<(), MDataAppError> {
    let ctx = SessionContext::new();
    let input_path = get_os_path(&opts.input)?;
    let output_path = get_os_path(&opts.output)?;
 
    // the inferred format is returned
    let inferred_input_format =
        match infer_file_type(&opts.input, true) {
            WriteFormat::Csv => {
                ctx.register_csv("input", input_path, CsvReadOptions::new())
                    .await?;
                WriteFormat::Csv
            }
            WriteFormat::Parquet => {
                ctx.register_parquet("input", input_path,
                                     ParquetReadOptions::default()).await?;
                WriteFormat::Parquet
            }
            WriteFormat::Undefined => {
                return Err(MDataAppError::InputFormat { path: opts.input })
                ;
            }
        };
        let df_input = ctx.table("input")?;
 
        // get the table schema
        let schema = df_input.schema();
         
        // print-it
        if opts.schema {
            info!("# Schema\n{:#?}", schema)
        }
         
        // process user filters
        let df_flt =
            if let Some(Filters::Eq { column: column_name, value }) = &opts.filter {
                // get the data type of the filtered column
                let filter_type = schema.field_with_name(None,
                                                 column_name)?.data_type();
                // parse the filter value based on column type
                let filter_value = match filter_type {
                    DataType::Boolean => lit(value.to_string().parse::<bool>()?),
                    DataType::Utf8 => lit(value.to_string()),
                    x if DataType::is_numeric(&x) =>
                        lit(value.to_string().parse::<f64>()?),
                    _ => return Err(MDataAppError::FilterValue {
                        error_message: "invalid filter value".to_string()
                    })
                };
         
                // filter the current dataframe
                df_input.filter(col(column_name).eq(filter_value))?
            } else {
                df_input
            };
         
        // if limit is active, only N first rows are written
        let df_lmt = if opts.limit > 0 {
            df_flt.limit(opts.limit)?
        } else {
            df_flt
        };
         
        // if the output format is undefined, default to input format
        let output_format = match opts.format {
            WriteFormat::Undefined => {
                if inferred_input_format == WriteFormat::Undefined {
                    WriteFormat::Undefined
                } else {
                    inferred_input_format
                }
            }
            _ => opts.format.clone()
        };
         
        match output_format {
            WriteFormat::Csv => df_lmt.write_csv(output_path).await?,
            WriteFormat::Parquet => df_lmt.write_parquet(output_path, None).await?,
            WriteFormat::Undefined => {
                return Err(
                    MDataAppError::OutputFormat {
                        format: opts.format.clone(),
                    },
                );
            }
        }
        
 
    Ok(())
}
