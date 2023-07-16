use polars::prelude::{Schema, Field, DataType};
use serde::Deserialize;

    
#[derive(Deserialize)]
pub struct StrokeRecord {
    id: u32,
    gender: String,
    age: f32,
    hypertension: u32,
    heart_disease: u32,
    ever_married: String,
    work_type: String,
    Residence_type: String,
    avg_glucose_level: f32,
    bmi: Option<f32>,
    smoking_status: String,
    stroke: u32
}


impl StrokeRecord {
    pub fn raw_schema() -> Schema {
        Schema::from_iter(
            vec![
                Field::new("id", DataType::Int32),
                Field::new("gender", DataType::Utf8),
                Field::new("age", DataType::Float64),
                Field::new("hypertension", DataType::Int32),
                Field::new("heart_disease", DataType::Int32),
                Field::new("ever_married", DataType::Utf8),
                Field::new("work_type", DataType::Utf8),
                Field::new("Residence_type", DataType::Utf8),
                Field::new("avg_glucose_level", DataType::Float64),
                Field::new("bmi", DataType::Utf8),
                Field::new("smoking_status", DataType::Utf8),
                Field::new("stroke", DataType::Int32),
            ])
    }

}


// pub fn convert_to_batch(table: &DeltaTable, records: &Vec<StrokeRecord>) -> RecordBatch {
//     let metadata = table
//         .get_metadata()
//         .expect("Failed to get metadata for the table");
//     let arrow_schema = <deltalake::arrow::datatypes::Schema as TryFrom<&Schema>>::try_from(
//         &metadata.schema.clone(),
//     )
//     .expect("Failed to convert to arrow schema");
//     let arrow_schema_ref = Arc::new(arrow_schema);

//     let mut ts = vec![];
//     let mut temp = vec![];
//     let mut lat = vec![];
//     let mut long = vec![];

//     for record in records {
//         ts.push(record.id);
//         temp.push(record.id);
//         lat.push(record.bmi);
//         long.push(record.bmi);
//     }

//     let arrow_array: Vec<Arc<dyn Array>> = vec![
//         Arc::new(Int32Array::from(ts)),
//         Arc::new(Int32Array::from(temp)),
//         Arc::new(Float64Array::from(lat)),
//         Arc::new(Float64Array::from(long)),
//     ];

//     RecordBatch::try_new(arrow_schema_ref, arrow_array).expect("Failed to create RecordBatch")
// }