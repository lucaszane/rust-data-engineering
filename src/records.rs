use polars::prelude::{Schema, Field, DataType};


pub struct StrokeRecord {
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
