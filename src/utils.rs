use ndarray::prelude::*;
use std::io::prelude::*;
//use std::fs::File;
use std::fmt::Display;
use std::error::Error;
use std::process;
use csv::{Reader, Writer};

/// Loads a CSV data file.
///
/// The file format should be, for each row:
///     label, x1, x2, ...
/// where x1, x2, ... are features forming a feature vector.
pub fn load_data(fname: String) -> Result<(Array2<f64>, Array1<usize>), Box<Error>> {
    let mut reader = Reader::from_path(fname)?;

    let mut inputs: Vec<f64> = Vec::new();
    let mut targets: Vec<usize> = Vec::new();

    let mut d: Option<usize> = None;

    for result in reader.records() {
        let record = result?;

        inputs.extend(record.iter()
                            .skip(1)  // First one is the label.
                            .map(|x| x.trim()
                                      .parse::<f64>().ok()
                                                     .expect("Failed to parse")));
        targets.push(record[0].parse::<usize>()?);

        if let Some(x) = d {
            if x != record.len() - 1 {
                panic!("File has wrong format");
            }
        } else {
            d = Some(record.len() - 1);
        }
    }

    let inputs_a = if let Some(d) = d {
        let n = inputs.len() / d;
        Array::from_vec(inputs)
              .into_shape((n, d))?
    } else {
        panic!("File has wrong format");
    };

    Ok((inputs_a, Array::from_vec(targets)))
}

pub fn store_predictions<T>(predictions: ArrayView2<T>, fname: String)
        -> Result<(), Box<Error>> where T: Display {
    let mut writer = Writer::from_path(fname)?;

    for x in predictions.outer_iter() {
        writer.write_record(x.iter()
                             .map(|v| format!("{}", v)))?;
    }

    writer.flush();
    Ok(())
}
