//! Module defining Conformal Predictors.
//!
//! A `ConfidencePredictor<T>` implements all methods to provide a confidence
//! prediction for new input vectors.
//! Examples of confidence predictors are inductive and transductive
//! Conformal Predictors.
pub mod cp;

use ndarray::prelude::*;
use rusty_machine::learning::LearningResult;

pub use self::cp::CP;


/// A Confidence Predictor (either transductive or inductive)
///
/// This trait is parametrized over `T`, the element type.
/// It provides all the methods for making a confidence prediction.
pub trait ConfidencePredictor<T> {
    fn train(&mut self, inputs: &Array2<T>, targets: &Array1<usize>) -> LearningResult<()>;
    fn predict(&mut self, inputs: &Array2<T>) -> LearningResult<Array2<bool>>;
    fn predict_confidence(&mut self, inputs: &Array2<T>) -> LearningResult<Array2<f64>>;
    fn set_epsilon(&mut self, epsilon: f64);
}
