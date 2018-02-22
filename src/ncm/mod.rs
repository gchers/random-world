//! Module defining nonconformity measures.
//!
//! A `NonconformityScorer<T>` implements a nonconformity measure `score()`,
//! which determines how "strange" a new input vector looks like with
//! respect to previously observed ones.
pub mod knn;

use ndarray::prelude::*;
use rusty_machine::learning::LearningResult;

pub use self::knn::KNN;

/// A NonconformityScorer can be used to associate a
/// nonconformity score to a new example.
///
/// This trait is parametrized over `T`, the element type.
pub trait NonconformityScorer<T: Sync> {
    /// TODO doc
    fn train(&mut self, inputs: &ArrayView2<T>, targets: &ArrayView1<usize>,
             n_labels: usize) -> LearningResult<()>;
    /// TODO doc
    fn update(&mut self, inputs: &ArrayView2<T>, targets: &ArrayView1<usize>) -> LearningResult<()>;
    /// TODO doc
    fn scores(&self, input: &ArrayView1<T>, targets: usize) -> Vec<f64>;
    fn score_one(&self, i: usize, inputs: &ArrayView2<T>) -> f64;
}
