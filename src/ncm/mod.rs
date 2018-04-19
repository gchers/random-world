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
    /// Trains a `NonconformityScorer`.
    ///
    /// Note: `train()` should be only called once. To update the training
    /// data of the `NonconformityScorer` use `update()`.
    ///
    ///
    /// # Arguments
    ///
    /// * `inputs` - Matrix (Array2<T>) with values of type T of training
    ///              vectors.
    /// * `targets` - Vector (Array1<T>) of labels corresponding to the
    ///               training vectors.
    fn train(&mut self, inputs: &ArrayView2<T>, targets: &ArrayView1<usize>,
             n_labels: usize) -> LearningResult<()>;
    /// Calibrates a `NonconformityScorer` for an ICP.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Matrix (Array2<T>) with values of type T of training
    ///              vectors.
    /// * `targets` - Vector (Array1<T>) of labels corresponding to the
    ///               training vectors.
    fn calibrate(&mut self, inputs: &ArrayView2<T>, targets: &ArrayView1<usize>,
             n_labels: usize) -> LearningResult<()>;
    /// Updates a `NonconformityScorer` with more training data.
    ///
    /// After calling `train()` once, `update()` allows to add
    /// inputs to the scorer's training data, which will be used
    /// for future predictions.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Matrix (Array2<T>) with values of type T of training
    ///              vectors.
    /// * `targets` - Vector (Array1<T>) of labels corresponding to the
    ///               training vectors.
    fn update(&mut self, inputs: &ArrayView2<T>, targets: &ArrayView1<usize>) -> LearningResult<()>;
    /// Computes the nonconformity scores of training inputs and of a new
    /// test example.
    ///
    /// Specifically, nonconformity scores a_i, where (a_1, ..., a_{n-1})
    /// are those corresponding to training examples, and a_n is the
    /// nonconformity score of the new example (x, y), are returned in the
    /// following order:
    ///     (a_n, a_1, a_2, ..., a_{n-1}).
    ///
    /// # Arguments
    ///
    /// * `x` - Test object.
    /// * `y` - (Candidate) label for the test object.
    fn scores(&self, input: &ArrayView1<T>, targets: usize) -> Vec<f64>;
}
