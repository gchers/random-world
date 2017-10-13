pub mod knn;

use ndarray::prelude::*;

pub use self::knn::KNN;

/// A NonconformityScorer can be used to associate a
/// nonconformity score to a new example.
///
/// This trait is parametrized over `T`, the element type.
pub trait NonconformityScorer<T: Sync> {
    /// Scores the `i`-th input vector given the remaining
    /// `0, 1, ..., i-1, i+1, ..., n`.
    ///
    /// # Arguments
    ///
    /// `i` - Input to score.
    /// `inputs` - View on matrix `ArrayView2<2>` containing all examples.
    fn score(&self, i: usize, &ArrayView2<T>) -> f64;
}
