use std::f64;
use std::cmp::min;
use ndarray::prelude::*;
use std::collections::BinaryHeap;
use std::iter::FromIterator;
use ordered_float::OrderedFloat;

use ncm::NonconformityScorer;


/// Returns the Euclidean distance between two vectors of f64 values.
fn euclidean_distance(v1: &ArrayView1<f64>, v2: &ArrayView1<f64>) -> f64 {
    v1.iter()
      .zip(v2.iter())
      .map(|(x,y)| (x - y).powi(2))
      .sum::<f64>()
      .sqrt()
}

/// A k-NN nonconformity measure.
///
/// The score is defined for some distance metric and number of
/// neighbors.
pub struct KNN<T: Sync> {
    k: usize,
    distance: fn(&ArrayView1<T>, &ArrayView1<T>) -> f64,
}

impl KNN<f64> {
    /// Constructs a k-NN nonconformity measure.
    ///
    /// # Arguments
    ///
    /// `k` - Number of nearest neighbors.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_world::ncm::*;
    ///
    /// let k = 2;
    /// let ncm = KNN::new(k);
    /// ```
    pub fn new(k: usize) -> KNN<f64> {
        KNN {k: k, distance: euclidean_distance}
    }
}

impl<T: Sync> NonconformityScorer<T> for KNN<T> {
    /// Scores the `i`-th input vector given the remaining
    /// ones with the k-NN nonconformity measure.
    ///
    /// # Arguments
    ///
    /// `i` - Input to score.
    /// `inputs` - View on matrix `ArrayView2<2>` containing all examples.
    ///
    /// # Examples
    ///
    /// ```
    /// #[macro_use(array)]
    /// extern crate ndarray;
    /// extern crate random_world;
    ///
    /// # fn main() {
    /// use random_world::ncm::*;
    ///
    /// let ncm = KNN::new(2);
    ///
    /// let train_inputs = array![[0., 0.],
    ///                           [1., 0.],
    ///                           [0., 1.],
    ///                           [2., 2.]];
    ///
    /// assert!(ncm.score(0, &train_inputs.view()) == 2.);
    /// # }
    /// ```
    fn score(&self, i: usize, inputs: &ArrayView2<T>) -> f64 {
        let k = min(self.k, inputs.len()-1);

        let ii = i as isize;
        let input = inputs.slice(s![ii..ii+1, ..])
                          .into_shape((inputs.cols()))
                          .expect("Unexpected error in extracting row");

        let mut heap = BinaryHeap::from_iter(inputs.outer_iter()
                                                   .enumerate()
                                                   .filter(|&(j, _)| j != i)
                                                   // Compute distances.
                                                   .map(|(_, x)|
                                                        (self.distance)(&x, &input))
                                                   // Need Ord floats to sort.
                                                   // NOTE: we take the negative
                                                   // value because we're using
                                                   // a max heap.
                                                   .map(|d| OrderedFloat(-d)));
        let mut sum = 0.;

        for _ in 0..k {
            sum -= heap.pop()
                       .expect("Unexpected error in finding k-smallest")
                       .0;
        }

        sum
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knn() {
        let ncm = KNN::new(2);

        let train_inputs = array![[0., 0.],
                                  [1., 0.],
                                  [0., 1.],
                                  [2., 2.]];
        let expected_scores = vec![2., 2.414213562373095, 2.414213562373095,
                                   4.47213595499958];

        let scores = (0..4).into_iter()
                           .map(|i| ncm.score(i, &train_inputs.view()))
                           .collect::<Vec<_>>();

        assert!(scores == expected_scores);
    }
}
