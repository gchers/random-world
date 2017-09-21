use std::f64;
use std::cmp::min;

//// Should use
//// https://athemathmo.github.io/rulinalg/doc/rulinalg/vector/struct.Vector.html#method.metric
fn euclidean_distance(v1: &Vec<f64>, v2: &Vec<f64>) -> f64 {
    let dist: f64 = v1.iter()
                      .zip(v2.iter())
                      .map(|(x,y)| (x - y).powi(2))
                      .sum();
    dist.sqrt()
}

/// T: type of a feature object
pub trait NonConformityScorer<T> {
    /// Compute a k-NN nonconformity score on the i-th input
    /// of inputs given all the rest of inputs.
    fn score(&self, usize, &[T]) -> f64;
}

pub struct KNN<T> {
    k: usize,
    distance: fn(&T, &T) -> f64,
}

impl KNN<Vec<f64>> {
    pub fn new(k: usize) -> KNN<Vec<f64>> {
        KNN {k: k, distance: euclidean_distance}
    }
}

impl<T> NonConformityScorer<T> for KNN<T> {
    fn score(&self, i: usize, inputs: &[T]) -> f64 {
        let input = &inputs[i];

        let distances = inputs.iter()
                              .enumerate()
                              .filter(|&(j, _)| j != i)
                              .map(|(_, x)| (self.distance)(x, input))
                              .collect::<Vec<_>>();

        let k = min(self.k, distances.len());
        let mut k_smallest = vec![f64::INFINITY; k];

        for d in distances {
            for i in 0..k {
                if d < k_smallest[i] {
                    k_smallest[i] = d;
                    break;
                }
            }
        }
        k_smallest.iter()
                  .sum()
    }
}
