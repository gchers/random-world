use std::f64;
use std::cmp::min;
use rusty_machine::linalg::{Matrix, BaseMatrix};


//// Should use
//// https://athemathmo.github.io/rulinalg/doc/rulinalg/vector/struct.Vector.html#method.metric
fn euclidean_distance(v1: &[f64], v2: &[f64]) -> f64 {
    let dist: f64 = v1.iter()
                      .zip(v2.iter())
                      .map(|(x,y)| (x - y).powi(2))
                      .sum();
    dist.sqrt()
}

pub trait NonConformityScorer<T> {
    /// Compute a k-NN nonconformity score on the i-th input
    /// of inputs given all the rest of inputs.
    fn score(&self, usize, &T) -> f64;
}

pub struct KNN {
    k: usize,
    distance: fn(&[f64], &[f64]) -> f64,
}

impl KNN {
    pub fn new(k: usize) -> KNN {
        KNN {k: k, distance: euclidean_distance}
    }
}

impl NonConformityScorer<Matrix<f64>> for KNN {
    fn score(&self, i: usize, inputs: &Matrix<f64>) -> f64 {
        let input = inputs.get_row(i).expect("Invalid index");

        let distances = inputs.iter_rows()
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
