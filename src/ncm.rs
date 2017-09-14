use rusty_machine::linalg::{Matrix, BaseMatrix};
use std::cmp::Ordering::Equal;


//// Should use
//// https://athemathmo.github.io/rulinalg/doc/rulinalg/vector/struct.Vector.html#method.metric
pub fn euclidean_distance(v1: &[f64], v2: &[f64]) -> f64 {
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

        //inf = 1000; // XXX
        //let mut smallest: (f64, usize) = (inf, 0);
        //let mut k_smallest = vec![inf; k];
        let mut distances = inputs.iter_rows()
                                  .enumerate()
                                  .filter(|&(j, _)| j != i)
                                  .map(|(_, x)| (self.distance)(x, input))
                                  //.filter(|d| { if d > smallest[0] {
                                  //                  k_smallest[smallest[1]] = d;
                                  //                  // Recompute idx to smallest
                                  //                  smallest[1] = d;
                                  //                  smallest[0] = ?;
                                  //            }
                                  .collect::<Vec<_>>();

        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Equal));

        distances.iter()
             .take(self.k)
             .sum()
    }
}


//trait Klargest {
//    fn k_largest(&self, k: usize);
//}

///// Cheap method to get k largest numbers in iterator (without sorting).
//impl Klargest for Iterator<f64> {
//    fn k_largest(&self, k: usize) -> Iterator<f64> {
//        let mut out: &[f64] = self.take(k);
//
//        for i in (k..v.len()) {
//            for j in (0..out) {
//                if let x = Some(out) {
//                    if x > v[i] {
//                        v[i] = x;
//                        continue;
//                    }
//                } else {
//                    out[j] = Some(v[i]);
//                    continue;
//                }
//            }
//        }
//        out
//    }
