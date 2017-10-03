use std::f64;
use std::cmp::min;
use ordered_float::OrderedFloat;
use lazysort::Sorted;
use ndarray::prelude::*;

//cached!{ EUCLIDEAN: UnboundCache >>
fn cached_euclidean_distance(v: (Vec<OrderedFloat<f64>>, Vec<OrderedFloat<f64>>))
        -> OrderedFloat<f64> { //= {
    let (ref v1, ref v2) = v;
    let dist: f64 = v1.iter()
                      .zip(v2.iter())
                      .map(|(x,y)| (x.into_inner() - y.into_inner()).powi(2))
                      .sum();
    OrderedFloat::from(dist.sqrt())
//}}
}

// TODO: Should use
// https://athemathmo.github.io/rulinalg/doc/rulinalg/vector/struct.Vector.html#method.metric
fn euclidean_distance(v1: &ArrayView1<f64>, v2: &ArrayView1<f64>) -> f64 {

    /* Convert into Vec<OrderedFloat<f64>> before calling the
     * cached function.
     */
    let v1o = v1.iter()
                .map(|x| OrderedFloat::from(x.clone()))
                .collect::<Vec<_>>();

    let v2o = v2.iter()
                .map(|x| OrderedFloat::from(x.clone()))
                .collect::<Vec<_>>();

    cached_euclidean_distance((v1o, v2o)).into_inner()
}

/// T: type of a feature object
pub trait NonConformityScorer<T: Sync> {
    /// Compute a k-NN nonconformity score on the i-th input
    /// of inputs given all the rest of inputs.
    fn score(&self, i: usize, &Array2<T>) -> f64;
}

pub struct KNN<T: Sync> {
    k: usize,
    distance: fn(&ArrayView1<T>, &ArrayView1<T>) -> f64,
}

impl KNN<f64> {
    pub fn new(k: usize) -> KNN<f64> {
        KNN {k: k, distance: euclidean_distance}
    }
}

impl<T: Sync> NonConformityScorer<T> for KNN<T> {
    fn score(&self, i: usize, inputs: &Array2<T>) -> f64 {

        let ii = i as isize;
        
        let input = inputs.slice(s![ii..ii+1, ..])
                          .into_shape((inputs.cols()))
                          .expect("Unexpected error in extracting row");
        let k = min(self.k, inputs.len()-1);

        inputs.outer_iter()
              .enumerate()
              .filter(|&(j, _)| j != i)
              /* Compute distances and convert to OrderedFloat for sorting */
              .map(|(_, x)| OrderedFloat((self.distance)(&x, &input)))
              /* Lazy sort: faster because generally k << inputs.len() */
              .sorted()
              .take(k)
              /* Convert back to f64 */
              .map(|d| d.into_inner())
              .sum()
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
        let expected_scores = array![2., 2.414213562373095, 2.414213562373095,
                                   4.47213595499958];

        for i in 0..4 {
            println!("{} {} {}", i, ncm.score(i, &train_inputs), expected_scores[i]);
            assert!(ncm.score(i, &train_inputs) == expected_scores[i]);
        }
    }
}
