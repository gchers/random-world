//! k-NN nonconformity measure.
use std::f64;
use std::cmp::min;
use lazysort::SortedPartial;
use ndarray::prelude::*;
use rusty_machine::learning::LearningResult;

use ncm::NonconformityScorer;


/// Returns the Euclidean distance between two vectors of f64 values.
fn euclidean_distance(v1: &ArrayView1<f64>, v2: &ArrayView1<f64>) -> f64 {
    v1.iter()
      .zip(v2.iter())
      .map(|(x,y)| (x - y).powi(2))
      .sum::<f64>()
      .sqrt()
}

/// Splits inputs according to their labels.
///
/// Returns as output a `train_inputs: Vec<Array2<T>>`, such that for each
/// unique label `y`, `train_inputs[y]` contains a matrix with the inputs with
/// label `y`.
fn split_inputs<T>(inputs: &ArrayView2<T>, targets: &ArrayView1<usize>,
         n_labels: usize) -> Vec<Array2<T>> where T: Clone + Sync + Copy {
    // Split examples w.r.t. their labels. For each unique label y,
    // train_inputs[y] will contain a matrix with the inputs with
    // label y.
    // We first put them into a vector, and then will convert them
    // into array. This should guarantee memory contiguity.
    // XXX: there may exist a better (faster) way.
    let mut train_inputs_vec = vec![vec![]; n_labels];

    for (x, y) in inputs.outer_iter().zip(targets) {
        // Implicitly asserts that 0 <= y < self.n_labels.
        train_inputs_vec[*y].extend(x.iter());
    }

    let d = inputs.cols();

    // Convert into arrays.
    let mut train_inputs = vec![];
    for inputs_y in train_inputs_vec {
        let n = inputs_y.len() / d;
        train_inputs.push(Array::from_shape_vec((n, d), inputs_y)
                                .expect("Unexpected error in reshaping"));
    }

    train_inputs
}

/// A k-NN nonconformity measure.
///
/// The score is defined for some distance metric and number of
/// neighbors.
pub struct KNN<T: Sync> {
    k: usize,
    distance: fn(&ArrayView1<T>, &ArrayView1<T>) -> f64,
    n_labels: Option<usize>,
    // Training inputs are stored in a train_inputs, indexed
    // by a label y, where train_inputs[y] contains all training
    // inputs with label y.
    train_inputs: Option<Vec<Array2<T>>>,
    // Calibration inputs are optional. If set, then the
    // NCM is trained on train_inputs, and the scores are
    // computed on calibration_inputs.
    calibration_inputs: Option<Vec<Array2<T>>>,
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
        KNN {
            k: k,
            distance: euclidean_distance,
            train_inputs: None,
            calibration_inputs: None,
            n_labels: None,
        }
    }
}

impl<T: Sync> NonconformityScorer<T> for KNN<T>
        where T: Clone + Sync + Copy {
    /// Trains a k-NN nonconformity scorer.
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
    /// * `n_labels` - Number of unique labels in the classification problem.
    fn train(&mut self, inputs: &ArrayView2<T>, targets: &ArrayView1<usize>,
             n_labels: usize) -> LearningResult<()> {
        if self.train_inputs.is_some() {
            panic!("Can only train once");
        }
        self.n_labels = Some(n_labels);
        self.train_inputs = Some(split_inputs(inputs, targets, n_labels));

        Ok(())
    }
    /// Calibrates a k-NN nonconformity scorer for an ICP.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Matrix (Array2<T>) with values of type T of training
    ///              vectors.
    /// * `targets` - Vector (Array1<T>) of labels corresponding to the
    ///               training vectors.
    /// * `n_labels` - Number of unique labels in the classification problem.
    fn calibrate(&mut self, inputs: &ArrayView2<T>, targets: &ArrayView1<usize>)
             -> LearningResult<()> {
        if self.train_inputs.is_none() {
            panic!("Need to train before calibrate()-ing");
        }
        self.calibration_inputs = Some(split_inputs(inputs, targets,
                                                    self.n_labels.unwrap()));

        Ok(())
    }

    /// Updates a k-NN nonconformity scorer with more training data.
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
    fn update(&mut self, inputs: &ArrayView2<T>, targets: &ArrayView1<usize>)
        -> LearningResult<()> {

        let train_inputs = match self.train_inputs {
            Some(ref mut train_inputs) => train_inputs,
            None => panic!("Call train() once before update()"),
        };

        // NOTE: when ndarray will have cheap concatenation, we
        // should iterate once through (inputs, targets) and just
        // append each (x, y) to the appropriate self.train_inputs[y].
        // The current method is less efficient than that.
        for (x, y) in inputs.outer_iter().zip(targets) {
            train_inputs[*y] = stack![Axis(0), train_inputs[*y],
                                      x.clone().into_shape((1, x.len()))
                                               .expect("Unexpected reshaping error")];
        }

        Ok(())
    }
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
    fn scores(&self, x: &ArrayView1<T>, y: usize) -> Vec<f64> {
        let train_inputs = self.train_inputs.as_ref()
                                            .expect("You should train the model first");
        let train_inputs_y = &train_inputs[y];
        let mut scores;

        // ICP.
        if let Some(calibration_inputs) = self.calibration_inputs.as_ref() {
            //// TODO: update to new version of ndarray and rejoyce.
            let test_inputs = stack![Axis(0), x.into_shape((1, train_inputs_y.cols()))
                                               .expect("Unexpected error in reshaping"),
                                     calibration_inputs[y].clone()];
            scores = Vec::with_capacity(test_inputs.len());
            let k = min(self.k, train_inputs_y.rows());
            for input in test_inputs.outer_iter() {
                let score = train_inputs_y.outer_iter()
                                          .map(|x| (self.distance)(&x, &input))
                                          .sorted_partial_last()
                                          .take(k)
                                          .sum::<f64>();
                scores.push(score);
            }
        }
        // TCP.
        else {
            // XXX: once ndarray supports appending a row, we should
            // append to the matrix rather than creating a new one.
            let test_inputs = stack![Axis(0), x.into_shape((1, train_inputs_y.cols()))
                                               .expect("Unexpected error in reshaping"),
                                     train_inputs_y.clone()];
            scores = Vec::with_capacity(test_inputs.len());
            let k = min(self.k, test_inputs.rows()-1);
            for i in 0..test_inputs.rows() {
                let input = test_inputs.row(i);
                let score = test_inputs.outer_iter()
                                       .enumerate()
                                       .filter(|&(j, _)| j != i)
                                       .map(|(_, x)| (self.distance)(&x, &input))
                                       .sorted_partial_last()
                                       .take(k)
                                       .sum::<f64>();
                scores.push(score);
            }
        }

        scores
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that KNN training succeeds properly (i.e., it
    /// correctly splits training inputs per label).
    #[test]
    fn train() {
        let mut knn = KNN::new(2);
        let n_labels = 3;

        let train_inputs = array![[2., 2.],
                                  [1., 2.],
                                  [0., 0.],
                                  [1., 0.],
                                  [0., 1.],
                                  [1., 1.]];
        let train_targets = array![2, 2, 0, 0, 1, 1];

        let expected_train_inputs = vec![array![[0., 0.],
                                                [1., 0.]],
                                         array![[0., 1.],
                                                [1., 1.]],
                                         array![[2., 2.],
                                                [1., 2.]]];

        knn.train(&train_inputs.view(), &train_targets.view(), n_labels).unwrap();

        assert!(knn.train_inputs.unwrap() == expected_train_inputs);
    }

    /// Verify that train() + update() on partial datasets is
    /// equivalent to train()-ing on full dataset.
    #[test]
    fn update() {
        // Train a k-NN
        let mut ncm = KNN::new(2);
        let n_labels = 3;
        let train_inputs_1 = array![[0., 0.],
                                    [0., 1.],
                                    [2., 2.]];
        let train_targets_1 = array![0, 1, 2];
        let train_inputs_2 = array![[1., 1.]];
        let train_targets_2 = array![0];
        let train_inputs_3 = array![[1., 2.],
                                    [2., 1.]];
        let train_targets_3 = array![1, 2];

        // First, train().
        ncm.train(&train_inputs_1.view(), &train_targets_1.view(), n_labels)
           .expect("Failed to train model");
        // Update with new data.
        ncm.update(&train_inputs_2.view(), &train_targets_2.view())
           .expect("Failed to train model");
        ncm.update(&train_inputs_3.view(), &train_targets_3.view())
           .expect("Failed to train model");

        // All this is identical to training the
        // k-NN ncm on all data once.
        let mut ncm_alt = KNN::new(2);

        let train_inputs = array![[0., 0.],
                                  [0., 1.],
                                  [2., 2.],
                                  [1., 1.],
                                  [1., 2.],
                                  [2., 1.]];
        let train_targets = array![0, 1, 2, 0, 1, 2];

        ncm_alt.train(&train_inputs.view(), &train_targets.view(), n_labels)
               .expect("Failed to train model");

        assert!(ncm.train_inputs == ncm_alt.train_inputs);
    }



    #[test]
    fn knn() {
        let mut knn = KNN::new(2);

        let train_inputs = array![[0., 0.],
                                  [1., 0.],
                                  [0., 1.]];
        let train_targets = array![0, 0, 0];
        let expected_scores = vec![4.47213595499958, 2., 2.414213562373095,
                                   2.414213562373095];
        let test_input = array![2., 2.];
        let test_target = 0;
        let n_labels = 1;

        knn.train(&train_inputs.view(), &train_targets.view(), n_labels)
           .expect("Failed to train k-NN ncm");
        let scores = knn.scores(&test_input.view(), test_target);

        assert!(scores == expected_scores);
    }
}
