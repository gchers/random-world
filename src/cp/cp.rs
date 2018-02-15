//! Transductive deterministic or smooth Conformal Predictors.
use pcg_rand::Pcg32;
use rand::{Rng, SeedableRng};
use rusty_machine::learning::LearningResult;
use ndarray::prelude::*;
use ndarray::{Axis};
use std::f64::NAN;

use cp::ConfidencePredictor;
use ncm::NonconformityScorer;


/// A Conformal Predictor, for some nonconformity scorer N and
/// matrix element type T.
///
/// CP can either be deterministic, where `smooth=false` and `rng=None`,
/// or smooth, where `smooth=true` and rng is some pseudo random number
/// generator (PRNG).
/// Let `Y` be a list of values ("prediction region") predicted by a CP
/// for a test input vector `x` with true label `y`.
/// If CP is constructed as deterministic, then:
/// $Pr(y \notin Y) \leq \varepsilon$, where $\varepsilon$ is the specified
/// significance level `epsilon`;
/// if CP is smooth, then:
/// $Pr(y \notin Y) = \varepsilon$.
pub struct CP<T: Sync, N: NonconformityScorer<T>> {
    ncm: N,
    epsilon: Option<f64>,
    smooth: bool,
    rng: Option<Pcg32>,
    n_labels: usize,
    // Training inputs are stored in a train_inputs, indexed
    // by a label y, where train_inputs[y] contains all training
    // inputs with label y.
    train_inputs: Option<Vec<Array2<T>>>,
}

impl<T: Sync, N: NonconformityScorer<T>> CP<T, N> {
    /// Constructs a new deterministic Transductive Conformal Predictor
    /// `CP<T,N>` from a nonconformity score NonconformityScorer.
    ///
    /// # Arguments
    ///
    /// * `ncm` - An object implementing NonconformityScorer.
    /// * `n_labels` - The number of labels.
    /// * `epsilon` - Either Some() significance level in [0,1] or None.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_world::cp::*;
    /// use random_world::ncm::*;
    ///
    /// let ncm = KNN::new(2);
    /// let n_labels = 2;
    /// let epsilon = 0.1;
    /// let mut cp = CP::new(ncm, n_labels, Some(epsilon));
    /// ```
    pub fn new(ncm: N, n_labels: usize, epsilon: Option<f64>) -> CP<T, N> {
        assert!(n_labels > 0);

        if let Some(e) = epsilon {
            assert!(e >= 0. && e <= 1.);
        }

        CP {
            ncm: ncm,
            epsilon: epsilon,
            smooth: false,
            n_labels: n_labels,
            train_inputs: None,
            rng: None,
        }
    }

    /// Constructs a new smooth Transductive Conformal Predictor
    /// `CP<T,N>` from a nonconformity score NonconformityScorer.
    ///
    /// # Arguments
    ///
    /// * `ncm` - An object implementing NonconformityScorer.
    /// * `n_labels` - The number of labels.
    /// * `epsilon` - Either Some() significance level in [0,1] or None.
    /// * `seed` - Optionally, a slice of 2 elements is provided as seed
    ///            to the random number generator.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_world::cp::*;
    /// use random_world::ncm::*;
    ///
    /// let ncm = KNN::new(2);
    /// let n_labels = 2;
    /// let epsilon = 0.1;
    /// let seed = [0, 0];
    /// let mut cp = CP::new_smooth(ncm, n_labels, Some(epsilon), Some(seed));
    /// ```
    pub fn new_smooth(ncm: N, n_labels: usize, epsilon: Option<f64>,
                      seed: Option<[u64; 2]>) -> CP<T, N> {

        if let Some(e) = epsilon {
            assert!(e >= 0. && e <= 1.);
        }

        CP {
            ncm: ncm,
            epsilon: epsilon,
            smooth: true,
            n_labels: n_labels,
            train_inputs: None,
            rng: match seed {
                Some(seed) => Some(Pcg32::from_seed(seed)),
                None => Some(Pcg32::new_unseeded())
            },
        }
    }
}

impl<T, N> ConfidencePredictor<T> for CP<T, N>
        where T: Clone + Sync + Copy, N: NonconformityScorer<T> + Sync {

    /// Sets the significance level.
    ///
    /// # Arguments
    ///
    /// * `epsilon` - Significance level in [0,1].
    fn set_epsilon(&mut self, epsilon: f64) {
        assert!(epsilon >= 0. && epsilon <= 1.);

        self.epsilon = Some(epsilon);
    }

    /// Trains a Conformal Predictor on a training set.
    ///
    /// Pedantic note: because CP is a transductive method, it never
    /// actually trains a model.
    /// This function, however, structures the training data so that
    /// it can be easily used in the prediction phase.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Matrix (Array2<T>) with values of type T of training
    ///              vectors.
    /// * `targets` - Vector (Array1<T>) of labels corresponding to the
    ///               training vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// #[macro_use(array)]
    /// extern crate ndarray;
    /// extern crate random_world;
    ///
    /// # fn main() {
    /// use random_world::cp::*;
    /// use random_world::ncm::*;
    ///
    /// // Train a CP
    /// let ncm = KNN::new(2);
    /// let n_labels = 3;
    /// let epsilon = 0.1;
    /// let mut cp = CP::new(ncm, n_labels, Some(epsilon));
    /// let train_inputs = array![[0., 0.],
    ///                           [1., 0.],
    ///                           [0., 1.],
    ///                           [1., 1.],
    ///                           [2., 2.],
    ///                           [1., 2.]];
    /// let train_targets = array![0, 0, 1, 1, 2, 2];
    ///
    /// cp.train(&train_inputs.view(), &train_targets.view())
    ///   .expect("Failed to train model");
    /// # }
    /// ```
    ///
    /// # Panics
    ///
    /// - if the number of training examples is not consistent
    ///   with the number of respective labels.
    /// - If labels are not numbers starting from 0 and containing all
    ///   numbers up to n_labels-1.
    fn train(&mut self, inputs: &ArrayView2<T>, targets: &ArrayView1<usize>)
             -> LearningResult<()> {

        assert!(inputs.rows() == targets.len());
        if self.train_inputs.is_some() {
            panic!("Can only train() once. Use update() to update training data");
        }

        // Split examples w.r.t. their labels. For each unique label y,
        // self.train_inputs[y] will contain a matrix with the inputs with
        // label y.
        // We first put them into a vector, and then will convert
        // into array. This should guarantee memory contiguity.
        // XXX: there may exist a better (faster) way.
        let mut train_inputs_vec = vec![vec![]; self.n_labels];

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
        
        self.train_inputs = Some(train_inputs);

        Ok(())
    }

    /// Updates a Conformal Predictor with more training data.
    ///
    /// After calling `train()` once, `update()` allows to add
    /// inputs to the Conformal Predictor's training data,
    /// which will be used for future predictions.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Matrix (Array2<T>) with values of type T of training
    ///              vectors.
    /// * `targets` - Vector (Array1<T>) of labels corresponding to the
    ///               training vectors.
    ///
    /// # Examples
    ///
    /// The following examples creates two CPs, `cp` and `cp_alt`,
    /// one `train()`-ed and `update()`d on partial data, the other
    /// one `train()`-ed on full data;
    /// these CPs are equivalent (i.e., their training data is identical).
    ///
    /// ```
    /// #[macro_use(array)]
    /// extern crate ndarray;
    /// extern crate random_world;
    ///
    /// # fn main() {
    /// use random_world::cp::*;
    /// use random_world::ncm::*;
    ///
    /// // Train a CP
    /// let ncm = KNN::new(2);
    /// let n_labels = 3;
    /// let epsilon = 0.1;
    /// let mut cp = CP::new(ncm, n_labels, Some(epsilon));
    /// let train_inputs_1 = array![[0., 0.],
    ///                             [0., 1.],
    ///                             [2., 2.]];
    /// let train_targets_1 = array![0, 1, 2];
    /// let train_inputs_2 = array![[1., 1.]];
    /// let train_targets_2 = array![0];
    /// let train_inputs_3 = array![[1., 2.],
    ///                             [2., 1.]];
    /// let train_targets_3 = array![1, 2];
    ///
    /// // First, train().
    /// cp.train(&train_inputs_1.view(), &train_targets_1.view())
    ///   .expect("Failed to train model");
    /// // Update with new data.
    /// cp.update(&train_inputs_2.view(), &train_targets_2.view())
    ///   .expect("Failed to train model");
    /// cp.update(&train_inputs_3.view(), &train_targets_3.view())
    ///   .expect("Failed to train model");
    ///
    /// // All this is identical to training the
    /// // CP on all data once.
    /// let ncm_alt = KNN::new(2);
    /// let mut cp_alt = CP::new(ncm_alt, n_labels, Some(0.1));
    ///
    /// let train_inputs = array![[0., 0.],
    ///                           [0., 1.],
    ///                           [2., 2.],
    ///                           [1., 1.],
    ///                           [1., 2.],
    ///                           [2., 1.]];
    /// let train_targets = array![0, 1, 2, 0, 1, 2];
    ///
    /// cp_alt.train(&train_inputs.view(), &train_targets.view())
    ///       .expect("Failed to train model");
    ///
    /// // The two CPs are equivalent.
    /// let preds = cp.predict(&train_inputs.view())
    ///               .expect("Failed to predict");
    /// let preds_alt = cp_alt.predict(&train_inputs.view())
    ///                       .expect("Failed to predict");
    /// assert!(preds == preds_alt);
    ///
    /// # }
    /// ```
    ///
    /// # Panics
    ///
    /// - if the number of training examples is not consistent
    ///   with the number of respective labels.
    /// - if labels are not numbers starting from 0 and containing all
    ///   numbers up to n_labels-1.
    /// - if `train()` hasn't been called once before (i.e., if
    ///   `self.train_inputs` is `None`.
    fn update(&mut self, inputs: &ArrayView2<T>, targets: &ArrayView1<usize>)
             -> LearningResult<()> {

        assert!(inputs.rows() == targets.len());

        let mut train_inputs = match self.train_inputs {
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

    /// Returns candidate labels (region prediction) for test vectors.
    ///
    /// The return value is a matrix of `bool` (`Array2<bool>`) with shape
    /// `(n_inputs, n_labels)`, where `n_inputs = inputs.rows()` and
    /// `n_labels` is the number of possible labels;
    /// in such matrix, each column `y` corresponds to a label,
    /// each row `i` to an input object, and the value at `[i,y]` is
    /// true if the label conforms the distribution, false otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// #[macro_use(array)]
    /// extern crate ndarray;
    /// extern crate random_world;
    ///
    /// # fn main() {
    /// use random_world::cp::*;
    /// use random_world::ncm::*;
    ///
    /// // Construct a deterministic CP with k-NN nonconformity measure (k=2).
    /// let ncm = KNN::new(2);
    /// let n_labels = 2;
    /// let mut cp = CP::new(ncm, n_labels, Some(0.3));
    /// let train_inputs = array![[0., 0.],
    ///                           [1., 0.],
    ///                           [0., 1.],
    ///                           [1., 1.],
    ///                           [2., 2.],
    ///                           [1., 2.]];
    /// let train_targets = array![0, 0, 0, 1, 1, 1];
    /// let test_inputs = array![[2., 1.],
    ///                          [2., 2.]];
    ///
    /// // Train and predict
    /// cp.train(&train_inputs.view(), &train_targets.view())
    ///   .expect("Failed prediction");
    /// let preds = cp.predict(&test_inputs.view())
    ///               .expect("Failed to predict");
    /// assert!(preds == array![[false, true],
    ///                         [false, true]]);
    /// # }
    /// ```
    fn predict(&mut self, inputs: &ArrayView2<T>) -> LearningResult<Array2<bool>> {
        let epsilon = self.epsilon.expect("Specify epsilon to perform a standard predict()");

        let pvalues = self.predict_confidence(inputs).expect("Failed to predict p-values");

        let preds = Array::from_iter(pvalues.iter()
                                            .map(|p| *p > epsilon))
                          .into_shape((pvalues.rows(), pvalues.cols()))
                          .expect("Unexpected error in converting p-values into predictions");

        Ok(preds)
    }

    /// Returns the p-values for test vectors.
    ///
    /// The return value is a matrix of `f64` (`Array2<f64>`) with shape
    /// `(n_inputs, n_labels)`, where `n_inputs = inputs.rows()` and
    /// `n_labels` is the number of possible labels;
    /// in such matrix, each column `y` corresponds to a label,
    /// each row `i` to an input object, and the value at `[i,y]` is
    /// the p-value obtained when assuming `y` as a label for the
    /// `i`-th input object.
    ///
    /// # Examples
    ///
    /// ```
    /// #[macro_use(array)]
    /// extern crate ndarray;
    /// extern crate random_world;
    ///
    /// # fn main() {
    /// use random_world::cp::*;
    /// use random_world::ncm::*;
    ///
    /// // Construct a deterministic CP with k-NN nonconformity measure (k=2).
    /// let ncm = KNN::new(2);
    /// let n_labels = 2;
    /// let mut cp = CP::new(ncm, n_labels, Some(0.1));
    /// let train_inputs = array![[0., 0.],
    ///                           [1., 0.],
    ///                           [0., 1.],
    ///                           [1., 1.],
    ///                           [2., 2.],
    ///                           [1., 2.]];
    /// let train_targets = array![0, 0, 0, 1, 1, 1];
    /// let test_inputs = array![[2., 1.],
    ///                          [2., 2.]];
    ///
    /// // Train and predict p-values
    /// cp.train(&train_inputs.view(), &train_targets.view()).unwrap();
    /// let pvalues = cp.predict_confidence(&test_inputs.view())
    ///                 .expect("Failed prediction");
    /// assert!(pvalues == array![[0.25, 1.],
    ///                           [0.25, 1.]]);
    /// }
    /// ```
    fn predict_confidence(&mut self, inputs: &ArrayView2<T>) -> LearningResult<Array2<f64>> {
        let CP { smooth,
                 ref train_inputs,
                 ref ncm,
                 ref mut rng, .. } = *self;

        let train_inputs = train_inputs.as_ref()
                                       .expect("You should train the model first");

        // Init with NaN to ease future debugging.
        let mut pvalues = Array2::<f64>::from_elem((inputs.rows(), train_inputs.len()),
                                                   NAN);

        for (y, train_inputs_y) in train_inputs.into_iter().enumerate() {

            // n_tmp accounts for the object we will temporarily append.
            let n_tmp = train_inputs_y.rows() + 1;

            for (i, test_x) in inputs.outer_iter().enumerate() {

                // If no training examples of label y, set the p-value to 1.
                if train_inputs_y.rows() == 0 {
                    pvalues[[i,y]] = 1.;
                    break;
                }

                // Temporarily add test_x to training inputs with label y.
                // XXX: once ndarray supports appending a row, we should
                // append to the matrix rather than creating a new one.
                let train_inputs_tmp = stack![Axis(0), *train_inputs_y,
                                              test_x.into_shape((1, inputs.cols()))
                                                    .expect("Unexpected error in reshaping")];

                let x_score = ncm.score(n_tmp-1, &train_inputs_tmp.view());

                let mut gt = 0.;
                let mut eq = 1.;

                for j in 0..n_tmp-1 {
                    // Compute nonconformity scores.
                    let score = ncm.score(j, &train_inputs_tmp.view());

                    // Keep track of greater than and equal.
                    match () {
                        _ if score > x_score => gt += 1.,
                        _ if score == x_score => eq += 1.,
                        _ => {},
                    }
                };

                // Compute p-value.
                let pvalue = if smooth {
                    let tau = rng.as_mut()
                                 .expect("Initialize as smooth CP to use")
                                 .gen::<f64>();
                    (gt + eq*tau) / (n_tmp as f64)
                } else {
                    (gt + eq) / (n_tmp as f64)
                };

                pvalues[[i,y]] = pvalue;
            }
        }

        Ok(pvalues)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ncm::KNN;
    
    /// Verify that training CP succeeds properly (i.e., it
    /// correctly splits training inputs per label).
    #[test]
    fn train() {
        let ncm = KNN::new(2);
        let n_labels = 3;
        let mut cp = CP::new(ncm, n_labels, Some(0.1));

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

        cp.train(&train_inputs.view(), &train_targets.view()).unwrap();

        assert!(cp.train_inputs.unwrap() == expected_train_inputs);
    }

    /// Verify that train() + update() on partial datasets is
    /// equivalent to train()-ing on full dataset.
    #[test]
    fn update() {
        // Train a CP
        let ncm = KNN::new(2);
        let n_labels = 3;
        let epsilon = 0.1;
        let mut cp = CP::new(ncm, n_labels, Some(epsilon));
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
        cp.train(&train_inputs_1.view(), &train_targets_1.view())
          .expect("Failed to train model");
        // Update with new data.
        cp.update(&train_inputs_2.view(), &train_targets_2.view())
          .expect("Failed to train model");
        cp.update(&train_inputs_3.view(), &train_targets_3.view())
          .expect("Failed to train model");

        // All this is identical to training the
        // CP on all data once.
        let ncm_alt = KNN::new(2);
        let mut cp_alt = CP::new(ncm_alt, n_labels, Some(epsilon));

        let train_inputs = array![[0., 0.],
                                  [0., 1.],
                                  [2., 2.],
                                  [1., 1.],
                                  [1., 2.],
                                  [2., 1.]];
        let train_targets = array![0, 1, 2, 0, 1, 2];

        cp_alt.train(&train_inputs.view(), &train_targets.view())
              .expect("Failed to train model");

        assert!(cp.train_inputs == cp_alt.train_inputs);
    }

    /// Verify that the internal PRNG generates the same sequence
    /// of numbers when seeded.
    /// NOTE: if we ever want to change the PRNG, the hardcoded
    /// values in this function also need to be changed appropriately.
    #[test]
    fn rnd_seeded() {
        let ncm = KNN::new(2);
        let n_labels = 2;
        let seed = [0, 0];
        let mut cp = CP::new_smooth(ncm, n_labels, Some(0.1), Some(seed));

        let r  = cp.rng.as_mut()
                  .expect("Initialize smooth CP to use")
                  .gen_iter::<f64>()
                  .take(5)
                  .collect::<Vec<_>>();

        assert!(r == vec![0., 0.07996389124884251, 0.6688798670240814,
                          0.5106323435126732, 0.5024848655054046]);
    }
}
