use pcg_rand::Pcg32;
use rand::{Rng, SeedableRng};
use itertools::Itertools;
use rusty_machine::learning::LearningResult;
use ndarray::prelude::*;
use ndarray::{Axis};

use ncm::NonconformityScorer;


/// A Confidence Predictor (either transductive or inductive CP)
pub trait ConfidencePredictor<T> {
    fn train(&mut self, inputs: &Array2<T>, targets: &Array1<usize>) -> LearningResult<()>;
    fn predict(&mut self, inputs: &Array2<T>) -> LearningResult<Array2<bool>>;
    fn predict_confidence(&mut self, inputs: &Array2<T>) -> LearningResult<Array2<f64>>;
    fn set_epsilon(&mut self, epsilon: f64);
    // TODO:
    // fn predict_region(&self, pvalues: &Matrix<f64>, epsilon: f64) -> ...
    // fn update(&self, inputs: &Vec<T>, targets: &Vec<usize>) -> LearningResult<()>;

}

/// Transductive Conformal Predictor
/// 
/// T: type of an object (e.g., Vec<f64>).
pub struct CP<T: Sync, N: NonconformityScorer<T>> {
    ncm: N,
    epsilon: Option<f64>,
    smooth: bool,
    rng: Option<Pcg32>,
    /* Training inputs are stored in a train_inputs, indexed
     * by a label y, where train_inputs[y] contains all training
     * inputs with label y.
     */
    train_inputs: Option<Vec<Array2<T>>>,
}

impl<T: Sync, N: NonconformityScorer<T>> CP<T,N> {
    pub fn new(ncm: N, epsilon: Option<f64>) -> CP<T,N> {
        CP {
            ncm: ncm,
            epsilon: epsilon,
            smooth: false,
            train_inputs: None,
            rng: None,
        }
    }

    pub fn new_smooth(ncm: N, epsilon: Option<f64>,
                      seed: Option<[u64; 2]>) -> CP<T,N> {
        CP {
            ncm: ncm,
            epsilon: epsilon,
            train_inputs: None,
            smooth: true,
            rng: match seed{
                    Some(seed) => Some(Pcg32::from_seed(seed)),
                    None => Some(Pcg32::new_unseeded())
                 },
        }
    }
}

impl<T, N> ConfidencePredictor<T> for CP<T,N>
        where T: Clone + Sync + Copy, N: NonconformityScorer<T> + Sync {

    fn set_epsilon(&mut self, epsilon: f64) {
        self.epsilon = Some(epsilon);
    }

    fn train(&mut self, inputs: &Array2<T>, targets: &Array1<usize>)
            -> LearningResult<()> {

        /* Split examples w.r.t. their labels. For each unique label y,
         * self.train_inputs[y] will contain a matrix with the inputs with
         * label y.
         */
        let mut train_inputs = vec![]; //Vec::with_capacity(n_labels);

        // TODO: keep track of label ordering, shrink_to_fit()
        // use .select()?
        for y in targets.iter().unique() {
            let inputs_y = inputs.outer_iter()
                                 .zip(targets)
                                 .filter(|&(_, _y)| _y== y)
                                 .flat_map(|(x, _)| x.to_vec())
                                 .collect::<Vec<_>>();

            let d = inputs.cols();
            let n = inputs_y.len() / d;

            train_inputs.push(Array::from_shape_vec((n, d), inputs_y)
                                    .expect("Unexpected error in reshaping"))
        }

        train_inputs.shrink_to_fit();

        self.train_inputs = Some(train_inputs);

        Ok(())
    }

    /// Returns a region prediction as a matrix of boolean
    /// values, where each column corresponds to a label,
    /// each value to an input object, and the value is
    /// true if the label conforms the distribution, false
    /// otherwise.
    fn predict(&mut self, inputs: &Array2<T>) -> LearningResult<Array2<bool>> {
        let epsilon = self.epsilon.expect("Specify epsilon to perform a standard predict()");

        let pvalues = self.predict_confidence(inputs).expect("Failed to predict p-values");

        let preds = Array::from_iter(pvalues.iter()
                                            .map(|p| *p > epsilon))
                          .into_shape((pvalues.rows(), pvalues.cols()))
                          .expect("Unexpected error in converting p-values into predictions");

        Ok(preds)
    }

    /// Returns the p-values corresponding to the labels
    /// for each object provided as input.
    fn predict_confidence(&mut self, inputs: &Array2<T>) -> LearningResult<Array2<f64>> {
        let CP { smooth,
                 ref train_inputs,
                 ref ncm,
                 ref mut rng, .. } = *self;

        let train_inputs = train_inputs.as_ref()
                                       .expect("You should train the model first");

        let mut pvalues = Array2::<f64>::zeros((inputs.rows(), train_inputs.len()));

        for (y, train_inputs_y) in train_inputs.into_iter().enumerate() {

            /* The number accounts of the object we will temporarily append.
             */
            let n_tmp = train_inputs_y.rows() + 1;

            for (i, test_x) in inputs.outer_iter().enumerate() {

                /* Temporarily add test_x to training inputs with label y */
                /* XXX: once ndarray supports appending a row, we should
                 * append to the matrix rather than creating a new one.
                 */
                let train_inputs_tmp = stack![Axis(0), *train_inputs_y,
                                              test_x.into_shape((1, inputs.cols()))
                                                    .unwrap()];

                let x_score = ncm.score(n_tmp-1, &train_inputs_tmp);

                let mut gt = 0.;
                let mut eq = 1.;

                for j in 0..n_tmp-1 {
                    /* Compute nonconformity scores.
                     */
                    let score = ncm.score(j, &train_inputs_tmp);

                    /* Keep track of greater than and equal */
                    match () {
                        _ if score > x_score => gt += 1.,
                        _ if score == x_score => eq += 1.,
                        _ => {},
                    }
                };

                /* Compute p-value.
                 */
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
        let mut cp = CP::new(ncm, Some(0.1));

        let train_inputs = array![[0., 0.],
                                  [1., 0.],
                                  [0., 1.],
                                  [1., 1.],
                                  [2., 2.],
                                  [1., 2.]];
        let train_targets = array![0, 0, 1, 1, 2, 2];

        let expected_train_inputs = vec![array![[0., 0.],
                                                [1., 0.]],
                                         array![[0., 1.],
                                                [1., 1.]],
                                         array![[2., 2.],
                                                [1., 2.]]];

        cp.train(&train_inputs, &train_targets).unwrap();

        assert!(cp.train_inputs.unwrap() == expected_train_inputs);
    }

    /// Verify that the internal PRNG generates the same sequence
    /// of numbers when seeded.
    /// NOTE: if we ever want to change the PRNG, the hardcoded
    /// values in this function also need to be changed appropriately.
    #[test]
    fn rnd_seeded() {
        let ncm = KNN::new(2);
        let seed = [0, 0];
        let mut cp = CP::new_smooth(ncm, Some(0.1), Some(seed));

        let r  = cp.rng.as_mut()
                  .expect("Initialize smooth CP to use")
                  .gen_iter::<f64>()
                  .take(5)
                  .collect::<Vec<_>>();

        assert!(r == vec![0., 0.07996389124884251, 0.6688798670240814,
                          0.5106323435126732, 0.5024848655054046]);
    }
}
