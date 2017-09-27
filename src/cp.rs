use rayon::prelude::*;
use pcg_rand::Pcg32;
use rand::{Rng, SeedableRng};
use itertools::Itertools;
use rusty_machine::linalg::{Matrix, BaseMatrix};
use rusty_machine::learning::LearningResult;

use ncm::NonConformityScorer;


/// A Confidence Predictor (either transductive or inductive CP)
pub trait ConfidencePredictor<T> {
    fn train(&mut self, inputs: &Vec<T>, targets: &Vec<usize>) -> LearningResult<()>;
    fn predict(&mut self, inputs: &Vec<T>) -> LearningResult<Matrix<bool>>;
    fn predict_confidence(&mut self, inputs: &Vec<T>) -> LearningResult<Matrix<f64>>;
    fn set_epsilon(&mut self, epsilon: f64);
    // TODO:
    // fn predict_region(&self, pvalues: &Matrix<f64>, epsilon: f64) -> ...
    // fn update(&self, inputs: &Vec<T>, targets: &Vec<usize>) -> LearningResult<()>;

}

/// Transductive Conformal Predictor
/// 
/// T: type of an object (e.g., Vec<f64>).
pub struct CP<T: Sync, N: Fn(usize, &[T]) -> f64> {
    ncm: N,
    epsilon: Option<f64>,
    smooth: bool,
    rng: Option<Pcg32>,
    /* Training inputs are stored in a train_inputs, indexed
     * by a label y, where train_inputs[y] contains all training
     * inputs with label y.
     */
    train_inputs: Option<Vec<Vec<T>>>,
}

impl<T: Sync, N: Fn(usize, &[T]) -> f64> CP<T,N> {
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
        where T: Clone + Sync, N:  Fn(usize, &[T]) -> f64 + Sync {

    fn set_epsilon(&mut self, epsilon: f64) {
        self.epsilon = Some(epsilon);
    }

    fn train(&mut self, inputs: &Vec<T>, targets: &Vec<usize>)
            -> LearningResult<()> {

        /* Split examples w.r.t. their labels. For each unique label y,
         * self.train_inputs[y] will contain a vector of the inputs with
         * label y.
         */
        let n_labels = targets.iter()
                              .unique()
                              .count();
        self.train_inputs = Some(inputs.iter()
                                       .zip(targets)
                                       .fold(vec![vec![]; n_labels],
                                             |mut res, (x, y)| {
                                                res[*y].push(x.clone());
                                                res
                                             }));

        Ok(())
    }

    /// Returns a region prediction as a matrix of boolean
    /// values, where each column corresponds to a label,
    /// each value to an input object, and the value is
    /// true if the label conforms the distribution, false
    /// otherwise.
    fn predict(&mut self, inputs: &Vec<T>) -> LearningResult<Matrix<bool>> {
        let epsilon = self.epsilon.expect("Specify epsilon to perform a standard predict()");

        let pvalues = self.predict_confidence(inputs).expect("Failed to predict p-values");

        let preds = Matrix::from_fn(pvalues.rows(), pvalues.cols(),
                                    |j, i| pvalues[[i,j]] > epsilon);
        
        Ok(preds)
    }

    /// Returns the p-values corresponding to the labels
    /// for each object provided as input.
    fn predict_confidence(&mut self, inputs: &Vec<T>) -> LearningResult<Matrix<f64>> {
        let CP { smooth,
                 ref mut train_inputs,
                 ref ncm,
                 ref mut rng, .. } = *self;

        let error_msg = "You should train the model first";

        let n_labels = train_inputs.as_ref()
                                   .expect(error_msg)
                                   .len();

        let n_test = inputs.len();

        let mut pvalues = Matrix::new(n_test, n_labels,
                                      vec![0.0; n_test*n_labels]);

        /* We first iterate through labels and then through input
         * examples.
         */
        for y in 0..n_labels {

            //train_inputs_l.reserve(1);
            let n_tmp = train_inputs.as_ref()
                                    .expect(error_msg)[y]
                                    .len() + 1; /* Count includes 1 test example */

            for (i, x) in inputs.iter().enumerate() {
                /* Temporarily add x to the training data with the
                 * current label.
                 */
                {
                    train_inputs.as_mut()
                                .expect(error_msg)[y]
                                .push(x.clone());
                }

                /* Compute nonconformity scores.
                 */
                let tau = match smooth {
                    true => rng.as_mut()
                               .expect("Initialize as smooth CP to use")
                               .gen::<f64>(),
                    false => 1.
                };

                let pvalue = {
                    let train_inputs = train_inputs.as_ref()
                                                   .expect(error_msg)[y]
                                                   .as_slice();

                    let x_score = ncm(n_tmp-1, train_inputs);

                    let (gt, eq) = (0..n_tmp-1).into_iter()
                                               /* Compute NCMs */
                                               .map(|j| ncm(j, train_inputs))
                                               /* Get count of scores > x_score
                                                * and = x_score.
                                                */
                                               .fold((0., 1.), |(gt, eq), s| {
                                                   match s {
                                                       _ if s > x_score => (gt+1., eq),
                                                       _ if s == x_score => (gt, eq+1.),
                                                       _ => (gt, eq),
                                                   }
                                               });

                    (gt + eq*tau) / (n_tmp as f64)
                };

                pvalues[[i,y]] = pvalue;

                /* Remove x from training data. */
                {
                    train_inputs.as_mut()
                                .expect(error_msg)[y]
                                .pop();
                }
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
        let mut cp = CP::new(|x, y| ncm.score(x, y), Some(0.1));

        let train_inputs = vec![vec![0., 0.],
                                vec![1., 0.],
                                vec![0., 1.],
                                vec![1., 1.],
                                vec![2., 2.],
                                vec![1., 2.]];
        let train_targets = vec![0, 0, 1, 1, 2, 2];

        let expected_train_inputs = vec![vec![vec![0., 0.],
                                              vec![1., 0.]],
                                         vec![vec![0., 1.],
                                              vec![1., 1.]],
                                         vec![vec![2., 2.],
                                              vec![1., 2.]]];

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
        let mut cp = CP::new_smooth(|x, y| ncm.score(x, y), Some(0.1),
                                    Some(seed));

        let r  = cp.rng.as_mut()
                  .expect("Initialize smooth CP to use")
                  .gen_iter::<f64>()
                  .take(5)
                  .collect::<Vec<_>>();

        assert!(r == vec![0., 0.07996389124884251, 0.6688798670240814,
                          0.5106323435126732, 0.5024848655054046]);
    }
}
