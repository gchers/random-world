pub use rusty_machine::linalg::{Matrix, BaseMatrix, Vector};

use ncm::NonConformityScorer;


/// A Confidence Predictor (either transductive or inductive CP)
trait ConfidencePredictor<T> {
    fn train(&mut self, inputs: &T, targets: &Vector<i64>) -> LearningResult<()>;
    fn predict(&self, inputs: &T) -> LearningResult<Matrix<bool>>;
    fn predict_confidence(&self, inputs: &T) -> LearningResult<Matrix<f64>>;
}

/// Transductive Conformal Predictor
pub struct CP<T> {
    ncm: Box<NonConformityScorer<T>>,
    epsilon: Option<f64>,
    smooth: bool,
    train_inputs: Option<T>,
    train_targets: Option<Vector<i64>>,
    n_labels: Option<usize>,
}

impl<T> CP<T> {
    pub fn new(ncm: Box<NonConformityScorer<T>>, epsilon: Option<f64>, smooth: bool)
            -> CP<T> {
        CP {
            ncm: ncm,
            epsilon: epsilon,
            smooth: smooth,
            train_inputs: None,
            train_targets: None,
            n_labels: None,
        }
    }
}

//generic doesn't work:
//impl ConfPredictor<T> for CP<T> where T: Matrix {
impl ConfidencePredictor<Matrix<f64>> for CP<Matrix<f64>> {

    fn train(&mut self, inputs: &Matrix<f64>, targets: &Vector<i64>)
            -> LearningResult<()> {
        self.train_inputs = Some(inputs.clone());
        self.train_targets = Some(targets.clone());
        // NOTE: is there a better way?
        let mut sorted_targets = targets.clone()
                                        .into_vec();
        sorted_targets.sort();
        sorted_targets.dedup();
        self.n_labels = Some(sorted_targets.len());

        Ok(())
    }

    /// Returns a region prediction as a matrix of boolean
    /// values, where each column corresponds to a label,
    /// each value to an input object, and the value is
    /// true if the label conforms the distribution, false
    /// otherwise.
    fn predict(&self, inputs: &Matrix<f64>) -> LearningResult<Matrix<bool>> {
        let epsilon = self.epsilon.expect("Specify epsilon to perform a standard predict()");

        let pvalues = self.predict_confidence(inputs).expect("Failed to predict p-values");

        let preds = Matrix::from_fn(pvalues.rows(), pvalues.cols(),
                                    |j, i| pvalues[[i,j]] > epsilon);
        
        Ok(preds)
    }

    // TODO:
    // fn predict_region(&self, pvalues: &Matrix<f64>, epsilon: f64) -> ...

    /// Returns the p-values corresponding to the labels
    /// for each object provided as input.
    fn predict_confidence(&self, inputs: &Matrix<f64>) -> LearningResult<Matrix<f64>> {

        let error_msg = "You should train the model first";
        // XXX: try with if let...?
        let train_inputs = self.train_inputs.as_ref().expect(error_msg);
        let train_targets = self.train_targets.as_ref().expect(error_msg);
        let n_labels = self.n_labels.expect(error_msg);

        let n_test = inputs.rows();

        let mut pvalues = Matrix::new(n_test, n_labels,
                                      vec![0.0; n_test*n_labels]);

        // We first iterate through labels and then through input
        // examples. This so we only compute train_inputs_l once
        // for each label
        for label in 0..n_labels {

            // Select examples with current label
            let train_inputs_l = train_inputs.iter_rows()
                                             .zip(train_targets.iter())
                                             .filter(|&(_, y)| *y==label as i64)
                                             .map(|(x, _)| x)
                                             .collect::<Matrix<_>>();

            for (i, x) in inputs.iter_rows().enumerate() {
                let inputs_tmp = train_inputs_l.vcat(&Matrix::new(1, x.len(),
                                                                  x));
                let n_tmp = inputs_tmp.rows();

                let scores = (0..n_tmp).map(|j| self.ncm.score(j, &inputs_tmp))
                                       .collect::<Vec<_>>();

                let pvalue = if self.smooth {
                        unimplemented!();
                        let r = 0.1;
                        let a = scores.iter()
                                      .filter(|&s| *s > scores[n_tmp-1])
                                      .count() as f64;
                        let b = scores.iter()
                                      .filter(|&s| *s == scores[n_tmp-1])
                                      .count() as f64;
                        (a + r*b) / n_tmp as f64
                    } else {
                        scores.iter()
                              .filter(|&s| *s >= scores[n_tmp-1])
                              .count() as f64 / n_tmp as f64
                };

                pvalues[[i,label]] = pvalue;
            }
        }

        Ok(pvalues)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use ncm::KNN;
    
    #[test]
    fn cp() {
        let k = 2;
        let ncm = KNN::new(k);
        let mut cp = CP::new(Box::new(ncm), Some(0.1), false);

        let train_inputs = Matrix::new(6, 2, vec![0., 0.,
                                                  1., 0.,
                                                  0., 1.,
                                                  1., 1.,
                                                  2., 2.,
                                                  1., 2.]);
        let train_targets = Vector::new(vec![0, 0, 0, 1, 1, 1]);
        let test_inputs = Matrix::new(2, 2, vec![2., 1.,
                                                 2., 2.]);
        let expected_pvalues = Matrix::new(2, 2, vec![0.25, 1.,
                                                      0.25, 1.]);
        let epsilon_1 = 0.3;
        let epsilon_2 = 0.2;
        let expected_preds_1 = Matrix::new(2, 2, vec![false, true,
                                                      false, true]);
        let expected_preds_2 = Matrix::new(2, 2, vec![true, true,
                                                      true, true]);

        cp.train(&train_inputs, &train_targets);
        assert!(cp.predict_confidence(&test_inputs).unwrap() == expected_pvalues);
        cp.epsilon = Some(epsilon_1);
        assert!(cp.predict(&test_inputs).unwrap() == expected_preds_1);
        cp.epsilon = Some(epsilon_2);
        assert!(cp.predict(&test_inputs).unwrap() == expected_preds_2);

    }
}
