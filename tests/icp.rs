extern crate ndarray;

extern crate random_world;

#[cfg(test)]
mod tests {
    use random_world::cp::*;
    use random_world::ncm::*;
    use ndarray::*;
    
    #[test]
    fn cp() {
        let k = 2;
        let ncm = KNN::new(k);
        let n_labels = 2;
        let mut cp = CP::new_inductive(ncm, n_labels, Some(0.1));

        let train_inputs = array![[0., 0.],
                                  [1., 0.],
                                  [1., 1.],
                                  [0., 1.],
                                  [2., 2.],
                                  [1., 2.]];
        let n_train = 3;            // Size of proper training set.
        let train_targets = array![0, 0, 1, 0, 1, 1];
        let test_inputs = array![[2., 1.],
                                 [2., 2.]];
        let expected_pvalues = array![[0.5, 1.],
                                      [0.5, 2./3.]];

        let epsilon_1 = 0.5;
        let epsilon_2 = 0.2;
        let expected_preds_1 = array![[false, true],
                                      [false, true]];
        let expected_preds_2 = array![[true, true],
                                      [true, true]];

        cp.train(&train_inputs.slice(s![..n_train, ..]),
                 &train_targets.slice(s![..n_train]))
          .expect("Failed to train the model");

        cp.calibrate(&train_inputs.slice(s![n_train.., ..]),
                     &train_targets.slice(s![n_train..]))
          .expect("Failed to train the model");

        let pvalues = cp.predict_confidence(&test_inputs.view()).unwrap();
        println!("Expected p-values: {:?}. P-values: {:?}.", expected_pvalues,
                 pvalues);
        assert!(pvalues == expected_pvalues);
        cp.set_epsilon(epsilon_1);
        assert!(cp.predict(&test_inputs.view()).unwrap() == expected_preds_1);
        cp.set_epsilon(epsilon_2);
        assert!(cp.predict(&test_inputs.view()).unwrap() == expected_preds_2);
    }
}
