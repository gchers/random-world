extern crate confident;
#[macro_use(array)]
extern crate ndarray;

#[cfg(test)]
mod tests {
    use confident::cp::*;
    use confident::ncm::*;
    
    #[test]
    fn cp() {
        let k = 2;
        let ncm = KNN::new(k);
        let mut cp = CP::new(ncm, Some(0.1));

        let train_inputs = array![[0., 0.],
                                  [1., 0.],
                                  [0., 1.],
                                  [1., 1.],
                                  [2., 2.],
                                  [1., 2.]];
        let train_targets = array![0, 0, 0, 1, 1, 1];
        let test_inputs = array![[2., 1.],
                                 [2., 2.]];
        let expected_pvalues = array![[0.25, 1.],
                                      [0.25, 1.]];

        let epsilon_1 = 0.3;
        let epsilon_2 = 0.2;
        let expected_preds_1 = array![[false, true],
                                      [false, true]];
        let expected_preds_2 = array![[true, true],
                                      [true, true]];

        cp.train(&train_inputs, &train_targets).unwrap();
        let pvalues = cp.predict_confidence(&test_inputs).unwrap();
        println!("Expected p-values: {:?}. P-values: {:?}.", expected_pvalues,
                 pvalues);
        assert!(pvalues == expected_pvalues);
        cp.set_epsilon(epsilon_1);
        assert!(cp.predict(&test_inputs).unwrap() == expected_preds_1);
        cp.set_epsilon(epsilon_2);
        assert!(cp.predict(&test_inputs).unwrap() == expected_preds_2);
    }

    #[test]
    fn smooth_cp() {
        let k = 2;
        let ncm = KNN::new(k);
        let seed = [0, 0];
        let mut cp = CP::new_smooth(ncm, Some(0.1), Some(seed));

        let train_inputs = array![[0., 0.],
                                  [1., 0.],
                                  [0., 1.],
                                  [1., 1.],
                                  [2., 2.],
                                  [1., 2.]];
        let train_targets = array![0, 0, 0, 1, 1, 1];
        let test_inputs = array![[2., 1.],
                                 [2., 2.]];
        let expected_pvalues = array![[0., 0.6688798670240814],
                                      [0.019990972812210628, 0.7553161717563366]];

        cp.train(&train_inputs, &train_targets).unwrap();

        let pvalues = cp.predict_confidence(&test_inputs).unwrap();
        println!("Expected p-values: {:?}.", expected_pvalues);
        println!("Actual p-values: {:?}.", pvalues);
        assert!(pvalues == expected_pvalues);
    }
}
