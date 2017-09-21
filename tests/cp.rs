extern crate confident;
extern crate rusty_machine;

#[cfg(test)]
mod tests {
    use confident::cp::*;
    use confident::ncm::*;
    use rusty_machine::linalg::Matrix;
    
    #[test]
    fn cp() {
        let k = 2;
        let ncm = KNN::new(k);
        let mut cp = CP::new(Box::new(ncm), Some(0.1), false);

        let train_inputs = vec![vec![0., 0.],
                                vec![1., 0.],
                                vec![0., 1.],
                                vec![1., 1.],
                                vec![2., 2.],
                                vec![1., 2.]];
        let train_targets = vec![0, 0, 0, 1, 1, 1];
        let test_inputs = vec![vec![2., 1.],
                               vec![2., 2.]];
        let expected_pvalues = Matrix::new(2, 2, vec![0.25, 1., 0.25, 1.]);

        let epsilon_1 = 0.3;
        let epsilon_2 = 0.2;
        let expected_preds_1 = Matrix::new(2, 2, vec![false, true, false, true]);
        let expected_preds_2 = Matrix::new(2, 2, vec![true, true, true, true]);

        cp.train(&train_inputs, &train_targets);
        assert!(cp.predict_confidence(&test_inputs).unwrap() == expected_pvalues);
        cp.set_epsilon(epsilon_1);
        assert!(cp.predict(&test_inputs).unwrap() == expected_preds_1);
        cp.set_epsilon(epsilon_2);
        assert!(cp.predict(&test_inputs).unwrap() == expected_preds_2);
    }
}
