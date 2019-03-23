extern crate ndarray;
extern crate itertools;

extern crate random_world;

#[cfg(test)]
mod tests {
    use ndarray::*;
    use random_world::cp::*;
    use random_world::ncm::*;
    use random_world::utils::*;
    use itertools::multizip;
    
    #[test]
    fn cp() {
        let k = 2;
        let ncm = KNN::new(k);
        let n_labels = 2;
        let mut cp = CP::new(ncm, n_labels, Some(0.1));

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

        cp.train(&train_inputs.view(), &train_targets.view()).unwrap();
        let pvalues = cp.predict_confidence(&test_inputs.view()).unwrap();
        println!("Expected p-values: {:?}", expected_pvalues);
        println!("P-values: {:?}", pvalues);
        assert!(pvalues == expected_pvalues);
        cp.set_epsilon(epsilon_1);
        assert!(cp.predict(&test_inputs.view()).unwrap() == expected_preds_1);
        cp.set_epsilon(epsilon_2);
        assert!(cp.predict(&test_inputs.view()).unwrap() == expected_preds_2);
    }

    #[test]
    fn smooth_cp() {
        let k = 2;
        let ncm = KNN::new(k);
        let n_labels = 2;
        let seed = [0, 0];
        let mut cp = CP::new_smooth(ncm, n_labels, Some(0.1), Some(seed));

        let train_inputs = array![[0., 0.],
                                  [1., 0.],
                                  [0., 1.],
                                  [1., 1.],
                                  [2., 2.],
                                  [1., 2.]];
        let train_targets = array![0, 0, 0, 1, 1, 1];
        let test_inputs = array![[2., 1.],
                                 [2., 2.]];
        let expected_pvalues = array![[0., 0.07996389124884251],
                                      [0.16721996675602036, 0.7553161717563366]];

        cp.train(&train_inputs.view(), &train_targets.view()).unwrap();

        let pvalues = cp.predict_confidence(&test_inputs.view()).unwrap();
        println!("Expected p-values: {:?}.", expected_pvalues);
        println!("Actual p-values: {:?}.", pvalues);
        assert!(pvalues == expected_pvalues);
    }

   /// Tests CP in batch mode (i.e., train on training set, predict
   /// test set. For simplicity of the test, training and test sets are
   /// identical.
    #[test]
    fn cp_batch_iris() {
        let k = 3;
        let ncm = KNN::new(k);
        let n_labels = 3;
        let mut cp = CP::new(ncm, n_labels, None);

        let (train_inputs, train_targets) = load_data("tests/data/iris.csv")
                                            .expect("Failed to load data");
        let expected_pvalues = load_pvalues("tests/data/iris-batch-expected.csv")
                                    .expect("Failed to load p-values");

        cp.train(&train_inputs.view(), &train_targets.view())
          .expect("Failed to train the model");

        let pvalues = cp.predict_confidence(&train_inputs.view())
                        .expect("Failed to predict");

        println!("Expected p-values: {:?}.", expected_pvalues);
        println!("Actual p-values: {:?}.", pvalues);
        assert!(pvalues == expected_pvalues);
    }

    /// Tests CP in online mode (i.e., iteratively train on i points,
    /// test on the i+1-th point, and add the i+1-th example into the
    /// new training set.
    #[test]
    fn cp_online_iris() {
        let k = 3;
        let ncm = KNN::new(k);
        let n_labels = 3;
        let mut cp = CP::new(ncm, n_labels, None);

        let (train_inputs, train_targets) = load_data("tests/data/iris.csv")
                                            .expect("Failed to load data");
        let expected_pvalues = load_pvalues("tests/data/iris-online-expected.csv")
                                    .expect("Failed to load p-values");

        // Train on first data point.
        let x = train_inputs.slice(s![0..1, ..]);
        let y = train_targets[[0]];
        cp.train(&x, &array![y].view())
          .expect("Failed to train CP");

        // Update and predict the remaining points in on-line mode.
        for (x, y, exp_pvals) in multizip((train_inputs.outer_iter().skip(1),
                                           train_targets.iter().skip(1),
                                           expected_pvalues.outer_iter())) {
            let x_ = x.into_shape((1, x.len())).unwrap();
            let preds = cp.predict_confidence(&x_)
                          .expect("Failed to predict");
            println!("Expected: {:?}", exp_pvals);
            println!("P-values: {:?}", preds);
            assert!(preds.row(0) == exp_pvals);

            cp.update(&x_, &array![*y].view())
              .expect("Failed to update CP");
        }
    }
}
