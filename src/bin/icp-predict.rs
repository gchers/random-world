#[macro_use]
extern crate ndarray;
#[macro_use]
extern crate serde_derive;
extern crate docopt;
extern crate random_world;
extern crate itertools;

use random_world::cp::*;
use random_world::ncm::*;
use random_world::utils::{load_data, store_predictions};
use itertools::Itertools;
use docopt::Docopt;
use ndarray::*;

const USAGE: &'static str = "
Predict data using Conformal Prediction.

If no <testing-file> is specified, on-line mode is assumed.

Usage: icp-predict knn [--knn=<k>] [options] [--] <output-file> <training-file> [<testing-file>]
       icp-predict kde [--kernel<kernel>] [--bandwidth=<bw>] [options] [--] <output-file> <training-file> [<testing-file>]
       icp-predict (--help | --version)

Options:
    -e, --epsilon=<epsilon>     Significance level. If specified, the output are
                                label predictions rather than p-values.
    -s, --smooth                Smooth CP.
    --seed                      PRNG seed. Only used if --smooth set.
    -k, --knn=<kn>              Number of neighbors for k-NN [default: 5].
    --n-labels=<n>              Number of labels. If specified in advance it
                                slightly improves performances.
    -h, --help                  Show help.
    --version                   Show the version.
";

#[derive(Deserialize)]
struct Args {
    flag_epsilon: Option<f64>,
    flag_smooth: bool,
    flag_seed: Option<u64>,
    flag_knn: usize,
    flag_kernel: Option<String>,
    flag_bandwidth: Option<f64>,
    flag_n_labels: Option<usize>,
    arg_training_file: String,
    arg_testing_file: Option<String>,
    arg_output_file: String,
    cmd_knn: bool,
    cmd_kde: bool,
}


fn main() {
    // Parse args from command line.
    let args: Args = Docopt::new(USAGE)
                            .and_then(|d| d.deserialize())
                            .unwrap_or_else(|e| e.exit());

    // Nonconformity measure.
    let ncm = if args.cmd_knn {
        KNN::new(args.flag_knn)
    } else if args.cmd_kde {
        unimplemented!();
    } else {
        // Docopt shouldn't let this happen.
        panic!("This shouldn't happen");
    };

    // Load training and test data.
    let (train_inputs, train_targets) = load_data(&args.arg_training_file)
                                        .expect("Failed to load data");

    // Number of labels.
    let n_labels = match args.flag_n_labels {
        Some(n_labels) => n_labels,
        None => train_targets.into_iter()
                             .unique()
                             .count()
    };

    // Initialize CP.
    let mut cp = if args.flag_smooth {
        let seed = match args.flag_seed {
            Some(s) => Some([0, s]),
            None => None,
        };
        // Not sure why we'd need smooth ICP, as it's mostly of theoretical
        // interest (and implemented for transductive CP in cp-predict).
        unimplemented!();
    } else {
        CP::new_inductive(ncm, n_labels, args.flag_epsilon)
    };

    // If testing file is specified, predict test data.
    // Otherwise, use CP in on-line mode.
    if let Some(testing_file) = args.arg_testing_file {
        println!("Predicting {}", testing_file);
        let (test_inputs, _) = load_data(&testing_file)
                                    .expect("Failed to load data");
        // Train on half, calibrate on the other half.
        let n_train = (train_inputs.rows() / 2) as isize;

        cp.train(&train_inputs.slice(s![..n_train, ..]),
                 &train_targets.slice(s![..n_train]))
          .expect("Failed to train the model");

        cp.calibrate(&train_inputs.slice(s![n_train.., ..]),
                     &train_targets.slice(s![n_train..]))
          .expect("Failed to train the model");

        // Predict and store results.
        if let Some(_) = args.flag_epsilon {
            let preds = cp.predict(&test_inputs.view())
                          .expect("Failed to predict");
            store_predictions(preds.view(), &args.arg_output_file, false)
                .expect("Failed to store the output");
        } else {
            let preds = cp.predict_confidence(&test_inputs.view())
                          .expect("Failed to predict");
            store_predictions(preds.view(), &args.arg_output_file, false)
                .expect("Failed to store the output");
        }
    } else {
        // On-line version of ICP is not implemented. It's not clear to
        // me whether it'd be useful at all.
        unimplemented!();
    }
}
