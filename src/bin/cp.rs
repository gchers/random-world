#[macro_use]
extern crate ndarray;
#[macro_use]
extern crate serde_derive;
extern crate docopt;
extern crate random_world;

use ndarray::prelude::*;
use random_world::cp::*;
use random_world::ncm::*;
use random_world::utils::{load_data, store_predictions};
use std::io::prelude::*;
use docopt::Docopt;

const USAGE: &'static str = "
Predict data using Conformal Prediction.

Usage: cp knn [--knn=<k>] [options] [--] <training-file> <testing-file> <output-file>
       cp kde [--kernel<kernel>] [--bandwidth=<bw>] [options] [--] <training-file> <testing-file> <output-file>
       cp (--help | --version)

Options:
    -e, --epsilon=<epsilon>     Significance level. If specified, the output are
                                label predictions rather than p-values.
    -s, --smooth                Smooth CP.
    --seed                      PRNG seed. Only used if --smooth set.
    -k, --knn=<kn>              Number of neighbors for k-NN [default: 5].
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
    arg_training_file: String,
    arg_testing_file: String,
    arg_output_file: String,
    cmd_knn: bool,
    cmd_kde: bool,
}


fn main() {
    // Parse args from command line
    let argv = std::env::args();

    let args: Args = Docopt::new(USAGE)
                            .and_then(|d| d.deserialize())
                            .unwrap_or_else(|e| e.exit());

    // Nonconformity measure
    let ncm = if args.cmd_knn {
        KNN::new(args.flag_knn)
    } else if args.cmd_kde {
        unimplemented!();
    } else {
        // Docopt shouldn't let this happen.
        panic!("This shouldn't happen");
    };

    let mut cp = if args.flag_smooth {
        let seed = match args.flag_seed {
            Some(s) => Some([0, s]),
            None => None,
        };
        CP::new_smooth(ncm, args.flag_epsilon, seed)
    } else {
        CP::new(ncm, args.flag_epsilon)
    };

    // Load training and test data.
    let (train_inputs, train_targets) = load_data(args.arg_training_file)
                                        .expect("Failed to load");
    let (test_inputs, test_targets) = load_data(args.arg_testing_file)
                                        .expect("Failed to load");

    // Train.
    cp.train(&train_inputs.view(), &train_targets.view())
      .expect("Failed to train the model");

    // Predict and store results.
    // TODO: store predictions on the fly.
    if let Some(_) = args.flag_epsilon {
        let preds = cp.predict(&test_inputs.view())
                      .expect("Failed to predict");
        store_predictions(preds.view(), args.arg_output_file);
    } else {
        let preds = cp.predict_confidence(&test_inputs.view())
                      .expect("Failed to predict");
        store_predictions(preds.view(), args.arg_output_file);
    };
}
