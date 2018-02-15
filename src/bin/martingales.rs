#[macro_use]
extern crate ndarray;
#[macro_use]
extern crate serde_derive;
extern crate docopt;
extern crate random_world;

use ndarray::prelude::*;
use random_world::cp::*;
use random_world::ncm::*;
use random_world::exchangeability::*;
use random_world::utils::{load_pvalues, store_predictions};
use docopt::Docopt;

const USAGE: &'static str = "
Test exchangeability using martingales.

Usage: martingales plugin [--bandwidth=<bw>] [options] <output-file> <pvalues-file>
       martingales power [--epsilon=<e>] [options] <output-file> <pvalues-file>
       martingales (--help | --version)

Options:
    --seed                      PRNG seed.
    -h, --help                  Show help.
    --version                   Show the version.
";

#[derive(Deserialize)]
struct Args {
    cmd_plugin: bool,
    cmd_power: bool,
    flag_epsilon: Option<f64>,
    flag_seed: Option<u64>,
    flag_bandwidth: Option<f64>,
    arg_pvalues_file: String,
    arg_output_file: String,
}


fn main() {
    // Parse args from command line
    let args: Args = Docopt::new(USAGE)
                            .and_then(|d| d.deserialize())
                            .unwrap_or_else(|e| e.exit());

    // Load p-values.
    let pvalues = load_pvalues(&args.arg_pvalues_file)
                    .expect("Failed to load p-values");
    if pvalues.cols() != 1 {
        panic!("Martingales can only be computed for single-label \
                predictions (i.e., one p-value per example).");
    }
        
    let mut martingale = if args.cmd_plugin {
        Martingale::new_plugin(args.flag_bandwidth)
    } else if args.cmd_power {
        Martingale::new_power(args.flag_epsilon.unwrap())
    } else {
        // Docopt shouldn't let this happen.
        panic!("This shouldn't happen");
    };
    
    // Reset output file.
    store_predictions(Array2::<f64>::zeros((0,0)).view(),
                      &args.arg_output_file, false).expect("Failed to initialize file");

    for p in pvalues.outer_iter() {
        let m = martingale.update(p[[0]]);
        store_predictions(arr2(&[[m]]).view(), &args.arg_output_file, true)
            .expect("Failed to store results");
    }
}
