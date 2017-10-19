#[macro_use]
extern crate ndarray;
extern crate random_world;
extern crate rayon;
extern crate docopt;

use ndarray::prelude::*;
use random_world::cp::*;
use random_world::ncm::*;
use std::io::prelude::*;
use std::fs::File;
use rayon::prelude::*;
use docopt::Docopt;

const USAGE: &'static str = "
    Usage: martingales <src> <out-pvalues> <out-martingales> <k>
";

fn compute_pvalues(inputs: &Array2<f64>) -> Vec<f64> {
    let k = 2;
    let seed = [0, 0];
    let n_inputs = inputs.rows();

    let mut pvalues = Vec::with_capacity(n_inputs);

    for n in 2..n_inputs { //.into_par_iter() {
        let ncm = KNN::new(k);
        let mut cp = CP::new_smooth(ncm, None, Some(seed));

        let ni = n as isize;
        let train_inputs = &inputs.slice(s![..ni, ..]);
        let fake_labels = Array::from_vec(vec![0; n]);

        cp.train(&train_inputs, &fake_labels.view())
          .expect("Failed to train");
        let pvalue = cp.predict_confidence(&inputs.slice(s![ni..ni+1, ..]))
                       .expect("Didn't predict");

        pvalues.push(pvalue[[0, 0]]);
    }

    pvalues
}

fn compute_martingales(pvalues: &Vec<f64>) -> Vec<f64> {
    let e = 0.92;
    let mut m = 1.;
    let mut martingales = Vec::with_capacity(pvalues.len());

    for p in pvalues {
        if *p == 0. {
            m *= 0.001;
        } else {
            m *= e * p.powf(e - 1.);
        }
        martingales.push(m);
    }

    martingales
}

fn std(v: &[f64]) -> f64 {
    let m = v.iter()
             .sum::<f64>() / v.len() as f64;

    (v.iter()
      .map(|x| (x - m).powi(2))
      .sum::<f64>() / v.len() as f64).powf(0.5)
}

fn kde(x: f64, inputs: &[f64]) -> f64 {
    let q = 1. / (2f64*3.14).powf(0.5);
    let n = inputs.len() as f64;
    let h = std(inputs)*(4./3./n).powf(1./5.);

    inputs.iter()
          .map(|v| q * (-0.5*((v - x) / h).powi(2)).exp())
          .sum::<f64>() / (n*h)
}

fn compute_plugin_martingales(pvalues: &Vec<f64>) -> Vec<f64> {
    let mut m = 1.;
    let mut martingales = Vec::with_capacity(pvalues.len());

    // NOTE: fix for 0..
    for i in 1..pvalues.len() {
        let k = kde(pvalues[i], &pvalues[..i+1]);
        m *= k;
        martingales.push(m);
    }

    martingales
}

fn vec_to_file(v: &Vec<f64>, fname: &str) {
    let mut file = File::create(fname)
                        .expect("Couldn't create file");
    file.write(v.iter()
                .map(|x| x.to_string())
                .collect::<Vec<String>>()
                .join("\n")
                .as_bytes());
}

fn main() {
    // Parse args from command line
    let argv = std::env::args();

    let args = Docopt::new(USAGE)
                      .and_then(|d| d.argv(argv.into_iter()).parse())
                      .unwrap_or_else(|e| e.exit());
    // NOTE: use args.src?
    let in_fname = args.get_str("<src>");
    let pvalues_fname = args.get_str("<out-pvalues>");
    let martingales_fname = args.get_str("<out-martingales>");
    let k = args.get_str("<k>")
                .parse::<usize>()
                .expect("k: integer >= 0");

    // Load data
    let mut file = File::open(in_fname)
                        .expect("Unable to open the file");
    let mut contents = String::new();

    file.read_to_string(&mut contents)
        .expect("Unable to read the file");
    let inputs = contents.split("\n")
                         //.skip(1)
                         .take(k)
                         .filter_map(|s| s.parse::<f64>().ok())
                         .collect::<Vec<_>>();

    // Compute p-values, martingales
    let n = inputs.len();
    println!("computing p-values");
    let pvalues = compute_pvalues(&Array::from_vec(inputs)
                                         .into_shape((n, 1))
                                         .unwrap());
    println!("writing into file");
    vec_to_file(&pvalues, pvalues_fname);

    println!("computing martingales");
    let martingales = compute_plugin_martingales(&pvalues);
    println!("writing into file");
    vec_to_file(&martingales, martingales_fname);
}
