#![feature(test)]

extern crate test;
extern crate rand;
extern crate ndarray;
extern crate pcg_rand;
extern crate random_world;

use test::{Bencher, black_box};
use pcg_rand::Pcg32;
use rand::{Rng, SeedableRng};
use ndarray::prelude::*;

use random_world::cp::*;
use random_world::ncm::*;

fn generate_data(n: usize, d: usize, n_labels: usize, seed: [u64; 2])
        -> (Array2<f64>, Array1<usize>) {
    let mut rng = Pcg32::from_seed(seed);

    let inputs = Array::from_iter(rng.gen_iter::<f64>()
                                     .take(n*d)).into_shape((n, d))
                                                .unwrap();
    let targets = Array::from_iter((0..n).into_iter()
                                         .map(|_| rng.gen_range::<usize>(0, n_labels)));
    (inputs, targets)
}
    
#[bench]
fn bench_cp_train(b: &mut Bencher) {
    let ncm = KNN::new(2);
    let mut cp = CP::new(ncm, Some(0.1));

    let n = 1000;
    let d = 2;
    let n_labels = 2;
    let seed = [0, 0];

    let (inputs, targets) = generate_data(n, d, n_labels, seed);

    b.iter(|| {
        let _ = black_box(cp.train(&inputs.view(), &targets.view()));
    });
}

#[bench]
fn bench_cp_predict(b: &mut Bencher) {
    let ncm = KNN::new(2);
    let mut cp = CP::new(ncm, Some(0.1));

    let n = 10;
    let d = 1;
    let n_labels = 2;
    let seed = [0, 0];

    let (inputs, targets) = generate_data(n, d, n_labels, seed);

    let _ = cp.train(&inputs.view(), &targets.view());

    b.iter(|| {
        let _ = black_box(cp.predict_confidence(&inputs.view()).unwrap());
    });
}
