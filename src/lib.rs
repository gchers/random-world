//! The random-world crate.
//!
//! A crate implementing Machine Learning ML methods for confident prediction
//! (e.g., Conformal Predictors) and related ones introduced in the book
//! Algorithmic Learning in a Random World ([ALRW](http://alrw.net/)).
//!
//!
//! # Goals
//! * Fast implementation of methods introduced in the book ALRW.
//! * Should easily allow to wrap existing rust implementations of ML
//!   classifiers/scorers.
//! * (Maybe) allow interfacing to Python.
//! * (Maybe) can be called as a binary.
//!
//! # Examples
//!
//! Create a Conformal Predictor with k-NN nonconformity measure, `k=2`,
//! and with significance level `epsilon=0.3`, train it on some training
//! set and use it to predict two test vector inputs.
//!
//! The output predictions will be a matrix, one row per each training
//! input, and one column per label, where each `bool` element is `true`
//! if the label conforms the distribution, `false` otherwise.
//!
//! ```
//! #[macro_use(array)]
//! extern crate ndarray;
//! extern crate random_world;
//!
//! # fn main() {
//! use random_world::cp::*;
//! use random_world::ncm::*;
//!
//! let ncm = KNN::new(2);
//! let mut cp = CP::new(ncm, Some(0.3));
//! let train_inputs = array![[0., 0.],
//!                           [1., 0.],
//!                           [0., 1.],
//!                           [1., 1.],
//!                           [2., 2.],
//!                           [1., 2.]];
//! let train_targets = array![0, 0, 0, 1, 1, 1];
//! let test_inputs = array![[2., 1.],
//!                          [2., 2.]];
//!
//! // Train and predict
//! cp.train(&train_inputs.view(), &train_targets.view())
//!   .expect("Failed prediction");
//! let preds = cp.predict(&test_inputs.view())
//!               .expect("Failed to predict");
//! assert!(preds == array![[false, true],
//!                         [false, true]]);
//! # }
//! ```
//!
//! More examples on deterministic/smooth Conformal Predictors at
//! [CP](/cp/cp/struct.CP.html).
//#![warn(missing_docs)]

extern crate rand;
extern crate rayon;
extern crate pcg_rand;
extern crate itertools;
extern crate rusty_machine;
extern crate ordered_float;
#[macro_use]
extern crate ndarray;

pub mod cp;
pub mod ncm;
