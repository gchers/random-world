//! Defines Exchangeability Martingales.
//!
//! Exchangeability martingales are tools for testing the
//! exchangeability (and i.i.d.) assumption.
//!
//! # Examples
//!
//! ```
//! extern crate rand;
//! extern crate statrs;
//! #[macro_use(array)]
//! extern crate ndarray;
//! extern crate random_world;
//! extern crate pcg_rand;
//!
//! # fn main() {
//! use pcg_rand::Pcg32;
//! use ndarray::prelude::*;
//! use rand::{Rng, SeedableRng};
//! use statrs::distribution::{Normal, Distribution};
//!
//! use random_world::cp::*;
//! use random_world::ncm::*;
//! use random_world::exchangeability::*;
//!
//! // Generate sequence. Trend change after 100 examples.
//! let seed = [0, 0];
//! let mut rng = Pcg32::from_seed(seed);
//! let n = Normal::new(0.0, 1.0).unwrap();
//! let n_anomaly = Normal::new(40.0, 10.0).unwrap();
//!
//! let mut data_sequence = (0..100).into_iter()
//!                             .map(|_| n.sample(&mut rng))
//!                             .collect::<Vec<_>>();
//! data_sequence.extend((0..100).into_iter()
//!                             .map(|_| n_anomaly.sample(&mut rng)));
//!
//! // Create a smoothed CP with k-NN nonconformity measure (k=1).
//! // Note that the CP needs to be smoothed for the method to work.
//! let seed = [1, 1];
//! let ncm = KNN::new(1);
//! let n_labels = 1;
//! let mut cp = CP::new_smooth(ncm, n_labels, None, Some(seed));
//!
//! // Create a new Plug-in martingale with `bandwidth=None` (i.e.,
//! // `bandwidth` will be automatically chosen using Silverman's method).
//! let mut m = Martingale::new_plugin(Some(0.1));
//! 
//! let y = array![0];          // Arbitrary label for data points.
//!
//! // Train on first data point.
//! cp.train(&array![[data_sequence[0]]].view(), &y.view())
//!   .expect("Failed to train CP");
//! // Update and evaluate the martingale on the remaining points.
//! for x in data_sequence.into_iter().skip(1) {
//!     let p = cp.predict_confidence(&array![[x]].view())
//!               .expect("Failed to predict")[[0,0]];
//!     let _ = m.update(p);    // Returns the martingale's value.
//!     cp.update(&array![[x]].view(), &y.view());
//! }
//!
//! assert!(m.is_large());
//! # }
//! ```
pub mod martingales;

pub use self::martingales::Martingale;
