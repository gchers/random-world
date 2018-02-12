//! Defines Exchangeability Martingales.
//!
//! Exchangeability martingales are tools for testing the
//! exchangeability (and i.i.d.) assumption.
//!
//! # Examples
//!
//! NOTE: this is just a sketch example which *won't* run.
//! ```
//! use random_world::cp::*;
//! use random_world::ncm::*;
//! use random_world::exchangeability::*;
//!
//! // Create sequence. Anomalies start from the 11th observation.
//! let data_sequence = vec![0, 1, 0, 2, -1, 2, 0, 1, 1, 0, 7, 10, -20, 40, 30];
//!
//! // Create a smoothed CP with k-NN nonconformity measure (k=1).
//! // Note that the CP needs to be smoothed for the method to work.
//! let ncm = KNN::new(1);
//! let mut cp = CP::new(ncm);
//! // Create a new Plug-in martingale with `bandwidth=None` (i.e.,
//! // `bandwidth` will be automatically chosen using Silverman's method).
//! let mut m = Martingale::new_plugin(None);
//! 
//! //for x in data_sequence {
//! //    p = cp.predict(x[i]);
//! //    M = m.next(p);
//! //    cp.update(x[i], y[i]);
//! //}
//! ```
pub mod martingales;

pub use self::martingales::Martingale;
